# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot policy runner for URLab Bridge.

Loads any pretrained LeRobot policy (ACT, Diffusion, SmolVLA, XVLA, etc.)
from HuggingFace Hub or a local path and runs inference against an Unreal
articulation over ZMQ.

Observations are built from ZMQ joint state + camera images.
Actions (joint position targets) are sent back via ZMQ control.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import msgpack
import numpy as np
import zmq

from urlab_client.transports import parse_camera_frame, resolve_endpoint

from ._state_stream import StateStream, art_names, art_qpos

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from lerobot.policies.pretrained import PreTrainedPolicy
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False


# ─── Configuration ───


@dataclass
class LeRobotRunnerCfg:
    """Configuration for a LeRobot policy runner."""

    # Model source: HuggingFace Hub repo ID or local directory path
    pretrained_path: str = ""

    # ZMQ endpoints
    state_endpoint: str = "tcp://127.0.0.1:5555"
    control_endpoint: str = "tcp://127.0.0.1:5556"
    camera_endpoint: str = "tcp://127.0.0.1:5558"
    info_endpoint: str = "tcp://127.0.0.1:5557"

    # Which articulation to target
    articulation_prefix: str = ""

    # Joint names in the order the policy expects them.
    # If empty, auto-discovered from ZMQ and sorted by ZMQ ID.
    joint_names: list[str] = field(default_factory=list)

    # Camera name mapping: URLab camera name -> LeRobot observation key
    # e.g. {"SceneCapture_top": "observation.images.top"}
    camera_map: dict[str, str] = field(default_factory=dict)

    # Camera resolution expected by the policy
    camera_width: int = 640
    camera_height: int = 480

    # Inference frequency (Hz)
    freq: float = 30.0

    # Device override ("cpu", "cuda", "cuda:0"). If empty, auto-detect.
    device: str = ""

    # Task instruction for language-conditioned policies (SmolVLA, XVLA, Pi0)
    task: str = ""

    # MJCF path for IK solver (needed for EE action policies like XVLA).
    # If empty, auto-detected from common locations.
    mjcf_path: str = ""


# ─── EE-to-Joint IK ───


class EEtoJointIK:
    """
    Converts end-effector delta actions (from VLA policies like XVLA)
    to joint position targets using MuJoCo's Jacobian-based damped
    least-squares IK.

    XVLA ee6d action format (20D, bimanual):
      [0:3]   arm1 xyz delta
      [3:9]   arm1 6D rotation
      [9]     arm1 gripper (0-1, after sigmoid)
      [10:13] arm2 xyz delta
      [13:19] arm2 6D rotation
      [19]    arm2 gripper (0-1)

    For single-arm (Franka), we use arm1 only.
    """

    def __init__(self, mjcf_path: str, ee_site: str = "gripper",
                 num_arm_joints: int = 7, damping: float = 1e-4,
                 max_delta: float = 0.05, ik_steps: int = 50,
                 world_offset: np.ndarray | None = None):
        import mujoco

        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.mujoco = mujoco
        self.ee_site_id = self.model.site(ee_site).id
        self.num_arm_joints = num_arm_joints
        self.damping = damping
        self.max_delta = max_delta  # clamp per-joint step (rad)
        self.ik_steps = ik_steps    # max iterations per solve
        self.nv = self.model.nv
        # Offset to convert policy world-frame targets to MuJoCo-relative
        # e.g. LIBERO base offset [0, 0, 0.25]
        self.world_offset = world_offset if world_offset is not None else np.zeros(3)

        logger.info(f"  IK solver: site='{ee_site}', nq={self.model.nq}, nv={self.nv}, steps={ik_steps}")
        if np.any(self.world_offset != 0):
            logger.info(f"  World offset: {self.world_offset}")

    def ee_to_joint(self, processed_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """
        Convert processed 7D action [pos(3), axis_angle(3), gripper(1)]
        to joint position targets via iterative Jacobian IK.

        Args:
            processed_action: 7D [ee_pos(3), axis_angle(3), gripper(1)]
            current_joints: current joint positions from ZMQ

        Returns:
            joint_targets: array matching current_joints length
        """
        mujoco = self.mujoco
        n_arm = self.num_arm_joints

        target_pos = processed_action[0:3].astype(np.float64) - self.world_offset
        gripper = float(processed_action[6]) if len(processed_action) > 6 else 0.0

        # Set current joint state
        nq = min(len(current_joints), self.model.nq)
        self.data.qpos[:nq] = current_joints[:nq]

        # Iterative IK: solve for target position
        for _ in range(self.ik_steps):
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            dx = target_pos - current_pos
            err = np.linalg.norm(dx)
            if err < 1e-4:
                break

            jacp = np.zeros((3, self.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
            J = jacp[:, :n_arm]

            JJT = J @ J.T + self.damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, dx)
            dq = np.clip(dq, -self.max_delta, self.max_delta)
            self.data.qpos[:n_arm] += dq

        # Read out joint targets
        joint_targets = current_joints.copy()
        joint_targets[:n_arm] = self.data.qpos[:n_arm].astype(np.float32)

        # Gripper: 1 = open (0.04m), -1 = closed (0.0m)
        if len(joint_targets) > n_arm:
            joint_targets[n_arm] = 0.04 if gripper > 0 else 0.0

        return joint_targets


# ─── Runner ───


class LeRobotRunner:
    """
    Connects to Unreal over ZMQ, builds observations, runs a LeRobot policy,
    and sends actions back as joint position targets.
    """

    def __init__(self, cfg: LeRobotRunnerCfg):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required. Install with: uv sync --extra robojudo"
            )
        if not HAS_LEROBOT:
            raise ImportError(
                "LeRobot is required. Install with: uv pip install lerobot"
            )

        self.cfg = cfg
        self._stop = False

        # Resolve device
        if cfg.device:
            self.device = torch.device(cfg.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"LeRobot runner — device: {self.device}")

        # Load policy
        self.policy, self.preprocessor, self.postprocessor = self._load_policy(
            cfg.pretrained_path
        )

        # ZMQ connections
        self._ctx = zmq.Context()
        # State arrives as the canonical `state/full` msgpack snapshot; joint
        # values are indexed out of the per-art `qpos` block.
        self._state = StateStream(self._ctx, cfg.state_endpoint)

        self._cam_sub = self._ctx.socket(zmq.SUB)
        self._cam_sub.connect(cfg.camera_endpoint)
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._cam_sub.setsockopt(zmq.RCVTIMEO, 100)
        self._connected_cam_endpoints = {cfg.camera_endpoint}

        # Discover additional camera endpoints from info socket
        self._discover_camera_endpoints()

        self._ctrl_pub = self._ctx.socket(zmq.PUB)
        self._ctrl_pub.connect(cfg.control_endpoint)

        time.sleep(0.2)  # let PUB socket settle

        # State tracking
        self._joint_positions = {}   # joint_name -> position
        self._joint_ids = {}         # joint_name -> discovery index
        self._images = {}            # camera_name -> np.ndarray (H, W, 4) BGRA
        self._prefix = cfg.articulation_prefix
        self._actuator_ids = None    # ordered list of actuator IDs for control
        # Ordered component names for the target articulation, discovered from
        # the info-socket `actuator_list`. Used to key the concatenated `qpos`
        # block of `state/full` back to named joints (the snapshot carries no
        # names, only per-art discovery order).
        self._art_names: list[str] = []

        # Runtime stats
        self.step_count = 0
        self.freq_hz = 0.0
        self.phase = "idle"
        self.error = ""
        self.active_camera_map = {}  # resolved mapping: urlab_name -> lerobot_key
        self.active_joint_order = []  # resolved joint order
        # Note: _language_tokenizer is set by _load_policy above

    def _load_policy(self, pretrained_path: str):
        """Load a LeRobot policy from HF Hub or local path.

        Uses a manual loading path to avoid draccus temp-file issues on Windows.
        Downloads config.json, resolves the policy class, then loads weights
        via HuggingFace's safetensors integration.
        """
        import json
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from lerobot.policies.factory import get_policy_class

        logger.info(f"Loading LeRobot policy from: {pretrained_path}")

        # 1. Download and parse config
        p = Path(pretrained_path)
        if p.is_dir():
            config_path = p / "config.json"
            model_dir = p
        else:
            config_path = Path(hf_hub_download(pretrained_path, "config.json"))
            model_dir = config_path.parent

        with open(config_path) as f:
            cfg_dict = json.load(f)

        policy_type = cfg_dict.get("type", "act")
        logger.info(f"  Policy type from config: {policy_type}")

        # 2. Get the policy class and its config class
        #    Patch GR00T module if it fails to import (dataclass bug on Python 3.12)
        try:
            PolicyClass = get_policy_class(policy_type)
        except (TypeError, ImportError):
            import sys
            import types
            for mod_name in [
                "lerobot.policies.groot",
                "lerobot.policies.groot.configuration_groot",
                "lerobot.policies.groot.__init__",
            ]:
                if mod_name not in sys.modules:
                    sys.modules[mod_name] = types.ModuleType(mod_name)
            sys.modules["lerobot.policies.groot.configuration_groot"].GrootConfig = type("GrootConfig", (), {})
            logger.warning("  Patched GR00T module (dataclass incompatibility)")
            PolicyClass = get_policy_class(policy_type)
        ConfigClass = PolicyClass.config_class

        # 3. Build config from dict — convert nested dicts to proper types
        from lerobot.configs.policies import PolicyFeature, FeatureType

        # NormalizationMode was removed in LeRobot 0.5.0
        try:
            from lerobot.policies.normalize import NormalizationMode
            has_norm_mode = True
        except ImportError:
            has_norm_mode = False

        build_kwargs = {}
        for k, v in cfg_dict.items():
            if k not in ConfigClass.__dataclass_fields__:
                continue
            # Convert input/output feature dicts to PolicyFeature objects
            if k in ("input_features", "output_features") and isinstance(v, dict):
                build_kwargs[k] = {
                    feat_name: PolicyFeature(
                        type=FeatureType(feat_val["type"]),
                        shape=tuple(feat_val["shape"]),
                    )
                    for feat_name, feat_val in v.items()
                }
            # Convert normalization_mapping string values to NormalizationMode enums
            elif k == "normalization_mapping" and isinstance(v, dict):
                if has_norm_mode:
                    build_kwargs[k] = {
                        FeatureType(ftype): NormalizationMode(nmode)
                        for ftype, nmode in v.items()
                    }
                else:
                    # LeRobot 0.5.0+ doesn't use NormalizationMode — pass as-is or skip
                    build_kwargs[k] = v
            else:
                build_kwargs[k] = v

        config = ConfigClass(**build_kwargs)
        config.pretrained_path = pretrained_path

        # Override device — force to what we selected
        config.device = str(self.device)

        # 4. Instantiate policy with config
        policy = PolicyClass(config)

        # 5. Load weights via state_dict (ensures buffers like normalization stats are loaded)
        from safetensors.torch import load_file
        model_file = model_dir / "model.safetensors"
        if not model_file.exists():
            model_file = Path(hf_hub_download(pretrained_path, "model.safetensors"))

        state_dict = load_file(str(model_file))
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"  Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.info(f"  Unexpected keys (ignored): {len(unexpected)}")
        logger.info("  Loaded weights from model.safetensors")

        # Fix uninitialized normalization stats (base models ship with inf sentinels).
        # These are stored as nn.Parameter (not buffers) inside ParameterDict.
        # Replace inf mean/std/min/max with identity normalization.
        for name, param in policy.named_parameters():
            if not torch.isinf(param.data).any():
                continue
            if "mean" in name or "min" in name:
                param.data.fill_(0.0)
                logger.info(f"  Fixed uninitialized stat: {name} -> 0.0")
            elif "std" in name:
                param.data.fill_(1.0)
                logger.info(f"  Fixed uninitialized stat: {name} -> 1.0")
            elif "max" in name:
                param.data.fill_(1.0)
                logger.info(f"  Fixed uninitialized stat: {name} -> 1.0")

        policy.to(self.device)
        policy.eval()
        policy.reset()

        # Load language tokenizer if the policy needs pre-tokenized input
        tokenizer_name = getattr(config, 'tokenizer_name', None)
        if tokenizer_name:
            from transformers import AutoTokenizer
            self._language_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"  Language tokenizer: {tokenizer_name}")
        else:
            self._language_tokenizer = getattr(policy, 'language_tokenizer', None)
            if self._language_tokenizer:
                logger.info("  Language tokenizer: from policy")

        # Log info
        logger.info(f"  Policy class: {type(policy).__name__}")
        logger.info(f"  Input features: {list(config.input_features.keys())}")
        logger.info(f"  Output features: {list(config.output_features.keys())}")
        logger.info(f"  Device: {self.device}")

        # No preprocessor/postprocessor — we handle normalization manually
        preprocessor = lambda x: x
        postprocessor = lambda x: x

        return policy, preprocessor, postprocessor

    def _drain_state(self):
        """Read the latest `state/full` snapshot and index out joint positions.

        The snapshot's per-art `qpos` is concatenated in joint discovery order
        with no names, so it is keyed back to names via `self._art_names`
        (the info-socket actuator order). A leading free joint, if present,
        occupies the head of `qpos`; the names align to its tail."""
        snap = self._state.drain()
        if snap is None:
            return
        names = art_names(snap)
        if not names:
            return
        if not self._prefix:
            self._prefix = self.cfg.articulation_prefix or names[0]
            logger.info(f"Auto-detected prefix: {self._prefix}")
        if self._prefix not in names:
            return

        qpos = art_qpos(snap, self._prefix)
        order = self._art_names or list(self.cfg.joint_names)
        if not order:
            order = [f"joint_{i}" for i in range(qpos.size)]
        offset = max(0, qpos.size - len(order))
        for i, jname in enumerate(order):
            idx = offset + i
            if idx < qpos.size:
                self._joint_positions[jname] = float(qpos[idx])
                self._joint_ids[jname] = i

    def _drain_cameras(self):
        """Read all pending camera messages from ZMQ."""
        w, h = self.cfg.camera_width, self.cfg.camera_height
        while True:
            try:
                topic_bytes = self._cam_sub.recv(zmq.NOBLOCK)
                topic = topic_bytes.decode("utf-8").strip()
                if not self._cam_sub.getsockopt(zmq.RCVMORE):
                    continue
                payload = self._cam_sub.recv()

                parts = topic.split("/", 1)
                if len(parts) < 2:
                    continue
                prefix, subtopic = parts

                if prefix != self._prefix:
                    continue

                # Camera topics are canonical `<art>/<cam>` (no `camera/` infix).
                cam_name = subtopic.strip()
                # Streamed frames carry a 40-byte FMjCameraFrameMeta header
                # ahead of the pixels; strip it before reshaping or every
                # frame fails the size check and is silently dropped.
                pixels, _fid, _sim_t, _cap_t = parse_camera_frame(payload)
                expected = w * h * 4
                if len(pixels) == expected:
                    img = np.frombuffer(pixels, dtype=np.uint8).reshape((h, w, 4))
                    self._images[cam_name] = img

            except zmq.Again:
                break

    def _discover_camera_endpoints(self):
        """Discover additional camera endpoints from the info socket.

        NOTE: the `:5557` info broadcast was retired in 5.3, so this subscribe
        now receives nothing and simply times out (RCVTIMEO=2000). No endpoints
        are added, which is harmless -- the runner falls back to whatever camera
        endpoints are already connected / configured. A replacement discovery
        mechanism is a pending design decision."""
        import json
        info_sub = self._ctx.socket(zmq.SUB)
        info_sub.connect(self.cfg.info_endpoint)
        info_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        info_sub.setsockopt(zmq.RCVTIMEO, 2000)

        try:
            for _ in range(20):
                try:
                    payload = info_sub.recv().decode("utf-8")
                    data = json.loads(payload)
                    if "camera_list" in data:
                        for entry in data["camera_list"]:
                            ep = entry.get("endpoint")
                            if ep:
                                # Advertised endpoints are bind form
                                # (tcp://0.0.0.0:NNNN); rewrite to the host the
                                # state stream reached, else connect() silently
                                # yields no frames.
                                ep = resolve_endpoint(ep, self.cfg.state_endpoint)
                            if ep and ep not in self._connected_cam_endpoints:
                                logger.info(f"  Discovered camera endpoint: {ep}")
                                self._cam_sub.connect(ep)
                                self._connected_cam_endpoints.add(ep)
                except zmq.Again:
                    break
        finally:
            info_sub.close()

    def _discover_actuator_ids(self):
        """Query the info endpoint for actuator ID mapping.

        NOTE: the `:5557` `actuator_list` broadcast was retired in 5.3, so this
        subscribe now receives nothing and simply times out (RCVTIMEO=3000),
        returning `{}`. That is intentional graceful degradation: an empty map
        makes `_send_action` fall back to ordinal actuator ids, which keeps the
        control path working without a crash. A replacement discovery mechanism
        is a pending design decision (full migration not done yet)."""
        info_sub = self._ctx.socket(zmq.SUB)
        info_sub.connect(self.cfg.info_endpoint)
        info_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        info_sub.setsockopt(zmq.RCVTIMEO, 3000)

        import json
        try:
            for _ in range(20):
                try:
                    payload = info_sub.recv().decode("utf-8")
                    data = json.loads(payload)
                    if (data.get("type") == "actuator_list"
                            and data.get("robot") == self._prefix):
                        names = data.get("names", [])
                        ids = data.get("ids", [])
                        name_to_id = {}
                        for n, i in zip(names, ids):
                            short = n.replace(self._prefix + "_", "", 1)
                            name_to_id[short] = int(i)
                        return name_to_id
                except zmq.Again:
                    break
        finally:
            info_sub.close()
        return {}

    def _compute_ee_state(self, joint_order: list[str], state_dim: int) -> np.ndarray:
        """
        Compute EE state for LIBERO format:
        [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D

        Then zero-padded to state_dim (typically 20 for XVLA).
        """
        from scipy.spatial.transform import Rotation

        ik = self._ee_ik
        mujoco = ik.mujoco

        # Set joint positions in MuJoCo
        all_joints = self._get_joint_order()
        nq = min(len(all_joints), ik.model.nq)
        for i in range(nq):
            jname = all_joints[i] if i < len(all_joints) else ""
            ik.data.qpos[i] = self._joint_positions.get(jname, 0.0)

        mujoco.mj_forward(ik.model, ik.data)

        # EE position and orientation
        ee_pos = ik.data.site_xpos[ik.ee_site_id].copy().astype(np.float32)
        ee_rot = ik.data.site_xmat[ik.ee_site_id].reshape(3, 3).copy()

        # Rotation matrix to axis-angle (LIBERO format)
        axis_angle = Rotation.from_matrix(ee_rot).as_rotvec().astype(np.float32)

        # Gripper: both finger joints
        gripper_qpos = np.zeros(2, dtype=np.float32)
        n_arm = ik.num_arm_joints
        for i, jname in enumerate(all_joints[n_arm:n_arm + 2]):
            gripper_qpos[i] = self._joint_positions.get(jname, 0.0)

        # Convert EE pos to policy's world frame (add offset)
        ee_pos_world = ee_pos + ik.world_offset.astype(np.float32)

        # Build state: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
        state = np.concatenate([ee_pos_world, axis_angle, gripper_qpos])

        # Pad to max_state_dim (20 for XVLA)
        if len(state) < state_dim:
            state = np.pad(state, (0, state_dim - len(state)))

        return state[:state_dim]

    def _get_joint_order(self) -> list[str]:
        """Determine joint order, respecting config or auto-detecting from ZMQ."""
        if self.cfg.joint_names:
            return self.cfg.joint_names

        # Sort by ZMQ ID for consistent ordering
        return sorted(
            self._joint_positions.keys(),
            key=lambda n: self._joint_ids.get(n, 999),
        )

    def _build_observation(self) -> dict:
        """
        Build a LeRobot-compatible observation dict from current ZMQ state.

        For EE-action policies (ee6d), the state is computed as
        [ee_pos(3), ee_quat(4), gripper(1)] via MuJoCo FK instead of
        raw joint angles — matching LIBERO's observation format.

        Returns dict with keys like:
            "observation.state"  -> np.ndarray of joint positions or EE state
            "observation.images.{name}" -> np.ndarray (H, W, 3) uint8 RGB
        """
        obs = {}

        # Determine expected state dimension from policy config
        state_feature = self.policy.config.input_features.get("observation.state")
        expected_state_dim = state_feature.shape[0] if state_feature else None

        joint_order = self._get_joint_order()

        # Truncate to policy's expected dimension if needed
        if expected_state_dim and len(joint_order) > expected_state_dim:
            joint_order = joint_order[:expected_state_dim]

        # For EE-action policies, compute EE state via FK
        if self._ee_ik is not None:
            state = self._compute_ee_state(joint_order, expected_state_dim or 8)
        else:
            state = np.array(
                [self._joint_positions.get(n, 0.0) for n in joint_order],
                dtype=np.float32,
            )

        # Pad if we have fewer values than expected
        if expected_state_dim and len(state) < expected_state_dim:
            state = np.pad(state, (0, expected_state_dim - len(state)))

        obs["observation.state"] = state

        # Camera images — map URLab names to LeRobot observation keys
        # Determine which image keys the policy expects
        expected_image_keys = [
            k for k, v in self.policy.config.input_features.items()
            if v.type.name == "VISUAL"
        ]

        if self.cfg.camera_map:
            for urlab_name, lerobot_key in self.cfg.camera_map.items():
                if urlab_name in self._images:
                    bgra = self._images[urlab_name]
                    rgb = bgra[:, :, [2, 1, 0]]
                    obs[lerobot_key] = rgb
            self.active_camera_map = dict(self.cfg.camera_map)
        elif self._images and expected_image_keys:
            # Smart auto-map: match cameras by name heuristics
            available = list(self._images.keys())
            new_map = {}

            overhead_names = ["overhead", "agentview", "front", "top", "scene"]
            wrist_names = ["wrist", "hand", "gripper", "eye_in_hand"]

            for expected_key in expected_image_keys:
                if "empty" in expected_key:
                    continue  # skip empty camera slots

                matched = None
                # Determine what type of camera this key expects
                key_lower = expected_key.lower()
                is_wrist_key = any(w in key_lower for w in ["image2", "wrist", "hand"])
                is_overhead_key = any(w in key_lower for w in ["image", "top", "front", "scene", "agentview"])

                if is_wrist_key and not is_overhead_key:
                    for cam in available:
                        if any(w in cam.lower() for w in wrist_names):
                            matched = cam
                            break
                else:
                    for cam in available:
                        if any(w in cam.lower() for w in overhead_names):
                            matched = cam
                            break

                if matched is None and available:
                    # Fallback: use first unmatched camera that isn't a wrist cam
                    # (avoids mapping wrist to overhead slot before overhead camera arrives)
                    used = set(new_map.keys())
                    for cam in available:
                        if cam not in used and not any(w in cam.lower() for w in wrist_names):
                            matched = cam
                            break
                    # If still nothing, use any unmatched
                    if matched is None:
                        for cam in available:
                            if cam not in used:
                                matched = cam
                            break

                if matched:
                    bgra = self._images[matched]
                    rgb = bgra[:, :, [2, 1, 0]]
                    obs[expected_key] = rgb
                    new_map[matched] = expected_key

            if new_map != self.active_camera_map:
                self.active_camera_map = new_map
                for urlab_name, lr_key in new_map.items():
                    logger.info(f"  Camera mapped: '{urlab_name}' -> '{lr_key}'")

        # Task instruction for language-conditioned policies
        if self.cfg.task:
            obs["task"] = self.cfg.task

        return obs

    def _send_action(self, action: np.ndarray, joint_order: list[str]):
        """Send action as joint position targets via ZMQ.

        Control-in is a msgpack `{ids:[...], vals:[...]}` payload parsed
        UE-side by FURLabMsgpackUtil (5.3). The legacy little-endian
        `[i32 n][i32 id, f32 val]*` binary format is retired -- the UE
        unsafe-cast parser was removed, so emitting it now mismatches.
        """
        n = len(action)

        # Build actuator ID list from joint order.
        name_to_id = {}
        if self._actuator_ids:
            name_to_id = self._actuator_ids
        else:
            # Fallback: use ZMQ joint IDs (or bare ordinal `i` per joint if
            # even those are absent). This is the path taken now that the
            # `:5557` actuator_list broadcast is gone (see _discover_actuator_ids).
            name_to_id = self._joint_ids

        ids = []
        vals = []
        for i in range(n):
            jname = joint_order[i] if i < len(joint_order) else f"joint_{i}"
            # Try exact match, then without _joint suffix, then ordinal.
            aid = name_to_id.get(jname,
                   name_to_id.get(jname.removesuffix("_joint"), i))
            ids.append(int(aid))
            vals.append(float(action[i]))

        payload = msgpack.packb({"ids": ids, "vals": vals}, use_bin_type=True)
        self._ctrl_pub.send_string(f"{self._prefix}/control ", zmq.SNDMORE)
        self._ctrl_pub.send(payload)

    def run(self, stop_event=None):
        """
        Main inference loop. Runs until stop_event is set or self._stop is True.

        Args:
            stop_event: optional threading.Event to signal stop from outside.
        """
        self.phase = "connecting"
        self.error = ""
        self.step_count = 0

        # Detect the target articulation from the state stream, then pull the
        # ordered component names from the info socket. state/full carries qpos
        # in discovery order with no names, so this ordering is what keys the
        # joint values -- it must be resolved before the joint-data wait below.
        if not self._prefix:
            t0 = time.time()
            while time.time() - t0 < 5.0:
                names = art_names(self._state.drain())
                if names:
                    self._prefix = self.cfg.articulation_prefix or names[0]
                    break
                if stop_event and stop_event.is_set():
                    return
                time.sleep(0.1)
        if self._prefix:
            logger.info(f"Target articulation: {self._prefix}")

        self._actuator_ids = self._discover_actuator_ids()
        if self._actuator_ids:
            self._art_names = [
                n for n, _ in sorted(self._actuator_ids.items(), key=lambda kv: kv[1])
            ]
            logger.info(f"Discovered {len(self._actuator_ids)} actuator IDs")

        # Wait for joint data
        logger.info("Waiting for state stream...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            self._drain_state()
            if self._joint_positions:
                break
            if stop_event and stop_event.is_set():
                return
            time.sleep(0.1)

        if not self._joint_positions:
            self.error = "No joint data received from ZMQ"
            self.phase = "error"
            logger.error(self.error)
            return

        logger.info(f"Received {len(self._joint_positions)} joints from '{self._prefix}'")

        # Wait for cameras to arrive
        num_expected_cams = sum(
            1 for k, v in self.policy.config.input_features.items()
            if v.type.name == "VISUAL" and "empty" not in k
        )
        if num_expected_cams > 0:
            logger.info(f"Waiting for cameras (expecting {num_expected_cams})...")
            t0 = time.time()
            while time.time() - t0 < 5.0:
                self._drain_cameras()
                if len(self._images) >= num_expected_cams:
                    break
                if len(self._images) >= 1:
                    # Give a bit more time for remaining cameras
                    time.sleep(0.5)
                    self._drain_cameras()
                    break
                time.sleep(0.1)
            logger.info(f"  Cameras available: {list(self._images.keys())}")

        # Determine joint order (truncated to policy's expected dim)
        joint_order = self._get_joint_order()
        state_feature = self.policy.config.input_features.get("observation.state")
        expected_state_dim = state_feature.shape[0] if state_feature else len(joint_order)
        if len(joint_order) > expected_state_dim:
            logger.info(f"Truncating {len(joint_order)} joints to policy's expected {expected_state_dim}")
            joint_order = joint_order[:expected_state_dim]
        self.active_joint_order = list(joint_order)
        logger.info(f"Joint order ({len(joint_order)}): {joint_order[:5]}...")

        # Determine action dimension and mode
        action_feature = self.policy.config.output_features.get("action")
        self._action_dim = action_feature.shape[0] if action_feature else len(joint_order)
        self._action_mode = getattr(self.policy.config, 'action_mode', 'joint').lower()
        self._ee_ik = None

        if self._action_mode == 'ee6d':
            logger.info("  Action mode: ee6d (end-effector) — initializing IK solver")
            mjcf = self.cfg.mjcf_path or str(
                Path.home() / "Documents" / "mujoco_menagerie" / "franka_emika_panda" / "mjx_panda_ue.xml"
            )
            if mjcf:
                # LIBERO world-frame offset: converts LIBERO absolute EE targets
                # to MuJoCo-relative (base at origin).
                # LIBERO Franka base is at ~[0, 0, 0.912] on a 0.8m table.
                # MuJoCo EE home = [0.088, 0, 0.826], LIBERO EE home ~ [0, 0, 1.05]
                libero_offset = np.array([-0.088, 0.0, 0.224])
                self._ee_ik = EEtoJointIK(
                    mjcf_path=mjcf, num_arm_joints=7,
                    world_offset=libero_offset,
                )
            else:
                logger.warning("  No MJCF found for IK — EE actions will be sent raw")
                logger.warning("  Set mjcf_path in config or place Franka XML in ~/Documents/mujoco_menagerie/")

        # Reset policy for new episode
        self.policy.reset()

        self.phase = "running"
        dt = 1.0 / self.cfg.freq
        logger.info(f"Running at {self.cfg.freq:.0f} Hz — {len(joint_order)} state DOFs, {self._action_dim} action DOFs")

        try:
            while True:
                if self._stop or (stop_event and stop_event.is_set()):
                    break

                step_start = time.time()

                # 1. Read latest state
                self._drain_state()
                self._drain_cameras()

                # 2. Build observation
                obs = self._build_observation()

                # 3. Run inference
                with torch.inference_mode():
                    # ImageNet normalization constants
                    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    is_xvla = self._action_mode == 'ee6d'

                    # Convert to tensors, normalize images, add batch dim
                    batch = {}
                    for key, val in obs.items():
                        if key == "task":
                            batch[key] = val
                            continue
                        t = torch.from_numpy(np.asarray(val))
                        if t.dtype == torch.uint8:
                            # Image: uint8 (H,W,C) -> float32 (C,H,W) in [0,1]
                            t = t.float().div(255.0).permute(2, 0, 1)

                            # Note: LIBERO flips images 180 for robosuite camera convention.
                            # Unreal cameras are already right-side-up — no flip needed.

                            # Resize to match policy's expected image shape
                            feat = self.policy.config.input_features.get(key)
                            if feat and len(feat.shape) == 3:
                                _, target_h, target_w = feat.shape
                                if t.shape[1] != target_h or t.shape[2] != target_w:
                                    t = torch.nn.functional.interpolate(
                                        t.unsqueeze(0), size=(target_h, target_w),
                                        mode="bilinear", align_corners=False,
                                    ).squeeze(0)

                            if is_xvla:
                                # Apply ImageNet normalization
                                t = (t - IMAGENET_MEAN) / IMAGENET_STD

                        t = t.unsqueeze(0).to(self.device)
                        batch[key] = t

                    # Add domain_id for XVLA (3 = LIBERO)
                    if is_xvla:
                        batch["domain_id"] = torch.tensor([3], dtype=torch.long, device=self.device)

                    # Ensure task is always present for VLA policies
                    if "task" not in batch:
                        batch["task"] = self.cfg.task or "do the task"

                    # Tokenize language for policies that expect pre-tokenized input
                    if self.step_count == 0:
                        logger.info(f"  Batch keys before tokenize: {list(batch.keys())}")
                        logger.info(f"  Has tokenizer: {self._language_tokenizer is not None}")
                    if self._language_tokenizer is not None:
                        task_text = batch.pop("task", "do the task")
                        if isinstance(task_text, str):
                            task_text = [task_text]
                        # Use short padding — long padding creates too many tokens
                        # and overflows the action transformer's max_len_seq
                        tokenized = self._language_tokenizer(
                            task_text,
                            padding="longest",
                            truncation=True,
                            max_length=64,
                            return_tensors="pt",
                        )
                        batch["observation.language.tokens"] = tokenized["input_ids"].to(self.device)
                    elif "task" in batch:
                        batch.pop("task")  # non-VLA policies don't need it

                    # Apply preprocessor
                    batch = self.preprocessor(batch)

                    # Get action
                    action = self.policy.select_action(batch)

                    # Apply postprocessor
                    action = self.postprocessor(action)

                action_np = action.squeeze(0).cpu().numpy()

                # Log first few steps for debugging
                if self.step_count < 3:
                    logger.info(f"  [step {self.step_count}] Raw action (first 10): {action_np[:10]}")
                    logger.info(f"  [step {self.step_count}] EE target pos: {action_np[0:3]}, gripper: {action_np[9] if len(action_np) > 9 else 'N/A'}")
                    if self._ee_ik is not None:
                        self._ee_ik.mujoco.mj_forward(self._ee_ik.model, self._ee_ik.data)
                        cur_ee = self._ee_ik.data.site_xpos[self._ee_ik.ee_site_id].copy()
                        logger.info(f"  [step {self.step_count}] Current EE pos: {cur_ee}")

                # 4. Convert EE actions to joint targets if needed
                if self._ee_ik is not None:
                    # Postprocess: 20D ee6d → 7D [pos(3), axis_angle(3), gripper(1)]
                    ee_pos = action_np[0:3]
                    rot6d = action_np[3:9]
                    gripper_raw = action_np[9]

                    # 6D rotation → axis-angle
                    a1 = rot6d[:3]
                    a2 = rot6d[3:6]
                    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
                    b2 = a2 - np.dot(b1, a2) * b1
                    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
                    b3 = np.cross(b1, b2)
                    rot_mat = np.stack([b1, b2, b3], axis=-1)
                    from scipy.spatial.transform import Rotation
                    axis_angle = Rotation.from_matrix(rot_mat).as_rotvec().astype(np.float32)

                    # Binarize gripper: >0.5 = open (1), <=0.5 = closed (-1)
                    gripper = 1.0 if gripper_raw > 0.5 else -1.0

                    # Build processed action: [pos(3), axis_angle(3), gripper(1)]
                    processed_action = np.concatenate([ee_pos, axis_angle, [gripper]])

                    if self.step_count < 3:
                        logger.info(f"  [step {self.step_count}] Processed: pos={ee_pos}, grip={gripper}")

                    # IK: absolute EE target → joint positions
                    current_joints = np.array(
                        [self._joint_positions.get(n, 0.0) for n in joint_order],
                        dtype=np.float32,
                    )
                    action_np = self._ee_ik.ee_to_joint(processed_action, current_joints)

                    # Safety: clamp total joint change per step
                    max_joint_step = 0.15  # ~8.5 degrees per step
                    delta = action_np[:len(current_joints)] - current_joints
                    delta = np.clip(delta, -max_joint_step, max_joint_step)
                    action_np[:len(current_joints)] = current_joints + delta

                # 5. Send action to Unreal
                self._send_action(action_np, joint_order)

                self.step_count += 1
                elapsed = time.time() - step_start
                self.freq_hz = 1.0 / max(elapsed, 0.001)

                # Sleep to maintain target frequency
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            import traceback
            self.error = str(e)
            self.phase = "error"
            logger.error(f"LeRobot runner error:\n{traceback.format_exc()}")
        finally:
            if not self.error:
                self.phase = "stopped"
            logger.info(f"LeRobot runner stopped after {self.step_count} steps")

    def stop(self):
        """Signal the runner to stop."""
        self._stop = True

    def shutdown(self):
        """Clean up ZMQ sockets."""
        self.stop()
        self._state.close()
        self._cam_sub.close()
        self._ctrl_pub.close()
        self._ctx.term()


def get_policy_info(pretrained_path: str) -> dict:
    """
    Fetch policy metadata without loading weights.
    Returns dict with policy_type, input_features, output_features, etc.
    """
    if not HAS_LEROBOT:
        return {"error": "lerobot not installed"}

    try:
        from huggingface_hub import hf_hub_download
        import json

        # Try loading config.json from hub or local
        p = Path(pretrained_path)
        if p.is_dir():
            config_path = p / "config.json"
        else:
            config_path = Path(hf_hub_download(pretrained_path, "config.json"))

        with open(config_path) as f:
            cfg = json.load(f)

        return {
            "policy_type": cfg.get("type", "unknown"),
            "input_features": cfg.get("input_features", {}),
            "output_features": cfg.get("output_features", {}),
            "n_obs_steps": cfg.get("n_obs_steps", 1),
        }
    except Exception as e:
        return {"error": str(e)}
