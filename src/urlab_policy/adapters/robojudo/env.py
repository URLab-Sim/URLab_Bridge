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

"""URLab <-> RoboJuDo environment glue.

``URLabRoboJuDoEnv`` is the URLabClient-backed env (use this). ``UnrealEnv``
is the legacy ZMQ-streaming variant kept for back-compat. ``URLabEnv`` is
a gymnasium adapter that doesn't depend on RoboJuDo.

When RoboJuDo isn't importable, only the RoboJuDo-free classes are exposed.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import zmq

from urlab_client import URLabArticulation, URLabClient
from urlab_client.enums import ObservationLevel, SpaceMode, StepMode, coerce, wire

from ..._state_stream import (
    StateStream,
    art_block,
    art_names,
    art_qpos,
    art_qvel,
    art_sensors,
    art_twist,
    free_base_state,
)

from .joint_specs import (
    G1_12DOF, G1_29DOF, GO2_12DOF,
    G1_12DOF_JOINT_NAMES, G1_29DOF_JOINT_NAMES, GO2_12DOF_JOINT_NAMES,
)


logger = logging.getLogger(__name__)


try:
    from robojudo.environment import Environment, env_registry
    from robojudo.environment.env_cfgs import EnvCfg
    from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg
    from robojudo.utils.util_func import quatToEuler, quat_rotate_inverse_np
    HAS_ROBOJUDO = True
except ImportError:
    HAS_ROBOJUDO = False

    # Minimal stand-in so the gym adapter and standalone helpers below
    # can still construct without RoboJuDo on the path.
    class Environment:  # type: ignore[no-redef]
        def __init__(self, cfg_env, device="cpu"):
            self.cfg_env = cfg_env
            self.device = device


try:  # pragma: no cover - import guard
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:
        gym = None  # type: ignore
        spaces = None  # type: ignore


# ---------------------------------------------------------------------------
# Legacy ZMQ PUB/SUB env (UnrealEnv + UnrealEnvCfg + per-robot configs)
# ---------------------------------------------------------------------------


class ZmqLink:
    """Manages ZMQ sockets for the streaming PUB/SUB Unreal transport.

    State is read from the canonical `state/full` msgpack snapshot; control
    is written on the raw ctrl PUB (unchanged)."""

    def __init__(self, state_endpoint: str, control_endpoint: str):
        self.ctx = zmq.Context()

        self.state = StateStream(self.ctx, state_endpoint)

        self.ctrl_pub = self.ctx.socket(zmq.PUB)
        self.ctrl_pub.connect(control_endpoint)
        time.sleep(0.2)

        logger.info(f"ZMQ state: {state_endpoint}  control: {control_endpoint}")

    def set_prefix_filter(self, prefix: str):
        # state/full carries every articulation in one snapshot; there is no
        # per-prefix wire filtering to apply. Kept for call-site compatibility.
        pass

    def snapshot(self) -> "dict | None":
        """Latest decoded `state/full` snapshot, or None if none yet."""
        return self.state.drain()

    def send_control(self, prefix: str, targets: np.ndarray,
                     actuator_ids: list[int] | None = None):
        n = len(targets)
        data = struct.pack("<i", n)
        for i, val in enumerate(targets):
            aid = actuator_ids[i] if actuator_ids else i
            data += struct.pack("<if", aid, float(val))
        self.ctrl_pub.send_string(f"{prefix}/control ", zmq.SNDMORE)
        self.ctrl_pub.send(data)

    def send_gains(self, prefix: str, joint_names: list[str],
                   kp: np.ndarray, kv: np.ndarray, torque_limits: np.ndarray):
        gains = {}
        for i, name in enumerate(joint_names):
            gains[name] = {
                "kp": float(kp[i]) if i < len(kp) else 100.0,
                "kv": float(kv[i]) if i < len(kv) else 5.0,
                "torque_limit": float(torque_limits[i]) if i < len(torque_limits) else 200.0,
            }
        self.ctrl_pub.send_string(f"{prefix}/set_gains ", zmq.SNDMORE)
        self.ctrl_pub.send_string(json.dumps(gains))

    def close(self):
        self.state.close()
        self.ctrl_pub.close()
        self.ctx.term()


def _detect_prefix(zmq_link: ZmqLink, forced_prefix: str = "",
                   timeout: float = 5.0) -> str:
    """Detect the target articulation prefix from the `state/full` snapshot.

    Returns the forced prefix if given (once the stream confirms it), the
    sole articulation when there is exactly one, or "" if nothing arrives."""
    if forced_prefix:
        logger.info(f"Using specified articulation prefix: '{forced_prefix}'")
    else:
        logger.info("Auto-detecting articulation prefix from state stream...")

    start = time.time()
    while time.time() - start < timeout:
        names = art_names(zmq_link.snapshot())
        if names:
            if forced_prefix:
                if forced_prefix in names:
                    return forced_prefix
            else:
                logger.info(f"  Detected prefix: '{names[0]}'")
                return names[0]
        time.sleep(0.05)

    logger.warning("Could not detect prefix -- no state snapshot received")
    return forced_prefix or ""


if HAS_ROBOJUDO:

    class UnrealEnvCfg(EnvCfg):
        """RoboJuDo-compatible config for the legacy ZMQ-streaming env."""

        env_type: str = "UnrealEnv"
        is_sim: bool = True
        xml: str = ""

        # ZMQ endpoints
        state_endpoint: str = "tcp://127.0.0.1:5555"
        control_endpoint: str = "tcp://127.0.0.1:5556"

        # Which articulation to target (auto-detected if empty)
        articulation_prefix: str = ""

        # Timing: must match Unreal's MuJoCo timestep
        sim_dt: float = 0.002
        sim_decimation: int = 10

    class G1UnrealEnvCfg(UnrealEnvCfg):
        """G1 12-DOF for Unitree locomotion policy."""

        articulation_prefix: str = "g1"
        dof: DoFConfig = G1_12DOF
        forward_kinematic: ForwardKinematicCfg | None = None
        update_with_fk: bool = False
        torso_name: str = "pelvis"

    # Path to the G1 29DOF XML for forward kinematics. parents[4] from
    # this file -> URLab_Bridge/ root.
    _G1_29_XML = (
        Path(__file__).resolve().parents[4]
        / "RoboJuDo" / "assets" / "robots" / "g1" / "g1_29dof_rev_1_0.xml"
    ).as_posix()

    class G1_29UnrealEnvCfg(UnrealEnvCfg):
        """G1 29-DOF for full-body policies (BeyondMimic, H2H, AMO, ...)."""

        articulation_prefix: str = "g1"
        dof: DoFConfig = G1_29DOF
        forward_kinematic: ForwardKinematicCfg = ForwardKinematicCfg(
            xml_path=_G1_29_XML,
            debug_viz=False,
            kinematic_joint_names=G1_29DOF_JOINT_NAMES,
        )
        update_with_fk: bool = True
        torso_name: str = "torso_link"

    class Go2UnrealEnvCfg(UnrealEnvCfg):
        """Go2 12-DOF for walk-these-ways locomotion policy."""

        articulation_prefix: str = "go2"
        dof: DoFConfig = GO2_12DOF
        forward_kinematic: ForwardKinematicCfg | None = None
        update_with_fk: bool = False
        torso_name: str = "base"
        sim_dt: float = 0.005
        sim_decimation: int = 4

    @env_registry.register
    class UnrealEnv(Environment):
        """RoboJuDo Environment over the legacy ZMQ PUB/SUB streaming
        transport. Auto-detects articulation prefix and joint mapping at
        startup."""

        cfg_env: UnrealEnvCfg

        def __init__(self, cfg_env: UnrealEnvCfg, device="cpu"):
            super().__init__(cfg_env=cfg_env, device=device)

            self._base_pos = np.zeros(3)
            self._base_lin_vel = np.zeros(3)
            self._torso_pos = np.zeros(3)
            self._torso_quat = np.array([0.0, 0.0, 0.0, 1.0])
            self._torso_ang_vel = np.zeros(3)

            self.zmq = ZmqLink(cfg_env.state_endpoint, cfg_env.control_endpoint)
            self.control_dt = cfg_env.sim_dt * cfg_env.sim_decimation

            self._connected = False
            self._last_update = 0.0
            self._base_state_received = False
            self._twist_cmd = np.zeros(3)
            self._force_twist = None

            forced = getattr(cfg_env, "articulation_prefix", "") or ""
            self.prefix = _detect_prefix(self.zmq, forced_prefix=forced)

            self._cfg_env = cfg_env
            # Ordered actuator short-names (info-socket discovery order) and the
            # per-DoF index into the art's state/full qpos block.
            self._ordered_act_names: list[str] = []
            self._dof_to_stream_idx: list[int] = []
            self._actuator_ids = self._discover_actuator_ids(cfg_env)
            self._build_stream_mapping()

            self.zmq.set_prefix_filter(self.prefix)

            logger.info(
                f"UnrealEnv ready -- prefix='{self.prefix}', "
                f"{self.num_dofs} DOFs, control_dt={self.control_dt:.4f}s "
                f"({1.0 / self.control_dt:.0f}Hz)"
            )

        def _build_stream_mapping(self):
            """Map each policy DoF to the ordinal of its actuator within the
            info-socket discovery order. state/full concatenates qpos in that
            same order, so update() reads qpos[free_offset + ordinal]."""
            ordinal = {n: k for k, n in enumerate(self._ordered_act_names)}
            self._dof_to_stream_idx = []
            for name in self.joint_names:
                k = ordinal.get(name)
                if k is None:
                    k = ordinal.get(name.removesuffix("_joint"), -1)
                if k < 0:
                    logger.warning(f"  Joint '{name}' not found in actuator order")
                self._dof_to_stream_idx.append(k)

        def update_dof_cfg(self, override_cfg=None):
            super().update_dof_cfg(override_cfg)
            if hasattr(self, "_cfg_env"):
                self._actuator_ids = self._discover_actuator_ids(self._cfg_env)
                self._build_stream_mapping()
                logger.info(
                    f"DOF config updated -- {self.num_dofs} DOFs, "
                    f"joints: {self.joint_names[:3]}..."
                )

        def _discover_actuator_ids(self, cfg_env) -> list[int] | None:
            info_endpoint = getattr(cfg_env, "info_endpoint", "tcp://127.0.0.1:5557")
            ctx = zmq.Context()
            info_sub = ctx.socket(zmq.SUB)
            info_sub.connect(info_endpoint)
            info_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            info_sub.setsockopt(zmq.RCVTIMEO, 5000)

            actuator_ids = None
            try:
                for _ in range(20):
                    try:
                        payload = info_sub.recv().decode("utf-8")
                        data = json.loads(payload)
                        if data.get("type") == "actuator_list" and data.get("robot") == self.prefix:
                            names = data.get("names", [])
                            ids = data.get("ids", [])
                            name_to_id: dict[str, int] = {}
                            for n, i in zip(names, ids):
                                short = n.replace(self.prefix + "_", "", 1)
                                name_to_id[short] = int(i)

                            # Actuator order == state/full joint discovery order.
                            self._ordered_act_names = [
                                n for n, _ in sorted(name_to_id.items(), key=lambda kv: kv[1])
                            ]

                            actuator_ids = []
                            for jname in self.joint_names:
                                if jname in name_to_id:
                                    actuator_ids.append(name_to_id[jname])
                                elif jname.removesuffix("_joint") in name_to_id:
                                    actuator_ids.append(name_to_id[jname.removesuffix("_joint")])
                                else:
                                    logger.warning(f"  Actuator for joint '{jname}' not found in info")
                                    actuator_ids.append(-1)

                            logger.info(f"  Discovered actuator IDs: {actuator_ids}")
                            break
                    except zmq.Again:
                        break
            finally:
                info_sub.close()
                ctx.term()

            return actuator_ids

        def self_check(self):
            logger.info("Running self-check...")
            for i in range(50):
                self.update()
                if self._connected:
                    logger.info(f"Self-check passed -- data flowing after {i * 100}ms")
                    return
                time.sleep(0.1)
            logger.warning("Self-check: no data in 5s")

        def reset(self):
            self._dof_pos = np.zeros(self.num_dofs)
            self._dof_vel = np.zeros(self.num_dofs)
            self._base_quat = np.array([0.0, 0.0, 0.0, 1.0])
            self._base_ang_vel = np.zeros(3)
            self._base_pos = np.zeros(3)
            self._base_lin_vel = np.zeros(3)

            if self.born_place_align:
                self.update()
                self.set_born_place()
                self.update()

        def set_born_place(self, quat=None, pos=None):
            super().set_born_place(
                quat if quat is not None else self.base_quat,
                pos if pos is not None else self.base_pos,
            )

        def set_gains(self, stiffness, damping):
            self.stiffness = np.asarray(stiffness)
            self.damping = np.asarray(damping)

        def update(self, simple=False):
            snap = self.zmq.snapshot()
            block = art_block(snap, self.prefix) if snap is not None else None
            if block is not None:
                qpos = art_qpos(snap, self.prefix)
                qvel = art_qvel(snap, self.prefix)
                # A leading free joint (if any) occupies the head of qpos/qvel;
                # the actuated joints are the tail, aligned to actuator order.
                na = len(self._ordered_act_names)
                qpos_off = max(0, qpos.size - na)
                qvel_off = max(0, qvel.size - na)
                for dof_idx, k in enumerate(self._dof_to_stream_idx):
                    if k < 0:
                        continue
                    pi, vi = qpos_off + k, qvel_off + k
                    if pi < qpos.size:
                        self._dof_pos[dof_idx] = qpos[pi]
                    if vi < qvel.size:
                        self._dof_vel[dof_idx] = qvel[vi]
                    self._connected = True

                # Root/base state from the leading free joint. qpos/qvel are raw
                # MuJoCo frame (the retired base_state binary carried the same
                # slots), matching the state/full sensors, which are raw MuJoCo SI
                # too; base state is still derived here from qpos/qvel rather than
                # from framepos/framequat sensors.
                fb = free_base_state(qpos, qvel)
                if fb is not None:
                    pos, quat_xyzw, lin_vel, ang_vel = fb
                    if self.born_place_align:
                        quat_xyzw, pos = self.base_align.align_transform(quat_xyzw, pos)
                    self._base_pos = pos
                    self._base_quat = quat_xyzw
                    self._base_lin_vel = lin_vel
                    self._base_ang_vel = ang_vel
                    self._connected = True
                    if not self._base_state_received:
                        self._base_state_received = True
                        logger.info(
                            f"Base state received! pos={self._base_pos}, "
                            f"quat={self._base_quat}, angvel={self._base_ang_vel}"
                        )

                # Twist command: (linear xyz, angular xyz) -> (lin.x, lin.y, ang.z).
                tw = art_twist(snap, self.prefix)
                if tw is not None:
                    self._twist_cmd = np.array([tw[0], tw[1], tw[5]])

            self._last_update = time.time()

            if (not simple and self.update_with_fk and self.kinematics is not None
                    and self._connected):
                try:
                    fk_info = self.fk()
                    self._fk_info = fk_info.copy()
                    if self._torso_name in fk_info:
                        self._torso_pos = fk_info[self._torso_name]["pos"]
                        self._torso_quat = fk_info[self._torso_name]["quat"]
                        self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
                except Exception:
                    pass

        def step(self, pd_target, hand_pose=None):
            assert len(pd_target) == self.num_dofs, (
                f"pd_target len {len(pd_target)} != num_dofs {self.num_dofs}"
            )
            self.zmq.send_control(self.prefix, pd_target, self._actuator_ids)

        @property
        def twist_cmd(self) -> np.ndarray:
            if self._force_twist is not None:
                return self._force_twist
            return self._twist_cmd

        def shutdown(self):
            self.zmq.close()
            logger.info("UnrealEnv shut down")


else:  # pragma: no cover - RoboJuDo-less fallback for legacy cfgs

    @dataclass
    class UnrealEnvCfg:
        state_endpoint: str = "tcp://127.0.0.1:5555"
        control_endpoint: str = "tcp://127.0.0.1:5556"
        articulation_prefix: str = ""
        joint_names: list[str] = field(default_factory=list)

    @dataclass
    class G1UnrealEnvCfg(UnrealEnvCfg):
        joint_names: list[str] = field(default_factory=lambda: G1_12DOF_JOINT_NAMES)


class UnrealEnvStandalone:
    """Minimal ZMQ bridge for testing without RoboJuDo. Used by ad-hoc
    diagnostic scripts -- normal callers should prefer ``URLabClient``."""

    def __init__(self, cfg_env: "UnrealEnvCfg"):
        self.cfg = cfg_env
        self.zmq = ZmqLink(cfg_env.state_endpoint, cfg_env.control_endpoint)
        self.prefix = cfg_env.articulation_prefix
        self.num_dofs = len(getattr(cfg_env, "joint_names", [])) or 12

        self._dof_pos = np.zeros(self.num_dofs)
        self._dof_vel = np.zeros(self.num_dofs)
        self._connected = False

    @property
    def connected(self):
        return self._connected

    def update(self):
        snap = self.zmq.snapshot()
        if snap is None:
            return
        prefix = self.prefix or (art_names(snap)[0] if art_names(snap) else "")
        if not prefix:
            return
        self.prefix = prefix
        qpos = art_qpos(snap, prefix)
        qvel = art_qvel(snap, prefix)
        n = min(self.num_dofs, qpos.size, qvel.size)
        if n > 0:
            self._dof_pos[:n] = qpos[:n]
            self._dof_vel[:n] = qvel[:n]
            self._connected = True

    def step(self, pd_target):
        self.zmq.send_control(self.prefix, pd_target)

    def shutdown(self):
        self.zmq.close()


# ---------------------------------------------------------------------------
# URLabClient-backed RoboJuDo env (canonical -- replaces UnrealEnv above)
# ---------------------------------------------------------------------------


if HAS_ROBOJUDO:

    class URLabRoboJuDoEnvCfg(EnvCfg):
        """RoboJuDo-compatible config for the URLabClient-backed env."""

        env_type: str = "URLabRoboJuDoEnv"
        is_sim: bool = True
        xml: str = ""

        # URLabClient connection
        address: str = "tcp://localhost"
        step_port: int = 5559
        state_port: int = 5555

        # Step semantics. The mode is the whole point of this port:
        #  - "direct": synchronous; bridge sends ctrl + n_substeps, UE
        #    advances exactly that many physics steps and returns.
        #  - "live": UE runs physics autonomously; bridge sends ctrl
        #    per RPC and reads the latest snapshot.
        #  - "puppet": bridge owns the integrator -- not yet supported.
        step_mode: str = "direct"

        sim_dt: float = 0.002
        sim_decimation: int = 10

        articulation_prefix: str = ""

        # Observation level for `client.step(observations=...)`. "full"
        # includes per-body xpos/xquat (needed if FK / torso state is on).
        observation_level: str = "full"

        push_gains_to_unreal: bool = True

        # Wire transport for URLabClient. "zmq" goes over TCP loopback;
        # "shm" uses the same-host shared-memory ring.
        transport: str = "zmq"

        # Optional explicit SHM session directory. Empty -> use the path
        # the UE handshake reports (`shm_session_dir`).
        shm_dir: str = ""

        # If set, reuse this URLabClient instead of constructing one — UE's
        # dispatcher tracks one active session, so a second client would
        # expire the first. connect() and close() are skipped when set.
        existing_client: Any = None

    class G1URLabRoboJuDoEnvCfg(URLabRoboJuDoEnvCfg):
        """G1 12-DoF (locomotion-style policies)."""

        articulation_prefix: str = "g1"
        dof: DoFConfig = G1_12DOF
        forward_kinematic: ForwardKinematicCfg | None = None
        update_with_fk: bool = False
        torso_name: str = "pelvis"

    class G1_29URLabRoboJuDoEnvCfg(URLabRoboJuDoEnvCfg):
        """G1 29-DoF (BeyondMimic, AMO, H2H, ...)."""

        articulation_prefix: str = "g1"
        dof: DoFConfig = G1_29DOF
        forward_kinematic: ForwardKinematicCfg = ForwardKinematicCfg(
            xml_path=_G1_29_XML,
            debug_viz=False,
            kinematic_joint_names=G1_29DOF_JOINT_NAMES,
        )
        update_with_fk: bool = True
        torso_name: str = "torso_link"

    class Go2URLabRoboJuDoEnvCfg(URLabRoboJuDoEnvCfg):
        """Go2 12-DoF (walk-these-ways / WTW)."""

        articulation_prefix: str = "go2"
        dof: DoFConfig = GO2_12DOF
        forward_kinematic: ForwardKinematicCfg | None = None
        update_with_fk: bool = False
        torso_name: str = "base"
        sim_dt: float = 0.005
        sim_decimation: int = 4

    @env_registry.register
    class URLabRoboJuDoEnv(Environment):
        """RoboJuDo Environment backed by URLabClient.

        Drop-in replacement for ``UnrealEnv``. Once constructed every
        ``RlPipeline`` / ``RlLocoMimicPipeline`` consumer just sees the
        same ``update()`` / ``step()`` / ``get_data()`` contract -- the
        policies don't know they're talking to a different transport.
        """

        cfg_env: "URLabRoboJuDoEnvCfg"

        def __init__(self, cfg_env: "URLabRoboJuDoEnvCfg", device: str = "cpu"):
            super().__init__(cfg_env=cfg_env, device=device)

            self._base_pos = np.zeros(3)
            self._base_lin_vel = np.zeros(3)
            self._base_lin_acc = np.zeros(3)
            self._torso_pos = np.zeros(3)
            self._torso_quat = np.array([0.0, 0.0, 0.0, 1.0])
            self._torso_ang_vel = np.zeros(3)
            self._twist_cmd = np.zeros(3)
            self._force_twist: Optional[np.ndarray] = None
            self._twist_full = np.zeros(6)
            self._actions: int = 0
            self._twist_log_t0 = 0.0
            self._sim_time_sec: int = 0
            self._sim_time_nsec: int = 0
            self._publish_time_sec: int = 0
            self._publish_time_nsec: int = 0
            self._transport_latency_ns: int = 0

            if cfg_env.existing_client is not None:
                # Reuse an already-connected client (dashboard / Notebook
                # integration). The caller owns the lifetime; shutdown()
                # below skips close().
                self.client = cfg_env.existing_client
                self._owns_client = False
                logger.info(
                    "URLabRoboJuDoEnv: reusing existing client (session=%s, "
                    "step_mode=%s, manager_present=%s)",
                    self.client.session_id[:8] if self.client.session_id else "?",
                    self.client.step_mode.value,
                    self.client.manager_present,
                )
            else:
                self.client = URLabClient(
                    cfg_env.address,
                    step_mode=cfg_env.step_mode,
                    step_port=cfg_env.step_port,
                    state_port=cfg_env.state_port,
                    auto_promote_step_mode=True,
                    transport=cfg_env.transport,
                    shm_dir=cfg_env.shm_dir or None,
                )
                self._owns_client = True
                logger.info(
                    "URLabRoboJuDoEnv: connecting to %s (step_mode=%s, transport=%s)",
                    cfg_env.address, cfg_env.step_mode, cfg_env.transport,
                )
                self.client.connect(observations=cfg_env.observation_level)

            try:
                self.client.runtime.set_sim_options(timestep=cfg_env.sim_dt)
                logger.info(
                    "URLabRoboJuDoEnv: pushed sim timestep=%.4fs to UE",
                    cfg_env.sim_dt,
                )
            except Exception as exc:
                logger.warning(
                    "URLabRoboJuDoEnv: set_sim_options failed: %s "
                    "(UE will use its compiled timestep)", exc,
                )

            self.prefix = self._resolve_articulation_prefix()
            self.art = self.client.articulations[self.prefix]
            logger.info(
                "URLabRoboJuDoEnv: articulation '%s' (%d joints, %d actuators, %d sensors)",
                self.prefix, len(self.art.joints), len(self.art.actuators),
                len(self.art.sensors),
            )

            self.client.runtime.claim_control(self.prefix)
            logger.info("URLabRoboJuDoEnv: claimed control of '%s'", self.prefix)

            self._dof_to_joint: List[Optional[str]] = []
            self._dof_to_actuator: List[Optional[str]] = []
            self._build_dof_mapping()

            if not self.art.has_free_base:
                logger.info(
                    "URLabRoboJuDoEnv: no free-base joint detected on '%s' "
                    "-- base/torso state will stay zero (fixed-base robot)",
                    self.prefix,
                )

            if cfg_env.push_gains_to_unreal:
                self._sync_gains_to_unreal(self.stiffness, self.damping)

            self._ctrl_zero = np.zeros(len(self.art.actuators), dtype=np.float64)
            self.art.ctrl_array[:] = self._ctrl_zero

            self._connected = False
            self.control_dt = cfg_env.sim_dt * cfg_env.sim_decimation
            logger.info(
                "URLabRoboJuDoEnv ready (control_dt=%.4fs / %.0fHz, num_dofs=%d, "
                "free_base=%s, fk=%s)",
                self.control_dt, 1.0 / self.control_dt, self.num_dofs,
                self.art.has_free_base, self.update_with_fk,
            )

        def _resolve_articulation_prefix(self) -> str:
            arts = list(self.client.articulations.keys())
            if not arts:
                raise RuntimeError(
                    "URLabRoboJuDoEnv: handshake exposed no articulations"
                )
            wanted = self.cfg_env.articulation_prefix or ""
            if wanted:
                if wanted in arts:
                    return wanted
                raise RuntimeError(
                    f"URLabRoboJuDoEnv: articulation_prefix={wanted!r} not in "
                    f"handshake (available: {arts})"
                )
            if len(arts) == 1:
                return arts[0]
            raise RuntimeError(
                f"URLabRoboJuDoEnv: handshake exposes multiple articulations "
                f"({arts}); set cfg.articulation_prefix to disambiguate"
            )

        def _build_dof_mapping(self) -> None:
            self._dof_to_joint.clear()
            self._dof_to_actuator.clear()
            for jname in self.joint_names:
                j_key = self.art.resolve_joint(jname)
                a_key = self.art.resolve_actuator(jname)
                if j_key is None:
                    logger.warning(
                        "URLabRoboJuDoEnv: dof joint %r not found in art.joints; "
                        "state will be zero for this dof", jname,
                    )
                if a_key is None:
                    logger.warning(
                        "URLabRoboJuDoEnv: dof actuator for %r not found; "
                        "ctrl will be dropped for this dof", jname,
                    )
                self._dof_to_joint.append(j_key)
                self._dof_to_actuator.append(a_key)

        def _sync_gains_to_unreal(self, stiffness: np.ndarray, damping: np.ndarray) -> None:
            if self.art.controller is None:
                logger.info(
                    "URLabRoboJuDoEnv: '%s' has no controller in handshake; "
                    "skipping gain sync (UE-side gains stay as configured)",
                    self.prefix,
                )
                return
            try:
                pushed = self.art.push_gains(self.joint_names, stiffness, damping)
                logger.info(
                    "URLabRoboJuDoEnv: pushed PD gains to UE (%d joints)", pushed,
                )
            except Exception as exc:
                logger.warning(
                    "URLabRoboJuDoEnv: failed to push gains to UE: %s "
                    "(continuing with whatever UE already has)", exc,
                )

        def self_check(self) -> None:
            logger.info("URLabRoboJuDoEnv: running self-check (10 steps)...")
            for i in range(10):
                self.step(self.default_pos.copy())
                if self._connected and (self._dof_pos != 0).any():
                    logger.info(
                        "URLabRoboJuDoEnv: self-check passed after %d steps", i + 1,
                    )
                    return
            logger.warning("URLabRoboJuDoEnv: self-check saw no nonzero dof_pos")

        def reset(self) -> None:
            self._dof_pos[:] = 0.0
            self._dof_vel[:] = 0.0
            self._base_quat = np.array([0.0, 0.0, 0.0, 1.0])
            self._base_ang_vel[:] = 0.0
            self._base_pos[:] = 0.0
            self._base_lin_vel[:] = 0.0

            try:
                self.client.reset()
            except Exception as exc:
                logger.warning("URLabRoboJuDoEnv: reset RPC failed: %s", exc)

            self.update()

            if self.born_place_align:
                self.set_born_place()
                self.update()

        def reborn(self, init_qpos=None) -> None:
            """SIM_REBORN entry point (pipeline calls when the user fires
            the command). `init_qpos` accepted for parity with MujocoEnv
            but ignored for now -- client.reset() resets to keyframe / default."""
            self.reset()

        def update(self, simple: bool = False) -> None:
            art_joints = self.art.joints

            for dof_idx, j_key in enumerate(self._dof_to_joint):
                if j_key is None:
                    continue
                j = art_joints[j_key]
                self._dof_pos[dof_idx] = float(self.art.qpos_array[j.qpos_local_offset])
                self._dof_vel[dof_idx] = float(self.art.qvel_array[j.qvel_local_offset])

            self._connected = True

            self._twist_full[:3] = self.art.twist_linear
            self._twist_full[3:] = self.art.twist_angular
            self._actions = self.art.actions
            self._twist_cmd[0] = self.art.twist_linear[0]
            self._twist_cmd[1] = self.art.twist_linear[1]
            self._twist_cmd[2] = self.art.twist_angular[2]
            if logger.isEnabledFor(logging.DEBUG):
                self._maybe_log_twist()

            self._sim_time_sec = self.client.sim_time_sec
            self._sim_time_nsec = self.client.sim_time_nsec
            self._publish_time_sec = self.client.wall_time_sec
            self._publish_time_nsec = self.client.wall_time_nsec
            publish_ns = self._publish_time_sec * 1_000_000_000 + self._publish_time_nsec
            self._transport_latency_ns = max(0, self.client.recv_wall_time_ns - publish_ns)

            if simple:
                return

            if self.art.has_free_base:
                base_pos = self.art.root_pos_w.copy()
                quat_xyzw = self.art.root_quat_xyzw
                lin_vel = self.art.root_lin_vel_w
                # MuJoCo free joint stores ang_vel in BODY frame (asymmetric
                # with lin_vel); stored raw below, no rotation.
                ang_vel = self.art.root_ang_vel_b

                if self.born_place_align:
                    quat_xyzw, base_pos = self.base_align.align_transform(quat_xyzw, base_pos)

                lin_vel_body = quat_rotate_inverse_np(quat_xyzw, lin_vel)

                self._base_pos[:] = base_pos
                self._base_quat = quat_xyzw
                self._base_ang_vel[:] = ang_vel
                self._base_lin_vel[:] = lin_vel_body
                self._base_rpy = quatToEuler(quat_xyzw)

            if self.update_with_fk and self.kinematics is not None:
                try:
                    fk_info = self.fk()
                    self._fk_info = fk_info.copy() if fk_info else None
                    if self._fk_info and self._torso_name in self._fk_info:
                        torso = self._fk_info[self._torso_name]
                        self._torso_pos[:] = torso["pos"]
                        self._torso_quat = np.asarray(torso["quat"])
                        self._torso_ang_vel[:] = torso["ang_vel"]
                except Exception as exc:
                    logger.debug("URLabRoboJuDoEnv: fk() failed: %s", exc)

        def step(self, pd_target: np.ndarray, hand_pose=None) -> None:
            assert len(pd_target) == self.num_dofs, (
                f"pd_target len {len(pd_target)} != num_dofs {self.num_dofs}"
            )
            if hand_pose is not None:
                logger.debug("URLabRoboJuDoEnv: hand_pose received but not yet wired")

            ctrl_map = {
                a_key: float(pd_target[dof_idx])
                for dof_idx, a_key in enumerate(self._dof_to_actuator)
                if a_key is not None
            }
            self.art.set_ctrl(ctrl_map)

            self.client.step(
                n_steps=self.cfg_env.sim_decimation,
                observations=self.cfg_env.observation_level,
            )

            self.update(simple=False)

        def shutdown(self) -> None:
            # Only close the client if we own it. Dashboard / Notebook
            # integrations pass an existing client via cfg_env.existing_client;
            # closing that would tear down their connection too.
            if getattr(self, "_owns_client", True):
                try:
                    self.client.close()
                except Exception:
                    logger.exception("URLabRoboJuDoEnv client close failed")
            logger.info("URLabRoboJuDoEnv shut down")

        def set_gains(self, stiffness, damping) -> None:
            self.stiffness = np.asarray(stiffness)
            self.damping = np.asarray(damping)
            if not getattr(self, "art", None):
                return
            if getattr(self.cfg_env, "push_gains_to_unreal", False):
                self._sync_gains_to_unreal(self.stiffness, self.damping)

        def update_dof_cfg(self, override_cfg=None) -> None:
            super().update_dof_cfg(override_cfg)
            if getattr(self, "art", None):
                self._build_dof_mapping()
                logger.info(
                    "URLabRoboJuDoEnv: dof config updated (%d dofs)",
                    self.num_dofs,
                )

        def set_born_place(self, quat: Optional[np.ndarray] = None,
                           pos: Optional[np.ndarray] = None) -> None:
            super().set_born_place(
                quat if quat is not None else self.base_quat,
                pos if pos is not None else self.base_pos,
            )

        def _maybe_log_twist(self) -> None:
            now = time.monotonic()
            if now - self._twist_log_t0 < 1.0:
                return
            self._twist_log_t0 = now
            if (np.any(self.art.twist_linear) or np.any(self.art.twist_angular)
                    or self.art.actions):
                logger.debug(
                    "art.twist: linear=%s angular=%s actions=0x%X",
                    self.art.twist_linear.tolist(),
                    self.art.twist_angular.tolist(),
                    self.art.actions,
                )

        @property
        def twist_cmd(self) -> np.ndarray:
            if self._force_twist is not None:
                return self._force_twist
            return self._twist_cmd

        @property
        def twist(self) -> np.ndarray:
            return self._twist_full

        @property
        def actions(self) -> int:
            return self._actions

        @property
        def sim_time_ns(self) -> int:
            return self._sim_time_sec * 1_000_000_000 + self._sim_time_nsec

        @property
        def publish_time_ns(self) -> int:
            return self._publish_time_sec * 1_000_000_000 + self._publish_time_nsec

        @property
        def transport_latency_ns(self) -> int:
            return self._transport_latency_ns


# ---------------------------------------------------------------------------
# Gymnasium adapter (no RoboJuDo dependency)
# ---------------------------------------------------------------------------


RewardFn = Callable[[Dict[str, Any]], float]
TerminationFn = Callable[[Dict[str, Any]], bool]


def _articulation_obs_size(art: URLabArticulation) -> int:
    """Length of the flat per-articulation observation vector.

    Layout: qpos | qvel | concatenated sensor readings (in discovery
    order).
    """
    qpos_n = len(art.joints) and sum(j.qpos_dim for j in art.joints.values()) or 0
    qvel_n = len(art.joints) and sum(j.qvel_dim for j in art.joints.values()) or 0
    sensor_n = sum(s.dim for s in art.sensors.values())
    return qpos_n + qvel_n + sensor_n


def _articulation_act_size(art: URLabArticulation) -> int:
    return len(art.actuators)


class URLabEnv:
    """gymnasium-compatible env wrapping a URLabClient.

    Constructor:
        URLabEnv(
            client,                          # already-discovered URLabClient
            *,
            space_mode="flat",               # "flat" | "dict"
            n_steps=1,                       # physics substeps per env step
            observations="standard",         # "minimal" | "standard" | "full"
            reward_fn=None,                  # (obs_dict) -> float; default 0.0
            termination_fn=None,             # (obs_dict) -> bool; default False
            max_episode_steps=None,          # None = no truncation
            include_cameras=False,           # forwarded to client.step
        )

    Action / observation layout:
        - "flat"  -> Box. Action is concatenated ctrl across all articulations
                     in discovery order. Observation is concatenated
                     qpos + qvel + sensors per articulation in discovery order.
        - "dict"  -> Dict keyed by articulation prefix. Same per-articulation
                     layout per value.

    Reward / termination are caller-supplied. The default env returns 0.0
    reward and never terminates -- it is a transport, not a reward shaper.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        client: URLabClient,
        *,
        space_mode: Union[str, SpaceMode] = "flat",
        n_steps: int = 1,
        observations: Union[str, ObservationLevel] = "standard",
        reward_fn: Optional[RewardFn] = None,
        termination_fn: Optional[TerminationFn] = None,
        max_episode_steps: Optional[int] = None,
        include_cameras: bool = False,
    ):
        if spaces is None:  # pragma: no cover
            raise RuntimeError(
                "URLabEnv requires gymnasium (or legacy gym) installed"
            )
        self.client = client
        self.space_mode = coerce(SpaceMode, space_mode)
        self.n_steps = int(n_steps)
        self.observations = coerce(ObservationLevel, observations)
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.max_episode_steps = max_episode_steps
        self.include_cameras = bool(include_cameras)
        self._step_count = 0

        if not client.articulations:
            raise RuntimeError(
                "URLabClient has no articulations; call client.connect() first"
            )

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def _build_action_space(self):
        if self.space_mode is SpaceMode.FLAT:
            sizes = [_articulation_act_size(a) for a in self.client.articulations.values()]
            total = sum(sizes)
            lows, highs = self._flat_action_bounds()
            return spaces.Box(low=lows, high=highs, shape=(total,), dtype=np.float64)
        return spaces.Dict(
            {
                prefix: spaces.Box(
                    low=self._per_arm_action_bounds(art)[0],
                    high=self._per_arm_action_bounds(art)[1],
                    shape=(_articulation_act_size(art),),
                    dtype=np.float64,
                )
                for prefix, art in self.client.articulations.items()
            }
        )

    def _build_observation_space(self):
        if self.space_mode is SpaceMode.FLAT:
            total = sum(_articulation_obs_size(a) for a in self.client.articulations.values())
            return spaces.Box(
                low=-np.inf, high=np.inf, shape=(total,), dtype=np.float64
            )
        return spaces.Dict(
            {
                prefix: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(_articulation_obs_size(art),),
                    dtype=np.float64,
                )
                for prefix, art in self.client.articulations.items()
            }
        )

    def _per_arm_action_bounds(self, art: URLabArticulation) -> Tuple[np.ndarray, np.ndarray]:
        lows, highs = [], []
        for actuator in art.actuators.values():
            if actuator.ctrlrange is not None:
                lows.append(float(actuator.ctrlrange[0]))
                highs.append(float(actuator.ctrlrange[1]))
            else:
                lows.append(-np.inf)
                highs.append(np.inf)
        return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)

    def _flat_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lows, highs = [], []
        for art in self.client.articulations.values():
            l, h = self._per_arm_action_bounds(art)
            lows.append(l)
            highs.append(h)
        return np.concatenate(lows), np.concatenate(highs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        keyframe = options.get("keyframe_name") if options else None
        per_art_qpos = options.get("per_articulation_qpos") if options else None
        self.client.reset(
            keyframe_name=keyframe,
            seed=seed,
            per_articulation_qpos=per_art_qpos,
        )
        self._step_count = 0
        obs = self._build_obs()
        return obs, self._build_info()

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._apply_action(action)
        self.client.step(
            n_steps=self.n_steps,
            include_cameras=self.include_cameras,
            observations=wire(self.observations),
        )
        self._step_count += 1

        obs = self._build_obs()
        info = self._build_info()
        reward = float(self.reward_fn(info)) if self.reward_fn else 0.0
        terminated = bool(self.termination_fn(info)) if self.termination_fn else False
        truncated = (
            self.max_episode_steps is not None
            and self._step_count >= self.max_episode_steps
        )
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.client.close()

    def _apply_action(self, action: Any) -> None:
        if self.space_mode is SpaceMode.FLAT:
            arr = np.asarray(action, dtype=np.float64).ravel()
            offset = 0
            for art in self.client.articulations.values():
                n = _articulation_act_size(art)
                ctrl_map = {
                    name: float(arr[offset + i])
                    for i, name in enumerate(art.actuators.keys())
                }
                art.set_ctrl(ctrl_map)
                offset += n
            return
        for prefix, vec in action.items():
            art = self.client.articulations[prefix]
            arr = np.asarray(vec, dtype=np.float64).ravel()
            ctrl_map = {
                name: float(arr[i]) for i, name in enumerate(art.actuators.keys())
            }
            art.set_ctrl(ctrl_map)

    def _per_arm_obs(self, art: URLabArticulation) -> np.ndarray:
        parts = []
        if art.qpos_array is not None:
            parts.append(np.asarray(art.qpos_array, dtype=np.float64).ravel())
        if art.qvel_array is not None:
            parts.append(np.asarray(art.qvel_array, dtype=np.float64).ravel())
        for sensor in art.sensors.values():
            if sensor.latest is not None:
                parts.append(np.asarray(sensor.latest, dtype=np.float64).ravel())
            else:
                parts.append(np.zeros(sensor.dim, dtype=np.float64))
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)

    def _build_obs(self) -> Any:
        if self.space_mode is SpaceMode.FLAT:
            return np.concatenate(
                [self._per_arm_obs(a) for a in self.client.articulations.values()]
            )
        return {
            prefix: self._per_arm_obs(art)
            for prefix, art in self.client.articulations.items()
        }

    def _build_info(self) -> Dict[str, Any]:
        return {
            "client": self.client,
            "step_count": self._step_count,
            "sim_time": getattr(self.client, "sim_time", 0.0),
            "articulations": dict(self.client.articulations),
        }
