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

"""Articulation + per-kind handles for the URLab remote-stepping client."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from .enums import (
    ActuatorType,
    CameraMode,
    ControlMode,
    ControllerKind,
    StepMode,
    coerce,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .client import URLabClient

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import mujoco  # type: ignore
except ImportError:  # pragma: no cover
    mujoco = None  # noqa: N816


# ---------------------------------------------------------------------------
# Dataclasses: per-kind accessors walked from MjModel at connect()
# ---------------------------------------------------------------------------


@dataclass
class Actuator:
    """Actuator accessor.

    `type` is the *authored* actuator kind (e.g. `"position"`, `"motor"`),
    shipped in the handshake because MuJoCo's compiled model collapses
    `<position>` / `<velocity>` shortcuts into `<general>` and loses the
    authored kind. Everything else walks straight out of `MjModel`.
    """

    name: str
    id: int
    type: Optional[ActuatorType] = None
    trn_type: Optional[str] = None
    joint: Optional[str] = None
    ctrlrange: Optional[Tuple[float, float]] = None
    forcerange: Optional[Tuple[float, float]] = None
    gear: Optional[np.ndarray] = None
    gainprm: Optional[np.ndarray] = None
    dynprm: Optional[np.ndarray] = None
    kp: Optional[float] = None
    kv: Optional[float] = None
    # Populated per-step from the step reply's `actuator_forces` field
    # (only present at observation level "full"). None when no reply has
    # carried it yet, or when running at a lower observation level.
    force: Optional[float] = None

    # Populated by the owning articulation after construction so that
    # `.set_ctrl(v)` / `.value` round-trip through the articulation's
    # local ctrl buffer without every actuator holding a back-reference
    # to the client transport.
    _art: "URLabArticulation | None" = field(default=None, repr=False)
    _local_index: int = field(default=-1, repr=False)

    def set_ctrl(self, v: float) -> None:
        if self._art is None or self._local_index < 0:
            raise RuntimeError(f"Actuator {self.name!r} not bound to an articulation")
        self._art.ctrl_array[self._local_index] = float(v)

    @property
    def value(self) -> float:
        if self._art is None or self._local_index < 0:
            raise RuntimeError(f"Actuator {self.name!r} not bound to an articulation")
        return float(self._art.ctrl_array[self._local_index])


@dataclass
class Joint:
    name: str
    id: int
    jnt_type: int  # mjtJoint enum int, from MjModel
    # Global offsets in MjData.qpos / qvel -- index `client.data.qpos[...]`
    # (the global mirror) with these.
    qpos_offset: int
    qpos_dim: int
    qvel_offset: int
    qvel_dim: int
    # Local offsets within the owning articulation's qpos_array / qvel_array
    # (densely packed in discovery order). Use these to slice `art.qpos_array`
    # and `art.qvel_array`. With scene bodies present, local != global.
    qpos_local_offset: int = 0
    qvel_local_offset: int = 0
    range: Optional[Tuple[float, float]] = None
    body_id: int = -1


@dataclass
class Sensor:
    name: str
    id: int
    sensor_type: int  # mjtSensor enum int
    offset: int
    dim: int
    # Populated per-step from the step reply's `sensors` block.
    latest: Optional[np.ndarray] = None


@dataclass
class Body:
    name: str
    id: int
    # Populated per-step from the MJB walk applied over the reply qpos
    # (via mj_forward on the local mirror). For articulations, cheap to
    # fetch any time via `client.data.xpos[body.id]` directly too.
    xpos: Optional[np.ndarray] = None
    xquat: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# URLabCameraView
# ---------------------------------------------------------------------------


_CAMERA_DTYPE_BY_MODE = {
    CameraMode.REAL: np.uint8,       # BGRA8 on the wire, decoded to RGBA
    CameraMode.DEPTH: np.float32,    # single-channel float32, in scene units
    # Seg modes ship as BGRA8 with a per-class / per-instance color tint
    # baked in by the seg post-process material. Consumers map color back
    # to a class id with their own LUT.
    CameraMode.SEMANTIC: np.uint8,
    CameraMode.INSTANCE: np.uint8,
}


@dataclass
class URLabCameraView:
    """Shared camera accessor used by both `art.cameras[name]` and
    `client.global_cameras[name]`.

    `latest_frame` is None until the first frame arrives (streaming mode)
    or until the first `step(include_cameras=True)` reply populates it.
    Shape depends on mode: `(H, W, 4)` for real, `(H, W)` for depth /
    semantic / instance.
    """

    name: str
    mode: CameraMode
    resolution: Tuple[int, int]  # (width, height)
    fovy: float
    owner: Optional[str] = None
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.uint8))
    latest_frame: Optional[np.ndarray] = None
    sim_time: Optional[float] = None
    frame_count: int = 0
    # Wall-clock (time.monotonic) when `latest_frame` was last stored by the
    # streaming callback. Lets a consumer measure how stale the frame it's
    # about to display is, independent of sim_time / frame_id. None until the
    # first frame arrives.
    recv_monotonic: Optional[float] = None
    # Unix-epoch seconds (UE FDateTime::UtcNow) when this frame was captured,
    # from the v2 stream header. Directly comparable to Python time.time(), so
    # content latency = time.time() - capture_unix_time. None on a v1 header.
    capture_unix_time: Optional[float] = None
    # Post-step render-snapshot id of `latest_frame` (the step state it shows).
    # Set from the SHM/ZMQ stream's per-frame metadata header, together with
    # `latest_frame` so the two never diverge. A "fresh" step query waits until
    # this is >= the step reply's frame_id. None until the first frame arrives.
    frame_id: Optional[int] = None

    @classmethod
    def from_handshake(
        cls, name: str, payload: Mapping[str, Any], *, owner: Optional[str] = None
    ) -> "URLabCameraView":
        mode_str = payload.get("mode", "real")
        try:
            mode = coerce(CameraMode, mode_str)
        except ValueError:
            # Future UE modes: fall back to uint8, user sees a warning
            # from coerce(). We still instantiate so the name is visible.
            mode = CameraMode.REAL
        w, h = payload.get("resolution", (0, 0))
        dtype = np.dtype(_CAMERA_DTYPE_BY_MODE.get(mode, np.uint8))
        view = cls(
            name=name,
            mode=mode,
            resolution=(int(w), int(h)),
            fovy=float(payload.get("fovy", 0.0)),
            owner=owner,
            dtype=dtype,
        )
        # Stash the per-camera ZMQ endpoint + topic from the handshake so
        # the streaming SUB threads can subscribe in free-running mode.
        # These fields are not part of the dataclass schema (they're
        # implementation details for the SUB plumbing).
        view._zmq_endpoint = payload.get("zmq_endpoint")
        view._zmq_topic = payload.get("zmq_topic")
        return view


# ---------------------------------------------------------------------------
# URLabController + URLabPDController
# ---------------------------------------------------------------------------


class URLabController:
    """Kind-tagged controller config surface.

    Wraps the `{kind, params, schema}` bundle the handshake ships per
    articulation. `.configure(**kwargs)` validates against the schema and
    delegates to a `configure_controller` RPC on the step server, which
    in turn calls `UMjArticulationController::ApplyConfig`.

    Setters are safe to call in any step mode and under any `control_mode`
    (they route through the same UE path the `{prefix}/set_gains` PUB/SUB
    topic uses). Under `"raw"` control the server still tracks params for
    when the caller flips back to `"ue_controller"`.
    """

    def __init__(
        self,
        articulation_prefix: str,
        kind: str,
        params: Mapping[str, Any],
        schema: Mapping[str, Any],
        *,
        client: "URLabClient | None" = None,
    ):
        self.articulation_prefix = articulation_prefix
        try:
            self.kind: Union[ControllerKind, str] = coerce(ControllerKind, kind)
        except ValueError:
            # Unknown kind: keep the raw string so `.kind == "new_kind"` works
            self.kind = kind
        self.params: Dict[str, Any] = _deep_copy_params(params)
        self.schema: Dict[str, Any] = dict(schema)
        self._client = client

    # -- introspection -----------------------------------------------------

    def __repr__(self) -> str:
        kind = self.kind.value if isinstance(self.kind, ControllerKind) else self.kind
        return f"URLabController(prefix={self.articulation_prefix!r}, kind={kind!r})"

    def refresh(self) -> Dict[str, Any]:
        """Re-pull params from the server. No-op in tests without a client."""
        if self._client is None:
            return self.params
        reply = self._client._rpc_configure_controller(
            articulation=self.articulation_prefix, params={}
        )
        self.params = _deep_copy_params(reply.get("params", {}))
        return self.params

    # -- generic config ----------------------------------------------------

    def configure(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate `kwargs` against the schema and send them via RPC."""
        self._validate(kwargs)
        if self._client is None:
            # Local-only mode (tests, or pre-connection wiring)
            _merge_params(self.params, kwargs)
            return self.params
        reply = self._client._rpc_configure_controller(
            articulation=self.articulation_prefix, params=kwargs
        )
        new_params = reply.get("params", {})
        if new_params:
            self.params = _deep_copy_params(new_params)
        return self.params

    def _validate(self, patch: Mapping[str, Any]) -> None:
        """Schema check; raises ValueError on a violation."""
        for key, value in patch.items():
            entry = self.schema.get(key)
            if entry is None:
                raise ValueError(
                    f"Unknown controller param {key!r} for {self.kind!r} "
                    f"(known: {sorted(self.schema.keys())})"
                )
            kind = entry.get("type", "scalar")
            if kind == "per_joint":
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{key!r} expects a per-joint mapping, got {type(value).__name__}"
                    )
                for jname, jval in value.items():
                    _check_bounds(entry, jval, f"{key}.{jname}")
            elif kind == "scalar":
                _check_bounds(entry, value, key)
            else:
                # Unknown schema kind: don't block, just pass through
                logger.debug("Unvalidated schema kind %r for %r", kind, key)


class URLabPDController(URLabController):
    """PD-specific typed setters.

    `set_gains(kp={...}, kv={...}, torque_limit={...})` uses partial-patch
    semantics: any joint not mentioned keeps its current value. This
    matches the `{prefix}/set_gains` PUB/SUB contract.

    `set_defaults(kp=..., kv=..., torque_limit=...)` sets the `default_*`
    scalars which UE uses as fallbacks for unmentioned joints.
    """

    def set_gains(
        self,
        kp: Optional[Mapping[str, float]] = None,
        kv: Optional[Mapping[str, float]] = None,
        torque_limit: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}
        if kp:
            patch["kp"] = {str(k): float(v) for k, v in kp.items()}
        if kv:
            patch["kv"] = {str(k): float(v) for k, v in kv.items()}
        if torque_limit:
            patch["torque_limit"] = {str(k): float(v) for k, v in torque_limit.items()}
        if not patch:
            return self.params
        return self.configure(**patch)

    def set_defaults(
        self,
        kp: Optional[float] = None,
        kv: Optional[float] = None,
        torque_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}
        if kp is not None:
            patch["default_kp"] = float(kp)
        if kv is not None:
            patch["default_kv"] = float(kv)
        if torque_limit is not None:
            patch["default_torque_limit"] = float(torque_limit)
        if not patch:
            return self.params
        return self.configure(**patch)

    # live dict views
    @property
    def kp(self) -> Mapping[str, float]:
        return self.params.get("kp", {})

    @property
    def kv(self) -> Mapping[str, float]:
        return self.params.get("kv", {})

    @property
    def torque_limit(self) -> Mapping[str, float]:
        return self.params.get("torque_limit", {})


def _controller_for_handshake(
    prefix: str, payload: Mapping[str, Any], *, client: "URLabClient | None"
) -> URLabController:
    kind = payload.get("kind", "pd")
    params = payload.get("params", {})
    schema = payload.get("schema", {})
    if kind == ControllerKind.PD.value:
        return URLabPDController(prefix, kind, params, schema, client=client)
    return URLabController(prefix, kind, params, schema, client=client)


def _deep_copy_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, Mapping):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def _merge_params(dst: Dict[str, Any], patch: Mapping[str, Any]) -> None:
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            merged = dict(dst[k])
            merged.update(v)
            dst[k] = merged
        else:
            dst[k] = v


def _check_bounds(entry: Mapping[str, Any], value: Any, label: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label!r} must be numeric, got {value!r}")
    if "min" in entry and numeric < float(entry["min"]):
        raise ValueError(f"{label!r}={numeric} < min={entry['min']}")
    if "max" in entry and numeric > float(entry["max"]):
        raise ValueError(f"{label!r}={numeric} > max={entry['max']}")


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw / SciPy / ROS convention)
# ---------------------------------------------------------------------------


def _quat_apply(q_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3-vec by a unit quaternion (xyzw) -- v_world = R(q) v_body."""
    qx, qy, qz, qw = (
        float(q_xyzw[0]),
        float(q_xyzw[1]),
        float(q_xyzw[2]),
        float(q_xyzw[3]),
    )
    u = np.array([qx, qy, qz], dtype=np.float64)
    cross_uv = np.cross(u, vec)
    return vec + 2.0 * qw * cross_uv + 2.0 * np.cross(u, cross_uv)


def _quat_apply_inverse(q_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3-vec by the inverse of a unit quaternion (xyzw) --
    v_body = R(q)^T v_world. Same as `quat_rotate_inverse` in mjlab/RoboJuDo."""
    qx, qy, qz, qw = (
        float(q_xyzw[0]),
        float(q_xyzw[1]),
        float(q_xyzw[2]),
        float(q_xyzw[3]),
    )
    u = np.array([qx, qy, qz], dtype=np.float64)
    cross_uv = np.cross(u, vec)
    return vec - 2.0 * qw * cross_uv + 2.0 * np.cross(u, cross_uv)


# ---------------------------------------------------------------------------
# URLabEntity -- base class for any dynamic body in the scene
# ---------------------------------------------------------------------------


class URLabEntity:
    """Accessor for any dynamic body the simulator tracks.

    Plain entities (pallets, props, free-jointed scene bodies) are this
    class directly; articulations are this class with extras. Exposes the
    common surface every dynamic body has: identity, root link state in
    world / body frame, and an external-wrench buffer.

    World-frame fields read from `client.data.xpos` / `xquat` / `qvel`,
    populated from each step reply (plus `mj_forward` over the local
    mirror so non-shipped fields stay consistent). Articulations override
    velocity / pose accessors to use their dense local buffers, which
    avoids the `mj_forward` round-trip.
    """

    def __init__(
        self,
        *,
        name: str,
        body_id: int,
        has_free_base: bool = False,
        client: "URLabClient | None" = None,
    ):
        self.name = name
        self.body_id = int(body_id)
        self.has_free_base = bool(has_free_base)
        self._client = client
        # Cached free-joint qpos / qvel offsets if the scene-body has one;
        # used by `apply_xfrc` puppet-warn diagnostic and by future
        # qpos/qvel inspection helpers. None for fixed-position scene props.
        self.free_joint: Optional[str] = None
        self.free_joint_id: Optional[int] = None
        self.qpos_offset: Optional[int] = None
        self.qvel_offset: Optional[int] = None
        # Buffered external wrench applied to this entity's root body;
        # cleared after the next step.
        self._pending_xfrc: Optional[np.ndarray] = None

    # ---- root-link state (world frame) ----

    @property
    def root_pos_w(self) -> np.ndarray:
        """World-frame root link position (3,). Reads from `client.data.xpos`
        which `mj_forward` keeps current for every body in the scene."""
        if self._client is None or self._client.data is None or self.body_id < 0:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self._client.data.xpos[self.body_id], dtype=np.float64).copy()

    @property
    def root_quat_w(self) -> np.ndarray:
        """World-frame root quaternion in MuJoCo (w, x, y, z) order."""
        if self._client is None or self._client.data is None or self.body_id < 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(self._client.data.xquat[self.body_id], dtype=np.float64).copy()

    @property
    def root_quat_xyzw(self) -> np.ndarray:
        """Root quaternion in (x, y, z, w) order -- the convention SciPy /
        ROS / mjlab / RoboJuDo / most policy stacks use."""
        q = self.root_quat_w
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

    @property
    def root_lin_vel_w(self) -> np.ndarray:
        """World-frame root linear velocity (3,). Plain entities return zero
        -- we don't compute body twists from `cvel` for them. Articulations
        override to read from their local qvel buffer."""
        return np.zeros(3, dtype=np.float64)

    @property
    def root_ang_vel_w(self) -> np.ndarray:
        """World-frame root angular velocity (3,). See `root_lin_vel_w`."""
        return np.zeros(3, dtype=np.float64)

    # ---- root-link state (body frame, derived) ----

    @property
    def root_lin_vel_b(self) -> np.ndarray:
        """Root linear velocity rotated into the body frame."""
        return _quat_apply_inverse(self.root_quat_xyzw, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> np.ndarray:
        """Root angular velocity in the body frame."""
        return _quat_apply_inverse(self.root_quat_xyzw, self.root_ang_vel_w)

    @property
    def projected_gravity_b(self) -> np.ndarray:
        """[0, 0, -1] rotated into the body frame. Standard locomotion-policy
        observation; matches `mjlab.envs.mdp.observations.projected_gravity`."""
        return _quat_apply_inverse(
            self.root_quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float64)
        )

    # ---- external wrench ----

    def apply_xfrc(
        self,
        force: Optional[Sequence[float]] = None,
        torque: Optional[Sequence[float]] = None,
    ) -> None:
        """Buffer an external 6-dof wrench applied to this entity's root body.
        Cleared after the next step. Inert in puppet mode -- UE's
        `d->xfrc_applied` is overwritten by the bridge's local `MjData`
        each step."""
        if self._client is None:
            raise RuntimeError(f"Entity {self.name!r} not bound to a client")
        if self._client.step_mode == StepMode.PUPPET:
            warnings.warn(
                f"apply_xfrc on entity {self.name!r} is inert in puppet mode "
                "(client's mj_step is authoritative).",
                stacklevel=2,
            )
        vec = np.zeros(6, dtype=np.float64)
        if force is not None:
            vec[:3] = np.asarray(force, dtype=np.float64)
        if torque is not None:
            vec[3:] = np.asarray(torque, dtype=np.float64)
        self._pending_xfrc = vec
        self._client._pending_entity_xfrc[self.name] = vec

    def clear_xfrc(self) -> None:
        self._pending_xfrc = None


# ---------------------------------------------------------------------------
# URLabArticulation
# ---------------------------------------------------------------------------


def _mj_obj_for_kind(kind: str) -> int:
    """Map a wrapper kind string to the mujoco.mjtObj integer."""
    if mujoco is None:  # pragma: no cover
        raise RuntimeError("mujoco not installed")
    mapping = {
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
        "keyframe": mujoco.mjtObj.mjOBJ_KEY,
    }
    if kind not in mapping:
        raise KeyError(
            f"Unknown kind {kind!r}; expected one of {sorted(mapping.keys())}"
        )
    return mapping[kind]


class URLabArticulation(URLabEntity):
    """Articulation: an entity with joints, actuators, sensors, bodies,
    cameras, and (optionally) a controller.

    Built from an MjModel walk at `URLabClient.connect()`. The
    articulation's `prefix` buckets MjModel names via
    `name.startswith(prefix + "_")`, mirroring how URLab generates names
    during XML import. Everything else comes straight from `MjModel`
    (offsets, ranges, gear, gainprm, dynprm, body ids) or from the
    handshake for info MJB cannot carry (actuator types, controller
    state, camera metadata).

    Inherits `URLabEntity`'s root-link state interface (`root_pos_w`,
    `root_quat_xyzw`, `root_lin_vel_w`, `root_lin_vel_b`,
    `projected_gravity_b`, ...). For articulations with a free base, the
    velocity / pose accessors are overridden to read from the local
    qpos/qvel buffers directly -- avoids the `client.data` round-trip.
    """

    def __init__(
        self,
        *,
        prefix: str,
        model: Any,  # mujoco.MjModel
        data: Any,  # mujoco.MjData
        handshake: Mapping[str, Any],
        client: "URLabClient | None" = None,
    ):
        # Defer base-class init until we know body_id (resolved during walk).
        self.prefix = prefix
        self.name = prefix  # entity-level alias; matches `entity.name`
        self._model = model
        self._data = data
        self._client = client

        # Bridge-owned actor id echoed by the handshake. Stable across
        # PIE sessions; survives UE's volatile actor naming. Empty when
        # nothing was set on the actor — callers fall back to ``prefix``.
        self.actor_id: str = str(handshake.get("actor_id", "") or "")

        # Default control mode defaults to "ue_controller" iff a controller
        # block is shipped. Callers can flip to "raw" before step().
        default_mode_str = handshake.get("default_control_mode", "raw")
        self.control_mode: ControlMode = coerce(
            ControlMode, default_mode_str, default=ControlMode.RAW
        )

        # Controller (may be None in "raw" articulations)
        controller_payload = handshake.get("controller")
        self.controller: Optional[URLabController]
        if controller_payload:
            self.controller = _controller_for_handshake(
                prefix, controller_payload, client=client
            )
        else:
            self.controller = None

        actuator_types_map = handshake.get("actuator_types", {})

        # Dicts populated by _walk_model()
        self.actuators: Dict[str, Actuator] = {}
        self.joints: Dict[str, Joint] = {}
        self.sensors: Dict[str, Sensor] = {}
        self.bodies: Dict[str, Body] = {}
        self.cameras: Dict[str, URLabCameraView] = {}

        # Name -> local index in articulation-ordered flat arrays
        self._actuator_local: Dict[str, int] = {}
        self._joint_local: Dict[str, int] = {}

        # Original-XML-name -> live-MJB-short-name reconciliation. The
        # handshake ships {live: original} per category for components
        # whose name diverged from the XML during import (SCS uniqueness)
        # or spec-build (intra-namespace dedup). We invert to
        # {original: live} so resolve_* can map policy-side names that
        # reference the original XML names back to the live keys we use
        # in self.actuators / self.joints / etc.
        original_names_payload = handshake.get("original_names", {}) or {}
        self._original_to_live: Dict[str, Dict[str, str]] = {
            "actuators": {v: k for k, v in (original_names_payload.get("actuators") or {}).items()},
            "joints":    {v: k for k, v in (original_names_payload.get("joints")    or {}).items()},
            "sensors":   {v: k for k, v in (original_names_payload.get("sensors")   or {}).items()},
            "bodies":    {v: k for k, v in (original_names_payload.get("bodies")    or {}).items()},
        }

        self._walk_model(actuator_types_map)

        # Free-base detection. A floating-base joint is `mjtJoint.mjJNT_FREE`
        # (jnt_type==0) with qpos_dim==7 / qvel_dim==6. Cached after
        # `_walk_model` so `has_free_base` / `root_pos_w` / `root_quat_xyzw`
        # work without re-scanning every call.
        self._free_base_joint: Optional[Joint] = None
        for j in self.joints.values():
            if j.qpos_dim == 7:
                self._free_base_joint = j
                break

        # Resolve the entity's body_id: the root body of the articulation.
        # MJCF authoring convention puts the root body at the prefix name
        # (or as the only body that owns the free joint, when present).
        # Default to -1 if nothing matches; root_pos_w guards against that.
        root_body_id = -1
        if self._free_base_joint is not None and self._free_base_joint.body_id >= 0:
            root_body_id = self._free_base_joint.body_id
        elif self.bodies:
            # Fixed-base: take the first walked body (lowest jid).
            root_body_id = next(iter(self.bodies.values())).id

        # Lift entity-level state. body_id + has_free_base feed root_pos_w
        # / root_quat_w and apply_xfrc; the rest of the entity surface is
        # served by overrides on the articulation.
        self.body_id = int(root_body_id)
        self.has_free_base = self._free_base_joint is not None
        self.free_joint = self._free_base_joint.name if self._free_base_joint else None
        self.free_joint_id = self._free_base_joint.id if self._free_base_joint else None
        self.qpos_offset = (
            self._free_base_joint.qpos_offset if self._free_base_joint else None
        )
        self.qvel_offset = (
            self._free_base_joint.qvel_offset if self._free_base_joint else None
        )
        self._pending_xfrc: Optional[np.ndarray] = None

        # Flat-array fast paths. Indexed in the articulation's local order
        # (discovery order). Callers who want global MjData slot indices
        # use art.mj(kind, name) instead.
        n_act = len(self.actuators)
        self.ctrl_array: np.ndarray = np.zeros(n_act, dtype=np.float64)
        # Last ctrl actually applied on UE side, echoed back in step replies.
        # Distinct from ctrl_array (the user's outgoing setpoint buffer).
        # In live this is UE's PD-controller torque output; in
        # direct/puppet it equals what the bridge sent. Read-only mirror.
        self.last_applied_ctrl: np.ndarray = np.zeros(n_act, dtype=np.float64)
        # Activation state for stateful actuators (cylinder, integrated
        # velocity, etc). Populated from `act` in step replies at
        # observation level standard or full. Shape is (n_act,) for
        # convenience even though only a subset of actuators have act state;
        # entries for stateless actuators stay zero.
        self.act_array: np.ndarray = np.zeros(n_act, dtype=np.float64)

        # qpos / qvel flat arrays are views when contiguous, otherwise a
        # materialised per-step snapshot. Implementation here keeps a
        # dedicated buffer and refreshes it in _refresh_state(). Addresses
        # the common case: joints in a single articulation are contiguous.
        self.qpos_array: np.ndarray = np.zeros(self._qpos_dim(), dtype=np.float64)
        self.qvel_array: np.ndarray = np.zeros(self._qvel_dim(), dtype=np.float64)

        # ROS-Twist-aligned input: linear xyz + angular xyz + bitfield of
        # active discrete actions. Populated per step from the
        # `per_articulation[prefix]` reply when UE has a UMjTwistController
        # attached (auto-spawned on AMjArticulation today). Stays zero
        # otherwise. Use the `twist` property for a flat 6-vec.
        self.twist_linear: np.ndarray = np.zeros(3, dtype=np.float64)
        self.twist_angular: np.ndarray = np.zeros(3, dtype=np.float64)
        self.actions: int = 0

        # xfrc buffered between steps
        self._pending_xfrc: Dict[str, np.ndarray] = {}

        # Cameras from handshake (MJB does not carry camera mode / resolution)
        for cam_name, cam_payload in handshake.get("camera_topics", {}).items():
            self.cameras[cam_name] = URLabCameraView.from_handshake(
                cam_name, cam_payload, owner=prefix
            )

    # -- helpers ----------------------------------------------------------

    def _prefix_match(self, name: Optional[str]) -> bool:
        if not name:
            return False
        return name.startswith(self.prefix + "_") or name == self.prefix

    def _qpos_dim(self) -> int:
        return int(sum(j.qpos_dim for j in self.joints.values()))

    def _qvel_dim(self) -> int:
        return int(sum(j.qvel_dim for j in self.joints.values()))

    def _walk_model(self, actuator_types_map: Mapping[str, str]) -> None:
        model = self._model
        if model is None:
            return

        # Actuators
        local_idx = 0
        for aid in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if not self._prefix_match(name):
                continue
            trn_type_int = int(model.actuator_trntype[aid])
            trn_type_str = _trn_type_name(trn_type_int)
            joint_name: Optional[str] = None
            if trn_type_str in ("joint", "joint_in_parent"):
                jnt_id = int(model.actuator_trnid[aid, 0])
                joint_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id
                )
            ctrlrange = None
            if bool(model.actuator_ctrllimited[aid]):
                ctrlrange = tuple(float(x) for x in model.actuator_ctrlrange[aid])
            forcerange = None
            if bool(model.actuator_forcelimited[aid]):
                forcerange = tuple(float(x) for x in model.actuator_forcerange[aid])
            gear = np.array(model.actuator_gear[aid], dtype=np.float64)
            gainprm = np.array(model.actuator_gainprm[aid], dtype=np.float64)
            dynprm = np.array(model.actuator_dynprm[aid], dtype=np.float64)
            kp = float(gainprm[0]) if gainprm.size > 0 else None
            kv = float(gainprm[1]) if gainprm.size > 1 else None

            # Actuator type from handshake (authored). Handshake key can be
            # the short (unprefixed) or full name — try short first.
            short = _strip_prefix(name, self.prefix)
            type_str = (
                actuator_types_map.get(short)
                or actuator_types_map.get(name)
            )
            actuator_type: Optional[ActuatorType] = None
            if type_str:
                try:
                    actuator_type = coerce(ActuatorType, type_str)
                except ValueError:
                    actuator_type = None

            act = Actuator(
                name=short,
                id=aid,
                type=actuator_type,
                trn_type=trn_type_str,
                joint=_strip_prefix(joint_name, self.prefix) if joint_name else None,
                ctrlrange=ctrlrange,
                forcerange=forcerange,
                gear=gear,
                gainprm=gainprm,
                dynprm=dynprm,
                kp=kp,
                kv=kv,
                _art=self,
                _local_index=local_idx,
            )
            self.actuators[short] = act
            self._actuator_local[short] = local_idx
            local_idx += 1

        # Joints
        joint_local_idx = 0
        local_qpos_off = 0
        local_qvel_off = 0
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not self._prefix_match(name):
                continue
            jtype = int(model.jnt_type[jid])
            qpos_off = int(model.jnt_qposadr[jid])
            qvel_off = int(model.jnt_dofadr[jid])
            qpos_dim = _qpos_dim_for_jnt(jtype)
            qvel_dim = _qvel_dim_for_jnt(jtype)
            rng = None
            if bool(model.jnt_limited[jid]):
                rng = tuple(float(x) for x in model.jnt_range[jid])
            short = _strip_prefix(name, self.prefix)
            self.joints[short] = Joint(
                name=short,
                id=jid,
                jnt_type=jtype,
                qpos_offset=qpos_off,
                qpos_dim=qpos_dim,
                qvel_offset=qvel_off,
                qvel_dim=qvel_dim,
                qpos_local_offset=local_qpos_off,
                qvel_local_offset=local_qvel_off,
                range=rng,
                body_id=int(model.jnt_bodyid[jid]),
            )
            local_qpos_off += qpos_dim
            local_qvel_off += qvel_dim
            self._joint_local[short] = joint_local_idx
            joint_local_idx += 1

        # Sensors
        for sid in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            if not self._prefix_match(name):
                continue
            short = _strip_prefix(name, self.prefix)
            self.sensors[short] = Sensor(
                name=short,
                id=sid,
                sensor_type=int(model.sensor_type[sid]),
                offset=int(model.sensor_adr[sid]),
                dim=int(model.sensor_dim[sid]),
            )

        # Bodies (owned by the articulation root — name-prefix bucketed)
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not self._prefix_match(name):
                continue
            short = _strip_prefix(name, self.prefix)
            self.bodies[short] = Body(name=short, id=bid)

    # -- mj(...) helpers -------------------------------------

    def mj(self, kind: str, name: str) -> int:
        """Return the global MuJoCo id for `name` under the given kind.

        `kind ∈ {"joint", "actuator", "sensor", "body", "geom", "site",
        "camera", "keyframe"}`. Accepts both the short (unprefixed) name
        and the full `prefix_name` form. Raises KeyError for unknown ids.
        """
        if mujoco is None:  # pragma: no cover
            raise RuntimeError("mujoco not installed")
        obj = _mj_obj_for_kind(kind)
        # try full name first, then prefixed form
        candidates: List[str] = []
        if name.startswith(self.prefix + "_") or name == self.prefix:
            candidates.append(name)
        else:
            candidates.append(f"{self.prefix}_{name}")
            candidates.append(name)
        for candidate in candidates:
            mjid = mujoco.mj_name2id(self._model, obj, candidate)
            if mjid >= 0:
                return int(mjid)
        raise KeyError(
            f"No {kind} named {name!r} (tried {candidates!r}) in articulation "
            f"{self.prefix!r}"
        )

    def mj_joint(self, name: str) -> int:
        return self.mj("joint", name)

    def mj_actuator(self, name: str) -> int:
        return self.mj("actuator", name)

    def mj_sensor(self, name: str) -> int:
        return self.mj("sensor", name)

    def mj_body(self, name: str) -> int:
        return self.mj("body", name)

    def mj_camera(self, name: str) -> int:
        return self.mj("camera", name)

    # -- control / state getters -----------------------------------------

    @property
    def twist(self) -> np.ndarray:
        """ROS-Twist-aligned 6-vec [linear xyz, angular xyz]. Zeros when
        the articulation has no UMjTwistController on the UE side."""
        return np.concatenate([self.twist_linear, self.twist_angular])

    # -- root-link state overrides --------------------------------------
    #
    # qpos_array layout for a free-base articulation:
    #   [root_pos_w(3), root_quat_wxyz(4), dof_qpos(num_dofs)]
    # qvel_array layout:
    #   [root_lin_vel_w(3), root_ang_vel_b(3), dof_qvel(num_dofs)]
    # MuJoCo stores free-joint angular velocity in the BODY frame; we
    # rotate it to world for `root_ang_vel_w` and expose it directly via
    # `root_ang_vel_b`. The free joint is always first in the local
    # buffers because URLab/MJCF compile it as jnt_id 0 of the
    # articulation. Fixed-base articulations fall through to the
    # `URLabEntity` defaults (zero velocities, pose from `client.data`).

    @property
    def root_pos_w(self) -> np.ndarray:
        if self._free_base_joint is None:
            return super().root_pos_w
        off = self._free_base_joint.qpos_local_offset
        return np.asarray(self.qpos_array[off : off + 3], dtype=np.float64).copy()

    @property
    def root_quat_w(self) -> np.ndarray:
        if self._free_base_joint is None:
            return super().root_quat_w
        off = self._free_base_joint.qpos_local_offset
        return np.asarray(self.qpos_array[off + 3 : off + 7], dtype=np.float64).copy()

    @property
    def root_lin_vel_w(self) -> np.ndarray:
        if self._free_base_joint is None:
            return np.zeros(3, dtype=np.float64)
        off = self._free_base_joint.qvel_local_offset
        return np.asarray(self.qvel_array[off : off + 3], dtype=np.float64).copy()

    @property
    def root_ang_vel_b(self) -> np.ndarray:
        # MuJoCo free joint already stores ang vel in body frame -- read direct.
        if self._free_base_joint is None:
            return np.zeros(3, dtype=np.float64)
        off = self._free_base_joint.qvel_local_offset
        return np.asarray(self.qvel_array[off + 3 : off + 6], dtype=np.float64).copy()

    @property
    def root_ang_vel_w(self) -> np.ndarray:
        # Forward rotation: body -> world.
        if self._free_base_joint is None:
            return np.zeros(3, dtype=np.float64)
        return _quat_apply(self.root_quat_xyzw, self.root_ang_vel_b)

    # -- per-DoF joint state (free joint excluded) -----------------------
    #
    # mjlab / IsaacLab / RSL-RL style obs vectors expect joint pos / vel
    # over the actuated DoFs only (free-joint root pose / velocity is
    # exposed separately via root_pos_w / root_lin_vel_b). These slices
    # drop the leading free-joint block when the articulation is free-base.

    @property
    def dof_qpos(self) -> np.ndarray:
        """Per-actuated-DoF joint positions, free-base root pose excluded.
        Shape `(num_dofs,)`. For fixed-base articulations this is the full
        local qpos."""
        if self._free_base_joint is None:
            return np.asarray(self.qpos_array, dtype=np.float64).copy()
        head = (
            self._free_base_joint.qpos_local_offset
            + self._free_base_joint.qpos_dim
        )
        return np.asarray(self.qpos_array[head:], dtype=np.float64).copy()

    @property
    def dof_qvel(self) -> np.ndarray:
        """Per-actuated-DoF joint velocities, free-base root velocity
        excluded. Shape `(num_dofs,)`."""
        if self._free_base_joint is None:
            return np.asarray(self.qvel_array, dtype=np.float64).copy()
        head = (
            self._free_base_joint.qvel_local_offset
            + self._free_base_joint.qvel_dim
        )
        return np.asarray(self.qvel_array[head:], dtype=np.float64).copy()

    # -- joint / actuator name resolution --------------------------------

    def resolve_joint(self, name: str) -> Optional[str]:
        """Resolve a caller-supplied joint name into the key used in
        `self.joints`. URLab keys are the unprefixed MJCF names; some
        callers (RoboJuDo) use a `_joint` suffix that Menagerie strips.
        Tries the raw name first, then the original-XML-name reverse map
        (handles SCS / spec-time renames), then with `_joint` stripped
        against both. Returns None when no match -- callers can warn / skip."""
        if name in self.joints:
            return name
        live = self._original_to_live["joints"].get(name)
        if live is not None and live in self.joints:
            return live
        stripped = name.removesuffix("_joint")
        if stripped in self.joints:
            return stripped
        live = self._original_to_live["joints"].get(stripped)
        if live is not None and live in self.joints:
            return live
        return None

    def resolve_actuator(self, name: str) -> Optional[str]:
        """Resolve a caller-supplied actuator name into the key used in
        `self.actuators`. Mirrors `resolve_joint`: tries raw, then the
        original-XML-name reverse map, then with `_joint` stripped."""
        if name in self.actuators:
            return name
        live = self._original_to_live["actuators"].get(name)
        if live is not None and live in self.actuators:
            return live
        stripped = name.removesuffix("_joint")
        if stripped in self.actuators:
            return stripped
        live = self._original_to_live["actuators"].get(stripped)
        if live is not None and live in self.actuators:
            return live
        return None

    # -- gain push convenience -------------------------------------------

    def push_gains(
        self,
        joint_names: Sequence[str],
        stiffness: Sequence[float],
        damping: Optional[Sequence[float]] = None,
        torque_limit: Optional[Sequence[float]] = None,
    ) -> int:
        """Push per-joint PD gains to the UE-side controller via
        `configure_controller`.

        Joint names are resolved through `resolve_actuator` (strips a
        `_joint` suffix when needed). Joints whose actuator can't be
        resolved are silently skipped -- the caller's policy is the
        authority on which DoFs exist; warning would spam.

        Returns the number of joints whose gains were forwarded. Raises
        when the articulation has no controller exposed (caller can
        catch + log)."""
        if self.controller is None:
            raise RuntimeError(
                f"articulation {self.prefix!r} has no controller; cannot push gains"
            )
        kp_map: Dict[str, float] = {}
        kv_map: Dict[str, float] = {}
        tl_map: Dict[str, float] = {}
        for i, jname in enumerate(joint_names):
            a_key = self.resolve_actuator(jname)
            if a_key is None:
                continue
            kp_map[a_key] = float(stiffness[i])
            if damping is not None:
                kv_map[a_key] = float(damping[i])
            if torque_limit is not None:
                tl_map[a_key] = float(torque_limit[i])
        if not kp_map:
            return 0
        # PDController has typed setters; fall back to configure() for
        # other controller kinds.
        if hasattr(self.controller, "set_gains"):
            self.controller.set_gains(
                kp=kp_map or None,
                kv=kv_map or None,
                torque_limit=tl_map or None,
            )
        else:  # pragma: no cover - non-PD controllers
            patch: Dict[str, Any] = {}
            if kp_map:
                patch["kp"] = kp_map
            if kv_map:
                patch["kv"] = kv_map
            if tl_map:
                patch["torque_limit"] = tl_map
            self.controller.configure(**patch)
        return len(kp_map)

    def set_ctrl(self, ctrl_map: Mapping[str, float]) -> None:
        """Bulk ctrl setter. Actuators not mentioned keep their last value."""
        for name, val in ctrl_map.items():
            if name not in self._actuator_local:
                raise KeyError(f"Unknown actuator {name!r} on {self.prefix!r}")
            self.ctrl_array[self._actuator_local[name]] = float(val)

    def get_ctrl(self) -> Dict[str, float]:
        return {
            name: float(self.ctrl_array[idx])
            for name, idx in self._actuator_local.items()
        }

    def get_qpos(self) -> Dict[str, float]:
        """First-component qpos per joint (free / ball joints return their
        position component, not the full 7-vec / 4-vec)."""
        if self._data is None:
            return {}
        return {
            name: float(self._data.qpos[j.qpos_offset])
            for name, j in self.joints.items()
        }

    def get_sensors(self) -> Dict[str, np.ndarray]:
        return {
            name: (s.latest.copy() if s.latest is not None else np.zeros(s.dim))
            for name, s in self.sensors.items()
        }

    # -- xfrc buffering --------------------------------------------------

    def apply_xfrc(
        self,
        body: str,
        force: Optional[Sequence[float]] = None,
        torque: Optional[Sequence[float]] = None,
    ) -> None:
        if body not in self.bodies:
            raise KeyError(f"Unknown body {body!r} on articulation {self.prefix!r}")
        vec = np.zeros(6, dtype=np.float64)
        if force is not None:
            vec[:3] = np.asarray(force, dtype=np.float64)
        if torque is not None:
            vec[3:] = np.asarray(torque, dtype=np.float64)
        self._pending_xfrc[body] = vec

    def clear_xfrc(self) -> None:
        self._pending_xfrc.clear()

    # -- refresh from step reply -----------------------------------------

    def _apply_step_reply(self, block: Mapping[str, Any]) -> None:
        """Populate local views from a `per_articulation[prefix]` reply block."""
        # Do NOT write reply ctrl back into self.ctrl_array. ctrl_array is
        # the user's outgoing send buffer (set via set_ctrl()); echoing
        # UE's d->ctrl into it creates a feedback loop in live
        # mode, where d->ctrl is the PD controller's torque output, not a
        # setpoint. The next step would send that torque as a new
        # setpoint, blowing up the joint. Keep echoed ctrl in a separate
        # `last_applied_ctrl` field for callers that want to inspect it.
        if "ctrl" in block:
            ctrl = np.asarray(block["ctrl"], dtype=np.float64)
            if ctrl.size == self.ctrl_array.size:
                self.last_applied_ctrl = ctrl
        qpos_all = block.get("qpos")
        if qpos_all is not None:
            qpos_all = np.asarray(qpos_all, dtype=np.float64)
            # Build a slice from local order
            off = 0
            for name, j in self.joints.items():
                n = j.qpos_dim
                # The reply's qpos is per-articulation, sorted by
                # discovery order; UE builds the reply the same way.
                self.qpos_array[off : off + n] = qpos_all[off : off + n]
                off += n
        qvel_all = block.get("qvel")
        if qvel_all is not None:
            qvel_all = np.asarray(qvel_all, dtype=np.float64)
            off = 0
            for name, j in self.joints.items():
                n = j.qvel_dim
                self.qvel_array[off : off + n] = qvel_all[off : off + n]
                off += n
        sensors_block = block.get("sensors") or {}
        for sname, raw in sensors_block.items():
            if sname in self.sensors:
                self.sensors[sname].latest = np.asarray(raw, dtype=np.float64)
        # Activation state (observation level standard / full).
        act_raw = block.get("act")
        if act_raw is not None:
            act_arr = np.asarray(act_raw, dtype=np.float64)
            if act_arr.size == self.act_array.size:
                self.act_array[:] = act_arr
            elif act_arr.size == 0:
                # Empty list = no stateful actuators; reset to zero.
                self.act_array[:] = 0.0
        # Per-body xpos / xquat (observation level full).
        bodies_block = block.get("bodies") or {}
        for bname, body_obs in bodies_block.items():
            if bname not in self.bodies:
                continue
            body = self.bodies[bname]
            xp = body_obs.get("xpos")
            if xp is not None:
                body.xpos = np.asarray(xp, dtype=np.float64)
            xq = body_obs.get("xquat")
            if xq is not None:
                body.xquat = np.asarray(xq, dtype=np.float64)
        # Actuator forces (observation level full). Either a flat array in
        # discovery order, or a dict keyed by actuator name.
        forces_raw = block.get("actuator_forces") or block.get("actuator_force")
        if isinstance(forces_raw, Mapping):
            for aname, fval in forces_raw.items():
                if aname in self.actuators:
                    self.actuators[aname].force = float(fval)
        elif forces_raw is not None:
            forces_arr = np.asarray(forces_raw, dtype=np.float64)
            for actuator, fval in zip(self.actuators.values(), forces_arr):
                actuator.force = float(fval)
        # Twist + actions: absent on articulations without UMjTwistController.
        twist_block = block.get("twist")
        if isinstance(twist_block, Mapping):
            lin = twist_block.get("linear")
            if lin is not None:
                lin_arr = np.asarray(lin, dtype=np.float64)
                if lin_arr.size >= 3:
                    self.twist_linear[:] = lin_arr[:3]
            ang = twist_block.get("angular")
            if ang is not None:
                ang_arr = np.asarray(ang, dtype=np.float64)
                if ang_arr.size >= 3:
                    self.twist_angular[:] = ang_arr[:3]
        actions_raw = block.get("actions")
        if actions_raw is not None:
            self.actions = int(actions_raw)

    def _build_step_request(self, *, control_mode: Optional[ControlMode]) -> Dict[str, Any]:
        """Serialize the articulation's outgoing step payload."""
        from .enums import wire as _wire
        out: Dict[str, Any] = {
            "ctrl": self.ctrl_array.tolist(),
        }
        mode_to_use = control_mode if control_mode is not None else self.control_mode
        if mode_to_use is not None:
            out["control_mode"] = _wire(mode_to_use)
        if self._pending_xfrc:
            out["xfrc_applied"] = {
                b: vec.tolist() for b, vec in self._pending_xfrc.items()
            }
        return out


def _qpos_dim_for_jnt(jtype: int) -> int:
    # mjtJoint: mjJNT_FREE=0 (7), mjJNT_BALL=1 (4), mjJNT_SLIDE=2 (1), mjJNT_HINGE=3 (1)
    return {0: 7, 1: 4, 2: 1, 3: 1}.get(jtype, 1)


def _qvel_dim_for_jnt(jtype: int) -> int:
    # free=6, ball=3, slide/hinge=1
    return {0: 6, 1: 3, 2: 1, 3: 1}.get(jtype, 1)


_TRN_TYPE_NAMES = {
    0: "joint",
    1: "joint_in_parent",
    2: "slider_crank",
    3: "tendon",
    4: "site",
    5: "body",
}


def _trn_type_name(tid: int) -> str:
    return _TRN_TYPE_NAMES.get(int(tid), "undefined")


def _strip_prefix(name: str, prefix: str) -> str:
    if name.startswith(prefix + "_"):
        return name[len(prefix) + 1 :]
    return name
