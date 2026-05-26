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

"""Typed result dataclasses returned by URLab client namespaces.
Each namespace method returns a typed object (or `None` for void acks);
`_*_from_wire` helpers keep one decode site per dataclass."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# PIE lifecycle (sim namespace)
# ---------------------------------------------------------------------------


class PIEState(str, enum.Enum):
    """Wire-string PIE lifecycle state.

    ``str`` mixin so values JSON / msgpack-serialise as their wire
    string and equality with raw strings still works.
    """
    OFF = "off"
    COMPILING = "compiling"
    COMPILE_FAILED = "compile_failed"
    READY = "ready"
    TIMEOUT = "timeout"


@dataclass
class PIEStartResult:
    """Typed result of :meth:`URLabClient.sim.start`.

    ``state`` is the PIE lifecycle outcome. ``handshake_payload`` carries
    the post-PIE handshake (already absorbed into the client) iff
    ``state == READY``. ``compile_error`` is non-empty iff
    ``state == COMPILE_FAILED``.
    """

    state: PIEState
    compile_error: str = ""
    handshake_payload: Optional[Dict[str, Any]] = None

    @property
    def is_ready(self) -> bool:
        return self.state == PIEState.READY


@dataclass
class PIEStatus:
    state: PIEState
    compile_error: str = ""
    sim_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Sim options (runtime namespace)
# ---------------------------------------------------------------------------


@dataclass
class SimOptions:
    """Typed echo of :meth:`URLabClient.runtime.set_sim_options`.

    Field names match wire keys; values reflect the resulting
    ``m_model->opt.*`` on the UE side. ``None`` for fields the server
    did not echo on this call.
    """

    timestep: Optional[float] = None
    gravity: Optional[Tuple[float, float, float]] = None
    wind: Optional[Tuple[float, float, float]] = None
    magnetic: Optional[Tuple[float, float, float]] = None
    density: Optional[float] = None
    viscosity: Optional[float] = None
    impratio: Optional[float] = None
    tolerance: Optional[float] = None
    iterations: Optional[int] = None
    ls_iterations: Optional[int] = None
    integrator: Optional[str] = None    # "euler" | "rk4" | "implicit" | "implicitfast"
    cone: Optional[str] = None          # "pyramidal" | "elliptic"
    solver: Optional[str] = None        # "pgs" | "cg" | "newton"
    noslip_iterations: Optional[int] = None
    noslip_tolerance: Optional[float] = None
    ccd_iterations: Optional[int] = None
    ccd_tolerance: Optional[float] = None
    enable_multiccd: Optional[bool] = None
    enable_sleep: Optional[bool] = None
    sleep_tolerance: Optional[float] = None


def _sim_options_from_wire(opts: Mapping[str, Any]) -> SimOptions:
    def _vec3(v: Any) -> Optional[Tuple[float, float, float]]:
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return (float(v[0]), float(v[1]), float(v[2]))
        return None

    def _f(v: Any) -> Optional[float]:
        return float(v) if isinstance(v, (int, float)) else None

    def _i(v: Any) -> Optional[int]:
        return int(v) if isinstance(v, (int, float)) else None

    def _s(v: Any) -> Optional[str]:
        return str(v) if isinstance(v, str) else None

    def _b(v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        return None

    return SimOptions(
        timestep=_f(opts.get("timestep")),
        gravity=_vec3(opts.get("gravity")),
        wind=_vec3(opts.get("wind")),
        magnetic=_vec3(opts.get("magnetic")),
        density=_f(opts.get("density")),
        viscosity=_f(opts.get("viscosity")),
        impratio=_f(opts.get("impratio")),
        tolerance=_f(opts.get("tolerance")),
        iterations=_i(opts.get("iterations")),
        ls_iterations=_i(opts.get("ls_iterations")),
        integrator=_s(opts.get("integrator")),
        cone=_s(opts.get("cone")),
        solver=_s(opts.get("solver")),
        noslip_iterations=_i(opts.get("noslip_iterations")),
        noslip_tolerance=_f(opts.get("noslip_tolerance")),
        ccd_iterations=_i(opts.get("ccd_iterations")),
        ccd_tolerance=_f(opts.get("ccd_tolerance")),
        enable_multiccd=_b(opts.get("enable_multiccd")),
        enable_sleep=_b(opts.get("enable_sleep")),
        sleep_tolerance=_f(opts.get("sleep_tolerance")),
    )


# ---------------------------------------------------------------------------
# Outliner namespace
# ---------------------------------------------------------------------------


@dataclass
class ActorInfo:
    """One row of :meth:`URLabClient.outliner.list_actors`.

    Field names match wire keys. ``actor_id`` is empty when the actor
    isn't an URLab-managed (id-tagged) actor. The ``static`` /
    ``complex_mesh`` / ``driven_by_unreal`` / ``coacd_threshold`` /
    ``friction`` fields mirror the inner ``quick_convert`` block when
    present; ``None`` when ``has_quick_convert`` is False.
    """

    name: str
    actor_class: str                          # wire key: "class"
    actor_id: str
    is_articulation: bool
    has_quick_convert: bool
    location: Tuple[float, float, float]
    rotation_quat: Tuple[float, float, float, float]
    label: str = ""
    is_static_mesh_actor: bool = False
    is_light: bool = False
    static: Optional[bool] = None
    complex_mesh: Optional[bool] = None
    coacd_threshold: Optional[float] = None
    driven_by_unreal: Optional[bool] = None
    friction: Optional[Tuple[float, float, float]] = None


def _actor_info_from_wire(a: Mapping[str, Any]) -> ActorInfo:
    loc = a.get("location") or (0.0, 0.0, 0.0)
    rq = a.get("rotation_quat") or (0.0, 0.0, 0.0, 1.0)
    qc = a.get("quick_convert") if a.get("has_quick_convert") else None
    fr = qc.get("friction") if qc else None
    return ActorInfo(
        name=str(a.get("name", "") or ""),
        actor_class=str(a.get("class", "") or ""),
        actor_id=str(a.get("actor_id", "") or ""),
        is_articulation=bool(a.get("is_articulation", False)),
        has_quick_convert=bool(a.get("has_quick_convert", False)),
        location=(float(loc[0]), float(loc[1]), float(loc[2])),
        rotation_quat=(float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])),
        label=str(a.get("label", "") or ""),
        is_static_mesh_actor=bool(a.get("is_static_mesh_actor", False)),
        is_light=bool(a.get("is_light", False)),
        static=bool(qc["static"]) if qc and "static" in qc else None,
        complex_mesh=bool(qc["complex_mesh"]) if qc and "complex_mesh" in qc else None,
        coacd_threshold=(
            float(qc["coacd_threshold"]) if qc and "coacd_threshold" in qc else None
        ),
        driven_by_unreal=(
            bool(qc["driven_by_unreal"]) if qc and "driven_by_unreal" in qc else None
        ),
        friction=(
            (float(fr[0]), float(fr[1]), float(fr[2]))
            if isinstance(fr, (list, tuple)) and len(fr) == 3 else None
        ),
    )


@dataclass
class BlueprintInfo:
    class_path: str       # wire key: "blueprint_class_path"
    short_name: str       # wire key: "blueprint_short_name"


def _blueprint_info_from_wire(b: Mapping[str, Any]) -> BlueprintInfo:
    return BlueprintInfo(
        class_path=str(b.get("blueprint_class_path", "") or ""),
        short_name=str(b.get("blueprint_short_name", "") or ""),
    )


@dataclass
class ActorBounds:
    """AABB of an actor's components, MJ metres."""

    actor_name: str
    min: Tuple[float, float, float]
    max: Tuple[float, float, float]
    center: Tuple[float, float, float]
    extents: Tuple[float, float, float]


def _vec3(v: Any, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    return default


def _actor_bounds_from_wire(r: Mapping[str, Any]) -> ActorBounds:
    return ActorBounds(
        actor_name=str(r.get("actor_name", "") or ""),
        min=_vec3(r.get("min")),
        max=_vec3(r.get("max")),
        center=_vec3(r.get("center")),
        extents=_vec3(r.get("extents")),
    )


@dataclass
class SceneSnapshotActor:
    """One actor entry in :meth:`URLabClient.scene.snapshot`.

    ``urlab`` is set only for AMjArticulation actors and carries the
    MJ-side metadata (joint / actuator / sensor / camera names)."""

    name: str
    actor_class: str
    actor_id: str
    label: str
    location: Tuple[float, float, float]
    rotation_quat: Tuple[float, float, float, float]
    tags: List[str] = field(default_factory=list)
    urlab: Optional[Dict[str, Any]] = None


@dataclass
class SceneSnapshot:
    level_path: str
    in_pie: bool
    actors: List[SceneSnapshotActor] = field(default_factory=list)


def _scene_snapshot_from_wire(r: Mapping[str, Any]) -> SceneSnapshot:
    raw = r.get("actors") or []
    actors: List[SceneSnapshotActor] = []
    for a in raw:
        rq = a.get("rotation_quat") or (0.0, 0.0, 0.0, 1.0)
        actors.append(SceneSnapshotActor(
            name=str(a.get("name", "") or ""),
            actor_class=str(a.get("class", "") or ""),
            actor_id=str(a.get("actor_id", "") or ""),
            label=str(a.get("label", "") or ""),
            location=_vec3(a.get("location")),
            rotation_quat=(float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])),
            tags=[str(t) for t in (a.get("tags") or [])],
            urlab=a.get("urlab"),
        ))
    return SceneSnapshot(
        level_path=str(r.get("level_path", "") or ""),
        in_pie=bool(r.get("in_pie", False)),
        actors=actors,
    )


@dataclass
class ActorHierarchyNode:
    name: str
    actor_class: str
    location: Tuple[float, float, float]
    children: List["ActorHierarchyNode"] = field(default_factory=list)


def _hierarchy_from_wire(r: Mapping[str, Any]) -> ActorHierarchyNode:
    return ActorHierarchyNode(
        name=str(r.get("name", "") or ""),
        actor_class=str(r.get("class", "") or ""),
        location=_vec3(r.get("location")),
        children=[_hierarchy_from_wire(c) for c in (r.get("children") or [])],
    )


# ---------------------------------------------------------------------------
# Mocap / contacts (runtime namespace)
# ---------------------------------------------------------------------------


@dataclass
class MocapPose:
    """MuJoCo mocap body pose. ``pos`` MJ metres, ``quat`` wxyz."""

    body: str
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]


def _mocap_pose_from_wire(r: Mapping[str, Any]) -> MocapPose:
    q = r.get("quat") or (1.0, 0.0, 0.0, 0.0)
    return MocapPose(
        body=str(r.get("body", "") or ""),
        pos=_vec3(r.get("pos")),
        quat=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )


@dataclass
class Contact:
    """One MuJoCo contact entry from :meth:`URLabClient.runtime.get_contacts`.

    ``force`` is a 6-vector ``[fx, fy, fz, tx, ty, tz]`` in the contact frame
    as returned by ``mj_contactForce``. ``dist`` is negative when penetrating.
    Geom / body names may be empty for anonymous entities."""

    geom1: str
    geom2: str
    body1: str
    body2: str
    pos: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    dist: float
    force: Tuple[float, float, float, float, float, float]


@dataclass
class ContactsResult:
    n_contacts: int
    truncated: bool
    contacts: List[Contact] = field(default_factory=list)


def _contact_from_wire(c: Mapping[str, Any]) -> Contact:
    f = c.get("force") or (0.0,) * 6
    return Contact(
        geom1=str(c.get("geom1", "") or ""),
        geom2=str(c.get("geom2", "") or ""),
        body1=str(c.get("body1", "") or ""),
        body2=str(c.get("body2", "") or ""),
        pos=_vec3(c.get("pos")),
        normal=_vec3(c.get("normal")),
        dist=float(c.get("dist", 0.0) or 0.0),
        force=(
            float(f[0]), float(f[1]), float(f[2]),
            float(f[3]), float(f[4]), float(f[5]),
        ),
    )


def _contacts_result_from_wire(r: Mapping[str, Any]) -> ContactsResult:
    return ContactsResult(
        n_contacts=int(r.get("n_contacts", 0) or 0),
        truncated=bool(r.get("truncated", False)),
        contacts=[_contact_from_wire(c) for c in (r.get("contacts") or [])],
    )


# ---------------------------------------------------------------------------
# Keyframes (runtime namespace)
# ---------------------------------------------------------------------------


@dataclass
class KeyframeInfo:
    """One MJCF ``<keyframe>`` entry compiled into the model.

    All MJ arrays are emitted in compiled order: ``qpos[nq]``,
    ``qvel[nv]``, ``ctrl[nu]``, ``mocap_pos[3 * nmocap]``,
    ``mocap_quat[4 * nmocap]``. Use the keyframe ``name`` with
    :meth:`URLabClient.reset` (``keyframe_name=...``) to load.
    """

    name: str
    time: float
    qpos: List[float] = field(default_factory=list)
    qvel: List[float] = field(default_factory=list)
    ctrl: List[float] = field(default_factory=list)
    mocap_pos: List[float] = field(default_factory=list)
    mocap_quat: List[float] = field(default_factory=list)


def _keyframe_info_from_wire(k: Mapping[str, Any]) -> KeyframeInfo:
    return KeyframeInfo(
        name=str(k.get("name", "") or ""),
        time=float(k.get("time", 0.0) or 0.0),
        qpos=[float(x) for x in (k.get("qpos")       or [])],
        qvel=[float(x) for x in (k.get("qvel")       or [])],
        ctrl=[float(x) for x in (k.get("ctrl")       or [])],
        mocap_pos=[float(x) for x in (k.get("mocap_pos")  or [])],
        mocap_quat=[float(x) for x in (k.get("mocap_quat") or [])],
    )


# ---------------------------------------------------------------------------
# Viewport (viewport namespace)
# ---------------------------------------------------------------------------


@dataclass
class CameraPose:
    """Perspective-viewport camera pose. ``location`` in MJ metres;
    ``rotation_quat`` is xyzw (UE FQuat order); ``rotation_euler`` is
    (roll, pitch, yaw) degrees; ``fov`` is horizontal FOV in degrees."""

    location: Tuple[float, float, float]
    rotation_quat: Tuple[float, float, float, float]
    rotation_euler: Tuple[float, float, float]
    fov: float


def _camera_pose_from_wire(r: Mapping[str, Any]) -> CameraPose:
    q = r.get("rotation_quat") or (0.0, 0.0, 0.0, 1.0)
    e = r.get("rotation_euler") or (0.0, 0.0, 0.0)
    return CameraPose(
        location=_vec3(r.get("location")),
        rotation_quat=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        rotation_euler=(float(e[0]), float(e[1]), float(e[2])),
        fov=float(r.get("fov", 0.0) or 0.0),
    )


# ---------------------------------------------------------------------------
# Recording / replay namespaces
# ---------------------------------------------------------------------------


@dataclass
class RecordingHandle:
    name: str
    max_duration_s: Optional[float] = None


@dataclass
class RecordingSummary:
    frame_count: int
    sim_duration_s: float


@dataclass
class ReplaySession:
    name: str
    total_frames: int = 0
    source_path: Optional[Path] = None


@dataclass
class ReplayStatus:
    active_session: str
    total_frames: int = 0
