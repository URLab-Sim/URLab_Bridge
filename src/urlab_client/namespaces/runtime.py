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

"""`client.runtime.*` — physics-runtime mutators while PIE is live."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING, Union

from .base import _RpcNamespace
from ..enums import StepMode, coerce, wire
from .._op_helpers import target_payload
from ..results import (
    ContactsResult,
    KeyframeInfo,
    MocapPose,
    SimOptions,
    _contacts_result_from_wire,
    _keyframe_info_from_wire,
    _mocap_pose_from_wire,
    _sim_options_from_wire,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient


class _RuntimeNamespace(_RpcNamespace):
    """`client.runtime.*` — physics-runtime mutators while PIE is live.

    Result type: :class:`SimOptions` lives in ``urlab_client.results``.
    """

    def __init__(self, client: "URLabClient"):
        super().__init__(client, "runtime")

    def set_paused(self, paused: bool) -> bool:
        reply = self._client._rpc(
            "set_paused", {"paused": bool(paused)},
            expected_op="set_paused_ok",
        )
        return bool(reply.get("paused", paused))

    def set_camera_streaming(
        self, cameras: Mapping[str, Union[bool, Mapping[str, bool]]]
    ) -> Dict[str, Any]:
        """Enable/disable per-camera ZMQ/SHM broadcast streams at runtime.

        Needed because UE's ``bEnableAllCameras`` now defaults off: a camera
        only runs its pub streams while broadcast-enabled (here) or requested
        via ``include_cameras``. Keys are canonical camera names (the
        ``camera_topics`` keys from the handshake). Values:

        - ``True`` / ``False`` — both transports on / off
        - ``{"zmq": bool, "shm": bool}`` — per-transport

        Returns the per-camera reply ``{canonical: {streaming, zmq, shm,
        zmq_endpoint, zmq_topic}}`` so you know exactly where to subscribe.
        """
        wire: Dict[str, Any] = {}
        for key, val in cameras.items():
            if isinstance(val, Mapping):
                entry: Dict[str, bool] = {}
                if "zmq" in val:
                    entry["zmq"] = bool(val["zmq"])
                if "shm" in val:
                    entry["shm"] = bool(val["shm"])
                wire[str(key)] = entry
            else:
                wire[str(key)] = bool(val)
        reply = self._client._rpc(
            "set_camera_streaming", {"cameras": wire},
            expected_op="set_camera_streaming_ok",
        )
        return dict(reply.get("cameras") or {})

    def set_sim_speed(self, percent: float) -> float:
        reply = self._client._rpc(
            "set_sim_speed", {"percent": float(percent)},
            expected_op="set_sim_speed_ok",
        )
        return float(reply.get("percent", percent))

    def set_control_source(
        self, source: str, *, articulation: Optional[str] = None
    ) -> None:
        """Choose which input writes to ``mjData.ctrl`` on the UE side.

        ``source`` is ``"zmq"`` (this client / the legacy ctrl SUB) or
        ``"ui"`` (the in-editor dashboard sliders). Pass
        ``articulation=<prefix>`` to scope the change to a single
        articulation; omitted, it sets the global default on the manager.
        """
        if source not in ("zmq", "ui"):
            raise ValueError(f"source must be 'zmq' or 'ui', got {source!r}")
        payload: Dict[str, Any] = {"source": source}
        if articulation is not None:
            payload["articulation"] = str(articulation)
        self._client._rpc(
            "set_control_source", payload, expected_op="set_control_source_ok",
        )

    def set_twist(
        self,
        articulation: str,
        *,
        linear: Sequence[float] = (0.0, 0.0, 0.0),
        angular: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Set the ROS-Twist input on an articulation's
        ``UMjTwistController``.

        ``linear`` is ``(vx, vy, vz)`` in m/s; ``angular`` is
        ``(wx, wy, wz)`` in rad/s. Both are body-frame, matching the
        ROS Twist message convention the gait clock reads. No-op (silent)
        for articulations that don't have a ``UMjTwistController``."""
        self._client._rpc(
            "set_twist",
            {
                "articulation": str(articulation),
                "linear":  [float(x) for x in linear],
                "angular": [float(x) for x in angular],
            },
            expected_op="set_twist_ok",
        )

    def set_qpos(
        self,
        target: str,
        qpos: Sequence[float],
        *,
        by_name: bool = False,
    ) -> None:
        """Write qpos for a single articulation; mirrors echo into client.data."""
        payload: Dict[str, Any] = {
            **target_payload(target, by_name=by_name),
            "qpos": [float(x) for x in qpos],
        }
        reply = self._client._rpc("set_qpos", payload, expected_op="set_qpos_ok")
        self._client._mirror_set_qpos_locally(reply)

    def set_sim_options(
        self,
        *,
        timestep: Optional[float] = None,
        gravity: Optional[Sequence[float]] = None,
        wind: Optional[Sequence[float]] = None,
        magnetic: Optional[Sequence[float]] = None,
        density: Optional[float] = None,
        viscosity: Optional[float] = None,
        impratio: Optional[float] = None,
        tolerance: Optional[float] = None,
        iterations: Optional[int] = None,
        ls_iterations: Optional[int] = None,
        integrator: Optional[str] = None,
        cone: Optional[str] = None,
        solver: Optional[str] = None,
        noslip_iterations: Optional[int] = None,
        noslip_tolerance: Optional[float] = None,
        ccd_iterations: Optional[int] = None,
        ccd_tolerance: Optional[float] = None,
        enable_multiccd: Optional[bool] = None,
        enable_sleep: Optional[bool] = None,
        sleep_tolerance: Optional[float] = None,
        disableflags: Optional[int] = None,
        enableflags: Optional[int] = None,
        num_worker_threads: Optional[int] = None,
    ) -> SimOptions:
        """Push MuJoCo sim options into the live UE model. MuJoCo-native
        SI units. Only fields you pass override; everything else keeps
        its compiled value. Mirrors back into client.model.opt so local
        readers (decimation calc, etc.) see live UE state."""
        opts: Dict[str, Any] = {}
        if timestep          is not None: opts["timestep"]          = float(timestep)
        if gravity           is not None: opts["gravity"]           = [float(x) for x in gravity]
        if wind              is not None: opts["wind"]              = [float(x) for x in wind]
        if magnetic          is not None: opts["magnetic"]          = [float(x) for x in magnetic]
        if density           is not None: opts["density"]           = float(density)
        if viscosity         is not None: opts["viscosity"]         = float(viscosity)
        if impratio          is not None: opts["impratio"]          = float(impratio)
        if tolerance         is not None: opts["tolerance"]         = float(tolerance)
        if iterations        is not None: opts["iterations"]        = int(iterations)
        if ls_iterations     is not None: opts["ls_iterations"]     = int(ls_iterations)
        if integrator        is not None: opts["integrator"]        = str(integrator)
        if cone              is not None: opts["cone"]              = str(cone)
        if solver            is not None: opts["solver"]            = str(solver)
        if noslip_iterations is not None: opts["noslip_iterations"] = int(noslip_iterations)
        if noslip_tolerance  is not None: opts["noslip_tolerance"]  = float(noslip_tolerance)
        if ccd_iterations    is not None: opts["ccd_iterations"]    = int(ccd_iterations)
        if ccd_tolerance     is not None: opts["ccd_tolerance"]     = float(ccd_tolerance)
        if enable_multiccd   is not None: opts["enable_multiccd"]   = bool(enable_multiccd)
        if enable_sleep      is not None: opts["enable_sleep"]      = bool(enable_sleep)
        if sleep_tolerance   is not None: opts["sleep_tolerance"]   = float(sleep_tolerance)
        if disableflags      is not None: opts["disableflags"]      = int(disableflags)
        if enableflags       is not None: opts["enableflags"]       = int(enableflags)
        if num_worker_threads is not None: opts["num_worker_threads"] = int(num_worker_threads)

        if not opts:
            raise ValueError("set_sim_options requires at least one field")

        client = self._client
        reply = client._rpc(
            "set_sim_options", {"options": opts},
            expected_op="set_sim_options_ok",
        )
        result = dict(reply.get("options", {}))

        if client.model is not None:
            opt = client.model.opt
            for key, attr in (
                ("timestep",          "timestep"),
                ("density",           "density"),
                ("viscosity",         "viscosity"),
                ("impratio",          "impratio"),
                ("tolerance",         "tolerance"),
                ("iterations",        "iterations"),
                ("ls_iterations",     "ls_iterations"),
                ("noslip_iterations", "noslip_iterations"),
                ("noslip_tolerance",  "noslip_tolerance"),
                ("ccd_iterations",    "ccd_iterations"),
                ("ccd_tolerance",     "ccd_tolerance"),
                ("sleep_tolerance",   "sleep_tolerance"),
            ):
                if key in opts and key in result:
                    setattr(opt, attr, result[key])
            for key, attr in (("gravity", "gravity"), ("wind", "wind"), ("magnetic", "magnetic")):
                if key in opts and key in result:
                    vec = result[key]
                    for i, v in enumerate(vec):
                        opt.__getattribute__(attr)[i] = v

        return _sim_options_from_wire(result)

    def set_mode(self, mode: Union[str, StepMode]) -> StepMode:
        """Promote / demote the server's step mode. Side effect: in
        live we keep streaming SUB sockets up, in direct/puppet
        we tear them down (server pauses publishers in those modes)."""
        coerced = coerce(StepMode, mode)
        reply = self._client._rpc(
            "set_mode", {"mode": wire(coerced)},
            expected_op="set_mode_ok",
        )
        new_mode = coerce(StepMode, reply.get("current_mode", coerced))
        self._client.step_mode = new_mode
        if new_mode == StepMode.LIVE:
            self._client._start_streaming_subs()
        else:
            self._client._stop_streaming_subs()
        return new_mode

    def set_mocap_pose(
        self,
        body: str,
        *,
        pos: Optional[Sequence[float]] = None,
        quat: Optional[Sequence[float]] = None,
    ) -> MocapPose:
        """Write a mocap body's pose into the live MJ data. ``body`` is
        the compiled MJ body name (URLab prefixes already applied; check
        :meth:`URLabClient.scene.snapshot` if unsure). At least one of
        ``pos`` (3-vec, MJ metres) or ``quat`` (4-vec wxyz) must be set;
        the other slot keeps its current value. Reply echoes the applied
        pose. Must be called between steps for the override to apply.
        """
        if pos is None and quat is None:
            raise ValueError("set_mocap_pose requires at least one of pos or quat")
        payload: Dict[str, Any] = {"body": str(body)}
        if pos is not None:
            payload["pos"] = [float(x) for x in pos]
        if quat is not None:
            payload["quat"] = [float(x) for x in quat]
        reply = self._client._rpc(
            "set_mocap_pose", payload, expected_op="set_mocap_pose_ok",
        )
        return _mocap_pose_from_wire(reply)

    def read_mocap_pose(self, body: str) -> MocapPose:
        """Read a mocap body's current pose from MJ data."""
        reply = self._client._rpc(
            "read_mocap_pose", {"body": str(body)},
            expected_op="read_mocap_pose_ok",
        )
        return _mocap_pose_from_wire(reply)

    def list_keyframes(self) -> "list[KeyframeInfo]":
        """Enumerate ``<keyframe>`` entries compiled into the model.
        Each entry carries the full MJ-side state (qpos / qvel / ctrl /
        mocap). Pair with :meth:`URLabClient.reset` (``keyframe_name=...``)
        to load one."""
        reply = self._client._rpc(
            "list_keyframes", {}, expected_op="list_keyframes_ok",
        )
        return [_keyframe_info_from_wire(k) for k in (reply.get("keyframes") or [])]

    def get_contacts(
        self,
        *,
        body1: Optional[str] = None,
        body2: Optional[str] = None,
        geom1: Optional[str] = None,
        geom2: Optional[str] = None,
        max_contacts: int = 64,
    ) -> ContactsResult:
        """Snapshot active MuJoCo contacts. Filters are AND-combined and
        match exact compiled MJ names. ``max_contacts`` (default 64)
        caps the reply; the ``truncated`` flag indicates the cap was
        hit. ``force`` per contact is the 6-vec from ``mj_contactForce``
        (contact-frame [fx, fy, fz, tx, ty, tz]); ``dist`` is negative
        when penetrating."""
        payload: Dict[str, Any] = {"max_contacts": int(max_contacts)}
        flt: Dict[str, str] = {}
        if body1 is not None: flt["body1"] = str(body1)
        if body2 is not None: flt["body2"] = str(body2)
        if geom1 is not None: flt["geom1"] = str(geom1)
        if geom2 is not None: flt["geom2"] = str(geom2)
        if flt:
            payload["filter"] = flt
        reply = self._client._rpc(
            "get_contacts", payload, expected_op="get_contacts_ok",
        )
        return _contacts_result_from_wire(reply)
