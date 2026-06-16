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

"""URLab remote-stepping client. See :mod:`urlab_client` for the public surface."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .articulation import URLabArticulation, URLabCameraView, URLabEntity
from .enums import (
    CameraMode,
    ObservationLevel,
    StepMode,
    coerce,
    wire,
)
from .errors import URLabRPCError, URLabTimeoutError, URLabVersionMismatch
from .namespaces.debug import _DebugNamespace
from .namespaces.outliner import _OutlinerNamespace
from .namespaces.recording import URLabRecordingAPI
from .namespaces.replay import URLabReplayAPI
from .namespaces.runtime import _RuntimeNamespace
from .namespaces.scene import _SceneNamespace
from .namespaces.viewport import _ViewportNamespace
from .namespaces.sim import _SimNamespace
from .transports import Transport, make_transport

logger = logging.getLogger(__name__)

# Per-op recv-timeout defaults (seconds), applied by ``_rpc`` when the caller
# doesn't pass an explicit ``rcv_timeout_ms``. Long editor / PIE / handshake ops
# get a generous window so callers never set a huge GLOBAL timeout just to make
# one slow op survive. Ops not listed use the transport default (~5s).
_OP_TIMEOUTS_S: Dict[str, float] = {
    "hello": 30.0,          # handshake embeds the (possibly large) MJB
    "begin_pie": 35.0,      # UE compile + PIE start
    "stop_pie": 30.0,
    "import_xml": 120.0,    # mesh clean subprocess + Blueprint compile
    "create_level": 30.0,
    "load_level": 30.0,
    "save_level": 30.0,
    "spawn_actor": 30.0,
    "spawn_grid": 60.0,
    "spawn_light": 30.0,
    "duplicate_actor": 30.0,
}


@dataclass
class Readiness:
    """Summary returned by :meth:`URLabClient.bringup` once the session is set
    up and ready to drive."""
    mode: "StepMode"
    n_articulations: int
    n_cameras: int = 0
    cameras_ready: int = 0
    sim_dt_applied: Optional[float] = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Readiness(mode={self.mode}, articulations={self.n_articulations}, "
            f"cameras_ready={self.cameras_ready}/{self.n_cameras}, "
            f"sim_dt={self.sim_dt_applied})"
        )


# Optional at import-time so `from urlab_client import StepMode`
# works in environments without msgpack / zmq / mujoco installed (the
# enum tests want that). Real use requires all three.
try:  # pragma: no cover - trivial import guard
    import msgpack  # type: ignore
except ImportError:  # pragma: no cover
    msgpack = None  # noqa: N816

try:  # pragma: no cover
    import zmq  # type: ignore
except ImportError:  # pragma: no cover
    zmq = None  # noqa: N816

try:  # pragma: no cover
    import mujoco  # type: ignore
except ImportError:  # pragma: no cover
    mujoco = None  # noqa: N816


class URLabClient:
    """Session-oriented step client. `step_mode` accepts a string or `StepMode` member."""

    def __init__(
        self,
        address: str = "tcp://localhost",
        *,
        step_mode: Union[str, StepMode] = "auto",
        step_port: int = 5559,
        state_port: int = 5555,
        ctrl_port: int = 5556,
        info_port: int = 5557,
        mujoco_version_check: bool = True,
        local_model: bool = True,
        rcv_timeout_ms: int = 5000,
        auto_promote_step_mode: bool = True,
        transport: Union[str, Transport] = "zmq",
        shm_dir: Optional[str] = None,
    ):
        self.address = address
        self.step_mode: StepMode = coerce(StepMode, step_mode, default=StepMode.AUTO)
        self.step_port = step_port
        self.state_port = state_port
        self.ctrl_port = ctrl_port
        self.info_port = info_port
        self.mujoco_version_check = mujoco_version_check
        self.local_model = local_model
        self._rcv_timeout_ms = rcv_timeout_ms
        self._auto_promote_step_mode = auto_promote_step_mode

        self.session_id: Optional[str] = None
        self.urlab_version: Optional[str] = None
        self.mujoco_version: Optional[str] = None
        # False until PIE starts; editor-only ops still work pre-PIE.
        self.manager_present: bool = False
        self.shm_session_dir: str = ""
        self.model: Any = None
        self.data: Any = None
        self.sim_time: float = 0.0
        self.step_count: int = 0

        # ROS-Time clocks: sim_time_* is d->time; wall_time_* is unix
        # epoch on UE; recv_wall_time_ns is bridge-local recv time.
        self.sim_time_sec: int = 0
        self.sim_time_nsec: int = 0
        self.wall_time_sec: int = 0
        self.wall_time_nsec: int = 0
        self.recv_wall_time_ns: int = 0

        self.articulations: Dict[str, URLabArticulation] = {}
        self.articulations_by_id: Dict[str, URLabArticulation] = {}
        # Flat dict of every dynamic body. Articulations appear here too
        # (subclass of URLabEntity); plain bodies are URLabEntity instances.
        self.entities: Dict[str, URLabEntity] = {}
        self.global_cameras: Dict[str, URLabCameraView] = {}

        self.recording = URLabRecordingAPI(self)
        self.replay = URLabReplayAPI(self)

        # Server meta payload: { op_name: decl }. Namespace proxies consult
        # this to decide whether an attribute exists.
        self._ops_meta: Dict[str, Dict[str, Any]] = {}
        self.scene = _SceneNamespace(self)
        self.sim = _SimNamespace(self)
        self.runtime = _RuntimeNamespace(self)
        self.outliner = _OutlinerNamespace(self)
        self.debug = _DebugNamespace(self)
        self.viewport = _ViewportNamespace(self)

        # Entity-level xfrc buffer; cleared post-step. Per-articulation
        # xfrc is tracked separately on each URLabArticulation.
        self._pending_entity_xfrc: Dict[str, np.ndarray] = {}

        # transport="shm" defers actual SHM construction to discover()
        # so we can pull the session dir out of the handshake; until then
        # we use ZMQ for the hello round-trip.
        self._shm_dir_override: Optional[str] = shm_dir
        self._pending_shm_swap: bool = False
        if isinstance(transport, str):
            if transport in ("zmq", "shm"):
                self._transport: Transport = make_transport(
                    "zmq",
                    address,
                    step_port=step_port,
                    state_port=state_port,
                    rcv_timeout_ms=rcv_timeout_ms,
                )
                self._pending_shm_swap = (transport == "shm")
            else:
                raise ValueError(
                    f"unknown transport name {transport!r}; expected "
                    f"'zmq' or 'shm', or pass a Transport instance"
                )
        else:
            self._transport = transport

        # State-snapshot bookkeeping. Transport state thread fires
        # `_on_state_snapshot`; live-mode step waits on this cond.
        self._state_lock = threading.Lock()
        self._state_cond = threading.Condition(self._state_lock)
        self._latest_state_snapshot: Optional[Dict[str, Any]] = None
        # Post-state frame_id of the most recent step() reply; get_camera(fresh=True)
        # waits for a streamed frame >= this so the image matches the step state.
        self._last_step_frame_id: Optional[int] = None
        # Monotonic timestamp of the last state-stream snapshot; the liveness
        # oracle (server_alive) reads it to tell "busy" from "dead" while awaiting.
        self._last_snapshot_monotonic: Optional[float] = None
        # Idempotency guard for close().
        self._closed: bool = False
        # Wall-clock of the previous step(), for optional target_hz pacing.
        self._last_step_monotonic: Optional[float] = None
        self._state_msg_count: int = 0

        # Guards writes through `self.data` + the mj_forward calls.
        # Reentrant: `_absorb_step_reply` → `_mirror_state_into_data`
        # both acquire. Cross-thread readers must take this lock too.
        self._data_lock: threading.RLock = threading.RLock()

    # -- transport --------------------------------------------------------

    def _rpc(
        self,
        op: str,
        payload: Mapping[str, Any],
        *,
        expected_op: Optional[str] = None,
        rcv_timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send one request and unpack one reply. Raises on error replies.

        ``rcv_timeout_ms`` overrides the transport's default recv
        timeout for this single call. When omitted, a per-op default from
        ``_OP_TIMEOUTS_S`` is applied so long editor/PIE ops get a generous
        window automatically (no caller-guessed global timeout); ops not in
        the registry use the transport default."""
        if rcv_timeout_ms is None:
            op_default_s = _OP_TIMEOUTS_S.get(op)
            if op_default_s is not None:
                rcv_timeout_ms = int(op_default_s * 1000)
        request = {"op": op, "session_id": self.session_id, **payload}
        reply = self._transport.rpc(request, rcv_timeout_ms=rcv_timeout_ms)
        if not isinstance(reply, dict):
            raise RuntimeError(f"non-dict reply to {op!r}: {type(reply).__name__}")
        reply_op = reply.get("op")
        if reply_op == "error":
            code = reply.get("code", "unknown")
            message = reply.get("message", "")
            raise URLabRPCError(code, message, op=op)
        if expected_op is not None and reply_op != expected_op:
            raise URLabRPCError(
                "unexpected_reply_op",
                f"wanted {expected_op!r}, got {reply_op!r}",
                op=op,
            )
        return dict(reply)

    def _rpc_configure_controller(
        self, *, articulation: str, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._rpc(
            "configure_controller",
            {"articulation": articulation, "params": dict(params)},
            expected_op="configure_controller_ok",
        )

    # -- session lifecycle ------------------------------------------------

    def discover(self, observations: Union[str, ObservationLevel] = "standard") -> None:
        """Handshake: send `hello`, load the MJB, construct articulation
        wrappers. Raises on a version mismatch unless
        `mujoco_version_check=False` was set.

        After the handshake, if the user constructed the client with an
        explicit `step_mode` (`direct` or `puppet`), tell the server to
        switch into that mode. The UE step server defaults to
        `live` and rejects `step` requests until a `set_mode`
        promotes it.
        """
        obs_str = wire(coerce(ObservationLevel, observations))
        # Always pin encoding=msgpack on hello. The server's encoding
        # flag is global, so leaving it implicit means we inherit
        # whatever the previous session set (e.g. a debugging client
        # that asked for JSON, leaving the server stuck in JSON mode).
        reply = self._rpc(
            "hello",
            {
                "client_version": self._client_version(),
                "observations": obs_str,
                "encoding": "msgpack",
            },
            expected_op="hello_ok",
        )
        self._apply_handshake(reply)

        # Fetch the server schema via `meta`. Lock-step bridge ↔ server:
        # every op the server registers becomes available on the right
        # `client.<namespace>` namespace via __getattr__. New server ops
        # appear without a bridge release. Older servers without `meta`
        # reply `unknown_op` — we tolerate that and leave `_ops_meta`
        # empty; only synthesised paths break, hand-written methods keep
        # working.
        try:
            meta_reply = self._rpc("meta", {}, expected_op="meta_ok")
            ops = meta_reply.get("ops", []) or []
            self._ops_meta = {
                str(o["name"]): {
                    "name": str(o["name"]),
                    "category": str(o.get("category", "")),
                    "namespace": str(o.get("namespace", "")),
                    "required_fields": list(o.get("required_fields", []) or []),
                    "reply_fields": list(o.get("reply_fields", []) or []),
                }
                for o in ops
                if isinstance(o, dict) and o.get("name")
            }
        except URLabRPCError as exc:
            if exc.code in ("unknown_op", "missing_op"):
                logger.debug(
                    "discover(): server has no `meta` op; namespace "
                    "synthesis disabled, hand-written wrappers still work"
                )
                self._ops_meta = {}
            else:
                raise

        # Editor-time / pre-PIE handshake: no manager registered, no MJB,
        # no articulations. Skip every PIE-only follow-up (SHM swap, mode
        # promote, streaming SUB startup). Caller can still drive editor
        # ops (import_xml, spawn_actor, begin_pie) and re-discover via
        # begin_pie's embedded handshake when PIE comes up.
        if not self.manager_present:
            logger.info(
                "discover(): no manager registered (editor-time / pre-PIE). "
                "Editor-only ops are available; call begin_pie or wait for "
                "the user to hit Play before stepping."
            )
            return

        # If the user asked for transport="shm", swap the temporary ZMQ
        # transport for a real SHM transport now that the handshake has
        # told us where to look. The ZMQ transport stays alive as the
        # SHM transport's fallback for ops too large for the slot.
        if self._pending_shm_swap:
            self._activate_shm_transport()

        if (
            self._auto_promote_step_mode
            and self.step_mode in (StepMode.DIRECT, StepMode.PUPPET)
        ):
            try:
                self.runtime.set_mode(self.step_mode)
            except URLabRPCError as exc:
                if exc.code == "mode_locked_by_server":
                    logger.warning(
                        "Server StepMode is locked; client requested %s but "
                        "server stays on its pinned mode. Subsequent step "
                        "requests may fail with mode_mismatch.",
                        self.step_mode.value,
                    )
                else:
                    raise

        # Spin up streaming SUBs in EVERY mode. Cameras are served from the
        # async SHM/ZMQ streams in all step modes now (not bundled into the
        # step reply), so puppet and direct need the SUBs running too. This is
        # what decouples camera rate from step rate -- a puppet step at 30Hz
        # no longer blocks on (or bloats its RPC reply with) a camera readback.
        # set_camera_streaming inside enables the per-camera broadcast.
        self._start_streaming_subs()

    def _apply_handshake(self, reply: Mapping[str, Any]) -> None:
        """Shared entry point used by `discover()` and tests that inject
        a canned handshake without the socket round-trip."""
        # Hold _data_lock around model/data swap so a concurrent reader
        # (state-stream worker, future async caller) can't see a
        # half-replaced model.
        with self._data_lock:
            self._apply_handshake_locked(reply)

    def _apply_handshake_locked(self, reply: Mapping[str, Any]) -> None:
        self.session_id = reply.get("session_id")
        self.urlab_version = reply.get("urlab_version")
        self.mujoco_version = reply.get("mujoco_version")
        # Server defaults to manager_present=true for replies that omit
        # the field (older server builds + the PIE-time begin_pie reply
        # always has a manager). Editor-time hello explicitly sets false.
        self.manager_present = bool(reply.get("manager_present", True))
        self.shm_session_dir = str(reply.get("shm_session_dir", "") or "")

        if self.mujoco_version_check and mujoco is not None:
            server_ver = str(self.mujoco_version or "")
            client_ver = mujoco.__version__
            if server_ver and server_ver != client_ver:
                raise URLabVersionMismatch(
                    f"MuJoCo version mismatch: server={server_ver!r} "
                    f"client={client_ver!r}. Pin both sides or pass "
                    f"mujoco_version_check=False to bypass."
                )

        # Load MJB into a local MjModel.
        mjb_bytes = reply.get("mjb")
        if self.local_model and mjb_bytes and mujoco is not None:
            self.model, self.data = _load_mjb(mjb_bytes)

        # Build articulations
        self.articulations = {}
        self.articulations_by_id = {}
        for art in reply.get("articulations", []):
            prefix = art.get("prefix")
            if not prefix:
                continue
            wrapper = URLabArticulation(
                prefix=prefix,
                model=self.model,
                data=self.data,
                handshake=art,
                client=self,
            )
            self.articulations[prefix] = wrapper
            if wrapper.actor_id:
                self.articulations_by_id[wrapper.actor_id] = wrapper

        # Non-articulation entities -- ship in handshake under `entities`,
        # optional. Modeled as plain `URLabEntity` instances; articulations
        # are the same type with extras (joints / actuators / etc.) and
        # ride in the `articulations` block.
        self.entities = {}
        self.entities.update(self.articulations)
        for name, payload in (reply.get("entities") or {}).items():
            entity = URLabEntity(
                name=name,
                body_id=int(payload.get("id", -1)),
                has_free_base=bool(payload.get("has_free_base", False)),
                client=self,
            )
            entity.free_joint = payload.get("free_joint")
            entity.free_joint_id = payload.get("free_joint_id")
            entity.qpos_offset = payload.get("qpos_offset")
            entity.qvel_offset = payload.get("qvel_offset")
            self.entities[name] = entity

        # Global cameras — reserved slot, empty today but accept any
        # payload for forward compatibility
        for cam_name, cam_payload in (reply.get("global_cameras") or {}).items():
            self.global_cameras[cam_name] = URLabCameraView.from_handshake(
                cam_name, cam_payload, owner=None
            )

    def _client_version(self) -> str:
        return "urlab_bridge/0.1.0-alpha"

    def _activate_shm_transport(self) -> None:
        """Replace the bootstrap ZMQ transport with a real ShmTransport
        once the handshake has provided the session dir. The existing ZMQ
        transport is reused as the SHM transport's fallback (for ops too
        large for the SHM slot, notably `hello`)."""
        shm_dir = self._shm_dir_override or self.shm_session_dir
        if not shm_dir:
            raise RuntimeError(
                "transport='shm' requested but neither shm_dir override nor "
                "handshake `shm_session_dir` was set; pass shm_dir explicitly"
            )
        # The SHM session id is the basename of the dir -- UE's
        # USmStepTransport uses it to name its kernel events
        # (`Local\URLab_<sid>_req_ready`), and the bridge must use the
        # same name to OpenEventW. Distinct from `self.session_id`,
        # which is the dispatcher's per-hello RPC session GUID.
        shm_session_id = os.path.basename(os.path.normpath(shm_dir)) or "live"
        self._transport = make_transport(
            "shm",
            self.address,
            shm_dir=shm_dir,
            shm_session_id=shm_session_id,
            fallback=self._transport,
        )
        self._pending_shm_swap = False
        logger.info(
            "URLabClient: SHM transport active (dir=%s, session=%s)",
            shm_dir, shm_session_id,
        )

    # -- step / reset -----------------------------------------------------

    def step(
        self,
        n_steps: int = 1,
        *,
        include_cameras: Union[bool, Mapping[str, Any]] = False,
        camera_query: str = "latest",
        camera_timeout_s: float = 0.5,
        observations: Union[str, ObservationLevel] = "standard",
        target_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advance the sim. Behaviour per `self.step_mode`:

        - `direct`: UE steps `n_steps`. Payload carries `ctrl`.
        - `puppet`: client calls `mj_step(client.model, client.data) × n_steps`
          locally, pushes the resulting full qpos/qvel to UE for rendering.
          `n_steps=0` is a supported escape hatch: skip local `mj_step`
          entirely and just push whatever is already in `client.data`
          (for MJX / manual state authors).
        - `live` / `auto`: same RPC as `direct` -- UE's autonomous
          physics is what advances the sim; the request just stamps the
          requested ctrl and reads back the current state.

        Cameras are served from the async SHM/ZMQ streams in EVERY mode now
        (not bundled into the step RPC reply -- that bloated the reply and
        stalled high-rate puppet stepping). ``include_cameras`` selects which
        cameras to attach to the reply: ``True`` for all, or a mapping/iterable
        of camera names. ``camera_query`` picks the freshness policy:

        - ``"latest"`` (default): attach whatever frame is currently cached.
          Never blocks; may be a frame or two behind the just-stepped state.
        - ``"fresh"``: wait (up to ``camera_timeout_s``) for a streamed frame
          whose ``frame_id`` is >= this step's post-state ``frame_id`` before
          attaching, guaranteeing the frame was rendered from this step's state
          (or newer). If the wait times out, the latest frame is attached and
          ``reply["cameras_stale"]`` is set True.

        Either way the frames also land on ``art.cameras[name].latest_frame``
        via the background stream; the reply's ``cameras`` block is a snapshot.

        Returns the raw step reply, useful when you need fields like
        ``sim_time`` or ``step`` directly; for state, prefer
        ``client.data`` and articulation accessors (``art.qpos_array``,
        ``art.get_sensors()``, etc.).
        """
        obs_str = wire(coerce(ObservationLevel, observations))

        # Optional real-time pacing: hold the loop to `target_hz` by sleeping
        # off any time remaining since the previous step() -- absorbs the manual
        # `sleep(dt - elapsed)` pattern from policy/demo loops.
        if target_hz and target_hz > 0 and self._last_step_monotonic is not None:
            slack = (1.0 / target_hz) - (time.monotonic() - self._last_step_monotonic)
            if slack > 0:
                time.sleep(slack)

        if self.step_mode == StepMode.PUPPET:
            reply = self._step_puppet(n_steps, observations=obs_str)
        else:
            # Live and Direct both use the RPC step path. UE's step
            # server applies ctrl + returns a state snapshot in either mode;
            # the difference is whether mj_step actually runs (Direct) or the
            # request just stamps NetworkValue and reads current state with
            # UE's autonomous physics continuing to advance (Live).
            reply = self._step_direct(n_steps, observations=obs_str)

        fid = reply.get("frame_id")
        if fid is not None:
            self._last_step_frame_id = int(fid)

        if include_cameras:
            self._attach_streamed_cameras(
                reply, include_cameras, camera_query, camera_timeout_s
            )
        self._last_step_monotonic = time.monotonic()
        return reply

    # -- camera access (decoupled getter API) -----------------------------

    def camera_names(self) -> "List[str]":
        """Canonical names of every discovered camera (per-articulation +
        global). These are the exact keys ``get_camera`` expects."""
        names: List[str] = []
        for art in self.articulations.values():
            names.extend(art.cameras.keys())
        names.extend(self.global_cameras.keys())
        return names

    def _find_camera_view(self, name: str) -> "URLabCameraView":
        for art in self.articulations.values():
            view = art.cameras.get(name)
            if view is not None:
                return view
        view = self.global_cameras.get(name)
        if view is not None:
            return view
        raise KeyError(
            f"camera {name!r} not found. Available cameras: {self.camera_names()}"
        )

    def warmup_cameras(
        self,
        names: "Optional[Sequence[str]]" = None,
        *,
        timeout_s: float = 10.0,
        require_all: bool = True,
    ) -> "List[str]":
        """Block until every camera (or the named subset) is streaming.

        Ensures the SHM/ZMQ streams are running (idempotent), then waits for
        each camera to deliver its first frame. Call this once after
        ``set_mode`` / scene setup so subsequent ``get_camera`` calls return
        pixels immediately instead of ``None`` during stream warm-up.

        Returns the list of cameras that became ready. With ``require_all``
        (default) a timeout raises :class:`URLabTimeoutError` naming the cameras
        that never produced a frame -- a loud, debuggable signal instead of a
        silent empty image. (``URLabTimeoutError`` is also a ``TimeoutError``.)
        """
        self._start_streaming_subs()  # idempotent: (re)enable + subscribe
        target = list(names) if names is not None else self.camera_names()
        views = {n: self._find_camera_view(n) for n in target}

        def _poll() -> "Optional[List[str]]":
            ready = [n for n, v in views.items() if v.latest_frame is not None]
            return ready if len(ready) == len(target) else None

        try:
            return self.await_ready(
                _poll, timeout_s=timeout_s,
                description=f"camera warm-up ({len(target)} cameras)",
                poll_interval_s=0.02,
            )
        except URLabTimeoutError:
            ready = [n for n, v in views.items() if v.latest_frame is not None]
            if require_all:
                missing = [n for n in target if n not in ready]
                raise URLabTimeoutError(
                    f"camera warm-up: no frames from {missing} "
                    f"(ready: {ready}). Is PIE running and the scene lit?",
                    waited_s=timeout_s, server_alive=self.server_alive(),
                )
            return ready

    def get_camera(
        self,
        name: str,
        *,
        fresh: bool = False,
        timeout_s: float = 2.0,
    ) -> "Optional[np.ndarray]":
        """Return the latest streamed frame for camera ``name`` (canonical).

        Cameras stream asynchronously in every step mode, so this is fully
        decoupled from ``step()`` -- call it whenever you want the current
        image. ``fresh=True`` waits for a frame whose ``frame_id`` is >= the
        most recent ``step()``'s post-state id, guaranteeing the frame shows
        that step's state (or newer).

        Blocks up to ``timeout_s`` for a frame to be available (covers stream
        warm-up); returns the frame as an ``np.ndarray`` (HxWx4 RGBA for
        real/seg, HxW float32 for depth), or ``None`` if none arrived in time.
        Raises ``KeyError`` (listing available names) if ``name`` is unknown.
        """
        view = self._find_camera_view(name)
        deadline = time.monotonic() + max(0.0, timeout_s)
        target = self._last_step_frame_id if fresh else None
        while True:
            frame = view.latest_frame
            have = frame is not None
            fresh_ok = (
                target is None
                or (view.frame_id is not None and view.frame_id >= target)
            )
            if have and fresh_ok:
                return frame
            if time.monotonic() >= deadline:
                return frame  # may be None (never arrived) or stale (fresh timed out)
            time.sleep(0.002)

    def _step_direct(
        self, n_steps: int, *, observations: str,
    ) -> Dict[str, Any]:
        per_art: Dict[str, Any] = {}
        for prefix, art in self.articulations.items():
            per_art[prefix] = art._build_step_request(control_mode=None)

        # Cameras are NOT requested inline: they stream over SHM/ZMQ and are
        # merged into the reply by _attach_streamed_cameras after the step.
        # The reply's `frame_id` is what a "fresh" query synchronises against.
        request: Dict[str, Any] = {
            "n_steps": int(n_steps),
            "observations": observations,
            "per_articulation": per_art,
        }
        reply = self._rpc("step", request, expected_op="step_ok")
        self._absorb_step_reply(reply)
        # Clear xfrc post-step per MuJoCo semantics
        for art in self.articulations.values():
            art.clear_xfrc()
        self._pending_entity_xfrc.clear()
        return reply

    def _step_puppet(
        self, n_steps: int, *, observations: str,
    ) -> Dict[str, Any]:
        if mujoco is None:
            raise RuntimeError("mujoco not installed; puppet mode requires it")
        if self.model is None or self.data is None:
            raise RuntimeError(
                "puppet mode requires a local model (got local_model=False or "
                "no MJB in handshake)"
            )
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")

        # n_steps == 0: skip mj_step entirely (MJX / manual state authors
        # push whatever they already wrote into client.data).
        for _ in range(int(n_steps)):
            mujoco.mj_step(self.model, self.data)

        # Cameras stream over SHM/ZMQ (see _step_direct); not requested inline.
        request: Dict[str, Any] = {
            "mode": wire(StepMode.PUPPET),
            "n_steps": int(n_steps),
            "observations": observations,
            "time": float(self.data.time),
            "qpos": np.asarray(self.data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(self.data.qvel, dtype=np.float64).tolist(),
            "ctrl": np.asarray(self.data.ctrl, dtype=np.float64).tolist(),
            "per_articulation": {},
        }
        reply = self._rpc("step", request, expected_op="step_ok")
        self._absorb_step_reply(reply)
        return reply

    # -- streaming-mode SUB infrastructure --------------------------------

    def _start_streaming_subs(self) -> None:
        """Spin up the state-snapshot stream + one camera stream per
        registered camera. Idempotent. Called from discover() when
        live is active and from set_mode() on transitions back
        to live.
        """
        self._transport.start_state_stream(self._on_state_snapshot)
        # UE's bEnableAllCameras defaults off: a camera only runs its pub
        # streams while broadcast-enabled or requested. Enable broadcast on
        # every discovered camera and -- crucially -- read the ACTUAL per-camera
        # endpoints back from the reply. The handshake advertises a shared
        # default endpoint before streaming is on; each camera only binds its
        # real (distinct) ZMQ port once enabled, and set_camera_streaming
        # reports it. Subscribing to the stale handshake endpoint sends every
        # camera to one port, so all but one get no frames.
        enable: Dict[str, Any] = {}
        for art in self.articulations.values():
            for cam_name, view in art.cameras.items():
                # Only cameras the handshake advertised an endpoint/topic for are
                # streamable. Skipping the rest also keeps this a no-op (no RPC)
                # for camera-less / stub-transport scenes.
                if getattr(view, "_zmq_topic", None) and getattr(view, "_zmq_endpoint", None):
                    enable[cam_name] = {"zmq": True, "shm": True}
        reply: Dict[str, Any] = {}
        if enable:
            try:
                reply = self.runtime.set_camera_streaming(enable)
            except Exception as exc:  # pragma: no cover - older server / transport
                logger.debug("set_camera_streaming at stream startup failed: %s", exc)
        # One per-camera stream; the transport dedupes on (prefix, name).
        # Prefer the endpoint/topic from the set_camera_streaming reply (the
        # bound port); fall back to the handshake values for older servers.
        for art in self.articulations.values():
            for cam_name, view in art.cameras.items():
                info = reply.get(cam_name) or {}
                if info.get("zmq_endpoint"):
                    view._zmq_endpoint = info["zmq_endpoint"]
                if info.get("zmq_topic"):
                    view._zmq_topic = info["zmq_topic"]
                topic = getattr(view, "_zmq_topic", None)
                endpoint = getattr(view, "_zmq_endpoint", None)
                if not topic or not endpoint:
                    continue
                self._transport.start_camera_stream(
                    art.prefix, cam_name, endpoint, topic,
                    self._make_camera_callback(art.prefix, cam_name),
                )

    def _stop_streaming_subs(self) -> None:
        """Tear down all streaming subs. Idempotent."""
        self._transport.stop_state_stream()
        self._transport.stop_camera_streams()

    def _on_state_snapshot(self, snap: Mapping[str, Any]) -> None:
        """Transport-thread callback: store the latest snapshot and bump
        the counter that streaming-mode step waits on."""
        with self._state_lock:
            self._latest_state_snapshot = dict(snap)
            self._state_msg_count += 1
            self._last_snapshot_monotonic = time.monotonic()
            self._state_cond.notify_all()

    # -- readiness / await layer ------------------------------------------

    def server_alive(self, within_s: float = 2.0) -> bool:
        """True if a state-stream snapshot arrived within ``within_s`` seconds.

        The state stream only flows while PIE is stepping, so this is the
        liveness oracle for awaiting: during a step loop it distinguishes
        "server busy" from "server hung". Returns False if no stream is up or
        no snapshot has arrived yet (e.g. editor-time ops before PIE)."""
        ts = self._last_snapshot_monotonic
        return ts is not None and (time.monotonic() - ts) <= within_s

    def await_ready(
        self,
        poll: "Callable[[], Any]",
        *,
        timeout_s: float,
        description: str = "operation",
        poll_interval_s: float = 0.05,
        on_progress: "Optional[Callable[[str], None]]" = None,
        require_liveness: bool = False,
    ) -> Any:
        """Block until ``poll()`` returns a non-None value, then return it.

        The single readiness primitive every wait in the client builds on
        (camera warm-up, PIE start, ...). ``poll`` is called every
        ``poll_interval_s`` and should return ``None`` while pending or the
        result once ready.

        On timeout raises :class:`URLabTimeoutError`, whose ``server_alive``
        field reports whether the state stream looked fresh -- so the error
        says "alive but slow" vs "silent/hung". With ``require_liveness`` the
        ``timeout_s`` deadline is soft *while the server is alive*: the wait is
        extended in short grace windows (capped at 10x ``timeout_s``), so a
        genuinely-working-but-slow server is waited out instead of failed.
        ``on_progress`` (if given) is called ~once/second with an elapsed note.
        """
        start = time.monotonic()
        deadline = start + max(0.0, timeout_s)
        hard_deadline = start + max(0.0, timeout_s) * 10.0
        last_beat = start
        while True:
            result = poll()
            if result is not None:
                return result
            now = time.monotonic()
            if now >= deadline:
                alive = self.server_alive()
                if require_liveness and alive and now < hard_deadline:
                    deadline = now + min(max(timeout_s, 1.0), 5.0)
                else:
                    raise URLabTimeoutError(
                        description, waited_s=now - start, server_alive=alive
                    )
            if on_progress is not None and (now - last_beat) >= 1.0:
                last_beat = now
                try:
                    on_progress(f"{description}: {now - start:.0f}s elapsed")
                except Exception:  # pragma: no cover - progress is best-effort
                    pass
            time.sleep(poll_interval_s)

    # -- bootstrap / lifecycle --------------------------------------------

    def connect(self, observations: Union[str, ObservationLevel] = "standard") -> None:
        """Open the session: handshake, load the model, build articulations.
        Call once after construction. Same wire op as the legacy ``discover``;
        ``connect`` is the canonical name."""
        self.discover(observations=observations)

    def refresh(self, observations: Union[str, ObservationLevel] = "standard") -> None:
        """Re-run discovery to pick up scene changes (after spawn/import or an
        external edit). Idempotent."""
        self.discover(observations=observations)

    def articulation(self, prefix: Optional[str] = None) -> "URLabArticulation":
        """Return one articulation by ``prefix``, or the sole articulation when
        ``prefix`` is None. Raises ``KeyError`` (listing available names) if the
        prefix is unknown or the choice is ambiguous -- no more single-vs-multi
        disambiguation boilerplate at the call site."""
        if prefix is not None:
            try:
                return self.articulations[prefix]
            except KeyError:
                raise KeyError(
                    f"no articulation {prefix!r}; available: {list(self.articulations)}"
                ) from None
        n = len(self.articulations)
        if n == 1:
            return next(iter(self.articulations.values()))
        raise KeyError(
            f"{n} articulations present; pass prefix=. "
            f"available: {list(self.articulations)}"
        )

    def bringup(
        self,
        *,
        mode: "Optional[Union[str, StepMode]]" = None,
        cameras: bool = False,
        sim_dt: Optional[float] = None,
        start_pie: bool = False,
        camera_timeout_s: float = 10.0,
        observations: Union[str, ObservationLevel] = "standard",
    ) -> Readiness:
        """One call that leaves the session fully ready to drive.

        Order: optional ``sim.start`` -> ``refresh`` (discover) ->
        ``set_sim_options(timestep=sim_dt)`` (best-effort) -> ``set_mode(mode)``
        -> ``warmup_cameras``. Returns a :class:`Readiness` summary. Raises on
        the first stage that fails (e.g. :class:`URLabTimeoutError` if cameras
        never stream), with details attached. Replaces the hand-ordered
        connect/set_mode/refresh/warmup recipe."""
        if start_pie and not self.manager_present:
            self.sim.start()
        self.refresh(observations=observations)
        sim_dt_applied: Optional[float] = None
        if sim_dt is not None:
            applied = self.runtime.set_sim_options(timestep=float(sim_dt), required=False)
            sim_dt_applied = getattr(applied, "timestep", None) if applied else None
        if mode is not None:
            self.runtime.set_mode(mode)
        n_cams = len(self.camera_names())
        cams_ready = 0
        if cameras and n_cams:
            cams_ready = len(self.warmup_cameras(timeout_s=camera_timeout_s))
        return Readiness(
            mode=self.step_mode,
            n_articulations=len(self.articulations),
            n_cameras=n_cams,
            cameras_ready=cams_ready,
            sim_dt_applied=sim_dt_applied,
        )

    def _make_camera_callback(self, prefix: str, cam_name: str) -> "Callable[[bytes], None]":
        """Build a frame-bytes callback bound to a specific (prefix, cam)
        URLabCameraView. The closure captures only string keys and resolves
        the live view on each call so a re-attached camera still updates."""

        def _on_frame(
            pixels: bytes,
            frame_id: "Optional[int]" = None,
            sim_time: "Optional[float]" = None,
        ) -> None:
            art = self.articulations.get(prefix)
            if not art:
                return
            view = art.cameras.get(cam_name)
            if view is None:
                return
            try:
                w, h = view.resolution
                decoded = False
                if view.mode == CameraMode.DEPTH:
                    # Single-channel PF_R32_FLOAT; one float per pixel.
                    arr = np.frombuffer(pixels, dtype=np.float32)
                    if arr.size == w * h:
                        view.latest_frame = arr.reshape((h, w))
                        decoded = True
                else:
                    # REAL / SEMANTIC / INSTANCE all ship BGRA8. Real
                    # rotates to RGBA for consumer-friendliness; seg modes
                    # keep BGRA -- the seg material's tint convention is
                    # documented per-channel and consumers mapping color
                    # to class id need the original byte order.
                    arr = np.frombuffer(pixels, dtype=np.uint8)
                    if arr.size == w * h * 4:
                        bgra = arr.reshape((h, w, 4))
                        if view.mode == CameraMode.REAL:
                            view.latest_frame = bgra[..., [2, 1, 0, 3]]
                        else:
                            view.latest_frame = bgra
                        decoded = True
                if decoded:
                    view.frame_count += 1
                    # frame_id / sim_time ride the stream header now (both
                    # transports), so a "fresh" query can wait on the cache
                    # until view.frame_id >= the step's post-state id. Set
                    # them together with the pixels so they never diverge.
                    if frame_id is not None:
                        view.frame_id = frame_id
                    if sim_time is not None:
                        view.sim_time = sim_time
            except Exception as exc:
                logger.debug("camera decode failed (%s/%s): %s",
                             prefix, cam_name, exc)

        return _on_frame

    @staticmethod
    def _select_camera_names(include_cameras: Any) -> "Optional[set]":
        """Normalise the ``include_cameras`` argument to a set of camera names,
        or ``None`` meaning "all cameras". Accepts:

        - ``True``                 -> None (all)
        - ``"cam1"`` (a str)       -> {"cam1"} (single camera)
        - mapping ``{"cam1": ...}`` -> its keys (values are ignored; the
          freshness policy is the ``camera_query`` arg, not a per-camera value)
        - list / tuple / set       -> that set of names

        Anything else (e.g. ``False``/``None``) yields an empty set -> nothing.
        """
        if include_cameras is True:
            return None
        if isinstance(include_cameras, str):
            return {include_cameras}
        if isinstance(include_cameras, Mapping):
            return set(include_cameras.keys())
        if isinstance(include_cameras, (list, tuple, set, frozenset)):
            return set(include_cameras)
        return set()

    def _gather_cached_cameras(self, include_cameras: Any) -> Dict[str, Dict[str, Any]]:
        """Read the latest cached frame off each requested URLabCameraView
        and bundle into a {prefix: {cam_name: {pixels, mode, ...}}} dict.

        These frames arrive over the async SHM/ZMQ stream (in every step mode
        now), not the step RPC reply. ``frame_id`` is the post-step state the
        frame shows -- compare it against the step reply's ``frame_id`` to know
        how fresh the frame is."""
        wanted = self._select_camera_names(include_cameras)
        out: Dict[str, Dict[str, Any]] = {}
        for prefix, art in self.articulations.items():
            per_art: Dict[str, Any] = {}
            for cam_name, view in art.cameras.items():
                if wanted is not None and cam_name not in wanted:
                    continue
                if view.latest_frame is None:
                    continue
                mode = view.mode.value if hasattr(view.mode, "value") else str(view.mode)
                per_art[cam_name] = {
                    "pixels": view.latest_frame,
                    "mode": mode,
                    "resolution": list(view.resolution),
                    "frame_count": view.frame_count,
                    "frame_id": view.frame_id,
                    "sim_time": view.sim_time,
                }
            if per_art:
                out[prefix] = per_art
        return out

    def _wait_for_camera_frames(
        self, include_cameras: Any, target_frame_id: int, timeout_s: float
    ) -> bool:
        """Block until every requested camera has streamed a frame whose
        ``frame_id >= target_frame_id`` (the step's post-state id), or until
        ``timeout_s`` elapses. Returns True if all reached the target.

        This is the "fresh" guarantee: the monotonic frame_id is stamped when
        the stepped state is pushed to the render snapshot, so a streamed frame
        tagged >= it was rendered from a state at or after this step."""
        wanted: Optional[set] = None
        if isinstance(include_cameras, Mapping):
            wanted = set(include_cameras.keys())

        def _all_fresh() -> bool:
            for art in self.articulations.values():
                for cam_name, view in art.cameras.items():
                    if wanted is not None and cam_name not in wanted:
                        continue
                    fid = view.frame_id
                    if fid is None or fid < target_frame_id:
                        return False
            return True

        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if _all_fresh():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    def _attach_streamed_cameras(
        self,
        reply: Dict[str, Any],
        include_cameras: Any,
        camera_query: str,
        timeout_s: float,
    ) -> None:
        """Merge async-streamed camera frames into a step reply. ``latest``
        takes whatever is cached; ``fresh`` first waits for a frame matching
        this step's ``frame_id`` and sets ``reply["cameras_stale"]`` if the
        wait timed out."""
        if camera_query not in ("latest", "fresh"):
            raise ValueError(
                f"camera_query must be 'latest' or 'fresh', got {camera_query!r}"
            )
        if camera_query == "fresh":
            target = reply.get("frame_id")
            stale = False
            if target is not None:
                stale = not self._wait_for_camera_frames(
                    include_cameras, int(target), timeout_s
                )
            reply["cameras_stale"] = stale
        cams = self._gather_cached_cameras(include_cameras)
        if cams:
            reply["cameras"] = cams

    def reset(
        self,
        keyframe_name: Optional[str] = None,
        seed: Optional[int] = None,
        per_articulation_qpos: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Reset the sim. Returns the raw reset reply, useful for fields
        like ``sim_time``; for state, prefer ``client.data`` and
        articulation accessors."""
        request: Dict[str, Any] = {}
        if keyframe_name is not None:
            request["keyframe_name"] = keyframe_name
        if seed is not None:
            request["seed"] = int(seed)
        if per_articulation_qpos is not None:
            request["per_articulation_qpos"] = {
                prefix: dict(m) for prefix, m in per_articulation_qpos.items()
            }
        # UE returns `reset_ok` as the op name (different from `step_ok`)
        # but the payload shape mirrors step_ok, so `_absorb_step_reply`
        # handles it. Accept either op name to be robust against an older
        # or future UE that conflates the two.
        reply = self._rpc("reset", request, expected_op=None)
        reply_op = reply.get("op")
        if reply_op not in ("reset_ok", "step_ok"):
            raise URLabRPCError(
                "unexpected_reply_op",
                f"reset: wanted 'reset_ok' or 'step_ok', got {reply_op!r}",
                op="reset",
            )
        self._absorb_step_reply(reply)
        return reply

    def forward(self) -> Dict[str, Any]:
        """Run ``mj_forward`` on the server (kinematics + dynamics, no
        integration) and return observations.

        Use this after writing ``qpos`` / ``qvel`` via
        :meth:`runtime.set_qpos` to read consistent derived state
        (``xpos``, sensors, contacts) without advancing simulation time.
        The reply shape matches a normal step reply, so articulation
        accessors (``art.qpos_array``, ``art.root_pos_w``, etc.) refresh
        as usual.
        """
        reply = self._rpc("forward", {}, expected_op="forward_ok")
        self._absorb_step_reply(reply)
        return reply


    def _mirror_set_qpos_locally(self, reply: Mapping[str, Any]) -> None:
        if self.model is None or self.data is None:
            return
        echoed = reply.get("qpos")
        if not isinstance(echoed, (list, tuple)) or len(echoed) == 0:
            return
        # Identify the articulation. Reply carries actor_id + actor_name;
        # also fall back to `target` if the server emits prefix directly.
        art = None
        aid = reply.get("actor_id")
        if isinstance(aid, str) and aid in self.articulations_by_id:
            art = self.articulations_by_id[aid]
        if art is None:
            tgt = reply.get("target")
            if isinstance(tgt, str):
                art = self.articulations.get(tgt) or self.articulations_by_id.get(tgt)
        if art is None or not getattr(art, "joints", None):
            return

        try:
            import mujoco  # noqa: F401  -- ensures self.data lib is loaded
        except ImportError:
            return

        with self._data_lock:
            qpos_arr = self.data.qpos
            if reply.get("free_base_shortcut"):
                # 7-vec write to the articulation's free joint (xyz + xyzw).
                if len(echoed) < 7:
                    return
                free_jnt = next(
                    (j for j in art.joints.values() if j.jnt_type == 0),
                    None,
                )
                if free_jnt is None:
                    return
                start = int(free_jnt.qpos_offset)
                if start + 7 > qpos_arr.size:
                    return
                for i in range(7):
                    qpos_arr[start + i] = float(echoed[i])
            else:
                # Full per-articulation qpos: walk joints in registration
                # order, write each joint's slot from the contiguous echo.
                offset = 0
                for joint in art.joints.values():
                    width = int(joint.qpos_dim)
                    if width == 0:
                        continue
                    if offset + width > len(echoed):
                        break
                    start = int(joint.qpos_offset)
                    if start + width > qpos_arr.size:
                        offset += width
                        continue
                    for i in range(width):
                        qpos_arr[start + i] = float(echoed[offset + i])
                    offset += width
            try:
                import mujoco
                mujoco.mj_forward(self.model, self.data)
            except Exception as exc:
                logger.debug("mj_forward after qpos mirror failed: %s", exc)

    def close(self) -> None:
        # Idempotent: safe to call multiple times (context-manager exit + an
        # explicit close, double-close in error paths, etc.) and never raises.
        if self._closed:
            return
        self._closed = True
        # Revert URLab to live before tearing the transport down. If
        # the client used auto-promote to enter direct/puppet, the server
        # stays in that mode forever once we disconnect (publishers stay
        # paused, editor users see the sim "stuck"). Best-effort -- swallow
        # any error so close() never raises during teardown. Symmetric with
        # the auto_promote_step_mode flag: if the constructor opted out of
        # auto-promote, also opt out of auto-revert.
        if (
            self._auto_promote_step_mode
            and self.session_id is not None
            and self.manager_present
            and self.step_mode in (StepMode.DIRECT, StepMode.PUPPET)
        ):
            try:
                self.runtime.set_mode(StepMode.LIVE)
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug(
                    "URLabClient.close: revert to live failed: %s", exc
                )
        # Transport closes streaming subs and the RPC channel. Swallow so
        # close() never raises during teardown / atexit.
        try:
            self._transport.close()
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.debug("URLabClient.close: transport close failed: %s", exc)

    def __enter__(self) -> "URLabClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- reply absorption -------------------------------------------------

    def _absorb_step_reply(self, reply: Mapping[str, Any]) -> None:
        # Guard self.data + per-articulation buffer writes against any
        # concurrent reader thread. Reentrant lock —
        # `_mirror_state_into_data` re-acquires below.
        with self._data_lock:
            self._absorb_step_reply_locked(reply)

    def _absorb_step_reply_locked(self, reply: Mapping[str, Any]) -> None:
        _Art = URLabArticulation

        if "time" in reply:
            self.sim_time = float(reply["time"])
        if "step" in reply:
            self.step_count = int(reply["step"])
        # ROS-Time-aligned clocks (sec, nsec). Absent on replies from
        # pre-clock-fields servers; fields stay at their previous value.
        sim_t = reply.get("sim_time")
        if isinstance(sim_t, Mapping):
            self.sim_time_sec = int(sim_t.get("sec", self.sim_time_sec))
            self.sim_time_nsec = int(sim_t.get("nsec", self.sim_time_nsec))
        wall_t = reply.get("wall_time")
        if isinstance(wall_t, Mapping):
            self.wall_time_sec = int(wall_t.get("sec", self.wall_time_sec))
            self.wall_time_nsec = int(wall_t.get("nsec", self.wall_time_nsec))
        self.recv_wall_time_ns = time.time_ns()

        per_art = reply.get("per_articulation") or {}
        for prefix, block in per_art.items():
            art = self.articulations.get(prefix)
            if art is not None:
                art._apply_step_reply(block)

        # Mirror state into local MjData for non-puppet replies. In puppet
        # mode, client.data is already authoritative (it drove the step);
        # UE's reply just echoes what we pushed.
        if (
            self.step_mode != StepMode.PUPPET
            and self.model is not None
            and self.data is not None
        ):
            self._mirror_state_into_data(reply)

        # Cameras block: include_cameras=True / {name: mode} on the step
        # request makes the server return a `cameras` object keyed by
        # camera name. Decode each frame into the matching
        # URLabCameraView so `art.cameras[name].latest_frame` reflects
        # the freshest pull-mode capture. Streaming-mode captures go
        # through the SUB stream + _make_camera_callback instead; this
        # path is only for direct / puppet include_cameras=True.
        cams_block = reply.get("cameras") or {}
        if cams_block:
            for cam_name, cam_payload in cams_block.items():
                if not isinstance(cam_payload, Mapping):
                    continue
                pixels_obj = cam_payload.get("data")
                if pixels_obj is None:
                    continue
                # msgpack bin frames arrive as Python bytes; JSON
                # fallback would send base64 strings — handle both.
                if isinstance(pixels_obj, str):
                    import base64 as _b64
                    pixels = _b64.b64decode(pixels_obj)
                elif isinstance(pixels_obj, (bytes, bytearray, memoryview)):
                    pixels = bytes(pixels_obj)
                else:
                    continue
                # Find the matching URLabCameraView. Lookup by the
                # bare camera name across every articulation; in
                # practice each scene's cameras are name-unique because
                # the server's ByName map collapses on bare name too.
                for art in self.articulations.values():
                    view = art.cameras.get(cam_name)
                    if view is None:
                        continue
                    try:
                        w, h = view.resolution
                        if view.mode == CameraMode.DEPTH:
                            arr = np.frombuffer(pixels, dtype=np.float32)
                            if arr.size == w * h:
                                view.latest_frame = arr.reshape((h, w))
                                view.frame_count += 1
                                view.sim_time = (
                                    float(cam_payload["sim_time"])
                                    if isinstance(cam_payload.get("sim_time"), (int, float))
                                    else self.sim_time
                                )
                                _fid = cam_payload.get("frame_id")
                                if _fid is not None:
                                    view.frame_id = int(_fid)
                        else:
                            arr = np.frombuffer(pixels, dtype=np.uint8)
                            if arr.size == w * h * 4:
                                bgra = arr.reshape((h, w, 4))
                                if view.mode == CameraMode.REAL:
                                    view.latest_frame = bgra[..., [2, 1, 0, 3]]
                                else:
                                    view.latest_frame = bgra
                                view.frame_count += 1
                                view.sim_time = (
                                    float(cam_payload["sim_time"])
                                    if isinstance(cam_payload.get("sim_time"), (int, float))
                                    else self.sim_time
                                )
                                _fid = cam_payload.get("frame_id")
                                if _fid is not None:
                                    view.frame_id = int(_fid)
                    except Exception as exc:
                        logger.debug(
                            "include_cameras decode failed (%s/%s): %s",
                            art.prefix, cam_name, exc,
                        )
                    break

        # Non-articulation entities. Write the reply's xpos/xquat into the
        # local MjData at the body's slot so `entity.root_pos_w` /
        # `root_quat_w` reads consistent values regardless of whether the
        # body has a free joint being driven by qpos. Order matters: this
        # runs AFTER `_mirror_state_into_data` (which calls mj_forward over
        # qpos), so we override mj_forward's per-body xpos for
        # non-articulation entities with the wire-shipped value.
        entity_block = reply.get("entities") or {}
        for name, block in entity_block.items():
            entity = self.entities.get(name)
            if entity is None or self.data is None or isinstance(entity, _Art):
                continue
            if entity.body_id < 0:
                continue
            if "xpos" in block:
                self.data.xpos[entity.body_id] = np.asarray(
                    block["xpos"], dtype=np.float64
                )
            if "xquat" in block:
                self.data.xquat[entity.body_id] = np.asarray(
                    block["xquat"], dtype=np.float64
                )

    def _mirror_state_into_data(self, reply: Mapping[str, Any]) -> None:
        """Write the reply's qpos / qvel back into the local MjData so
        MPC / IK / observation derivation sees the UE-authoritative state.

        Threading contract: caller MUST hold `self._data_lock`. The
        `mj_forward` at the end is the main reason — it mutates many
        derived fields (xpos, xquat, sensors) inside `data` non-atomically.
        """
        per_art = reply.get("per_articulation") or {}
        for prefix, block in per_art.items():
            art = self.articulations.get(prefix)
            if art is None:
                continue
            qpos = block.get("qpos")
            qvel = block.get("qvel")
            if qpos is not None and self.data is not None:
                qpos_arr = np.asarray(qpos, dtype=np.float64)
                # Write per-joint into the global qpos buffer at each
                # joint's qpos_offset / qpos_dim. This tolerates gaps /
                # non-contiguous articulations.
                src_idx = 0
                for j in art.joints.values():
                    n = j.qpos_dim
                    if src_idx + n > qpos_arr.size:
                        break
                    self.data.qpos[j.qpos_offset : j.qpos_offset + n] = qpos_arr[
                        src_idx : src_idx + n
                    ]
                    src_idx += n
            if qvel is not None and self.data is not None:
                qvel_arr = np.asarray(qvel, dtype=np.float64)
                src_idx = 0
                for j in art.joints.values():
                    n = j.qvel_dim
                    if src_idx + n > qvel_arr.size:
                        break
                    self.data.qvel[j.qvel_offset : j.qvel_offset + n] = qvel_arr[
                        src_idx : src_idx + n
                    ]
                    src_idx += n
        if "time" in reply and self.data is not None:
            self.data.time = float(reply["time"])
        if mujoco is not None and self.model is not None and self.data is not None:
            mujoco.mj_forward(self.model, self.data)


def _load_mjb(buf: bytes) -> Tuple[Any, Any]:
    """Load an MJB buffer via the filesystem route.

    `mujoco.MjModel.from_binary_path` is the stable public API; there's
    also a `from_binary` in some versions but the path form is available
    everywhere. Write to a tempfile, load, delete.
    """
    if mujoco is None:  # pragma: no cover
        raise RuntimeError("mujoco not installed")
    with tempfile.NamedTemporaryFile(
        suffix=".mjb", delete=False
    ) as f:
        f.write(buf)
        path = f.name
    try:
        model = mujoco.MjModel.from_binary_path(path)
    finally:
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover
            pass
    data = mujoco.MjData(model)
    return model, data
