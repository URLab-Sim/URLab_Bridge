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
import socket
import tempfile
import threading
import time
import warnings
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
from .errors import (
    URLabPuppetDriftError,
    URLabRPCError,
    URLabTimeoutError,
)
from ._model_upload import (
    MAX_ASSETS,
    flatten_model,
    iter_chunks,
    require_bare_filename,
    sha256_hex,
)
from .results import StepResult
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
# doesn't pass an explicit ``recv_timeout_ms``. Long editor / PIE / handshake ops
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
    # Allocates a render target and binds a port per camera.
    "set_camera_streaming": 30.0,
    # Walk the whole world on the game thread; a populated level outgrows 5s.
    "list_actors": 30.0,
    "find_actors": 30.0,
    "get_actor_bounds": 30.0,
    "actor_hierarchy": 30.0,
    "snapshot": 30.0,
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
        mujoco_version_check: bool = True,
        local_model: bool = True,
        recv_timeout_ms: int = 5000,
        auto_promote_step_mode: bool = True,
        transport: Union[str, Transport] = "auto",
        shm_dir: Optional[str] = None,
        puppet_drift_check: str = "warn",
        broadcast_viewers: bool = False,
        viewer_port: int = 5560,
    ):
        self.address = address
        self.step_mode: StepMode = coerce(StepMode, step_mode, default=StepMode.AUTO)
        self.step_port = step_port
        self.state_port = state_port
        # When True, the owner (this client) re-broadcasts the raw {t,qpos,qvel}
        # it pushes each step onto a PUB so any number of read-only viewers can
        # subscribe. Bound lazily on the first broadcast step. Viewer-side is a
        # plain SUB + msgpack -- no client library required.
        self.broadcast_viewers = broadcast_viewers
        self.viewer_port = viewer_port
        self._viewer_bcast_bound = False
        self.mujoco_version_check = mujoco_version_check
        self.local_model = local_model
        if puppet_drift_check not in ("error", "warn", "off"):
            raise ValueError(
                "puppet_drift_check must be 'error', 'warn' or 'off', "
                f"got {puppet_drift_check!r}"
            )
        self.puppet_drift_check = puppet_drift_check
        self._recv_timeout_ms = recv_timeout_ms
        self._auto_promote_step_mode = auto_promote_step_mode

        self.session_id: Optional[str] = None
        self.urlab_version: Optional[str] = None
        self.mujoco_version: Optional[str] = None
        # False until PIE starts; editor-only ops still work pre-PIE.
        self.manager_present: bool = False
        # Render-farm identity advertised in the handshake `instance` block
        # (instance_id / index / host / ports / capabilities). Empty when the
        # server is not farm-aware. `instance.host` drives the transport=auto
        # locality decision in connect().
        self.instance: Dict[str, Any] = {}
        # Cooperative farm lease id, set by URLabPool.lease when this client
        # claimed its instance; released on close(). None for un-leased clients.
        self.lease_id: Optional[str] = None
        # Set by URLabPool.lease to the InstanceInfo this client leased.
        self._leased_instance: Any = None
        self.shm_session_dir: str = ""
        # Explicit SHM RPC contract from the handshake (session/paths/event
        # names for req.shm/rep.shm). The RPC region lives on its own static
        # session, distinct from the per-PIE camera/state stream dir; the
        # transport must use these verbatim or every RPC stalls its full timeout.
        self._shm_rpc_contract: Optional[Dict[str, Any]] = None
        self.model: Any = None
        self.data: Any = None
        # Set by _apply_handshake_locked when the MJB couldn't load and the
        # reply carried no compiled XML; the outer _apply_handshake then
        # refetches the handshake once with include_assets=true.
        self._model_fallback_pending: bool = False
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

        # Observation level requested at connect(); re-sent on any internal
        # handshake refetch so a model-fallback round-trip does not silently
        # revert the server to its default observation level.
        self._observation_level: str = "standard"

        # Per-camera server-side delay (seconds) last applied via
        # runtime.set_camera_delay, keyed by canonical camera name. Under a
        # nonzero delay a "fresh" frame for the just-stepped state can never
        # reveal within the step, so the fresh-wait paths degrade such cameras
        # to "latest" instead of burning the whole camera timeout.
        self._camera_applied_delay: Dict[str, float] = {}

        # transport="shm" defers actual SHM construction to connect()
        # so we can pull the session dir out of the handshake; until then
        # we use ZMQ for the hello round-trip.
        self._shm_dir_override: Optional[str] = shm_dir
        self._want_shm: bool = False
        # Config the live SHM transport was built from (stream dir + advertised
        # RPC paths/events). Re-diffed against every handshake so a PIE restart
        # that moves the session dir rebuilds the transport instead of spinning
        # forever on the dead files.
        self._active_shm_config: Optional[Tuple[Any, ...]] = None
        # Transport preference: "auto" bootstraps over ZMQ for hello, then
        # upgrades to SHM in connect() only when the handshake reports the
        # instance is co-located (same hostname + a locally-accessible SHM
        # session dir). "zmq"/"shm" force the choice. All three bootstrap the
        # ZMQ transport first; SHM construction is deferred to connect().
        self._transport_pref: str = "explicit"
        if isinstance(transport, str):
            if transport in ("zmq", "shm", "auto"):
                self._transport: Transport = make_transport(
                    "zmq",
                    address,
                    step_port=step_port,
                    state_port=state_port,
                    recv_timeout_ms=recv_timeout_ms,
                )
                self._want_shm = (transport == "shm")
                self._transport_pref = transport
            else:
                raise ValueError(
                    f"unknown transport name {transport!r}; expected "
                    f"'auto', 'zmq' or 'shm', or pass a Transport instance"
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
        self._camera_frame_cond = threading.Condition()

    def attach_puppet_simulation(self, model: Any, data: Any) -> None:
        """Use an externally owned MuJoCo simulation as the puppet source.

        Call this before :meth:`connect`. The client retains the exact model
        and data objects; ``step(n_steps=0)`` can then transmit their current
        state without advancing physics locally.

        The binding is made before the URLab scene exists -- levels are
        loaded, MJCF imported and actors spawned afterwards -- so the two
        models can diverge between here and the first push. Every handshake
        that carries an MJB is therefore checked against this model (see
        ``puppet_drift_check``), including the fresh one ``sim.start``
        absorbs once PIE is up. A mismatch cannot be repaired after the
        fact, since a puppet step is just a raw ``qpos``/``qvel``/``ctrl``
        vector, so it raises :class:`URLabPuppetDriftError` instead of
        pushing state onto the wrong degrees of freedom.
        """
        if mujoco is None:
            raise RuntimeError("mujoco not installed; puppet mode requires it")
        if self.session_id is not None:
            raise RuntimeError(
                "attach_puppet_simulation must be called before connect"
            )
        if self.step_mode not in (StepMode.AUTO, StepMode.PUPPET):
            raise ValueError(
                "external simulation attachment requires puppet or auto mode"
            )
        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be a mujoco.MjModel")
        if not isinstance(data, mujoco.MjData):
            raise TypeError("data must be a mujoco.MjData")
        if data.model is not model:
            raise ValueError("data must have been created from the same model")

        self.model = model
        self.data = data
        self.local_model = False
        self.step_mode = StepMode.PUPPET

    # -- transport --------------------------------------------------------

    def _rpc(
        self,
        op: str,
        payload: Mapping[str, Any],
        *,
        expected_op: Optional[str] = None,
        recv_timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send one request and unpack one reply. Raises on error replies.

        ``recv_timeout_ms`` overrides the transport's default recv
        timeout for this single call. When omitted, a per-op default from
        ``_OP_TIMEOUTS_S`` is applied so long editor/PIE ops get a generous
        window automatically (no caller-guessed global timeout); ops not in
        the registry use the transport default."""
        if recv_timeout_ms is None:
            op_default_s = _OP_TIMEOUTS_S.get(op)
            if op_default_s is not None:
                recv_timeout_ms = int(op_default_s * 1000)
        request = {"op": op, "session_id": self.session_id, **payload}
        reply = self._transport.rpc(request, recv_timeout_ms=recv_timeout_ms)
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

    def _run_editor_job(
        self,
        op: str,
        payload: Mapping[str, Any],
        *,
        expected_op: str,
        on_progress: "Optional[Callable[[str], None]]" = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a (possibly async) editor op and return its final reply dict.

        The server may answer EITHER synchronously (``op == expected_op`` --
        older/non-async ops) OR asynchronously with ``op == "op_started"`` +
        ``job_id``, in which case we poll ``op_status`` via :meth:`await_ready`
        until the job is done/failed. Either way the caller gets the same final
        reply dict, so the public method signatures are unchanged."""
        reply = self._rpc(op, payload)
        if reply.get("op") != "op_started":
            # Synchronous reply (or non-async server): validate and return.
            if reply.get("op") != expected_op:
                raise URLabRPCError(
                    "unexpected_reply_op",
                    f"wanted {expected_op!r}, got {reply.get('op')!r}",
                    op=op,
                )
            return reply

        job_id = reply.get("job_id")
        if not job_id:
            raise URLabRPCError("bad_job", "op_started reply missing job_id", op=op)
        tmo = timeout_s if timeout_s is not None else _OP_TIMEOUTS_S.get(op, 30.0)

        def _poll() -> "Optional[Dict[str, Any]]":
            st = self._rpc("op_status", {"job_id": job_id})
            if st.get("state") == "running":
                prog = st.get("progress")
                if prog and on_progress is not None:
                    try:
                        on_progress(prog)
                    except Exception:  # pragma: no cover - progress best-effort
                        pass
                return None
            return st  # done | failed

        final = self.await_ready(
            _poll, timeout_s=tmo, description=op,
            poll_interval_s=0.1, on_progress=on_progress, require_liveness=True,
        )
        result = final.get("result") or {}
        if final.get("state") == "failed":
            raise URLabRPCError(
                result.get("code", "job_failed"),
                result.get("message", f"editor job {op!r} failed"),
                op=op,
            )
        if result.get("op") != expected_op:
            raise URLabRPCError(
                "unexpected_reply_op",
                f"wanted {expected_op!r}, got {result.get('op')!r}",
                op=op,
            )
        return result

    def _rpc_configure_controller(
        self, *, articulation: str, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._rpc(
            "configure_controller",
            {"articulation": articulation, "params": dict(params)},
            expected_op="configure_controller_ok",
        )

    # -- session lifecycle ------------------------------------------------

    def connect(self, observations: Union[str, ObservationLevel] = "standard") -> None:
        """Handshake: send `hello`, build the local model (MJB fast path,
        compiled-XML fallback when the server runs a different MuJoCo
        version), construct articulation wrappers. A version skew logs a
        warning (silenced by `mujoco_version_check=False`); it no longer
        raises.

        After the handshake, if the user constructed the client with an
        explicit `step_mode` (`direct` or `puppet`), tell the server to
        switch into that mode. The UE step server defaults to
        `live` and rejects `step` requests until a `set_mode`
        promotes it.
        """
        obs_str = wire(coerce(ObservationLevel, observations))
        self._observation_level = obs_str
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
                    "connect(): server has no `meta` op; namespace "
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
                "connect(): no manager registered (editor-time / pre-PIE). "
                "Editor-only ops are available; call begin_pie or wait for "
                "the user to hit Play before stepping."
            )
            return

        # transport="auto": decide locality from the handshake now that the
        # instance host + SHM session dir are known. Same-host upgrades to SHM;
        # a remote client stays on ZMQ (shared memory does not cross machines).
        if self._transport_pref == "auto":
            self._detect_locality_transport()

        # If the user asked for transport="shm" (or auto concluded same-host),
        # (re)build the SHM transport from what this handshake advertised. Done
        # on EVERY handshake, not just the first: a PIE restart hands out a new
        # session dir / RPC contract, and a transport still bound to the previous
        # PIE's dead files would silently freeze all SHM streaming and stall
        # every RPC.
        if self._want_shm:
            self._ensure_shm_transport()

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

    @staticmethod
    def _model_layout(model: Any) -> Dict[str, Any]:
        """The parts of a model a raw puppet state vector depends on."""
        names = {
            "bodies": mujoco.mjtObj.mjOBJ_BODY,
            "joints": mujoco.mjtObj.mjOBJ_JOINT,
            "actuators": mujoco.mjtObj.mjOBJ_ACTUATOR,
        }
        counts = {"bodies": model.nbody, "joints": model.njnt, "actuators": model.nu}
        layout: Dict[str, Any] = {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
        }
        for key, obj_type in names.items():
            layout[key] = [
                mujoco.mj_id2name(model, obj_type, i) or "" for i in range(counts[key])
            ]
        # Joint type/address ordering decides which slice of qpos each
        # joint reads; a slide imported as a hinge keeps nq intact but
        # silently reinterprets the numbers.
        layout["jnt_type"] = [int(v) for v in model.jnt_type]
        layout["jnt_qposadr"] = [int(v) for v in model.jnt_qposadr]
        return layout

    def _check_puppet_drift(self, mjb_bytes: bytes) -> None:
        """Compare the server's model against the attached puppet source.

        A drift cannot be repaired from here -- the state vectors are
        already ambiguous -- so this only reports it.

        Dimension mismatches always raise: a puppet step transmits exactly
        ``nq``/``nv``/``nu`` floats, so there is no reading under which
        differing counts are benign. Name and ordering differences follow
        ``puppet_drift_check`` ('error' | 'warn' | 'off'), defaulting to
        'warn' because a scene may legitimately carry bodies the source
        model never had.
        """
        if self.puppet_drift_check == "off" or self.model is None:
            return
        try:
            server_model, _ = _load_mjb(mjb_bytes)
        except Exception as exc:  # unreadable MJB is not itself a drift
            logger.warning("puppet drift check skipped: MJB unreadable (%s)", exc)
            return

        local = self._model_layout(self.model)
        server = self._model_layout(server_model)
        fatal: List[str] = []
        for field in ("nq", "nv", "nu"):
            if local[field] != server[field]:
                fatal.append(f"{field} local={local[field]} server={server[field]}")
        if fatal:
            raise URLabPuppetDriftError(fatal)

        differences: List[str] = []
        for field in ("bodies", "joints", "actuators"):
            missing = [n for n in local[field] if n and n not in set(server[field])]
            if missing:
                head = ", ".join(missing[:5])
                tail = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
                differences.append(f"{field} missing server-side: {head}{tail}")
        for field in ("jnt_type", "jnt_qposadr"):
            if local[field] != server[field]:
                differences.append(f"{field} ordering differs")

        if not differences:
            logger.debug("puppet drift check: server model matches attached source")
            return
        if self.puppet_drift_check == "warn":
            logger.warning(
                "puppet model drift (pushes may land on the wrong DOFs): %s",
                "; ".join(differences),
            )
            return
        raise URLabPuppetDriftError(differences)

    def _apply_handshake(self, reply: Mapping[str, Any]) -> None:
        """Shared entry point used by `connect()` and tests that inject
        a canned handshake without the socket round-trip."""
        # Hold _data_lock around model/data swap so a concurrent reader
        # (state-stream worker, future async caller) can't see a
        # half-replaced model.
        with self._data_lock:
            self._apply_handshake_locked(reply)

        # The MJB didn't load (version skew) and this reply carried no
        # compiled XML: refetch the handshake once with assets included.
        # Outside _data_lock -- this is an RPC round-trip.
        if self._model_fallback_pending:
            self._model_fallback_pending = False
            logger.info(
                "refetching handshake with include_assets=true to build the "
                "local model from compiled XML"
            )
            fresh = self._rpc(
                "hello",
                {
                    "client_version": self._client_version(),
                    "encoding": "msgpack",
                    "include_assets": True,
                    # Carry the requested observation level: the server's
                    # observation flag is global, so omitting it here reverts
                    # UE to its default level for the rest of the session.
                    "observations": self._observation_level,
                },
                expected_op="hello_ok",
            )
            with self._data_lock:
                self._apply_handshake_locked(fresh)
            # One retry only -- an older server that never ships
            # mjcf_compiled would otherwise refetch on every handshake.
            self._model_fallback_pending = False

        if (
            self.local_model
            and mujoco is not None
            and self.model is None
            and self.articulations
        ):
            logger.warning(
                "no local MjModel could be built from this handshake; "
                "articulation joint/actuator maps are EMPTY, so name-based "
                "control (policy adapters, art.set_ctrl) will silently no-op"
            )

    def _apply_handshake_locked(self, reply: Mapping[str, Any]) -> None:
        # A handshake means a fresh (possibly restarted) PIE world: the old
        # render-snapshot frame_id counter is meaningless now, so drop it.
        # Leaving it set makes the first post-restart get_camera(fresh=True)
        # wait for a frame_id the new session may never reach, burning the
        # whole timeout.
        self._last_step_frame_id = None
        self.session_id = reply.get("session_id")
        self.urlab_version = reply.get("urlab_version")
        self.mujoco_version = reply.get("mujoco_version")
        # Server defaults to manager_present=true for replies that omit
        # the field (older server builds + the PIE-time begin_pie reply
        # always has a manager). Editor-time hello explicitly sets false.
        self.manager_present = bool(reply.get("manager_present", True))
        instance_block = reply.get("instance")
        if isinstance(instance_block, Mapping):
            self.instance = dict(instance_block)
        self.shm_session_dir = str(reply.get("shm_session_dir", "") or "")
        rpc_contract = reply.get("shm_rpc")
        if isinstance(rpc_contract, Mapping):
            self._shm_rpc_contract = dict(rpc_contract)

        if self.mujoco_version_check and mujoco is not None:
            server_ver = str(self.mujoco_version or "")
            client_ver = mujoco.__version__
            if server_ver and server_ver != client_ver:
                # Warn, never raise: a skew only matters for the local-model
                # MJB load, and that path falls back to the version-portable
                # compiled-XML route below.
                if _major_minor(server_ver) != _major_minor(client_ver):
                    logger.warning(
                        "MuJoCo version skew: server=%s client=%s. The MJB "
                        "binary format is version-locked, so the local model "
                        "will be built from the compiled XML instead; puppet "
                        "mode may still behave differently across versions.",
                        server_ver, client_ver,
                    )
                else:
                    logger.debug(
                        "MuJoCo patch-level skew: server=%s client=%s",
                        server_ver, client_ver,
                    )

        # Build the local MjModel: MJB fast path, then the version-portable
        # compiled-XML fallback (an MJB only loads into the exact MuJoCo
        # version that saved it). When neither is present in this reply but
        # a manager is live, flag a one-shot asset refetch -- everything
        # that resolves joints/actuators (URLabArticulation._walk_model)
        # needs a model, and building it silently empty bricks every
        # name-based consumer (issue #76).
        self._model_fallback_pending = False
        mjb_bytes = reply.get("mjb")
        mjcf_xml = reply.get("mjcf_compiled")
        if self.local_model and mujoco is not None and (mjb_bytes or mjcf_xml):
            model = data = None
            if mjb_bytes:
                try:
                    model, data = _load_mjb(mjb_bytes)
                except Exception as exc:
                    logger.warning(
                        "MJB load failed (%s); falling back to the compiled-"
                        "XML model (server mujoco %s, client %s)",
                        exc, self.mujoco_version,
                        mujoco.__version__,
                    )
            if model is None:
                if mjcf_xml:
                    model, data = _load_xml_with_assets(
                        str(mjcf_xml), reply.get("vfs_assets") or {}
                    )
                    logger.info(
                        "local model built from compiled XML (%d assets)",
                        len(reply.get("vfs_assets") or {}),
                    )
                else:
                    self._model_fallback_pending = bool(
                        reply.get("manager_present", True)
                    )
            if model is not None:
                self.model, self.data = model, data
        elif mjb_bytes and mujoco is not None:
            # An external simulation is attached (local_model=False), so the
            # server's MJB is not adopted -- but it is exactly what a puppet
            # push lands in, and it is the only evidence we get that the two
            # models still agree. `attach_puppet_simulation` runs before the
            # level is even loaded; the scene is imported, spawned and PIE'd
            # afterwards, and `sim.start` re-absorbs a handshake describing
            # whatever actually got built. Compare here or not at all.
            self._check_puppet_drift(mjb_bytes)

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

        # Global cameras. Rebuilt from scratch each handshake (like the
        # articulation set above); accumulating across refresh() would leak
        # stale views from a prior scene and make warmup_cameras wait on
        # cameras that no longer exist.
        self.global_cameras = {}
        for cam_name, cam_payload in (reply.get("global_cameras") or {}).items():
            self.global_cameras[cam_name] = URLabCameraView.from_handshake(
                cam_name, cam_payload, owner=None
            )

    def _client_version(self) -> str:
        return "urlab_bridge/0.1.0-alpha"

    def _detect_locality_transport(self) -> None:
        """transport=auto locality decision (see plan_render_farm.md 3.6).

        Conclude SAME-HOST when the handshake's ``instance.host`` matches this
        machine's hostname AND the advertised ``shm_session_dir`` exists on the
        local filesystem; only then flip ``_want_shm`` so connect() upgrades to
        the same-host SHM transport. Otherwise stay on ZMQ -- shared memory
        never crosses machines, so a remote client must use TCP for everything.
        Idempotent and safe to call on every handshake.
        """
        host = str((self.instance or {}).get("host") or "")
        local_host = socket.gethostname()
        shm_ok = bool(self.shm_session_dir) and os.path.isdir(self.shm_session_dir)
        same_host = bool(host) and host == local_host and shm_ok
        if same_host:
            logger.info(
                "URLabClient transport=auto: instance is co-located "
                "(host=%s, shm_dir=%s); upgrading to SHM.",
                host, self.shm_session_dir,
            )
            self._want_shm = True
        else:
            logger.info(
                "URLabClient transport=auto: staying on ZMQ "
                "(instance host=%r vs local %r, shm_dir=%r accessible=%s).",
                host, local_host, self.shm_session_dir, shm_ok,
            )
            self._want_shm = False

    def _ensure_shm_transport(self) -> None:
        """(Re)build the ShmTransport to match what the latest handshake
        advertised, rebuilding only when the config actually changed.

        First call swaps the bootstrap ZMQ transport for a real ShmTransport,
        reusing the ZMQ transport as the SHM fallback (for ops too large for
        the slot, notably `hello`). Subsequent handshakes re-diff the stream
        dir + advertised RPC contract against the live transport; a PIE restart
        that moves the session dir triggers a rebuild so streaming does not
        freeze on the previous PIE's dead files.
        """
        # shm_dir is the per-PIE camera/state STREAM session (state.shm +
        # cam_*.shm live here). The RPC region (req.shm/rep.shm + its kernel
        # events) lives on the RPC transport's OWN, static session, advertised
        # verbatim in the `shm_rpc` handshake block -- a DIFFERENT directory.
        # These must be wired separately: pointing RPC at the stream dir stalls
        # every RPC its full timeout (UE's RPC worker never services it), and
        # pointing the streams at the RPC dir starves state/cameras. Precedence
        # for the stream dir: explicit override > handshake stream dir.
        from .transports.shm import ShmTransport

        shm_dir = self._shm_dir_override or self.shm_session_dir
        if not shm_dir:
            raise RuntimeError(
                "transport='shm' requested but neither shm_dir override nor "
                "handshake `shm_session_dir` was set; pass shm_dir explicitly"
            )
        shm_session_id = os.path.basename(os.path.normpath(shm_dir)) or "live"

        # RPC region: use the advertised contract verbatim when present; else
        # fall back to deriving from the stream dir (legacy servers).
        contract = self._shm_rpc_contract or {}
        config = (
            shm_dir,
            shm_session_id,
            contract.get("req_path"),
            contract.get("rep_path"),
            contract.get("req_event"),
            contract.get("rep_event"),
        )
        if isinstance(self._transport, ShmTransport) and config == self._active_shm_config:
            # Already bound to exactly this session; nothing to rebuild.
            return

        # Recover the ZMQ fallback to reuse it under the new SHM transport. On
        # the first swap the live transport IS the ZMQ bootstrap; on a rebuild
        # it is the old SHM transport, whose fallback we lift out before tearing
        # its stream/RPC bindings down (without closing the shared fallback).
        if isinstance(self._transport, ShmTransport):
            fallback = self._transport._fallback
            self._transport.close(close_fallback=False)
        else:
            fallback = self._transport

        self._transport = make_transport(
            "shm",
            self.address,
            shm_dir=shm_dir,
            shm_session_id=shm_session_id,
            fallback=fallback,
            rpc_req_path=contract.get("req_path"),
            rpc_rep_path=contract.get("rep_path"),
            rpc_req_event=contract.get("req_event"),
            rpc_rep_event=contract.get("rep_event"),
        )
        self._active_shm_config = config
        logger.info(
            "URLabClient: SHM transport active (stream_dir=%s, rpc_session=%s)",
            shm_dir, contract.get("session", shm_session_id),
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
    ) -> "StepResult":
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
        - ``"sync"``: do not use the streams at all. Ask the server to render
          and read back inline, and return the pixels in this reply. The
          server renders from the state this very step applied and waits on
          that capture's readback serial, so the frame cannot be stale or
          reordered -- at the cost of blocking the step on a GPU readback.

          This is the only policy that works in puppet mode: the server pauses
          its camera publishers on puppet entry (the step server owns cadence
          there), so ``latest``/``fresh`` wait on streams that never tick.

        For a bounded, real-sensor-style camera lag that keeps the step loop
        running at full speed, configure a server-side delay with
        ``runtime.set_camera_delay(...)`` -- the delay is applied natively in UE
        so every consumer sees the already-delayed stream, and a delayed camera
        never blocks the step on the current render.

        In every policy the frames also land on ``art.cameras[name].latest_frame``,
        so ``get_camera(name)`` returns the last one served; the reply's
        ``cameras`` block is a snapshot.

        Returns the raw step reply, useful when you need fields like
        ``sim_time`` or ``step`` directly; for state, prefer
        ``client.data`` and articulation accessors (``art.qpos_array``,
        ``art.get_sensors()``, etc.).
        """
        obs_str = wire(coerce(ObservationLevel, observations))
        if camera_query not in ("latest", "fresh", "sync"):
            raise ValueError(
                "camera_query must be 'latest', 'fresh' or 'sync', "
                f"got {camera_query!r}"
            )

        # "sync" is requested on the wire rather than merged from the streams
        # afterwards, so it has to be resolved before the RPC goes out.
        inline_cameras = (
            self._inline_camera_request(include_cameras)
            if include_cameras and camera_query == "sync"
            else None
        )

        # Optional real-time pacing: hold the loop to `target_hz` by sleeping
        # off any time remaining since the previous step() -- absorbs the manual
        # `sleep(dt - elapsed)` pattern from policy/demo loops.
        if target_hz and target_hz > 0 and self._last_step_monotonic is not None:
            slack = (1.0 / target_hz) - (time.monotonic() - self._last_step_monotonic)
            if slack > 0:
                time.sleep(slack)

        if self.step_mode == StepMode.PUPPET:
            reply = self._step_puppet(
                n_steps, observations=obs_str, include_cameras=inline_cameras
            )
        else:
            # Live and Direct both use the RPC step path. UE's step
            # server applies ctrl + returns a state snapshot in either mode;
            # the difference is whether mj_step actually runs (Direct) or the
            # request just stamps NetworkValue and reads current state with
            # UE's autonomous physics continuing to advance (Live).
            reply = self._step_direct(
                n_steps, observations=obs_str, include_cameras=inline_cameras
            )

        fid = reply.get("frame_id")
        if fid is not None:
            self._last_step_frame_id = int(fid)

        # Inline pixels are already in the reply (and already decoded onto
        # each URLabCameraView by _absorb_step_reply); only the streamed
        # policies have anything left to merge.
        if include_cameras and inline_cameras is None:
            self._attach_streamed_cameras(
                reply, include_cameras, camera_query, camera_timeout_s
            )
        self._last_step_monotonic = time.monotonic()
        return StepResult(reply)

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
        real/seg, HxW float32 for depth). Returns ``None`` if no frame arrived
        in time OR if ``fresh=True`` could not be satisfied before the deadline
        -- a stale frame is never returned dressed up as fresh. Raises
        ``KeyError`` (listing available names) if ``name`` is unknown.
        """
        view = self._find_camera_view(name)
        deadline = time.monotonic() + max(0.0, timeout_s)
        target = self._last_step_frame_id if fresh else None
        # A camera under a server-side delay cannot reveal a frame for the
        # just-stepped state within the step (the feed is intentionally the
        # delayed past), so demanding frame_id >= target would burn the whole
        # timeout every call. Drop the fresh requirement for delayed cameras
        # and serve latest.
        if target is not None and self._camera_applied_delay.get(name, 0.0) > 0.0:
            target = None
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
                # fresh requested but no matching frame arrived: report the
                # miss as None instead of handing back a stale image.
                if target is not None and not fresh_ok:
                    return None
                return frame  # may be None (never arrived)
            time.sleep(0.002)

    def _step_direct(
        self,
        n_steps: int,
        *,
        observations: str,
        include_cameras: "Optional[Dict[str, str]]" = None,
    ) -> Dict[str, Any]:
        per_art: Dict[str, Any] = {}
        for prefix, art in self.articulations.items():
            per_art[prefix] = art._build_step_request(control_mode=None)

        # Cameras stream over SHM/ZMQ by default and are merged into the reply
        # by _attach_streamed_cameras after the step; the reply's `frame_id` is
        # what a "fresh" query synchronises against. `include_cameras` here is
        # non-None only for camera_query="sync", which asks the server to
        # render and read back inline instead.
        request: Dict[str, Any] = {
            "n_steps": int(n_steps),
            "observations": observations,
            "per_articulation": per_art,
        }
        # Entity-level external wrenches (URLabEntity.apply_xfrc) ride the step
        # request keyed by body name. This is the wire half of the feature;
        # the server must read `entity_xfrc` and stamp d->xfrc_applied for
        # those bodies (plugin-side work). Until it does the forces have no
        # effect, but they are no longer silently discarded on the client.
        if self._pending_entity_xfrc:
            request["entity_xfrc"] = {
                name: vec.tolist()
                for name, vec in self._pending_entity_xfrc.items()
            }
        if include_cameras:
            request["include_cameras"] = include_cameras
        reply = self._rpc("step", request, expected_op="step_ok")
        self._absorb_step_reply(reply)
        # Clear xfrc post-step per MuJoCo semantics
        for art in self.articulations.values():
            art.clear_xfrc()
        self._pending_entity_xfrc.clear()
        return reply

    def _step_puppet(
        self,
        n_steps: int,
        *,
        observations: str,
        include_cameras: "Optional[Dict[str, str]]" = None,
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

        # Pending external wrenches cannot be honoured in puppet mode: the
        # client's local mj_step is authoritative and UE's d->xfrc_applied is
        # overwritten by the pushed state every step. Rather than let them sit
        # in the buffers and silently resurrect on a later mode switch, warn
        # once and clear them here.
        if self._pending_entity_xfrc:
            warnings.warn(
                "puppet step: dropping pending entity xfrc "
                f"{sorted(self._pending_entity_xfrc)} (external wrenches are "
                "inert in puppet mode; the client's mj_step is authoritative).",
                stacklevel=2,
            )
            self._pending_entity_xfrc.clear()
        stale_art_xfrc = [
            art.prefix for art in self.articulations.values() if art._pending_xfrc
        ]
        if stale_art_xfrc:
            warnings.warn(
                f"puppet step: dropping pending articulation xfrc on "
                f"{stale_art_xfrc} (inert in puppet mode).",
                stacklevel=2,
            )
            for art in self.articulations.values():
                art.clear_xfrc()

        # n_steps == 0: skip mj_step entirely (MJX / manual state authors
        # push whatever they already wrote into client.data).
        for _ in range(int(n_steps)):
            mujoco.mj_step(self.model, self.data)

        # Cameras stream over SHM/ZMQ (see _step_direct) unless camera_query
        # was "sync". In puppet mode the server pauses those publishers, so
        # "sync" is in practice the only policy that yields a frame here.
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
        if include_cameras:
            request["include_cameras"] = include_cameras

        # Owner broadcast: fan the just-pushed kinematics out to any viewers,
        # synced to this step (the client is the authority in puppet mode). The
        # render-server RPC below is unchanged; viewers are a parallel PUB.
        if self.broadcast_viewers:
            if not self._viewer_bcast_bound:
                self._transport.enable_viewer_broadcast(self.viewer_port)
                self._viewer_bcast_bound = True
            self._transport.publish_viewer_state(
                {
                    "t": request["time"],
                    "qpos": request["qpos"],
                    "qvel": request["qvel"],
                }
            )

        reply = self._rpc("step", request, expected_op="step_ok")
        self._absorb_step_reply(reply)
        return reply

    # -- streaming-mode SUB infrastructure --------------------------------

    def _start_streaming_subs(self) -> None:
        """Spin up the state-snapshot stream + one camera stream per
        registered camera. Idempotent. Called from connect() when
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
        # camera to one port, so all but one get no frames. Global (scene-level)
        # cameras stream on the same path as per-articulation ones.
        enable: Dict[str, Any] = {}
        for _prefix, cam_name, view in self._iter_all_camera_views():
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
                # A failure here means no camera ever streams -- surface it
                # loudly rather than leaving the user with silent black feeds.
                logger.warning(
                    "set_camera_streaming failed at stream startup (%s); "
                    "camera feeds will not appear until it succeeds", exc,
                )
        # One per-camera stream; the transport dedupes on (prefix, name).
        # Prefer the endpoint/topic from the set_camera_streaming reply (the
        # bound port); fall back to the handshake values for older servers.
        for prefix, cam_name, view in self._iter_all_camera_views():
            info = reply.get(cam_name)
            if info is not None:
                if info.zmq_endpoint:
                    view._zmq_endpoint = info.zmq_endpoint
                if info.zmq_topic:
                    view._zmq_topic = info.zmq_topic
            topic = getattr(view, "_zmq_topic", None)
            endpoint = getattr(view, "_zmq_endpoint", None)
            if not topic or not endpoint:
                continue
            self._transport.start_camera_stream(
                prefix, cam_name, endpoint, topic,
                self._make_camera_callback(prefix, cam_name),
            )

    def _iter_all_camera_views(self):
        """Yield ``(stream_prefix, cam_name, view)`` for every discovered
        camera: per-articulation cameras under the articulation prefix and
        global (scene-level) cameras under the ``"global"`` prefix. The prefix
        is only the transport's stream key; ``cam_name`` stays canonical."""
        for art in self.articulations.values():
            for cam_name, view in art.cameras.items():
                yield art.prefix, cam_name, view
        for cam_name, view in self.global_cameras.items():
            yield "global", cam_name, view

    def _resolve_camera_view(
        self, prefix: str, cam_name: str
    ) -> "Optional[URLabCameraView]":
        """Resolve the live view for a (stream_prefix, cam_name) pair, tolerant
        of a re-attach that rebuilt the wrapper. ``"global"`` routes to the
        scene-level camera set."""
        if prefix == "global":
            return self.global_cameras.get(cam_name)
        art = self.articulations.get(prefix)
        return art.cameras.get(cam_name) if art else None

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

    def refresh(self, observations: Union[str, ObservationLevel] = "standard") -> None:
        """Re-run the handshake to pick up scene changes (after spawn/import or
        an external edit). Idempotent; same wire op as :meth:`connect`.

        Note: this rebuilds every ``URLabArticulation`` / ``URLabEntity`` /
        ``URLabCameraView`` wrapper from the fresh handshake, so any references
        you held (``art = client.articulation(...)``, ``cam = art.cameras[...]``)
        are orphaned and keep pointing at the pre-refresh objects. Re-fetch them
        from ``client.articulations`` / ``client.camera_names`` after a refresh.
        """
        self.connect(observations=observations)

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

    @staticmethod
    def _decode_camera_frame(
        view,
        pixels: bytes,
        frame_id: "Optional[int]" = None,
        sim_time: "Optional[float]" = None,
        capture_time: "Optional[float]" = None,
    ) -> bool:
        """Decode *pixels* into *view*, returning True on success.

        Depth frames are float32 (HxW); colour frames arrive as BGRA8
        (HxWx4), with REAL mode rotated to RGBA for consumer convenience.
        On success, updates ``view.latest_frame``, ``frame_count``,
        ``recv_monotonic``, and any non-``None`` metadata fields.
        """
        try:
            w, h = view.resolution
            if view.mode == CameraMode.DEPTH:
                arr = np.frombuffer(pixels, dtype=np.float32)
                if arr.size != w * h:
                    return False
                view.latest_frame = arr.reshape((h, w))
            else:
                arr = np.frombuffer(pixels, dtype=np.uint8)
                if arr.size != w * h * 4:
                    return False
                bgra = arr.reshape((h, w, 4))
                if view.mode == CameraMode.REAL:
                    view.latest_frame = bgra[..., [2, 1, 0, 3]]
                else:
                    view.latest_frame = bgra
            view.frame_count += 1
            view.recv_monotonic = time.monotonic()
            if frame_id is not None:
                view.frame_id = frame_id
            if sim_time is not None:
                view.sim_time = sim_time
            if capture_time is not None:
                view.capture_unix_time = capture_time
            return True
        except Exception:
            return False

    def _make_camera_callback(self, prefix: str, cam_name: str) -> "Callable[[bytes], None]":
        """Build a frame-bytes callback bound to a specific (prefix, cam)
        URLabCameraView. The closure captures only string keys and resolves
        the live view on each call so a re-attached camera still updates."""

        def _on_frame(
            pixels: bytes,
            frame_id: "Optional[int]" = None,
            sim_time: "Optional[float]" = None,
            capture_time: "Optional[float]" = None,
        ) -> None:
            view = self._resolve_camera_view(prefix, cam_name)
            if view is None:
                return
            if not URLabClient._decode_camera_frame(
                view, pixels,
                frame_id=frame_id, sim_time=sim_time, capture_time=capture_time,
            ):
                logger.debug("camera decode failed (%s/%s)",
                             prefix, cam_name)
            else:
                with self._camera_frame_cond:
                    self._camera_frame_cond.notify_all()

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

    def _inline_camera_request(self, include_cameras: Any) -> Dict[str, str]:
        """Build the wire ``include_cameras`` block for ``camera_query="sync"``.

        The server keys this by camera name with a per-camera capture mode;
        ``"sync"`` means render now and embed the pixels in the step reply
        rather than publishing them on a stream. ``True`` expands to every
        discovered camera, since the wire form has no "all" spelling.
        """
        names = self._select_camera_names(include_cameras)
        if names is None:
            names = self.camera_names()
        return {str(name): "sync" for name in names}

    def _gather_cached_cameras(self, include_cameras: Any) -> Dict[str, Dict[str, Any]]:
        """Read the latest cached frame off each requested URLabCameraView
        and bundle into a {prefix: {cam_name: {pixels, mode, ...}}} dict.

        These frames arrive over the async SHM/ZMQ stream (in every step mode
        now), not the step RPC reply. ``frame_id`` is the post-step state the
        frame shows -- compare it against the step reply's ``frame_id`` to know
        how fresh the frame is."""
        wanted = self._select_camera_names(include_cameras)
        out: Dict[str, Dict[str, Any]] = {}
        for prefix, cam_name, view in self._iter_all_camera_views():
            if wanted is not None and cam_name not in wanted:
                continue
            if view.latest_frame is None:
                continue
            mode = view.mode.value if hasattr(view.mode, "value") else str(view.mode)
            out.setdefault(prefix, {})[cam_name] = {
                "pixels": view.latest_frame,
                "mode": mode,
                "resolution": list(view.resolution),
                "frame_count": view.frame_count,
                "frame_id": view.frame_id,
                "sim_time": view.sim_time,
            }
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
        # Normalise the selector the same way the gather path does; a list /
        # tuple / set / str no longer silently degrades to "wait on every
        # camera" (which made one dormant camera burn the full timeout).
        wanted = self._select_camera_names(include_cameras)

        def _all_fresh() -> bool:
            for _prefix, cam_name, view in self._iter_all_camera_views():
                if wanted is not None and cam_name not in wanted:
                    continue
                # A camera under a server-side delay can never reveal a frame
                # tagged with the just-stepped state within the step, so it
                # would deadlock the fresh-wait; treat it as satisfied (it is
                # served "latest", i.e. the delayed past, by design).
                if self._camera_applied_delay.get(cam_name, 0.0) > 0.0:
                    continue
                fid = view.frame_id
                if fid is None or fid < target_frame_id:
                    return False
            return True

        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._camera_frame_cond:
            while True:
                if _all_fresh():
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._camera_frame_cond.wait(timeout=remaining)

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
    ) -> "StepResult":
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
        return StepResult(reply)

    def forward(self) -> "StepResult":
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
        return StepResult(reply)

    # -- network model upload ---------------------------------------------

    def upload_model(
        self,
        xml: "Union[str, bytes, os.PathLike]",
        assets: Optional[Mapping[str, bytes]] = None,
        *,
        step_mode: str = "direct",
        chunk_bytes: int = 4 * 1024 * 1024,
        asset_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload an MJCF model to the server over the content-addressed protocol.

        The model is flattened into one self-contained XML (every ``<include>``
        resolved, every asset ``file=`` rewritten to a bare filename), hashed,
        and offered to the server via ``upload_model_manifest``. The server
        checks its cache and asks only for the blobs it is missing; only those
        are streamed with ``upload_model_chunk`` (chunked at ``chunk_bytes``),
        then ``upload_model_commit`` compiles the model and returns its
        dimensions plus the compiled MJB.

        Parameters
        ----------
        xml:
            A filesystem path, an XML string, an XML ``bytes`` blob, or an
            ``os.PathLike`` to the model's MJCF.
        assets:
            Optional mapping of bare filename -> raw bytes. When given it is
            used verbatim as the asset set (keys must be bare filenames). When
            ``None`` the referenced ``file=`` assets are read from disk,
            resolved against ``asset_root`` (or the XML directory + meshdir /
            texturedir).
        step_mode:
            Advisory step mode recorded in the manifest (``"direct"`` default).
        chunk_bytes:
            Max bytes per ``upload_model_chunk`` (default 4 MiB). Small blobs
            are sent as a single chunk.
        asset_root:
            Base directory for resolving includes and asset files; overrides the
            XML's own directory.

        Returns
        -------
        dict
            The ``upload_model_commit_ok`` reply: ``imported``, ``nq`` / ``nv``
            / ``nu`` / ``nbody`` / ``ngeom``, ``mjb`` (compiled bytes) and
            ``warnings``.

        Raises
        ------
        ValueError
            On a non-bare asset key, too many assets, or a blob / total that
            exceeds the server's advertised limits.
        URLabRPCError
            If the server rejects the manifest or the commit (e.g.
            ``import_failed``), carrying the server's code and message.
        """
        xml_text, asset_paths = flatten_model(xml, asset_root=asset_root)
        xml_bytes = xml_text.encode("utf-8")

        blobs = self._gather_upload_assets(assets, asset_paths)

        # Client-side validation before any round-trip (the server enforces the
        # same rules; failing here is faster and friendlier).
        for name in blobs:
            require_bare_filename(name)
        if len(blobs) > MAX_ASSETS:
            raise ValueError(
                f"model references {len(blobs)} assets; the limit is {MAX_ASSETS}"
            )

        xml_sha = sha256_hex(xml_bytes)
        asset_shas = {name: sha256_hex(data) for name, data in blobs.items()}
        asset_manifest = [
            {"name": name, "sha256": asset_shas[name], "size": len(blobs[name])}
            for name in blobs
        ]
        total_bytes = len(xml_bytes) + sum(len(d) for d in blobs.values())

        manifest = self._rpc(
            "upload_model_manifest",
            {
                "xml_sha256": xml_sha,
                "assets": asset_manifest,
                "total_bytes": total_bytes,
                "step_mode": step_mode,
            },
            expected_op="upload_model_manifest_ok",
        )
        upload_id = manifest.get("upload_id")
        if not upload_id:
            raise URLabRPCError(
                "bad_manifest",
                "upload_model_manifest_ok missing upload_id",
                op="upload_model_manifest",
            )
        need_xml = bool(manifest.get("need_xml", True))
        need_assets = set(manifest.get("need_assets", []) or [])
        max_asset_bytes = manifest.get("max_asset_bytes")
        max_total_bytes = manifest.get("max_total_bytes")

        # Enforce the server's advertised ceilings before streaming a byte.
        if max_total_bytes is not None and total_bytes > int(max_total_bytes):
            raise ValueError(
                f"model upload is {total_bytes} bytes, over the server limit of "
                f"{int(max_total_bytes)}"
            )
        if max_asset_bytes is not None:
            cap = int(max_asset_bytes)
            if need_xml and len(xml_bytes) > cap:
                raise ValueError(
                    f"flattened XML is {len(xml_bytes)} bytes, over the per-blob "
                    f"limit of {cap}"
                )
            for name in need_assets:
                if name in blobs and len(blobs[name]) > cap:
                    raise ValueError(
                        f"asset {name!r} is {len(blobs[name])} bytes, over the "
                        f"per-blob limit of {cap}"
                    )

        # Stream only the blobs the server asked for (a cache hit skips them).
        if need_xml:
            self._upload_blob(upload_id, "xml", "model.xml", xml_bytes, xml_sha, chunk_bytes)
        for name in blobs:
            if name in need_assets:
                self._upload_blob(
                    upload_id, "asset", name, blobs[name], asset_shas[name], chunk_bytes
                )

        return self._rpc(
            "upload_model_commit",
            {"upload_id": upload_id},
            expected_op="upload_model_commit_ok",
        )

    def upload_model_file(
        self, path: "Union[str, os.PathLike]", **kwargs: Any
    ) -> Dict[str, Any]:
        """Upload an ``.xml`` model file, auto-gathering assets from its directory.

        Thin convenience over :meth:`upload_model`: reads ``path`` and resolves
        every referenced asset relative to the file's own directory (plus the
        model's meshdir / texturedir). Accepts the same keyword arguments as
        :meth:`upload_model` except ``assets`` (assets are always auto-gathered
        here).
        """
        return self.upload_model(os.fspath(path), **kwargs)

    @staticmethod
    def _gather_upload_assets(
        assets: Optional[Mapping[str, bytes]],
        asset_paths: Mapping[str, str],
    ) -> Dict[str, bytes]:
        """Return the {bare_name: bytes} blob set for an upload.

        With an explicit ``assets`` mapping the caller's bytes are used
        verbatim; otherwise every referenced asset is read from its resolved
        path on disk.
        """
        if assets is not None:
            out: Dict[str, bytes] = {}
            for name, data in assets.items():
                if isinstance(data, (bytearray, memoryview)):
                    data = bytes(data)
                elif not isinstance(data, bytes):
                    raise TypeError(
                        f"asset {name!r} must be bytes, got {type(data).__name__}"
                    )
                out[str(name)] = data
            return out
        out = {}
        for bare, abspath in asset_paths.items():
            if not os.path.isfile(abspath):
                raise FileNotFoundError(
                    f"asset {bare!r} referenced by the model was not found at "
                    f"{abspath!r}; pass assets=... or asset_root=... to locate it"
                )
            with open(abspath, "rb") as f:
                out[bare] = f.read()
        return out

    def _upload_blob(
        self,
        upload_id: str,
        kind: str,
        name: str,
        data: bytes,
        sha256: str,
        chunk_bytes: int,
    ) -> Dict[str, Any]:
        """Stream one blob to the server in ``chunk_bytes`` slices.

        Returns the final ``upload_model_chunk_ok`` reply. Each chunk carries
        the blob's full ``sha256`` / ``total`` and its own ``offset`` so the
        server can reassemble and verify content-addressed."""
        total = len(data)
        reply: Dict[str, Any] = {}
        for offset, chunk in iter_chunks(data, chunk_bytes):
            reply = self._rpc(
                "upload_model_chunk",
                {
                    "upload_id": upload_id,
                    "kind": kind,
                    "name": name,
                    "sha256": sha256,
                    "offset": offset,
                    "total": total,
                    "data": chunk,
                },
                expected_op="upload_model_chunk_ok",
            )
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
        # Release a cooperative farm lease (from URLabPool.lease) so the
        # instance frees up immediately instead of waiting out its TTL. Best-
        # effort: swallow any error so close() never raises during teardown.
        if self.lease_id is not None and self.session_id is not None:
            try:
                self._rpc("release_lease", {"lease_id": self.lease_id})
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("URLabClient.close: release_lease failed: %s", exc)
            self.lease_id = None
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

    def release(self) -> None:
        """Release a farm lease and close the client. Alias of :meth:`close`
        (which already releases the lease); provided so leased-client call
        sites read as ``client.release()``."""
        self.close()

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

        per_art = reply.get("arts") or {}
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
                # Find the matching URLabCameraView by bare camera name --
                # per-articulation first, then the global ones. A scene-level
                # camera (a worldbody <camera> in the imported MJCF) is ONLY
                # in global_cameras, and an articulations-only lookup dropped
                # its pixels silently.
                try:
                    view = self._find_camera_view(cam_name)
                except KeyError:
                    logger.debug("include_cameras: unknown camera %r", cam_name)
                    continue
                # Trust the payload's own dimensions over the view's
                # registered resolution: they are what `pixels` was sized
                # by, and a view registered at a stale resolution would
                # otherwise fail the size check and drop a good frame.
                try:
                    w = int(cam_payload.get("width") or view.resolution[0])
                    h = int(cam_payload.get("height") or view.resolution[1])
                    if view.mode == CameraMode.DEPTH:
                        arr = np.frombuffer(pixels, dtype=np.float32)
                        if arr.size != w * h:
                            raise ValueError(f"expected {w * h} depth samples, got {arr.size}")
                        view.latest_frame = arr.reshape((h, w))
                    else:
                        arr = np.frombuffer(pixels, dtype=np.uint8)
                        if arr.size != w * h * 4:
                            raise ValueError(f"expected {w * h * 4} bytes, got {arr.size}")
                        bgra = arr.reshape((h, w, 4))
                        view.latest_frame = (
                            bgra[..., [2, 1, 0, 3]] if view.mode == CameraMode.REAL else bgra
                        )
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
                    logger.debug("include_cameras decode failed (%s): %s", cam_name, exc)

        # Non-articulation entities. Write the reply's xpos/xquat into the
        # local MjData at the body's slot so `entity.root_pos_w` /
        # `root_quat_w` reads consistent values regardless of whether the
        # body has a free joint being driven by qpos. Order matters: this
        # runs AFTER `_mirror_state_into_data` (which calls mj_forward over
        # qpos), so we override mj_forward's per-body xpos for
        # non-articulation entities with the wire-shipped value.
        entity_block = reply.get("scene") or {}
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
        per_art = reply.get("arts") or {}
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


def _major_minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


def _load_xml_with_assets(xml: str, assets: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Build an MjModel from the handshake's compiled MJCF + VFS assets.

    Version-portable counterpart to `_load_mjb`: XML parses across MuJoCo
    releases, while an MJB only loads into the exact version that saved it.
    Asset keys are the bare filenames the server rewrote the `file=` refs to.
    """
    if mujoco is None:  # pragma: no cover
        raise RuntimeError("mujoco not installed")
    asset_dict = {}
    for name, blob in (assets or {}).items():
        if isinstance(blob, (bytearray, memoryview)):
            blob = bytes(blob)
        asset_dict[str(name)] = blob
    model = mujoco.MjModel.from_xml_string(xml, asset_dict)
    return model, mujoco.MjData(model)


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
