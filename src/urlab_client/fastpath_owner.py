"""Fast-path owner: advertise a live sim to UE fast-path renderers.

A fast-path *owner* is any process that holds a MuJoCo model and steps it (a
puppet script, or a UE live/direct instance). It offers two things to a
renderer:

* a **control channel** (ZMQ REQ/REP) that answers ``fastpath_hello`` with the
  model (declaring its ``model_format``: mjb | xml | mjz, plus the asset bundle
  an xml references) and the transform-bus endpoint, so a renderer can connect
  with no shared file, and
* the **render transform bus** (ZMQ PUB, topic ``render``) that carries the
  per-body render tier (transforms + an optional capability-gated debug tier)
  every step.

The owner also writes a **registry entry** (a JSON file in the shared registry
directory, the same directory UE bridge servers use) so a UE renderer's server
browser can discover it without being told an endpoint. The entry is refreshed
on a heartbeat and removed on close.

This keeps the renderer dumb and light: it discovers an owner, pulls the MJB,
subscribes to the bus, and renders. No physics runs on the renderer.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import tempfile
import threading
import time
from typing import Optional, Sequence

import msgpack
import zmq

from .pool import default_registry_dir

# Capability string a fast-path owner advertises; the UE server browser filters
# on it so it lists owners a renderer can actually consume.
FASTPATH_OWNER_CAP = "fastpath_owner"

# Canonical capability wire strings -- one vocabulary shared with UE, whose
# EMjCapability advertises exactly these (RpcDispatcher.cpp: "stream_cameras" /
# "accept_input"). Peek/viewer/VR are NOT modes: a consumer uses stream_cameras
# to observe the owner's view and accept_input to push perturbations back.
CAP_STREAM_CAMERAS = "stream_cameras"
CAP_ACCEPT_INPUT = "accept_input"
# Fold legacy / enum-style spellings onto the canonical wire strings so older
# callers (and the UE enum names) keep working.
_CAP_ALIASES = {
    "view": CAP_STREAM_CAMERAS,
    "streamcameras": CAP_STREAM_CAMERAS,
    "stream_cameras": CAP_STREAM_CAMERAS,
    "acceptinput": CAP_ACCEPT_INPUT,
    "accept_input": CAP_ACCEPT_INPUT,
}


def _canon_cap(c: str) -> str:
    """Map a capability spelling to its canonical wire string (UE vocabulary)."""
    return _CAP_ALIASES.get(str(c).strip().lower(), str(c))


def _vec3(v) -> list[float]:
    """Coerce an arbitrary wire value into exactly three floats, padding a short
    sequence with zeros and truncating a long one. Guarantees indexing [0..2] is
    always safe, so a malformed force/torque can't raise mid-handler."""
    out = [float(x) for x in v][:3]
    out += [0.0] * (3 - len(out))
    return out


class FastPathOwner:
    """Advertise + serve a MuJoCo model to UE fast-path renderers.

    Parameters
    ----------
    mjb_bytes:
        The compiled, version-matched MJB the renderer will load. Must match the
        model whose transforms are published on the bus (same geom order).
    scene:
        Human-readable scene id (e.g. "franka_emika_panda"); shown in the browser.
    control_port, bus_port:
        Ports to bind. control is REQ/REP; bus is the geoms PUB.
    bind:
        Interface to bind on ("0.0.0.0" for all, so remote renderers can reach it).
    advertise_host:
        Host a remote renderer should dial. Defaults to this machine's hostname;
        endpoints in the registry use it so cross-machine discovery works.
    ngeom:
        Geom count, published in the advertisement for display.
    instance_id:
        Registry instance id; defaults to the scene name.
    """

    def __init__(
        self,
        mjb_bytes: bytes,
        *,
        scene: str,
        control_port: int = 5571,
        bus_port: int = 5561,
        bind: str = "0.0.0.0",
        advertise_host: Optional[str] = None,
        ngeom: int = 0,
        model_format: str = "mjb",
        assets: "Optional[dict]" = None,
        instance_id: Optional[str] = None,
        registry_dir: Optional[str] = None,
        capabilities: Sequence[str] = (CAP_STREAM_CAMERAS, CAP_ACCEPT_INPUT),
    ) -> None:
        self._mjb = bytes(mjb_bytes)
        # Wire format of the model bytes served on fastpath_hello: "mjb" (compiled,
        # version-locked) or "xml"/"mjz" (source the renderer decodes in-engine, so
        # no MJB version match is required). For "xml", model_bytes is the flattened
        # MJCF and `assets` is {bare-filename: bytes} for its meshes/textures.
        self._model_format = str(model_format)
        self._assets: "dict[str, bytes]" = {
            str(k): bytes(v) for k, v in (assets or {}).items()
        }
        self._scene = scene
        self._ngeom = int(ngeom)
        # Granted capabilities advertised in the registry (canonical UE wire
        # strings). "stream_cameras" = a consumer may subscribe to this owner's
        # view; "accept_input" = interactive consumers may push perturbations.
        # Drop "accept_input" for a look-but-don't-touch owner.
        self._caps = {FASTPATH_OWNER_CAP} | {_canon_cap(c) for c in capabilities}
        self._host = advertise_host or socket.gethostname()
        self._control_port = int(control_port)
        self._bus_port = int(bus_port)
        self._instance_id = instance_id or scene or "live"
        self._registry_dir = registry_dir or default_registry_dir()
        self._pid = os.getpid()

        # Endpoints a renderer dials. The bus binds on `bind` but advertises a
        # dialable host; control likewise.
        self._bus_endpoint = f"tcp://{self._host}:{self._bus_port}"
        self._control_endpoint = f"tcp://{self._host}:{self._control_port}"

        self._ctx = zmq.Context.instance()
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.setsockopt(zmq.LINGER, 0)
        self._rep.bind(f"tcp://{bind}:{self._control_port}")

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(f"tcp://{bind}:{self._bus_port}")

        self._poller = zmq.Poller()
        self._poller.register(self._rep, zmq.POLLIN)

        self._registry_path = os.path.join(
            self._registry_dir, f"fastpath_{self._instance_id}_{self._pid}.json"
        )
        self._last_registry_write = 0.0
        # Latest drag INTENT from a renderer/viewer, not a computed force: a mirror
        # has no mjData, so it forwards {select, active, localpos, refselpos} (body,
        # grab point in body-local MuJoCo frame, drag target in world frame) and the
        # owner runs the real mjv_applyPerturbForce (mass-scaled + critically damped,
        # exactly like simulate's Ctrl-drag) in apply_perturbations(). Guarded by
        # _lock because a gRPC server (optional) submits from its own threads; the
        # latest published {t,qpos,qvel} is cached here too for a gRPC subscribe.
        self._lock = threading.Lock()
        self._perturb_intent: "Optional[dict]" = None
        # Raw wrench pushes (legacy/programmatic: apply an EXACT force/torque, e.g. a
        # Python peek or a test): body id -> accumulated 6-vector. Distinct from the
        # drag intent -- a raw push is a fixed force, a drag is a spring toward a point.
        self._perturb_force: "dict[int, list[float]]" = {}
        # mjv perturb machinery, built lazily on first apply (needs the live model).
        self._pert = None          # mujoco.MjvPerturb
        self._pert_scene = None    # throwaway mjvScene (only pert.scale uses it)
        self._pert_sel = None      # body id localmass was last initialised for
        # Latest LEAN per-body transform frame ({f,bxpos,bxquat,cxpos?,cxquat?})
        # -- transforms only, NEVER the debug tier. This is what the always-on ZMQ
        # `render` topic carries (it has no way to negotiate the debug tier) and the
        # base every gRPC subscribe(format=render) stream augments per ITS OWN caps.
        # Written by every publish_bodies/publish_mjdata.
        self._latest_lean: "Optional[dict]" = None
        # The (model, data) behind `_latest_lean`, kept so a gRPC stream can compute
        # the debug tier (§8.2) for its own subscription off the latest post-step
        # state. None for publish_bodies (no mjData -> transforms only).
        self._latest_model = None
        self._latest_data = None
        # Per-publish frame cache keyed by caps-key (`_caps_key`): the lean frame
        # under the None key, plus one pre-built debug frame per DISTINCT active
        # subscriber caps-set (usually 1-2). Built ONCE per publish on the owner's
        # main thread in `_send_transforms`, off the SAME (model, data) snapshot as
        # the transforms -- so the debug fields are consistent with the frame, and
        # the gRPC stream thread only ever RE-SENDS a cached frame (a cheap dict
        # copy) instead of rebuilding the debug tier on every ~200 Hz poll (which
        # held the GIL on the stream thread and starved the owner's step loop).
        self._latest_frames: "dict" = {}
        # Debug-tier caps (source-of-truth §8.2). This is NOT a per-subscription
        # value any more -- each gRPC stream carries its own caps (owner_server.py
        # `_stream_render`). This field is the AGGREGATE of all currently-active
        # render subscribers (union of their caps), recomputed on every
        # register/unregister, and gates only the shared `latest_transforms()`
        # cache view -- so it goes back to lean the moment the last debug
        # subscriber disconnects. `set_render_debug_caps` still writes it directly
        # for callers that set caps without going through the subscriber registry.
        self._render_debug_caps: dict = {
            "contacts": False, "overlay": False, "max_contacts": 0,
        }
        # Active render subscriptions -> their negotiated caps (token-keyed), so the
        # aggregate can be recomputed when any one connects/disconnects.
        self._render_subscribers: "dict[int, dict]" = {}
        self._render_sub_next = 0
        self._grpc_server = None  # optional; started by start_grpc_server()
        self._grpc_endpoint: Optional[str] = None
        self._write_registry()

    # -- model swap --------------------------------------------------------- #
    def update_model(self, mjb_bytes: bytes, ngeom: Optional[int] = None) -> None:
        """Swap the MJB served on ``fastpath_hello`` after a live scene change, so a
        renderer that discovers this owner LATER pulls the CURRENT scene, not the one
        it started on (otherwise its geometry would mismatch the transform stream)."""
        self._mjb = bytes(mjb_bytes)
        if ngeom is not None:
            self._ngeom = int(ngeom)
        self._write_registry()  # refresh advertised ngeom

    # -- properties --------------------------------------------------------- #
    @property
    def capabilities(self) -> "tuple[str, ...]":
        return tuple(sorted(self._caps))

    @property
    def accepts_input(self) -> bool:
        """Whether interactive consumers may push perturbations (accept_input cap)."""
        return CAP_ACCEPT_INPUT in self._caps

    @property
    def streams_view(self) -> bool:
        """Whether consumers may subscribe to this owner's view (stream_cameras cap)."""
        return CAP_STREAM_CAMERAS in self._caps

    @property
    def model_bytes(self) -> bytes:
        """The model served on fastpath_hello (see :attr:`model_format`)."""
        return self._mjb

    @property
    def model_format(self) -> str:
        """Wire format of :attr:`model_bytes`: 'mjb' | 'xml' | 'mjz'."""
        return self._model_format

    @property
    def grpc_endpoint(self) -> "Optional[str]":
        """The owner's gRPC endpoint ('host:port') once start_grpc_server() ran, else None."""
        return self._grpc_endpoint

    @property
    def assets(self) -> "dict[str, bytes]":
        """Mesh/texture assets for an xml model ({bare-filename: bytes}); empty for mjb."""
        return self._assets

    @property
    def scene(self) -> str:
        return self._scene

    @property
    def ngeom(self) -> int:
        return self._ngeom

    @property
    def bus_endpoint(self) -> str:
        return self._bus_endpoint

    @property
    def control_endpoint(self) -> str:
        return self._control_endpoint

    # -- hello -------------------------------------------------------------- #
    def hello_reply(self, bus: Optional[str] = None) -> dict:
        """Build the ONE ``fastpath_hello`` reply schema, identical on every
        transport face (render source-of-truth §11): ``{ok, scene, ngeom,
        capabilities, bus, model_format, model|mjb, vfs_assets?}``.

        Field names match the UE ``MjRendererDriverClient::FetchModel`` reader:
        ``model_format`` selects the loader on the renderer -- ``mjb`` ships the
        compiled bytes under ``mjb`` (version-locked), anything else ships the
        source under ``xml`` plus its asset bundle under ``vfs_assets`` (each
        asset base64 under a ``<name>__b64__`` key) so the renderer compiles it
        with its OWN libmujoco (version-independent). ``bus`` overrides the
        advertised ZMQ bus endpoint for faces that stream elsewhere (the gRPC
        face passes ``grpc://<endpoint>``).
        """
        reply = {
            "ok": True,
            "scene": self._scene,
            "ngeom": self._ngeom,
            "capabilities": list(self.capabilities),
            "model_format": self._model_format,
            "bus": self._bus_endpoint if bus is None else bus,
            # generic aliases (kept for non-UE consumers)
            "model": self._mjb, "format": self._model_format,
        }
        if self._model_format == "mjb":
            # bytes -> msgpack bin; UE reads it as base64 under `mjb__b64__`.
            reply["mjb"] = self._mjb
        else:
            # xml/mjz: FetchModel compiles this in-engine (version-independent).
            reply["xml"] = self._mjb.decode("utf-8", "replace")
            reply["vfs_assets"] = {
                f"{name}__b64__": base64.b64encode(data).decode("ascii")
                for name, data in self._assets.items()
            }
        return reply

    # -- registry ----------------------------------------------------------- #
    def _write_registry(self) -> None:
        """Write/refresh the discovery entry atomically."""
        entry = {
            "instance_id": self._instance_id,
            "role": FASTPATH_OWNER_CAP,
            "capabilities": sorted(self._caps),
            "pid": self._pid,
            "host": self._host,
            "scene": self._scene,
            "ngeom": self._ngeom,
            "control": self._control_endpoint,
            "control_port": self._control_port,
            "bus": self._bus_endpoint,
            "bus_port": self._bus_port,
            # Canonical port keys shared with the UE writer (InstanceRegistry.cpp)
            # and read by pool.read_registry (source-of-truth §12). A Python
            # owner's control REP is its step channel; it has no separate
            # full-state PUB or camera SHM ports, so those are 0 (pool.py falls
            # back to the index-derived port on a 0/absent value).
            "step_port": self._control_port,
            "state_port": 0,
            "cam_base_port": 0,
            # A fastpath owner always holds and steps a live model (it is its own
            # manager) and does not track a busy state.
            "manager_present": True,
            "busy": False,
            # Which transports viewers can reach this owner on. "zmq" is always up
            # (control REP + viewer PUB); "grpc" is added once start_grpc_server ran.
            "transports": ["zmq"] + (["grpc"] if getattr(self, "_grpc_endpoint", None) else []),
            "grpc": getattr(self, "_grpc_endpoint", None),
            # Timestamp key `registry_written_at` is shared with the UE writer and
            # read by discover_owners / pool.read_registry / MjDriverDiscovery.
            # int epoch here (pool.py only accepts a numeric value); the readers
            # normalize int-epoch vs the UE writer's ISO-8601 string.
            "registry_written_at": int(time.time()),
        }
        try:
            os.makedirs(self._registry_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._registry_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry, fh)
            os.replace(tmp, self._registry_path)
            self._last_registry_write = time.time()
        except OSError:
            pass  # discovery is best-effort; the sim keeps running regardless

    def _maybe_heartbeat(self, period_s: float = 10.0) -> None:
        if time.time() - self._last_registry_write >= period_s:
            self._write_registry()

    # -- control channel ---------------------------------------------------- #
    def serve_pending(self, heartbeat: bool = True) -> int:
        """Answer any queued control requests without blocking. Call once per
        step. Returns the number of requests served."""
        served = 0
        while dict(self._poller.poll(timeout=0)).get(self._rep) == zmq.POLLIN:
            try:
                raw = self._rep.recv()
            except zmq.ZMQError:
                break
            self._rep.send(self._handle_request(raw))
            served += 1
        if heartbeat:
            self._maybe_heartbeat()
        return served

    def _handle_request(self, raw: bytes) -> bytes:
        try:
            req = msgpack.unpackb(raw, raw=False)
        except Exception:  # noqa: BLE001
            return msgpack.packb({"error": "bad request"}, use_bin_type=True)
        op = req.get("op") if isinstance(req, dict) else None
        if op == "fastpath_hello":
            # One hello schema on every transport face (§11): declares
            # model_format so a renderer picks the right loader instead of
            # feeding xml/mjz source into mj_loadModelBuffer.
            return msgpack.packb(self.hello_reply(), use_bin_type=True)
        if op == "fastpath_perturb":
            # A renderer/viewer pushes an external force/torque on a body. Refused
            # unless the AcceptInput capability is granted (mirrors UE's
            # HandleFastpathPerturb capability check).
            if not self.accepts_input:
                return msgpack.packb(
                    {"ok": False, "error": "capability disabled: accept_input"},
                    use_bin_type=True)
            # Two shapes, both applied on the next step (apply_perturbations): an
            # interactive drag INTENT {select, active, localpos, refselpos} -> the real
            # mjv spring; or a raw wrench {body, force, torque} -> an exact force.
            # Vectors are padded to length 3 BEFORE use so a truncated wire vector
            # can't raise mid-handler and wedge the REP socket.
            try:
                if "refselpos" in req or "localpos" in req or "active" in req:
                    self.submit_perturb(
                        int(req.get("select", -1)),
                        bool(req.get("active", True)),
                        req.get("localpos", (0, 0, 0)),
                        req.get("refselpos", (0, 0, 0)))
                else:
                    self.submit_perturb_force(
                        int(req.get("body", -1)),
                        req.get("force", (0, 0, 0)),
                        req.get("torque", (0, 0, 0)))
                return msgpack.packb({"ok": True}, use_bin_type=True)
            except (TypeError, ValueError):
                return msgpack.packb({"error": "bad perturb"}, use_bin_type=True)
        return msgpack.packb(
            {"error": f"unknown op {op!r}"}, use_bin_type=True
        )

    def submit_perturb(self, select, active, localpos, refselpos) -> None:
        """Store the latest drag INTENT (thread-safe; called by the ZMQ REP AND an
        optional gRPC server). ``select`` is the body, ``localpos`` the grab point in
        the body's local MuJoCo frame, ``refselpos`` the drag target in the world
        frame (both metres, length-3, padded so a truncated wire vector can't raise).
        ``active=False`` releases the drag. The owner turns this into a force via
        :meth:`apply_perturbations` on its next step -- the real mjv spring."""
        self._perturb_intent = {
            "select": int(select),
            "active": bool(active),
            "localpos": _vec3(localpos),
            "refselpos": _vec3(refselpos),
        }

    def submit_perturb_force(self, body: int, force, torque) -> None:
        """Accumulate an EXACT body force/torque (thread-safe). For programmatic /
        legacy pushes that want a specific wrench, not a drag spring. Padded to
        length 3; ignored for a negative body id."""
        body = int(body)
        if body < 0:
            return
        f, t = _vec3(force), _vec3(torque)
        with self._lock:
            acc = self._perturb_force.setdefault(body, [0.0] * 6)
            for i in range(3):
                acc[i] += f[i]
                acc[i + 3] += t[i]

    def drain_perturbations(self) -> "dict[int, list[float]]":
        """Return the accumulated raw {body_id: 6-vector} wrenches and clear them.
        Apply to ``data.xfrc_applied`` before the next step. (Drag intents go through
        :meth:`apply_perturbations` instead, which also drains these.)"""
        with self._lock:
            perts = self._perturb_force
            self._perturb_force = {}
        return perts

    def apply_perturbations(self, model, data) -> None:
        """Write all pending perturbations to ``data.xfrc_applied`` -- call once per
        step, AFTER zeroing xfrc_applied and BEFORE ``mj_step``. Applies raw wrench
        pushes AND the interactive drag intent. The drag is exactly what ``simulate``
        does for a Ctrl-drag: a mass-scaled, critically-damped spring
        (``mjv_applyPerturbForce``), so the body settles on the target instead of
        flying off. localmass is (re)computed via ``mjv_initPerturb`` once per grab."""
        import mujoco  # noqa: PLC0415

        # Raw wrench pushes first (exact forces), then the drag spring on top.
        for body, wrench in self.drain_perturbations().items():
            if 0 <= body < model.nbody:
                data.xfrc_applied[body] = wrench

        with self._lock:
            intent = self._perturb_intent
        sel = int(intent["select"]) if intent else -1
        active = bool(intent["active"]) if intent else False

        if not active or sel <= 0 or sel >= model.nbody:
            # Released / invalid: clear the last-driven body so it stops drifting.
            if self._pert_sel is not None and 0 < self._pert_sel < model.nbody:
                data.xfrc_applied[self._pert_sel] = 0.0
            self._pert_sel = None
            return

        # Lazily build the perturb struct + a throwaway scene with a valid frustum
        # (mjv_initPerturb only touches the scene for pert.scale, which we don't use,
        # but it faults on a zero frustum).
        if self._pert is None:
            self._pert = mujoco.MjvPerturb()
            self._pert_scene = mujoco.MjvScene(model, 0)
            for cam in (self._pert_scene.camera[0], self._pert_scene.camera[1]):
                cam.frustum_near = 0.1
                cam.frustum_far = 100.0
                cam.frustum_bottom, cam.frustum_top = -0.1, 0.1
                cam.frustum_center, cam.frustum_width = 0.0, 0.1
                cam.pos[:] = [0.0, -2.0, 1.0]
                cam.forward[:] = [0.0, 1.0, 0.0]
                cam.up[:] = [0.0, 0.0, 1.0]

        pert = self._pert
        # New grab: set the anchor + (re)compute localmass for this body/point.
        if self._pert_sel != sel:
            pert.select = sel
            pert.localpos[:] = intent["localpos"]
            mujoco.mjv_initPerturb(model, data, self._pert_scene, pert)
            pert.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
            self._pert_sel = sel

        pert.localpos[:] = intent["localpos"]
        pert.refselpos[:] = intent["refselpos"]
        mujoco.mjv_applyPerturbForce(model, data, pert)

    def latest_transforms(self) -> "Optional[dict]":
        """The most recent per-body transform frame, augmented with the debug tier
        for the AGGREGATE of currently-active render subscribers (none -> lean). A
        diagnostic view of the shared cache; each gRPC stream builds its own frame
        via :meth:`render_frame_for_caps`. Thread-safe (returns a fresh copy)."""
        with self._lock:
            caps = dict(self._render_debug_caps)
        return self.render_frame_for_caps(caps)

    @staticmethod
    def _caps_key(caps) -> "Optional[tuple]":
        """Canonical hashable key for a parsed caps dict, used to index the
        per-publish frame cache. Lean (no contacts AND no overlay) collapses to
        ``None`` so every lean subscriber -- and the always-on ZMQ topic -- share
        the single cached lean frame. Two subscribers negotiating the same debug
        tier collapse to the same key, so it is built (and cached) exactly once."""
        if not caps or not (caps.get("contacts") or caps.get("overlay")):
            return None
        return (
            bool(caps.get("contacts")),
            bool(caps.get("overlay")),
            int(caps.get("max_contacts") or 0),
        )

    def render_frame_for_caps(self, caps=None) -> "Optional[dict]":
        """Return a fresh copy of the frame this caller's ``caps`` should receive
        for the latest publish. In the common case this is a CHEAP DICT COPY of a
        frame already built once per publish in :meth:`_send_transforms` (lean under
        the ``None`` key, or the pre-built debug frame whose caps match) -- the poll
        path does NOT rebuild the debug tier, so a ~200 Hz gRPC stream never holds
        the GIL doing per-poll MuJoCo work and never starves the owner's step loop.

        Returns None before the first publish. Two subscribers with different caps
        get different cached frames from the same publish, and the shared lean base
        is never mutated. If a caller's exact caps-set was not pre-built for this
        publish (rare: a subset differing from every active subscriber, e.g. the
        first poll of a just-registered subscriber before the next publish), this
        falls back to building the debug tier once from the latest snapshot."""
        parsed = self.parse_render_debug_caps(caps) if caps is not None else None
        key = self._caps_key(parsed)
        with self._lock:
            if self._latest_lean is None:
                return None
            cached = self._latest_frames.get(key)
            if cached is not None:
                return dict(cached)  # common path: re-send a pre-built frame
            # Fallback (rare): caps not pre-built for this publish. Build once from
            # the latest snapshot; the next publish will cache this caps-set.
            frame = dict(self._latest_lean)
            model = self._latest_model
            data = self._latest_data
        self._append_render_debug_fields(frame, model, data, parsed)
        return frame

    # -- per-subscription debug caps registry ------------------------------- #
    def _recompute_aggregate_caps_locked(self) -> None:
        """Recompute `_render_debug_caps` as the union of all active subscribers'
        caps (call with `self._lock` held). With no subscribers this resets to
        lean, so the shared cache stops paying for the debug tier the instant the
        last debug subscriber disconnects."""
        contacts = overlay = False
        maxc = 0
        unlimited = False
        for c in self._render_subscribers.values():
            contacts = contacts or bool(c.get("contacts"))
            overlay = overlay or bool(c.get("overlay"))
            mc = int(c.get("max_contacts") or 0)
            if mc == 0:
                unlimited = True  # 0 == "no cap" -> the widest request wins
            else:
                maxc = max(maxc, mc)
        self._render_debug_caps = {
            "contacts": contacts,
            "overlay": overlay,
            "max_contacts": 0 if unlimited else maxc,
        }

    def _active_debug_caps_sets_locked(self) -> "dict":
        """The DISTINCT non-lean caps-sets a publish must pre-build a debug frame
        for (call with `self._lock` held): every active render subscriber's caps,
        plus the directly-set aggregate (`_render_debug_caps`) so callers that set
        caps without registering a subscriber -- e.g. `set_render_debug_caps`, used
        by `latest_transforms()` -- also get a cached frame. Keyed by `_caps_key`,
        so identical caps-sets collapse to one build. Usually 0-2 entries."""
        sets: "dict" = {}
        for caps in self._render_subscribers.values():
            key = self._caps_key(caps)
            if key is not None:
                sets[key] = caps
        key = self._caps_key(self._render_debug_caps)
        if key is not None:
            sets[key] = dict(self._render_debug_caps)
        return sets

    def register_render_subscriber(self, caps) -> int:
        """Register an active render subscription with its negotiated caps; returns
        a token to pass to :meth:`unregister_render_subscriber` when it ends. The
        caps are per-subscription (used by that stream alone); registration only
        keeps the aggregate view (`latest_transforms`) in sync."""
        caps = self.parse_render_debug_caps(caps)
        with self._lock:
            token = self._render_sub_next
            self._render_sub_next += 1
            self._render_subscribers[token] = caps
            self._recompute_aggregate_caps_locked()
        return token

    def unregister_render_subscriber(self, token: int) -> None:
        """Drop a subscription (on stream end/disconnect) and recompute the
        aggregate caps, so a disconnect never leaves the debug tier enabled for
        the shared cache once nobody is asking for it."""
        with self._lock:
            self._render_subscribers.pop(token, None)
            self._recompute_aggregate_caps_locked()

    # -- transform bus ------------------------------------------------------ #
    def _build_render_frame(self, payload: dict, cxpos, cxquat, usercam=None) -> dict:
        """Assemble the always-present render tier (source-of-truth §8.1): the
        per-body/per-geom transforms already in ``payload``, plus the optional
        per-camera and operator free-camera pose. Returns the frame dict; debug
        fields (§8.2) are appended separately by :meth:`_append_render_debug_fields`.

        ``usercam``, when given, is a ``(pos, fwd, up)`` triple of MuJoCo-world
        3-vectors for the operator's free/user camera; a render slave in "copycat"
        mode points its viewport at it.
        """
        if cxpos is not None and cxquat is not None:
            payload["cxpos"] = list(cxpos)
            payload["cxquat"] = list(cxquat)
        if usercam is not None:
            pos, fwd, up = usercam
            payload["ucpos"] = [float(v) for v in pos]
            payload["ucfwd"] = [float(v) for v in fwd]
            payload["ucup"] = [float(v) for v in up]
        return payload

    @staticmethod
    def parse_render_debug_caps(req) -> dict:
        """Parse the debug-tier subscription capabilities off a subscribe request
        (source-of-truth §8.2): ``contacts`` (bool) toggles ``StreamContacts``,
        ``overlay`` (bool) toggles ``StreamOverlay``, and ``maxcontacts`` (int,
        alias ``max_contacts``) sets the contact cap. Absent/false keys leave the
        cap off, so a lean mirror that asks for nothing gets zero extra bytes."""
        req = req or {}
        try:
            maxc = int(req.get("maxcontacts", req.get("max_contacts", 0)) or 0)
        except (TypeError, ValueError):
            maxc = 0
        return {
            "contacts": bool(req.get("contacts", False)),
            "overlay": bool(req.get("overlay", False)),
            "max_contacts": max(0, maxc),
        }

    def set_render_debug_caps(self, caps) -> None:
        """Record the debug-tier caps a render subscriber negotiated. Gates what
        :meth:`_append_render_debug_fields` serializes onto every published frame;
        defaults to none so a lean mirror pays zero extra bytes."""
        with self._lock:
            self._render_debug_caps = self.parse_render_debug_caps(caps)

    def _append_render_debug_fields(self, frame: dict, model=None, data=None,
                                    caps=None) -> None:
        """Capability-gated, count-capped debug tier (source-of-truth §8.2) computed
        straight from ``model``/``data`` after the step. A subscriber that requests
        neither ``StreamContacts`` nor ``StreamOverlay`` pays zero extra bytes, so
        this returns immediately when no cap is set (or when no mjData is at hand).

        Serializes ``contacts`` (capped at ``caps['max_contacts']``) under
        ``StreamContacts``, and the derived-decor bundle (``xfrc_applied`` /
        ``subtree_com`` / ``ctrl`` / ``act`` / ``wrap_xpos`` (+ ``wrap_obj`` /
        ``ten_wrapadr`` / ``ten_wrapnum``) / ``eq_active`` + ``eq_anchor`` /
        ``sensordata`` / light ``light_xpos``+``light_xdir``) under
        ``StreamOverlay``.
        """
        if not caps or not (caps.get("contacts") or caps.get("overlay")):
            return
        if model is None or data is None:
            return
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        def flat(a):
            return [float(x) for x in np.asarray(a, dtype=float).ravel()]

        def flat_i(a):
            return [int(x) for x in np.asarray(a).ravel()]

        # --- StreamContacts: the contact list, capped at the subscriber's cap ---
        # contacts[] = [{pos[3], frame[9], dist, force[6], dim, g1, g2}]; force via
        # mj_contactForce (force:torque in the contact frame). Truncated to
        # caps['max_contacts'], mirroring MuJoCo's scn->maxgeom bound.
        if caps.get("contacts"):
            ncon = int(data.ncon)
            cap = int(caps.get("max_contacts") or 0)
            n = min(ncon, cap) if cap > 0 else ncon
            contacts = []
            force = np.zeros(6, dtype=float)
            for c in range(n):
                con = data.contact[c]
                mujoco.mj_contactForce(model, data, c, force)
                contacts.append({
                    "pos": flat(con.pos),
                    "frame": flat(con.frame),
                    "dist": float(con.dist),
                    "force": [float(x) for x in force],
                    "dim": int(con.dim),
                    "g1": int(con.geom[0]),
                    "g2": int(con.geom[1]),
                })
            frame["contacts"] = contacts

        # --- StreamOverlay: the derived-decor bundle, each array capped by its ----
        # natural model dimension and appended only when that dimension is non-zero.
        if caps.get("overlay"):
            nbody = int(model.nbody)
            if nbody > 0:
                # Perturbation / external-force arrows (computed every step anyway).
                frame["xfrc_applied"] = flat(data.xfrc_applied)   # 6*nbody
                frame["subtree_com"] = flat(data.subtree_com)     # 3*nbody, CoM spheres
            if int(model.nu) > 0:
                frame["ctrl"] = flat(data.ctrl)                   # actuator coloring
            if int(model.na) > 0:
                frame["act"] = flat(data.act)
            # Tendon wrap paths + slicing arrays (segment per tendon consumer-side).
            if int(model.nwrap) > 0:
                frame["wrap_xpos"] = flat(data.wrap_xpos)         # 6*nwrap
                frame["wrap_obj"] = flat_i(data.wrap_obj)         # 2*nwrap
            if int(model.ntendon) > 0:
                frame["ten_wrapadr"] = flat_i(data.ten_wrapadr)
                frame["ten_wrapnum"] = flat_i(data.ten_wrapnum)
            # Equality-constraint decor: eq_active is the only DYNAMIC quantity a
            # mirror can't reconstruct (it has eq_type/eq_obj*/eq_data statically).
            # eq_anchor carries the two world anchor endpoints per equality
            # (body-origin convention; world body -> 0), ready to draw.
            neq = int(model.neq)
            if neq > 0:
                xpos = np.asarray(data.xpos, dtype=float).reshape(-1, 3)
                eq_anchor = []
                for e in range(neq):
                    b1 = int(model.eq_obj1id[e])
                    b2 = int(model.eq_obj2id[e])
                    p1 = xpos[b1] if 0 < b1 < nbody else np.zeros(3)
                    p2 = xpos[b2] if 0 < b2 < nbody else np.zeros(3)
                    eq_anchor.extend(float(x) for x in p1)
                    eq_anchor.extend(float(x) for x in p2)
                frame["eq_active"] = flat_i(data.eq_active)        # neq
                frame["eq_anchor"] = eq_anchor                    # 6*neq
            # Rangefinder / sensor decor.
            if int(model.nsensordata) > 0:
                frame["sensordata"] = flat(data.sensordata)
            # Light glyphs (cameras already ride cxpos/cxquat §8.1; lights do not).
            if int(model.nlight) > 0:
                frame["light_xpos"] = flat(data.light_xpos)       # 3*nlight
                frame["light_xdir"] = flat(data.light_xdir)       # 3*nlight

    def _send_transforms(self, payload: dict, cxpos, cxquat, usercam=None,
                         model=None, data=None) -> None:
        """Build one LEAN render-tier frame and publish it on the ``render`` topic.
        Best-effort: a slow/absent renderer never stalls the sim.

        ``model``/``data``, when given (via :meth:`publish_mjdata`), are cached
        alongside the lean frame so a gRPC subscriber can compute the
        capability-gated debug tier (§8.2) for its OWN subscription later
        (:meth:`render_frame_for_caps`). The debug tier is never appended here, so
        the ZMQ topic and a lean mirror always pay zero extra bytes."""
        frame = self._build_render_frame(payload, cxpos, cxquat, usercam)
        # The frame is LEAN (transforms only). The debug tier (§8.2) is NEVER
        # appended here: it is per-subscription and can only be negotiated over
        # gRPC, so each gRPC stream appends its own via `render_frame_for_caps`.
        # This keeps the always-on ZMQ `render` topic -- which has no way to ask
        # for the debug tier -- paying zero extra bytes regardless of any gRPC
        # subscriber, and lets a lean gRPC mirror do the same.
        #
        # Cache the lean base + the (model, data) behind it, so a gRPC stream can
        # compute the debug tier for its OWN caps off the latest post-step state.
        lean = dict(frame)
        # Snapshot which distinct debug caps-sets are currently wanted (subscribers
        # + directly-set aggregate). Do the (potentially expensive) debug-tier build
        # ONCE PER PUBLISH here on the main thread, off THIS publish's (model, data)
        # -- so every cached debug frame is consistent with the transforms, and the
        # gRPC stream thread only ever re-sends a cached frame (never rebuilds the
        # debug tier on its ~200 Hz poll). Building outside the lock keeps the lock
        # hold to the cheap dict swap below; model/data are this publish's args, not
        # shared owner state, so no lock is needed to read them here.
        with self._lock:
            caps_sets = self._active_debug_caps_sets_locked()
        frames: "dict" = {None: lean}
        for key, caps in caps_sets.items():
            dbg = dict(lean)
            self._append_render_debug_fields(dbg, model, data, caps)
            frames[key] = dbg
        with self._lock:
            self._latest_lean = lean
            self._latest_model = model
            self._latest_data = data
            self._latest_frames = frames
        # Publish the LEAN frame on the ZMQ bus (best-effort; a slow/absent
        # renderer never stalls the sim).
        try:
            self._pub.send_multipart(
                [b"render", msgpack.packb(frame, use_bin_type=True)],
                flags=zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            pass

    def publish_bodies(self, frame: int, bxpos, bxquat, cxpos=None, cxquat=None,
                       usercam=None) -> None:
        """Publish one per-BODY transform frame on the ``render`` topic.

        bxpos is a flat length-3*nbody sequence, bxquat length-4*nbody (wxyz) --
        i.e. ``data.xpos`` / ``data.xquat``. The renderer composes each geom's world
        pose from its body transform and its body-relative offset (from the model),
        so the wire carries nbody transforms instead of ngeom, and mocap bodies are
        covered for free. cxpos/cxquat, when given, are the per-camera world
        transforms (3*ncam, 4*ncam wxyz). usercam, when given, is the operator's
        free-camera (pos, fwd, up) for a copycat render slave.
        """
        self._send_transforms(
            {"f": int(frame), "bxpos": list(bxpos), "bxquat": list(bxquat)},
            cxpos, cxquat, usercam,
        )

    def publish_mjdata(self, frame: int, model, data, usercam=None,
                       refresh_kinematics: bool = True) -> None:
        """Publish transforms straight from an MjModel/MjData pair, re-running FK
        after stepping so transforms reflect post-integration qpos."""
        from .render_client import poses_from_mjdata  # lazy import

        poses = poses_from_mjdata(model, data, refresh_kinematics=refresh_kinematics)
        # Route (model, data) through so the capability-gated debug tier (§8.2) can
        # be computed post-step; publish_bodies has no mjData so it stays transforms
        # only.
        self._send_transforms(
            {"f": int(frame), "bxpos": list(poses["bxpos"]),
             "bxquat": list(poses["bxquat"])},
            poses["cxpos"], poses["cxquat"], usercam, model=model, data=data,
        )

    # -- optional gRPC face ------------------------------------------------- #
    def start_grpc_server(self, port: int = 50051, bind: str = "0.0.0.0") -> str:
        """Serve this owner over gRPC too (mirrors subscribe + perturb over gRPC,
        not just ZMQ). Returns the ``host:port`` it listens on. Idempotent."""
        if self._grpc_server is not None:
            return self._grpc_endpoint
        from .owner_server import OwnerGrpcServer

        self._grpc_server = OwnerGrpcServer(self, port=port, bind=bind)
        self._grpc_server.start()
        # Advertise the gRPC endpoint in the registry for discovery. This is
        # the dialable host:port (resolved advertise_host), not the server's
        # own `.endpoint` (built from `bind`, e.g. "0.0.0.0:port") — callers
        # dial the return value of this method, so it must match what the
        # registry/`grpc_endpoint` property advertise.
        self._grpc_endpoint = f"{self._host}:{port}"
        self._write_registry()
        return self._grpc_endpoint

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        if self._grpc_server is not None:
            try:
                self._grpc_server.stop()
            except Exception:  # noqa: BLE001
                pass
            self._grpc_server = None
        try:
            os.remove(self._registry_path)
        except OSError:
            pass
        for sock in (self._rep, self._pub):
            try:
                sock.close(0)
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "FastPathOwner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
