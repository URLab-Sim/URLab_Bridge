"""Fast-path owner: advertise a live sim to UE fast-path renderers.

A fast-path *owner* is any process that holds a MuJoCo model and steps it (a
puppet script, or a UE live/direct instance). It offers two things to a
renderer:

* a **control channel** (ZMQ REQ/REP) that answers ``fastpath_hello`` with the
  model's compiled MJB bytes and the transform-bus endpoint, so a renderer can
  connect with no shared file, and
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
        self._latest_state: "Optional[tuple[float, list, list]]" = None
        # Latest per-body transform frame ({f,bxpos,bxquat,cxpos?,cxquat?}) cached
        # for a gRPC subscribe(format=render) stream -- the true mirror payload
        # (viewer runs zero MuJoCo). Written by every publish_bodies/publish_mjdata.
        self._latest_transforms: "Optional[dict]" = None
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
            # Which transports viewers can reach this owner on. "zmq" is always up
            # (control REP + viewer PUB); "grpc" is added once start_grpc_server ran.
            "transports": ["zmq"] + (["grpc"] if getattr(self, "_grpc_endpoint", None) else []),
            "grpc": getattr(self, "_grpc_endpoint", None),
            # int epoch so both the UE reader and pool.py parse it (pool.py only
            # accepts a numeric registry_written_at).
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
            reply = {
                "ok": True,
                "scene": self._scene,
                "ngeom": self._ngeom,
                "bus": self._bus_endpoint,
                # bytes -> msgpack bin; UE reads it as base64 under `mjb__b64__`.
                "mjb": self._mjb,
            }
            return msgpack.packb(reply, use_bin_type=True)
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

    def latest_state(self) -> "Optional[tuple[float, list, list]]":
        """The most recent ``(t, qpos, qvel)`` given to :meth:`publish_state`, or
        None. Read by a gRPC subscribe(format=qpos) stream; thread-safe."""
        with self._lock:
            return self._latest_state

    def latest_transforms(self) -> "Optional[dict]":
        """The most recent per-body transform frame ({f,bxpos,bxquat,cxpos?,cxquat?})
        published via :meth:`publish_bodies`/:meth:`publish_mjdata`, or None. Read by
        a gRPC subscribe(format=render) stream; thread-safe (returns a copy)."""
        with self._lock:
            return dict(self._latest_transforms) if self._latest_transforms else None

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

    def _append_render_debug_fields(self, frame: dict, caps=None) -> None:
        """Capability-gated, count-capped seam (source-of-truth §8.2) that appends
        the optional debug tier onto a render frame. A subscriber that requests
        neither ``StreamContacts`` nor ``StreamOverlay`` pays zero extra bytes, so
        this returns immediately when no cap is set.

        Phase 2.3 establishes the hook only; Phase 9.1 computes + serializes the
        §8.2 debug arrays here -- ``contacts`` (capped at ``caps['max_contacts']``)
        when ``StreamContacts``, and the derived-decor bundle (``xfrc_applied`` /
        ``subtree_com`` / ``ctrl`` / ``act`` / ``wrap_xpos`` / ``eq`` / ``sensor``
        / light glyphs) when ``StreamOverlay``.
        """
        if not caps or not (caps.get("contacts") or caps.get("overlay")):
            return
        # SEAM (Phase 9.1): populate the §8.2 debug arrays on ``frame`` here.

    def _send_transforms(self, payload: dict, cxpos, cxquat, usercam=None) -> None:
        """Build one render-tier frame and publish it on the ``render`` topic.
        Best-effort: a slow/absent renderer never stalls the sim."""
        frame = self._build_render_frame(payload, cxpos, cxquat, usercam)
        # Debug tier carries no extra bytes over the ZMQ bus: the topic-per-tier bus
        # advertises the render tier only; per-subscription debug caps are negotiated
        # on the gRPC selector (wired in 9.1). Pass no caps so the seam is a no-op.
        self._append_render_debug_fields(frame, None)
        # Cache the fully-assembled frame for a gRPC subscribe(format=render) stream,
        # so a gRPC mirror gets byte-identical frames to a ZMQ mirror. The ZMQ
        # publish below stays independent + best-effort.
        with self._lock:
            self._latest_transforms = dict(frame)
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
        self.publish_bodies(
            frame, poses["bxpos"], poses["bxquat"],
            cxpos=poses["cxpos"], cxquat=poses["cxquat"], usercam=usercam,
        )

    def publish_geoms(self, frame: int, xpos, xquat, cxpos=None, cxquat=None) -> None:
        """Publish one per-geom transform frame on the ``render`` topic.

        xpos is a flat length-3*ngeom sequence, xquat length-4*ngeom (wxyz).
        cxpos/cxquat, when given, are the per-camera world transforms (3*ncam and
        4*ncam wxyz) so streamed cameras track moving bodies.
        """
        self._send_transforms(
            {"f": int(frame), "xpos": list(xpos), "xquat": list(xquat)}, cxpos, cxquat
        )

    def publish_state(self, t: float, qpos, qvel) -> None:
        """Broadcast raw kinematics ``{t, qpos, qvel}`` on the ``viewer`` topic for
        read-only viewers -- a mujoco/pystudio peek window or a UE viewer instance.

        This is the smooth async channel, separate from the eval render path, and
        uses the exact wire format UE's ViewerSubscribeTransport consumes, so the
        same bus feeds a Python peek and a UE viewer alike. Best-effort: a slow or
        absent subscriber never stalls the sim."""
        qpos_l = [float(x) for x in qpos]
        qvel_l = [float(x) for x in qvel]
        with self._lock:
            self._latest_state = (float(t), qpos_l, qvel_l)  # for the gRPC stream
        payload = {"t": float(t), "qpos": qpos_l, "qvel": qvel_l}
        try:
            self._pub.send_multipart(
                [b"viewer", msgpack.packb(payload, use_bin_type=True)],
                flags=zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            pass

    # -- optional gRPC face ------------------------------------------------- #
    def start_grpc_server(self, port: int = 50051, bind: str = "0.0.0.0") -> str:
        """Serve this owner over gRPC too (viewers subscribe + perturb over gRPC,
        not just ZMQ). Returns the ``host:port`` it listens on. Idempotent."""
        if self._grpc_server is not None:
            return self._grpc_server.endpoint
        from .owner_server import OwnerGrpcServer

        self._grpc_server = OwnerGrpcServer(self, port=port, bind=bind)
        self._grpc_server.start()
        # Advertise the gRPC endpoint in the registry for discovery.
        self._grpc_endpoint = f"{self._host}:{port}"
        self._write_registry()
        return self._grpc_server.endpoint

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
