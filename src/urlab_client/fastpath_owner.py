"""Fast-path owner: advertise a live sim to UE fast-path renderers.

A fast-path *owner* is any process that holds a MuJoCo model and steps it (a
puppet script, or a UE live/direct instance). It offers two things to a
renderer:

* a **control channel** (ZMQ REQ/REP) that answers ``fastpath_hello`` with the
  model's compiled MJB bytes and the transform-bus endpoint, so a renderer can
  connect with no shared file, and
* the **geoms transform bus** (ZMQ PUB, topic ``geoms``) that carries per-geom
  world transforms every step.

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
import time
from typing import Optional

import msgpack
import zmq

from .pool import default_registry_dir

# Capability string a fast-path owner advertises; the UE server browser filters
# on it so it lists owners a renderer can actually consume.
FASTPATH_OWNER_CAP = "fastpath_owner"


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
        instance_id: Optional[str] = None,
        registry_dir: Optional[str] = None,
    ) -> None:
        self._mjb = bytes(mjb_bytes)
        self._scene = scene
        self._ngeom = int(ngeom)
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
        self._write_registry()

    # -- properties --------------------------------------------------------- #
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
            "capabilities": [FASTPATH_OWNER_CAP],
            "pid": self._pid,
            "host": self._host,
            "scene": self._scene,
            "ngeom": self._ngeom,
            "control": self._control_endpoint,
            "control_port": self._control_port,
            "bus": self._bus_endpoint,
            "bus_port": self._bus_port,
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
        return msgpack.packb(
            {"error": f"unknown op {op!r}"}, use_bin_type=True
        )

    # -- transform bus ------------------------------------------------------ #
    def publish_geoms(self, frame: int, xpos, xquat) -> None:
        """Publish one per-geom transform frame on the ``geoms`` topic.

        xpos is a flat length-3*ngeom sequence, xquat length-4*ngeom (wxyz).
        """
        payload = {"f": int(frame), "xpos": list(xpos), "xquat": list(xquat)}
        try:
            self._pub.send_multipart(
                [b"geoms", msgpack.packb(payload, use_bin_type=True)],
                flags=zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            pass  # best-effort; a slow/absent renderer never stalls the sim

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
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
