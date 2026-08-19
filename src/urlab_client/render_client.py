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
"""High-level client for the fast-path render server (the "mirror").

Wraps the ``fastpath_*`` bridge ops so callers never hand-roll op dicts. The render
server holds a MuJoCo model and renders its cameras; the client steps physics itself
(as a *puppet*), pushes the resulting body/camera poses, and gets camera frames back:

* ``delay == 0`` -- block until the *exact* pushed state is freshly rendered.
* ``delay  > 0`` -- return an ``N``-substep-stale frame from the server's ring, which
  is faster (no fresh-render wait) and matches real-camera latency for training data.

Typical use::

    import mujoco
    from urlab_client import RenderClient

    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    with RenderClient("tcp://127.0.0.1", step_port=5559) as rc:
        for _ in range(100):
            mujoco.mj_step(model, data)
            frames = rc.render_mjdata(model, data, cameras=["cam0"])   # fresh
            bgr = frames["cam0"].to_bgr()                              # HxWx3 for cv2

The server is launched separately (packaged exe or editor); see docs/render_server.md.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .transports import Transport, make_transport

__all__ = ["RenderClient", "CameraFrame", "RenderError", "poses_from_mjdata"]


class RenderError(RuntimeError):
    """The render server returned an error reply (not a transport failure)."""


@dataclass
class CameraFrame:
    """One rendered camera image and its metadata."""

    name: str
    width: int
    height: int
    dtype: str  # "bgra8" (color) or "float32" (depth)
    frame_id: int
    sim_time: float
    data: bytes

    def to_array(self) -> np.ndarray:
        """Raw pixels: ``HxWx4`` uint8 BGRA, or ``HxW`` float32 for depth."""
        if self.dtype == "float32":
            return np.frombuffer(self.data, np.float32).reshape(self.height, self.width)
        return np.frombuffer(self.data, np.uint8).reshape(self.height, self.width, 4)

    def to_bgr(self) -> np.ndarray:
        """``HxWx3`` uint8 BGR, ready for ``cv2.imshow`` / ``cv2.imwrite``."""
        return np.ascontiguousarray(self.to_array()[:, :, :3])

    def to_rgb(self) -> np.ndarray:
        """``HxWx3`` uint8 RGB (e.g. for matplotlib / PIL)."""
        return np.ascontiguousarray(self.to_array()[:, :, 2::-1])


def poses_from_mjdata(model, data) -> Dict[str, np.ndarray]:
    """Body + camera world transforms from a mujoco ``MjData``, ready for ``render()``.

    Returns ``bxpos``/``bxquat`` (per-body) and ``cxpos``/``cxquat`` (per-camera), the
    exact fields the server applies to its mirror before rendering.
    """
    import mujoco  # lazy: the client core does not require mujoco

    ncam = int(model.ncam)
    cxmat = np.asarray(data.cam_xmat, np.float64).reshape(ncam, 9)
    cxquat = np.empty(ncam * 4, np.float64)
    q = np.zeros(4)
    for c in range(ncam):
        mujoco.mju_mat2Quat(q, cxmat[c])
        cxquat[4 * c:4 * c + 4] = q
    return {
        "bxpos": np.asarray(data.xpos, np.float64).reshape(-1),
        "bxquat": np.asarray(data.xquat, np.float64).reshape(-1),
        "cxpos": np.asarray(data.cam_xpos, np.float64).reshape(-1),
        "cxquat": cxquat,
    }


class RenderClient:
    """Drives a running fast-path render server as a puppet.

    Parameters
    ----------
    address:
        ZMQ address of the server, e.g. ``"tcp://127.0.0.1"``.
    step_port:
        Bridge step/RPC port the server listens on (default 5559).
    transport:
        ``"zmq"`` (any host) or ``"shm"`` (co-located, lower latency).
    recv_timeout_ms:
        Default reply timeout; per-call overrides are derived from ``timeout_ms``.
    """

    def __init__(
        self,
        address: str = "tcp://127.0.0.1",
        *,
        step_port: int = 5559,
        transport: str = "zmq",
        recv_timeout_ms: int = 5000,
    ) -> None:
        self._t: Transport = make_transport(
            transport, address=address, step_port=step_port,
            recv_timeout_ms=recv_timeout_ms,
        )
        self._frame = 0

    # -- model lifecycle ---------------------------------------------------
    def load_model(
        self,
        data: "bytes | str",
        *,
        format: str = "mjb",  # noqa: A002 - wire field name
        assets: Optional[Mapping[str, "bytes | str"]] = None,
        timeout_ms: int = 120_000,
    ) -> None:
        """Hot-swap the server's model from any supported format.

        ``format`` is ``"mjb"`` | ``"xml"`` | ``"mjz"``:

        * ``mjb`` -- a compiled model, loaded directly (must be version-matched to
          the server's MuJoCo; use ``mjbcompile``).
        * ``xml`` -- MJCF source; the server compiles it with its OWN libmujoco, so
          it is immune to MJB version skew. ``assets`` supplies the referenced mesh/
          texture files (see :meth:`load_xml`, which gathers them for you).
        * ``mjz`` -- a self-contained MuJoCo archive, decoded by the server's spec
          decoder registry (``assets`` mounts any extras alongside).

        The server (``MjModelSource::FromBytes``) normalises whatever arrives back to
        an MJB with its own libmujoco, then swaps it into the live renderer.

        ``data`` and each ``assets`` value may be raw bytes or a file path. Assets are
        keyed by the **bare filename** the model references them under.
        """
        blob = data if isinstance(data, (bytes, bytearray)) else open(data, "rb").read()
        req: Dict[str, object] = {
            "op": "fastpath_load", "format": str(format), "model": bytes(blob),
        }
        if assets:
            # Asset bytes ride as base64 STRINGS (the server reads each asset value
            # as a base64 string). Unlike the top-level `model` bin -- which the
            # transport's msgpack-bin -> `__b64__` convention handles -- a nested bin
            # value would arrive under a `<name>__b64__` key and mount under the wrong
            # filename, so encode these ourselves and keep the bare-name key intact.
            enc: Dict[str, str] = {}
            for name, val in assets.items():
                raw = val if isinstance(val, (bytes, bytearray)) else open(val, "rb").read()
                enc[name] = base64.b64encode(bytes(raw)).decode("ascii")
            req["assets"] = enc
        rep = self._t.rpc(req, recv_timeout_ms=timeout_ms)
        _check(rep, "fastpath_load")

    def load_mjb(self, mjb: "bytes | str", *, timeout_ms: int = 120_000) -> None:
        """Hot-swap the server's model with a compiled MJB (raw bytes or a file path).

        Optional -- a server launched with ``-URLabFastMjb=<file>`` already has one.
        The MJB must be version-matched to the server's MuJoCo (use ``mjbcompile``).
        """
        self.load_model(mjb, format="mjb", timeout_ms=timeout_ms)

    def load_xml(
        self,
        xml: "str | bytes",
        *,
        asset_root: Optional[str] = None,
        assets: Optional[Mapping[str, "bytes | str"]] = None,
        timeout_ms: int = 120_000,
    ) -> None:
        """Hot-swap the server's model from MJCF source + its assets.

        ``xml`` is a path, an XML string, or an XML ``bytes`` blob. Every
        ``<include>`` is inlined and each ``file=`` rewritten to a bare filename;
        the referenced mesh/texture files are read from disk (relative to the XML,
        or ``asset_root``) and shipped alongside. Pass ``assets`` (bare name -> bytes/
        path) to supply any that aren't on disk; those take precedence.

        The server compiles the XML with its own libmujoco, so there is no MJB
        version-matching requirement.
        """
        from ._model_upload import flatten_model  # lazy: stdlib-only helper

        xml_text, asset_paths = flatten_model(xml, asset_root=asset_root)
        blobs: Dict[str, "bytes | str"] = {}
        for name, path in asset_paths.items():
            blobs[name] = path  # load_model reads paths itself
        if assets:  # caller-supplied bytes/paths override on-disk lookups
            blobs.update(dict(assets))
        self.load_model(xml_text.encode("utf-8"), format="xml", assets=blobs,
                        timeout_ms=timeout_ms)

    def load_mjz(
        self,
        mjz: "bytes | str",
        *,
        assets: Optional[Mapping[str, "bytes | str"]] = None,
        timeout_ms: int = 120_000,
    ) -> None:
        """Hot-swap the server's model from a MuJoCo ``.mjz`` archive (raw bytes or
        a file path).

        A ``.mjz`` is a zip of one MJCF plus its assets. The compact archive goes
        straight over the wire; the server decodes it with MuJoCo's own C decoder
        (``mj_parse`` into a VFS, then ``mj_compile`` against it) and compiles with
        its own libmujoco. The archive's root model must be named ``model.xml`` (the
        canonical layout MuJoCo's decoder expects). ``assets`` mounts extra files
        alongside the archive's own.
        """
        self.load_model(mjz, format="mjz", assets=assets, timeout_ms=timeout_ms)

    # -- rendering ---------------------------------------------------------
    def render(
        self,
        *,
        bxpos: Sequence[float],
        bxquat: Sequence[float],
        cxpos: Optional[Sequence[float]] = None,
        cxquat: Optional[Sequence[float]] = None,
        sim_time: float = 0.0,
        cameras: Optional[Sequence[str]] = None,
        delay: int = 0,
        timeout_ms: int = 2000,
    ) -> Dict[str, CameraFrame]:
        """Push a pose set and return ``{camera_name: CameraFrame}``.

        ``cameras`` selects which cameras to render (None = all; naming one is far
        cheaper -- each camera is a full scene capture). ``delay`` in substeps: 0 =
        fresh/blocking, >0 = stale-from-ring (faster, real-camera-latency emulation).
        """
        req: Dict[str, object] = {
            "op": "fastpath_render", "f": self._frame, "frame_id": self._frame,
            "sim_time": float(sim_time),
            "bxpos": _tolist(bxpos), "bxquat": _tolist(bxquat),
            "timeout_ms": int(timeout_ms),
        }
        if cxpos is not None:
            req["cxpos"] = _tolist(cxpos)
        if cxquat is not None:
            req["cxquat"] = _tolist(cxquat)
        if cameras:
            req["cameras"] = list(cameras)
        if delay:
            req["delay"] = int(delay)
        rep = self._t.rpc(req, recv_timeout_ms=timeout_ms + 5000)
        _check(rep, "fastpath_render")
        self._frame += 1
        return {c["name"]: _frame_from(c) for c in rep.get("cameras", [])}

    def render_mjdata(
        self, model, data, *, cameras: Optional[Sequence[str]] = None,
        delay: int = 0, timeout_ms: int = 2000,
    ) -> Dict[str, CameraFrame]:
        """Convenience: render straight from a mujoco ``(model, data)`` pair."""
        return self.render(
            sim_time=float(data.time), cameras=cameras, delay=delay,
            timeout_ms=timeout_ms, **poses_from_mjdata(model, data),
        )

    def camera_names(self) -> List[str]:
        """Names the server reports (one probe render of all cameras at default pose)."""
        rep = self._t.rpc(
            {"op": "fastpath_render", "f": self._frame, "frame_id": self._frame,
             "sim_time": 0.0, "bxpos": [], "bxquat": [], "timeout_ms": 4000},
            recv_timeout_ms=9000,
        )
        _check(rep, "fastpath_render")
        return [c.get("name", str(i)) for i, c in enumerate(rep.get("cameras", []))]

    def close(self) -> None:
        close = getattr(self._t, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "RenderClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _tolist(a) -> list:
    return a.tolist() if isinstance(a, np.ndarray) else list(a)


def _frame_from(c: Mapping) -> CameraFrame:
    return CameraFrame(
        name=str(c.get("name", "")), width=int(c["width"]), height=int(c["height"]),
        dtype=str(c.get("dtype", "bgra8")), frame_id=int(c.get("frame_id", -1)),
        sim_time=float(c.get("sim_time", 0.0)), data=bytes(c.get("data", b"")),
    )


def _check(rep, op: str) -> None:
    ok = isinstance(rep, dict) and rep.get("ok", rep.get("op") == f"{op}_ok")
    if not ok:
        detail = (rep.get("error") or rep.get("msg") or str(rep)[:200]
                  if isinstance(rep, dict) else str(rep))
        raise RenderError(f"{op} failed: {detail}")
