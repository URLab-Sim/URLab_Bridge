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

"""Peek at a running fast-path simulation -- a smooth, async viewer you can reach into.

An **owner** (a Python client *or* a UE instance) broadcasts raw kinematics on a
``viewer`` bus (``{t, qpos, qvel}``) and accepts ``fastpath_perturb`` on a control
channel. This attaches a local ``mujoco.viewer`` window to that bus and renders the
live state -- separate from any eval render pool, so it never disturbs it.

**Transport-agnostic:** the state stream + push-back go through the pluggable
``Transport`` layer (``transport.start_viewer_stream`` / ``transport.rpc``), so a
peek works over ZMQ or gRPC identically -- pick with ``transport=``. The wire
contract (topic ``"viewer"`` + ``{t,qpos,qvel}``, and ``fastpath_perturb``) is the
same one UE's ViewerSubscribeTransport / bridge speak, so it peeks at a Python or
a UE owner alike.

Ctrl-drag a body and the *exact* MuJoCo perturbation force (via
``mjv_applyPerturbForce``) is sent back, so the push shows up in every viewer.

    python -m urlab_client.peek --model scene.xml \
        --bus tcp://127.0.0.1:5561 --control tcp://127.0.0.1:5571      # zmq
    python -m urlab_client.peek --model scene.xml \
        --transport grpc --control 127.0.0.1:50051                    # grpc

``--control`` is optional; omit it for a read-only peek.
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .transports import make_transport

__all__ = ["PeekViewer", "decode_viewer_frame", "perturb_request", "VIEWER_TOPIC"]

# Topic every viewer subscribes to; matches ZmqTransport._VIEWER_TOPIC and UE's
# ViewerSubscribeTransport. Wire frame: [b"viewer", msgpack({"t","qpos","qvel"})].
VIEWER_TOPIC = b"viewer"


def decode_viewer_frame(payload: bytes) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Decode a ``viewer``-bus msgpack payload into ``(t, qpos, qvel)`` (qvel may be
    empty). Returns None if it carries no qpos."""
    import msgpack

    try:
        msg = msgpack.unpackb(payload, raw=False)
    except Exception:  # noqa: BLE001
        return None
    return _frame_from_mapping(msg)


def _frame_from_mapping(msg: Any) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    if not isinstance(msg, Mapping) or msg.get("qpos") is None:
        return None
    return (
        float(msg.get("t", 0.0)),
        np.asarray(msg["qpos"], np.float64),
        np.asarray(msg.get("qvel") or [], np.float64),
    )


def perturb_request(body: int, force, torque) -> dict:
    """The ``fastpath_perturb`` op dict an owner accepts (Python or UE alike)."""
    return {
        "op": "fastpath_perturb",
        "body": int(body),
        "force": [float(x) for x in force],
        "torque": [float(x) for x in torque],
    }


def _split_endpoint(ep: str, default_port: int) -> Tuple[str, int]:
    """('tcp://host:port' | 'host:port' | 'host') -> (host, port)."""
    s = ep.replace("tcp://", "", 1)
    if ":" in s:
        host, _, p = s.rpartition(":")
        return host or "127.0.0.1", int(p)
    return s or "127.0.0.1", default_port


class PeekViewer:
    """A mujoco viewer bound to an owner's viewer bus, with optional push-back.

    Parameters
    ----------
    model_source:
        The scene the owner is running (xml/mjb path or MJB bytes) -- must match the
        owner's model (same nq/nv).
    control:
        The owner endpoint that answers ``fastpath_perturb`` (and, over gRPC, also
        streams state). ZMQ: the owner control REP (``tcp://host:port``). gRPC: the
        owner's ``host:port`` gRPC endpoint. Omit for a read-only peek over gRPC;
        for ZMQ read-only, pass only ``bus``.
    bus:
        ZMQ only: the owner's viewer PUB (``tcp://host:port``). Ignored for gRPC
        (the state stream rides the gRPC channel).
    transport:
        ``"zmq"`` (default) or ``"grpc"``.
    """

    def __init__(
        self,
        model_source,
        *,
        control: Optional[str] = None,
        bus: Optional[str] = None,
        transport: str = "zmq",
        recv_timeout_ms: int = 2000,
    ) -> None:
        import mujoco  # lazy

        self._mj = mujoco
        self.model = self._load_model(model_source)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self._transport_name = transport
        self._bus = bus
        self._perturb = control is not None
        self._latest: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        self._lock = threading.Lock()

        # rpc (perturb) targets the control endpoint; over gRPC the same endpoint
        # also carries the state stream. A ZMQ read-only peek has no control, so
        # anchor the transport on the bus host for the (unused) rpc socket.
        anchor = control or bus or "tcp://127.0.0.1:5571"
        default_port = 50051 if transport == "grpc" else 5571
        host, port = _split_endpoint(anchor, default_port)
        self._t = make_transport(
            transport, address=f"tcp://{host}", step_port=port,
            recv_timeout_ms=recv_timeout_ms,
        )

    def _load_model(self, src):
        mj = self._mj
        if isinstance(src, (bytes, bytearray)):
            return mj.MjModel.from_binary_buffer(bytes(src))
        s = str(src)
        return mj.MjModel.from_binary_path(s) if s.endswith(".mjb") \
            else mj.MjModel.from_xml_path(s)

    # -- viewer bus receive (via transport) -------------------------------
    def _on_frame(self, frame: Mapping[str, Any]) -> None:
        parsed = _frame_from_mapping(frame)
        if parsed is not None:
            with self._lock:
                self._latest = parsed

    def _apply_latest(self) -> None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return
        t, qpos, qvel = latest
        m, d = self.model, self.data
        if qpos.shape[0] != m.nq:
            return  # model mismatch: don't scatter garbage
        d.qpos[:] = qpos
        if qvel.shape[0] == m.nv:
            d.qvel[:] = qvel
        d.time = t
        self._mj.mj_forward(m, d)

    # -- push-back (perturb, via transport.rpc) ---------------------------
    def _maybe_send_perturb(self, viewer) -> None:
        if not self._perturb:
            return
        with viewer.lock():
            pert = viewer.perturb
            body = int(pert.select)
            if not int(getattr(pert, "active", 0)) or body <= 0:
                return
            self.data.xfrc_applied[:] = 0.0
            self._mj.mjv_applyPerturbForce(self.model, self.data, pert)
            wrench = np.array(self.data.xfrc_applied[body], np.float64)
        try:
            self._t.rpc(perturb_request(body, wrench[:3], wrench[3:6]),
                        recv_timeout_ms=150)
        except Exception:  # noqa: BLE001 - best-effort; a dropped push is fine
            pass

    # -- run --------------------------------------------------------------
    def run(self) -> None:
        """Open the viewer window and render the owner's live state until closed."""
        import mujoco.viewer

        self._t.start_viewer_stream(self._on_frame, endpoint=self._bus)
        push = " (ctrl-drag pushes back)" if self._perturb else " (read-only)"
        print(f"peek: viewing over {self._transport_name}{push}")
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as v:
                while v.is_running():
                    self._apply_latest()
                    v.sync()
                    self._maybe_send_perturb(v)
                    time.sleep(1.0 / 120.0)
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._t.stop_viewer_stream()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._t.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="scene xml/mjb the owner is running")
    ap.add_argument("--transport", default="zmq", choices=["zmq", "grpc"])
    ap.add_argument("--control", default=None,
                    help="owner control endpoint for perturbs (omit = read-only)")
    ap.add_argument("--bus", default=None,
                    help="ZMQ only: owner viewer PUB, e.g. tcp://127.0.0.1:5561")
    args = ap.parse_args()
    PeekViewer(args.model, control=args.control, bus=args.bus,
               transport=args.transport).run()


if __name__ == "__main__":
    main()
