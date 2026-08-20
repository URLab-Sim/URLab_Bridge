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

An **owner** (a Python client *or* a UE instance) broadcasts raw kinematics on the
``viewer`` bus (``{t, qpos, qvel}``) and accepts ``fastpath_perturb`` on a control
channel. This module attaches a local ``mujoco.viewer`` window to that bus and
renders the live state -- separate from any eval render pool, so it never disturbs
it. It is owner-agnostic: the same wire contract UE's ViewerSubscribeTransport
uses, so it peeks at a Python owner or a UE owner identically.

Ctrl-drag a body and the *exact* MuJoCo perturbation force (via
``mjv_applyPerturbForce``) is sent back to the owner, which applies it to
``xfrc_applied`` on its next step -- so the push shows up in every viewer.

    python -m urlab_client.peek --model scene.xml \
        --bus tcp://127.0.0.1:5561 --control tcp://127.0.0.1:5571

``--control`` is optional; omit it for a read-only peek.
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Optional, Tuple

import numpy as np

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
    if not isinstance(msg, dict) or msg.get("qpos") is None:
        return None
    return (
        float(msg.get("t", 0.0)),
        np.asarray(msg["qpos"], np.float64),
        np.asarray(msg.get("qvel") or [], np.float64),
    )


def perturb_request(body: int, force, torque) -> dict:
    """The ``fastpath_perturb`` op dict an owner accepts (Python FastPathOwner or a
    UE bridge alike)."""
    return {
        "op": "fastpath_perturb",
        "body": int(body),
        "force": [float(x) for x in force],
        "torque": [float(x) for x in torque],
    }


class PeekViewer:
    """A mujoco viewer bound to an owner's ``viewer`` bus, with optional push-back.

    Parameters
    ----------
    model_source:
        The scene the owner is running (xml/mjb path or MJB bytes) -- the viewer
        renders its own copy, so it must match the owner's model (same nq/nv).
    bus:
        The owner's ``viewer`` PUB endpoint, e.g. ``tcp://127.0.0.1:5561``.
    control:
        The owner's control endpoint (ZMQ REQ/REP) for ``fastpath_perturb``; omit
        for a read-only peek.
    """

    def __init__(
        self,
        model_source,
        *,
        bus: str,
        control: Optional[str] = None,
        recv_timeout_ms: int = 200,
    ) -> None:
        import mujoco  # lazy
        import zmq

        self._mj = mujoco
        self._zmq = zmq
        self.model = self._load_model(model_source)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self._bus = bus
        self._control = control
        self._latest: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._ctx = zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, VIEWER_TOPIC)
        self._sub.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sub.connect(bus)
        self._ctrl = self._make_ctrl() if control else None

    def _load_model(self, src):
        mj = self._mj
        if isinstance(src, (bytes, bytearray)):
            return mj.MjModel.from_binary_buffer(bytes(src))
        s = str(src)
        return mj.MjModel.from_binary_path(s) if s.endswith(".mjb") \
            else mj.MjModel.from_xml_path(s)

    def _make_ctrl(self):
        z = self._zmq
        sock = self._ctx.socket(z.REQ)
        sock.setsockopt(z.LINGER, 0)
        sock.setsockopt(z.RCVTIMEO, 100)
        sock.setsockopt(z.SNDTIMEO, 100)
        sock.connect(self._control)
        return sock

    # -- bus receive (background) -----------------------------------------
    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                parts = self._sub.recv_multipart()
            except self._zmq.error.Again:
                continue
            except Exception:  # noqa: BLE001
                break
            if len(parts) < 2:
                continue
            frame = decode_viewer_frame(parts[-1])
            if frame is not None:
                with self._lock:
                    self._latest = frame

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

    # -- push-back (perturb) ----------------------------------------------
    def _maybe_send_perturb(self, viewer) -> None:
        if self._ctrl is None:
            return
        with viewer.lock():
            pert = viewer.perturb
            body = int(pert.select)
            if not int(getattr(pert, "active", 0)) or body <= 0:
                return
            # Reproduce MuJoCo's own perturbation force, then read it off the body.
            self.data.xfrc_applied[:] = 0.0
            self._mj.mjv_applyPerturbForce(self.model, self.data, pert)
            wrench = np.array(self.data.xfrc_applied[body], np.float64)
        self._send_perturb(body, wrench[:3], wrench[3:6])

    def _send_perturb(self, body, force, torque) -> None:
        import msgpack

        try:
            self._ctrl.send(msgpack.packb(perturb_request(body, force, torque),
                                          use_bin_type=True))
            self._ctrl.recv()
        except Exception:  # noqa: BLE001 - a REQ is wedged after a missed recv
            try:
                self._ctrl.close(0)
            except Exception:  # noqa: BLE001
                pass
            self._ctrl = self._make_ctrl()

    # -- run --------------------------------------------------------------
    def run(self) -> None:
        """Open the viewer window and render the owner's live state until closed."""
        import mujoco.viewer

        rx = threading.Thread(target=self._rx_loop, name="PeekRx", daemon=True)
        rx.start()
        push = " (ctrl-drag pushes back)" if self._ctrl is not None else " (read-only)"
        print(f"peek: viewing {self._bus}{push}")
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
        self._stop.set()
        for s in (self._sub, self._ctrl):
            try:
                if s is not None:
                    s.close(0)
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="scene xml/mjb the owner is running")
    ap.add_argument("--bus", required=True, help="owner viewer PUB, e.g. tcp://127.0.0.1:5561")
    ap.add_argument("--control", default=None,
                    help="owner control endpoint for perturbs (omit for read-only)")
    args = ap.parse_args()
    PeekViewer(args.model, bus=args.bus, control=args.control).run()


if __name__ == "__main__":
    main()
