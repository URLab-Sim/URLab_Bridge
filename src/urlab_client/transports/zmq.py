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

"""ZMQ REQ-REP + PUB-SUB transport for URLabClient."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse

from . import FrameCallback, SnapshotCallback, Transport, parse_camera_frame

logger = logging.getLogger(__name__)

try:  # pragma: no cover - trivial import guard
    import msgpack  # type: ignore
except ImportError:  # pragma: no cover
    msgpack = None  # noqa: N816

try:  # pragma: no cover
    import zmq  # type: ignore
except ImportError:  # pragma: no cover
    zmq = None  # noqa: N816


class ZmqTransport(Transport):
    """REQ-REP + PUB-SUB transport."""

    def __init__(
        self,
        address: str = "tcp://localhost",
        *,
        step_port: int = 5559,
        state_port: int = 5555,
        rcv_timeout_ms: int = 5000,
    ):
        self.address = address
        self.step_port = step_port
        self.state_port = state_port
        self._rcv_timeout_ms = rcv_timeout_ms

        self._ctx: Any = None
        self._socket: Any = None
        # Serialise REQ-socket ops + ctx/socket lifecycle. ZMQ sockets
        # are not thread-safe, and on Windows concurrent access from
        # the IO thread + a user thread + a UI tick thread can trip
        # libzmq's signaler (signaler.cpp:345 wsa_assert -> abort) when
        # a recv() on the internal TCP-loopback signaler pair returns
        # WSAECONNRESET. Holding this lock around every send/recv path
        # plus socket reset + ctx teardown collapses concurrent callers
        # into one, eliminating the race.
        self._sock_lock = threading.RLock()

        self._state_stop: Optional[threading.Event] = None
        self._state_thread: Optional[threading.Thread] = None
        self._cam_threads: Dict[Tuple[str, str], threading.Thread] = {}
        self._cam_stops: Dict[Tuple[str, str], threading.Event] = {}

    # -- RPC --------------------------------------------------------------

    def _ensure_socket(self) -> None:
        # Caller holds self._sock_lock.
        if self._socket is not None:
            return
        if zmq is None:
            raise RuntimeError("pyzmq not installed; cannot connect")
        # Per-instance Context so close() can join SUB threads before
        # term(). The pyzmq singleton would race on interpreter shutdown.
        if self._ctx is None:
            self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._rcv_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        endpoint = f"{self.address}:{self.step_port}"
        self._socket.connect(endpoint)
        logger.info("ZmqTransport REQ connected to %s", endpoint)

    def _reset_socket(self) -> None:
        """Tear down and recreate the REQ socket. Required after a recv
        timeout because pyzmq's REQ enforces strict send/recv alternation
        -- a missed recv leaves the socket in EFSM (operation cannot be
        accomplished in current state) for every subsequent send.
        Caller holds self._sock_lock."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._socket = None

    def rpc(
        self,
        request: Mapping[str, Any],
        *,
        rcv_timeout_ms: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """One-shot send + recv. Pass ``rcv_timeout_ms`` for ops that
        legitimately take longer than the constructor default (e.g.
        ``begin_pie`` blocks on UE compile, can be 30s+). The override
        is applied for this call only; the next call goes back to the
        default."""
        if msgpack is None:
            raise RuntimeError("msgpack not installed; cannot run RPC")
        # Serialise: REQ sockets aren't thread-safe, and concurrent use
        # from the user thread + a UI tick thread crashed libzmq's
        # Windows signaler (signaler.cpp:345 WSAECONNRESET assert).
        with self._sock_lock:
            self._ensure_socket()
            try:
                if rcv_timeout_ms is not None:
                    self._socket.setsockopt(zmq.RCVTIMEO, int(rcv_timeout_ms))
                try:
                    self._socket.send(msgpack.packb(dict(request), use_bin_type=True))
                    raw = self._socket.recv()
                finally:
                    if rcv_timeout_ms is not None and self._socket is not None:
                        self._socket.setsockopt(zmq.RCVTIMEO, self._rcv_timeout_ms)
            except Exception:
                # A REQ socket is unusable after send/recv error; reset so the
                # next call gets a clean retry.
                self._reset_socket()
                raise
        reply = msgpack.unpackb(raw, raw=False)
        if not isinstance(reply, dict):
            raise RuntimeError(
                f"non-dict reply: {type(reply).__name__}"
            )
        return reply

    # -- state stream -----------------------------------------------------

    def start_state_stream(self, on_snapshot: SnapshotCallback) -> None:
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        self._state_stop = threading.Event()
        self._state_thread = threading.Thread(
            target=self._state_loop,
            args=(on_snapshot,),
            name="URLabStateSub",
            daemon=True,
        )
        self._state_thread.start()

    def stop_state_stream(self) -> None:
        if self._state_stop is not None:
            self._state_stop.set()
        if self._state_thread is not None:
            # Generous timeout: the loop's RCVTIMEO is 200ms so the
            # thread observes the stop flag within ~250ms; 5s is a hard
            # ceiling for pathological cases. Joining without a timeout
            # would hang if a loop iteration is wedged.
            self._state_thread.join(timeout=5.0)
            self._state_thread = None
        self._state_stop = None

    def _state_loop(self, on_snapshot: SnapshotCallback) -> None:
        if zmq is None or msgpack is None:
            return
        # Use the transport's own context (set up in _ensure_socket).
        if self._ctx is None:
            self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(f"{self.address}:{self.state_port}")
            sock.setsockopt(zmq.SUBSCRIBE, b"state/full")
            sock.setsockopt(zmq.RCVTIMEO, 200)
            assert self._state_stop is not None
            while not self._state_stop.is_set():
                try:
                    sock.recv()  # topic frame, discard
                    payload = sock.recv()  # msgpack snapshot
                except zmq.Again:
                    continue
                except Exception:
                    break
                try:
                    snap = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                except Exception as exc:
                    logger.debug("state/full decode failed: %s", exc)
                    continue
                try:
                    on_snapshot(snap)
                except Exception as exc:  # pragma: no cover - callback-defensive
                    logger.debug("state snapshot callback raised: %s", exc)
        finally:
            try:
                sock.close(linger=0)
            except Exception:
                pass

    # -- camera streams ---------------------------------------------------

    def start_camera_stream(
        self,
        prefix: str,
        name: str,
        endpoint: str,
        topic: str,
        on_frame: FrameCallback,
    ) -> None:
        key = (prefix, name)
        if key in self._cam_threads and self._cam_threads[key].is_alive():
            return
        self._cam_stops[key] = threading.Event()
        t = threading.Thread(
            target=self._camera_loop,
            args=(key, endpoint, topic, on_frame),
            name=f"URLabCamSub-{prefix}-{name}",
            daemon=True,
        )
        self._cam_threads[key] = t
        t.start()

    def stop_camera_streams(self) -> None:
        for ev in self._cam_stops.values():
            ev.set()
        for t in self._cam_threads.values():
            # 5s ceiling — loops have 200ms RCVTIMEO so they exit within
            # ~250ms in the common case. Hard timeout prevents an orphan
            # thread from outliving close() and racing context termination.
            t.join(timeout=5.0)
        self._cam_stops.clear()
        self._cam_threads.clear()

    def _camera_loop(
        self,
        key: Tuple[str, str],
        endpoint: str,
        topic: str,
        on_frame: FrameCallback,
    ) -> None:
        if zmq is None:
            return
        if self._ctx is None:
            self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            # Endpoint from the handshake is "tcp://*:NNNN" (server bind
            # form). Connect to the same host the RPC is targeting on the
            # advertised port.
            connect_ep = f"{self.address}:{self._port_from_endpoint(endpoint)}"
            sock.connect(connect_ep)
            sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
            sock.setsockopt(zmq.RCVTIMEO, 200)
            stop_ev = self._cam_stops[key]
            while not stop_ev.is_set():
                try:
                    sock.recv()         # topic frame, discard
                    payload = sock.recv()  # [meta(32)][pixels]
                except zmq.Again:
                    continue
                except Exception:
                    break
                pixels, frame_id, sim_time = parse_camera_frame(payload)
                try:
                    on_frame(pixels, frame_id, sim_time)
                except Exception as exc:  # pragma: no cover - callback-defensive
                    logger.debug("camera frame callback raised: %s", exc)
        finally:
            try:
                sock.close(linger=0)
            except Exception:
                pass

    @staticmethod
    def _port_from_endpoint(endpoint: str) -> int:
        # Endpoint looks like "tcp://*:5558" or "tcp://127.0.0.1:5558".
        # urlparse needs a scheme it understands; substitute one.
        parsed = urlparse(endpoint.replace("tcp://", "http://"))
        if parsed.port is not None:
            return parsed.port
        # Fallback: take the trailing ":NNNN".
        return int(endpoint.rsplit(":", 1)[-1])

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        # Order matters: signal + join SUB threads first, then close the
        # REQ socket, then term the context. Reverse ordering races
        # libzmq's signaler (WSAECONNRESET on Windows).
        self.stop_state_stream()
        self.stop_camera_streams()
        with self._sock_lock:
            if self._socket is not None:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
            self._socket = None
            if self._ctx is not None:
                try:
                    # destroy(linger=0) is idempotent and revokes any pending
                    # operations on stuck sockets — the safe hammer for the
                    # case where stop_*_stream's join timed out and a daemon
                    # thread is still in recv. Equivalent to term() when
                    # everything closed cleanly.
                    self._ctx.destroy(linger=0)
                except Exception:
                    pass
                self._ctx = None
