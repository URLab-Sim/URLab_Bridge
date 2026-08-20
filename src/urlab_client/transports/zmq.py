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

from . import (
    FrameCallback,
    SnapshotCallback,
    Transport,
    parse_camera_frame,
    resolve_endpoint,
)
from ..errors import URLabTimeoutError

logger = logging.getLogger(__name__)

# Backoff bounds for stream-loop reconnection after a socket error.
_STREAM_RECONNECT_MIN_S = 0.25
_STREAM_RECONNECT_MAX_S = 5.0

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
        recv_timeout_ms: int = 5000,
    ):
        self.address = address
        self.step_port = step_port
        self.state_port = state_port
        self._recv_timeout_ms = recv_timeout_ms

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

        # Viewer bus: a PUB the owner binds so read-only viewers can subscribe
        # to raw kinematics. Lazily bound by enable_viewer_broadcast().
        self._viewer_pub: Any = None
        self._viewer_endpoint: Optional[str] = None
        # Viewer bus SUBSCRIBER (the read side, for a peek/viewer).
        self._viewer_sub_stop: Optional[threading.Event] = None
        self._viewer_sub_thread: Optional[threading.Thread] = None

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
        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
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
        recv_timeout_ms: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """One-shot send + recv. Pass ``recv_timeout_ms`` for ops that
        legitimately take longer than the constructor default (e.g.
        ``begin_pie`` blocks on UE compile, can be 30s+). The override
        is applied for this call only; the next call goes back to the
        default."""
        if msgpack is None:
            raise RuntimeError("msgpack not installed; cannot run RPC")
        # Serialise: REQ sockets aren't thread-safe, and concurrent use
        # from the user thread + a UI tick thread crashed libzmq's
        # Windows signaler (signaler.cpp:345 WSAECONNRESET assert).
        effective_timeout_ms = (
            int(recv_timeout_ms) if recv_timeout_ms is not None
            else self._recv_timeout_ms
        )
        with self._sock_lock:
            self._ensure_socket()
            try:
                if recv_timeout_ms is not None:
                    self._socket.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
                try:
                    self._socket.send(msgpack.packb(dict(request), use_bin_type=True))
                    raw = self._socket.recv()
                finally:
                    if recv_timeout_ms is not None and self._socket is not None:
                        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
            except zmq.Again as exc:
                # Recv timed out. Normalise to URLabTimeoutError so callers get
                # the same exception on both transports (SHM raises it too)
                # instead of a raw zmq.error.Again leaking through. The REQ
                # socket is in EFSM after a missed recv; reset it for the next call.
                self._reset_socket()
                raise URLabTimeoutError(
                    f"RPC {request.get('op')!r} over ZMQ",
                    waited_s=effective_timeout_ms / 1000.0,
                    op=request.get("op"),
                ) from exc
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
        assert self._state_stop is not None
        # Use the transport's own context (set up in _ensure_socket).
        if self._ctx is None:
            self._ctx = zmq.Context()
        backoff = _STREAM_RECONNECT_MIN_S
        # Outer loop rebuilds the SUB socket after a fatal recv error so a
        # transient publisher/context hiccup doesn't kill the stream for the
        # rest of the session (the old code broke out silently). Every exit
        # is logged so a dead stream is never invisible.
        while not self._state_stop.is_set():
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                sock.connect(f"{self.address}:{self.state_port}")
                sock.setsockopt(zmq.SUBSCRIBE, b"state/full")
                sock.setsockopt(zmq.RCVTIMEO, 200)
                while not self._state_stop.is_set():
                    try:
                        sock.recv()  # topic frame, discard
                        payload = sock.recv()  # msgpack snapshot
                    except zmq.Again:
                        # Idle timeout is the healthy path; a successful recv
                        # resets the reconnect backoff.
                        continue
                    backoff = _STREAM_RECONNECT_MIN_S
                    try:
                        snap = msgpack.unpackb(
                            payload, raw=False, strict_map_key=False
                        )
                    except Exception as exc:
                        logger.debug("state/full decode failed: %s", exc)
                        continue
                    try:
                        on_snapshot(snap)
                    except Exception as exc:  # pragma: no cover - callback-defensive
                        logger.debug("state snapshot callback raised: %s", exc)
            except Exception as exc:
                logger.warning(
                    "ZmqTransport state stream error (%s); reconnecting in %.2fs",
                    exc, backoff,
                )
            finally:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
            if self._state_stop.is_set():
                break
            self._state_stop.wait(backoff)
            backoff = min(backoff * 2.0, _STREAM_RECONNECT_MAX_S)
        logger.debug("ZmqTransport state stream loop exited")

    # -- viewer bus consumer ----------------------------------------------

    def start_viewer_stream(self, on_frame, *, endpoint=None) -> None:
        if self._viewer_sub_thread is not None and self._viewer_sub_thread.is_alive():
            return
        ep = endpoint or self._viewer_endpoint
        if not ep:
            raise ValueError(
                "start_viewer_stream needs an owner viewer endpoint (tcp://host:port)")
        self._viewer_sub_stop = threading.Event()
        self._viewer_sub_thread = threading.Thread(
            target=self._viewer_loop, args=(on_frame, ep),
            name="URLabViewerSub", daemon=True,
        )
        self._viewer_sub_thread.start()

    def stop_viewer_stream(self) -> None:
        if self._viewer_sub_stop is not None:
            self._viewer_sub_stop.set()
        if self._viewer_sub_thread is not None:
            self._viewer_sub_thread.join(timeout=5.0)
            self._viewer_sub_thread = None
        self._viewer_sub_stop = None

    def _viewer_loop(self, on_frame, endpoint: str) -> None:
        # Mirrors _state_loop but subscribes the owner's viewer PUB (topic
        # "viewer", {t,qpos,qvel}) instead of the server state/full snapshot.
        if zmq is None or msgpack is None:
            return
        assert self._viewer_sub_stop is not None
        if self._ctx is None:
            self._ctx = zmq.Context()
        backoff = _STREAM_RECONNECT_MIN_S
        while not self._viewer_sub_stop.is_set():
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                sock.connect(endpoint)
                sock.setsockopt(zmq.SUBSCRIBE, self._VIEWER_TOPIC)
                sock.setsockopt(zmq.RCVTIMEO, 200)
                while not self._viewer_sub_stop.is_set():
                    try:
                        parts = sock.recv_multipart()
                    except zmq.Again:
                        continue
                    backoff = _STREAM_RECONNECT_MIN_S
                    if len(parts) < 2:
                        continue
                    try:
                        frame = msgpack.unpackb(
                            parts[-1], raw=False, strict_map_key=False)
                    except Exception as exc:
                        logger.debug("viewer frame decode failed: %s", exc)
                        continue
                    try:
                        on_frame(frame)
                    except Exception as exc:  # pragma: no cover - callback-defensive
                        logger.debug("viewer frame callback raised: %s", exc)
            except Exception as exc:
                logger.warning(
                    "ZmqTransport viewer stream error (%s); reconnecting in %.2fs",
                    exc, backoff)
            finally:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
            if self._viewer_sub_stop.is_set():
                break
            self._viewer_sub_stop.wait(backoff)
            backoff = min(backoff * 2.0, _STREAM_RECONNECT_MAX_S)
        logger.debug("ZmqTransport viewer stream loop exited")

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
        stop_ev = self._cam_stops[key]
        # Endpoint from the handshake is a bind form ("tcp://*:NNNN" /
        # "tcp://0.0.0.0:NNNN"). Rewrite it to the host the RPC is targeting;
        # a concrete advertised host is preserved.
        connect_ep = resolve_endpoint(endpoint, self.address)
        backoff = _STREAM_RECONNECT_MIN_S
        # Rebuild the SUB socket after a fatal recv error rather than dying
        # silently, and log every loop exit.
        while not stop_ev.is_set():
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.LINGER, 0)
            # Bound the inbound queue: a live camera feed only cares about the
            # newest frame, so don't let a slow consumer accumulate seconds of
            # stale frames in the SUB queue (that's what made the dashboard lag
            # ~3s). HWM must be set before connect. Combined with the drain-to-
            # latest below, the delivered frame stays fresh regardless of how
            # fast the consumer renders.
            sock.setsockopt(zmq.RCVHWM, 4)
            try:
                sock.connect(connect_ep)
                sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
                sock.setsockopt(zmq.RCVTIMEO, 200)
                while not stop_ev.is_set():
                    try:
                        sock.recv()         # topic frame, discard
                        payload = sock.recv()  # [meta][pixels]
                    except zmq.Again:
                        continue
                    backoff = _STREAM_RECONNECT_MIN_S
                    # Drain any backlog and keep only the freshest frame, so a
                    # slow consumer (heavy UI) never falls behind the publisher.
                    # Multipart delivery is atomic, so a NOBLOCK topic recv
                    # guarantees its payload is also available.
                    while True:
                        try:
                            sock.recv(flags=zmq.NOBLOCK)            # newer topic
                            payload = sock.recv(flags=zmq.NOBLOCK)  # newer payload
                        except zmq.Again:
                            break
                    pixels, frame_id, sim_time, capture_time = parse_camera_frame(
                        payload
                    )
                    try:
                        on_frame(pixels, frame_id, sim_time, capture_time)
                    except Exception as exc:  # pragma: no cover - callback-defensive
                        logger.debug("camera frame callback raised: %s", exc)
            except Exception as exc:
                logger.warning(
                    "ZmqTransport camera %s/%s stream error (%s); "
                    "reconnecting in %.2fs", key[0], key[1], exc, backoff,
                )
            finally:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
            if stop_ev.is_set():
                break
            stop_ev.wait(backoff)
            backoff = min(backoff * 2.0, _STREAM_RECONNECT_MAX_S)
        logger.debug("ZmqTransport camera %s/%s stream loop exited", key[0], key[1])

    # -- viewer bus (owner -> viewers) ------------------------------------

    # Topic every viewer subscribes to. A bare prefix keeps the wire format
    # trivial: [topic, msgpack({"t","qpos","qvel"})].
    _VIEWER_TOPIC = b"viewer"
    _GEOMS_TOPIC = b"geoms"

    def enable_viewer_broadcast(self, port: int) -> Optional[str]:
        if zmq is None:
            raise RuntimeError("pyzmq not installed; cannot broadcast")
        if msgpack is None:
            raise RuntimeError("msgpack not installed; cannot broadcast")
        with self._sock_lock:
            if self._viewer_pub is not None:
                return self._viewer_endpoint
            if self._ctx is None:
                self._ctx = zmq.Context()
            pub = self._ctx.socket(zmq.PUB)
            pub.setsockopt(zmq.LINGER, 0)
            # Bind on every interface so a viewer on another host can reach it;
            # the owner advertises its own reachable address out of band.
            endpoint = f"tcp://0.0.0.0:{port}"
            pub.bind(endpoint)
            self._viewer_pub = pub
            self._viewer_endpoint = endpoint
            logger.info("ZmqTransport viewer PUB bound to %s", endpoint)
            return endpoint

    def publish_viewer_state(self, payload: Mapping[str, Any]) -> None:
        # Called once per owner step; drop silently if not enabled so callers
        # need no guard. PUB.send never blocks (it discards with no subscriber).
        with self._sock_lock:
            pub = self._viewer_pub
            if pub is None:
                return
            try:
                pub.send_multipart(
                    [self._VIEWER_TOPIC, msgpack.packb(dict(payload), use_bin_type=True)],
                    flags=zmq.NOBLOCK,
                )
            except Exception as exc:  # pragma: no cover - best-effort broadcast
                logger.debug("viewer publish dropped: %s", exc)

    def publish_geoms(self, payload: Mapping[str, Any]) -> None:
        # Per-geom world transforms for fast-path renderers. Shares the viewer
        # PUB socket, distinguished by the "geoms" topic.
        with self._sock_lock:
            pub = self._viewer_pub
            if pub is None:
                return
            try:
                pub.send_multipart(
                    [self._GEOMS_TOPIC, msgpack.packb(dict(payload), use_bin_type=True)],
                    flags=zmq.NOBLOCK,
                )
            except Exception as exc:  # pragma: no cover - best-effort broadcast
                logger.debug("geom publish dropped: %s", exc)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        # Order matters: signal + join SUB threads first, then close the
        # REQ socket, then term the context. Reverse ordering races
        # libzmq's signaler (WSAECONNRESET on Windows).
        self.stop_state_stream()
        self.stop_viewer_stream()
        self.stop_camera_streams()
        with self._sock_lock:
            if self._socket is not None:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
            self._socket = None
            if self._viewer_pub is not None:
                try:
                    self._viewer_pub.close(linger=0)
                except Exception:
                    pass
                self._viewer_pub = None
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
