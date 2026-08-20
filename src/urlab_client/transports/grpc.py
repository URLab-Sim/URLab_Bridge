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

"""dm_env_rpc (gRPC) RPC transport for URLabClient.

Drives the render server over DeepMind ``dm_env_rpc`` instead of ZMQ. The UE
``URLabDmEnvRpc`` backend exposes a ``dm_env_rpc.v1.Environment`` service whose
bidirectional ``Process`` stream tunnels our request/reply bytes through the
proto's ``extension`` field: each request is msgpack (exactly what ZmqTransport
sends), wrapped in a ``UrlabPacket`` packed into ``EnvironmentRequest.extension``.
The server hands the payload to the same ``ProcessRequestBytes`` dispatch every
transport uses, so every op -- including ``fastpath_render`` -- works unchanged.

Only ``rpc()`` is implemented: this is the request/reply path RenderClient uses.
The PUB/SUB stream methods (state + camera streams) have no dm_env_rpc analogue
here and raise ``NotImplementedError`` -- RenderClient never calls them.

The server binds ``0.0.0.0:50051`` (``UURLabDmEnvRpcTransport::ListenPort``), so
the default port here is 50051, not the ZMQ step port 5559.

Robustness. The bidi stream is a single shared channel, so ``rpc()`` serialises
under a lock and is hardened three ways so a downstream user never has to babysit
the connection:

* **Fail-fast connect** -- a fresh channel is waited to READY within
  ``connect_timeout_s``; nothing listening surfaces as a clear ``ConnectionError``
  naming the target, not an opaque render timeout.
* **Transparent reconnect** -- a *broken* stream (server restart, idle reap, a
  half-open socket) is rebuilt and the request re-sent once. A *timeout* is the
  caller's declared deadline and is surfaced, not retried.
* **Sequence-id validation** -- every request stamps ``sequence_id`` and the
  server echoes it; a mismatched reply means the stream desynced (a stale reply
  from an abandoned call) and is treated as a broken stream so the reconnect path
  realigns instead of handing back the wrong frame.
"""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from typing import Any, Mapping, Optional

from . import FrameCallback, SnapshotCallback, Transport
from ..errors import URLabTimeoutError

try:  # pragma: no cover - trivial import guard
    import msgpack  # type: ignore
except ImportError:  # pragma: no cover
    msgpack = None  # noqa: N816

dm_env_rpc_pb2 = None
dm_env_rpc_pb2_grpc = None
urlab_dm_env_rpc_pb2 = None
_import_error: Optional[str] = None

try:
    from dm_env_rpc.v1 import dm_env_rpc_pb2, dm_env_rpc_pb2_grpc  # type: ignore
    from ._dmenv import urlab_dm_env_rpc_pb2  # type: ignore
except Exception:
    try:
        from ._dmenv import dm_env_rpc_pb2, dm_env_rpc_pb2_grpc, urlab_dm_env_rpc_pb2  # type: ignore
    except Exception as _e:
        _import_error = str(_e)
        dm_env_rpc_pb2 = None
        dm_env_rpc_pb2_grpc = None
        urlab_dm_env_rpc_pb2 = None

# Default gRPC port the UE dm_env_rpc backend listens on (ListenPort).
DEFAULT_DMENV_PORT = 50051


class _StreamBroken(Exception):
    """Internal: the bidi stream is unusable (dropped / desynced / errored) and
    must be torn down and rebuilt. Never escapes ``rpc()``."""


class GrpcTransport(Transport):
    """dm_env_rpc bidi-stream RPC transport (request/reply only)."""

    def __init__(
        self,
        address: str = "tcp://127.0.0.1",
        *,
        step_port: int = DEFAULT_DMENV_PORT,
        recv_timeout_ms: int = 5000,
        connect_timeout_s: float = 5.0,
    ) -> None:
        # gRPC targets are bare "host:port"; accept a tcp://host[:port] address
        # for parity with the other transports and take only its host.
        host = address.replace("tcp://", "", 1).split("/", 1)[0].split(":", 1)[0]
        self._target = f"{host or '127.0.0.1'}:{step_port}"
        self._recv_timeout_ms = recv_timeout_ms
        self._connect_timeout_s = float(connect_timeout_s)

        self._lock = threading.Lock()
        self._seq = 0
        self._channel: Any = None
        self._resp_iter: Any = None
        self._req_q: "queue.Queue[Any]" = queue.Queue()
        # A single-thread executor lets us bound each blocking next() on the
        # response iterator by a per-call timeout without cancelling gRPC's own
        # (stream-wide) deadline. It is replaced on every reset so a timed-out
        # next() left blocked on a dead iterator can't starve the next call.
        self._recv_exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="URLabDmEnvRecv"
        )
        # Viewer subscription: its OWN channel + Process call (a dedicated
        # server-stream), independent of the rpc stream + lock so it runs
        # concurrently with blocking renders/perturbs.
        self._viewer_stop: Optional[threading.Event] = None
        self._viewer_thread: Optional[threading.Thread] = None

    # -- stream setup ------------------------------------------------------
    def _ensure_stream(self) -> None:
        # Caller holds self._lock.
        if self._channel is not None:
            return
        try:
            import grpc  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "grpcio not installed; cannot use transport='grpc'"
            ) from exc

        # -1 == unlimited, matching the server's SetMax*MessageSize(-1); camera
        # frames (BGRA8 at 1280x720+) exceed the 4 MB gRPC default. Keepalive keeps
        # an idle stream from being reaped by a NAT/proxy between renders.
        self._channel = grpc.insecure_channel(
            self._target,
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
                ("grpc.keepalive_time_ms", 20000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        )
        # Fail fast with a clear message if nothing is listening yet, rather than
        # letting the first next() block until the RPC timeout and surface as an
        # opaque "render timed out".
        try:
            grpc.channel_ready_future(self._channel).result(
                timeout=self._connect_timeout_s
            )
        except grpc.FutureTimeoutError as exc:
            ch, self._channel = self._channel, None
            try:
                ch.close()
            except Exception:  # pragma: no cover
                pass
            raise ConnectionError(
                f"render server not reachable at {self._target} "
                f"(no gRPC listener within {self._connect_timeout_s:.0f}s)"
            ) from exc

        stub = dm_env_rpc_pb2_grpc.EnvironmentStub(self._channel)

        def _request_gen():
            # One long-lived request stream fed one message per rpc() call. A
            # None sentinel (close()) ends it and lets the server's Read loop exit.
            while True:
                item = self._req_q.get()
                if item is None:
                    return
                yield item

        self._resp_iter = stub.Process(_request_gen())

    # -- RPC ---------------------------------------------------------------
    def rpc(
        self,
        request: Mapping[str, Any],
        *,
        recv_timeout_ms: Optional[int] = None,
    ) -> Mapping[str, Any]:
        if msgpack is None:
            raise RuntimeError("msgpack not installed; cannot run RPC")
        if dm_env_rpc_pb2 is None or urlab_dm_env_rpc_pb2 is None:
            raise RuntimeError(
                f"dm_env_rpc protobuf modules not available; cannot run gRPC RPC ({_import_error})"
            )

        payload = msgpack.packb(dict(request), use_bin_type=True)
        timeout_ms = (
            int(recv_timeout_ms) if recv_timeout_ms is not None
            else self._recv_timeout_ms
        )
        op = str(request.get("op", ""))

        # One transparent reconnect: a broken/dropped bidi stream is rebuilt and
        # the request re-sent. A *timeout* is the caller's declared deadline and
        # is NOT retried here -- it surfaces so the caller decides how long to wait.
        conn_err: Optional[BaseException] = None
        for _attempt in range(2):
            with self._lock:
                try:
                    return self._rpc_locked(op, bytes(payload), timeout_ms)
                except URLabTimeoutError:
                    self._reset_stream()
                    raise
                except _StreamBroken as exc:
                    conn_err = exc.__cause__ or exc
                    self._reset_stream()
                    # fall out of the lock, then rebuild + resend on the next pass
        raise ConnectionError(
            f"render server RPC {op!r} failed after reconnect to {self._target}: "
            f"{conn_err}"
        ) from conn_err

    def _rpc_locked(
        self, op: str, payload: bytes, timeout_ms: int
    ) -> Mapping[str, Any]:
        # Caller holds self._lock.
        self._ensure_stream()
        self._seq += 1
        seq = self._seq
        packet = urlab_dm_env_rpc_pb2.UrlabPacket(
            op=op, payload=payload, sequence_id=seq
        )
        env = dm_env_rpc_pb2.EnvironmentRequest()
        env.extension.Pack(packet)
        self._req_q.put(env)

        try:
            # next() blocks until UE renders + replies; bound it so a dead server
            # surfaces as a timeout instead of hanging the caller.
            resp = self._recv_exec.submit(next, self._resp_iter).result(
                timeout=timeout_ms / 1000.0
            )
        except concurrent.futures.TimeoutError as exc:
            raise URLabTimeoutError(
                f"RPC {op!r} over gRPC", waited_s=timeout_ms / 1000.0, op=op
            ) from exc
        except StopIteration as exc:
            raise _StreamBroken(f"stream closed by server during {op!r}") from exc
        except Exception as exc:  # gRPC RpcError, channel teardown, etc.
            raise _StreamBroken(f"stream error during {op!r}: {exc}") from exc

        out = urlab_dm_env_rpc_pb2.UrlabPacket()
        if not resp.extension.Unpack(out):
            raise _StreamBroken(f"reply for {op!r} carried no UrlabPacket extension")
        # The server echoes the request's sequence_id; a mismatch means the stream
        # desynced (a stale reply from an abandoned call), so realign via reconnect
        # instead of returning the wrong frame. (0 == a legacy server that does not
        # echo -> skip the check for back-compat.)
        if out.sequence_id and out.sequence_id != seq:
            raise _StreamBroken(
                f"reply seq {out.sequence_id} != request seq {seq} for {op!r}"
            )
        reply = msgpack.unpackb(bytes(out.payload), raw=False, strict_map_key=False)
        if not isinstance(reply, dict):
            raise RuntimeError(f"non-dict reply: {type(reply).__name__}")
        return reply

    def _reset_stream(self) -> None:
        # Caller holds self._lock. A broken bidi stream can't be reused (the
        # request generator is bound to the dead call); drop it so the next rpc()
        # rebuilds a fresh channel + stream.
        try:
            self._req_q.put_nowait(None)  # end the old request generator
        except Exception:  # pragma: no cover
            pass
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:  # pragma: no cover
                pass
        self._channel = None
        self._resp_iter = None
        self._req_q = queue.Queue()
        # Replace the single-worker executor: a timed-out next() may still be
        # blocked on the old (now-closed) iterator, and a fresh executor keeps the
        # next call's recv from queueing behind that zombie thread.
        old_exec, self._recv_exec = self._recv_exec, concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="URLabDmEnvRecv"
        )
        old_exec.shutdown(wait=False)

    # -- viewer subscription (server-stream over gRPC) --------------------
    def start_viewer_stream(self, on_frame, *, endpoint=None) -> None:
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            return
        self._viewer_stop = threading.Event()
        self._viewer_thread = threading.Thread(
            target=self._viewer_loop, args=(on_frame,),
            name="URLabDmEnvViewerSub", daemon=True,
        )
        self._viewer_thread.start()

    def stop_viewer_stream(self) -> None:
        if self._viewer_stop is not None:
            self._viewer_stop.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=5.0)
            self._viewer_thread = None
        self._viewer_stop = None

    def _viewer_loop(self, on_frame) -> None:
        # Its own channel + a dedicated Process call: send one subscribe_viewer
        # request, keep the request stream open, and dispatch the server's
        # streamed viewer_frame packets to on_frame. Independent of rpc()'s stream.
        if msgpack is None:
            return
        # Capture the event locally: stop_viewer_stream() nulls self._viewer_stop
        # after join, but the gRPC request-generator thread below outlives that, so
        # it must not read the instance attribute.
        stop = self._viewer_stop
        if stop is None:
            return
        try:
            import grpc  # type: ignore
        except ImportError:  # pragma: no cover
            return
        from ._dmenv import dm_env_rpc_pb2, dm_env_rpc_pb2_grpc, urlab_dm_env_rpc_pb2

        backoff = 0.25
        while not stop.is_set():
            channel = None
            try:
                channel = grpc.insecure_channel(self._target, options=[
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                ])
                stub = dm_env_rpc_pb2_grpc.EnvironmentStub(channel)

                def _req_gen():
                    pkt = urlab_dm_env_rpc_pb2.UrlabPacket(
                        op="subscribe_viewer", payload=b"", sequence_id=1)
                    env = dm_env_rpc_pb2.EnvironmentRequest()
                    env.extension.Pack(pkt)
                    yield env
                    while not stop.is_set():  # hold the stream open
                        stop.wait(0.5)

                for resp in stub.Process(_req_gen()):
                    if stop.is_set():
                        break
                    backoff = 0.25
                    if not resp.HasField("extension"):
                        continue
                    out = urlab_dm_env_rpc_pb2.UrlabPacket()
                    if not resp.extension.Unpack(out) or out.op != "viewer_frame":
                        continue
                    try:
                        frame = msgpack.unpackb(
                            bytes(out.payload), raw=False, strict_map_key=False)
                    except Exception:  # noqa: BLE001
                        continue
                    try:
                        on_frame(frame)
                    except Exception:  # noqa: BLE001 - callback-defensive
                        pass
            except Exception:  # noqa: BLE001 - reconnect below
                pass
            finally:
                if channel is not None:
                    try:
                        channel.close()
                    except Exception:  # noqa: BLE001
                        pass
            if stop.is_set():
                break
            stop.wait(backoff)
            backoff = min(backoff * 2.0, 5.0)

    # -- streams (unsupported over dm_env_rpc here) ------------------------
    def start_state_stream(self, on_snapshot: SnapshotCallback) -> None:
        raise NotImplementedError(
            "GrpcTransport has no state stream; use transport='zmq' for the dashboard"
        )

    def stop_state_stream(self) -> None:
        return None

    def start_camera_stream(
        self, prefix: str, name: str, endpoint: str, topic: str,
        on_frame: FrameCallback,
    ) -> None:
        raise NotImplementedError(
            "GrpcTransport has no PUB/SUB camera stream; RenderClient pulls frames "
            "synchronously via rpc('fastpath_render')"
        )

    def stop_camera_streams(self) -> None:
        return None

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.stop_viewer_stream()
        with self._lock:
            try:
                self._req_q.put_nowait(None)  # let the server's Read loop exit
            except Exception:  # pragma: no cover
                pass
            if self._channel is not None:
                try:
                    self._channel.close()
                except Exception:  # pragma: no cover
                    pass
            self._channel = None
            self._resp_iter = None
        self._recv_exec.shutdown(wait=False)
