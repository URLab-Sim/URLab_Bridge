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

# Default gRPC port the UE dm_env_rpc backend listens on (ListenPort).
DEFAULT_DMENV_PORT = 50051


class GrpcTransport(Transport):
    """dm_env_rpc bidi-stream RPC transport (request/reply only)."""

    def __init__(
        self,
        address: str = "tcp://127.0.0.1",
        *,
        step_port: int = DEFAULT_DMENV_PORT,
        recv_timeout_ms: int = 5000,
    ) -> None:
        # gRPC targets are bare "host:port"; accept a tcp://host[:port] address
        # for parity with the other transports and take only its host.
        host = address.replace("tcp://", "", 1).split("/", 1)[0].split(":", 1)[0]
        self._target = f"{host or '127.0.0.1'}:{step_port}"
        self._recv_timeout_ms = recv_timeout_ms

        self._lock = threading.Lock()
        self._seq = 0
        self._channel: Any = None
        self._resp_iter: Any = None
        self._req_q: "queue.Queue[Any]" = queue.Queue()
        # A single-thread executor lets us bound each blocking next() on the
        # response iterator by a per-call timeout without cancelling gRPC's own
        # (stream-wide) deadline.
        self._recv_exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="URLabDmEnvRecv"
        )

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
        from ._dmenv import dm_env_rpc_pb2_grpc  # lazy: only when grpc is used

        # -1 == unlimited, matching the server's SetMax*MessageSize(-1); camera
        # frames (BGRA8 at 1280x720+) exceed the 4 MB gRPC default.
        self._channel = grpc.insecure_channel(
            self._target,
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )
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
        from ._dmenv import dm_env_rpc_pb2, urlab_dm_env_rpc_pb2

        payload = msgpack.packb(dict(request), use_bin_type=True)
        timeout_ms = (
            int(recv_timeout_ms) if recv_timeout_ms is not None
            else self._recv_timeout_ms
        )
        op = str(request.get("op", ""))

        with self._lock:
            self._ensure_stream()
            self._seq += 1
            packet = urlab_dm_env_rpc_pb2.UrlabPacket(
                op=op, payload=bytes(payload), sequence_id=self._seq
            )
            env = dm_env_rpc_pb2.EnvironmentRequest()
            env.extension.Pack(packet)
            self._req_q.put(env)

            try:
                # next() blocks until UE renders + replies; bound it so a dead
                # server surfaces as a timeout instead of hanging the caller.
                resp = self._recv_exec.submit(next, self._resp_iter).result(
                    timeout=timeout_ms / 1000.0
                )
            except concurrent.futures.TimeoutError as exc:
                self._reset_stream()
                raise URLabTimeoutError(
                    f"RPC {op!r} over gRPC",
                    waited_s=timeout_ms / 1000.0,
                    op=op,
                ) from exc
            except StopIteration as exc:
                self._reset_stream()
                raise RuntimeError(
                    f"gRPC stream closed by server during {op!r}"
                ) from exc
            except Exception:
                self._reset_stream()
                raise

            out = urlab_dm_env_rpc_pb2.UrlabPacket()
            if not resp.extension.Unpack(out):
                raise RuntimeError(
                    f"gRPC reply for {op!r} carried no UrlabPacket extension"
                )
            reply = msgpack.unpackb(
                bytes(out.payload), raw=False, strict_map_key=False
            )

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
