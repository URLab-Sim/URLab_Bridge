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

"""Pluggable transports for `URLabClient`. ``ZmqTransport`` over TCP/IPC;
``ShmTransport`` over a same-host shared-memory ring (carries a ZMQ
fallback for ops too large for the SHM slot — notably hello, which
embeds the MJB)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional


SnapshotCallback = Callable[[Mapping[str, Any]], None]
FrameCallback = Callable[[bytes], None]


class Transport(ABC):
    """Abstract wire transport for URLabClient.

    Implementations must be safe to call `rpc()` from the main thread
    while `start_state_stream` / `start_camera_stream` callbacks fire on
    background threads. The transport owns its own threads and sockets;
    URLabClient never touches them directly.
    """

    @abstractmethod
    def rpc(
        self,
        request: Mapping[str, Any],
        *,
        rcv_timeout_ms: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """Send one msgpack-serialisable dict, return one dict reply.

        The transport is responsible for any encoding / decoding glue
        (msgpack on ZMQ, raw POD bytes on SHM, etc.). Raises a transport
        exception on connection / parse failure; URLab error replies are
        returned as-is so the caller can branch on `reply["op"] == "error"`.

        ``rcv_timeout_ms`` overrides the constructor-default recv timeout
        for this single call, then resets. Used for ops that legitimately
        block (e.g. ``begin_pie`` waiting on UE compile). Implementations
        that have no notion of a per-call timeout (in-process queues, etc.)
        may ignore the kwarg.
        """

    @abstractmethod
    def start_state_stream(self, on_snapshot: SnapshotCallback) -> None:
        """Spin up the state-snapshot stream. Idempotent. The callback is
        invoked from the transport's worker thread once per inbound
        snapshot."""

    @abstractmethod
    def stop_state_stream(self) -> None:
        """Tear the state stream down. Idempotent."""

    @abstractmethod
    def start_camera_stream(
        self,
        prefix: str,
        name: str,
        endpoint: str,
        topic: str,
        on_frame: FrameCallback,
    ) -> None:
        """Spin up a single camera stream. Idempotent per (prefix, name)."""

    @abstractmethod
    def stop_camera_streams(self) -> None:
        """Tear down every active camera stream. Idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Stop all streams and close the RPC channel. Idempotent."""


def make_transport(
    name: str,
    address: str,
    *,
    step_port: int = 5559,
    state_port: int = 5555,
    rcv_timeout_ms: int = 5000,
    shm_dir: Optional[str] = None,
    shm_session_id: str = "live",
    shm_open_timeout_s: float = 5.0,
    fallback: Optional[Transport] = None,
) -> Transport:
    """Pluggable transport factory. Supports `name in {"zmq", "shm"}`.

    For `"shm"`, `shm_dir` is required (the bridge passes the dir reported
    by the UE handshake, or the user's explicit override). The SHM
    transport always carries a ZMQ fallback for ops too large for the SHM
    slot (notably `hello`, which embeds the MJB) -- pass an existing one
    via `fallback`, or one will be constructed.
    """
    # Lazy imports so users can swap transports without dragging zmq/mmap
    # into a deployment that won't use them.
    if name == "zmq":
        from .zmq import ZmqTransport
        return ZmqTransport(
            address,
            step_port=step_port,
            state_port=state_port,
            rcv_timeout_ms=rcv_timeout_ms,
        )
    if name == "shm":
        from .zmq import ZmqTransport
        from .shm import ShmTransport
        if not shm_dir:
            raise ValueError("transport='shm' requires shm_dir")
        if fallback is None:
            fallback = ZmqTransport(
                address,
                step_port=step_port,
                state_port=state_port,
                rcv_timeout_ms=rcv_timeout_ms,
            )
        return ShmTransport(
            shm_dir,
            fallback=fallback,
            session_id=shm_session_id,
            open_timeout_s=shm_open_timeout_s,
        )
    raise ValueError(
        f"unknown transport name {name!r}; expected one of 'zmq', 'shm'"
    )


__all__ = [
    "FrameCallback",
    "SnapshotCallback",
    "Transport",
    "make_transport",
]
