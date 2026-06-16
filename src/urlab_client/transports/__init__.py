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

import struct
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional, Tuple


SnapshotCallback = Callable[[Mapping[str, Any]], None]
# (pixels, frame_id, sim_time). frame_id / sim_time are None when the stream
# carries no metadata header (legacy server, or a malformed frame).
FrameCallback = Callable[[bytes, Optional[int], Optional[float]], None]

# Per-frame metadata header prepended to streamed camera pixels on BOTH the
# ZMQ and SHM transports. Must match FMjCameraFrameMeta on the UE side: a
# fixed 32-byte little-endian POD, layout "<IIQdII":
#   magic(u32) version(u32) frame_id(u64) sim_time(f64) width(u32) height(u32)
CAMERA_META_MAGIC = 0x314D4355  # 'UCM1' little-endian
CAMERA_META_STRUCT = struct.Struct("<IIQdII")
CAMERA_META_SIZE = CAMERA_META_STRUCT.size  # 32


def parse_camera_frame(payload: bytes) -> Tuple[bytes, Optional[int], Optional[float]]:
    """Split a streamed camera payload into (pixels, frame_id, sim_time).

    The payload is ``[FMjCameraFrameMeta (32 bytes)][pixels]``. If the leading
    magic doesn't match (older server that streams bare pixels, or a runt
    frame) the whole payload is returned as pixels with no frame_id, so the
    "latest" query still works and "fresh" gracefully degrades to "latest".
    """
    if len(payload) >= CAMERA_META_SIZE:
        magic, _ver, frame_id, sim_time, _w, _h = CAMERA_META_STRUCT.unpack_from(payload, 0)
        if magic == CAMERA_META_MAGIC:
            return payload[CAMERA_META_SIZE:], int(frame_id), float(sim_time)
    return payload, None, None


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
        recv_timeout_ms: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """Send one msgpack-serialisable dict, return one dict reply.

        The transport is responsible for any encoding / decoding glue
        (msgpack on ZMQ, raw POD bytes on SHM, etc.). Raises a transport
        exception on connection / parse failure; URLab error replies are
        returned as-is so the caller can branch on `reply["op"] == "error"`.

        ``recv_timeout_ms`` overrides the constructor-default recv timeout
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
    recv_timeout_ms: int = 5000,
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
            recv_timeout_ms=recv_timeout_ms,
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
                recv_timeout_ms=recv_timeout_ms,
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
    "CAMERA_META_MAGIC",
    "CAMERA_META_SIZE",
    "FrameCallback",
    "SnapshotCallback",
    "Transport",
    "make_transport",
    "parse_camera_frame",
]
