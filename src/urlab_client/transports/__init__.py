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
from urllib.parse import urlparse


# Hosts that mean "bind on every interface" on the server side. A SUB/REQ
# socket cannot connect to any of these; they must be rewritten to the host
# the RPC channel actually reached the server on.
_BIND_WILDCARD_HOSTS = frozenset({"0.0.0.0", "*", "::", ""})


def resolve_endpoint(advertised: str, rpc_address: str) -> str:
    """Rewrite a server-advertised bind endpoint into a connectable one.

    UE advertises camera / stream endpoints in *bind* form (``tcp://0.0.0.0:NNNN``
    or ``tcp://*:NNNN``), which a subscriber cannot connect to. Substitute the
    host the RPC channel is already talking to (``rpc_address``), keeping the
    advertised port. An endpoint that already names a concrete host is returned
    unchanged, so this is safe to apply unconditionally in tooling.

    ``rpc_address`` is the ``tcp://host`` (optionally with a port) the client
    used for its RPC socket; only its host is used.
    """
    if not advertised:
        return advertised
    adv = urlparse(advertised.replace("tcp://", "http://", 1))
    host = adv.hostname
    port = adv.port
    if port is None:
        # No parseable port -- take the trailing ":NNNN" and treat the rest as host.
        head, _, tail = advertised.rpartition(":")
        try:
            port = int(tail)
        except ValueError:
            return advertised
        host = head.replace("tcp://", "", 1)
    if host is not None and host not in _BIND_WILDCARD_HOSTS:
        return advertised
    rpc = urlparse(rpc_address.replace("tcp://", "http://", 1))
    rpc_host = rpc.hostname or rpc_address.replace("tcp://", "", 1).split(":", 1)[0]
    return f"tcp://{rpc_host}:{port}"


SnapshotCallback = Callable[[Mapping[str, Any]], None]
# (pixels, frame_id, sim_time, capture_time). All but pixels are None when the
# stream carries no metadata header (legacy server, or a malformed frame).
# capture_time is Unix-epoch seconds the frame was captured (v2+ header), or
# None on a v1 header that predates it.
FrameCallback = Callable[
    [bytes, Optional[int], Optional[float], Optional[float]], None]

# Per-frame metadata header prepended to streamed camera pixels on BOTH the
# ZMQ and SHM transports. Must match FMjCameraFrameMeta on the UE side.
#   v1 (32B): magic(u32) version(u32) frame_id(u64) sim_time(f64) width(u32) height(u32)
#   v2 (40B): ... + capture_unix_time(f64)   <- Unix seconds at capture
CAMERA_META_MAGIC = 0x314D4355  # 'UCM1' little-endian
CAMERA_META_STRUCT_V1 = struct.Struct("<IIQdII")     # 32
CAMERA_META_STRUCT_V2 = struct.Struct("<IIQdIId")    # 40
# Back-compat aliases (older imports expected a single 32-byte struct).
CAMERA_META_STRUCT = CAMERA_META_STRUCT_V1
CAMERA_META_SIZE = CAMERA_META_STRUCT_V1.size  # 32


def parse_camera_frame(
    payload: bytes,
) -> Tuple[bytes, Optional[int], Optional[float], Optional[float]]:
    """Split a streamed camera payload into (pixels, frame_id, sim_time,
    capture_time).

    The payload is ``[FMjCameraFrameMeta][pixels]`` (32B v1 or 40B v2 header).
    Version-tolerant: a v2 client reads ``capture_time`` from a v2 header and
    leaves it ``None`` for a v1 header; pixels start after the version's header
    size. If the magic doesn't match (legacy bare-pixel stream or a runt frame)
    the whole payload is returned as pixels with no metadata, so "latest" still
    works and "fresh" degrades gracefully.
    """
    if len(payload) >= CAMERA_META_STRUCT_V1.size:
        magic, ver, frame_id, sim_time, _w, _h = \
            CAMERA_META_STRUCT_V1.unpack_from(payload, 0)
        if magic == CAMERA_META_MAGIC:
            if ver >= 2 and len(payload) >= CAMERA_META_STRUCT_V2.size:
                (_m, _v, _f, _s, _w2, _h2, capture) = \
                    CAMERA_META_STRUCT_V2.unpack_from(payload, 0)
                return (payload[CAMERA_META_STRUCT_V2.size:],
                        int(frame_id), float(sim_time), float(capture))
            return (payload[CAMERA_META_STRUCT_V1.size:],
                    int(frame_id), float(sim_time), None)
    return payload, None, None, None


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

    # -- viewer bus (owner -> viewers) ------------------------------------
    # The owner of the simulation (the puppet client here) broadcasts one raw
    # kinematics frame per step onto a PUB socket that any number of read-only
    # viewers subscribe to. Default no-ops so a transport that has no viewer
    # PUB (SHM, in-process) is safe to call unconditionally; ZmqTransport
    # overrides them.

    def enable_viewer_broadcast(self, port: int) -> Optional[str]:
        """Bind the viewer PUB on ``port`` and return its endpoint, or None if
        this transport cannot broadcast. Idempotent."""
        return None

    def publish_viewer_state(self, payload: Mapping[str, Any]) -> None:
        """Broadcast one owner-authored frame (``{t, qpos, qvel}``) to viewers.
        No-op until :meth:`enable_viewer_broadcast` has bound the PUB."""
        return None


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
    rpc_req_path: Optional[str] = None,
    rpc_rep_path: Optional[str] = None,
    rpc_req_event: Optional[str] = None,
    rpc_rep_event: Optional[str] = None,
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
            rpc_req_path=rpc_req_path,
            rpc_rep_path=rpc_rep_path,
            rpc_req_event=rpc_req_event,
            rpc_rep_event=rpc_rep_event,
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
    "resolve_endpoint",
]
