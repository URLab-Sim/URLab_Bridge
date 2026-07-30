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

"""Shared-memory transport for URLabClient.

Wire layout: a 64-byte FMjShmHeader followed by `n_buffers` slots of
`buffer_stride` bytes each. The publisher writes the msgpack-encoded
snapshot into a `[u32 size][bytes...]` mini-frame inside the slot it is
about to flip into, then atomically updates `latest_idx` and bumps
`sequence`. Consumers poll `sequence`, pick up the slot at `latest_idx`,
and decode the mini-frame. SHM covers state stream, RPC, and per-camera
streams; the ZMQ fallback handles ops too large for the slot (notably
`hello`, which embeds the MJB).
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

from . import FrameCallback, SnapshotCallback, Transport, parse_camera_frame
from ..errors import URLabTimeoutError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - trivial import guard
    import msgpack  # type: ignore
except ImportError:  # pragma: no cover
    msgpack = None  # noqa: N816

# Ops whose reply we may transparently fetch over the ZMQ fallback after a
# SHM timeout. These are read-only / idempotent, so a slow-but-alive UE
# servicing the already-signalled SHM request in addition to the fallback
# request has no side effect beyond wasted work. Any op NOT listed here is
# treated as mutating: a SHM timeout on it raises rather than silently
# re-executing it (a resend would double-step, double-reset, double-spawn).
# This is a conservative allowlist -- an unknown op defaults to "mutating".
_RETRY_SAFE_OPS = frozenset({
    "hello",
    "meta",
    "op_status",
    "pie_status",
    "get_contacts",
    "read_mocap_pose",
    "list_keyframes",
    "list_actors",
    "list_blueprints",
    "snapshot",
    "get_actor_bounds",
    "actor_hierarchy",
})

# --- Windows kernel events --------------------------------------------------
# UE creates two named events per session (req_ready, rep_ready). The bridge
# opens them by name to replace polling on req/rep.shm with kernel-blocking
# waits. On non-Windows or when ctypes/win32 isn't usable, ShmTransport falls
# back to the polling code path automatically.
_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:  # pragma: no cover - platform-specific
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32
    _kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel32.OpenEventW.restype = wintypes.HANDLE
    _kernel32.SetEvent.argtypes = [wintypes.HANDLE]
    _kernel32.SetEvent.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _EVENT_MODIFY_STATE = 0x0002
    _SYNCHRONIZE = 0x00100000
    _EVENT_RIGHTS = _EVENT_MODIFY_STATE | _SYNCHRONIZE
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 0x102


def _open_named_event(name: str):
    """Open an existing Windows named event by name. Returns a HANDLE
    (truthy) on success, None on failure."""
    if not _IS_WINDOWS:
        return None
    handle = _kernel32.OpenEventW(_EVENT_RIGHTS, False, name)
    return handle if handle else None


def _set_event(handle) -> None:
    if _IS_WINDOWS and handle:
        _kernel32.SetEvent(handle)


def _wait_event(handle, timeout_ms: int) -> bool:
    """Returns True if the event was signalled, False on timeout."""
    if not (_IS_WINDOWS and handle):
        return False
    return _kernel32.WaitForSingleObject(handle, timeout_ms) == _WAIT_OBJECT_0


def _close_event(handle) -> None:
    if _IS_WINDOWS and handle:
        _kernel32.CloseHandle(handle)

# --- SHM region layout (mirrors FMjShmHeader in MjShmRegion.h) ---
URLAB_SHM_MAGIC = 0x42_4C_52_55  # 'URLB' little-endian
URLAB_SHM_PROTOCOL_VERSION = 1
SHM_HEADER_SIZE = 64
# Header field offsets (see Source/URLab/Public/MuJoCo/Net/MjShmRegion.h).
SHM_OFF_MAGIC = 0
SHM_OFF_PROTOCOL = 4
SHM_OFF_STRIDE = 8
SHM_OFF_NBUFFERS = 12
SHM_OFF_SEQUENCE = 16  # uint64
SHM_OFF_LATEST_IDX = 24  # uint32


class ShmTransport(Transport):
    """Same-host shared-memory transport.

    Reads / writes `<shm_dir>/{state,req,rep,cam_*}.shm` on the same host
    as UE. Carries a `fallback` (typically a ZmqTransport) for ops too
    large for the SHM slot. Construction does not block on the file --
    each reader thread retries until the publisher creates it.

    Parameters
    ----------
    shm_dir : str
        Directory holding the SHM region files. Must resolve to the same
        filesystem path on both sides; single-host only.
    fallback : Transport, optional
        Used for `rpc` ops that don't fit in the SHM slot, and for any
        camera stream whose SHM region never appears.
    poll_interval_s : float
        How often the reader thread checks the sequence counter. Default
        1 ms. Drop to 100 us for >1 kHz step rates.
    open_timeout_s : float
        How long `start_state_stream` waits for the SHM file to exist.
        Default 5 s. Beyond this the reader thread exits silently.
    """

    def __init__(
        self,
        shm_dir: str,
        *,
        fallback: Optional[Transport] = None,
        poll_interval_s: float = 0.001,
        open_timeout_s: float = 5.0,
        rpc_timeout_s: float = 5.0,
        session_id: str = "live",
        use_kernel_events: bool = True,
        rpc_req_path: Optional[str] = None,
        rpc_rep_path: Optional[str] = None,
        rpc_req_event: Optional[str] = None,
        rpc_rep_event: Optional[str] = None,
    ):
        self.shm_dir = shm_dir
        self._fallback = fallback
        self._poll_interval_s = poll_interval_s
        self._open_timeout_s = open_timeout_s
        self._rpc_timeout_s = rpc_timeout_s
        self._session_id = session_id
        self._use_kernel_events = use_kernel_events and _IS_WINDOWS

        # state.shm + cam_*.shm live on the per-PIE STREAM session (shm_dir).
        # The RPC region (req/rep.shm + its kernel events) lives on the RPC
        # transport's own session, given verbatim by the `shm_rpc` handshake
        # contract; fall back to the stream dir / session name for legacy
        # servers that don't advertise it.
        self._state_path = os.path.join(shm_dir, "state.shm")
        self._req_path = rpc_req_path or os.path.join(shm_dir, "req.shm")
        self._rep_path = rpc_rep_path or os.path.join(shm_dir, "rep.shm")
        self._req_event_name = rpc_req_event
        self._rep_event_name = rpc_rep_event
        self._state_thread: Optional[threading.Thread] = None
        self._state_stop: Optional[threading.Event] = None

        # Kernel-event handles (Windows only). Opened lazily in
        # `_ensure_rpc_open` so we don't fight a startup race with UE.
        self._req_event = None
        self._rep_event = None

        # Lazy-opened RPC mappings. The publisher creates these so the
        # bridge waits in `rpc()` until they appear.
        self._rpc_lock = threading.Lock()
        self._req_mm: Optional[mmap.mmap] = None
        self._req_fd: Optional[int] = None
        self._req_stride: int = 0
        self._req_nbufs: int = 0
        self._rep_mm: Optional[mmap.mmap] = None
        self._rep_fd: Optional[int] = None
        self._rep_stride: int = 0
        self._rep_nbufs: int = 0
        self._last_rep_seq: int = 0

        # Per-camera reader threads.
        self._cam_threads: Dict[Tuple[str, str], threading.Thread] = {}
        self._cam_stops: Dict[Tuple[str, str], threading.Event] = {}

        # Sticky route table: ops that produced a reply_too_large or a
        # timeout once get routed to the fallback for the rest of the
        # session, so we don't pay the SHM round-trip + timeout on every
        # call. Cleared on close().
        self._ops_routed_to_fallback: Set[str] = set()

    def _open_rpc_region(self, path: str):
        """Open and validate an existing RPC SHM file. Returns
        (fd, mm, stride, nbufs) or None if the file isn't ready."""
        if not os.path.isfile(path):
            return None
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            return None
        try:
            file_size = os.fstat(fd).st_size
            if file_size < SHM_HEADER_SIZE:
                os.close(fd)
                return None
            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_WRITE)
        except OSError:
            os.close(fd)
            return None
        magic, protocol, stride, nbufs = struct.unpack_from(
            "<IIII", mm, SHM_OFF_MAGIC
        )
        if magic != URLAB_SHM_MAGIC or protocol != URLAB_SHM_PROTOCOL_VERSION:
            mm.close()
            os.close(fd)
            return None
        return (fd, mm, stride, nbufs)

    def _ensure_rpc_open(self) -> bool:
        """Make sure both req.shm and rep.shm are open. Wait up to
        `open_timeout_s` for the publisher to create them. On Windows,
        also open the named kernel events so `rpc()` can wake/wait on
        UE without polling."""
        if self._req_mm is not None and self._rep_mm is not None:
            return True
        deadline = time.monotonic() + self._open_timeout_s
        while time.monotonic() < deadline:
            if self._req_mm is None:
                opened = self._open_rpc_region(self._req_path)
                if opened is not None:
                    self._req_fd, self._req_mm, self._req_stride, self._req_nbufs = opened
            if self._rep_mm is None:
                opened = self._open_rpc_region(self._rep_path)
                if opened is not None:
                    self._rep_fd, self._rep_mm, self._rep_stride, self._rep_nbufs = opened
                    # Snapshot the current sequence so we don't pick up a
                    # stale reply from a prior session.
                    self._last_rep_seq = struct.unpack_from(
                        "<Q", self._rep_mm, SHM_OFF_SEQUENCE
                    )[0]
            if self._req_mm is not None and self._rep_mm is not None:
                # Open the named events. UE creates them in TransportInit
                # alongside the SHM regions, so by the time both files exist
                # the events should be openable. If they're not (different
                # UE version, permissions, etc.), fall back to polling.
                if self._use_kernel_events and self._req_event is None:
                    req_name = (self._req_event_name
                                or f"Local\\URLab_{self._session_id}_req_ready")
                    rep_name = (self._rep_event_name
                                or f"Local\\URLab_{self._session_id}_rep_ready")
                    self._req_event = _open_named_event(req_name)
                    self._rep_event = _open_named_event(rep_name)
                    if self._req_event and self._rep_event:
                        logger.info(
                            "ShmTransport: using kernel events for RPC sync (%s, %s)",
                            req_name, rep_name,
                        )
                    else:
                        # Couldn't open one or both -- close the survivor and
                        # disable kernel events for this transport instance.
                        if self._req_event: _close_event(self._req_event); self._req_event = None
                        if self._rep_event: _close_event(self._rep_event); self._rep_event = None
                        self._use_kernel_events = False
                        logger.info(
                            "ShmTransport: kernel events unavailable, falling back to polling"
                        )
                return True
            time.sleep(self._poll_interval_s)
        return False

    # -- Transport: RPC + cameras delegate to the fallback ---------------

    def rpc(
        self,
        request: Mapping[str, Any],
        *,
        recv_timeout_ms: Optional[int] = None,
    ) -> Mapping[str, Any]:
        if msgpack is None:
            raise RuntimeError("msgpack not installed; cannot run SHM RPC")
        # Some replies (notably hello, which embeds the MJB) can exceed the
        # SHM slot stride. UE writes back a `reply_too_large` error in that
        # case; we route the same RPC through the fallback (ZMQ) so the
        # caller transparently gets a real reply.
        op = request.get("op")
        if op in self._ops_routed_to_fallback and self._fallback is not None:
            return self._fallback.rpc(request, recv_timeout_ms=recv_timeout_ms)

        with self._rpc_lock:
            if not self._ensure_rpc_open():
                if self._fallback is not None:
                    logger.debug("ShmTransport: req/rep.shm not ready, "
                                 "falling back for op=%r", op)
                    return self._fallback.rpc(request, recv_timeout_ms=recv_timeout_ms)
                raise RuntimeError(
                    "ShmTransport.rpc: req.shm/rep.shm not available at "
                    f"{self.shm_dir} after {self._open_timeout_s}s"
                )

            assert self._req_mm is not None and self._rep_mm is not None

            # Pack request and write into the next req slot via the
            # double-buffer pattern the publisher uses on the UE side.
            payload = msgpack.packb(dict(request), use_bin_type=True)
            if len(payload) + 4 > self._req_stride:
                # Too large for SHM; route through the ZMQ fallback (same
                # path reply-too-large uses) instead of raising directly.
                if self._fallback is not None:
                    return self._fallback.rpc(request, recv_timeout_ms=recv_timeout_ms)
                raise RuntimeError(
                    f"SHM request payload {len(payload)}B exceeds slot stride "
                    f"{self._req_stride}B"
                )
            # Re-snapshot the reply sequence immediately before publishing the
            # request. Reply correlation here is "any reply newer than this
            # snapshot", so a late reply from a PRIOR (timed-out or fallback)
            # RPC must not be mistaken for this call's answer. Without this
            # re-snap the stale reply's sequence bump would satisfy the check
            # below and the two calls' payloads would swap undetected.
            # A robust fix needs a request-id echoed by UE in the reply slot
            # (plugin coordination); until that field exists this single-flight
            # re-snapshot under `_rpc_lock` is the correct minimum.
            self._last_rep_seq = struct.unpack_from(
                "<Q", self._rep_mm, SHM_OFF_SEQUENCE
            )[0]
            cur_latest = struct.unpack_from(
                "<I", self._req_mm, SHM_OFF_LATEST_IDX
            )[0]
            target = (cur_latest + 1) % self._req_nbufs if self._req_nbufs > 1 else 0
            slot_off = SHM_HEADER_SIZE + target * self._req_stride
            struct.pack_into("<I", self._req_mm, slot_off, len(payload))
            self._req_mm[slot_off + 4 : slot_off + 4 + len(payload)] = payload
            struct.pack_into(
                "<I", self._req_mm, SHM_OFF_LATEST_IDX, target
            )
            seq_before = struct.unpack_from(
                "<Q", self._req_mm, SHM_OFF_SEQUENCE
            )[0]
            struct.pack_into(
                "<Q", self._req_mm, SHM_OFF_SEQUENCE, seq_before + 1
            )
            # No mmap.flush(): the region is shared RAM (both processes map the
            # same file), so writes are visible to UE without an msync to the
            # backing file. Cross-process ordering is the seqlock plus UE's
            # acquire load on the sequence, not a file flush.

            # Wake UE's worker. Auto-reset event self-clears after one waiter.
            if self._use_kernel_events and self._req_event:
                _set_event(self._req_event)

            # Wait for a new reply -- kernel-event blocking wait when
            # available, polling otherwise. Per-call recv_timeout_ms
            # overrides the constructor default (used for begin_pie etc).
            effective_timeout_s = (
                recv_timeout_ms / 1000.0
                if recv_timeout_ms is not None
                else self._rpc_timeout_s
            )
            deadline = time.monotonic() + effective_timeout_s
            while time.monotonic() < deadline:
                if self._use_kernel_events and self._rep_event:
                    # Block until UE signals or 50 ms elapses (so the outer
                    # while-loop can check `deadline`). 50 ms is a poll on
                    # the deadline only, not on shm itself -- the inner wait
                    # is purely event-driven.
                    remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                    wait_ms = min(remaining_ms, 50)
                    _wait_event(self._rep_event, wait_ms)
                    # Whether signalled or timed out, fall through to the
                    # sequence check; on a stale reply we just go around.
                rep_seq = struct.unpack_from(
                    "<Q", self._rep_mm, SHM_OFF_SEQUENCE
                )[0]
                if rep_seq != self._last_rep_seq:
                    rep_idx = struct.unpack_from(
                        "<I", self._rep_mm, SHM_OFF_LATEST_IDX
                    )[0]
                    if rep_idx >= self._rep_nbufs:
                        time.sleep(self._poll_interval_s)
                        continue
                    rep_off = SHM_HEADER_SIZE + rep_idx * self._rep_stride
                    rep_size = struct.unpack_from(
                        "<I", self._rep_mm, rep_off
                    )[0]
                    if rep_size == 0 or rep_size + 4 > self._rep_stride:
                        time.sleep(self._poll_interval_s)
                        continue
                    rep_bytes = self._rep_mm[
                        rep_off + 4 : rep_off + 4 + rep_size
                    ]
                    seq_after = struct.unpack_from(
                        "<Q", self._rep_mm, SHM_OFF_SEQUENCE
                    )[0]
                    if seq_after - rep_seq > self._rep_nbufs:
                        # Producer raced past; re-poll.
                        self._last_rep_seq = seq_after
                        continue
                    self._last_rep_seq = rep_seq
                    reply = msgpack.unpackb(
                        rep_bytes, raw=False, strict_map_key=False
                    )
                    if not isinstance(reply, dict):
                        raise RuntimeError(
                            f"non-dict SHM reply: {type(reply).__name__}"
                        )
                    # Two synthetic UE errors mean "this op cannot travel over
                    # SHM": `reply_too_large` (the real reply won't fit the
                    # slot -- UE DID execute it) and `wrong_transport` (an
                    # EditorOnly op like begin_pie/stop_pie/op_status was sent
                    # on SHM -- UE rejected it BEFORE executing). Both are
                    # handled identically and safely: sticky-reroute the op to
                    # the fallback and re-run it there, forwarding the caller's
                    # recv_timeout_ms so a long editor op (30s hello refetch,
                    # begin_pie) does not die at the fallback's default.
                    if (reply.get("op") == "error"
                            and reply.get("code") in ("reply_too_large",
                                                      "wrong_transport")
                            and self._fallback is not None):
                        if op:
                            self._ops_routed_to_fallback.add(op)
                        logger.info(
                            "ShmTransport: op=%r not serviceable over SHM "
                            "(%s); rerouting through fallback transport",
                            op, reply.get("code"),
                        )
                        return self._fallback.rpc(
                            request, recv_timeout_ms=recv_timeout_ms
                        )
                    return reply
                # Reply not yet ready. With kernel events the wait above
                # already blocked for up to 50ms; fall through immediately
                # so the next iteration re-waits. Without events, sleep to
                # avoid busy-spinning.
                if not (self._use_kernel_events and self._rep_event):
                    time.sleep(self._poll_interval_s)

            # No reply within the timeout. Critical safety rule: the request
            # was already written to the slot and signalled, so a slow-but-
            # alive UE may still execute it. Transparently resending over ZMQ
            # would make UE execute the op TWICE -- a double step, a double
            # reset/set_qpos, a duplicate spawn. Only retry ops on the
            # read-only / idempotent allowlist; everything else raises.
            if op in _RETRY_SAFE_OPS and self._fallback is not None:
                if op:
                    self._ops_routed_to_fallback.add(op)
                logger.info(
                    "ShmTransport: read-only op=%r timed out; retrying over "
                    "fallback transport", op,
                )
                return self._fallback.rpc(request, recv_timeout_ms=recv_timeout_ms)
            raise URLabTimeoutError(
                f"SHM RPC {op!r}", waited_s=effective_timeout_s, op=op,
            )

    def start_camera_stream(
        self,
        prefix: str,
        name: str,
        endpoint: str,
        topic: str,
        on_frame: Callable[[bytes], None],
    ) -> None:
        """Open `cam_<prefix>_<name>.shm` and spin up a reader thread. If
        the file isn't present within `open_timeout_s`, fall back to ZMQ
        (when `fallback` is configured) or exit silently."""
        cam_path = os.path.join(self.shm_dir, f"cam_{prefix}_{name}.shm")
        key = (prefix, name)
        if key in self._cam_threads and self._cam_threads[key].is_alive():
            return
        self._cam_stops[key] = threading.Event()
        t = threading.Thread(
            target=self._camera_loop,
            args=(key, cam_path, endpoint, topic, on_frame),
            name=f"URLabShmCam-{prefix}-{name}",
            daemon=True,
        )
        self._cam_threads[key] = t
        t.start()

    def stop_camera_streams(self) -> None:
        for ev in self._cam_stops.values():
            ev.set()
        for t in self._cam_threads.values():
            t.join(timeout=1.0)
        self._cam_stops.clear()
        self._cam_threads.clear()
        if self._fallback is not None:
            self._fallback.stop_camera_streams()

    def _camera_loop(
        self,
        key: Tuple[str, str],
        cam_path: str,
        endpoint: str,
        topic: str,
        on_frame: Callable[[bytes], None],
    ) -> None:
        # Wait for the SHM file. If it never appears, fall back to ZMQ.
        deadline = time.monotonic() + self._open_timeout_s
        while time.monotonic() < deadline:
            if self._cam_stops[key].is_set():
                return
            if os.path.isfile(cam_path):
                break
            time.sleep(self._poll_interval_s)
        else:
            if self._fallback is not None:
                logger.debug(
                    "ShmTransport: camera %s/%s SHM unavailable, ZMQ fallback",
                    key[0], key[1],
                )
                self._fallback.start_camera_stream(
                    key[0], key[1], endpoint, topic, on_frame
                )
            return

        try:
            fd = os.open(cam_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
        except OSError as exc:
            logger.debug("ShmTransport: camera mmap failed for %s: %s",
                         cam_path, exc)
            return
        try:
            magic, protocol, stride, nbufs = struct.unpack_from(
                "<IIII", mm, SHM_OFF_MAGIC
            )
            if magic != URLAB_SHM_MAGIC or protocol != URLAB_SHM_PROTOCOL_VERSION:
                logger.debug("ShmTransport: camera %s header mismatch", cam_path)
                return
            last_seq = 0
            stop_ev = self._cam_stops[key]
            while not stop_ev.is_set():
                seq = struct.unpack_from("<Q", mm, SHM_OFF_SEQUENCE)[0]
                if seq == last_seq:
                    time.sleep(self._poll_interval_s)
                    continue
                idx = struct.unpack_from("<I", mm, SHM_OFF_LATEST_IDX)[0]
                if idx >= nbufs:
                    last_seq = seq
                    continue
                slot_off = SHM_HEADER_SIZE + idx * stride
                size = struct.unpack_from("<I", mm, slot_off)[0]
                if size == 0 or size + 4 > stride:
                    last_seq = seq
                    continue
                # Payload is [meta][pixels]; size covers both.
                payload = bytes(mm[slot_off + 4 : slot_off + 4 + size])
                seq_after = struct.unpack_from("<Q", mm, SHM_OFF_SEQUENCE)[0]
                if seq_after - seq > nbufs:
                    last_seq = seq_after
                    continue
                last_seq = seq
                pixels, frame_id, sim_time, capture_time = parse_camera_frame(payload)
                try:
                    on_frame(pixels, frame_id, sim_time, capture_time)
                except Exception as exc:  # pragma: no cover - callback-defensive
                    logger.debug(
                        "ShmTransport: camera frame callback raised: %s", exc
                    )
        finally:
            try:
                mm.close()
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass

    # -- state stream (SHM) ---------------------------------------------

    def start_state_stream(self, on_snapshot: SnapshotCallback) -> None:
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        self._state_stop = threading.Event()
        self._state_thread = threading.Thread(
            target=self._state_loop,
            args=(on_snapshot,),
            name="URLabShmStateSub",
            daemon=True,
        )
        self._state_thread.start()

    def stop_state_stream(self) -> None:
        if self._state_stop is not None:
            self._state_stop.set()
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
            self._state_thread = None
        self._state_stop = None

    def _open_state_region(self) -> Optional[Tuple[Any, mmap.mmap, int, int]]:
        """Wait for the SHM file to appear, validate its header, and return
        an mmap'd view + parsed (stride, n_buffers). Returns None on
        timeout or header mismatch."""
        deadline = time.monotonic() + self._open_timeout_s
        while time.monotonic() < deadline:
            if (self._state_stop is not None and self._state_stop.is_set()):
                return None
            if os.path.isfile(self._state_path):
                try:
                    fd = os.open(self._state_path, os.O_RDONLY)
                except OSError:
                    time.sleep(self._poll_interval_s)
                    continue
                try:
                    file_size = os.fstat(fd).st_size
                    if file_size < SHM_HEADER_SIZE:
                        # Producer hasn't finished initialising yet.
                        os.close(fd)
                        time.sleep(self._poll_interval_s)
                        continue
                    mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
                except OSError as exc:
                    os.close(fd)
                    logger.debug("ShmTransport: mmap failed for %s: %s",
                                 self._state_path, exc)
                    time.sleep(self._poll_interval_s)
                    continue

                # Header validation. magic + protocol must match.
                magic, protocol, stride, nbufs = struct.unpack_from(
                    "<IIII", mm, SHM_OFF_MAGIC
                )
                if magic != URLAB_SHM_MAGIC:
                    logger.debug("ShmTransport: magic mismatch (0x%x); retrying", magic)
                    mm.close()
                    os.close(fd)
                    time.sleep(self._poll_interval_s)
                    continue
                if protocol != URLAB_SHM_PROTOCOL_VERSION:
                    logger.error(
                        "ShmTransport: protocol mismatch (file=%d, client=%d)",
                        protocol, URLAB_SHM_PROTOCOL_VERSION,
                    )
                    mm.close()
                    os.close(fd)
                    return None
                logger.info(
                    "ShmTransport: opened %s (stride=%d, n_buffers=%d)",
                    self._state_path, stride, nbufs,
                )
                return (fd, mm, stride, nbufs)
            time.sleep(self._poll_interval_s)
        logger.warning("ShmTransport: state.shm did not appear at %s within %.1fs",
                       self._state_path, self._open_timeout_s)
        return None

    def _state_loop(self, on_snapshot: SnapshotCallback) -> None:
        if msgpack is None:
            return
        opened = self._open_state_region()
        if opened is None:
            return
        fd, mm, stride, nbufs = opened
        last_seq = 0
        try:
            assert self._state_stop is not None
            while not self._state_stop.is_set():
                # Acquire-load via reading the sequence first; the publisher
                # writes the slot, then updates latest_idx (release-store),
                # then bumps the sequence. Reading sequence > latest_idx >
                # buffer in this order gives us a happens-before chain.
                seq = struct.unpack_from("<Q", mm, SHM_OFF_SEQUENCE)[0]
                if seq == last_seq:
                    time.sleep(self._poll_interval_s)
                    continue
                idx = struct.unpack_from("<I", mm, SHM_OFF_LATEST_IDX)[0]
                if idx >= nbufs:
                    last_seq = seq
                    continue
                slot_off = SHM_HEADER_SIZE + idx * stride
                size = struct.unpack_from("<I", mm, slot_off)[0]
                if size == 0 or size + 4 > stride:
                    last_seq = seq
                    continue
                payload = mm[slot_off + 4 : slot_off + 4 + size]
                # Re-read sequence; if it changed during the copy we got a
                # torn read (publisher swapped indices and overwrote our
                # slot). Skip and try again next tick.
                seq_after = struct.unpack_from("<Q", mm, SHM_OFF_SEQUENCE)[0]
                if seq_after - seq > nbufs:
                    # Producer has wrapped around past our slot. Drop.
                    last_seq = seq_after
                    continue
                last_seq = seq
                try:
                    snap = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                except Exception as exc:
                    logger.debug("ShmTransport: state decode failed: %s", exc)
                    continue
                try:
                    on_snapshot(snap)
                except Exception as exc:  # pragma: no cover - callback-defensive
                    logger.debug("ShmTransport: snapshot callback raised: %s", exc)
        finally:
            try:
                mm.close()
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass

    # -- lifecycle -------------------------------------------------------

    def close(self, *, close_fallback: bool = True) -> None:
        """Stop all streams and close the RPC mappings. ``close_fallback``
        defaults to True (the Transport contract). The client passes
        ``close_fallback=False`` when it rebuilds the SHM transport across a
        PIE restart and wants to carry the shared ZMQ fallback over to the
        replacement instead of tearing it down."""
        self.stop_state_stream()
        self.stop_camera_streams()
        with self._rpc_lock:
            if self._req_mm is not None:
                try:
                    self._req_mm.close()
                except Exception:
                    pass
                self._req_mm = None
            if self._req_fd is not None:
                try:
                    os.close(self._req_fd)
                except Exception:
                    pass
                self._req_fd = None
            if self._rep_mm is not None:
                try:
                    self._rep_mm.close()
                except Exception:
                    pass
                self._rep_mm = None
            if self._rep_fd is not None:
                try:
                    os.close(self._rep_fd)
                except Exception:
                    pass
                self._rep_fd = None
            if self._req_event is not None:
                _close_event(self._req_event)
                self._req_event = None
            if self._rep_event is not None:
                _close_event(self._rep_event)
                self._rep_event = None
        if self._fallback is not None and close_fallback:
            self._fallback.close()
