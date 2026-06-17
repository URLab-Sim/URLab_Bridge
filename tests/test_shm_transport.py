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

"""Wire-level tests for the SHM state-stream transport (plan §13 Phase B).

Spins up a Python producer that writes the same FMjShmHeader layout the UE
publisher does, then verifies a ShmTransport consumer picks the snapshots
up via its `start_state_stream` callback. UE is not in the loop.
"""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time
from typing import Any, Dict, List

import pytest

from urlab_client.transports.shm import (
    SHM_HEADER_SIZE,
    SHM_OFF_LATEST_IDX,
    SHM_OFF_MAGIC,
    SHM_OFF_SEQUENCE,
    URLAB_SHM_MAGIC,
    URLAB_SHM_PROTOCOL_VERSION,
    ShmTransport,
)


def _create_state_shm(path: str, stride: int, n_buffers: int = 2) -> int:
    """Mirror FMjShmRegion::Open: truncate file, write header, zero slots.
    Returns the open file descriptor (caller closes)."""
    total = SHM_HEADER_SIZE + stride * n_buffers
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, total)
    mm = mmap.mmap(fd, total, access=mmap.ACCESS_WRITE)
    try:
        struct.pack_into(
            "<IIII", mm, 0,
            URLAB_SHM_MAGIC, URLAB_SHM_PROTOCOL_VERSION, stride, n_buffers,
        )
        struct.pack_into("<Q", mm, SHM_OFF_SEQUENCE, 0)
        struct.pack_into("<I", mm, SHM_OFF_LATEST_IDX, 0)
        # Body is implicitly zero from ftruncate.
    finally:
        mm.close()
    return fd


def _publish(path: str, stride: int, n_buffers: int, payload: bytes) -> None:
    """Mirror USmSnapshotPublisher::PublishSnapshot one-shot."""
    fd = os.open(path, os.O_RDWR)
    file_size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_WRITE)
    try:
        cur = struct.unpack_from("<I", mm, SHM_OFF_LATEST_IDX)[0]
        target = (cur + 1) & (n_buffers - 1) if n_buffers > 1 else 0
        slot_off = SHM_HEADER_SIZE + target * stride
        struct.pack_into("<I", mm, slot_off, len(payload))
        mm[slot_off + 4 : slot_off + 4 + len(payload)] = payload
        struct.pack_into("<I", mm, SHM_OFF_LATEST_IDX, target)
        seq = struct.unpack_from("<Q", mm, SHM_OFF_SEQUENCE)[0]
        struct.pack_into("<Q", mm, SHM_OFF_SEQUENCE, seq + 1)
        mm.flush()
    finally:
        mm.close()
        os.close(fd)


def test_state_stream_round_trip(msgpack_mod, tmp_path):
    """One snapshot in, one callback fire, decoded shape matches."""
    shm_dir = str(tmp_path)
    state_path = os.path.join(shm_dir, "state.shm")

    stride = 4096
    n_buffers = 2
    fd = _create_state_shm(state_path, stride, n_buffers)
    try:
        transport = ShmTransport(shm_dir, poll_interval_s=0.0005, open_timeout_s=2.0)
        received: List[Dict[str, Any]] = []
        evt = threading.Event()

        def on_snapshot(snap):
            received.append(dict(snap))
            evt.set()

        transport.start_state_stream(on_snapshot)
        try:
            # Give the reader a tick to mmap the region.
            time.sleep(0.05)
            payload = msgpack_mod.packb({
                "op": "state_full",
                "time": 1.5,
                "step": 7,
                "per_articulation": {"vx300s": {"qpos": [0.1, 0.2]}},
            }, use_bin_type=True)
            _publish(state_path, stride, n_buffers, payload)
            assert evt.wait(timeout=2.0), "no snapshot delivered"
        finally:
            transport.stop_state_stream()
    finally:
        os.close(fd)

    assert len(received) == 1
    snap = received[0]
    assert snap["op"] == "state_full"
    assert snap["time"] == pytest.approx(1.5)
    assert snap["step"] == 7
    assert "vx300s" in snap["per_articulation"]


def test_state_stream_many_snapshots(msgpack_mod, tmp_path):
    """Stream a burst of snapshots; reader picks up the latest each tick.
    With double buffering and a sequence counter the consumer may skip
    some intermediate frames; the contract is just "last snapshot wins"
    per poll, which matches UE semantics for free-running observation."""
    shm_dir = str(tmp_path)
    state_path = os.path.join(shm_dir, "state.shm")

    stride = 2048
    n_buffers = 2
    fd = _create_state_shm(state_path, stride, n_buffers)
    try:
        transport = ShmTransport(shm_dir, poll_interval_s=0.0005, open_timeout_s=2.0)
        latest_step: List[int] = []
        lock = threading.Lock()

        def on_snapshot(snap):
            with lock:
                latest_step.append(int(snap["step"]))

        transport.start_state_stream(on_snapshot)
        try:
            time.sleep(0.05)
            for k in range(50):
                payload = msgpack_mod.packb({"op": "state_full", "step": k},
                                            use_bin_type=True)
                _publish(state_path, stride, n_buffers, payload)
                time.sleep(0.001)
            # Drain a moment so the reader picks up the final snapshot.
            time.sleep(0.05)
        finally:
            transport.stop_state_stream()
    finally:
        os.close(fd)

    assert len(latest_step) >= 1
    # The reader must observe a strictly-increasing sequence of step ids
    # (it may skip values due to coalescing but must never go backwards).
    for prev, cur in zip(latest_step, latest_step[1:]):
        assert cur > prev, f"sequence regressed: {latest_step}"
    # And it must have caught the final value at some point.
    assert latest_step[-1] == 49


def test_open_timeout_returns_silently(tmp_path):
    """If the SHM file never appears, the reader thread exits cleanly
    without raising."""
    transport = ShmTransport(str(tmp_path), poll_interval_s=0.001, open_timeout_s=0.2)
    received: List[Any] = []
    transport.start_state_stream(lambda s: received.append(s))
    time.sleep(0.5)
    transport.stop_state_stream()
    assert received == []


def test_close_stops_reader_thread(msgpack_mod, tmp_path):
    """close() joins the reader thread cleanly even with no fallback."""
    shm_dir = str(tmp_path)
    state_path = os.path.join(shm_dir, "state.shm")
    fd = _create_state_shm(state_path, 1024, 2)
    try:
        transport = ShmTransport(shm_dir, poll_interval_s=0.001, open_timeout_s=2.0)
        transport.start_state_stream(lambda s: None)
        time.sleep(0.05)
        transport.close()
        # Calling close() twice must be a no-op.
        transport.close()
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# RPC ring-buffer wire-level tests.
# ---------------------------------------------------------------------------


class _ShmEchoServer:
    """Mimic USmStepTransport on the UE side: poll req.shm sequence, decode
    payload, run a user-supplied handler, write reply into rep.shm."""

    def __init__(self, msgpack_mod, shm_dir: str, *, stride: int = 4096,
                 n_buffers: int = 2, handler=None):
        self._msgpack = msgpack_mod
        self.shm_dir = shm_dir
        self.stride = stride
        self.n_buffers = n_buffers
        self.handler = handler or (lambda req: {"op": "echo", "echoed": req})
        self.req_path = os.path.join(shm_dir, "req.shm")
        self.rep_path = os.path.join(shm_dir, "rep.shm")
        os.makedirs(shm_dir, exist_ok=True)
        self.req_fd = _create_state_shm(self.req_path, stride, n_buffers)
        self.rep_fd = _create_state_shm(self.rep_path, stride, n_buffers)
        # Open mappings.
        sz = SHM_HEADER_SIZE + stride * n_buffers
        self.req_mm = mmap.mmap(self.req_fd, sz, access=mmap.ACCESS_WRITE)
        self.rep_mm = mmap.mmap(self.rep_fd, sz, access=mmap.ACCESS_WRITE)
        self.received = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        last_seen = struct.unpack_from("<Q", self.req_mm, SHM_OFF_SEQUENCE)[0]
        while not self._stop.is_set():
            cur = struct.unpack_from("<Q", self.req_mm, SHM_OFF_SEQUENCE)[0]
            if cur == last_seen:
                time.sleep(0.0005)
                continue
            idx = struct.unpack_from("<I", self.req_mm, SHM_OFF_LATEST_IDX)[0]
            slot_off = SHM_HEADER_SIZE + idx * self.stride
            size = struct.unpack_from("<I", self.req_mm, slot_off)[0]
            if size == 0 or size + 4 > self.stride:
                last_seen = cur
                continue
            payload = bytes(self.req_mm[slot_off + 4 : slot_off + 4 + size])
            last_seen = cur
            try:
                req = self._msgpack.unpackb(payload, raw=False, strict_map_key=False)
            except Exception:
                continue
            self.received.append(req)
            try:
                reply = self.handler(req)
            except Exception as exc:
                reply = {"op": "error", "code": "handler_raised",
                         "message": str(exc)}
            rep_bytes = self._msgpack.packb(reply, use_bin_type=True)

            cur_latest = struct.unpack_from(
                "<I", self.rep_mm, SHM_OFF_LATEST_IDX
            )[0]
            target = (cur_latest + 1) & (self.n_buffers - 1) \
                if self.n_buffers > 1 else 0
            rep_off = SHM_HEADER_SIZE + target * self.stride
            struct.pack_into("<I", self.rep_mm, rep_off, len(rep_bytes))
            self.rep_mm[rep_off + 4 : rep_off + 4 + len(rep_bytes)] = rep_bytes
            struct.pack_into("<I", self.rep_mm, SHM_OFF_LATEST_IDX, target)
            seq = struct.unpack_from("<Q", self.rep_mm, SHM_OFF_SEQUENCE)[0]
            struct.pack_into("<Q", self.rep_mm, SHM_OFF_SEQUENCE, seq + 1)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try: self.req_mm.close()
        except Exception: pass
        try: self.rep_mm.close()
        except Exception: pass
        try: os.close(self.req_fd)
        except Exception: pass
        try: os.close(self.rep_fd)
        except Exception: pass


def test_rpc_round_trip(msgpack_mod, tmp_path):
    """One RPC, one reply, payload survives the round trip."""
    server = _ShmEchoServer(msgpack_mod, str(tmp_path))
    try:
        transport = ShmTransport(
            str(tmp_path),
            poll_interval_s=0.0005,
            open_timeout_s=2.0,
            rpc_timeout_s=2.0,
        )
        try:
            reply = transport.rpc({"op": "hello", "session_id": "test", "n": 7})
        finally:
            transport.close()
    finally:
        server.close()
    assert reply["op"] == "echo"
    assert reply["echoed"]["op"] == "hello"
    assert reply["echoed"]["n"] == 7
    assert len(server.received) == 1
    assert server.received[0]["session_id"] == "test"


def test_rpc_many_sequential(msgpack_mod, tmp_path):
    """A burst of sequential RPCs all complete; each reply matches its
    request shape."""
    server = _ShmEchoServer(
        msgpack_mod, str(tmp_path),
        handler=lambda req: {"op": "step_ok", "i": req["i"]},
    )
    try:
        transport = ShmTransport(
            str(tmp_path),
            poll_interval_s=0.0005,
            open_timeout_s=2.0,
            rpc_timeout_s=2.0,
        )
        try:
            for i in range(20):
                reply = transport.rpc({"op": "step", "i": i})
                assert reply == {"op": "step_ok", "i": i}, f"iter {i}: {reply}"
        finally:
            transport.close()
    finally:
        server.close()
    assert len(server.received) == 20


def test_rpc_falls_back_when_files_missing(msgpack_mod, tmp_path):
    """If req/rep.shm don't exist, ShmTransport.rpc routes to fallback."""

    class _StubFallback(Transport_StubBase := __import__(
        "urlab_client.transports", fromlist=["Transport"]).Transport):
        # Minimal fallback that records what was sent.
        def __init__(self):
            self.sent = []

        def rpc(self, req, *, recv_timeout_ms=None):
            self.sent.append(dict(req))
            return {"op": "fallback_ok"}

        def start_state_stream(self, on_snapshot):
            pass

        def stop_state_stream(self):
            pass

        def start_camera_stream(self, *a, **kw):
            pass

        def stop_camera_streams(self):
            pass

        def close(self):
            pass

    fallback = _StubFallback()
    transport = ShmTransport(
        str(tmp_path),
        fallback=fallback,
        poll_interval_s=0.001,
        open_timeout_s=0.2,
        rpc_timeout_s=0.5,
    )
    try:
        reply = transport.rpc({"op": "hello"})
    finally:
        transport.close()
    assert reply == {"op": "fallback_ok"}
    assert fallback.sent == [{"op": "hello"}]


def test_camera_stream_round_trip(tmp_path):
    """A producer writes BGRA pixel buffers to cam_<prefix>_<name>.shm; the
    transport's camera reader thread delivers them to the on_frame callback."""
    shm_dir = str(tmp_path)
    cam_path = os.path.join(shm_dir, "cam_arm0_wrist.shm")

    # Allocate stride for a tiny 4x4 BGRA frame + 4-byte size prefix.
    width, height = 4, 4
    pixel_bytes = width * height * 4
    stride = pixel_bytes + 4
    n_buffers = 2
    fd = _create_state_shm(cam_path, stride, n_buffers)
    try:
        transport = ShmTransport(shm_dir, poll_interval_s=0.0005, open_timeout_s=2.0)
        received: List[bytes] = []
        evt = threading.Event()

        def on_frame(pixels: bytes, frame_id=None, sim_time=None,
                     capture_time=None) -> None:
            received.append(pixels)
            evt.set()

        transport.start_camera_stream(
            "arm0", "wrist", endpoint="tcp://*:5558",
            topic="arm0/camera/wrist", on_frame=on_frame,
        )
        try:
            time.sleep(0.05)  # let reader open the file
            payload = bytes(range(256))[:pixel_bytes]
            _publish(cam_path, stride, n_buffers, payload)
            assert evt.wait(timeout=2.0), "no camera frame delivered"
        finally:
            transport.stop_camera_streams()
    finally:
        os.close(fd)

    assert len(received) == 1
    # No metadata header -> whole payload passes through as pixels.
    assert received[0] == payload


def test_camera_stream_frame_meta_round_trip(tmp_path):
    """A frame written with the 40-byte v2 FMjCameraFrameMeta header is split by
    the reader into (pixels, frame_id, sim_time, capture_time)."""
    import struct as _struct

    from urlab_client.transports import CAMERA_META_MAGIC, CAMERA_META_STRUCT_V2

    shm_dir = str(tmp_path)
    cam_path = os.path.join(shm_dir, "cam_arm0_head.shm")

    width, height = 4, 4
    pixel_bytes = width * height * 4
    meta_size = CAMERA_META_STRUCT_V2.size  # 40
    stride = pixel_bytes + meta_size + 4  # + size prefix
    n_buffers = 2
    fd = _create_state_shm(cam_path, stride, n_buffers)
    try:
        transport = ShmTransport(shm_dir, poll_interval_s=0.0005, open_timeout_s=2.0)
        received = []
        evt = threading.Event()

        def on_frame(pixels, frame_id=None, sim_time=None,
                     capture_time=None) -> None:
            received.append((pixels, frame_id, sim_time, capture_time))
            evt.set()

        transport.start_camera_stream(
            "arm0", "head", endpoint="tcp://*:5558",
            topic="arm0/camera/head", on_frame=on_frame,
        )
        try:
            time.sleep(0.05)
            pixels = bytes(range(256))[:pixel_bytes]
            meta = CAMERA_META_STRUCT_V2.pack(
                CAMERA_META_MAGIC, 2, 4242, 1.5, width, height, 1234.5)
            _publish(cam_path, stride, n_buffers, meta + pixels)
            assert evt.wait(timeout=2.0), "no camera frame delivered"
        finally:
            transport.stop_camera_streams()
    finally:
        os.close(fd)

    assert len(received) == 1
    got_pixels, got_fid, got_time, got_capture = received[0]
    assert got_pixels == pixels
    assert got_fid == 4242
    assert got_time == 1.5
    assert got_capture == 1234.5


def test_rpc_timeout_when_server_silent(msgpack_mod, tmp_path):
    """If the SHM files exist but no server is replying, rpc raises
    TimeoutError after rpc_timeout_s."""
    # Manually create the regions so the open phase succeeds.
    req_path = os.path.join(str(tmp_path), "req.shm")
    rep_path = os.path.join(str(tmp_path), "rep.shm")
    fd_req = _create_state_shm(req_path, 1024, 2)
    fd_rep = _create_state_shm(rep_path, 1024, 2)
    try:
        transport = ShmTransport(
            str(tmp_path),
            poll_interval_s=0.001,
            open_timeout_s=2.0,
            rpc_timeout_s=0.3,
        )
        try:
            with pytest.raises(TimeoutError):
                transport.rpc({"op": "hello"})
        finally:
            transport.close()
    finally:
        os.close(fd_req)
        os.close(fd_rep)


def test_rpc_per_call_timeout_overrides_default(msgpack_mod, tmp_path):
    """ShmTransport.rpc(recv_timeout_ms=...) overrides the constructor
    default for that single call. Constructor default is very long
    (5s); the per-call override caps to 200ms; verify the call raises
    TimeoutError around 200ms, not 5s."""
    req_path = os.path.join(str(tmp_path), "req.shm")
    rep_path = os.path.join(str(tmp_path), "rep.shm")
    fd_req = _create_state_shm(req_path, 1024, 2)
    fd_rep = _create_state_shm(rep_path, 1024, 2)
    try:
        transport = ShmTransport(
            str(tmp_path),
            poll_interval_s=0.001,
            open_timeout_s=2.0,
            rpc_timeout_s=5.0,           # constructor default — long
        )
        try:
            t0 = time.monotonic()
            with pytest.raises(TimeoutError):
                transport.rpc({"op": "hello"}, recv_timeout_ms=200)
            elapsed = time.monotonic() - t0
            # Per-call override fires; full-budget (~5s) was bypassed.
            assert elapsed < 1.0, (
                f"per-call timeout did not override default: elapsed={elapsed:.3f}s"
            )
        finally:
            transport.close()
    finally:
        os.close(fd_req)
        os.close(fd_rep)
