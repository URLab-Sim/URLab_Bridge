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

#!/usr/bin/env python3
# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Benchmark camera streaming over ZMQ vs SHM transports.

Measures sustained frame rate (FPS), inter-frame gap distribution
(p50/p99/max), and time-to-first-frame for each camera × each transport
combination.

Run from the bridge repo root after starting URLab in the editor with at
least one camera that has streaming enabled and the broadcast flags set:

    uv run python scripts/bench_cameras.py --duration 5

Required UE-side setup per camera:
    - SetStreamingEnabled(true)            -- otherwise nothing is captured
    - bEnableZmqBroadcast = true           -- to bench ZMQ streaming
    - bEnableShmBroadcast = true           -- to bench SHM streaming
                                              (writes <Saved>/URLabShm/live/cam_<prefix>_<name>.shm)

Optional knobs:
    --address      ZMQ endpoint for the discovery handshake (default tcp://localhost)
    --shm-dir      SHM session directory (default <Saved>/URLabShm/live)
    --duration     seconds to sample per transport (default 5)
    --warmup       seconds to wait before counting (default 1)
    --transports   subset of "zmq,shm" (default both)
    --poll-us      bridge SHM poll interval in microseconds (default 100)

Notes:
    - End-to-end latency (publish→receive) requires UE to stamp each frame
      with a publish-time. The current wire format doesn't carry that, so
      this script reports inter-frame jitter on the receive side instead.
      That captures most of what you'd want to see (queueing, polling,
      kernel-loopback delays) -- it just doesn't separate one-way delay
      from per-frame variance.
    - Both transports run on the SAME UE instance; the bench does one
      transport at a time so they don't interfere with each other.
    - If UE is rendering at 60 FPS, expect ~60 FPS per camera regardless
      of transport (frame rate is producer-bound, not transport-bound).
      The interesting result is whether SHM has tighter jitter than ZMQ.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Tuple

# Make the urlab_policy package importable when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client.transports import Transport, resolve_endpoint  # noqa: E402
from urlab_client.transports.shm import ShmTransport  # noqa: E402
from urlab_client.transports.zmq import ZmqTransport  # noqa: E402


CameraSpec = Tuple[str, str, str, str, Tuple[int, int]]  # prefix, name, endpoint, topic, (w,h)


def _percentile(samples: List[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _discover_cameras(address: str) -> List[CameraSpec]:
    """Handshake, ENABLE per-camera broadcast, and return the connectable
    endpoints.

    UE's ``bEnableAllCameras`` defaults off, so a camera only binds its real
    (distinct) ZMQ port once ``set_camera_streaming`` is called; the endpoint
    advertised in the bare handshake is a shared placeholder. We enable
    streaming and read the bound endpoint back from that reply, then rewrite
    its bind-wildcard host to the RPC host so a subscriber can actually connect.
    """
    transport = ZmqTransport(address)
    try:
        reply = transport.rpc({
            "op": "hello",
            "client_version": "bench_cameras/1",
            "observations": "minimal",
            "encoding": "msgpack",
        })
        if reply.get("op") != "hello_ok":
            raise RuntimeError(
                f"unexpected hello reply op={reply.get('op')!r}: {reply}"
            )

        discovered: Dict[str, Tuple[str, Tuple[int, int]]] = {}
        cams: List[CameraSpec] = []
        for art in reply.get("articulations", []):
            prefix = art.get("prefix", "")
            for cam_name, cam_info in (art.get("camera_topics", {}) or {}).items():
                endpoint = cam_info.get("zmq_endpoint", "") or ""
                topic = cam_info.get("zmq_topic", "") or ""
                res = cam_info.get("resolution", [0, 0]) or [0, 0]
                discovered[cam_name] = (prefix, (int(res[0]), int(res[1])))
                cams.append((prefix, cam_name, endpoint, topic,
                             (int(res[0]), int(res[1]))))

        if not cams:
            return cams

        # Enable broadcast on every camera and use the bound endpoints/topics
        # the reply reports (falling back to the handshake values otherwise).
        enable = {name: {"zmq": True, "shm": True} for name in discovered}
        stream_reply = transport.rpc({
            "op": "set_camera_streaming",
            "cameras": enable,
        })
        bound = stream_reply.get("cameras") or {}
        resolved: List[CameraSpec] = []
        for prefix, cam_name, endpoint, topic, wh in cams:
            info = bound.get(cam_name) or {}
            endpoint = info.get("zmq_endpoint") or endpoint
            topic = info.get("zmq_topic") or topic
            resolved.append(
                (prefix, cam_name, resolve_endpoint(endpoint, address), topic, wh)
            )
        return resolved
    finally:
        transport.close()


def _bench_transport(
    transport: Transport,
    cameras: List[CameraSpec],
    duration_s: float,
    warmup_s: float,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Subscribe to every camera through `transport`, sample, return per-cam stats."""
    arrivals: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    arrivals_lock = threading.Lock()
    first_frame_at: Dict[Tuple[str, str], float] = {}
    subscribed_at = time.perf_counter()

    def make_cb(key: Tuple[str, str]):
        # Transport frame callbacks are 4-arg: (pixels, frame_id, sim_time,
        # capture_time). A 1-arg callback raises inside the transport and every
        # frame is silently dropped -- which is what made this bench always
        # report 0 frames.
        def _cb(_pixels: bytes, _frame_id=None, _sim_time=None,
                _capture_time=None) -> None:
            t = time.perf_counter()
            with arrivals_lock:
                if key not in first_frame_at:
                    first_frame_at[key] = t
                arrivals[key].append(t)
        return _cb

    for prefix, name, endpoint, topic, _res in cameras:
        transport.start_camera_stream(prefix, name, endpoint, topic, make_cb((prefix, name)))

    # Warmup (frames received during this window are discarded).
    time.sleep(warmup_s)
    with arrivals_lock:
        # Snapshot first-frame times before clearing.
        ttf = {k: v - subscribed_at for k, v in first_frame_at.items()}
        arrivals.clear()

    t_start = time.perf_counter()
    time.sleep(duration_s)
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    transport.stop_camera_streams()

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with arrivals_lock:
        for key, times in arrivals.items():
            in_window = [t for t in times if t_start <= t <= t_end]
            if len(in_window) < 2:
                results[key] = {
                    "frames": len(in_window),
                    "fps": len(in_window) / elapsed if elapsed > 0 else 0.0,
                    "gap_p50_ms": float("nan"),
                    "gap_p99_ms": float("nan"),
                    "gap_max_ms": float("nan"),
                    "ttf_ms": ttf.get(key, float("nan")) * 1000,
                }
                continue
            gaps = [b - a for a, b in zip(in_window, in_window[1:])]
            results[key] = {
                "frames": len(in_window),
                "fps": len(in_window) / elapsed,
                "gap_p50_ms": statistics.median(gaps) * 1000,
                "gap_p99_ms": _percentile(gaps, 0.99) * 1000,
                "gap_max_ms": max(gaps) * 1000,
                "ttf_ms": ttf.get(key, float("nan")) * 1000,
            }
    return results


def _make_transport(name: str, address: str, shm_dir: str, poll_us: int) -> Transport:
    if name == "zmq":
        return ZmqTransport(address)
    if name == "shm":
        # No fallback: we want pure-SHM measurements. If SHM files don't
        # exist (cameras don't have bEnableShmBroadcast set), the bench
        # will report 0 frames -- exactly the signal we want.
        return ShmTransport(
            shm_dir,
            fallback=None,
            poll_interval_s=poll_us / 1_000_000.0,
            open_timeout_s=2.0,
        )
    raise ValueError(f"unknown transport {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="tcp://localhost",
                        help="ZMQ address used for the discovery handshake")
    parser.add_argument("--shm-dir", default=None,
                        help="SHM session directory (default <Saved>/URLabShm/live)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="seconds to sample per transport")
    parser.add_argument("--warmup", type=float, default=1.0,
                        help="seconds to discard at the start of each run")
    parser.add_argument("--transports", default="zmq,shm",
                        help="comma-separated transports to bench")
    parser.add_argument("--poll-us", type=int, default=100,
                        help="bridge SHM poll interval in microseconds")
    args = parser.parse_args()

    if args.shm_dir is None:
        args.shm_dir = os.path.join(
            os.path.expanduser("~"),
            "Documents", "Unreal Projects", "url_proj",
            "Saved", "URLabShm", "live",
        )

    transports = [t.strip() for t in args.transports.split(",") if t.strip()]

    print("Camera bench configuration:")
    print(f"  address     = {args.address}")
    print(f"  shm_dir     = {args.shm_dir}")
    print(f"  duration    = {args.duration}s  warmup={args.warmup}s")
    print(f"  transports  = {transports}")
    print(f"  poll_us     = {args.poll_us}")
    print()

    try:
        cameras = _discover_cameras(args.address)
    except Exception as exc:
        print(f"[discover] failed: {exc}")
        return 1

    if not cameras:
        print("No cameras advertised in the handshake. Verify each UMjCamera has:")
        print("  - SetStreamingEnabled(true)")
        print("  - bEnableZmqBroadcast = true  (for ZMQ)")
        print("  - bEnableShmBroadcast = true  (for SHM)")
        print("  - a unique ZmqEndpoint per camera")
        return 1

    print(f"Discovered {len(cameras)} camera(s):")
    for prefix, name, endpoint, topic, (w, h) in cameras:
        print(f"  - {prefix}/{name}   {w}x{h}   endpoint={endpoint}   topic={topic}")
    print()

    rows: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for transport_name in transports:
        try:
            print(f"== {transport_name} ==")
            t = _make_transport(transport_name, args.address, args.shm_dir, args.poll_us)
        except Exception as exc:
            print(f"  [{transport_name}] make-transport failed: {exc}")
            print()
            continue

        try:
            results = _bench_transport(t, cameras, args.duration, args.warmup)
        except Exception as exc:
            print(f"  [{transport_name}] bench failed: {exc}")
            results = {}
        finally:
            try:
                t.close()
            except Exception:
                pass

        if not results:
            print(f"  [{transport_name}] no frames received -- check broadcast flags on UE")

        for key in sorted(results.keys()):
            stats = results[key]
            rows.append((transport_name, key[0], key[1], stats))
            cam = f"{key[0]}/{key[1]}"
            print(
                f"  [{transport_name}/{cam}] "
                f"frames={stats['frames']:5d}  "
                f"fps={stats['fps']:6.1f}  "
                f"gap_p50={stats['gap_p50_ms']:6.2f}ms  "
                f"gap_p99={stats['gap_p99_ms']:6.2f}ms  "
                f"gap_max={stats['gap_max_ms']:6.2f}ms  "
                f"ttf={stats['ttf_ms']:6.1f}ms"
            )
        print()

    print()
    print(
        f"{'transport':10s} {'camera':30s} "
        f"{'fps':>8s} {'p50_ms':>8s} {'p99_ms':>8s} {'max_ms':>8s} {'ttf_ms':>8s}"
    )
    print("-" * 90)
    for transport_name, prefix, name, stats in rows:
        cam = f"{prefix}/{name}"
        print(
            f"{transport_name:10s} {cam:30s} "
            f"{stats['fps']:8.1f} "
            f"{stats['gap_p50_ms']:8.2f} "
            f"{stats['gap_p99_ms']:8.2f} "
            f"{stats['gap_max_ms']:8.2f} "
            f"{stats['ttf_ms']:8.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
