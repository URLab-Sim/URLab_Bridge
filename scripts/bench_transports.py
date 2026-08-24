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

"""Benchmark ZMQ vs SHM transports across the three step modes.

Connects to a running UnrealEditor / packaged build with both transports
enabled (the default), runs N iterations of each (transport, mode)
combination, and prints a comparison table.

Run from the bridge repo root after starting URLab in the editor:

    uv run python scripts/bench_transports.py \\
        --address tcp://localhost --iters 500

Optional knobs:
    --shm-dir   path to <Saved>/URLabShm/<session>/ on the host running
                UE. Defaults to <Saved>/URLabShm/live which matches the
                publisher's default session id.
    --modes     subset of "direct,puppet,live" (default: all)
    --transports subset of "zmq,shm" (default: both)

Metrics:
    rpc_us       mean RPC latency, microseconds (step round-trip)
    p50_us       median latency
    p99_us       99th percentile
    steps/s      sustained step throughput
    state_hz     observed state-stream rate (live only)

Notes:
    - The SHM benchmark requires UE on the same host (single-host SHM).
      Cross-host runs use ZMQ.
    - The bench script does not start UE -- launch the editor / packaged
      build yourself, then run this against it.
    - This is intentionally a lightweight smoke benchmark, not a rigorous
      latency study. Numbers carry obvious noise from PIE rendering,
      Windows scheduling, etc. Run with the dashboard widget closed for
      cleaner numbers.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import List, Tuple

# Make the urlab_policy package importable when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import StepMode, URLabClient  # noqa: E402
from urlab_client.transports.shm import ShmTransport  # noqa: E402
from urlab_client.transports.zmq import ZmqTransport  # noqa: E402


def _percentile(samples: List[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _run_one(client: URLabClient, mode: str, iters: int) -> Tuple[List[float], float]:
    """Run `iters` step calls and collect per-call wall-clock latency.

    Returns (latencies_seconds, steps_per_second).
    """
    # Switch into the requested mode (no-op if already there).
    if mode != client.step_mode.value:
        client.set_mode(mode)

    latencies: List[float] = []
    t0 = time.perf_counter()
    for _ in range(iters):
        start = time.perf_counter()
        client.step(n_steps=1)
        latencies.append(time.perf_counter() - start)
    elapsed = time.perf_counter() - t0
    sps = iters / elapsed if elapsed > 0 else float("inf")
    return latencies, sps


def _make_client(transport: str, address: str, shm_dir: str, mode: str,
                 shm_poll_us: int) -> URLabClient:
    if transport == "zmq":
        client = URLabClient(
            address,
            step_mode=mode,
            transport="zmq",
            auto_promote_step_mode=True,
        )
    elif transport == "shm":
        # SHM transport uses ZMQ as a fallback for handshake-time RPC if
        # the SHM RPC region isn't ready yet (it should be after the
        # auto-spawn). Pass an explicit ZmqTransport as fallback so the
        # discover() call has a path even before SHM ring-buffers warm up.
        fallback = ZmqTransport(address)
        shm_t = ShmTransport(
            shm_dir,
            fallback=fallback,
            open_timeout_s=5.0,
            poll_interval_s=shm_poll_us / 1_000_000.0,
        )
        client = URLabClient(
            address,
            step_mode=mode,
            transport=shm_t,
            auto_promote_step_mode=True,
        )
    else:
        raise ValueError(f"unknown transport {transport!r}")
    client.connect()
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://localhost",
                        help="ZMQ address (default tcp://localhost)")
    parser.add_argument("--shm-dir", default=None,
                        help="SHM directory (default: <ProjectSavedDir>/URLabShm/live)")
    parser.add_argument("--iters", type=int, default=500,
                        help="step iterations per (transport, mode) pair")
    parser.add_argument("--warmup", type=int, default=20,
                        help="discarded warmup iterations before timing")
    parser.add_argument("--modes", default="direct,puppet,live",
                        help="comma-separated step modes to bench")
    parser.add_argument("--transports", default="zmq,shm",
                        help="comma-separated transports to bench")
    parser.add_argument("--shm-poll-us", type=int, default=100,
                        help="bridge-side SHM poll interval in microseconds. "
                             "Lower = lower RPC latency, higher idle CPU. "
                             "Default 100us; bump to 1000 to reproduce the older "
                             "1ms default and see how much polling cadence "
                             "moves the SHM numbers.")
    args = parser.parse_args()

    if args.shm_dir is None:
        # Best-effort default that matches USmSnapshotPublisher's path
        # scheme. The user can override with --shm-dir.
        args.shm_dir = os.path.join(
            os.path.expanduser("~"),
            "Documents",
            "Unreal Projects",
            "url_proj",
            "Saved",
            "URLabShm",
            "live",
        )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    transports = [t.strip() for t in args.transports.split(",") if t.strip()]

    print(f"Bench configuration:")
    print(f"  address     = {args.address}")
    print(f"  shm_dir     = {args.shm_dir}")
    print(f"  iters       = {args.iters}  warmup={args.warmup}")
    print(f"  modes       = {modes}")
    print(f"  transports  = {transports}")
    print(f"  shm_poll_us = {args.shm_poll_us}")
    print()

    rows: List[Tuple[str, str, float, float, float, float]] = []

    for transport in transports:
        for mode in modes:
            label = f"{transport:4s}/{mode}"
            try:
                client = _make_client(transport, args.address, args.shm_dir, mode,
                                      shm_poll_us=args.shm_poll_us)
            except Exception as exc:
                print(f"[{label}] connect failed: {exc}")
                continue
            try:
                # Warmup -- toss latencies, prime the JIT, ZMQ context, SHM
                # mappings.
                for _ in range(args.warmup):
                    try:
                        client.step(n_steps=1)
                    except Exception:
                        break
                latencies, sps = _run_one(client, mode, args.iters)
                if not latencies:
                    print(f"[{label}] no samples")
                    continue
                mean_us = statistics.mean(latencies) * 1e6
                p50_us = _percentile(latencies, 0.50) * 1e6
                p99_us = _percentile(latencies, 0.99) * 1e6
                rows.append((transport, mode, mean_us, p50_us, p99_us, sps))
                print(f"[{label}] mean={mean_us:7.1f}us p50={p50_us:7.1f}us "
                      f"p99={p99_us:7.1f}us steps/s={sps:7.1f}")
            except Exception as exc:
                print(f"[{label}] bench failed: {exc}")
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    print()
    print(f"{'transport':10s} {'mode':14s} {'mean_us':>10s} {'p50_us':>10s} "
          f"{'p99_us':>10s} {'steps/s':>10s}")
    print("-" * 70)
    for transport, mode, mean_us, p50_us, p99_us, sps in rows:
        print(f"{transport:10s} {mode:14s} {mean_us:10.1f} {p50_us:10.1f} "
              f"{p99_us:10.1f} {sps:10.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
