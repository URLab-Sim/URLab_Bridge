# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Measure TRUE camera content latency using the v2 stream header.

Each streamed frame now carries CaptureUnixTime (UE FDateTime::UtcNow at the
moment its GPU readback was requested). That's the same clock as Python's
time.time(), so content latency = time.time() - view.capture_unix_time, with no
cross-process clock sync and no dependency on sim_time / frame_id (which freeze
when the sim is paused).

Run with --load to also run the dashboard's per-tick 5-camera float conversion
in a background thread, so the camera SUB threads face the same GIL contention:
  --load     reproduces the dashboard's conditions
  (default)  isolates the transport

    uv run python scripts/diag_content_latency.py --seconds 10
    uv run python scripts/diag_content_latency.py --seconds 10 --load
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    ap.add_argument("--step-port", type=int, default=5559)
    ap.add_argument("--camera", default="")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--load", action="store_true",
                    help="run the dashboard's 5-camera conversion load")
    args = ap.parse_args()

    from urlab_client import URLabClient

    print(f"Connecting to {args.host} (load={args.load})...", flush=True)
    client = URLabClient(args.host, step_port=args.step_port,
                         mujoco_version_check=False, local_model=False,
                         recv_timeout_ms=120_000)
    client.connect()
    if not client.manager_present:
        client.sim.start()
    client.refresh()

    names = client.camera_names()
    if not names:
        print("FAIL: no cameras")
        return 1
    cam = args.camera or names[0]
    probe = client._find_camera_view(cam)
    client.warmup_cameras(names, timeout_s=15.0)
    print(f"Cameras: {names}\nProbe: {cam!r}", flush=True)

    if probe.capture_unix_time is None:
        print("\nFAIL: frames carry no capture_unix_time -- the editor is "
              "running a pre-v2 build. Rebuild UE with the v2 camera meta.")
        client.close()
        return 2

    stop = threading.Event()

    def load_loop():
        allv = []
        for art in client.articulations.values():
            allv.extend(art.cameras.values())
        allv.extend(client.global_cameras.values())
        inv = np.float32(1.0 / 255.0)
        while not stop.is_set():
            for v in allv:
                f = v.latest_frame
                if f is not None:
                    a = np.asarray(f, dtype=np.uint8)
                    if a.ndim == 3:
                        np.multiply(a, inv, out=np.empty(a.shape, np.float32))
            time.sleep(1.0 / 15.0)

    if args.load:
        threading.Thread(target=load_loop, daemon=True).start()

    print(f"\nSampling content latency for {args.seconds:.0f}s "
          "(time.time - frame capture time)...", flush=True)
    samples = []
    last_fc = -1
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        if probe.frame_count != last_fc and probe.capture_unix_time is not None:
            last_fc = probe.frame_count
            samples.append((time.time() - probe.capture_unix_time) * 1000.0)
        time.sleep(0.002)

    stop.set()
    if samples:
        print(f"\nCONTENT LATENCY (UE capture -> client has it):")
        print(f"  n={len(samples)}  mean={statistics.mean(samples):.0f}ms  "
              f"p50={statistics.median(samples):.0f}ms  "
              f"min={min(samples):.0f}ms  max={max(samples):.0f}ms")
        print("\n  ~2000ms => transport/SUB delivers stale content; that's the lag.")
        print("  ~200ms  => transport is fresh; the dashboard's dpg present is the lag.")
    else:
        print("No samples.")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
