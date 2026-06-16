# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Quantify camera-feed rate + end-to-end latency against a live URLab editor.

Two measurements:
  1. STREAM FPS (passive): how fast frames actually arrive on the bridge, sampled
     from the camera view's frame_count over a window. Cameras render once per UE
     game tick, so this is effectively the PIE tick rate -- if it's ~10-15 fps the
     editor is almost certainly throttling in the background (focus the editor
     window or disable "Use Less CPU when in Background").
  2. END-TO-END LATENCY (puppet): time from pushing a step's state (frame_id F) to
     a streamed frame with frame_id >= F arriving on the client. This is the
     render+readback+publish+SUB+decode pipeline delay -- the "fresh" latency.

Run with the editor in PIE and NO other client stepping (close the dashboard).

    uv run python scripts/measure_camera.py \
        --local-xml /c/Users/jonat/Downloads/reaf_test/reaf_test/base_scene.xml
"""

from __future__ import annotations

import argparse
import statistics
import time


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    ap.add_argument("--step-port", type=int, default=5559)
    ap.add_argument("--transport", choices=["zmq", "shm"], default="zmq")
    ap.add_argument("--local-xml", default="",
                    help="plain MJCF for the local model (puppet latency test)")
    ap.add_argument("--camera", default="",
                    help="canonical camera name (default: first discovered)")
    ap.add_argument("--fps-window", type=float, default=5.0,
                    help="seconds to sample stream FPS")
    ap.add_argument("--latency-steps", type=int, default=60)
    args = ap.parse_args()

    from urlab_client import URLabClient

    print(f"Connecting to {args.host} (transport={args.transport})...", flush=True)
    client = URLabClient(args.host, step_port=args.step_port, transport=args.transport,
                         mujoco_version_check=False, local_model=False,
                         recv_timeout_ms=120_000)
    client.connect()
    if not client.manager_present:
        client.sim.start()
    client.refresh()

    names = client.camera_names()
    if not names:
        print("FAIL: no cameras discovered (is PIE running with a camera scene?)")
        return 1
    cam = args.camera or names[0]
    if cam not in names:
        print(f"FAIL: camera {cam!r} not found; available: {names}")
        return 1
    view = client._find_camera_view(cam)
    print(f"Cameras: {names}\nMeasuring: {cam!r}", flush=True)

    client.warmup_cameras([cam], timeout_s=10.0)

    # --- 1. passive stream FPS (no stepping; cameras render per game tick) ---
    print(f"\n[1] Sampling stream FPS for {args.fps_window:.0f}s "
          "(no stepping -- pure render/publish rate)...", flush=True)
    c0 = view.frame_count
    f0 = view.frame_id
    t0 = time.monotonic()
    time.sleep(args.fps_window)
    dt = time.monotonic() - t0
    frames = view.frame_count - c0
    fid_adv = (view.frame_id - f0) if (view.frame_id is not None and f0 is not None) else None
    fps = frames / dt if dt > 0 else 0.0
    print(f"    received {frames} frames in {dt:.2f}s -> {fps:.1f} fps "
          f"(frame_id advanced {fid_adv})")
    if fps < 25:
        print("    NOTE: <25 fps -> almost certainly editor background throttling. "
              "Focus the UE editor window, or Editor Preferences > General > "
              "Performance > uncheck 'Use Less CPU when in Background'.")

    # --- 2. end-to-end latency (puppet: state push -> frame arrival) ---
    if not args.local_xml:
        print("\n[2] SKIPPED end-to-end latency (pass --local-xml for puppet).")
        client.close()
        return 0

    import mujoco
    m = mujoco.MjModel.from_xml_path(args.local_xml)
    client.model = m
    client.data = mujoco.MjData(m)
    client.local_model = False
    client.runtime.set_mode("puppet")
    client.warmup_cameras([cam], timeout_s=10.0)

    print(f"\n[2] End-to-end latency over {args.latency_steps} steps "
          "(push state -> frame_id>=F arrives)...", flush=True)
    import numpy as np
    lat_ms = []
    for i in range(args.latency_steps):
        if m.nq > 0:
            client.data.qpos[0] = 0.3 * np.sin(i * 0.15)
        mujoco.mj_forward(m, client.data)
        reply = client.step(n_steps=0)
        target = reply.frame_id
        if target is None:
            continue
        t_push = time.monotonic()
        deadline = t_push + 1.0
        while time.monotonic() < deadline:
            if view.frame_id is not None and view.frame_id >= target:
                lat_ms.append((time.monotonic() - t_push) * 1000.0)
                break
            time.sleep(0.0005)
        time.sleep(0.02)  # don't outrun the render

    if lat_ms:
        print(f"    n={len(lat_ms)}  mean={statistics.mean(lat_ms):.1f}ms  "
              f"p50={_pct(lat_ms,50):.1f}ms  p95={_pct(lat_ms,95):.1f}ms  "
              f"max={max(lat_ms):.1f}ms")
        print(f"    (at {fps:.0f} fps the inter-frame gap alone is "
              f"{1000.0/fps if fps else float('nan'):.0f}ms; latency >= that by nature)")
    else:
        print("    no frames reached the target frame_id within 1s -- stream stalled?")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
