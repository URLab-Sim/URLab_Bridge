# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Localise the dashboard camera latency.

Frames reach the client's ``view.latest_frame`` ~140ms fresh (see
measure_camera.py), yet the dashboard shows ~2s of lag. The suspect is the
dashboard *process*: its camera SUB threads share the GIL with a render loop
that, every ~66ms, float-converts and uploads 5x 640x480 frames. If that work
starves the SUB threads, frames back up and ``latest_frame`` itself goes stale
inside the dashboard process.

This script reproduces ONLY the CPU side of that load (the per-tick float
conversion of every camera) WITHOUT dearpygui, then reports per second:
  - arrival fps in THIS process (view.frame_count delta) -- if this collapses
    under load, the SUB thread is starved.
  - frame age = now - view.recv_monotonic -- how stale latest_frame is when a
    consumer reads it. ~2s here => the lag is client-side SUB starvation, not
    dpg. ~50ms here => the lag is dpg upload/present, not the SUB path.

    uv run python scripts/diag_dashboard_load.py
    uv run python scripts/diag_dashboard_load.py --no-load   # baseline, no conversion
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _convert(view) -> None:
    """Mirror tabs/cameras.py _frame_to_rgba_float: the per-tick cost."""
    frame = view.latest_frame
    if frame is None:
        return
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim != 3:
        return
    rgba = arr.astype(np.float32) / 255.0
    rgba.ravel()  # the array handed to dpg.set_value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    ap.add_argument("--step-port", type=int, default=5559)
    ap.add_argument("--render-fps", type=float, default=15.0,
                    help="match the dashboard render loop (default 15)")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--no-load", action="store_true",
                    help="skip the float conversion (isolate SUB health)")
    args = ap.parse_args()

    from urlab_client import URLabClient

    print(f"Connecting to {args.host} (load={'off' if args.no_load else 'on'})...",
          flush=True)
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
    client.warmup_cameras(names, timeout_s=15.0)
    print(f"Cameras: {names}", flush=True)

    def views():
        out = []
        for art in client.articulations.values():
            out.extend(art.cameras.values())
        out.extend(client.global_cameras.values())
        return out

    probe = client._find_camera_view(names[0])
    interval = 1.0 / max(args.render_fps, 1.0)
    last_count = probe.frame_count
    next_report = time.monotonic() + 1.0
    t_end = time.monotonic() + args.seconds
    loops = 0

    print("\n  t   loop/s  arrivalfps  frame_age_ms  (probe="
          f"{names[0]})", flush=True)
    while time.monotonic() < t_end:
        loop_start = time.monotonic()
        if not args.no_load:
            for v in views():
                _convert(v)
        loops += 1

        now = time.monotonic()
        if now >= next_report:
            arr_fps = probe.frame_count - last_count
            last_count = probe.frame_count
            age_ms = ((now - probe.recv_monotonic) * 1000.0
                      if probe.recv_monotonic else float("nan"))
            print(f"  {now - (t_end - args.seconds):4.0f}  {loops:6d}  "
                  f"{arr_fps:9d}  {age_ms:11.0f}", flush=True)
            loops = 0
            next_report += 1.0

        # Pace to the render fps like the dashboard does.
        sleep = interval - (time.monotonic() - loop_start)
        if sleep > 0:
            time.sleep(sleep)

    client.close()
    print("\nInterpretation:")
    print("  arrivalfps ~25 + age ~50ms  -> SUB healthy; lag is dpg display.")
    print("  arrivalfps collapses / age grows to ~seconds -> SUB starved by "
          "the main-thread load; fix = decode/convert off the render thread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
