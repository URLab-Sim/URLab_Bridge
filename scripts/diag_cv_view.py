# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Control experiment: display the live camera feed with OpenCV, not dearpygui.

Uses the exact same URLabClient + ZMQ camera streams as the dashboard, but
renders frames with cv2.imshow. If THIS window tracks your movement in real
time (and the printed CONTENT_AGE stays ~150ms) while the dpg dashboard lags
~3s, dearpygui's present path is conclusively the bottleneck -- not the
pipeline.

Shows every camera (same SUB load as the dashboard). Press 'q' in any window
to quit. The title bar of the probe camera shows live content-age.

    uv run python scripts/diag_cv_view.py
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    ap.add_argument("--step-port", type=int, default=5559)
    ap.add_argument("--mode", default="live", choices=("live", "direct", "keep"),
                    help="step mode to put the server in first (default live). "
                         "A camera captures on state change, so in direct mode "
                         "nothing publishes unless something is stepping; live "
                         "lets UE advance itself, which is what a latency read "
                         "wants. 'keep' leaves the server where it is.")
    args = ap.parse_args()

    from urlab_client import URLabClient

    print(f"Connecting to {args.host} ...", flush=True)
    client = URLabClient(args.host, step_port=args.step_port,
                         mujoco_version_check=False, local_model=False,
                         recv_timeout_ms=120_000)
    client.connect()
    if not client.manager_present:
        client.sim.start()
    client.refresh()

    if args.mode != "keep":
        client.runtime.set_mode(args.mode)

    names = client.camera_names()
    if not names:
        print("FAIL: no cameras")
        return 1
    client.warmup_cameras(names, timeout_s=15.0)
    views = {n: client._find_camera_view(n) for n in names}
    probe = names[0]
    print(f"Cameras: {names}\nProbe (title shows content-age): {probe}\n"
          "Move in front of a camera; compare this window's lag to the dpg UI. "
          "Press q to quit.", flush=True)

    for n in names:
        cv2.namedWindow(n, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(n, 320, 240)

    last_log = 0.0
    while True:
        for n, v in views.items():
            f = v.latest_frame
            if f is None:
                continue
            a = np.asarray(f)
            if a.ndim == 3 and a.shape[2] == 4:
                bgr = a[..., [2, 1, 0]]  # RGBA(real)/BGRA(seg) -> BGR-ish for view
            elif a.ndim == 2:
                bgr = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                continue
            cv2.imshow(n, np.ascontiguousarray(bgr.astype(np.uint8)))

        now = time.time()
        pv = views[probe]
        if now - last_log >= 1.0:
            last_log = now
            age = ((now - pv.capture_unix_time) * 1000.0
                   if pv.capture_unix_time else float("nan"))
            cv2.setWindowTitle(probe, f"{probe}  content_age={age:.0f}ms")
            print(f"[cvview] content_age={age:6.0f}ms  frames={pv.frame_count}",
                  flush=True)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
