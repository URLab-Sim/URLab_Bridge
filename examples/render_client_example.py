# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Minimal example: drive a fast-path render server with the clean RenderClient.

Prerequisite: a render server is already running and listening on the step port
(default 5559). Launch one from a packaged build, e.g.::

    url_proj.exe /Game/FirstPerson/Lvl_FirstPerson \
        -URLabDrive=push -URLabModel=scene.mjb -URLabCaps=serve,cameras \
        -URLabScene=cammax=0 -RenderOffScreen -nosplash

(See docs/render_server.md for packaging, launch flags, and MJB compilation.)

Run (from the URLab_Bridge dir)::

    uv run python examples/render_client_example.py --xml scene.xml --camera 0 --frames 100
    uv run python examples/render_client_example.py --xml scene.xml --show      # live cv2
    uv run python examples/render_client_example.py --xml scene.xml --delay 2   # stale/fast
"""
import argparse
import time

import mujoco
import numpy as np

from urlab_client import RenderClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, help="MuJoCo scene the server also loaded")
    ap.add_argument("--address", default="tcp://127.0.0.1")
    ap.add_argument("--step-port", type=int, default=5559)
    ap.add_argument("--transport", default="zmq", choices=["zmq", "shm", "grpc"])
    ap.add_argument("--camera", default="0", help="camera index or name (default: first)")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--delay", type=int, default=0,
                    help="0 = fresh/blocking, >0 = stale-from-ring (faster)")
    ap.add_argument("--dt", type=float, default=0.0, help="override model timestep (s)")
    ap.add_argument("--show", action="store_true", help="live cv2 window")
    ap.add_argument("--save", default="", help="write the last frame to this PNG")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    if args.dt > 0:
        model.opt.timestep = args.dt
    data = mujoco.MjData(model)

    with RenderClient(args.address, step_port=args.step_port,
                      transport=args.transport, recv_timeout_ms=10_000) as rc:
        # Resolve which camera to view (the server owns the authoritative names).
        names = rc.camera_names()
        cam = names[int(args.camera)] if args.camera.isdigit() else args.camera
        print(f"cameras: {names}\nviewing: {cam}   delay={args.delay} substeps")

        rts = []
        last = None
        for i in range(args.frames):
            mujoco.mj_step(model, data)
            t0 = time.perf_counter()
            frames = rc.render_mjdata(model, data, cameras=[cam], delay=args.delay)
            rts.append((time.perf_counter() - t0) * 1000.0)
            last = frames[cam]
            if args.show:
                import cv2
                cv2.imshow("render_client", cv2.resize(last.to_bgr(), (960, 720)))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        if rts:
            mean = float(np.mean(rts))
            print(f"\n{len(rts)} frames  {last.width}x{last.height} {last.dtype}")
            print(f"round-trip mean {mean:.2f} ms  ->  {1000/mean:.1f} Hz")
        if args.save and last is not None:
            import cv2
            cv2.imwrite(args.save, last.to_bgr())
            print(f"wrote {args.save}")
        if args.show:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
