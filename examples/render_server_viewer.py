# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Drive the fast-path render server from a live MuJoCo passive viewer.

This is the end-to-end integration: one Python process that

  1. loads a MuJoCo scene itself (any menagerie scene works),
  2. connects to the render server over gRPC and uploads the model -- the server
     spawns its renderer on demand, so it can be booted on a bare empty level
     with no ``-URLabFast*`` model flag,
  3. opens MuJoCo's own ``launch_passive`` viewer so you can fly around, and
  4. each frame mirrors the sim state + your viewer camera to the server, pulls
     the UE render of *your* viewpoint back, and paints it into the viewer as a
     picture-in-picture -- the "UE is the studio's main renderer" preview.

Everything goes through the high-level client (:class:`RenderClient` +
:mod:`urlab_client.viewer_sync`); there is not a single hand-rolled op dict or
wire key here.

Two server regimes, selected with ``--mode`` (they can't share one instance --
forced capture stalls the render thread a smooth stream needs):

  * ``forced``  -- exact-fresh, client-paced, blocking (``delay=0``). Boot the
    server with ``-URLabFastForcedOnly``. Deterministic, not buttery smooth.
  * ``viewer``  -- server-paced ring frames, a few substeps stale but smooth
    (``delay=N``). Boot the server WITHOUT ``-URLabFastForcedOnly``.

Boot a server (empty level, no model needed) e.g.::

    UnrealEditor URLabTest.uproject /Game/FastPath/FastPathRender -game \
        -URLabFastServe -URLabFastForcedOnly -URLabFastCameras -RenderOffScreen \
        -nosplash -unattended -stdout

Run::

    uv run python examples/render_server_viewer.py \
        --xml ../mujoco_menagerie/aloha/scene.xml --cameras 2 --mode forced
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from urlab_client import USER_CAMERA, RenderClient, viewer_sync
from urlab_client.errors import URLabTimeoutError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True, help="MuJoCo scene to load (menagerie ok)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50051, help="gRPC ListenPort")
    ap.add_argument("--cameras", type=int, default=1,
                    help="how many model cameras to also render each frame")
    ap.add_argument("--mode", choices=["forced", "viewer"], default="forced",
                    help="forced=exact/blocking (delay 0); viewer=smooth ring frames")
    ap.add_argument("--delay", type=int, default=2,
                    help="viewer mode: substeps of ring lag to trade for smoothness")
    ap.add_argument("--pip-scale", type=float, default=0.4,
                    help="picture-in-picture size as a fraction of the viewer width")
    args = ap.parse_args()

    delay = args.delay if args.mode == "viewer" else 0

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # One reusable scene for resolving the viewer's free camera to a world pose.
    cam_scene = viewer_sync.make_scene(model)

    with RenderClient.grpc(args.host, args.port, recv_timeout_ms=20_000) as rc, \
            mujoco.viewer.launch_passive(model, data) as viewer:
        # The server spawns its renderer from this upload (no boot model required).
        rc.load_xml(args.xml)

        names = rc.camera_names()
        model_cams = [n for n in names if n != USER_CAMERA][: max(0, args.cameras)]
        want = model_cams + [USER_CAMERA]
        print(f"server cameras: {names}")
        print(f"rendering {want}  mode={args.mode} delay={delay}")

        rts: list[float] = []
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            # Mirror the viewer's free/user camera -- pure MjvCamera -> (pos,fwd,up).
            user_pose = viewer_sync.pose_from_passive(viewer, scene=cam_scene)

            t0 = time.perf_counter()
            try:
                frames = rc.render_mjdata(
                    model, data, cameras=want, user_pose=user_pose,
                    delay=delay, timeout_ms=5000,
                )
            except URLabTimeoutError:
                # A single slow/cold frame: skip it, the transport stays connected.
                continue
            rts.append((time.perf_counter() - t0) * 1000.0)

            _paint_pip(viewer, frames.get(USER_CAMERA), args.pip_scale)

        if rts:
            mean = float(np.mean(rts))
            print(f"\n{len(rts)} frames  round-trip mean {mean:.1f} ms  ({1000 / mean:.1f} Hz)")


def _paint_pip(viewer, frame, scale: float) -> None:
    """Overlay the UE user-camera render in the viewer's top-right corner.

    ``set_images`` needs the image to match its viewport rect exactly, so the
    frame is resized to a rect sized off the current window and pinned top-right.
    """
    if frame is None:
        return
    vp = viewer.viewport
    if vp is None or vp.width <= 0 or vp.height <= 0:
        return
    w = max(1, int(vp.width * scale))
    h = max(1, int(w * frame.height / frame.width))
    rgb = _resize_rgb(frame.to_rgb(), w, h)
    # MjrRect origin is bottom-left; pin the PiP to the top-right corner.
    rect = mujoco.MjrRect(vp.width - w, vp.height - h, w, h)
    viewer.set_images((rect, rgb))


def _resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize an HxWx3 uint8 image to (h, w). Uses cv2 when present, else a
    dependency-free nearest-neighbour sampling."""
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return np.ascontiguousarray(rgb)
    try:
        import cv2
        return np.ascontiguousarray(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA))
    except ImportError:
        ys = (np.linspace(0, rgb.shape[0] - 1, h)).astype(np.intp)
        xs = (np.linspace(0, rgb.shape[1] - 1, w)).astype(np.intp)
        return np.ascontiguousarray(rgb[ys][:, xs])


if __name__ == "__main__":
    main()
