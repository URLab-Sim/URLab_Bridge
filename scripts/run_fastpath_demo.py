#!/usr/bin/env python3
# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Fast-path render demo.

Load a MuJoCo scene, apply random controls, and broadcast per-geom world
transforms so a URLab UE fast-path renderer (AMjbScene) mirrors it live -- with
NO physics on the UE side.

The owner here (pip ``mujoco``) and the renderer (UE, fork ``libmujoco``) must
compile the SAME scene to the same geom order. The transform stream itself is
version-independent (indexed float arrays); only the MJB the UE loads is
version-locked, so it is generated with the fork-linked ``mjbcompile`` tool.

Usage:
    python scripts/run_fastpath_demo.py \
        ../mujoco_menagerie/franka_emika_panda/scene.xml

Then, in a UE editor with the URLab plugin, drop an ``AMjbScene`` actor and set
``MjbFilePath`` + ``BusEndpoint`` as printed, and press Play.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:  # pragma: no cover
    sys.exit("this demo needs `mujoco` (pip install mujoco, matching your UE minor version)")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402

# Fork-linked toolchain that produces a UE-loadable (version-matched) MJB.
# Machine-specific paths come from env vars so this runs unchanged on another box
# (see docs/fast_path_render.md). Defaults match the original dev layout.
_REPO = os.environ.get("URLAB_ROOT", "/home/buzz/Documents/urlab_debug")
_MJBCOMPILE = os.environ.get("URLAB_MJBCOMPILE", os.path.join(_REPO, "mjb_test", "mjbcompile"))
_MJLIB = os.environ.get(
    "URLAB_MJLIB",
    os.path.join(_REPO, "UnrealRoboticsLab", "third_party", "install", "MuJoCo", "lib"))
_SYSLIB = os.environ.get("URLAB_SYSLIB", "/usr/lib/x86_64-linux-gnu")


def make_mjb(scene_xml: str, out_mjb: str) -> None:
    """Compile scene.xml to a version-matched MJB with the fork-linked tool."""
    if not os.path.exists(_MJBCOMPILE):
        sys.exit(f"mjbcompile not built at {_MJBCOMPILE}; build it first (see mjb_test/)")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{_SYSLIB}:{_MJLIB}")
    subprocess.run([_MJBCOMPILE, scene_xml, out_mjb], check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="URLab fast-path render demo")
    ap.add_argument("scene", help="path to a MuJoCo scene.xml (e.g. a menagerie scene)")
    ap.add_argument("--mjb", default="/tmp/urlab_fastpath.mjb", help="MJB path the UE renderer loads")
    ap.add_argument("--port", type=int, default=5561, help="transform-bus PUB port")
    ap.add_argument("--control-port", type=int, default=5571,
                    help="fast-path control REQ/REP port (serves the MJB, advertises)")
    ap.add_argument("--hz", type=float, default=60.0, help="broadcast rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="don't open the native MuJoCo viewer (headless broadcast only)")
    args = ap.parse_args()

    make_mjb(args.scene, args.mjb)
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(args.seed)

    # Own the model as a fast-path owner: serve its MJB on request, advertise in
    # the registry, and publish the geoms transform bus. The renderer discovers
    # us and pulls the MJB over the wire -- no shared file path needed.
    with open(args.mjb, "rb") as fh:
        mjb_bytes = fh.read()
    scene_id = os.path.splitext(os.path.basename(args.scene))[0]
    owner = FastPathOwner(
        mjb_bytes,
        scene=scene_id,
        control_port=args.control_port,
        bus_port=args.port,
        ngeom=model.ngeom,
    )

    print(f"[owner] scene={args.scene}")
    print(f"[owner] nbody={model.nbody} ngeom={model.ngeom} nu={model.nu} nmesh={model.nmesh}")
    print(f"[owner] advertising as '{scene_id}' in the registry")
    print(f"[owner] control (serves MJB): {owner.control_endpoint}")
    print(f"[owner] transform bus:        {owner.bus_endpoint} (topic 'geoms')")
    print()
    print("A UE fast-path renderer discovers this automatically (server browser")
    print("or -URLabFastConnect). To connect explicitly, point it at the control")
    print(f"endpoint: {owner.control_endpoint}")
    print("Ctrl-C here to stop.")
    if args.view:
        print("The native MuJoCo viewer (ground truth) opens next to compare "
              "side by side with the UE fast path.")
    print()

    # Native MuJoCo viewer alongside the UE fast path, both driven by this one
    # owner sim -- so the two renders can be compared side by side.
    viewer = mujoco.viewer.launch_passive(model, data) if args.view else None

    # Random-control ranges: use ctrlrange where limited, else a modest default.
    lo = model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
    hi = model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
    unlim = ~model.actuator_ctrllimited.astype(bool)
    lo[unlim] = -1.0
    hi[unlim] = 1.0

    # Broadcast at args.hz, but advance a real dt of sim each frame. mj_step only
    # advances model.opt.timestep (e.g. 2ms), so a single step per 1/60s frame
    # would crawl at ~0.12x real-time. Substep round(dt/timestep) times so the
    # motion runs at wall-clock speed.
    dt = 1.0 / args.hz
    n_sub = max(1, round(dt / model.opt.timestep))
    print(f"[owner] rate: {args.hz:.0f} Hz broadcast, timestep={model.opt.timestep*1e3:.1f} ms, "
          f"{n_sub} steps/frame (~real-time)")
    quat = np.zeros(4)
    frame = 0
    try:
        while viewer is None or viewer.is_running():
            # Answer any renderer that just connected (serves the MJB + bus) and
            # collect any external forces a renderer pushed back.
            owner.serve_pending()

            if model.nu:
                data.ctrl[:] = rng.uniform(lo, hi)

            # Apply renderer-sent perturbations as body external forces for this
            # step, then clear (transient impulse; a sustained drag resends).
            perts = owner.drain_perturbations()
            data.xfrc_applied[:] = 0.0
            for body_id, ft in perts.items():
                if 0 <= body_id < model.nbody:
                    data.xfrc_applied[body_id] = ft

            for _ in range(n_sub):
                mujoco.mj_step(model, data)

            # Per-geom world transforms: position + quat (from the 3x3 xmat).
            xpos = np.asarray(data.geom_xpos, dtype=np.float64).reshape(-1)
            gx = np.asarray(data.geom_xmat, dtype=np.float64).reshape(model.ngeom, 9)
            xquat = np.empty(model.ngeom * 4, dtype=np.float64)
            for g in range(model.ngeom):
                mujoco.mju_mat2Quat(quat, gx[g])
                xquat[4 * g : 4 * g + 4] = quat

            # Per-camera world transforms, so a render-server renderer's cameras
            # track moving bodies.
            cxpos = cxquat = None
            if model.ncam:
                cxpos = np.asarray(data.cam_xpos, dtype=np.float64).reshape(-1)
                cx = np.asarray(data.cam_xmat, dtype=np.float64).reshape(model.ncam, 9)
                cxquat = np.empty(model.ncam * 4, dtype=np.float64)
                for c in range(model.ncam):
                    mujoco.mju_mat2Quat(quat, cx[c])
                    cxquat[4 * c : 4 * c + 4] = quat

            owner.publish_geoms(frame, xpos, xquat, cxpos, cxquat)
            if viewer is not None:
                viewer.sync()  # native MuJoCo viewer = ground truth, side by side
            frame += 1
            if frame % 120 == 0:
                print(f"[owner] step {frame}  sim_t={data.time:.2f}s")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[owner] stopping")
    finally:
        if viewer is not None:
            viewer.close()
        owner.close()


if __name__ == "__main__":
    main()
