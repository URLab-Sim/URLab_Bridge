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
except ImportError:  # pragma: no cover
    sys.exit("this demo needs `mujoco` (pip install mujoco, matching your UE minor version)")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from urlab_client.transports.zmq import ZmqTransport  # noqa: E402

# Fork-linked toolchain that produces a UE-loadable (version-matched) MJB.
_REPO = "/home/buzz/Documents/urlab_debug"
_MJBCOMPILE = os.path.join(_REPO, "mjb_test", "mjbcompile")
_MJLIB = os.path.join(_REPO, "UnrealRoboticsLab", "third_party", "install", "MuJoCo", "lib")
_SYSLIB = "/usr/lib/x86_64-linux-gnu"


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
    ap.add_argument("--hz", type=float, default=60.0, help="broadcast rate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    make_mjb(args.scene, args.mjb)
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(args.seed)

    transport = ZmqTransport("tcp://localhost")
    endpoint = transport.enable_viewer_broadcast(args.port)

    print(f"[owner] scene={args.scene}")
    print(f"[owner] nbody={model.nbody} ngeom={model.ngeom} nu={model.nu} nmesh={model.nmesh}")
    print(f"[owner] MJB written -> {args.mjb}")
    print(f"[owner] broadcasting per-geom transforms on {endpoint} (topic 'geoms')")
    print()
    print("In UE (URLab plugin), add an AMjbScene actor and set:")
    print(f"    MjbFilePath = {args.mjb}")
    print(f"    BusEndpoint = tcp://127.0.0.1:{args.port}")
    print("    bTestSweep  = false")
    print("then press Play.  Ctrl-C here to stop.\n")

    # Random-control ranges: use ctrlrange where limited, else a modest default.
    lo = model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
    hi = model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
    unlim = ~model.actuator_ctrllimited.astype(bool)
    lo[unlim] = -1.0
    hi[unlim] = 1.0

    dt = 1.0 / args.hz
    quat = np.zeros(4)
    frame = 0
    try:
        while True:
            if model.nu:
                data.ctrl[:] = rng.uniform(lo, hi)
            mujoco.mj_step(model, data)

            # Per-geom world transforms: position + quat (from the 3x3 xmat).
            xpos = np.asarray(data.geom_xpos, dtype=np.float64).reshape(-1)
            gx = np.asarray(data.geom_xmat, dtype=np.float64).reshape(model.ngeom, 9)
            xquat = np.empty(model.ngeom * 4, dtype=np.float64)
            for g in range(model.ngeom):
                mujoco.mju_mat2Quat(quat, gx[g])
                xquat[4 * g : 4 * g + 4] = quat

            transport.publish_geoms(
                {"f": frame, "xpos": xpos.tolist(), "xquat": xquat.tolist()}
            )
            frame += 1
            if frame % 120 == 0:
                print(f"[owner] step {frame}  sim_t={data.time:.2f}s")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[owner] stopping")
    finally:
        transport.close()


if __name__ == "__main__":
    main()
