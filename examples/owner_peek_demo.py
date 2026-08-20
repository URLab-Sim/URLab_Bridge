# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""A minimal fast-path OWNER you can peek into and push on.

Steps a MuJoCo scene in ~real time, broadcasts its state on the ``viewer`` bus,
and applies any perturbations peek viewers push back -- the Python-owner side of
the peek loop. (A UE Direct instance is the other kind of owner; a peek viewer
attaches to either identically.)

Run the owner, then attach a viewer in another terminal:

    uv run python examples/owner_peek_demo.py --xml ../mujoco_menagerie/aloha/scene.xml
    # it prints the exact peek command, e.g.:
    #   python -m urlab_client.peek --model <xml> --bus tcp://127.0.0.1:5561 \
    #       --control tcp://127.0.0.1:5571
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from urlab_client.fastpath_owner import FastPathOwner


def to_mjb(model) -> bytes:
    buf = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return buf.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--bus-port", type=int, default=5561)
    ap.add_argument("--control-port", type=int, default=5571)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until Ctrl-C")
    ap.add_argument("--realtime", action="store_true", default=True)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    owner = FastPathOwner(
        to_mjb(model), scene="peek_demo",
        control_port=args.control_port, bus_port=args.bus_port,
        advertise_host="127.0.0.1", ngeom=int(model.ngeom),
    )
    print(f"owner up: bus {owner.bus_endpoint}  control {owner.control_endpoint}")
    print("peek with:\n"
          f"  python -m urlab_client.peek --model {args.xml} "
          f"--bus {owner.bus_endpoint} --control {owner.control_endpoint}")

    t0 = time.time()
    try:
        while args.seconds <= 0.0 or (time.time() - t0) < args.seconds:
            # 1. apply perturbations pushed by peek viewers (set, not accumulate)
            data.xfrc_applied[:] = 0.0
            for body, wrench in owner.drain_perturbations().items():
                data.xfrc_applied[body] = wrench
            # 2. step
            mujoco.mj_step(model, data)
            # 3. answer hello/perturb + broadcast state for viewers
            owner.serve_pending()
            owner.publish_state(data.time, data.qpos, data.qvel)
            if args.realtime:
                time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        pass
    finally:
        owner.close()
        print("\nowner stopped")


if __name__ == "__main__":
    main()
