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

"""Minimal URLabClient example -- no RoboJuDo, no env wrappers.

Shows the full life of a custom policy on top of the URLabClient API:

    1. Connect with `transport=zmq` or `transport=shm`. SHM pulls the
       on-disk session dir from the handshake -- no hardcoded paths.
    2. Pull the live `URLabArticulation` (`art = client.articulations[...]`).
    3. Read root state via the entity accessors (`art.root_pos_w`,
       `art.root_quat_xyzw`, `art.root_lin_vel_w`, `art.root_ang_vel_w`).
    4. Push PD gains with `art.push_gains(joint_names, kp, kv)`.
    5. Send PD targets per step with `art.set_ctrl({...})`.
    6. Read twist + clocks straight off the client / articulation.

Run:

    uv run python scripts/run_raw_policy.py --duration 30
    uv run python scripts/run_raw_policy.py --transport shm --duration 30
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient  # noqa: E402

logger = logging.getLogger("run_raw_policy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="tcp://localhost")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--step-mode", default="stepped",
                        choices=["stepped", "freerun"])
    parser.add_argument("--transport", default="zmq", choices=["zmq", "shm"])
    parser.add_argument("--shm-dir", default=None,
                        help="explicit SHM session dir (default: pulled "
                             "from the handshake's shm_session_dir field)")
    parser.add_argument("--articulation", default="",
                        help="prefix to drive (default: the only one)")
    parser.add_argument("--sim-decimation", type=int, default=10)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--kp", type=float, default=80.0,
                        help="uniform per-joint stiffness")
    parser.add_argument("--kv", type=float, default=2.0,
                        help="uniform per-joint damping")
    parser.add_argument("--ctrl-scale", type=float, default=0.2,
                        help="random ctrl amplitude (rad). Clamped to "
                             "actuator ctrlrange when authored.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    client = URLabClient(
        args.address,
        step_mode=args.step_mode,
        step_port=args.step_port,
        state_port=args.state_port,
        transport=args.transport,
        shm_dir=args.shm_dir,
    )
    client.connect(observations="full")

    if args.articulation:
        prefix = args.articulation
    elif len(client.articulations) == 1:
        prefix = next(iter(client.articulations))
    else:
        raise SystemExit(
            f"multiple articulations available {list(client.articulations)}; "
            f"pass --articulation"
        )
    art = client.articulations[prefix]

    joint_names = list(art.joints.keys())
    actuator_names = list(art.actuators.keys())
    logger.info("connected: prefix=%s joints=%d actuators=%d free_base=%s",
                prefix, len(joint_names), len(actuator_names), art.has_free_base)
    logger.info("shm_session_dir from handshake: %r", client.shm_session_dir)

    # Push uniform PD gains to UE's controller. Real policies would
    # read kp/kv from a checkpoint or config; this is the minimum a
    # raw policy needs for stable PD tracking.
    if art.controller is not None:
        kp = np.full(len(joint_names), args.kp)
        kv = np.full(len(joint_names), args.kv)
        pushed = art.push_gains(joint_names, kp, kv)
        logger.info("pushed PD gains for %d joints", pushed)

    # Trivial policy: random per-step setpoints sampled in `[-scale, +scale]`,
    # clamped to each actuator's authored ctrlrange when present. Demonstrates
    # the smallest useful set_ctrl loop -- swap the sampler for a real policy.
    rng = np.random.default_rng(args.seed)

    def sample_ctrl() -> dict:
        out: dict = {}
        for name, act in art.actuators.items():
            v = float(rng.uniform(-args.ctrl_scale, args.ctrl_scale))
            if act.ctrlrange is not None:
                lo, hi = act.ctrlrange
                v = float(np.clip(v, lo, hi))
            out[name] = v
        return out

    logger.info("policy: random ctrl (scale=%.3f, seed=%d) for %.1fs",
                args.ctrl_scale, args.seed, args.duration)
    dt = 1.0 / args.freq
    deadline = time.perf_counter() + args.duration
    stop = False

    def _on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    iters = 0
    try:
        while not stop and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            art.set_ctrl(sample_ctrl())
            client.step(n_steps=args.sim_decimation, observations="full")
            iters += 1
            if iters % 50 == 0 and art.has_free_base:
                logger.info(
                    "t=%.2fs base_pos=%s base_lin_vel=%s twist=%s",
                    client.sim_time,
                    np.round(art.root_pos_w, 3).tolist(),
                    np.round(art.root_lin_vel_w, 3).tolist(),
                    np.round(art.twist, 3).tolist(),
                )
            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        client.close()

    logger.info("done: %d iterations", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
