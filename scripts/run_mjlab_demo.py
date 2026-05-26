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

"""End-to-end eval: mjlab's public demo policy against URLab.

Pulls together everything the mjlab adapter needs to drive URLab from
mjlab's `Mjlab-Tracking-Flat-Unitree-G1` task:

    - URLabClient over ZMQ or SHM
    - mjlab task config + checkpoint loading
    - LAFAN motion file as the tracking command source

Prerequisites:

    1. URLab editor in PIE with a G1 scene imported from
       `<demo>/g1/g1_with_actuators.xml`. Run the bridge's mjlab_export
       helper to materialise that XML if you don't have it:

         uv run python -m urlab_policy.adapters.mjlab.export \\
             --robot unitree_g1 \\
             --out C:/Users/jonat/Documents/mjlab_demo/g1/g1_with_actuators.xml

    2. Demo checkpoint + motion file under `<demo>/`:

         curl -L -o ckpt.pt    https://storage.googleapis.com/mjlab_beta/model_49999.pt
         curl -L -o motion.npz https://storage.googleapis.com/mjlab_beta/lafan_dance1_subject1.npz

    3. mjlab + rsl_rl + torch installed in the bridge env:

         uv pip install mjlab

Run:

    uv run python scripts/run_mjlab_demo.py
    uv run python scripts/run_mjlab_demo.py --transport shm
    uv run python scripts/run_mjlab_demo.py --steps 5000 --device cuda:0

The script exits cleanly on Ctrl+C or after `--steps` is reached.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_policy.adapters.mjlab import MjlabRunner  # noqa: E402
from urlab_client import URLabClient  # noqa: E402

DEFAULT_DEMO_DIR = r"C:/Users/jonat/Documents/mjlab_demo"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="tcp://localhost",
                        help="URLab step server address.")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--transport", default="zmq", choices=["zmq", "shm"])
    parser.add_argument("--demo-dir", default=DEFAULT_DEMO_DIR,
                        help="Directory containing ckpt.pt + motion.npz.")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path. Default: <demo-dir>/ckpt.pt")
    parser.add_argument("--motion", default=None,
                        help="Override motion file. Default: <demo-dir>/motion.npz")
    parser.add_argument("--task-id", default="Mjlab-Tracking-Flat-Unitree-G1",
                        help="mjlab task id.")
    parser.add_argument("--articulation", default="g1",
                        help="URLab articulation prefix. Default 'g1'.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for the policy. cpu / cuda:0 / etc.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Stop after this many policy steps. Default: forever (Ctrl+C to stop).")
    parser.add_argument("--hold-default", action="store_true",
                        help="Sanity check: bypass the policy and send action=zeros every step. "
                             "With use_default_offset=True this drives URLab to mjlab's init "
                             "crouch pose continuously -- if URLab holds it stably, the wire / "
                             "scale / offset path is sound and any 'going crazy' is on the "
                             "obs/policy side.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_mjlab_demo")

    checkpoint = args.checkpoint or os.path.join(args.demo_dir, "ckpt.pt")
    if not os.path.isfile(checkpoint):
        logger.error("missing checkpoint: %s", checkpoint)
        return 2

    # Motion file is only needed by tracking tasks. For velocity / standing
    # locomotion tasks, the policy reads the twist command from the URLab
    # actor's UMjTwistController (WASD/gamepad in PIE), no motion file
    # required. We let the runner figure it out: if an explicit --motion is
    # given, use it; otherwise use <demo-dir>/motion.npz when present, else
    # pass None and the adapter will silently skip MotionContext.
    if args.motion:
        motion = args.motion
        if not os.path.isfile(motion):
            logger.error("missing motion file: %s", motion)
            return 2
    else:
        candidate = os.path.join(args.demo_dir, "motion.npz")
        motion = candidate if os.path.isfile(candidate) else None
        if motion is None:
            logger.info("no motion.npz in --demo-dir; running without motion file "
                        "(fine for velocity/locomotion tasks, mandatory for tracking)")

    logger.info("connecting to URLab at %s (transport=%s)", args.address, args.transport)
    client = URLabClient(
        args.address,
        step_mode="direct",
        step_port=args.step_port,
        state_port=args.state_port,
        transport=args.transport,
    )
    client.discover()
    logger.info("session=%s articulations=%s",
                client.session_id, sorted(client.articulations.keys()))

    if args.articulation not in client.articulations:
        logger.error("articulation prefix %r not in session (have %s)",
                     args.articulation, sorted(client.articulations.keys()))
        client.close()
        return 3

    runner = MjlabRunner(
        client,
        task_id=args.task_id,
        checkpoint=checkpoint,
        motion_file=motion,
        articulation_prefix=args.articulation,
        device=args.device,
    )
    logger.info("policy ready. running...")

    stop = False

    def _handler(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handler)

    try:
        i = 0
        if args.hold_default:
            logger.info("--hold-default: bypassing policy, sending init crouch pose every step")
        while not stop and (args.steps is None or i < args.steps):
            runner.step(hold_default=args.hold_default)
            i += 1
            if i % 50 == 0:
                logger.info("step=%d sim_time=%.3fs", i, client.sim_time)
    except Exception:
        logger.exception("policy loop crashed")
        return 4
    finally:
        try:
            runner.close()
        except Exception:
            logger.exception("close failed")
    logger.info("done after %d steps", i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
