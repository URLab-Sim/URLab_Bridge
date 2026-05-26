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

"""End-to-end eval for the mjlab-HOMIERL H1 policy in URLab.

HOMIERL is a custom mjlab task (not in mjlab core) that trains the
Unitree H1 to walk while its upper body is randomly perturbed. The
policy uses a single custom obs term (`him_obs`, history_length=6),
two commands (`twist` velocity + `height` relative target), and a
joint-position action restricted to the lower body.

Prerequisites:

    1. URLab editor in PIE with a Unitree H1 actor imported.

       Export the H1 XML once:
         uv run python -m urlab_policy.adapters.mjlab.export \\
             --robot unitree_h1 \\
             --out C:/Users/jonat/Documents/mjlab_h1_demo/h1_with_actuators.xml

       Place the H1 mesh `assets/` directory next to the exported XML
       (the export resolves `meshdir="assets"`), then drag the XML
       into the URLab editor.

    2. HOMIERL checkpoint (.pt) downloaded from the HuggingFace repo
       `Nagi-ovo/HOMIERL-loco`. Drop it at:
         C:/Users/jonat/Documents/mjlab_h1_demo/ckpt.pt
       Or pass `--checkpoint` explicitly.

    3. `mjlab-homierl` installed in this venv (already done by the
       bridge setup; the version-pin compat shim handles the
       mjlab>=1.2.0,<1.3.0 incompatibility against our 1.3.0).

Run:

    uv run python scripts/run_homierl_demo.py --steps 2000
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

# Import order matters: install the mjlab.utils.os.update_assets shim
# BEFORE any mjlab.tasks auto-discovery scans installed task packages.
from urlab_policy.adapters.robojudo import _compat as _homierl_compat  # noqa: E402
import mjlab_homierl  # noqa: E402, F401  -- registers Mjlab-Homie-* tasks
_homierl_compat.register_homierl_command_mappings()

from urlab_policy.adapters.mjlab import MjlabRunner  # noqa: E402
from urlab_client import URLabClient  # noqa: E402

DEFAULT_DEMO_DIR = r"C:/Users/jonat/Documents/mjlab_h1_demo"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="tcp://localhost")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--transport", default="zmq", choices=["zmq", "shm"])
    parser.add_argument("--demo-dir", default=DEFAULT_DEMO_DIR,
                        help="Directory containing ckpt.pt + (optional) the H1 export.")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path. Default: <demo-dir>/ckpt.pt")
    parser.add_argument("--task-id", default="Mjlab-Homie-Unitree-H1",
                        choices=["Mjlab-Homie-Unitree-H1", "Mjlab-Homie-Unitree-H1-with_hands"])
    parser.add_argument("--articulation", default=None,
                        help="URLab H1 articulation prefix. Default: the only one in the session.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=None,
                        help="Stop after this many policy steps. Default: forever (Ctrl+C).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_homierl_demo")

    checkpoint = args.checkpoint or os.path.join(args.demo_dir, "ckpt.pt")
    if not os.path.isfile(checkpoint):
        logger.error("missing checkpoint: %s", checkpoint)
        logger.error("download from https://huggingface.co/Nagi-ovo/HOMIERL-loco")
        return 2

    logger.info("connecting to URLab at %s (transport=%s)", args.address, args.transport)
    client = URLabClient(
        args.address,
        step_mode="direct",
        step_port=args.step_port,
        state_port=args.state_port,
        transport=args.transport,
    )
    client.discover()
    arts = sorted(client.articulations.keys())
    logger.info("session=%s articulations=%s", client.session_id, arts)

    prefix = args.articulation
    if prefix is None:
        if len(arts) != 1:
            logger.error("multiple articulations available %s; pass --articulation", arts)
            client.close()
            return 3
        prefix = arts[0]
    if prefix not in client.articulations:
        logger.error("articulation prefix %r not in session (have %s)", prefix, arts)
        client.close()
        return 3

    runner = MjlabRunner(
        client,
        task_id=args.task_id,
        checkpoint=checkpoint,
        articulation_prefix=prefix,
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
        while not stop and (args.steps is None or i < args.steps):
            runner.step()
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
