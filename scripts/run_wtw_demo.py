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

"""End-to-end eval for the Walk-These-Ways quadruped policy on a Go2 in URLab.

Prerequisites:

    1. URLab editor in PIE with a Go2 actor imported. Add a
       `UMjTwistController` component to the Go2 BP if you want to
       drive it via WASD/gamepad in PIE.

    2. WTW assets present at `assets/models/go2/wtw/`:
         - body_latest.jit
         - adaptation_module_latest.jit

    3. RoboJuDo on the Python path (the `RoboJuDo` submodule next to
       this repo, installed in the same venv).

Run:

    uv run python scripts/run_wtw_demo.py
    uv run python scripts/run_wtw_demo.py --gait Pronk --duration 30
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time


_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_policy.native_runner import NativePolicyRunner  # noqa: E402
from urlab_client import URLabClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="tcp://localhost")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--transport", default="zmq", choices=["zmq", "shm"])
    parser.add_argument("--articulation", default=None,
                        help="URLab Go2 articulation prefix. Default: the only one in the session.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=None,
                        help="Stop after this many policy steps. Default: forever (Ctrl+C).")
    parser.add_argument("--gait", default="Trot",
                        help="WTW gait preset (e.g. Trot, Pronk, Bound, Pace).")
    parser.add_argument("--push-gains", action="store_true",
                        help="Push the policy's PD gains to URLab via the controller surface "
                             "(overrides the MJB-baked actuator gains for the Go2 BP).")
    parser.add_argument("--sim-dt", type=float, default=0.002,
                        help="Push this MuJoCo timestep into UE via set_sim_options before "
                             "stepping (default 0.002s — Menagerie go2 default). Pass 0 to skip.")
    parser.add_argument("--freq", type=float, default=50.0,
                        help="Wall-clock policy loop pacing in Hz (default 50). Without this "
                             "the loop runs as fast as the RPC roundtrip permits and sim time "
                             "outpaces wall clock.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_wtw_demo")

    # Construct the policy config + instance. WTW lives under the
    # `urlab_policy.policies` package; importing it auto-registers the
    # class with RoboJuDo's `policy_registry`.
    from urlab_policy.configs.go2.policy.go2_wtw_policy_cfg import Go2WtwPolicyCfg
    from urlab_policy.policies.wtw_policy import WalkTheseWaysPolicy

    cfg = Go2WtwPolicyCfg()
    if args.gait != "Trot":
        if args.gait not in cfg.GAIT_PRESETS:
            logger.error("unknown gait %r; available: %s",
                         args.gait, list(cfg.GAIT_PRESETS))
            return 2

    policy = WalkTheseWaysPolicy(cfg_policy=cfg, device=args.device)
    if args.gait != "Trot":
        policy.set_gait_preset(args.gait)

    logger.info("connecting to URLab at %s (transport=%s)", args.address, args.transport)
    client = URLabClient(
        args.address,
        step_mode="direct",
        step_port=args.step_port,
        state_port=args.state_port,
        transport=args.transport,
    )
    client.discover()

    if args.sim_dt > 0:
        try:
            client.set_sim_options(timestep=args.sim_dt)
            logger.info("pushed sim timestep=%.4fs to UE", args.sim_dt)
        except Exception as exc:
            logger.warning("set_sim_options failed: %s -- using whatever UE compiled with", exc)

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
    art = client.articulations[prefix]

    runner = NativePolicyRunner(
        client=client, art=art, policy=policy,
        push_gains=args.push_gains,
    )
    logger.info("policy ready (gait=%s). running...", args.gait)

    stop = False
    def _handler(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _handler)

    loop_dt = 1.0 / args.freq if args.freq > 0 else 0.0
    try:
        i = 0
        while not stop and (args.steps is None or i < args.steps):
            t0 = time.perf_counter()
            runner.step()
            i += 1
            if i % 50 == 0:
                logger.info("step=%d sim_time=%.3fs", i, client.sim_time)
            if loop_dt > 0:
                sleep_s = loop_dt - (time.perf_counter() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
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
