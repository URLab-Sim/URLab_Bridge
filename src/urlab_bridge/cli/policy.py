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

"""`urlab-policy` — headless RoboJuDo policy loop on the legacy
streaming transport. Requires RoboJuDo installed."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .common import add_common_args, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="urlab-policy",
        description="Run a RoboJuDo policy pipeline against URLab over legacy streaming.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--policy", type=str, required=True,
        help="Policy name (unitree, unitree_nogait, ...).",
    )
    parser.add_argument(
        "--freq", type=int, default=50, help="Policy step rate in Hz.",
    )
    parser.add_argument(
        "--twist-source", choices=("keyboard", "zmq"), default="keyboard",
        help="Twist command source.",
    )
    args = parser.parse_args()
    setup_logging()

    try:
        from robojudo.pipeline.rl_pipeline import RlPipeline
        from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg
        from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
        from robojudo.config.g1.policy.g1_unitree_policy_cfg import (
            G1UnitreePolicyCfg,
            G1UnitreeWoGaitPolicyCfg,
        )
    except ImportError:
        logger.error(
            "RoboJuDo not installed. Run:\n"
            "  cd urlab_bridge/RoboJuDo && pip install -e ."
        )
        sys.exit(1)

    from urlab_policy.adapters.robojudo.env import G1UnrealEnvCfg
    import urlab_policy.adapters.robojudo  # noqa: F401  -- explicit register_controller()

    env_cfg = G1UnrealEnvCfg()
    env_cfg.state_endpoint = args.state_ep
    env_cfg.control_endpoint = args.ctrl_ep
    env_cfg.articulation_prefix = args.prefix

    if args.policy == "unitree":
        policy_cfg = G1UnitreePolicyCfg()
    elif args.policy == "unitree_nogait":
        policy_cfg = G1UnitreeWoGaitPolicyCfg()
    else:
        logger.error("Unknown policy: %s", args.policy)
        sys.exit(1)

    policy_cfg.freq = args.freq

    if args.twist_source == "zmq":
        from urlab_policy.adapters.robojudo.twist_ctrl import UnrealTwistCtrlCfg
        ctrl_cfgs = [UnrealTwistCtrlCfg()]
    else:
        ctrl_cfgs = [KeyboardCtrlCfg()]

    pipeline_cfg = RlPipelineCfg(
        robot="g1",
        env=env_cfg,
        ctrl=ctrl_cfgs,
        policy=policy_cfg,
    )

    logger.info(
        "Initializing pipeline: %s @ %dHz -> %s (twist: %s)",
        args.policy, args.freq, args.prefix, args.twist_source,
    )
    pipeline = RlPipeline(cfg=pipeline_cfg)

    logger.info("Preparing robot (moving to init pose)...")
    pipeline.prepare()

    logger.info("Running policy loop -- Ctrl+C to stop")
    dt = 1.0 / args.freq
    try:
        while True:
            step_start = time.time()
            pipeline.step()
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        pipeline.env.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
