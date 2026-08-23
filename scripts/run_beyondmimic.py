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

"""Run the G1 BeyondMimic loco-mimic pipeline against a URLab UE session.

This is the canonical example of running a RoboJuDo policy through the
**new** URLabClient API (vs the legacy ZMQ PUB/SUB `unreal_env.py` path).

The whole point is to be able to pick the **step mode** -- direct,
live, or (eventually) puppet -- because the BeyondMimic dances
were exhibiting drift that we suspected was a rate-mismatch artefact of
the legacy free-running path. Switch modes from the CLI; same policy,
same scene.

Usage (UE editor running with the G1 scene):

    uv run python scripts/run_beyondmimic.py --step-mode direct
    uv run python scripts/run_beyondmimic.py --step-mode live

Keyboard while running:
    [    -> activate mimic (start a dance)
    ]    -> activate loco (locomotion)
    ;    -> next dance / pose
    '    -> previous dance / pose
    o    -> shutdown
    i    -> sim reborn
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

logger = logging.getLogger("run_beyondmimic")


def _make_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="tcp://localhost",
                        help="URLab step-server address (default tcp://localhost)")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--step-mode", default="stepped",
                        choices=["stepped", "freerun"],
                        help="step mode (default: direct -- tightest "
                             "policy/physics coupling, recommended for "
                             "motion-tracking).")
    parser.add_argument("--articulation-prefix", default="",
                        help="UE articulation prefix (auto if a single "
                             "articulation is present).")
    parser.add_argument("--sim-decimation", type=int, default=10,
                        help="physics ticks per env step (default 10 -- "
                             "500 Hz physics / 50 Hz policy)")
    parser.add_argument("--sim-dt", type=float, default=0.002,
                        help="physics timestep, must match UE (default 0.002s)")
    parser.add_argument("--no-push-gains", action="store_true",
                        help="don't sync RoboJuDo PD gains to the UE controller "
                             "(use whatever UE has configured)")
    parser.add_argument("--observation-level", default="full",
                        choices=["minimal", "standard", "full"],
                        help="observation level (default: full -- needed for FK)")
    parser.add_argument("--freq", type=float, default=50.0,
                        help="loop pacing in Hz (default 50)")
    parser.add_argument("--prepare", action="store_true",
                        help="run Pipeline.prepare() (1000-step move-to-init-pose) "
                             "before the main loop. Off by default since URLab's "
                             "actor already starts at the BP-default pose.")
    parser.add_argument("--verbose", action="store_true",
                        help="DEBUG-level logging")
    args = parser.parse_args()

    _make_logger(args.verbose)

    # RoboJuDo imports happen here so the help text works without it installed.
    try:
        from robojudo.config import cfg_registry
        from robojudo.config.g1.g1_loco_mimic_cfg import g1_locomimic_beyondmimic  # noqa: F401  -- registers the cfg
        from robojudo.environment import env_registry  # noqa: F401
        from robojudo.pipeline import pipeline_registry
        from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline  # noqa: F401
    except ImportError as exc:
        logger.error("RoboJuDo not installed: %s", exc)
        logger.error("Install it from urlab_bridge/RoboJuDo: pip install -e .")
        return 1

    # Register our env subclass (must be imported AFTER robojudo core so the
    # @env_registry.register decorator runs).
    from urlab_policy.adapters.robojudo import (  # noqa: F401
        G1_29URLabRoboJuDoEnvCfg,
        URLabRoboJuDoEnv,  # noqa: F401  -- registers in env_registry
    )

    # Pull the registered loco-mimic config; swap the env for our URLab one.
    pipeline_cfg = cfg_registry.get("g1_locomimic_beyondmimic")()
    pipeline_cfg.env = G1_29URLabRoboJuDoEnvCfg(
        address=args.address,
        step_port=args.step_port,
        state_port=args.state_port,
        step_mode=args.step_mode,
        sim_dt=args.sim_dt,
        sim_decimation=args.sim_decimation,
        articulation_prefix=args.articulation_prefix,
        observation_level=args.observation_level,
        push_gains_to_unreal=not args.no_push_gains,
        born_place_align=True,
    )

    logger.info(
        "Pipeline: %s | env=URLabRoboJuDoEnv (step_mode=%s, decim=%d, dt=%.4fs)",
        pipeline_cfg.__class__.__name__,
        args.step_mode, args.sim_decimation, args.sim_dt,
    )
    logger.info(
        "Policies: loco=%s, mimics=%s",
        pipeline_cfg.loco_policy.__class__.__name__,
        [p.policy_name for p in pipeline_cfg.mimic_policies],
    )

    pipeline_cls = pipeline_registry.get(pipeline_cfg.pipeline_type)
    pipeline = pipeline_cls(cfg=pipeline_cfg)

    # The 1000-step move-to-init-pose is off by default — URLab spawns at
    # the BP pose and `set_born_place` re-anchors the policy on the first update.
    if getattr(args, "prepare", False):
        logger.info("Preparing robot (moving to init pose)...")
        pipeline.prepare()

    logger.info("Running -- Ctrl+C to stop. Keyboard: [ mimic ] loco ; next ' prev")
    dt = 1.0 / args.freq
    stop = False

    def _on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        while not stop:
            t0 = time.perf_counter()
            pipeline.step()
            elapsed = time.perf_counter() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
    except Exception:
        logger.exception("pipeline crashed")
        return 2
    finally:
        try:
            pipeline.env.shutdown()
        except Exception:
            logger.exception("env shutdown failed")
        logger.info("done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
