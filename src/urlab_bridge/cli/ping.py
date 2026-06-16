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

"""`urlab-ping` — handshake against UZmqStepServer and print a summary.

Useful as a smoke test that the UE side is reachable and the wire
protocol matches the bridge's expectations."""

from __future__ import annotations

import argparse
import logging
import sys

from .common import add_common_args, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="urlab-ping",
        description="Handshake against the URLab step server and print the session summary.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--step-mode", choices=("auto", "direct", "puppet", "live"), default="auto",
        help="Step mode to negotiate with the server. Default: auto.",
    )
    args = parser.parse_args()
    setup_logging()

    from urlab_client import URLabClient

    logger.info("Handshake to %s:%d (step_mode=%s)", args.address, args.step_port, args.step_mode)
    client = URLabClient(
        args.address,
        step_mode=args.step_mode,
        step_port=args.step_port,
    )
    try:
        client.connect()
    except Exception as exc:
        logger.error("Handshake failed: %s", exc)
        sys.exit(1)
    try:
        logger.info(
            "Session: %s (urlab=%s mujoco=%s)",
            client.session_id, client.urlab_version, client.mujoco_version,
        )
        for prefix, art in client.articulations.items():
            logger.info(
                "  %s: %d actuators, %d joints, %d sensors (%s)",
                prefix, len(art.actuators), len(art.joints), len(art.sensors),
                art.control_mode.value,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
