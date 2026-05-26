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

"""Shared argparse helpers for the urlab-* console scripts."""

from __future__ import annotations

import argparse
import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s -- %(message)s",
        datefmt="%H:%M:%S",
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--address", type=str, default="tcp://127.0.0.1",
        help="Base ZMQ address (no port). Default: tcp://127.0.0.1",
    )
    parser.add_argument(
        "--prefix", type=str, default="g1",
        help="Articulation prefix in Unreal",
    )
    parser.add_argument(
        "--step-port", type=int, default=5559,
        help="UZmqStepServer REP port (default 5559)",
    )
    parser.add_argument(
        "--state-ep", type=str, default="tcp://127.0.0.1:5555",
        help="Legacy streaming state PUB endpoint",
    )
    parser.add_argument(
        "--ctrl-ep", type=str, default="tcp://127.0.0.1:5556",
        help="Legacy streaming control SUB endpoint",
    )
