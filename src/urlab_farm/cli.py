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

"""``urlab-farm`` command-line entry point (``up`` / ``down`` / ``ps``).

Runnable as ``python -m urlab_farm``, or as the ``urlab-farm`` executable
installed via the ``[project.scripts]`` entry
(``urlab-farm = "urlab_farm.cli:main"``) in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

from urlab_client.pool import (
    DEFAULT_PORT_BASE,
    DEFAULT_PORT_STRIDE,
    InstanceInfo,
    default_registry_dir,
    pid_alive,
)

from . import launcher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urlab-farm",
        description="Launch and manage a pool of Unreal editor render instances.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="spawn N editor instances")
    up.add_argument("--count", "-n", type=int, required=True, help="number of instances")
    up.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    up.add_argument("--port-stride", type=int, default=DEFAULT_PORT_STRIDE)
    up.add_argument("--project", default=None, help="path to the .uproject")
    up.add_argument("--editor", default=None, help="path to UnrealEditor (or engine root)")
    up.add_argument(
        "--headless",
        action="store_true",
        help="run offscreen (-RenderOffscreen; keeps a real GPU for readback)",
    )
    up.add_argument(
        "--isolate-projects",
        action="store_true",
        help="give each instance its own project working copy (see docs)",
    )

    down = sub.add_parser("down", help="terminate the pool and clean the registry")
    down.add_argument("--registry-dir", default=None)

    listp = sub.add_parser("ps", help="print the registry")
    listp.add_argument("--registry-dir", default=None)

    return parser


def _cmd_up(args: argparse.Namespace) -> int:
    try:
        launched = launcher.up(
            args.count,
            port_base=args.port_base,
            port_stride=args.port_stride,
            project=args.project,
            editor=args.editor,
            headless=args.headless,
            isolate_projects=args.isolate_projects,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"urlab-farm up: {exc}", file=sys.stderr)
        return 2
    print(f"launched {len(launched)} instance(s):")
    header = f"{'idx':>3}  {'pid':>7}  {'step':>6}  {'state':>6}  {'cam_base':>8}  log"
    print(header)
    print("-" * len(header))
    for inst in launched:
        print(
            f"{inst.index:>3}  {inst.pid:>7}  {inst.step_port:>6}  "
            f"{inst.state_port:>6}  {inst.cam_base_port:>8}  {inst.log_path}"
        )
    return 0


def _cmd_down(args: argparse.Namespace) -> int:
    instances = launcher.down(args.registry_dir)
    if not instances:
        print("no instances in registry; nothing to stop")
        return 0
    print(f"terminated {len(instances)} instance(s):")
    for inst in instances:
        print(f"  {inst.instance_id or '?'} (pid {inst.pid})")
    return 0


def _fmt_caps(caps: Sequence[str]) -> str:
    return ",".join(caps) if caps else "-"


def _cmd_ps(args: argparse.Namespace) -> int:
    instances: List[InstanceInfo] = launcher.ps(args.registry_dir)
    registry_dir = args.registry_dir or default_registry_dir()
    if not instances:
        print(f"no instances registered under {registry_dir}")
        return 0
    header = (
        f"{'instance_id':<24}  {'idx':>3}  {'host':<15}  {'step':>6}  "
        f"{'busy':>4}  {'alive':>5}  capabilities"
    )
    print(header)
    print("-" * len(header))
    for inst in instances:
        alive = "yes" if pid_alive(inst.pid) else "no"
        print(
            f"{(inst.instance_id or '?'):<24}  {inst.index:>3}  "
            f"{inst.host:<15}  {inst.step_port:>6}  "
            f"{('yes' if inst.busy else 'no'):>4}  {alive:>5}  "
            f"{_fmt_caps(inst.capabilities)}"
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "up":
        return _cmd_up(args)
    if args.command == "down":
        return _cmd_down(args)
    if args.command == "ps":
        return _cmd_ps(args)
    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover - argparse exits first


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
