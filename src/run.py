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

"""Compatibility shim for `uv run src/run.py --<flag>`.

The per-mode logic lives in `urlab_bridge.cli.{ui, ping, test, policy}`,
each wired into `pyproject.toml` `[project.scripts]` as a console script
(`urlab-ui`, `urlab-ping`, `urlab-test`, `urlab-policy`). Prefer those
when invoking from the command line directly.

This shim survives the old README's instructions:

    uv run src/run.py --ui
    uv run src/run.py --ping --prefix vx300s
    uv run src/run.py --test --prefix g1
    uv run src/run.py --policy unitree --prefix g1
"""

from __future__ import annotations

import sys


_FLAG_TO_MODULE = {
    "--ui":     ("urlab_bridge.cli.ui",     "main"),
    "--ping":   ("urlab_bridge.cli.ping",   "main"),
    "--test":   ("urlab_bridge.cli.test",   "main"),
    "--policy": ("urlab_bridge.cli.policy", "main"),
}


def main() -> None:
    flag = next((arg for arg in sys.argv[1:] if arg in _FLAG_TO_MODULE), None)
    if flag is None:
        print(
            "usage: run.py {--ui | --ping | --test | --policy NAME} [options]\n"
            "       (use the per-mode console scripts directly: "
            "urlab-ui, urlab-ping, urlab-test, urlab-policy)"
        )
        sys.exit(2)

    # Strip the dispatch flag; per-mode argparse doesn't expect it.
    if flag != "--policy":
        sys.argv = [arg for arg in sys.argv if arg != flag]

    module_name, attr = _FLAG_TO_MODULE[flag]
    import importlib
    mod = importlib.import_module(module_name)
    getattr(mod, attr)()


if __name__ == "__main__":
    main()
