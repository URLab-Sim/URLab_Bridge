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

"""`urlab-ui` — launch the dashboard."""

from __future__ import annotations


def main() -> None:
    try:
        from urlab_dashboard.app import main as run_dashboard
    except ModuleNotFoundError as exc:
        # The dashboard deps (dearpygui, opencv) live in the `ui` extra, which
        # a bare `uv sync` does not install. Turn the raw ModuleNotFoundError
        # into an actionable message instead of a traceback.
        raise SystemExit(
            f"urlab-ui needs the dashboard dependencies ({exc.name} is "
            "missing). Install them with:\n\n    uv sync --extra ui\n"
        ) from exc
    run_dashboard()


if __name__ == "__main__":
    main()
