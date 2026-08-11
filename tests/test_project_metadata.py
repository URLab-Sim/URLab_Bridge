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

"""Regression checks for the published dependency constraints.

The bridge and the plugin have to agree about MuJoCo. The MJB the server
hands over at handshake is a version-locked binary, and the policy adapters
import `mujoco` directly, so a pin that drifts from the plugin's submodule
build is not a packaging detail: it decides whether a client can adopt the
server's model at all.

These assert the pins rather than the resolved environment, so the failure
arrives when someone edits pyproject.toml, not when a user's install picks a
different wheel months later.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# The plugin builds MuJoCo from source at 3.11.1; 3.11.0 is the newest
# release on PyPI. The handshake compares major.minor, so this pin is
# deliberately one patch behind rather than unpinned.
EXPECTED_MUJOCO = "mujoco==3.11.0"
EXPECTED_MUJOCO_WARP = "mujoco-warp==3.11.0"


def _metadata() -> dict:
    root = Path(__file__).resolve().parent.parent
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_mujoco_pin_matches_the_urlab_runtime():
    assert EXPECTED_MUJOCO in _metadata()["project"]["dependencies"]


def test_mujoco_warp_pin_matches_mujoco():
    mjlab = _metadata()["project"]["optional-dependencies"]["mjlab"]
    assert EXPECTED_MUJOCO_WARP in mjlab


def test_the_two_mujoco_pins_agree_on_version():
    """mujoco-warp reaches into mujoco's enums, so a split pin breaks mjlab.

    The failure mode is quiet: the enum lookup raises at import time and the
    mjlab registry is skipped, leaving a runner that simply is not there.
    """
    meta = _metadata()
    mujoco = next(
        d for d in meta["project"]["dependencies"] if d.startswith("mujoco==")
    )
    warp = next(
        d
        for d in meta["project"]["optional-dependencies"]["mjlab"]
        if d.startswith("mujoco-warp==")
    )

    def release(spec: str) -> tuple[str, str, str]:
        # mujoco-warp carries a fourth build component (3.10.0.3), so compare
        # the release the two share rather than the literal strings.
        major, minor, patch = spec.split("==")[1].split(".")[:3]
        return major, minor, patch

    assert release(mujoco) == release(warp)
