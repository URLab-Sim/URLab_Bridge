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

"""mjlab-homierl compatibility shim (RoboJuDo adapter side).

`mjlab-homierl` pins ``mjlab>=1.2.0,<1.3.0`` and calls
``mjlab.utils.os.update_assets``, which was removed in mjlab 1.3.
We're pinned to mjlab 1.3.0 (everything else in the bridge expects the
1.3 API), so we monkey-patch a back-port of ``update_assets`` onto
``mjlab.utils.os`` before ``mjlab_homierl`` imports.

Import this module BEFORE any ``mjlab_homierl`` import::

    import urlab_policy.adapters.robojudo._compat  # noqa: F401
    import mjlab_homierl  # now safe

The monkey-patch is idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union


def _update_assets(
    assets: Dict[str, bytes],
    asset_dir: Union[str, Path],
    meshdir: str,
) -> None:
    """Walk ``asset_dir``, read each file as bytes, and write it into
    ``assets`` keyed by ``<meshdir>/<filename>``. Mirrors the mjlab 1.2
    helper of the same name -- the only behaviour ``mjlab-homierl``
    relies on."""
    base = Path(asset_dir)
    if not base.is_dir():
        return
    for path in base.iterdir():
        if not path.is_file():
            continue
        key = f"{meshdir}/{path.name}" if meshdir else path.name
        assets[key] = path.read_bytes()


def _install() -> None:
    import mjlab.utils.os as _mjlab_utils_os

    if not hasattr(_mjlab_utils_os, "update_assets"):
        _mjlab_utils_os.update_assets = _update_assets


_install()


def register_homierl_command_mappings() -> None:
    """Register the mjlab-loader mappings for HOMIERL's custom command
    types so ``mjlab_cfg_to_taskspec`` can convert a HOMIE env_cfg
    without edits to our adapter. Idempotent.

    Two mappings:
      - ``RelativeHeightCommandCfg`` -> ``constant`` source with a 1-D
        zero command ("stand at default height"; the policy never sees a
        non-zero height delta at eval).
      - ``UniformVelocityCommandCfg`` (HOMIERL's, NOT mjlab core's --
        different class, not a subclass) -> ``urlab_twist``. HOMIERL
        forked the cfg class so we register it explicitly.
    """
    from mjlab_homierl.mdp.velocity_command import (  # type: ignore
        RelativeHeightCommandCfg,
        UniformVelocityCommandCfg as HomieVelocityCommandCfg,
    )
    from urlab_policy.adapters.mjlab.loader import register_mjlab_command_mapping

    def _height_params(cfg):
        return {"value": [0.0]}

    register_mjlab_command_mapping(
        RelativeHeightCommandCfg, "constant", _height_params,
    )
    register_mjlab_command_mapping(
        HomieVelocityCommandCfg, "urlab_twist", lambda cfg: {},
    )
