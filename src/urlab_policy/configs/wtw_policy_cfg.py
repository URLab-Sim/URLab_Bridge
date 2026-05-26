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

"""
Walk-These-Ways policy config for quadruped locomotion.
Extends RoboJuDo's PolicyCfg with WTW-specific observation structure and gait presets.
"""

from pathlib import Path
from typing import ClassVar
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.config import Config

# Bridge-local assets directory (`<repo>/assets/`). WTW checkpoints
# live under `assets/models/<robot>/wtw/` -- see the README there for
# download instructions.
_BRIDGE_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"


class WalkTheseWaysPolicyCfg(PolicyCfg):
    """Config for walk-these-ways gait-conditioned agility policies (quadrupeds)."""

    class ObsScalesCfg(Config):
        lin_vel: float = 2.0
        ang_vel: float = 0.25
        dof_pos: float = 1.0
        dof_vel: float = 0.05

    policy_type: str = "WalkTheseWaysPolicy"
    disable_autoload: bool = True

    @property
    def policy_file(self) -> str:
        return (_BRIDGE_ASSETS_DIR / f"models/{self.robot}/wtw/body_latest.jit").as_posix()

    @property
    def adaptation_module_file(self) -> str:
        return (_BRIDGE_ASSETS_DIR / f"models/{self.robot}/wtw/adaptation_module_latest.jit").as_posix()

    action_scale: float = 0.25
    action_clip: float | None = None
    action_beta: float = 1.0

    history_length: int = 30
    num_obs_per_step: int = 70
    num_commands: int = 15
    adaptation_latent_dim: int = 2

    @property
    def history_obs_size(self) -> int:
        return self.num_obs_per_step

    obs_scales: ObsScalesCfg = ObsScalesCfg()
    max_cmd: list[float] = [2.0, 0.6, 1.57]
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]

    hip_scale_reduction: float = 0.5

    default_gait_commands: list[float] = [
        0.0, 3.0, 0.5, 0.0, 0.0, 0.5,
        0.09, 0.0, 0.0, 0.25, 0.4, 0.0,
    ]

    GAIT_PRESETS: ClassVar[dict[str, list[float]]] = {
        "Trot":           [0.0,   3.0, 0.5, 0.0, 0.0, 0.5, 0.09, 0.0, 0.0, 0.25, 0.40, 0.0],
        "Relaxed Trot":   [0.0,   2.0, 0.5, 0.0, 0.0, 0.5, 0.12, 0.0, 0.0, 0.30, 0.42, 0.0],
        "High Step Trot": [0.0,   2.5, 0.5, 0.0, 0.0, 0.5, 0.30, 0.0, 0.0, 0.25, 0.40, 0.0],
        "Relaxed High":   [0.0,   2.0, 0.5, 0.0, 0.0, 0.5, 0.20, 0.0, 0.0, 0.30, 0.42, 0.0],
        "Fast Trot":      [0.0,   4.0, 0.5, 0.0, 0.0, 0.5, 0.09, 0.0, 0.0, 0.25, 0.40, 0.0],
        "Pronk":          [0.0,   2.0, 0.0, 0.0, 0.0, 0.5, 0.20, 0.0, 0.0, 0.25, 0.40, 0.0],
        "Bound":          [0.0,   3.0, 0.0, 0.5, 0.0, 0.5, 0.15, 0.0, 0.0, 0.25, 0.40, 0.0],
        "Pace":           [0.0,   3.0, 0.0, 0.0, 0.5, 0.5, 0.09, 0.0, 0.0, 0.25, 0.40, 0.0],
        "Low Crouch":     [-0.15, 3.0, 0.5, 0.0, 0.0, 0.5, 0.06, 0.0, 0.0, 0.25, 0.40, 0.0],
        "Wide Stance":    [0.0,   3.0, 0.5, 0.0, 0.0, 0.5, 0.09, 0.0, 0.0, 0.45, 0.45, 0.0],
    }
