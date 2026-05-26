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

"""RoboJuDo-bound policy registry entries.

Every entry here points at a RoboJuDo policy cfg class (under
``robojudo.config.*``) plus an env cfg from
:mod:`urlab_policy.adapters.robojudo.env`. They're contributed to the
top-level :data:`urlab_policy.registry.POLICIES` map via the merge in
``urlab_policy/registry.py`` (a guarded import: missing torch /
RoboJuDo silently drops these entries).

The ``"robot"`` field references :data:`.joint_specs.ROBOTS` — joint
names, PD gains, default pose, etc. live there (one entry per
hardware platform, shared across the policies that target it).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .joint_specs import ROBOTS, RobotSpec


def robot_for(entry: Dict[str, Any]) -> Optional[RobotSpec]:
    """Resolve an entry's ``"robot"`` field to a :class:`RobotSpec`.
    Returns ``None`` when the entry has no robot key (legacy / mjlab /
    lerobot adapters that don't fit this RoboJuDo schema)."""
    key = entry.get("robot")
    return ROBOTS.get(key) if key else None


POLICIES = {
    "unitree_12dof": {
        "label": "G1 Unitree Locomotion (12 DOF)",
        "policy_cfg": "robojudo.config.g1.policy.g1_unitree_policy_cfg.G1UnitreePolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.G1UnrealEnvCfg",
        "robot": "g1_12dof",
        "dofs": 12,
        "xml": "g1_12dof",
        "desc": "Basic walking -- WASD twist control",
        "ctrl_type": "twist",
    },
    "unitree_wo_gait": {
        "label": "G1 Unitree Full Body (29 DOF)",
        "policy_cfg": "robojudo.config.g1.policy.g1_unitree_policy_cfg.G1UnitreeWoGaitPolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.G1_29UnrealEnvCfg",
        "robot": "g1_29dof",
        "dofs": 29,
        "xml": "g1_29dof",
        "desc": "Full body walking without gait clock",
        "ctrl_type": "twist",
    },
    "smooth": {
        "label": "G1 Smooth Locomotion (29 DOF)",
        "policy_cfg": "robojudo.config.g1.policy.g1_smooth_policy_cfg.G1SmoothPolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.G1_29UnrealEnvCfg",
        "robot": "g1_29dof",
        "dofs": 29,
        "xml": "g1_29dof",
        "desc": "Smoother walking policy",
        "ctrl_type": "twist",
    },
    "beyondmimic_dance": {
        "label": "G1 BeyondMimic Dance (29 DOF) [PHC]",
        "policy_cfg": "robojudo.config.g1.policy.g1_beyondmimic_policy_cfg.G1BeyondMimicPolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.G1_29UnrealEnvCfg",
        "robot": "g1_29dof",
        "dofs": 29,
        "xml": "g1_29dof",
        "desc": "Motion imitation -- dance. Requires PHC submodule",
        "ctrl_type": "motion",
        "requires_phc": True,
    },
    "h2h": {
        "label": "G1 Human2Humanoid (21 DOF) [PHC]",
        "policy_cfg": "robojudo.config.g1.policy.g1_h2h_policy_cfg.G1H2HPolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.G1_29UnrealEnvCfg",
        "robot": "g1_29dof",
        "dofs": 21,
        "xml": "g1_29dof",
        "desc": "Human motion retargeting. Requires PHC submodule",
        "ctrl_type": "motion_h2h",
        "requires_phc": True,
    },
    "amo": {
        "label": "G1 AMO Locomotion (29 DOF) [PHC]",
        "policy_cfg": "robojudo.config.g1.policy.g1_amo_policy_cfg.G1AmoPolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.G1_29UnrealEnvCfg",
        "robot": "g1_29dof",
        "dofs": 29,
        "xml": "g1_29dof",
        "desc": "Adaptive motion optimization. Requires PHC submodule",
        "ctrl_type": "twist",
        "requires_phc": True,
    },
    "twist_tracker": {
        "label": "G1 Twist General Motion (12 DOF) [PHC]",
        "policy_cfg": "robojudo.config.g1.policy.g1_twist_policy_cfg.G1TwistPolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.G1UnrealEnvCfg",
        "robot": "g1_12dof",
        "dofs": 12,
        "xml": "g1_12dof",
        "desc": "Motion tracker with twist. Requires PHC submodule",
        "ctrl_type": "motion_twist",
        "requires_phc": True,
    },
    # WTW Go2: policy class lives at ``urlab_policy.policies.wtw_policy``
    # and stays there (URLab-bundled). The registry entry is RoboJuDo-bound
    # because the launcher pipeline drives it through RoboJuDo's policy
    # registry decorator (see wtw_policy.py for the @policy_registry call).
    "go2_wtw": {
        "label": "Go2 Walk-These-Ways (12 DOF)",
        "policy_cfg": "robojudo.config.go2.policy.go2_wtw_policy_cfg.Go2WtwPolicyCfg",
        "env_cfg": "urlab_policy.adapters.robojudo.env.Go2UnrealEnvCfg",
        "robot": "go2",
        "dofs": 12,
        "xml": "go2",
        "desc": "Gait-conditioned agility -- rough terrain locomotion (WASD twist)",
        "ctrl_type": "twist",
    },
}
