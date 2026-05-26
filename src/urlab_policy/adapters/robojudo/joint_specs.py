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

"""Per-robot joint specs (names, default poses, PD gains, torque limits).

Plain-Python lists at the top work without RoboJuDo. The wrapped
``DoFConfig`` instances at the bottom only materialise when RoboJuDo is
on the path (a typed cfg used by ``robojudo`` policies and pipelines).

The :data:`ROBOTS` dict at the bottom is the canonical lookup —
``ROBOTS["g1_12dof"].joint_names`` etc. Registry entries in
``adapters/robojudo/registry.py`` carry a ``"robot"`` key referencing
this dict, so callers can derive joint metadata from a policy entry
without a second module lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class RobotSpec:
    """Per-robot joint metadata. Frozen so registry consumers can rely
    on it not mutating mid-session."""

    name: str                                   # short key, e.g. "g1_12dof"
    joint_names: Tuple[str, ...]
    default_pos: Tuple[float, ...]
    stiffness: Tuple[float, ...]
    damping: Tuple[float, ...]
    torque_limits: Tuple[float, ...]
    position_limits: Optional[Tuple[Tuple[float, float], ...]] = None
    xml_asset_key: str = ""                      # MJCF asset key under assets/models/

    @property
    def num_dofs(self) -> int:
        return len(self.joint_names)


try:
    from robojudo.tools.tool_cfgs import DoFConfig
    HAS_ROBOJUDO = True
except ImportError:
    HAS_ROBOJUDO = False


# G1 12-DOF (lower body only, locomotion policies)

G1_12DOF_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

G1_12DOF_DEFAULT_POS = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2

G1_12DOF_STIFFNESS = [100, 100, 100, 150, 40, 40] * 2
G1_12DOF_DAMPING = [2, 2, 2, 4, 2, 2] * 2
G1_12DOF_TORQUE_LIMITS = [88, 88, 88, 139, 50, 50] * 2
G1_12DOF_POSITION_LIMITS = [
    [-2.5307, 2.8798], [-0.5236, 2.9671], [-2.7576, 2.7576],
    [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618],
    [-2.5307, 2.8798], [-2.9671, 0.5236], [-2.7576, 2.7576],
    [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618],
]


# G1 29-DOF (full-body policies: BeyondMimic, AMO, H2H, ...)

G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

G1_29DOF_DEFAULT_POS = [
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[0, 0, 0],
    *[0, 0, 0, 0, 0, 0, 0],
    *[0, 0, 0, 0, 0, 0, 0],
]

# Gains matching BeyondMimic policy (most common 29DOF policy). Order
# matches G1_29DOF_JOINT_NAMES (XML joint order). Swap if you wire a
# different 29-DoF policy that needs different PD.
G1_29DOF_STIFFNESS = [
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 28.501, 28.501,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
]

G1_29DOF_DAMPING = [
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 1.814, 1.814,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
]

G1_29DOF_TORQUE_LIMITS = [
    *[200, 200, 200, 300, 40, 40],
    *[200, 200, 200, 300, 40, 40],
    *[200, 200, 200],
    *[40, 40, 18, 18, 10, 10, 10],
    *[40, 40, 18, 18, 10, 10, 10],
]


# Go2 12-DOF (walk-these-ways order: FL, FR, RL, RR)

GO2_12DOF_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

GO2_12DOF_DEFAULT_POS = [
    0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5,
]

GO2_12DOF_STIFFNESS = [25.0] * 12
GO2_12DOF_DAMPING = [0.6] * 12
GO2_12DOF_TORQUE_LIMITS = [45.0] * 12


# DoFConfig instances (RoboJuDo-typed)

G1_12DOF = None
G1_29DOF = None
GO2_12DOF = None

if HAS_ROBOJUDO:
    GO2_12DOF = DoFConfig(
        joint_names=GO2_12DOF_JOINT_NAMES,
        default_pos=GO2_12DOF_DEFAULT_POS,
        stiffness=GO2_12DOF_STIFFNESS,
        damping=GO2_12DOF_DAMPING,
        torque_limits=GO2_12DOF_TORQUE_LIMITS,
    )

    G1_12DOF = DoFConfig(
        joint_names=G1_12DOF_JOINT_NAMES,
        default_pos=G1_12DOF_DEFAULT_POS,
        stiffness=G1_12DOF_STIFFNESS,
        damping=G1_12DOF_DAMPING,
        torque_limits=G1_12DOF_TORQUE_LIMITS,
        position_limits=G1_12DOF_POSITION_LIMITS,
    )

    G1_29DOF = DoFConfig(
        joint_names=G1_29DOF_JOINT_NAMES,
        default_pos=G1_29DOF_DEFAULT_POS,
        stiffness=G1_29DOF_STIFFNESS,
        damping=G1_29DOF_DAMPING,
        torque_limits=G1_29DOF_TORQUE_LIMITS,
    )


# Canonical lookup. Registry entries reference these by key, so
# callers can do ``ROBOTS[entry["robot"]].joint_names`` without
# importing this module's per-robot constants directly.

ROBOTS = {
    "g1_12dof": RobotSpec(
        name="g1_12dof",
        joint_names=tuple(G1_12DOF_JOINT_NAMES),
        default_pos=tuple(G1_12DOF_DEFAULT_POS),
        stiffness=tuple(G1_12DOF_STIFFNESS),
        damping=tuple(G1_12DOF_DAMPING),
        torque_limits=tuple(G1_12DOF_TORQUE_LIMITS),
        position_limits=tuple(tuple(p) for p in G1_12DOF_POSITION_LIMITS),
        xml_asset_key="g1_12dof",
    ),
    "g1_29dof": RobotSpec(
        name="g1_29dof",
        joint_names=tuple(G1_29DOF_JOINT_NAMES),
        default_pos=tuple(G1_29DOF_DEFAULT_POS),
        stiffness=tuple(G1_29DOF_STIFFNESS),
        damping=tuple(G1_29DOF_DAMPING),
        torque_limits=tuple(G1_29DOF_TORQUE_LIMITS),
        position_limits=None,
        xml_asset_key="g1_29dof",
    ),
    "go2": RobotSpec(
        name="go2",
        joint_names=tuple(GO2_12DOF_JOINT_NAMES),
        default_pos=tuple(GO2_12DOF_DEFAULT_POS),
        stiffness=tuple(GO2_12DOF_STIFFNESS),
        damping=tuple(GO2_12DOF_DAMPING),
        torque_limits=tuple(GO2_12DOF_TORQUE_LIMITS),
        position_limits=None,
        xml_asset_key="go2",
    ),
}
