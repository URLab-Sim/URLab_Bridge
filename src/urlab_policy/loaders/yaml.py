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

"""YAML loader: build a `TaskSpec` from a static description file.

For policies not trained against an mjlab task config (your own RL
runs, Isaac Gym, mujoco_playground, etc.) you describe what the policy
expects in a YAML file. Each obs term names a builder by string; the
builder is looked up in `urlab_policy.obs_builders.OBS_BUILDER_REGISTRY`.

The YAML schema mirrors `TaskSpec`:

    robot_name: robot
    physics_dt: 0.005
    decimation: 4
    init_state:
      ".*_hip_pitch_joint": -0.312
      ".*_knee_joint": 0.669
    obs_groups:
      - name: actor
        terms:
          - {name: joint_pos, builder: joint_pos_rel, params: {biased: true}}
          - {name: joint_vel, builder: joint_vel_rel}
          - {name: imu_lin_vel, builder: builtin_sensor, params: {sensor_name: robot/imu_lin_vel}}
          - {name: actions, builder: last_action}
          - {name: command, builder: command_value, params: {command_name: twist}}
    action:
      kind: joint_position
      joint_patterns: [".*"]
      scale: 0.5
      use_default_offset: true
    commands:
      - {name: twist, source: urlab_twist}

This module is intentionally minimal -- the obs_builders library
referenced above isn't shipped yet (the mjlab loader covers all current
use cases by pointing builders at mjlab's own functions). Add to
`urlab_policy.obs_builders` as you onboard non-mjlab policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..task_spec import (
    ActionSpec,
    CommandSpec,
    ObsGroupSpec,
    ObsTermSpec,
    TaskSpec,
)


def _resolve_builder(name: str):
    """Look up an obs builder by string name. Local registry; extend as
    non-mjlab tasks land."""
    try:
        from .. import obs_builders  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"YAML loader needs `urlab_policy.obs_builders` to resolve "
            f"builder {name!r}, but the module isn't available: {exc}"
        )
    registry = getattr(obs_builders, "OBS_BUILDER_REGISTRY", {})
    builder = registry.get(name)
    if builder is None:
        raise KeyError(
            f"obs builder {name!r} not registered. Available: "
            f"{sorted(registry)}. Add it to "
            f"`urlab_policy.obs_builders.OBS_BUILDER_REGISTRY`."
        )
    return builder


def yaml_to_taskspec(path: str) -> TaskSpec:
    """Read a YAML file describing a policy and return a `TaskSpec`."""
    import yaml  # PyYAML, likely already a transitive dep

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    obs_groups: List[ObsGroupSpec] = []
    for group in raw.get("obs_groups", []):
        terms = []
        for t in group.get("terms", []):
            terms.append(ObsTermSpec(
                name=t["name"],
                builder=_resolve_builder(t["builder"]),
                params=dict(t.get("params") or {}),
                history_length=int(t.get("history_length", 0)),
                flatten_history_dim=bool(t.get("flatten_history_dim", True)),
                scale=t.get("scale"),
                clip=tuple(t["clip"]) if "clip" in t else None,
            ))
        obs_groups.append(ObsGroupSpec(
            name=group["name"],
            terms=terms,
            concatenate_dim=int(group.get("concatenate_dim", -1)),
        ))

    action_raw = raw.get("action") or {}
    action = ActionSpec(
        kind=action_raw.get("kind", "joint_position"),
        joint_patterns=list(action_raw.get("joint_patterns") or [".*"]),
        scale=action_raw.get("scale", 1.0),
        offset=action_raw.get("offset", 0.0),
        use_default_offset=bool(action_raw.get("use_default_offset", True)),
    )

    commands: List[CommandSpec] = []
    for c in raw.get("commands", []):
        commands.append(CommandSpec(
            name=c["name"],
            source=c["source"],
            params=dict(c.get("params") or {}),
        ))

    return TaskSpec(
        robot_name=raw.get("robot_name", "robot"),
        physics_dt=float(raw.get("physics_dt", 0.005)),
        decimation=int(raw.get("decimation", 4)),
        init_state={str(k): float(v) for k, v in (raw.get("init_state") or {}).items()},
        obs_groups=obs_groups,
        action=action,
        commands=commands,
        episode_length_s=raw.get("episode_length_s"),
    )
