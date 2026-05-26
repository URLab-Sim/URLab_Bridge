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

"""TaskSpec: the contract between "the policy's expectations" and "how
URLab feeds it". Loaders (mjlab cfg, YAML) emit a TaskSpec; the runner
consumes one. The runner has no idea where the spec came from.

A TaskSpec captures only what the runner actually needs at inference:

- Robot binding (which URLab articulation to drive).
- Per-step physics rate (`physics_dt × decimation`).
- Per-joint init defaults (mjlab style: `{regex: value}`).
- An ordered list of obs groups, each with ordered obs terms. Each term
  carries a `builder` callable that the runner invokes per-step.
- The action decoding spec (joint patterns, scale, offset semantics).
- The command sources (twist input, motion file, ...). Built once at
  setup time; their `command` property is read by obs builders.

The dataclasses are deliberately minimal. Anything that's per-task
*logic* lives in the builders and command-source factories, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Builder signature: `(env, **params) -> torch.Tensor`. The first arg is
# whatever env shim the runner hands in -- typically an object with a
# `scene` mapping (`scene["robot"]` returns an entity facade with
# `.data.X` properties), an `action_manager` shim, and a
# `command_manager` shim. Builders read off this surface.
ObsBuilder = Callable[..., Any]


@dataclass
class ObsTermSpec:
    """One element of an obs group. The `builder` is invoked per-step
    with `(env, **params)`; its return is post-processed (scale → clip →
    history) and concatenated with siblings."""

    name: str
    builder: ObsBuilder
    params: Dict[str, Any] = field(default_factory=dict)
    history_length: int = 0
    flatten_history_dim: bool = True
    scale: Optional[Union[float, Any]] = None  # float or torch.Tensor
    clip: Optional[Tuple[float, float]] = None


@dataclass
class ObsGroupSpec:
    """A named group of obs terms. Locomotion / tracking policies use
    `actor` for inference and `critic` for value-fn estimation; rsl_rl
    needs both groups present even at eval time."""

    name: str
    terms: List[ObsTermSpec] = field(default_factory=list)
    concatenate_dim: int = -1


@dataclass
class ActionSpec:
    """How to decode a raw policy action vector into URLab ctrl values.

    `kind` is currently only `"joint_position"` (mjlab's
    `JointPositionAction` semantics: `ctrl = action * scale + offset`,
    where offset can absorb `default_joint_pos` when
    `use_default_offset=True`). Other kinds (joint_velocity,
    joint_effort) are easy to add when a policy needs them; the runner
    dispatches on `kind`.
    """

    kind: str
    joint_patterns: List[str]
    scale: Union[float, Dict[str, float]] = 1.0
    offset: Union[float, Dict[str, float]] = 0.0
    use_default_offset: bool = True


@dataclass
class CommandSpec:
    """Declares one command source the policy reads via
    `env.command_manager.get_command(name)`. The actual context object
    (with the `.command` tensor property) is built by a factory in the
    command-source registry, keyed by `source` string.

    `params` is passed to the factory as kwargs. For example:
        CommandSpec(name="motion", source="motion_file",
                    params={"motion_file": "...", "body_names": [...], "anchor_body_name": "pelvis"})
        CommandSpec(name="twist", source="urlab_twist")
        CommandSpec(name="height", source="constant", params={"value": [0.0]})
    """

    name: str
    source: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    """Complete declarative description of what a policy needs."""

    robot_name: str = "robot"
    physics_dt: float = 0.005
    decimation: int = 4
    init_state: Dict[str, float] = field(default_factory=dict)
    obs_groups: List[ObsGroupSpec] = field(default_factory=list)
    action: Optional[ActionSpec] = None
    commands: List[CommandSpec] = field(default_factory=list)
    # Episode budget for the rsl_rl wrapper, otherwise unused at eval.
    episode_length_s: Optional[float] = None
