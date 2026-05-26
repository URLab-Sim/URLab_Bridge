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

"""Command source registry. Maps a `CommandSpec.source` string to a
factory that builds a context object exposing `.command` (the tensor
read by obs term `command_value` / mjlab's `generated_commands`) plus
any other read-only attributes per-source obs functions need.

A YAML loader maps a `commands:` section by string. The mjlab loader
maps an `mjlab.CommandTermCfg` subclass to one of these strings before
emitting the `TaskSpec`. Either way, the runner only ever sees the
string-indirected registry.

Built-in sources:
    - `urlab_twist`     : reads `art.twist_linear` / `art.twist_angular`
    - `motion_file`     : loads an mjlab-format motion .npz, advances per step
    - `constant`        : returns a fixed tensor (e.g., "stand still")

Custom sources register themselves before constructing a runner:

    from urlab_policy.command_sources import register_command_source
    register_command_source("my_height", lambda art, scene, entity, device, **params: MyHeightCtx(...))
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Sequence

logger = logging.getLogger(__name__)


# Factory: `(art, scene, entity, device, **params) -> context object`.
# The context object must expose `.command` (a torch.Tensor) and may
# expose additional attributes consumed by per-task obs functions
# (e.g. `anchor_pos_w` for tracking).
CommandSourceFactory = Callable[..., Any]
_COMMAND_SOURCE_REGISTRY: Dict[str, CommandSourceFactory] = {}


def register_command_source(name: str, factory: CommandSourceFactory) -> None:
    """Register a command-source factory under `name`. Later calls with
    the same name overwrite the previous entry."""
    _COMMAND_SOURCE_REGISTRY[name] = factory


def build_command_source(
    source: str, art, scene, entity, device: str, **params
) -> Any:
    """Look up `source` in the registry and call its factory. Raises
    `KeyError` with the available source names if unknown."""
    factory = _COMMAND_SOURCE_REGISTRY.get(source)
    if factory is None:
        raise KeyError(
            f"unknown command source {source!r}. Registered: "
            f"{sorted(_COMMAND_SOURCE_REGISTRY)}. Use "
            f"`urlab_policy.command_sources.register_command_source(...)` "
            f"to add a custom source."
        )
    return factory(art=art, scene=scene, entity=entity, device=device, **params)


# ---------------------------------------------------------------------------
# Built-in sources
# ---------------------------------------------------------------------------


class TwistSource:
    """Body-frame velocity command read live from URLab's
    `UMjTwistController` (WASD/gamepad in PIE). Returns
    `(num_envs, 3) = [vx, vy, yaw_rate]`. Falls back to zeros when no
    twist controller is attached on the UE side."""

    def __init__(self, art, device: str):
        import torch

        self._art = art
        self._buf = torch.zeros((1, 3), dtype=torch.float32, device=device)

    @property
    def command(self):
        lin = self._art.twist_linear
        ang = self._art.twist_angular
        self._buf[0, 0] = float(lin[0])
        self._buf[0, 1] = float(lin[1])
        self._buf[0, 2] = float(ang[2])
        return self._buf


class ConstantSource:
    """Returns a fixed constant tensor. Use for held command values you
    don't intend to drive interactively (e.g., a height-delta of zero
    meaning "stand at default height")."""

    def __init__(self, value: Sequence[float], device: str):
        import torch

        v = torch.tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
        self._buf = v

    @property
    def command(self):
        return self._buf


class MotionFileSource:
    """Wraps an mjlab-format motion .npz. Tracks `time_steps` per env
    and exposes the same property surface as
    `mjlab.tasks.tracking.mdp.commands.MotionCommand` so the standard
    tracking obs functions just work.

    Required `params`:
        - `motion_file`: path to the .npz
        - `body_names`: tracked-body short names (matches the policy
          training order)
        - `anchor_body_name`: which tracked body is the anchor frame
    """

    def __init__(
        self,
        art,
        scene,
        entity,
        device: str,
        motion_file: str,
        body_names: Sequence[str],
        anchor_body_name: str,
    ):
        import torch
        from types import SimpleNamespace
        from mjlab.tasks.tracking.mdp.commands import MotionLoader  # type: ignore
        from urlab_client._name_match import resolve_matching_names

        self.cfg_body_names = list(body_names)
        # mjlab's tracking obs functions read `command.cfg.body_names`
        # and `command.cfg.anchor_body_name`. Expose a SimpleNamespace
        # `cfg` with those fields so the obs functions just work.
        self.cfg = SimpleNamespace(
            body_names=list(body_names),
            anchor_body_name=anchor_body_name,
            motion_file=motion_file,
        )
        self.device = device
        self.num_envs = 1
        self._scene = scene
        self._entity = entity

        # Map tracked body names → entity-local body indices. mjlab's
        # MotionLoader expects body_indexes as positions into the entity's
        # body_names list.
        ids, _ = resolve_matching_names(
            self.cfg_body_names, list(entity.body_names), preserve_order=True
        )
        body_indexes = torch.tensor(ids, dtype=torch.long, device=device)
        self.body_indexes = body_indexes

        self._motion_anchor_idx = self.cfg_body_names.index(anchor_body_name)
        self.robot_anchor_body_index = list(entity.body_names).index(anchor_body_name)

        self.motion = MotionLoader(motion_file, body_indexes, device=device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)

    def advance(self) -> None:
        self.time_steps = (self.time_steps + 1).clamp(
            max=self.motion.time_step_total - 1
        )

    def frame0_joint_pos(self):
        return self.motion.joint_pos[0]

    def frame0_anchor_pos_w(self):
        return self.motion.body_pos_w[0, self._motion_anchor_idx]

    # ------ properties read by tracking obs functions ------

    @property
    def joint_pos(self):
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self):
        return self.motion.joint_vel[self.time_steps]

    @property
    def command(self):
        import torch

        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def body_pos_w(self):
        return (
            self.motion.body_pos_w[self.time_steps]
            + self._scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self):
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self):
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self):
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self):
        return (
            self.motion.body_pos_w[self.time_steps, self._motion_anchor_idx]
            + self._scene.env_origins
        )

    @property
    def anchor_quat_w(self):
        return self.motion.body_quat_w[self.time_steps, self._motion_anchor_idx]

    @property
    def anchor_lin_vel_w(self):
        return self.motion.body_lin_vel_w[self.time_steps, self._motion_anchor_idx]

    @property
    def anchor_ang_vel_w(self):
        return self.motion.body_ang_vel_w[self.time_steps, self._motion_anchor_idx]

    @property
    def robot_anchor_pos_w(self):
        return self._entity.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self):
        return self._entity.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self):
        return self._entity.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self):
        return self._entity.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_body_pos_w(self):
        return self._entity.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self):
        return self._entity.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self):
        return self._entity.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self):
        return self._entity.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_joint_pos(self):
        return self._entity.data.joint_pos

    @property
    def robot_joint_vel(self):
        return self._entity.data.joint_vel


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

register_command_source(
    "urlab_twist",
    lambda art, scene, entity, device, **_: TwistSource(art, device),
)
register_command_source(
    "constant",
    lambda art, scene, entity, device, value, **_: ConstantSource(value, device),
)
register_command_source(
    "motion_file",
    lambda art, scene, entity, device, **params: MotionFileSource(
        art, scene, entity, device, **params
    ),
)
