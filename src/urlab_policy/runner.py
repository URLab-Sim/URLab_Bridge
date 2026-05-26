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

"""Framework-agnostic per-step policy runner. Drives a URLabArticulation
from a TaskSpec + policy callable; imports nothing from mjlab."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from urlab_client import URLabArticulation, URLabClient
from .task_spec import (
    ActionSpec,
    CommandSpec,
    ObsGroupSpec,
    ObsTermSpec,
    TaskSpec,
)
from .command_sources import build_command_source

logger = logging.getLogger(__name__)


class PolicyRunner:
    """Drive a URLab articulation with a pretrained policy described by
    a `TaskSpec`. The `env_shim` is whatever object the obs builders
    expect as their first argument -- typically a structure with
    `scene`, `action_manager`, `command_manager`. Callers construct the
    shim and pass it in; the runner doesn't care how it's built (mjlab
    loader builds an mjlab-shaped shim, a YAML loader could build a
    leaner one whose builders read URLab directly).
    """

    def __init__(
        self,
        client: URLabClient,
        art: URLabArticulation,
        spec: TaskSpec,
        env_shim: Any,
        policy: Callable[[Dict[str, Any]], Any],
        device: str = "cpu",
    ):
        import torch

        self.client = client
        self.art = art
        self.spec = spec
        self.env = env_shim
        self.policy = policy
        self.device = device
        self.num_envs = 1

        # Command sources. If the env shim already has a populated
        # `command_manager._terms` (the mjlab loader plugs it in before
        # the obs probe to avoid double motion-file loads), reuse those.
        # Otherwise build via the canonical string-keyed registry.
        existing = (
            getattr(getattr(env_shim, "command_manager", None), "_terms", None)
            or None
        )
        if existing:
            self.command_sources: Dict[str, Any] = dict(existing)
            for name, ctx in self.command_sources.items():
                logger.info("command %r -> %s (reused from env shim)",
                            name, type(ctx).__name__)
        else:
            self.command_sources = {}
            for cs in spec.commands:
                try:
                    ctx = build_command_source(
                        cs.source, art, env_shim.scene, env_shim.scene[spec.robot_name],
                        device, **cs.params,
                    )
                except Exception as exc:
                    logger.warning(
                        "command source %r (kind=%r) failed to build: %s -- "
                        "obs reading it will see zeros",
                        cs.name, cs.source, exc,
                    )
                    continue
                self.command_sources[cs.name] = ctx
                logger.info("command %r -> %s", cs.name, type(ctx).__name__)
            if hasattr(env_shim, "command_manager"):
                env_shim.command_manager._terms = self.command_sources  # type: ignore[attr-defined]

        # Resolve action mapping. mjlab's `JointPositionAction` is the
        # only kind we handle today. Patterns resolve over the obs side's
        # joint name list, which the env_shim already exposes via
        # `entity.joint_names`.
        if spec.action is None:
            raise ValueError("TaskSpec.action is required")
        from urlab_client._name_match import (
            resolve_matching_names,
            resolve_matching_names_values,
        )

        entity = env_shim.scene[spec.robot_name]
        joint_names = list(entity.joint_names)
        ids, names = resolve_matching_names(
            list(spec.action.joint_patterns), joint_names, preserve_order=True
        )
        self._action_joint_indices = ids
        self._action_urlab_keys: List[str] = []
        for n in names:
            key = art.resolve_actuator(n) or art.resolve_joint(n)
            if key is None:
                raise KeyError(
                    f"action joint {n!r} not found on URLab articulation {art.prefix!r}"
                )
            self._action_urlab_keys.append(key)
        self._num_actions = len(self._action_urlab_keys)

        # Pre-resolve scale + offset to flat (num_actions,) tensors.
        if isinstance(spec.action.scale, dict):
            scale_t = torch.ones(self._num_actions, dtype=torch.float32, device=device)
            idx_list, _, val_list = resolve_matching_names_values(
                spec.action.scale, names
            )
            scale_t[idx_list] = torch.tensor(
                val_list, dtype=torch.float32, device=device
            )
            self._action_scale = scale_t
        else:
            self._action_scale = float(spec.action.scale)

        offset_t = torch.zeros(self._num_actions, dtype=torch.float32, device=device)
        if isinstance(spec.action.offset, dict):
            idx_list, _, val_list = resolve_matching_names_values(
                spec.action.offset, names
            )
            offset_t[idx_list] = torch.tensor(
                val_list, dtype=torch.float32, device=device
            )
        else:
            offset_t[:] = float(spec.action.offset)
        if spec.action.use_default_offset:
            offset_t = offset_t + entity.data.default_joint_pos[
                0, self._action_joint_indices
            ]
        self._action_offset = offset_t

        # Frozen obs pipeline.
        self._obs_groups: List[
            Tuple[str, int, List[Tuple[ObsTermSpec, ...]]]
        ] = []
        for group in spec.obs_groups:
            self._obs_groups.append(
                (group.name, group.concatenate_dim, list(group.terms))
            )

        # Per-term rolling history buffer, lazily filled on first
        # compute (the first sample populates every slot so the policy
        # never sees a zero-padded prefix).
        self._obs_history: Dict[str, Dict[str, List[Any]]] = {
            g.name: {t.name: [] for t in g.terms if t.history_length > 0}
            for g in spec.obs_groups
        }

        # Step / decimation.
        self.cfg_decimation = int(spec.decimation)
        self.physics_dt = float(spec.physics_dt)
        self.step_dt = self.physics_dt * self.cfg_decimation

        # Pre-fill URLab's ctrl array with default joint positions for
        # every actuator the env shim's entity knows about, even ones
        # the action term doesn't drive. Tasks like HOMIERL declare
        # multiple action terms (lower body via JointPositionAction +
        # upper body via UpperBodyPoseAction with action_dim=0). The
        # policy only emits lower-body actions; without this prefill,
        # upper-body actuators would sit at ctrl=0 and the limbs would
        # drop. Per-step `set_ctrl(...)` writes only the action-driven
        # keys; non-driven slots retain whatever was last written.
        self._prefill_ctrl_to_defaults(entity)

    def _prefill_ctrl_to_defaults(self, entity: Any) -> None:
        """Write `default_joint_pos[i]` to URLab's ctrl slot for every
        joint name in the env shim's `entity.joint_names` list (i.e.
        every joint mjlab considers actuated). Non-driven actuators
        thus hold default pose instead of slamming to ctrl=0."""
        defaults = entity.data.default_joint_pos[0].detach().cpu().numpy()
        joint_names = list(entity.joint_names)
        prefill: Dict[str, float] = {}
        for i, name in enumerate(joint_names):
            key = self.art.resolve_actuator(name) or self.art.resolve_joint(name)
            if key is None or key not in self.art.actuators:
                continue
            prefill[key] = float(defaults[i])
        if prefill:
            self.art.set_ctrl(prefill)
            logger.info(
                "pre-filled ctrl for %d/%d actuators with default joint pos",
                len(prefill), len(joint_names),
            )

    # ----- obs computation -----

    def compute_observations(self) -> Dict[str, Any]:
        import torch

        out: Dict[str, Any] = {}
        for group_name, cat_dim, terms in self._obs_groups:
            parts = []
            history = self._obs_history.get(group_name, {})
            for t in terms:
                y = t.builder(self.env, **t.params)
                if t.scale is not None:
                    if isinstance(t.scale, torch.Tensor):
                        y = y * t.scale.to(y.device)
                    else:
                        y = y * float(t.scale)
                if t.clip is not None:
                    lo, hi = t.clip
                    y = y.clamp(min=lo, max=hi)
                if t.history_length > 0:
                    buf = history[t.name]
                    if not buf:
                        for _ in range(t.history_length):
                            buf.append(y.detach().clone())
                    else:
                        buf.append(y.detach().clone())
                        del buf[0]
                    stacked = torch.stack(buf, dim=1)
                    if t.flatten_history_dim:
                        stacked = stacked.reshape(stacked.shape[0], -1)
                    y = stacked
                parts.append(y)
            out[group_name] = torch.cat(parts, dim=cat_dim)
        return out

    def reset_obs_history(self) -> None:
        for group_buffers in self._obs_history.values():
            for buf in group_buffers.values():
                buf.clear()

    # ----- action decoding + step -----

    def step(self, hold_default: bool = False) -> None:
        import torch

        with torch.no_grad():
            if hold_default:
                action = torch.zeros(
                    (self.num_envs, self._num_actions),
                    dtype=torch.float32, device=self.device,
                )
            else:
                obs = self.compute_observations()
                action = self.policy(obs)

        # Cache for `last_action` obs builder + decode source.
        if hasattr(self.env, "action_manager"):
            self.env.action_manager.action = action.detach()

        # Decode and push.
        scaled = action[0] * self._action_scale + self._action_offset
        ctrl_np = scaled.detach().cpu().numpy()
        self.art.set_ctrl({
            k: float(v) for k, v in zip(self._action_urlab_keys, ctrl_np)
        })

        # Advance URLab.
        self.client.step(n_steps=self.cfg_decimation, observations="standard")

        # Tick any per-step command sources (e.g. motion file).
        for ctx in self.command_sources.values():
            advance = getattr(ctx, "advance", None)
            if callable(advance):
                advance()

    def run(self, num_steps: Optional[int] = None, hold_default: bool = False) -> None:
        i = 0
        while num_steps is None or i < num_steps:
            self.step(hold_default=hold_default)
            i += 1

    # ----- alignment helper for tracking tasks -----

    def align_to_motion(self) -> None:
        """If a `motion_file` command source is present, write its
        frame-0 joint pose into URLab and shift `scene.env_origins` so
        the motion's anchor lines up with the robot's current world
        position. No-op for tasks without motion."""
        import torch

        motion = next(
            (
                ctx for ctx in self.command_sources.values()
                if hasattr(ctx, "frame0_joint_pos") and hasattr(ctx, "frame0_anchor_pos_w")
            ),
            None,
        )
        if motion is None:
            return

        # 1. Per-articulation joint reset.
        frame0_jp = motion.frame0_joint_pos().detach().cpu().numpy()
        qpos_map = {
            urlab_key: float(val)
            for urlab_key, val in zip(self._action_urlab_keys, frame0_jp.tolist())
        }
        try:
            self.client.reset(per_articulation_qpos={self.art.prefix: qpos_map})
        except Exception as exc:
            logger.warning("client.reset() to motion frame-0 failed: %s", exc)
            return

        # 2. Shift env_origins.
        anchor_world = motion.frame0_anchor_pos_w().detach().cpu().numpy()
        entity = self.env.scene[self.spec.robot_name]
        robot_anchor_world = entity.data.body_link_pos_w[
            0, motion.robot_anchor_body_index
        ].detach().cpu().numpy()
        delta = robot_anchor_world - anchor_world
        self.env.scene.env_origins = torch.tensor(
            np.asarray([delta], dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )
        self.reset_obs_history()
        logger.info(
            "aligned to motion frame-0: anchor_delta=%s",
            [round(float(v), 3) for v in delta.tolist()],
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
