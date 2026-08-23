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

"""mjlab integration adapter — drives URLab physics from an mjlab-trained policy.

Sensor stubs (touch / FT / height-scan) return zeros; tasks that observe
beyond the standard locomotion set will not produce correct obs.
Single-env eval only (URLab is single-process). To add an obs field,
extend ``_EntityDataFacade``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from urlab_client import URLabArticulation, URLabClient

logger = logging.getLogger(__name__)


# Command context registry. mjlab CommandTerm classes pull in the manager
# scaffolding we deliberately removed, so each command type maps to a shim
# exposing a `.command` property. Register custom types BEFORE constructing
# MjlabRunner via `register_command_context(cfg_class, factory)`; unknown
# types fall through with a warning and the obs term reads zeros.

CommandContextFactory = Callable[
    [Any, "URLabArticulation", "_SceneFacade", "_EntityFacade", str], Any
]
_COMMAND_CONTEXT_REGISTRY: Dict[type, CommandContextFactory] = {}


def register_command_context(
    cfg_class: type, factory: CommandContextFactory
) -> None:
    """Register a context shim for an mjlab CommandCfg subclass.

    `factory(cfg, art, scene, entity, device)` returns an object with a
    `.command` property (torch tensor of the shape the obs terms expect)
    plus any other attributes the task's obs functions read off the
    command term (e.g. `anchor_pos_w` for tracking).
    """
    _COMMAND_CONTEXT_REGISTRY[cfg_class] = factory


def _resolve_command_context(
    cfg: Any,
    art: "URLabArticulation",
    scene: "_SceneFacade",
    entity: "_EntityFacade",
    device: str,
) -> Optional[Any]:
    """Look up `type(cfg)` (and its MRO) in the registry, build the
    context. Returns None if no factory is registered."""
    for cls in type(cfg).__mro__:
        factory = _COMMAND_CONTEXT_REGISTRY.get(cls)
        if factory is not None:
            return factory(cfg, art, scene, entity, device)
    return None


def _require_mjlab():
    """Lazy import. Raises a clear error if mjlab isn't installed."""
    try:
        import mjlab  # noqa: F401  # type: ignore
        import torch  # noqa: F401  # type: ignore
        return mjlab
    except ImportError as exc:
        raise ImportError(
            "URLabMjlabRunner requires `mjlab` (and `torch`) to be installed. "
            "See https://github.com/mujocolab/mjlab for installation."
        ) from exc


# ---------------------------------------------------------------------------
# Fake EntityData -- exposes URLab state under mjlab's EntityData property
# names, with a leading num_envs=1 batch dim and torch tensors.
# ---------------------------------------------------------------------------


class _EntityDataFacade:
    """Fake `mjlab.entity.EntityData` over a `URLabArticulation`.

    Properties match `mjlab.entity.data.EntityData` field names so mjlab's
    obs / action term implementations work without modification. All
    properties produce torch tensors with shape `(1, ...)` -- mjlab's
    leading batch dim.

    `default_joint_pos` / `default_joint_vel` / `encoder_bias` are loaded
    from the task config at adapter init and stored as fixed tensors;
    URLab doesn't carry them. Joint targets written by mjlab's action
    terms (`joint_pos_target`, `joint_vel_target`, `joint_effort_target`)
    land in the corresponding buffers on this facade, ready to be
    forwarded to URLab.
    """

    def __init__(self, art: URLabArticulation, joint_names: Sequence[str], device: str):
        import torch

        self._art = art
        self._device = device
        self._joint_names = list(joint_names)

        # Resolve each mjlab joint name to a key on URLab's articulation.
        self._joint_keys: List[str] = []
        for name in self._joint_names:
            key = art.resolve_joint(name) or art.resolve_actuator(name)
            if key is None:
                raise KeyError(
                    f"mjlab task references joint {name!r} which doesn't resolve "
                    f"to any URLab joint or actuator on articulation {art.prefix!r}"
                )
            self._joint_keys.append(key)

        # Index each tracked joint into URLab's local qpos / qvel array.
        self._dof_qpos_idx: List[int] = []
        self._dof_qvel_idx: List[int] = []
        head_qpos = 0
        head_qvel = 0
        if art.has_free_base and art._free_base_joint is not None:
            head_qpos = (
                art._free_base_joint.qpos_local_offset
                + art._free_base_joint.qpos_dim
            )
            head_qvel = (
                art._free_base_joint.qvel_local_offset
                + art._free_base_joint.qvel_dim
            )
        # Build a name->local-qpos-offset map (skip free joint head).
        # Joints in URLab walk MjModel jid order; we trust resolve_joint
        # to give us valid keys, then look up offsets by walking joints.
        name_to_qpos = {}
        name_to_qvel = {}
        for jname, j in art.joints.items():
            if j.qpos_dim != 1:
                continue  # skip free joint, ball joint, etc.
            name_to_qpos[jname] = j.qpos_local_offset
            name_to_qvel[jname] = j.qvel_local_offset
        for key in self._joint_keys:
            if key not in name_to_qpos:
                raise KeyError(
                    f"joint {key!r} on articulation {art.prefix!r} isn't a single-DoF "
                    "joint; mjlab adapter only supports 1-DoF joint observations"
                )
            self._dof_qpos_idx.append(name_to_qpos[key])
            self._dof_qvel_idx.append(name_to_qvel[key])

        n = len(self._joint_keys)
        # Defaults / encoder bias: filled in from task config by the adapter
        # via `set_defaults_from_cfg`. Zero until set.
        self.default_joint_pos = torch.zeros((1, n), dtype=torch.float32, device=device)
        self.default_joint_vel = torch.zeros((1, n), dtype=torch.float32, device=device)
        self.encoder_bias = torch.zeros((1, n), dtype=torch.float32, device=device)
        # Soft joint pos limits: copy from MjModel jnt_range when available.
        self.soft_joint_pos_limits = torch.zeros(
            (1, n, 2), dtype=torch.float32, device=device
        )
        for i, key in enumerate(self._joint_keys):
            j = art.joints[key]
            if j.range is not None:
                self.soft_joint_pos_limits[0, i, 0] = float(j.range[0])
                self.soft_joint_pos_limits[0, i, 1] = float(j.range[1])

        # Action target buffers -- written by mjlab's action terms via the
        # `set_joint_*_target` methods on the entity. The adapter reads
        # them out each step and forwards to URLab as ctrl.
        self.joint_pos_target = torch.zeros_like(self.default_joint_pos)
        self.joint_vel_target = torch.zeros_like(self.default_joint_pos)
        self.joint_effort_target = torch.zeros_like(self.default_joint_pos)

        # Cache of `_quat_apply_inverse` for mjlab's gravity vec (world -1z).
        self._gravity_world = torch.tensor(
            [[0.0, 0.0, -1.0]], dtype=torch.float32, device=device
        )

        # Body indexing. mjlab's `Entity.body_names` is per-entity; tensors
        # like `body_link_pos_w` have shape `(num_envs, len(body_names), 3)`
        # and `find_bodies` returns positions into that list. Cache the
        # global mj body ids in `art.bodies.keys()` order so we can slice
        # `data.xpos[body_ids]` once and hand back an entity-local tensor.
        self._body_ids = np.asarray(
            [b.id for b in art.bodies.values()], dtype=np.int64
        )

    # ---- joint-level state (per-DoF) ----

    @property
    def joint_pos(self) -> "torch.Tensor":
        import torch

        # URLab's qpos_array is local, dense; index by precomputed offsets.
        v = self._art.qpos_array
        out = torch.tensor(
            [v[i] for i in self._dof_qpos_idx], dtype=torch.float32, device=self._device
        )
        return out.unsqueeze(0)  # (1, n)

    @property
    def joint_pos_biased(self) -> "torch.Tensor":
        return self.joint_pos + self.encoder_bias

    @property
    def joint_vel(self) -> "torch.Tensor":
        import torch

        v = self._art.qvel_array
        out = torch.tensor(
            [v[i] for i in self._dof_qvel_idx], dtype=torch.float32, device=self._device
        )
        return out.unsqueeze(0)

    @property
    def joint_acc(self) -> "torch.Tensor":
        # URLab doesn't expose qacc per articulation. Return zeros; obs
        # terms that depend on this will be flat (typical RL policies
        # don't observe joint accelerations, so this is rarely an issue).
        import torch

        return torch.zeros_like(self.joint_pos)

    # ---- root-link state ----

    @property
    def root_link_pos_w(self) -> "torch.Tensor":
        import torch

        v = self._art.root_pos_w
        return torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)

    @property
    def root_link_quat_w(self) -> "torch.Tensor":
        import torch

        # mjlab uses (w, x, y, z) order. URLab's root_quat_w matches.
        v = self._art.root_quat_w
        return torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)

    @property
    def root_link_pose_w(self) -> "torch.Tensor":
        import torch

        return torch.cat([self.root_link_pos_w, self.root_link_quat_w], dim=-1)

    @property
    def root_link_lin_vel_w(self) -> "torch.Tensor":
        import torch

        v = self._art.root_lin_vel_w
        return torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)

    @property
    def root_link_ang_vel_w(self) -> "torch.Tensor":
        import torch

        v = self._art.root_ang_vel_w
        return torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)

    @property
    def root_link_lin_vel_b(self) -> "torch.Tensor":
        import torch

        v = self._art.root_lin_vel_b
        return torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)

    @property
    def root_link_ang_vel_b(self) -> "torch.Tensor":
        import torch

        v = self._art.root_ang_vel_b
        return torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)

    @property
    def projected_gravity_b(self) -> "torch.Tensor":
        import torch

        v = self._art.projected_gravity_b
        return torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)

    # ---- per-body pose / velocity (world frame) ----
    #
    # mjlab obs terms like `motion_anchor_pos_b` / `robot_body_pos_b` /
    # `body_pos_w` index into per-body arrays. mjlab's natural shape is
    # `(num_envs, num_bodies, K)` with body indices matching the MuJoCo
    # MjModel.body order. We mirror that by reading directly from
    # `client.data.xpos` / `xquat` -- mj_forward keeps these current for
    # every body in the model after each step reply absorption.

    @property
    def body_link_pos_w(self) -> "torch.Tensor":
        import torch

        client = self._art._client
        n = len(self._body_ids)
        if client is None or client.data is None or n == 0:
            return torch.zeros((1, max(n, 1), 3), dtype=torch.float32, device=self._device)
        xpos = np.asarray(client.data.xpos, dtype=np.float32)[self._body_ids]
        return torch.tensor(xpos, device=self._device).unsqueeze(0)

    @property
    def body_link_quat_w(self) -> "torch.Tensor":
        import torch

        client = self._art._client
        n = len(self._body_ids)
        if client is None or client.data is None or n == 0:
            return torch.zeros((1, max(n, 1), 4), dtype=torch.float32, device=self._device)
        xquat = np.asarray(client.data.xquat, dtype=np.float32)[self._body_ids]
        return torch.tensor(xquat, device=self._device).unsqueeze(0)

    @property
    def body_link_pose_w(self) -> "torch.Tensor":
        import torch

        return torch.cat([self.body_link_pos_w, self.body_link_quat_w], dim=-1)

    @property
    def body_link_lin_vel_w(self) -> "torch.Tensor":
        """Per-body world-frame linear velocity, derived from `cvel`.

        mjData.cvel layout is `[ang_vel(3), lin_vel(3)]` per body in the
        body's COM frame. To convert to world-frame linear velocity at the
        body's link (origin), apply `v_link = lin_com - ang × (subtree_com - pos)`.
        Result is sliced to entity-local body order."""
        import torch

        client = self._art._client
        n = len(self._body_ids)
        if client is None or client.data is None or n == 0:
            return torch.zeros((1, max(n, 1), 3), dtype=torch.float32, device=self._device)
        d = client.data
        ids = self._body_ids
        cvel = np.asarray(d.cvel, dtype=np.float32)[ids]
        ang = cvel[:, 0:3]
        lin_com = cvel[:, 3:6]
        pos = np.asarray(d.xpos, dtype=np.float32)[ids]
        subtree_com = np.asarray(d.subtree_com, dtype=np.float32)[ids]
        offset = subtree_com - pos
        lin_link = lin_com - np.cross(ang, offset)
        return torch.tensor(lin_link, device=self._device).unsqueeze(0)

    @property
    def body_link_ang_vel_w(self) -> "torch.Tensor":
        import torch

        client = self._art._client
        n = len(self._body_ids)
        if client is None or client.data is None or n == 0:
            return torch.zeros((1, max(n, 1), 3), dtype=torch.float32, device=self._device)
        ang = np.asarray(client.data.cvel, dtype=np.float32)[self._body_ids, 0:3]
        return torch.tensor(ang, device=self._device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Fake Sensor -- exposes `.data` reading from URLab's `art.sensors[name].latest`
# ---------------------------------------------------------------------------


# mjlab's `builtin_sensor` obs term asserts `isinstance(sensor, BuiltinSensor)`
# before reading `.data`, so the facade has to subclass it (duck-typing isn't
# enough). The import sits behind a try so this module stays importable on
# bridge installs that don't have mjlab; downstream code that actually builds
# a MjlabRunner already requires the dependency.
try:
    from mjlab.sensor.builtin_sensor import BuiltinSensor as _BuiltinSensor  # type: ignore
except ImportError:
    _BuiltinSensor = object  # type: ignore[assignment,misc]


class _SensorFacade(_BuiltinSensor):  # type: ignore[misc,valid-type]
    """`BuiltinSensor` subclass that pulls live data from a URLab sensor.

    mjlab's `builtin_sensor` obs term does:
        sensor = env.scene[sensor_name]
        assert isinstance(sensor, BuiltinSensor)
        return sensor.data
    We subclass via the `name`-only ctor (cfg=None) so we satisfy the
    isinstance check without owning a `BuiltinSensorCfg`, then override
    `data` to read from URLab's `art.sensors[short_name].latest`.
    `latest` is `None` until the first step reply -- return zeros in
    that case so observation-shape probing during init doesn't choke.
    """

    def __init__(self, art: URLabArticulation, short_name: str, device: str):
        if _BuiltinSensor is not object:
            super().__init__(cfg=None, name=short_name)
        self._art = art
        self._short_name = short_name
        self._device = device

    @property  # type: ignore[override]
    def data(self) -> "torch.Tensor":
        import torch

        sensor = self._art.sensors.get(self._short_name)
        if sensor is None or sensor.latest is None:
            dim = sensor.dim if sensor is not None else 0
            return torch.zeros((1, max(dim, 1)), dtype=torch.float32, device=self._device)
        arr = np.asarray(sensor.latest, dtype=np.float32)
        return torch.tensor(arr, device=self._device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Fake Entity -- exposes find_joints_by_actuator_names + set_joint_*_target,
# wraps an _EntityDataFacade.
# ---------------------------------------------------------------------------


class _EntityFacade:
    """Fake `mjlab.entity.Entity` over a `URLabArticulation`.

    Implements the small subset of `Entity` methods mjlab's action terms
    call: `find_joints_by_actuator_names`, `set_joint_position_target`,
    `set_joint_velocity_target`, `set_joint_effort_target`, and the
    `.data` accessor.
    """

    def __init__(self, art: URLabArticulation, joint_names: Sequence[str], device: str):
        self._art = art
        self.data = _EntityDataFacade(art, joint_names, device)
        self._device = device

    @property
    def num_instances(self) -> int:
        return 1

    @property
    def device(self) -> str:
        return self._device

    @property
    def body_names(self) -> List[str]:
        """Short body names in MJB id order. mjlab's tracking command resolves
        anchor / tracked-body names by `body_names.index(name)`, so the order
        and naming must match what the policy was trained against. URLab
        keys `art.bodies` by short XML name in `mj_id` walk order, which
        matches mjlab's spec order for the same robot."""
        return list(self._art.bodies.keys())

    @property
    def joint_names(self) -> List[str]:
        """Joint names in the order the policy expects (taken from the
        mjlab task's robot Entity at adapter init)."""
        return list(self.data._joint_names)

    def find_joints_by_actuator_names(
        self, actuator_names: Sequence[str], preserve_order: bool = False
    ) -> Tuple[List[int], List[str]]:
        """Match regex patterns against the adapter's `joint_names` list
        and return matched indices + names. Mirrors mjlab's
        `Entity.find_joints_by_actuator_names`: every entry in the joint
        list came from a procedurally-built mjlab `Entity`, so each is
        already actuated -- no extra filtering needed."""
        from urlab_client._name_match import resolve_matching_names

        keys = [actuator_names] if isinstance(actuator_names, str) else list(actuator_names)
        return resolve_matching_names(keys, self.data._joint_names, preserve_order)

    def find_bodies(
        self, body_names: Sequence[str], preserve_order: bool = False
    ) -> Tuple[List[int], List[str]]:
        """Match regex patterns against the entity-local body name list and
        return positions into that list. Mirrors mjlab's `Entity.find_bodies`,
        which is what `entity.data.body_link_pos_w` is indexed by (shape
        `(num_envs, len(body_names), ...)`)."""
        from urlab_client._name_match import resolve_matching_names

        keys = [body_names] if isinstance(body_names, str) else list(body_names)
        return resolve_matching_names(keys, self.body_names, preserve_order)

    def set_joint_position_target(
        self, target: "torch.Tensor", joint_ids: "torch.Tensor"
    ) -> None:
        self.data.joint_pos_target[:, joint_ids] = target

    def set_joint_velocity_target(
        self, target: "torch.Tensor", joint_ids: "torch.Tensor"
    ) -> None:
        self.data.joint_vel_target[:, joint_ids] = target

    def set_joint_effort_target(
        self, target: "torch.Tensor", joint_ids: "torch.Tensor"
    ) -> None:
        self.data.joint_effort_target[:, joint_ids] = target


# ---------------------------------------------------------------------------
# Fake Scene + Env -- enough for ObservationManager / ActionManager to run.
# ---------------------------------------------------------------------------


class _SceneFacade:
    """`scene[name]` returns the fake entity or sensor.

    mjlab obs terms use two name shapes:
      * Bare entity name (`"robot"`) -> resolved via `_DEFAULT_ASSET_CFG`.
      * `"<entity>/<sensor>"` (e.g. `"robot/imu_lin_vel"`) -> resolved by
        `mdp.builtin_sensor` to a `BuiltinSensor`-shaped object exposing
        `.data` as a torch tensor.

    Anything else falls through to a `KeyError`."""

    def __init__(
        self,
        entities: Dict[str, _EntityFacade],
        sensors: Dict[str, _SensorFacade],
        device: str,
        num_envs: int = 1,
    ):
        import torch

        self._entities = entities
        self._sensors = sensors
        # mjlab's MotionCommand offsets motion-frame body positions by the
        # per-env origin. Single-env URLab parks the origin at world zero;
        # tensor lives on the policy device so MotionCommand's `+ env_origins`
        # broadcast doesn't trigger a host<->device copy each tick.
        self.env_origins = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    def __getitem__(self, key: str) -> Any:
        if key in self._entities:
            return self._entities[key]
        if key in self._sensors:
            return self._sensors[key]
        raise KeyError(
            f"scene[{key!r}] not found "
            f"(entities={sorted(self._entities)}, sensors={sorted(self._sensors)})"
        )

    def __contains__(self, key: str) -> bool:
        return key in self._entities or key in self._sensors


# ---------------------------------------------------------------------------
# Manager shims -- enough surface for mjlab's pure obs functions to read,
# without instantiating real `ObservationManager` / `ActionManager` /
# `CommandManager` (those mutate sim state on reset / probe shapes at init,
# which doesn't work over a remote-stepping bridge).
# ---------------------------------------------------------------------------


class _ActionShim:
    """Stand-in for `ActionManager`. The only obs term that reads from this
    is `last_action`, which fetches `env.action_manager.action`. We keep a
    single `(1, num_actions)` tensor and overwrite it after each policy
    forward. `total_action_dim` and `get_term(...)` are exposed for
    `RslRlVecEnvWrapper.__init__` and any obs term that does
    `.get_term(name).raw_action`."""

    def __init__(self, num_actions: int, device: str):
        import torch

        self._num_actions = num_actions
        self.action = torch.zeros((1, num_actions), dtype=torch.float32, device=device)

    @property
    def total_action_dim(self) -> int:
        return self._num_actions

    @property
    def raw_action(self) -> "torch.Tensor":
        return self.action

    def get_term(self, name: str) -> "_ActionShim":
        return self


class _CommandShim:
    """Stand-in for `CommandManager`. mjlab's tracking obs terms call
    `env.command_manager.get_term("motion")` (returns the `MotionCommand`)
    or `env.command_manager.get_command("motion")` (returns its `command`
    tensor). Both routes resolve through the `_MotionContext`."""

    def __init__(self, terms: Dict[str, Any]):
        self._terms = terms

    def get_term(self, name: str) -> Any:
        return self._terms[name]

    def get_command(self, name: str) -> "torch.Tensor":
        return self._terms[name].command


class _MotionContext:
    """Minimal stand-in for `mjlab.tasks.tracking.mdp.commands.MotionCommand`.

    Owns the loaded motion file, advances `time_steps` per `advance()` call,
    and exposes the read-only properties tracking obs terms consume:

      * `command` -> joint_pos + joint_vel concatenation
      * `joint_pos`, `joint_vel`
      * `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
      * `anchor_pos_w`, `anchor_quat_w`, `anchor_lin_vel_w`, `anchor_ang_vel_w`

    Mirrors `MotionCommand` semantics (anchor-frame offsets via
    `scene.env_origins`, body indexes resolved against the policy's body
    name list) but skips the metric / write-state-to-sim side effects --
    inference doesn't need them.
    """

    def __init__(
        self,
        cfg,
        mjlab_body_names: Sequence[str],
        scene_facade: "_SceneFacade",
        entity_facade: "_EntityFacade",
        device: str,
    ):
        import torch
        from mjlab.tasks.tracking.mdp.commands import MotionLoader  # type: ignore
        from urlab_client._name_match import resolve_matching_names  # type: ignore

        self.cfg = cfg
        self.device = device
        self.num_envs = 1
        self._scene = scene_facade
        self._entity = entity_facade

        cfg_body_names = list(cfg.body_names)
        # body_indexes for the motion file: position of each tracked body
        # name in the entity's full body_names list (matches MotionLoader's
        # expectation -- it slices motion_data[:, body_indexes]).
        ids, _ = resolve_matching_names(
            cfg_body_names, list(mjlab_body_names), preserve_order=True
        )
        body_indexes = torch.tensor(ids, dtype=torch.long, device=device)
        self.body_indexes = body_indexes  # mirror MotionCommand for parity
        # robot-side anchor body index: position of the anchor body in the
        # entity's full body_names list, used by `robot_anchor_*_w`
        # properties below to slice into `entity.data.body_link_*_w`.
        self.robot_anchor_body_index = list(mjlab_body_names).index(cfg.anchor_body_name)
        # motion-side anchor body index: position of the anchor inside the
        # tracked-body sublist, for `anchor_*_w` (motion-frame) reads.
        self._motion_anchor_body_index = cfg_body_names.index(cfg.anchor_body_name)
        self.motion = MotionLoader(cfg.motion_file, body_indexes, device=device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)

    def advance(self) -> None:
        """Advance one policy step. Clamps at the end of the motion (no
        looping) -- callers can detect end-of-motion via `time_steps`."""
        self.time_steps = (self.time_steps + 1).clamp(max=self.motion.time_step_total - 1)

    def frame0_joint_pos(self) -> "torch.Tensor":
        """Motion joint positions at frame 0 (shape `(num_joints,)`).
        Used by the runner to teleport URLab to the motion's start pose
        before the first policy step."""
        return self.motion.joint_pos[0]

    def frame0_anchor_pos_w(self) -> "torch.Tensor":
        """Motion anchor body world position at frame 0 (shape `(3,)`)."""
        return self.motion.body_pos_w[0, self._motion_anchor_body_index]

    @property
    def joint_pos(self) -> "torch.Tensor":
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> "torch.Tensor":
        return self.motion.joint_vel[self.time_steps]

    @property
    def command(self) -> "torch.Tensor":
        import torch

        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def body_pos_w(self) -> "torch.Tensor":
        return self.motion.body_pos_w[self.time_steps] + self._scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> "torch.Tensor":
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> "torch.Tensor":
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> "torch.Tensor":
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> "torch.Tensor":
        return (
            self.motion.body_pos_w[self.time_steps, self._motion_anchor_body_index]
            + self._scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> "torch.Tensor":
        return self.motion.body_quat_w[self.time_steps, self._motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> "torch.Tensor":
        return self.motion.body_lin_vel_w[self.time_steps, self._motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> "torch.Tensor":
        return self.motion.body_ang_vel_w[self.time_steps, self._motion_anchor_body_index]

    # -- robot-side reads (read through the entity facade so they reflect
    #    URLab's current sim state, not the motion file) --------------------

    @property
    def robot_anchor_pos_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_body_pos_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> "torch.Tensor":
        return self._entity.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_joint_pos(self) -> "torch.Tensor":
        return self._entity.data.joint_pos

    @property
    def robot_joint_vel(self) -> "torch.Tensor":
        return self._entity.data.joint_vel


class _TwistContext:
    """Stand-in for `mjlab.tasks.velocity.mdp.UniformVelocityCommand`.

    mjlab's locomotion policies obs the body-frame twist setpoint
    `[vx, vy, yaw_rate]` via `command.command` (read by the
    `generated_commands` obs term). In training that vector is
    randomly resampled every few seconds; for URLab eval we let the
    user drive it interactively via the actor's `UMjTwistController`
    (WASD / gamepad), which UE ships back in each step reply
    (`twist.linear.x`, `twist.linear.y`, `twist.angular.z` →
    `art.twist_linear` / `art.twist_angular`). When no UMjTwistController
    is attached on the UE side, those vectors stay zero and the policy
    sees a "stand still" command.
    """

    def __init__(self, cfg, art: URLabArticulation, device: str):
        import torch

        self.cfg = cfg
        self.device = device
        self.num_envs = 1
        self._art = art
        # Pre-allocate the (1, 3) buffer; refilled each `command` read.
        self._buf = torch.zeros((1, 3), dtype=torch.float32, device=device)

    @property
    def command(self) -> "torch.Tensor":
        import torch

        lin = self._art.twist_linear  # numpy (3,)
        ang = self._art.twist_angular  # numpy (3,)
        self._buf[0, 0] = float(lin[0])
        self._buf[0, 1] = float(lin[1])
        self._buf[0, 2] = float(ang[2])
        return self._buf


def _register_builtin_command_contexts() -> None:
    """Register the command context factories shipped with this adapter.
    Called lazily from `_EnvFacade.__init__` so the import-time cost of
    `mjlab.tasks.tracking.mdp` etc. only fires when actually building a
    runner."""
    try:
        from mjlab.tasks.tracking.mdp import MotionCommandCfg  # type: ignore

        def _make_motion_ctx(cfg, art, scene, entity, device):
            return _MotionContext(
                cfg, entity.body_names, scene, entity, device
            )
        register_command_context(MotionCommandCfg, _make_motion_ctx)
    except ImportError:
        pass

    try:
        from mjlab.tasks.velocity.mdp.velocity_command import (  # type: ignore
            UniformVelocityCommandCfg,
        )

        def _make_twist_ctx(cfg, art, scene, entity, device):
            return _TwistContext(cfg, art, device)
        register_command_context(UniformVelocityCommandCfg, _make_twist_ctx)
    except ImportError:
        pass


class _EnvFacade:
    """Lightweight env shim. Walks the task's obs config directly and
    dispatches each obs term's `func(env, **params)` against the URLab
    facades. No real mjlab managers are instantiated -- those probe shapes
    at init and mutate sim state on reset, neither of which works over the
    remote-stepping bridge. Manager surfaces (`action_manager`,
    `command_manager`) are minimal shims sufficient for the obs functions
    that read them."""

    def __init__(
        self,
        client: URLabClient,
        art: URLabArticulation,
        env_cfg,
        device: str,
        scene_name: str = "robot",
        skip_command_contexts: bool = False,
        defer_obs_probe: bool = False,
    ):
        import torch

        self.client = client
        self.art = art
        self.cfg = env_cfg
        self.device = device
        self.num_envs = 1
        self.scene_name = scene_name

        # Canonical joint / body order from a real mjlab `Entity` build.
        # This is the order the trained policy expects, regardless of what
        # URLab's discovery walk produced. Joint defaults / action mapping /
        # body indexing all key off these.
        from mjlab.entity.entity import Entity as _MjlabEntity  # type: ignore

        mjlab_entity = _MjlabEntity(env_cfg.scene.entities[scene_name])
        self._mjlab_joint_names = list(mjlab_entity.joint_names)
        self._mjlab_body_names = list(mjlab_entity.body_names)

        # Facades. These are the actual surfaces mjlab obs functions read
        # off `env.scene[name]` / `env.scene[name].data`.
        self.entity = _EntityFacade(art, self._mjlab_joint_names, device)
        sensors: Dict[str, _SensorFacade] = {}
        for short, _sensor in art.sensors.items():
            sensors[f"{scene_name}/{short}"] = _SensorFacade(art, short, device)
        self.scene = _SceneFacade(
            {scene_name: self.entity}, sensors, device=device, num_envs=self.num_envs
        )

        # Per-joint init defaults from the task config.
        self._populate_defaults_from_cfg(env_cfg, scene_name)

        # Decimation / dt.
        self.cfg_decimation = int(env_cfg.decimation)
        self.dt = float(getattr(env_cfg.sim, "mujoco", env_cfg.sim).timestep)
        self.step_dt = self.dt * self.cfg_decimation
        episode_length_s = getattr(env_cfg, "episode_length_s", None)
        if episode_length_s is not None and self.step_dt > 0.0:
            self.max_episode_length = int(float(episode_length_s) / self.step_dt)
        else:
            self.max_episode_length = 1_000_000

        # Push mjlab's expected physics timestep to UE. The imported MJCF
        # may have compiled with a different timestep (UE uses whatever
        # `<option timestep="..."/>` the XML had), and a mismatch causes
        # client.step(n_steps=decimation) to advance the *wrong amount*
        # of sim time -- ten times too fast (or slow) per env step.
        # Motion-tracking policies drift visibly under this; the
        # dashboard sim_time pill looks frozen because each env step
        # increments by ~0.1x the expected dt. Pushed here so every
        # MjlabRunner caller (CLI script, dashboard launcher, notebook)
        # gets the same fix.
        try:
            self.client.runtime.set_sim_options(timestep=self.dt)
            logger.info(
                "MjlabRunner: pushed sim_dt=%.5f decim=%d (env step=%.4fs)",
                self.dt, self.cfg_decimation, self.step_dt,
            )
        except Exception as exc:
            logger.warning(
                "MjlabRunner: set_sim_options(timestep=%.5f) failed: %s "
                "(UE will use its compiled timestep; motion-tracking "
                "policies may drift)", self.dt, exc,
            )

        # Action mapping. `JointPositionActionCfg.actuator_names` is a
        # regex-or-name list; resolve against mjlab's joint_names (the
        # policy's training order), then map each name to a URLab actuator
        # key. The policy's action vector index `i` drives URLab actuator
        # `self._action_urlab_keys[i]`.
        from urlab_client._name_match import resolve_matching_names  # type: ignore

        if not env_cfg.actions:
            raise RuntimeError("env_cfg.actions is empty -- no action term to drive")
        action_cfg = next(iter(env_cfg.actions.values()))
        ids, names = resolve_matching_names(
            list(action_cfg.actuator_names), self._mjlab_joint_names, preserve_order=True,
        )
        self._action_joint_indices = ids  # positions into _mjlab_joint_names
        self._action_urlab_keys: List[str] = []
        for n in names:
            key = art.resolve_actuator(n) or art.resolve_joint(n)
            if key is None:
                raise KeyError(
                    f"action joint {n!r} not found on URLab articulation {art.prefix!r}"
                )
            self._action_urlab_keys.append(key)
        self._num_actions = len(self._action_urlab_keys)
        # Build a (num_actions,) scale tensor. mjlab supports `scale` as
        # either a scalar or a `dict[regex, value]`; pre-resolve into a
        # flat tensor so the per-step decode is one mul + one add.
        from urlab_client._name_match import resolve_matching_names_values  # type: ignore

        cfg_scale = getattr(action_cfg, "scale", 1.0)
        if isinstance(cfg_scale, dict):
            scale_t = torch.ones(self._num_actions, dtype=torch.float32, device=device)
            idx_list, _, val_list = resolve_matching_names_values(cfg_scale, names)
            scale_t[idx_list] = torch.tensor(val_list, dtype=torch.float32, device=device)
            self._action_scale: "torch.Tensor" = scale_t  # type: ignore[assignment]
        else:
            self._action_scale = float(cfg_scale)  # type: ignore[assignment]

        # Combined offset = cfg.offset (rare) + (default_joint_pos when
        # use_default_offset). Pre-resolved to a (num_actions,) tensor.
        offset_t = torch.zeros(self._num_actions, dtype=torch.float32, device=device)
        cfg_offset = getattr(action_cfg, "offset", 0.0)
        if isinstance(cfg_offset, dict):
            idx_list, _, val_list = resolve_matching_names_values(cfg_offset, names)
            offset_t[idx_list] = torch.tensor(val_list, dtype=torch.float32, device=device)
        else:
            offset_t[:] = float(cfg_offset)
        if bool(getattr(action_cfg, "use_default_offset", True)):
            offset_t = offset_t + self.entity.data.default_joint_pos[0, self._action_joint_indices]
        self._action_offset = offset_t

        # Manager shims (`last_action`, `motion_anchor_*`, `generated_commands`
        # all read through these).
        self.action_manager = _ActionShim(self._num_actions, device)
        all_command_terms: Dict[str, Any] = {}
        if not skip_command_contexts:
            _register_builtin_command_contexts()
            commands_cfg = getattr(env_cfg, "commands", None) or {}
            for cname, ccfg in commands_cfg.items():
                ctx = _resolve_command_context(ccfg, art, self.scene, self.entity, device)
                if ctx is not None:
                    all_command_terms[cname] = ctx
                    logger.info("command term %r -> %s", cname, type(ctx).__name__)
                else:
                    logger.warning(
                        "command term %r has unregistered type %s -- "
                        "obs reading it will see zeros.",
                        cname, type(ccfg).__name__,
                    )
        self._motion_terms = {
            k: v for k, v in all_command_terms.items() if isinstance(v, _MotionContext)
        }
        self.command_manager = _CommandShim(all_command_terms)

        # Walk every obs group declared by the task ("actor", "critic", ...
        # rsl_rl needs every group present at construction even when only
        # `actor` runs at inference). Frozen list of
        # (group_name, cat_dim, [(term_name, func, params, scale, clip,
        #                         history_length, flatten_history_dim), ...]).
        # `history_length` and `flatten_history_dim` resolve mjlab's
        # group-overrides-term precedence: when the group sets
        # `history_length is not None`, that wins for every term.
        self._obs_groups: List[
            Tuple[str, int, List[Tuple[str, Any, Dict[str, Any], Any, Any, int, bool]]]
        ] = []
        for group_name, group_cfg in env_cfg.observations.items():
            group_hist = getattr(group_cfg, "history_length", None)
            group_flatten = getattr(group_cfg, "flatten_history_dim", True)
            terms = []
            for term_name, term in group_cfg.terms.items():
                if group_hist is not None:
                    hist_len = int(group_hist)
                    flatten = bool(group_flatten)
                else:
                    hist_len = int(getattr(term, "history_length", 0))
                    flatten = bool(getattr(term, "flatten_history_dim", True))
                terms.append(
                    (term_name, term.func, dict(term.params),
                     term.scale, term.clip, hist_len, flatten)
                )
            cat_dim = getattr(group_cfg, "concatenate_dim", -1)
            self._obs_groups.append((group_name, cat_dim, terms))

        # Per-term rolling history buffer. Lists of past tensors, oldest
        # first. Lazily filled on first compute (the first sample is
        # cloned `history_length` times to avoid a step-0 transient where
        # the policy sees zeros for `history_length-1` of its frames).
        self._obs_history: Dict[str, Dict[str, List["torch.Tensor"]]] = {
            g: {t[0]: [] for t in terms if t[5] > 0}
            for g, _, terms in self._obs_groups
        }

        # Probe obs dim for `RslRlVecEnvWrapper` / runner construction.
        # Skipped (and run later via `finalize_obs_probe()`) when the
        # caller is going to plug in command contexts after construction
        # -- otherwise the probe walks the obs tree before commands are
        # populated and crashes inside `motion_anchor_pos_b` etc.
        self._action_space_dim = self._num_actions
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        if not defer_obs_probe:
            self.finalize_obs_probe()

    def finalize_obs_probe(self) -> None:
        """Run the obs pipeline once to set `num_obs` and emit the
        startup log. Idempotent. Called automatically from `__init__`
        unless `defer_obs_probe=True`, in which case the loader runs it
        after plugging in command contexts."""
        import torch

        with torch.no_grad():
            obs0 = self._compute_observations()
        self.num_obs = int(obs0["actor"].shape[-1])
        group_dims = {g: int(obs0[g].shape[-1]) for g, _, _ in self._obs_groups}
        logger.info(
            "obs pipeline: groups=%s, num_actions=%d",
            group_dims, self._num_actions,
        )


    @property
    def unwrapped(self) -> "_EnvFacade":
        return self

    @property
    def action_space(self):
        # rsl_rl wrapper queries this for shape; gym Box equivalent.
        import gymnasium as gym  # type: ignore

        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self._action_space_dim,),
            dtype=np.float32,
        )

    def _joint_names_from_cfg(self, env_cfg, scene_name: str) -> Optional[List[str]]:
        """Try to discover joint names by building a real mjlab `Entity`
        from the task's robot config. This is the canonical source --
        order matches what mjlab's policy was trained against. Returns
        `None` when the cfg path doesn't have a robot config we can
        instantiate; the adapter falls back to URLab's joint list."""
        try:
            robot_cfg = env_cfg.scene.entities[scene_name]
        except (AttributeError, KeyError, TypeError):
            return None

        try:
            from mjlab.entity.entity import Entity  # type: ignore

            entity = Entity(robot_cfg)
            return list(entity.joint_names)
        except Exception as exc:
            logger.warning(
                "could not build mjlab Entity to discover joint names "
                "(%s); falling back to URLab's joint list: %s",
                type(exc).__name__, exc,
            )
            return None

    def _infer_joint_names(self, env_cfg, art: URLabArticulation) -> List[str]:
        """Fallback joint-name list: every single-DoF joint on the URLab
        articulation in MJB discovery order. Should match mjlab's order
        when both compile from the same MJCF, but we prefer
        `_joint_names_from_cfg` when available."""
        return [name for name, j in art.joints.items() if j.qpos_dim == 1]

    def _populate_defaults_from_cfg(self, env_cfg, scene_name: str) -> None:
        """Set `entity.data.default_joint_pos` from the task's per-joint
        init config. mjlab's `EntityCfg.init_state.joint_pos` is a dict
        keyed by regex (e.g. `".*_hip_pitch_joint": -0.1`); we resolve
        each pattern against our joint names and write the matched value
        into the defaults buffer."""
        import re

        try:
            robot = env_cfg.scene.entities[scene_name]
            init_pos = getattr(robot.init_state, "joint_pos", None)
        except (AttributeError, KeyError, TypeError):
            return
        if not init_pos:
            return

        for i, name in enumerate(self.entity.data._joint_names):
            # Direct hit.
            if name in init_pos:
                self.entity.data.default_joint_pos[0, i] = float(init_pos[name])
                continue
            # Regex hit (mjlab convention: dict keys are regex patterns).
            for pattern, value in init_pos.items():
                try:
                    if re.fullmatch(pattern, name):
                        self.entity.data.default_joint_pos[0, i] = float(value)
                        break
                except re.error:
                    continue

    # ---- gym-style surfaces ------------------------------------------------

    @property
    def render_mode(self) -> Optional[str]:
        return None

    @property
    def observation_space(self):
        import gymnasium as gym  # type: ignore

        # Probe each group's dim once, then build a Dict of Boxes.
        with __import__("torch").no_grad():
            obs = self._compute_observations()
        return gym.spaces.Dict({
            name: gym.spaces.Box(
                low=-float("inf"), high=float("inf"),
                shape=(int(t.shape[-1]),), dtype=np.float32,
            )
            for name, t in obs.items()
        })

    def seed(self, seed: int = -1) -> int:
        return seed

    @property
    def is_finite_horizon(self) -> bool:
        return bool(getattr(self.cfg, "is_finite_horizon", False))

    # ---- obs pipeline ------------------------------------------------------

    def _compute_observations(self) -> Dict[str, "torch.Tensor"]:
        """Walk every obs group's term list, dispatch each
        `func(env, **params)` through the URLab facades, apply per-term
        scale / clip, optionally fold history, concat per group."""
        import torch

        dbg = getattr(self, "_dbg_obs_dump", False)
        out: Dict[str, "torch.Tensor"] = {}
        for group_name, cat_dim, terms in self._obs_groups:
            parts = []
            history = self._obs_history.get(group_name, {})
            for term_name, func, params, scale, clip, hist_len, flatten in terms:
                y = func(self, **params)
                if scale is not None:
                    if isinstance(scale, torch.Tensor):
                        y = y * scale.to(y.device)
                    else:
                        y = y * float(scale)
                if clip is not None:
                    lo, hi = clip
                    y = y.clamp(min=lo, max=hi)
                if hist_len > 0:
                    buf = history[term_name]
                    if not buf:
                        # Lazy prefill: first observation populates every
                        # slot so the policy never sees zero-padding on
                        # step 0. Use detached clones so subsequent
                        # in-place writes on `y` don't propagate.
                        for _ in range(hist_len):
                            buf.append(y.detach().clone())
                    else:
                        buf.append(y.detach().clone())
                        del buf[0]
                    # (1, hist_len, term_dim) ordered oldest -> newest.
                    stacked = torch.stack(buf, dim=1)
                    if flatten:
                        # mjlab's term-major layout:
                        # [t_oldest_dims, t_next_dims, ..., t_newest_dims]
                        stacked = stacked.reshape(stacked.shape[0], -1)
                    y = stacked
                if dbg and group_name == "actor":
                    flat = y.detach().cpu().numpy().reshape(-1)
                    n = flat.size
                    head = flat[: min(8, n)].tolist()
                    logger.info(
                        "  obs term %-22s shape=%-12s |x|=%.4f range=[%.3f,%.3f] head=%s",
                        term_name, tuple(y.shape), float(np.abs(flat).mean()),
                        float(flat.min()), float(flat.max()),
                        [round(v, 3) for v in head],
                    )
                parts.append(y)
            out[group_name] = torch.cat(parts, dim=cat_dim)
        return out

    def reset_obs_history(self) -> None:
        """Clear all obs history buffers. Call after teleporting URLab to
        a new pose (e.g. `align_to_motion`) so the policy doesn't see a
        history that mixes pre- and post-warp state."""
        for group_buffers in self._obs_history.values():
            for buf in group_buffers.values():
                buf.clear()

    def get_observations(self) -> Dict[str, "torch.Tensor"]:
        return self._compute_observations()

    @property
    def observation_manager(self):
        """Tiny shim. `RslRlVecEnvWrapper.get_observations` calls
        `unwrapped.observation_manager.compute()`; we forward to our own
        obs walker, which already returns the per-group dict."""
        outer = self

        class _ObsManagerShim:
            def compute(self_inner):
                return outer._compute_observations()

        return _ObsManagerShim()

    # ---- step / reset (RslRlVecEnvWrapper compatible) ---------------------

    def reset(self, env_ids=None) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Any]]:
        """`RslRlVecEnvWrapper.__init__` calls `env.reset()` once. We don't
        actually reset URLab here -- the top-level `MjlabRunner` owns that
        decision. Just hand back the current obs so the wrapper's init can
        finish."""
        return self._compute_observations(), {}

    def step(self, actions: "torch.Tensor") -> Tuple[Any, Any, Any, Any, Any]:
        """Single inference step: cache the raw policy action (for
        `last_action` obs term), decode to URLab ctrl
        (`scale * action + default_joint_pos`), advance URLab by
        `decimation` mj_steps, advance the motion command, recompute obs."""
        import torch

        # 1. Cache for `last_action` obs term + decode source. Detach so we
        #    don't carry policy gradients into the next compute.
        self.action_manager.action = actions.detach()

        # 2. Decode to URLab actuator ctrl. scale + offset are pre-resolved
        # at init (handles both scalar and per-joint dict configs); offset
        # already absorbs `default_joint_pos` when `use_default_offset=True`.
        scaled = actions[0] * self._action_scale + self._action_offset
        ctrl_np = scaled.detach().cpu().numpy()
        ctrl_map = {
            k: float(v) for k, v in zip(self._action_urlab_keys, ctrl_np)
        }
        self.art.set_ctrl(ctrl_map)

        self._dbg_step_count = getattr(self, "_dbg_step_count", 0) + 1

        # 3. Advance URLab.
        self.client.step(n_steps=self.cfg_decimation, observations="standard")

        # 4. Advance motion command (if any).
        for ctx in self._motion_terms.values():
            ctx.advance()

        # 5. Recompute obs. Inference doesn't need rew/term/trunc, but
        #    `RslRlVecEnvWrapper.step` unpacks 5-tuple -- return zeros.
        obs = self._compute_observations()
        rewards = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, rewards, terminated, truncated, {}

    def align_to_motion(self) -> None:
        """Warp URLab onto the motion's first frame: write motion[0]
        joint positions to URLab's `qpos` and shift `scene.env_origins`
        so the motion's frame-0 anchor coincides with the robot's current
        world position. Without this, a tracking policy trained to start
        at the motion's first pose sees a multi-meter offset on step 0
        and produces apparently-random actions trying to chase the
        target. Mirrors mjlab `MotionCommand._write_reference_state_to_sim`
        for the joint side; the env-origin shift is the eval-time
        equivalent of mjlab's per-env spawn-point alignment."""
        import torch

        if not self._motion_terms:
            return
        motion = next(iter(self._motion_terms.values()))

        # 1. Write motion[0] joint positions into URLab. Per-articulation
        #    qpos resets are name-keyed; use the URLab keys we resolved
        #    at init.
        frame0_jp = motion.frame0_joint_pos().detach().cpu().numpy()
        qpos_map: Dict[str, float] = {}
        for urlab_key, val in zip(self._action_urlab_keys, frame0_jp.tolist()):
            qpos_map[urlab_key] = float(val)
        try:
            self.client.reset(per_articulation_qpos={self.art.prefix: qpos_map})
        except Exception as exc:
            logger.warning("client.reset() to motion frame-0 failed: %s", exc)
            return

        # 2. Shift env_origins so motion[0].anchor lands on the robot's
        #    current world anchor. After this, `motion_anchor_pos_b == 0`
        #    on step 0.
        anchor_world = motion.frame0_anchor_pos_w().detach().cpu().numpy()
        robot_anchor_world = self.entity.data.body_link_pos_w[
            0, motion.robot_anchor_body_index
        ].detach().cpu().numpy()
        delta = robot_anchor_world - anchor_world
        self.scene.env_origins = torch.tensor(
            [delta], dtype=torch.float32, device=self.device
        )
        # Drop any obs history captured against the pre-warp pose so the
        # next `_compute_observations` lazy-fills from the warped state.
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


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


class MjlabRunner:
    """Run a mjlab eval-time policy against a URLab session.

    Usage (velocity-tracking task, no motion file)::

        from urlab_client import URLabClient, StepMode
        from urlab_policy.adapters.mjlab import MjlabRunner

        client = URLabClient("tcp://localhost", step_mode=StepMode.STEPPED)
        client.connect()

        runner = MjlabRunner(
            client,
            task_id="Mjlab-Velocity-Flat-Unitree-G1-Play",
            checkpoint="path/to/model.pt",
            articulation_prefix="g1",
        )
        runner.run(num_steps=2000)
        runner.close()

    Usage (motion-tracking task, with reference motion)::

        runner = MjlabRunner(
            client,
            task_id="Mjlab-Tracking-Flat-Unitree-G1",
            checkpoint="/tmp/mjlab_demo_ckpt.pt",
            motion_file="/tmp/mjlab_demo_motion.npz",
            articulation_prefix="g1",
        )

    The runner wires URLab into the same `obs / policy / action` loop
    `mjlab.scripts.play.run_play` uses internally, so anything mjlab can
    play, this can play -- subject to the adapter's coverage of obs /
    action terms (see this module's docstring for caveats).
    """

    def __init__(
        self,
        client: URLabClient,
        *,
        task_id: str,
        checkpoint: str,
        articulation_prefix: Optional[str] = None,
        device: str = "cpu",
        scene_name: str = "robot",
        motion_file: Optional[str] = None,
    ):
        _require_mjlab()
        import torch  # noqa: F401

        from mjlab.rl import MjlabOnPolicyRunner  # type: ignore
        import mjlab.tasks  # noqa: F401  -- registers tasks
        from mjlab.tasks.registry import (  # type: ignore
            load_env_cfg, load_rl_cfg, load_runner_cls,
        )

        self.client = client
        self.task_id = task_id
        self.device = device
        self.scene_name = scene_name

        # Resolve the articulation. Default: only one in the handshake.
        if articulation_prefix is None:
            keys = list(client.articulations.keys())
            if len(keys) != 1:
                raise RuntimeError(
                    f"multiple articulations available {keys}; pass "
                    "articulation_prefix to disambiguate"
                )
            articulation_prefix = keys[0]
        self.art = client.articulations[articulation_prefix]

        # Load mjlab task + agent config.
        env_cfg = load_env_cfg(task_id, play=True)
        agent_cfg = load_rl_cfg(task_id)
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner

        # Apply num_envs=1 (URLab is single-process).
        env_cfg.scene.num_envs = 1

        # Motion-tracking tasks need a `motion_file` -- either passed in or
        # already set on the task's command config. Patch it onto the
        # `motion` command if present so the same pattern as mjlab's
        # `play.py` works (`--motion-file path.npz`).
        if motion_file is not None:
            try:
                from mjlab.tasks.tracking.mdp import MotionCommandCfg  # type: ignore

                motion_cmd = env_cfg.commands.get("motion")
                if isinstance(motion_cmd, MotionCommandCfg):
                    motion_cmd.motion_file = motion_file
                else:
                    logger.warning(
                        "motion_file=%r passed but task %r has no MotionCommandCfg; "
                        "ignoring", motion_file, task_id,
                    )
            except ImportError:
                logger.warning(
                    "motion_file passed but mjlab.tasks.tracking.mdp not importable"
                )

        # Build TaskSpec + env shim via the new mjlab loader, then load
        # the policy weights via mjlab's RslRl runner (which still
        # wraps our env shim with `RslRlVecEnvWrapper` for shape
        # discovery), and finally hand everything to the agnostic
        # `PolicyRunner`.
        from .loader import load_taskspec
        from ...runner import PolicyRunner

        load_result = load_taskspec(
            env_cfg, self.art, device=device, scene_name=scene_name,
        )
        self.spec = load_result.spec
        self.env = load_result.env_shim

        # Policy load: still uses mjlab's wrapper to discover obs/action
        # space dims from the env shim. Once `.get_inference_policy()`
        # returns, the wrapper isn't used again -- the agnostic runner
        # drives step / obs directly.
        from mjlab.rl import RslRlVecEnvWrapper  # type: ignore
        from dataclasses import asdict

        wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
        self._rsl_runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        self._rsl_runner.load(
            checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
        )
        self.policy = self._rsl_runner.get_inference_policy(device=device)

        self._inner = PolicyRunner(
            client=client,
            art=self.art,
            spec=self.spec,
            env_shim=self.env,
            policy=self.policy,
            device=device,
        )

        # Re-export the env shim's timing constants so callers (the
        # dashboard launcher, anyone reading the spec at runtime) can
        # access `runner.dt` and `runner.cfg_decimation` without
        # piercing the `.env` shim. The shim is the one that actually
        # pushes sim_dt to UE in its __init__; we just mirror the
        # numbers up so they're discoverable on the public surface.
        self.dt = float(getattr(self.env, "dt", 0.0))
        self.cfg_decimation = int(getattr(self.env, "cfg_decimation", 1))
        self.step_dt = float(getattr(self.env, "step_dt", self.dt * self.cfg_decimation))

        # Warp URLab onto the motion's first frame for tracking tasks.
        # No-op for non-motion tasks.
        self._inner.align_to_motion()

        logger.info(
            "MjlabRunner ready: task=%s, articulation=%s, decim=%d, dt=%.4fs",
            task_id, articulation_prefix, env_cfg.decimation,
            self.env.dt,
        )

    def step(self, hold_default: bool = False) -> None:
        """One policy step. `hold_default=True` bypasses the policy and
        decodes `action=zeros`, driving URLab to the mjlab init pose --
        useful as a wire / scale / offset sanity check."""
        self._inner.step(hold_default=hold_default)

    def run(self, num_steps: Optional[int] = None, hold_default: bool = False) -> None:
        self._inner.run(num_steps, hold_default=hold_default)

    def close(self) -> None:
        self._inner.close()


__all__ = ["MjlabRunner"]
