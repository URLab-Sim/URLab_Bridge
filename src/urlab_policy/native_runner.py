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

"""Imperative runner for RoboJuDo-shape `Policy` objects (the ones that
own their own gait clock / motion playback / heading alignment).
Counterpart to declarative-TaskSpec `PolicyRunner`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from urlab_client import URLabArticulation, URLabClient

from .base import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RoboJuDo-compatible env_data namespace
# ---------------------------------------------------------------------------


@dataclass
class EnvData:
    """The fields RoboJuDo / WTW policies read off `env_data`. Standard
    fields are always populated. Per-policy needs (e.g. BeyondMimic's
    `torso_pos` / `torso_quat`) are added dynamically via the runner's
    `body_field_map`, which sets attributes on the namespace at build
    time."""

    dof_pos: np.ndarray             # (num_obs_dofs,)
    dof_vel: np.ndarray             # (num_obs_dofs,)
    base_pos: np.ndarray            # (3,)
    base_quat: np.ndarray           # (4,) xyzw -- RoboJuDo convention
    base_lin_vel: np.ndarray        # (3,) body frame
    base_ang_vel: np.ndarray        # (3,) body frame


def _resolve_joint_indices(
    art: URLabArticulation, joint_names: Sequence[str], kind: str
) -> List[int]:
    """Map each joint name onto a URLab local qpos / qvel offset for
    obs reads, or onto an actuator local index for action writes.

    `kind="qpos"` -> per-joint local qpos offset (single-DoF assumed)
    `kind="qvel"` -> per-joint local qvel offset
    `kind="act"`  -> URLab actuator local index
    """
    indices: List[int] = []
    for name in joint_names:
        if kind in ("qpos", "qvel"):
            key = art.resolve_joint(name) or art.resolve_actuator(name)
            if key is None:
                raise KeyError(
                    f"native runner: joint name {name!r} not found on "
                    f"articulation {art.prefix!r} (have "
                    f"{sorted(art.joints)})"
                )
            j = art.joints[key]
            if j.qpos_dim != 1:
                raise ValueError(
                    f"joint {key!r} has qpos_dim={j.qpos_dim}; only single-DoF "
                    "joints are supported in native runner obs"
                )
            indices.append(j.qpos_local_offset if kind == "qpos" else j.qvel_local_offset)
        elif kind == "act":
            key = art.resolve_actuator(name) or art.resolve_joint(name)
            if key is None:
                raise KeyError(
                    f"native runner: actuator name {name!r} not found on "
                    f"articulation {art.prefix!r} (have "
                    f"{sorted(art.actuators)})"
                )
            indices.append(art._actuator_local[key])
        else:
            raise ValueError(f"unknown kind: {kind}")
    return indices


# ---------------------------------------------------------------------------
# Twist controller -> joystick-shaped ctrl_data
# ---------------------------------------------------------------------------


def _build_ctrl_data(art: URLabArticulation) -> Dict[str, Any]:
    """RoboJuDo policies expect `ctrl_data["JoystickCtrl"]["axes"]`
    with `LeftX` / `LeftY` / `RightX`. We synthesize those from
    URLab's twist input (the `UMjTwistController` on the actor maps
    WASD / gamepad to `art.twist_linear` / `art.twist_angular`).

    Some policies (e.g. WTW) pre-scale by `max_cmd` then convert via
    `command_remap`. URLab's twist values are already in m/s and rad/s,
    but the policies expect normalized [-1, 1] joystick axes. We
    inverse-normalize using the controller's max settings -- but those
    aren't shipped on the wire. Pragmatic shortcut: send already-scaled
    values via the JoystickCtrl 'axes' map, and rely on the policy's
    `command_remap` being the identity-ish for normalized input.

    For policies that don't go through `command_remap`, this format is
    fine: they read the axes as-is.
    """
    lin = art.twist_linear   # (3,)
    ang = art.twist_angular  # (3,)
    return {
        "JoystickCtrl": {
            "axes": {
                # WTW's _get_commands maps LeftY -> vx, LeftX -> vy, RightX -> yaw.
                # URLab's twist already gives (Vx, Vy, YawRate); pass through.
                "LeftY": float(lin[0]),
                "LeftX": float(lin[1]),
                "RightX": float(ang[2]),
            },
        },
    }


# ---------------------------------------------------------------------------
# NativePolicyRunner
# ---------------------------------------------------------------------------


class NativePolicyRunner:
    """Drive a RoboJuDo-shape `Policy` against URLab.

    Args:
        client: a discovered `URLabClient`.
        art: the `URLabArticulation` to control.
        policy: the `Policy` instance. Must expose `cfg_obs_dof`,
            `cfg_action_dof`, `get_observation(env_data, ctrl_data)`,
            `get_action(obs)`, `post_step_callback()`, and (typically)
            `reset()`.
        decimation: how many URLab mj_steps per policy step. Defaults
            to `int(round(physics_rate / policy.freq))` when both are
            inferable; otherwise pass explicitly.
        push_gains: when True, send the policy's `cfg_action_dof.stiffness`
            and `damping` to URLab via `art.push_gains(...)` once at init.
            Lets the policy's intended PD gains override whatever the
            imported MJB has. Default False.
    """

    def __init__(
        self,
        client: URLabClient,
        art: URLabArticulation,
        policy: "Policy | Any",  # ABC for URLab-bundled; Any for duck-typed RoboJuDo policies
        decimation: Optional[int] = None,
        push_gains: bool = False,
        body_field_map: Optional[Dict[str, str]] = None,
    ):
        self.client = client
        self.art = art
        self.policy = policy

        # Optional per-policy body lookups (e.g. BeyondMimic reads
        # `env_data.torso_pos` / `torso_quat`). Map: EnvData attr name
        # -> URLab body short name. Validated up front so the per-step
        # build path is just `art._client.data.xpos[bid]`.
        self._body_field_map: Dict[str, int] = {}
        for attr, body_name in (body_field_map or {}).items():
            if body_name not in art.bodies:
                raise KeyError(
                    f"native runner: body {body_name!r} (for {attr!r}) not "
                    f"found on articulation {art.prefix!r}; have "
                    f"{sorted(art.bodies)}"
                )
            self._body_field_map[attr] = art.bodies[body_name].id

        # When body lookups are wired, force a `mj_forward` on the
        # client's local model so `data.xpos` / `data.xquat` are
        # populated for the very first `build_env_data()` call. Without
        # this, the first observation reads zero positions / identity
        # quats for any body in `body_field_map`, which can confuse
        # standalone policies that compute setpoints from the initial
        # robot pose (BeyondMimic's anchor-frame transform).
        if self._body_field_map:
            try:
                import mujoco
                if client.model is not None and client.data is not None:
                    mujoco.mj_forward(client.model, client.data)
            except Exception as exc:
                logger.warning(
                    "mj_forward failed during body-field init: %s -- "
                    "first-step body positions may be uninitialised", exc,
                )

        # Resolve joint mappings.
        obs_dof = policy.cfg_obs_dof
        action_dof = policy.cfg_action_dof
        self._obs_qpos_idx = _resolve_joint_indices(art, obs_dof.joint_names, "qpos")
        self._obs_qvel_idx = _resolve_joint_indices(art, obs_dof.joint_names, "qvel")
        self._action_act_idx = _resolve_joint_indices(art, action_dof.joint_names, "act")
        # URLab actuator KEY per action index (for set_ctrl by name).
        local_to_key = {idx: name for name, idx in art._actuator_local.items()}
        self._action_keys = [local_to_key[i] for i in self._action_act_idx]
        # Action default offset (added to scaled action to form ctrl).
        self._action_default = np.asarray(
            action_dof.default_pos if action_dof.default_pos is not None
            else np.zeros(action_dof.num_dofs),
            dtype=np.float64,
        )

        # Optional gain push.
        if push_gains and (action_dof.stiffness is not None or action_dof.damping is not None):
            try:
                art.push_gains(
                    action_dof.joint_names,
                    stiffness=action_dof.stiffness or [0.0] * action_dof.num_dofs,
                    damping=action_dof.damping,
                )
                logger.info("pushed PD gains to URLab via the controller surface")
            except Exception as exc:
                logger.warning("push_gains failed: %s -- continuing with MJB-baked gains", exc)

        # Decimation: prefer explicit; otherwise infer from policy.freq
        # vs URLab's physics dt (read off the local model when present).
        if decimation is not None:
            self.cfg_decimation = int(decimation)
        else:
            policy_dt = 1.0 / float(policy.freq) if getattr(policy, "freq", 0) else 0.02
            urlab_dt = float(client.model.opt.timestep) if client.model is not None else 0.005
            self.cfg_decimation = max(1, int(round(policy_dt / urlab_dt)))
        logger.info(
            "NativePolicyRunner ready: policy=%s decimation=%d obs_dofs=%d action_dofs=%d",
            type(policy).__name__, self.cfg_decimation,
            obs_dof.num_dofs, action_dof.num_dofs,
        )

    # ----- per-step state assembly -----

    def build_env_data(self) -> EnvData:
        art = self.art
        qpos = np.asarray(art.qpos_array, dtype=np.float64)
        qvel = np.asarray(art.qvel_array, dtype=np.float64)
        env_data = EnvData(
            dof_pos=qpos[self._obs_qpos_idx].copy(),
            dof_vel=qvel[self._obs_qvel_idx].copy(),
            base_pos=art.root_pos_w.astype(np.float64),
            base_quat=art.root_quat_xyzw.astype(np.float64),  # xyzw, RoboJuDo convention
            base_lin_vel=art.root_lin_vel_b.astype(np.float64),
            base_ang_vel=art.root_ang_vel_b.astype(np.float64),
        )
        # Per-policy body lookups: write each (attr, body_id) pair as an
        # attribute on the namespace. `_pos` suffixed attrs read xpos,
        # `_quat` suffixed attrs read xquat (mujoco wxyz convention --
        # RoboJuDo policies that read torso_quat etc. expect this order).
        if self._body_field_map and art._client is not None and art._client.data is not None:
            d = art._client.data
            for attr, bid in self._body_field_map.items():
                if attr.endswith("_pos"):
                    setattr(env_data, attr,
                            np.asarray(d.xpos[bid], dtype=np.float64).copy())
                elif attr.endswith("_quat"):
                    setattr(env_data, attr,
                            np.asarray(d.xquat[bid], dtype=np.float64).copy())
                else:
                    # Fallback: provide both pos+quat for unknown suffix.
                    setattr(env_data, attr + "_pos",
                            np.asarray(d.xpos[bid], dtype=np.float64).copy())
                    setattr(env_data, attr + "_quat",
                            np.asarray(d.xquat[bid], dtype=np.float64).copy())
        return env_data

    # ----- step + run -----

    def step(self) -> None:
        env_data = self.build_env_data()
        ctrl_data = _build_ctrl_data(self.art)
        obs, _extras = self.policy.get_observation(env_data, ctrl_data)
        action = self.policy.get_action(obs)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        # Action -> ctrl: decode = action + default. The policy already
        # applies its own internal scale / clip / smoothing in
        # `get_action`, so by the time we see `action` it's the joint
        # delta to add to default pos. (RoboJuDo convention.)
        ctrl = action + self._action_default
        self.art.set_ctrl({k: float(v) for k, v in zip(self._action_keys, ctrl)})
        self.client.step(n_steps=self.cfg_decimation, observations="standard")
        if hasattr(self.policy, "post_step_callback"):
            self.policy.post_step_callback()

    def run(self, num_steps: Optional[int] = None) -> None:
        i = 0
        while num_steps is None or i < num_steps:
            self.step()
            i += 1

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


__all__ = ["NativePolicyRunner", "EnvData"]
