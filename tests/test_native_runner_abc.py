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

"""NativePolicyRunner accepts URLab-side `Policy` ABC implementations.

Proves the RoboJuDo coupling inversion: a tiny mock policy that
implements `urlab_policy.base.Policy` (no torch, no robojudo) drives
through `NativePolicyRunner.step()` end-to-end without any
backend-specific glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from urlab_client import URLabClient
from urlab_policy.base import Policy
from urlab_policy.native_runner import NativePolicyRunner

from . import wire_replies as wr


@dataclass
class _CfgDof:
    joint_names: List[str]
    stiffness: List[float] = field(default_factory=list)
    damping: List[float] = field(default_factory=list)
    default_pos: List[float] = field(default_factory=list)

    @property
    def num_dofs(self) -> int:
        return len(self.joint_names)


class _ConstantPolicy(Policy):
    """Returns the same action every step. No state, no reset semantics."""

    freq = 50.0

    def __init__(self, joint_names: List[str], constant: float = 0.1):
        self.cfg_obs_dof = _CfgDof(joint_names=list(joint_names))
        self.cfg_action_dof = _CfgDof(
            joint_names=list(joint_names),
            stiffness=[100.0] * len(joint_names),
            damping=[5.0] * len(joint_names),
            default_pos=[0.0] * len(joint_names),
        )
        self._action = np.full(len(joint_names), constant, dtype=np.float64)
        self.observed = []
        self.callbacks = 0

    def get_observation(
        self, env_data: Any, ctrl_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.observed.append((np.asarray(env_data.dof_pos).copy(),
                              np.asarray(env_data.dof_vel).copy()))
        return env_data.dof_pos, {}

    def get_action(self, obs: Any) -> np.ndarray:
        return self._action.copy()

    def post_step_callback(self) -> None:
        self.callbacks += 1


def _make_client(port: int) -> URLabClient:
    return URLabClient(
        "tcp://127.0.0.1",
        step_mode="direct",
        step_port=port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )


def test_native_runner_accepts_urlab_policy_abc(
    mock_step_server, base_handshake, mujoco_mod
):
    """A `urlab_policy.base.Policy` subclass drives through
    NativePolicyRunner.step() — no RoboJuDo, no torch, no duck-typing
    surprises."""
    client = _make_client(mock_step_server.port)
    try:
        mock_step_server.replies.append(base_handshake)
        client.connect()
        art = client.articulations["vx300s"]

        # The synthetic MJB has 'waist' and 'shoulder' joints.
        joint_names = ["waist", "shoulder"]
        policy = _ConstantPolicy(joint_names, constant=0.05)

        runner = NativePolicyRunner(
            client=client,
            art=art,
            policy=policy,
            decimation=1,
            push_gains=False,
        )

        # Queue a step reply for the URLabClient.step() inside runner.step().
        mock_step_server.replies.append(
            wr.step_ok(
                time=0.001, step=1,
                arts={
                    "vx300s": wr.art_block(
                        qpos=[0.0, 0.0],
                        qvel=[0.0, 0.0],
                        ctrl=[0.05, 0.05],
                        act=[0.0, 0.0],
                    ),
                },
            )
        )
        runner.step()

        # Policy was driven once.
        assert len(policy.observed) == 1
        assert policy.callbacks == 1

        # Last applied ctrl on URLab side reflects the constant action
        # we wrote (action=0.05, default=0.0 -> ctrl=0.05).
        assert np.allclose(art.last_applied_ctrl[:2], 0.05)
    finally:
        client.close()


def test_policy_abc_subclass_must_implement_methods():
    """`Policy` is an ABC — instantiating without get_observation /
    get_action raises."""

    class Incomplete(Policy):
        cfg_obs_dof = _CfgDof([])
        cfg_action_dof = _CfgDof([])

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
