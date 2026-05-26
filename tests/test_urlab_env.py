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

"""Tests for URLabEnv (gymnasium adapter on top of URLabClient).

Uses the shared `mock_step_server` fixture for end-to-end wire round-trips.
gymnasium is required; the test module skips cleanly if it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from urlab_client import URLabClient  # noqa: E402
from urlab_client.enums import SpaceMode  # noqa: E402
from urlab_policy.adapters.robojudo.env import URLabEnv  # noqa: E402


def _make_discovered_client(port: int, mock_server, base_handshake) -> URLabClient:
    mock_server.replies.append(base_handshake)
    client = URLabClient(
        "tcp://127.0.0.1",
        step_mode="direct",
        step_port=port,
        rcv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )
    client.discover()
    return client


def _step_reply(time: float, step: int):
    return {
        "op": "step_ok",
        "time": time,
        "step": step,
        "per_articulation": {
            "vx300s": {
                "qpos": [0.1, 0.2],
                "qvel": [0.01, 0.02],
                "ctrl": [0.5, 0.0],
                "act": [],
                "sensors": {
                    "waist_pos": [0.1],
                    "tip_pos": [0.1, 0.0, 0.05],
                },
            },
            "go2": {
                "qpos": [0.0],
                "qvel": [0.0],
                "ctrl": [0.0],
                "act": [],
                "sensors": {},
            },
        },
    }


def test_flat_env_action_and_obs_spaces(mock_step_server, base_handshake):
    client = _make_discovered_client(mock_step_server.port, mock_step_server, base_handshake)
    try:
        env = URLabEnv(client, space_mode="flat")
        # 2 vx300s actuators + 1 go2 actuator = 3
        assert env.action_space.shape == (3,)
        # 2 vx300s qpos + 2 vx300s qvel + 1 sensor (waist_pos) + 3 sensor (tip_pos)
        # + 1 go2 qpos + 1 go2 qvel = 10
        assert env.observation_space.shape == (10,)
    finally:
        client.close()


def test_dict_env_spaces(mock_step_server, base_handshake):
    client = _make_discovered_client(mock_step_server.port, mock_step_server, base_handshake)
    try:
        env = URLabEnv(client, space_mode="dict")
        assert "vx300s" in env.action_space.spaces
        assert "go2" in env.action_space.spaces
        assert env.action_space["vx300s"].shape == (2,)
        assert env.action_space["go2"].shape == (1,)
    finally:
        client.close()


def test_step_returns_gym_quintuple(mock_step_server, base_handshake):
    client = _make_discovered_client(mock_step_server.port, mock_step_server, base_handshake)
    try:
        env = URLabEnv(client, space_mode="flat", n_steps=2)
        mock_step_server.replies.append(_step_reply(0.02, 1))
        obs, reward, terminated, truncated, info = env.step(np.array([0.5, -0.2, 0.1]))
        assert obs.shape == (10,)
        assert isinstance(reward, float) and reward == 0.0
        assert terminated is False and truncated is False
        assert info["step_count"] == 1
        assert "client" in info
        # Check the step request shape on the wire
        req = mock_step_server.received[-1]
        assert req["op"] == "step"
        assert req["n_steps"] == 2
        assert "vx300s" in req["per_articulation"]
    finally:
        client.close()


def test_reset_returns_obs_and_info(mock_step_server, base_handshake):
    client = _make_discovered_client(mock_step_server.port, mock_step_server, base_handshake)
    try:
        env = URLabEnv(client, space_mode="flat")
        mock_step_server.replies.append(_step_reply(0.0, 0))
        obs, info = env.reset(seed=42)
        assert obs.shape == (10,)
        assert info["step_count"] == 0
        # Verify seed went out on the wire
        req = mock_step_server.received[-1]
        assert req["op"] == "reset"
        assert req["seed"] == 42
    finally:
        client.close()


def test_max_episode_steps_truncates(mock_step_server, base_handshake):
    client = _make_discovered_client(mock_step_server.port, mock_step_server, base_handshake)
    try:
        env = URLabEnv(client, space_mode="flat", max_episode_steps=2)
        for i in range(3):
            mock_step_server.replies.append(_step_reply(0.01 * i, i + 1))
        action = np.zeros(3, dtype=np.float64)
        _, _, _, trunc1, _ = env.step(action)
        _, _, _, trunc2, _ = env.step(action)
        assert trunc1 is False
        assert trunc2 is True
    finally:
        client.close()


def test_reward_and_termination_callbacks(mock_step_server, base_handshake):
    client = _make_discovered_client(mock_step_server.port, mock_step_server, base_handshake)
    try:
        rewards_seen = []

        def reward_fn(info):
            rewards_seen.append(info["step_count"])
            return float(info["step_count"]) * 0.1

        def termination_fn(info):
            return info["step_count"] >= 2

        env = URLabEnv(
            client,
            space_mode="flat",
            reward_fn=reward_fn,
            termination_fn=termination_fn,
        )
        for _ in range(2):
            mock_step_server.replies.append(_step_reply(0.0, 0))

        action = np.zeros(3, dtype=np.float64)
        _, r1, term1, _, _ = env.step(action)
        _, r2, term2, _, _ = env.step(action)
        assert pytest.approx(r1) == 0.1
        assert pytest.approx(r2) == 0.2
        assert term1 is False
        assert term2 is True
        assert rewards_seen == [1, 2]
    finally:
        client.close()


def test_dict_env_action_routing(mock_step_server, base_handshake):
    client = _make_discovered_client(mock_step_server.port, mock_step_server, base_handshake)
    try:
        env = URLabEnv(client, space_mode="dict")
        mock_step_server.replies.append(_step_reply(0.01, 1))
        action = {
            "vx300s": np.array([0.7, -0.1]),
            "go2": np.array([0.3]),
        }
        env.step(action)
        # Inspect the step request the server received; ctrl values from the
        # action should appear in per-articulation ctrl in actuator discovery
        # order. Post-step actuator.value reflects the *reply* (server-echoed
        # post-step state), not the inbound action — so we verify intent by
        # looking at the wire instead.
        req = mock_step_server.received[-1]
        per_art = req["per_articulation"]
        assert pytest.approx(per_art["vx300s"]["ctrl"][0]) == 0.7
        assert pytest.approx(per_art["vx300s"]["ctrl"][1]) == -0.1
        assert pytest.approx(per_art["go2"]["ctrl"][0]) == 0.3
    finally:
        client.close()
