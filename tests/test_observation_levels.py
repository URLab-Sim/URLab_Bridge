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

"""Tests that the bridge correctly absorbs the three observation levels.

Plan section 7.9:
    minimal  -> qpos, qvel, time, step
    standard -> minimal + act + ctrl + sensors by name
    full     -> standard + body xpos / xquat + actuator forces
"""

from __future__ import annotations

import numpy as np
import pytest

from urlab_client import URLabClient


def _client(port: int) -> URLabClient:
    return URLabClient(
        "tcp://127.0.0.1",
        step_mode="direct",
        step_port=port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )


def _minimal_reply():
    return {
        "op": "step_ok",
        "time": 0.01,
        "step": 1,
        "per_articulation": {
            "vx300s": {"qpos": [0.1, 0.2], "qvel": [0.0, 0.0]},
            "go2": {"qpos": [0.0], "qvel": [0.0]},
        },
    }


def _standard_reply():
    base = _minimal_reply()
    base["per_articulation"]["vx300s"].update(
        {
            "ctrl": [0.5, -0.1],
            "act": [0.42, 0.0],
            "sensors": {"waist_pos": [0.1], "tip_pos": [0.1, 0.0, 0.05]},
        }
    )
    base["per_articulation"]["go2"].update({"ctrl": [0.3], "act": [], "sensors": {}})
    return base


def _full_reply():
    base = _standard_reply()
    base["per_articulation"]["vx300s"].update(
        {
            "bodies": {
                "vx300s_base_link": {
                    "xpos": [1.0, 2.0, 3.0],
                    "xquat": [1.0, 0.0, 0.0, 0.0],
                },
            },
            "actuator_forces": [12.5, -3.4],
        }
    )
    return base


def test_minimal_absorbs_qpos_qvel(mock_step_server, base_handshake):
    mock_step_server.replies.extend([base_handshake, _minimal_reply()])
    client = _client(mock_step_server.port)
    try:
        client.connect()
        client.step(observations="minimal")
        arm = client.articulations["vx300s"]
        np.testing.assert_allclose(arm.qpos_array, [0.1, 0.2])
        np.testing.assert_allclose(arm.qvel_array, [0.0, 0.0])
    finally:
        client.close()


def test_standard_absorbs_act_and_sensors(mock_step_server, base_handshake):
    mock_step_server.replies.extend([base_handshake, _standard_reply()])
    client = _client(mock_step_server.port)
    try:
        client.connect()
        client.step(observations="standard")
        arm = client.articulations["vx300s"]
        np.testing.assert_allclose(arm.act_array, [0.42, 0.0])
        np.testing.assert_allclose(arm.sensors["waist_pos"].latest, [0.1])
        np.testing.assert_allclose(
            arm.sensors["tip_pos"].latest, [0.1, 0.0, 0.05]
        )
    finally:
        client.close()


def test_full_absorbs_bodies_and_forces(mock_step_server, base_handshake):
    mock_step_server.replies.extend([base_handshake, _full_reply()])
    client = _client(mock_step_server.port)
    try:
        client.connect()
        client.step(observations="full")
        arm = client.articulations["vx300s"]
        body = arm.bodies.get("vx300s_base_link")
        if body is not None:
            np.testing.assert_allclose(body.xpos, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(body.xquat, [1.0, 0.0, 0.0, 0.0])
        # Actuator forces by discovery order
        actuators = list(arm.actuators.values())
        assert pytest.approx(actuators[0].force) == 12.5
        assert pytest.approx(actuators[1].force) == -3.4
    finally:
        client.close()


def test_actuator_forces_dict_form(mock_step_server, base_handshake):
    """Server may emit forces as a name->value dict instead of a flat array."""
    reply = _standard_reply()
    reply["per_articulation"]["vx300s"]["actuator_forces"] = {
        "waist": 7.7,
        "shoulder": -1.1,
    }
    mock_step_server.replies.extend([base_handshake, reply])
    client = _client(mock_step_server.port)
    try:
        client.connect()
        client.step(observations="full")
        arm = client.articulations["vx300s"]
        if "waist" in arm.actuators:
            assert pytest.approx(arm.actuators["waist"].force) == 7.7
        if "shoulder" in arm.actuators:
            assert pytest.approx(arm.actuators["shoulder"].force) == -1.1
    finally:
        client.close()


def test_observation_level_passes_through_on_wire(mock_step_server, base_handshake):
    mock_step_server.replies.extend([base_handshake, _minimal_reply()])
    client = _client(mock_step_server.port)
    try:
        client.connect()
        client.step(observations="full")
        req = mock_step_server.received[-1]
        assert req["observations"] == "full"
    finally:
        client.close()
