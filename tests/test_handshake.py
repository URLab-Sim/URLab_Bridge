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

"""Handshake parse + MJB walk tests.

These tests use `_apply_handshake` directly (no socket) so they are
deterministic and cheap. Socket round-trip tests live in
`test_transport.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from urlab_client import (
    URLabClient,
    URLabPDController,
    URLabVersionMismatch,
)
from urlab_client.enums import ActuatorType, CameraMode, ControlMode


@pytest.fixture
def client(base_handshake):
    c = URLabClient(step_mode="direct")
    c._apply_handshake(base_handshake)
    return c


def test_client_normalises_step_mode_string():
    assert URLabClient(step_mode="direct").step_mode.value == "direct"
    assert URLabClient(step_mode="puppet").step_mode.value == "puppet"


def test_handshake_populates_session_id_and_versions(client, base_handshake):
    assert client.session_id == base_handshake["session_id"]
    assert client.urlab_version == base_handshake["urlab_version"]
    assert client.mujoco_version == base_handshake["mujoco_version"]


def test_handshake_loads_mjb_into_local_model(client):
    assert client.model is not None
    assert client.data is not None
    assert client.model.nu >= 3  # waist + shoulder + hip


def test_handshake_builds_articulations_by_prefix(client):
    assert set(client.articulations.keys()) == {"vx300s", "go2"}
    vx = client.articulations["vx300s"]
    go2 = client.articulations["go2"]
    assert vx.control_mode == ControlMode.UE_CONTROLLER
    assert go2.control_mode == ControlMode.RAW


def test_articulation_groups_actuators_by_prefix(client):
    vx = client.articulations["vx300s"]
    go2 = client.articulations["go2"]
    assert set(vx.actuators.keys()) == {"waist", "shoulder"}
    assert set(go2.actuators.keys()) == {"hip"}


def test_articulation_groups_joints_by_prefix(client):
    vx = client.articulations["vx300s"]
    assert set(vx.joints.keys()) == {"waist", "shoulder"}
    assert vx.joints["waist"].qpos_dim == 1
    assert vx.joints["waist"].qvel_dim == 1


def test_articulation_groups_sensors_by_prefix(client):
    vx = client.articulations["vx300s"]
    # synthetic MJB ships vx300s_waist_pos (jointpos, dim 1) and vx300s_tip_pos (framepos, dim 3)
    assert "waist_pos" in vx.sensors
    assert "tip_pos" in vx.sensors
    assert vx.sensors["waist_pos"].dim == 1
    assert vx.sensors["tip_pos"].dim == 3


def test_actuator_types_from_handshake(client):
    vx = client.articulations["vx300s"]
    assert vx.actuators["waist"].type is ActuatorType.POSITION
    assert vx.actuators["shoulder"].type is ActuatorType.MOTOR


def test_actuator_ranges_and_gainprm_walked_from_mjb(client):
    vx = client.articulations["vx300s"]
    waist = vx.actuators["waist"]
    # <position kp=100> compiles to gainprm[0]=100
    assert waist.kp == pytest.approx(100.0)
    assert waist.gear.size > 0


def test_arm_mj_helpers_match_mj_name2id(client, mujoco_mod):
    vx = client.articulations["vx300s"]
    # shortcut + kinded resolution
    assert vx.mj_joint("waist") == mujoco_mod.mj_name2id(
        client.model, mujoco_mod.mjtObj.mjOBJ_JOINT, "vx300s_waist"
    )
    assert vx.mj_actuator("shoulder") == mujoco_mod.mj_name2id(
        client.model, mujoco_mod.mjtObj.mjOBJ_ACTUATOR, "vx300s_shoulder"
    )
    assert vx.mj_body("waist_link") == mujoco_mod.mj_name2id(
        client.model, mujoco_mod.mjtObj.mjOBJ_BODY, "vx300s_waist_link"
    )


def test_arm_mj_accepts_full_prefixed_name(client):
    vx = client.articulations["vx300s"]
    a = vx.mj_joint("waist")
    b = vx.mj_joint("vx300s_waist")
    assert a == b


def test_arm_mj_unknown_kind_raises(client):
    vx = client.articulations["vx300s"]
    with pytest.raises(KeyError):
        vx.mj("bogus", "waist")


def test_arm_mj_unknown_name_raises(client):
    vx = client.articulations["vx300s"]
    with pytest.raises(KeyError):
        vx.mj_joint("no_such_joint")


def test_pd_controller_constructed(client):
    vx = client.articulations["vx300s"]
    assert isinstance(vx.controller, URLabPDController)
    assert vx.controller.kp["waist"] == 300.0


def test_raw_articulation_has_no_controller(client):
    go2 = client.articulations["go2"]
    assert go2.controller is None


def test_camera_view_parsed_from_handshake(client):
    vx = client.articulations["vx300s"]
    cam = vx.cameras["wrist"]
    assert cam.mode is CameraMode.REAL
    assert cam.resolution == (320, 240)
    assert cam.fovy == pytest.approx(45.0)
    assert cam.owner == "vx300s"
    assert cam.latest_frame is None


def test_ctrl_array_default_zero(client):
    vx = client.articulations["vx300s"]
    assert vx.ctrl_array.shape == (2,)
    assert np.allclose(vx.ctrl_array, 0.0)


def test_set_ctrl_bulk(client):
    vx = client.articulations["vx300s"]
    vx.set_ctrl({"waist": 0.5})
    # waist is actuator 0 in discovery order
    assert vx.ctrl_array[vx._actuator_local["waist"]] == pytest.approx(0.5)


def test_set_ctrl_unknown_actuator_raises(client):
    vx = client.articulations["vx300s"]
    with pytest.raises(KeyError):
        vx.set_ctrl({"no_such": 1.0})


def test_apply_xfrc_buffers_pending(client):
    vx = client.articulations["vx300s"]
    vx.apply_xfrc("waist_link", force=[0, 5, 0])
    assert "waist_link" in vx._pending_xfrc
    vx.clear_xfrc()
    assert not vx._pending_xfrc


def test_apply_xfrc_unknown_body_raises(client):
    vx = client.articulations["vx300s"]
    with pytest.raises(KeyError):
        vx.apply_xfrc("no_such_body", force=[1, 0, 0])


def test_actuator_set_control_and_value(client):
    vx = client.articulations["vx300s"]
    vx.actuators["waist"].set_control(1.25)
    assert vx.actuators["waist"].value == pytest.approx(1.25)


def test_version_mismatch_raises(base_handshake):
    bad = dict(base_handshake)
    bad["mujoco_version"] = "99.99.99"
    client = URLabClient(step_mode="direct", mujoco_version_check=True)
    with pytest.raises(URLabVersionMismatch):
        client._apply_handshake(bad)


def test_version_mismatch_bypass(base_handshake):
    bad = dict(base_handshake)
    bad["mujoco_version"] = "99.99.99"
    client = URLabClient(step_mode="direct", mujoco_version_check=False)
    client._apply_handshake(bad)  # must not raise
    assert client.mujoco_version == "99.99.99"
