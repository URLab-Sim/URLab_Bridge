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

"""URLabEntity (non-articulation) + URLabCameraView tests."""

from __future__ import annotations

import warnings

import numpy as np

from urlab_client import URLabCameraView, URLabClient, URLabEntity
from urlab_client.enums import CameraMode


def test_entity_from_handshake():
    client = URLabClient(step_mode="direct")
    handshake = {
        "op": "hello_ok",
        "session_id": "s",
        "urlab_version": "u",
        "mujoco_version": "3.7.0",
        "mjb": b"",
        "articulations": [],
        "entities": {
            "pallet": {
                "id": 5,
                "has_free_base": True,
                "free_joint": "pallet_free",
                "free_joint_id": 7,
                "qpos_offset": 14,
                "qvel_offset": 12,
            }
        },
    }
    client.mujoco_version_check = False
    client.local_model = False
    client._apply_handshake(handshake)
    entity = client.entities["pallet"]
    assert isinstance(entity, URLabEntity)
    assert entity.has_free_base
    assert entity.qpos_offset == 14
    assert entity.body_id == 5
    assert entity.root_pos_w.shape == (3,)
    assert np.allclose(entity.root_quat_w, [1.0, 0.0, 0.0, 0.0])


def test_articulations_appear_in_entities():
    """Articulations are entities too. `client.entities[prefix]` returns
    the same `URLabArticulation` object as `client.articulations[prefix]`."""
    import mujoco

    mjcf = """
    <mujoco>
      <worldbody>
        <body name="vx300s_waist" pos="0 0 0.1">
          <joint name="vx300s_waist" type="hinge" axis="0 0 1" range="-3 3"/>
          <geom type="capsule" size="0.02 0.05" fromto="0 0 0 0.1 0 0"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="vx300s_waist" joint="vx300s_waist" gear="1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    client = URLabClient(step_mode="direct")
    client.mujoco_version_check = False
    client.model = model
    client.data = data
    client._apply_handshake(
        {
            "op": "hello_ok",
            "session_id": "s",
            "urlab_version": "u",
            "mujoco_version": mujoco.__version__,
            "articulations": [{"prefix": "vx300s", "default_control_mode": "raw"}],
            "entities": {},
        }
    )
    assert "vx300s" in client.entities
    assert client.entities["vx300s"] is client.articulations["vx300s"]


def test_entity_root_pose_absorbs_from_step_reply():
    """xpos/xquat from a step reply land in client.data and surface as
    `entity.root_pos_w` / `root_quat_w`."""
    import mujoco

    mjcf = """
    <mujoco>
      <worldbody>
        <body name="pallet" pos="0 0 0">
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    pallet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pallet")

    client = URLabClient(step_mode="direct")
    client.mujoco_version_check = False
    client.model = model
    client.data = data
    client._apply_handshake(
        {
            "op": "hello_ok",
            "session_id": "s",
            "urlab_version": "u",
            "mujoco_version": mujoco.__version__,
            "articulations": [],
            "entities": {"pallet": {"id": pallet_id, "has_free_base": False}},
        }
    )
    client._absorb_step_reply(
        {
            "time": 0.5,
            "step": 50,
            "arts": {},
            "scene": {
                "pallet": {"xpos": [1.0, 2.0, 3.0], "xquat": [1.0, 0.0, 0.0, 0.0]}
            },
        }
    )
    entity = client.entities["pallet"]
    assert np.allclose(entity.root_pos_w, [1.0, 2.0, 3.0])
    assert np.allclose(entity.root_quat_w, [1.0, 0.0, 0.0, 0.0])


def test_entity_apply_xfrc_in_puppet_warns():
    client = URLabClient(step_mode="puppet")
    client.mujoco_version_check = False
    client.local_model = False
    client._apply_handshake(
        {
            "op": "hello_ok",
            "session_id": "s",
            "urlab_version": "u",
            "mujoco_version": "3.7.0",
            "articulations": [],
            "entities": {"pallet": {"id": 5, "has_free_base": False}},
        }
    )
    entity = client.entities["pallet"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        entity.apply_xfrc(force=[1, 0, 0])
    assert any("puppet mode" in str(x.message) for x in w)


def test_camera_view_parses_mode_and_dtype():
    view = URLabCameraView.from_handshake(
        "depth_cam",
        {"mode": "depth", "resolution": [640, 480], "fovy": 60.0},
        owner="vx300s",
    )
    assert view.mode is CameraMode.DEPTH
    assert view.resolution == (640, 480)
    assert view.owner == "vx300s"
    assert view.dtype == np.dtype(np.float32)
