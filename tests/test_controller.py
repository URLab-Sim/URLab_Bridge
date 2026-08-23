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

"""URLabController + URLabPDController unit tests (no socket)."""

from __future__ import annotations

import pytest

from urlab_client import (
    URLabClient,
    URLabController,
    URLabPDController,
)
from urlab_client.enums import ControllerKind


@pytest.fixture
def client(base_handshake):
    c = URLabClient(step_mode="stepped")
    c._apply_handshake(base_handshake)
    return c


def test_pd_controller_is_pd_kind(client):
    vx = client.articulations["vx300s"]
    assert isinstance(vx.controller, URLabPDController)
    assert vx.controller.kind is ControllerKind.PD


def test_pd_controller_live_views(client):
    vx = client.articulations["vx300s"]
    assert vx.controller.kp == {"waist": 300.0, "shoulder": 280.0}
    assert vx.controller.kv == {"waist": 20.0, "shoulder": 18.0}


def test_configure_validates_unknown_key(client):
    vx = client.articulations["vx300s"]
    with pytest.raises(ValueError):
        vx.controller.configure(bogus_param=1.0)


def test_configure_validates_per_joint_shape(client):
    vx = client.articulations["vx300s"]
    with pytest.raises(ValueError):
        # `kp` wants a per-joint mapping
        vx.controller.configure(kp=300.0)


def test_configure_validates_min_bound(client):
    vx = client.articulations["vx300s"]
    with pytest.raises(ValueError):
        vx.controller.configure(kp={"waist": -10.0})


def test_set_gains_partial_patch_keeps_other_joints(client):
    """Missing joints keep their current values (plan: mirror PUB/SUB semantics)."""
    vx = client.articulations["vx300s"]
    # Detach the controller from the client so no RPC is made
    vx.controller._client = None
    result = vx.controller.set_gains(kp={"waist": 500.0})
    assert result["kp"]["waist"] == 500.0
    # shoulder untouched
    assert result["kp"]["shoulder"] == 280.0


def test_set_defaults_patches_scalars_only(client):
    vx = client.articulations["vx300s"]
    vx.controller._client = None
    result = vx.controller.set_defaults(kp=120.0, kv=7.0)
    assert result["default_kp"] == 120.0
    assert result["default_kv"] == 7.0
    # torque_limit default untouched
    assert result["default_torque_limit"] == 200.0


def test_generic_controller_kind_with_unknown_string():
    """Unknown controller kind round-trips as a string, not an enum."""
    ctrl = URLabController(
        "xyz",
        "future_kind",
        {"alpha": 1.0},
        {"alpha": {"type": "scalar", "min": 0.0}},
    )
    assert ctrl.kind == "future_kind"
    ctrl.configure(alpha=2.5)
    assert ctrl.params["alpha"] == 2.5
