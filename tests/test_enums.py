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

"""Round-trip + identity tests for urlab_client.enums."""

from __future__ import annotations

import json

import pytest

from urlab_client.enums import (
    ActuatorType,
    CameraMode,
    CameraTiming,
    ControlMode,
    ControllerKind,
    ObservationLevel,
    SpaceMode,
    StepMode,
    coerce,
    wire,
)


def test_step_mode_values():
    assert StepMode.LIVE.value == "live"
    assert StepMode.DIRECT.value == "direct"
    assert StepMode.PUPPET.value == "puppet"
    assert StepMode.AUTO.value == "auto"


def test_control_mode_values():
    assert ControlMode.UE_CONTROLLER.value == "ue_controller"
    assert ControlMode.RAW.value == "raw"


def test_actuator_type_values_match_plan():
    # Every actuator element MJCF's schema declares.
    expected = {
        "general", "motor", "position", "velocity", "intvelocity",
        "damper", "cylinder", "muscle", "adhesion", "dcmotor",
    }
    assert {m.value for m in ActuatorType} == expected


def test_camera_mode_and_timing():
    assert {m.value for m in CameraMode} == {"real", "depth", "semantic", "instance"}
    assert {m.value for m in CameraTiming} == {"sync", "latest"}


def test_observation_level_and_space_mode():
    assert {m.value for m in ObservationLevel} == {"minimal", "standard", "full"}
    assert {m.value for m in SpaceMode} == {"flat", "dict"}


def test_controller_kind_has_pd_and_passthrough():
    values = {m.value for m in ControllerKind}
    assert "pd" in values
    assert "passthrough" in values


def test_enum_is_str_mixin_serialises_to_wire_string():
    """The `str` mixin means enums JSON-serialise as their wire string."""
    encoded = json.dumps({"mode": StepMode.PUPPET})
    assert encoded == '{"mode": "puppet"}'


def test_coerce_string_to_enum():
    assert coerce(StepMode, "direct") is StepMode.DIRECT
    assert coerce(ControlMode, "raw") is ControlMode.RAW


def test_coerce_enum_passthrough():
    assert coerce(StepMode, StepMode.DIRECT) is StepMode.DIRECT


def test_coerce_with_default():
    assert coerce(StepMode, None, default=StepMode.AUTO) is StepMode.AUTO


def test_coerce_unknown_string_raises_and_warns(caplog):
    with pytest.raises(ValueError), caplog.at_level("WARNING"):
        coerce(StepMode, "bogus")
    assert any("Unknown StepMode" in r.message for r in caplog.records)


def test_wire_passthrough_and_enum():
    assert wire(StepMode.PUPPET) == "puppet"
    assert wire("puppet") == "puppet"


def test_round_trip_enum_to_wire_to_enum():
    for mode in StepMode:
        assert coerce(StepMode, wire(mode)) is mode
    for t in ActuatorType:
        assert coerce(ActuatorType, wire(t)) is t
