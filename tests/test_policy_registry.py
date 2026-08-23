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

"""Tests for the urlab_policy.registry required_step_mode helpers."""

from __future__ import annotations

import pytest

from urlab_client.enums import StepMode
from urlab_policy.registry import (
    check_step_mode_compatible,
    get_required_step_mode,
)


def test_no_requirement_returns_none():
    assert get_required_step_mode({}) is None
    assert get_required_step_mode({"label": "x"}) is None


def test_single_mode_normalises_to_tuple():
    assert get_required_step_mode(
        {"required_step_mode": "stepped"}
    ) == (StepMode.STEPPED,)
    assert get_required_step_mode(
        {"required_step_mode": StepMode.STATEPUSHED}
    ) == (StepMode.STATEPUSHED,)


def test_tuple_of_modes_preserves_each():
    result = get_required_step_mode(
        {"required_step_mode": ("stepped", StepMode.STATEPUSHED)}
    )
    assert result == (StepMode.STEPPED, StepMode.STATEPUSHED)


def test_check_step_mode_compatible_passes():
    entry = {"required_step_mode": "stepped"}
    check_step_mode_compatible(entry, "stepped")
    check_step_mode_compatible(entry, StepMode.STEPPED)


def test_check_step_mode_compatible_rejects():
    entry = {"required_step_mode": ("stepped", "statepushed")}
    with pytest.raises(ValueError, match="step_mode"):
        check_step_mode_compatible(entry, "freerun")


def test_check_skips_when_no_requirement():
    check_step_mode_compatible({}, "freerun")
    check_step_mode_compatible({}, StepMode.STATEPUSHED)


def test_robot_spec_resolves_from_registry_entry():
    """Every RoboJuDo-bound policy entry references a known RobotSpec."""
    from urlab_policy.adapters.robojudo.joint_specs import ROBOTS, RobotSpec
    from urlab_policy.adapters.robojudo.registry import POLICIES, robot_for

    for name, entry in POLICIES.items():
        assert "robot" in entry, f"entry {name!r} missing 'robot' field"
        spec = robot_for(entry)
        assert isinstance(spec, RobotSpec), f"{name!r}: got {type(spec)}"
        # The resolved spec must be the one named in the entry, not
        # just any RobotSpec — without this the test would pass even
        # if robot_for ignored the lookup and always returned the
        # same default.
        assert spec.name == entry["robot"], (
            f"{name!r}: robot_for returned RobotSpec(name={spec.name!r}); "
            f"entry asked for {entry['robot']!r}"
        )
        assert spec is ROBOTS[entry["robot"]], (
            f"{name!r}: robot_for returned a different object than "
            f"ROBOTS[{entry['robot']!r}]"
        )
        # dofs in the entry should match the spec's joint count when
        # they describe the same surface (some H2H-style policies
        # observe a 21-dof subset of a 29-dof robot).
        if entry.get("ctrl_type") == "twist" and entry["robot"] != "g1_29dof":
            assert spec.num_dofs == entry["dofs"], (
                f"{name!r}: entry dofs={entry['dofs']} != "
                f"robot {entry['robot']!r} num_dofs={spec.num_dofs}"
            )


def test_robot_for_returns_none_when_robot_field_absent():
    from urlab_policy.adapters.robojudo.registry import robot_for

    assert robot_for({}) is None
    assert robot_for({"label": "x"}) is None
    assert robot_for({"robot": "nonexistent"}) is None
