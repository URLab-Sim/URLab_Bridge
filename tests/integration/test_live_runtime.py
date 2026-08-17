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

"""Runtime-mutator integration tests.

Covers: set_qpos round-trip, set_twist, set_sim_options echo,
set_paused / set_sim_speed echoes, PD controller set_gains
round-trip. Requires a live editor with PIE running and at least one
articulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from urlab_client.results import SimOptions


def _first_articulation(client):
    if not client.articulations:
        pytest.skip("No articulations in the current scene")
    return next(iter(client.articulations.values()))


def test_set_qpos_round_trip(pie_client):
    """runtime.set_qpos(...) writes mjData->qpos and the reply mirrors
    it back into client.data — check the value **before** stepping
    physics. The controller is at its default ctrl=0 and would yank
    qpos back toward zero on the next step, which would mask whether
    set_qpos actually landed."""
    art = _first_articulation(pie_client)
    if art.qpos_array.size < 1:
        pytest.skip("Articulation has zero qpos")
    target = np.zeros_like(art.qpos_array)
    # Pick a small value for the first DoF so we can detect it.
    if art.has_free_base:
        # Free-base: index 7 is the first non-base dof.
        if art.qpos_array.size <= 7:
            pytest.skip("Free-base articulation with no extra DoFs")
        target[:] = art.qpos_array
        target[7] = 0.1
    else:
        target[0] = 0.1
    # set_qpos resolves the target via actor_id by default (or UE actor
    # name with by_name=True). art.prefix is the MJ-side identifier; we
    # need the ActorId or the UE actor name. Prefer actor_id when it's
    # been set (the golden_session fixture sets it to "golden_root").
    if art.actor_id:
        result = pie_client.runtime.set_qpos(art.actor_id, qpos=list(target))
    else:
        result = pie_client.runtime.set_qpos(art.prefix, qpos=list(target), by_name=True)
    assert result is None
    # set_qpos mirrors the new qpos into client.data.qpos (the live
    # mjData buffer). art.qpos_array is a per-articulation local
    # snapshot that's only refreshed by step / reset, so it doesn't
    # see the write until physics advances. Read client.data directly
    # to verify the immediate write landed without controller drift.
    joints = list(art.joints.values())
    first_dof_joint = joints[1] if art.has_free_base else joints[0]
    data_idx = int(first_dof_joint.qpos_offset)
    assert abs(float(pie_client.data.qpos[data_idx]) - 0.1) < 1e-9


def test_set_sim_options_echoes(pie_client):
    """runtime.set_sim_options(timestep=0.002) returns SimOptions with
    timestep echoed."""
    result = pie_client.runtime.set_sim_options(timestep=0.002)
    assert isinstance(result, SimOptions)
    if result.timestep is not None:
        assert abs(result.timestep - 0.002) < 1e-9


def test_set_paused_returns_bool(pie_client):
    pre = pie_client.runtime.set_paused(True)
    assert isinstance(pre, bool)
    pie_client.runtime.set_paused(False)


def test_set_sim_speed_returns_float(pie_client):
    result = pie_client.runtime.set_sim_speed(100.0)
    assert isinstance(result, float)
    assert abs(result - 100.0) < 1e-3


def test_set_twist_returns_none(pie_client):
    art = _first_articulation(pie_client)
    result = pie_client.runtime.set_twist(art.prefix, linear=(0.0, 0.0, 0.0))
    assert result is None


def test_pd_controller_set_gains_round_trip(pie_client):
    """For a PD-controlled articulation, set_gains updates the live
    kp/kv view."""
    art = _first_articulation(pie_client)
    if art.controller is None:
        pytest.skip("Articulation runs in raw mode (no controller)")
    if not hasattr(art.controller, "set_gains"):
        pytest.skip("Controller is not PD-shaped")
    if not art.controller.kp:
        pytest.skip("PD controller has no joints to tune")

    first_joint = next(iter(art.controller.kp))
    art.controller.set_gains(kp={first_joint: 250.0})
    assert abs(float(art.controller.kp[first_joint]) - 250.0) < 1e-3
