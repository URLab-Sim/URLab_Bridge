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

"""PIE lifecycle tests against a real UE editor.

Covers: sim.status / sim.start / sim.stop typed-result shapes.

These tests are ordered to run **last** in the session (see
``conftest.pytest_collection_modifyitems``) because ``test_sim_stop``
leaves PIE off and UE's PIE re-compile path is fragile across many
stop/start cycles — putting these tests at the end means a single
recompile fails contained at most one test instead of cascading
through the rest of the suite.
"""

from __future__ import annotations

import time

from urlab_client.results import PIEStartResult, PIEState, PIEStatus


def test_sim_status_returns_typed(scene_loaded_client):
    """sim.status returns a typed PIEStatus regardless of PIE state."""
    status = scene_loaded_client.sim.status()
    assert isinstance(status, PIEStatus)
    assert isinstance(status.state, PIEState)
    if status.sim_time is not None:
        assert isinstance(status.sim_time, float)


def test_sim_start_returns_typed(scene_loaded_client):
    """sim.start returns a typed PIEStartResult. Calling sim.start
    on a running PIE is a no-op that returns state=READY without
    triggering a recompile — that's exactly what we want here, since
    forcing a stop+start cycle was tripping a UE compile-wedge bug
    intermittently."""
    result = scene_loaded_client.sim.start(raise_on_failure=False, timeout_s=30.0)
    assert isinstance(result, PIEStartResult)
    assert isinstance(result.state, PIEState)
    assert result.is_ready, (
        f"PIE start returned unexpected state for the session-bootstrapped "
        f"scene: state={result.state} "
        f"compile_error={result.compile_error[:200] if result.compile_error else ''}"
    )


def test_sim_stop_returns_none(scene_loaded_client):
    """sim.stop returns None. After this, PIE is off — by virtue of
    the collection ordering hook this is the last test in the session,
    so the session-fixture's teardown is what handles cleanup."""
    if not scene_loaded_client.manager_present:
        # Defensive: prior test left PIE off.
        scene_loaded_client.sim.start(raise_on_failure=False, timeout_s=30.0)
        time.sleep(0.5)
        scene_loaded_client.discover()
    result = scene_loaded_client.sim.stop()
    assert result is None
