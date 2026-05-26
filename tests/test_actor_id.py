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

"""ActorId / articulations_by_id.

The bridge owns a stable string handle per articulation that the UE
side echoes back in the handshake. Empty when unset; bridge falls back
to ``prefix`` / ``actor_name`` resolution.
"""

from __future__ import annotations

import copy

import pytest

from urlab_client import URLabArticulation, URLabClient


@pytest.fixture
def handshake_no_ids(base_handshake):
    return copy.deepcopy(base_handshake)


@pytest.fixture
def handshake_with_ids(base_handshake):
    h = copy.deepcopy(base_handshake)
    # vx300s gets an actor id; go2 stays unset.
    for art in h["articulations"]:
        if art["prefix"] == "vx300s":
            art["actor_id"] = "robot_a"
    return h


def test_actor_id_defaults_empty(handshake_no_ids):
    c = URLabClient(step_mode="direct")
    c._apply_handshake(handshake_no_ids)
    for art in c.articulations.values():
        assert art.actor_id == ""


def test_articulations_by_id_empty_without_ids(handshake_no_ids):
    c = URLabClient(step_mode="direct")
    c._apply_handshake(handshake_no_ids)
    assert c.articulations_by_id == {}


def test_actor_id_populates_field_and_index(handshake_with_ids):
    c = URLabClient(step_mode="direct")
    c._apply_handshake(handshake_with_ids)

    vx = c.articulations["vx300s"]
    go2 = c.articulations["go2"]

    assert vx.actor_id == "robot_a"
    assert go2.actor_id == ""

    # vx in the id index, go2 not (empty id is not indexed)
    assert "robot_a" in c.articulations_by_id
    assert c.articulations_by_id["robot_a"] is vx
    assert "go2" not in c.articulations_by_id  # would be the prefix, not an actor id


def test_articulations_by_id_is_alias_not_copy(handshake_with_ids):
    """Looking up by id and by prefix yields the SAME URLabArticulation
    object — both maps point at the same wrappers, no shadow state."""
    c = URLabClient(step_mode="direct")
    c._apply_handshake(handshake_with_ids)
    assert c.articulations_by_id["robot_a"] is c.articulations["vx300s"]


def test_apply_handshake_resets_articulations_by_id(handshake_with_ids):
    """A second discover should rebuild articulations_by_id from scratch
    (drop stale ids, pick up new ones)."""
    c = URLabClient(step_mode="direct")
    c._apply_handshake(handshake_with_ids)
    assert "robot_a" in c.articulations_by_id

    # Reapply with no ids.
    no_ids = copy.deepcopy(handshake_with_ids)
    for art in no_ids["articulations"]:
        art.pop("actor_id", None)
    c._apply_handshake(no_ids)
    assert c.articulations_by_id == {}
