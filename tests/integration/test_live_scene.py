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

"""Scene-authoring lifecycle against a real UE editor.

Covers: create_level, import_xml (returns URLabBlueprint),
spawn_actor (returns URLabSpawnHandle with was_existing), apply_scene
idempotency, BP-class mismatch error, set_actor_transform,
destroy_actor, list_actors / list_blueprints typed returns.

Each test creates its own scratch level under /Game/Levels/urlab_test_*
so they don't pollute the user's working scene. Defaults to the
golden scene MJCF that ships with the test fixtures; override with
``URLAB_LIVE_XML`` to point at a different MJCF.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from urlab_client import (
    URLabAsset,
    URLabBlueprint,
    URLabRPCError,
    URLabSpawnHandle,
)
from urlab_client.results import ActorInfo, BlueprintInfo

from .conftest import GOLDEN_SCENE_PATH

XML_PATH = os.environ.get("URLAB_LIVE_XML") or str(GOLDEN_SCENE_PATH)
# Second MJCF with a distinct file stem so the importer derives a
# different BP class path. Used by class-mismatch coverage.
GOLDEN_SCENE_ALT_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "golden_scene_alt.xml"
)


def _scratch_level_name(suffix: str) -> str:
    import uuid
    return f"urlab_test_{suffix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def scratch_level(fresh_live_client, _live_session):
    """A fresh empty scratch level. Auto-named per test. Returns the
    level's name.

    UE refuses scene-authoring ops while PIE is active (the level
    subsystem holds a lock on the PIE world), so this fixture stops
    PIE on entry. PIE is left off on teardown -- the next pie_client
    test's _ensure_pie_for_pie_client restarts it. Teardown does
    reload the golden level so the PIE restart finds the scene's
    AAMjManager; without that, sim.start would launch into the empty
    scratch level and the manager-poll loop in HandleBeginPie would
    time out.

    No destroy_asset on teardown -- UE's ObjectTools::ForceDeleteObjects
    path has an intermittent null deref under repeated invocations.
    Scratch levels accumulate under /Game/Levels until the user
    clears them; they don't affect the rest of the suite.
    """
    import time as _t

    if fresh_live_client.manager_present:
        fresh_live_client.sim.stop()
        _t.sleep(0.5)
        fresh_live_client.connect()
    name = _scratch_level_name("scene")
    fresh_live_client.scene.create_level(name, force_overwrite=True)
    try:
        yield name
    finally:
        if _live_session is not None:
            try:
                fresh_live_client.scene.load_level(
                    _live_session.level_object_path
                )
            except Exception:
                pass


def test_import_xml_returns_blueprint(fresh_live_client):
    """import_xml(...) returns URLabBlueprint, not a dict."""
    if not XML_PATH:
        pytest.skip("URLAB_LIVE_XML not set; can't exercise import_xml")
    bp = fresh_live_client.scene.import_xml(XML_PATH)
    assert isinstance(bp, URLabBlueprint)
    assert bp.class_path
    assert bp.short_name


def test_spawn_actor_round_trip(fresh_live_client, scratch_level):
    """spawn_actor accepts URLabBlueprint, returns URLabSpawnHandle
    with was_existing=False on first spawn."""
    if not XML_PATH:
        pytest.skip("URLAB_LIVE_XML not set")
    fresh_live_client.scene.load_level(scratch_level)
    bp = fresh_live_client.scene.import_xml(XML_PATH)
    handle = fresh_live_client.scene.spawn_actor(
        blueprint=bp, actor_id="alpha",
        location=(0.0, 0.0, 0.5),
    )
    assert isinstance(handle, URLabSpawnHandle)
    assert handle.actor_id == "alpha"
    assert handle.actor_name
    assert handle.was_existing is False


def test_apply_scene_idempotent(fresh_live_client, scratch_level):
    """Calling apply_scene twice with the same actor_id list returns
    was_existing=True on the second pass and does not duplicate
    actors.

    Must use ``save=True``: apply_scene calls ``load_level`` first,
    which reloads the level package from disk and wipes any in-memory
    actors. Without saving, the second call sees a fresh empty level
    and creates a brand-new actor."""
    if not XML_PATH:
        pytest.skip("URLAB_LIVE_XML not set")
    assets = [URLabAsset(actor_id="bravo", xml=XML_PATH, location=(0, 0, 0.5))]
    out1 = fresh_live_client.scene.apply_scene(scratch_level, assets, save=True)
    out2 = fresh_live_client.scene.apply_scene(scratch_level, assets, save=True)
    assert out1["bravo"].was_existing is False
    assert out2["bravo"].was_existing is True
    # Same actor (by name) — idempotent path returned the existing one.
    assert out1["bravo"].actor_name == out2["bravo"].actor_name

    # Outliner confirms exactly one actor with this actor_id.
    actors = fresh_live_client.scene.client.outliner.list_actors() if False else \
        fresh_live_client.outliner.list_actors()
    bravo_count = sum(1 for a in actors if a.actor_id == "bravo")
    assert bravo_count == 1


def test_force_reimport_round_trip(fresh_live_client, scratch_level):
    """``force_reimport=True`` destroys the existing BP and re-imports
    cleanly. Returns a BP at the same class path (the importer derives
    the BP name from the file stem). Without the explicit destroy step
    in ImportXmlSync, the second import's FactoryCreateFile asserts on
    a duplicate UBlueprint and crashes the editor."""
    if not XML_PATH:
        pytest.skip("URLAB_LIVE_XML not set")
    fresh_live_client.scene.load_level(scratch_level)
    bp1 = fresh_live_client.scene.import_xml(XML_PATH)
    bp2 = fresh_live_client.scene.import_xml(XML_PATH, force_reimport=True)
    assert bp1.class_path == bp2.class_path
    # The freshly re-imported BP should still spawn cleanly.
    handle = fresh_live_client.scene.spawn_actor(
        blueprint=bp2, actor_id="reimport_target", location=(0, 0, 0.5),
    )
    assert handle.was_existing is False


def test_spawn_actor_class_mismatch_rejected(fresh_live_client, scratch_level):
    """Same actor_id with a different BP class is a hard error. Uses two
    distinct MJCF files so the importer produces two distinct BP class
    paths (the BP name derives from the file stem, so force_reimport on
    a single file would just give back the same class path)."""
    if not XML_PATH:
        pytest.skip("URLAB_LIVE_XML not set")
    if not GOLDEN_SCENE_ALT_PATH.is_file():
        pytest.skip(f"alt MJCF fixture missing at {GOLDEN_SCENE_ALT_PATH}")
    fresh_live_client.scene.load_level(scratch_level)
    bp1 = fresh_live_client.scene.import_xml(XML_PATH)
    bp2 = fresh_live_client.scene.import_xml(str(GOLDEN_SCENE_ALT_PATH))
    assert bp1.class_path != bp2.class_path, (
        "alt MJCF should produce a distinct BP class path"
    )
    fresh_live_client.scene.spawn_actor(
        blueprint=bp1, actor_id="charlie", location=(0, 0, 0.5),
    )
    with pytest.raises(URLabRPCError) as exc_info:
        fresh_live_client.scene.spawn_actor(
            blueprint=bp2, actor_id="charlie", location=(0, 0, 0.5),
        )
    assert exc_info.value.code == "spawn_failed"


def test_destroy_actor_removes_from_world(fresh_live_client, scratch_level):
    """destroy_actor is None-returning; actor disappears from list_actors."""
    if not XML_PATH:
        pytest.skip("URLAB_LIVE_XML not set")
    fresh_live_client.scene.load_level(scratch_level)
    bp = fresh_live_client.scene.import_xml(XML_PATH)
    fresh_live_client.scene.spawn_actor(blueprint=bp, actor_id="delta")

    result = fresh_live_client.scene.remove_actor("delta")
    assert result is None

    actors = fresh_live_client.outliner.list_actors()
    assert all(a.actor_id != "delta" for a in actors)


def test_set_actor_transform_returns_none(fresh_live_client, scratch_level):
    if not XML_PATH:
        pytest.skip("URLAB_LIVE_XML not set")
    fresh_live_client.scene.load_level(scratch_level)
    bp = fresh_live_client.scene.import_xml(XML_PATH)
    fresh_live_client.scene.spawn_actor(blueprint=bp, actor_id="echo", location=(0, 0, 0.5))
    result = fresh_live_client.scene.set_actor_transform("echo", location=(1, 0, 0.5))
    assert result is None


def test_list_actors_returns_typed(pie_client):
    """list_actors() returns a list of ActorInfo objects with attribute
    access (not dicts). Uses pie_client so the golden actor is in the
    world; otherwise an empty editor would force a skip."""
    actors = pie_client.outliner.list_actors()
    assert isinstance(actors, list)
    assert actors, "pie_client should expose at least the spawned golden actor"
    for a in actors:
        assert isinstance(a, ActorInfo)
        assert isinstance(a.name, str)
        assert isinstance(a.is_articulation, bool)


def test_list_blueprints_returns_typed(pie_client):
    """list_blueprints() returns BlueprintInfo objects. pie_client has
    imported the golden scene, so the registry contains at least one
    URLab blueprint."""
    bps = pie_client.outliner.list_blueprints()
    assert isinstance(bps, list)
    assert bps, "pie_client should expose the golden-scene blueprint"
    for bp in bps:
        assert isinstance(bp, BlueprintInfo)
        assert isinstance(bp.class_path, str)
        assert isinstance(bp.short_name, str)
