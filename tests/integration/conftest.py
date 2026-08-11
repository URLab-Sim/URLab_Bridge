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

"""Shared integration-test fixtures + skip gate.

All tests under tests/integration/ require URLAB_LIVE=1 and a running
URLab editor. Without those, the whole package is skipped at
collection time.

**Tests do not assume any particular editor state**. One session-
scoped fixture authors the golden scene and starts PIE **once** at
session start, leaves PIE running for the whole run, and tears
everything down once at the end. Per-test fixtures open a fresh client
+ call ``client.reset()`` so each test gets a clean physics state
without paying the cost (and risking the UE recompile churn) of
sim.stop / sim.start between tests.

Tests that explicitly exercise the PIE lifecycle (sim.start /
sim.stop / pie_status) manage their own state and are designed to
restore PIE-on by the time they exit so the next pie_client test
finds the scene where it expects it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import NamedTuple

import pytest

LIVE = os.environ.get("URLAB_LIVE") == "1"
HOST = os.environ.get("URLAB_HOST", "tcp://127.0.0.1")
STEP_PORT = int(os.environ.get("URLAB_STEP_PORT", "5559"))

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_SCENE_PATH = _FIXTURES_DIR / "golden_scene.xml"

GOLDEN_LEVEL_NAME = "URLabGoldenTest"
GOLDEN_LEVEL_PACKAGE = f"/Game/Levels/{GOLDEN_LEVEL_NAME}"
GOLDEN_LEVEL_OBJECT = f"{GOLDEN_LEVEL_PACKAGE}.{GOLDEN_LEVEL_NAME}"
GOLDEN_ACTOR_ID = "golden_root"


class SceneBootstrap(NamedTuple):
    level_object_path: str
    blueprint_object_path: str
    original_level: str


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help="Rewrite the golden trajectory reference for test_live_golden "
             "instead of comparing. Run locally against a known-good editor.",
    )


@pytest.fixture
def regenerate_golden(request):
    return request.config.getoption("--regenerate-golden")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if not LIVE:
        skip = pytest.mark.skip(reason="URLAB_LIVE=1 not set; live-UE tests skipped")
        for item in items:
            if "tests/integration" in item.nodeid.replace("\\", "/"):
                item.add_marker(skip)
        return
    # Ordering: push test_live_pie last. Those tests stop / start PIE,
    # and UE's PIE-recompile path is fragile under repeated cycles.
    # Putting them last means a single failed recompile only loses
    # itself, not the entire downstream suite.
    pie_lifecycle = []
    other = []
    for item in items:
        if "test_live_pie" in item.nodeid.replace("\\", "/"):
            pie_lifecycle.append(item)
        else:
            other.append(item)
    items[:] = other + pie_lifecycle


@pytest.fixture
def live_endpoint():
    return {"address": HOST, "port": STEP_PORT}


def _make_client(recv_timeout_ms: int = 120_000):
    """Default recv timeout is generous (120s) because editor ops like
    ``import_xml`` shell out to a Python subprocess + drive the BP factory,
    which can take 10-30s on a cold editor. The server side blocks on the
    game thread without its own deadline, so this is the only operational
    cap. Per-test code can override via ``recv_timeout_ms`` on individual
    RPC calls (e.g. sim.start passes its own).
    """
    from urlab_client import URLabClient

    # The bridge pins mujoco==3.8.1 (mjlab / mujoco-warp compat) while the
    # UE plugin may run a newer mujoco. The client handles the skew itself
    # now: the version check warns instead of raising, and an unloadable
    # MJB falls back to building the model from the compiled XML.
    return URLabClient(
        HOST, step_port=STEP_PORT, recv_timeout_ms=recv_timeout_ms,
    )


@pytest.fixture(scope="session")
def _live_session():
    """Session-scoped bootstrap: author the golden scene once, start PIE
    once, hold the level + blueprint references for the whole test
    session. Yields the SceneBootstrap so per-test fixtures can read
    the asset paths. Teardown at session end stops PIE, destroys the
    level + blueprint, returns the editor to whatever level the user
    was on before the run.
    """
    if not LIVE:
        yield None
        return
    if not GOLDEN_SCENE_PATH.is_file():
        pytest.fail(f"golden scene MJCF missing at {GOLDEN_SCENE_PATH}")

    from urlab_client import StepMode
    from urlab_client.results import PIEState

    client = _make_client()
    state = None
    try:
        client.connect()
        if client.manager_present:
            client.sim.stop()
            time.sleep(0.5)
            client.connect()
        try:
            original_level = client.scene.current_level()
        except Exception:
            original_level = ""
        client.scene.create_level(GOLDEN_LEVEL_NAME, force_overwrite=True)
        client.scene.ensure_manager()
        blueprint = client.scene.import_xml(str(GOLDEN_SCENE_PATH), force_reimport=True)
        client.scene.spawn_actor(blueprint, actor_id=GOLDEN_ACTOR_ID)
        # Persist the populated level to disk. Without this, the manager
        # and spawned actor are in-memory only -- if any test switches
        # the active level (the scratch_level fixture does), the in-memory
        # state is discarded and a subsequent load_level reloads an empty
        # shell from disk. PIE-on-empty-level then has no AAMjManager and
        # the begin_pie poll loop times out.
        client.scene.save_level()
        blueprint_object_path = (
            f"/Game/MuJoCoImports/{GOLDEN_SCENE_PATH.stem}.{GOLDEN_SCENE_PATH.stem}"
        )
        state = SceneBootstrap(
            level_object_path=GOLDEN_LEVEL_OBJECT,
            blueprint_object_path=blueprint_object_path,
            original_level=original_level,
        )
        result = client.sim.start(raise_on_failure=False, timeout_s=60.0)
        if result.state != PIEState.READY:
            pytest.fail(
                f"PIE start failed for golden scene: state={result.state} "
                f"compile_error={result.compile_error[:200] if result.compile_error else ''}"
            )
        time.sleep(0.5)
        client.connect()
        if client.step_mode != StepMode.DIRECT:
            client.runtime.set_mode(StepMode.DIRECT)
        yield state
    finally:
        try:
            if client.manager_present:
                client.sim.stop()
                time.sleep(0.3)
        except Exception:
            pass
        if state is not None:
            for path in (state.level_object_path, state.blueprint_object_path):
                try:
                    client.scene.destroy_asset(path)
                except Exception:
                    pass
            if state.original_level and state.original_level != GOLDEN_LEVEL_PACKAGE:
                try:
                    client.scene.load_level(state.original_level)
                except Exception:
                    pass
        try:
            client.close()
        except Exception:
            pass


def _ensure_pie_for_pie_client(client) -> None:
    """Idempotent: if PIE is off (a prior PIE-lifecycle test stopped
    it), restart it. The session-scoped fixture is the canonical PIE
    starter; this is just a safety net so pie_client always yields a
    PIE-on client without depending on test execution order.
    """
    from urlab_client import StepMode
    from urlab_client.results import PIEState

    if not client.manager_present:
        result = client.sim.start(raise_on_failure=False, timeout_s=30.0)
        if result.state != PIEState.READY:
            pytest.fail(
                f"PIE restart failed: state={result.state} "
                f"compile_error={result.compile_error[:200] if result.compile_error else ''}"
            )
        time.sleep(0.5)
        client.connect()
    if client.step_mode != StepMode.DIRECT:
        client.runtime.set_mode(StepMode.DIRECT)


@pytest.fixture
def fresh_live_client():
    """Bare client + discover. No scene bootstrap, no PIE. Use for
    scene-authoring tests that manage their own level lifecycle.

    Uses the default 120s recv timeout; scene-authoring ops like
    import_xml and create_level can each legitimately take tens of
    seconds on a cold editor.
    """
    client = _make_client()
    try:
        client.connect()
        yield client
    finally:
        try:
            client.close()
        except Exception:
            pass


def _claim_all(client) -> None:
    """Take control of every articulation the client can see.

    The server refuses actuator writes from a client that holds no claim
    (`not_control_owner`), which is what `claim_control` exists for. A test
    driving ctrl or twist is doing exactly what a policy runner does, so it
    claims the same way; `force` because a previous run that died without
    releasing would otherwise lock every later one out.
    """
    for prefix in list(getattr(client, "articulations", {}) or {}):
        try:
            client.runtime.claim_control(prefix, force=True)
        except Exception:
            pass


@pytest.fixture
def pie_client(_live_session):
    """Per-test client on the session-bootstrapped scene with PIE on.

    Each test gets a fresh ``URLabClient`` but reuses the editor's
    single PIE session. ``client.reset()`` resets physics state to the
    keyframe default so tests start with clean qpos / qvel / ctrl. The
    fixture does **not** stop PIE on teardown — PIE stays up for the
    whole session, only torn down at session end.
    """
    state: SceneBootstrap = _live_session
    client = _make_client()
    try:
        client.connect()
        _ensure_pie_for_pie_client(client)
        _claim_all(client)
        # Clean physics state for every test. reset() preserves the
        # active step mode and articulation count; only qpos/qvel/ctrl
        # snap back to keyframe defaults.
        try:
            client.reset()
        except Exception:
            pass
        # Clear any leftover recording state from a prior test (or a
        # prior pytest run — the server's recording flag survives
        # across bridge sessions). The server only allows one active
        # recording at a time, so stop it if one is running, then
        # clear the buffer. Both ops are safe no-ops when nothing is
        # active.
        try:
            client.recording.stop()
        except Exception:
            pass
        try:
            client.recording.clear_buffer()
        except Exception:
            pass
        yield client
    finally:
        try:
            client.close()
        except Exception:
            pass


@pytest.fixture
def golden_session(pie_client):
    """Alias for pie_client. Kept so test_live_golden.py reads naturally."""
    yield pie_client


@pytest.fixture
def scene_loaded_client(_live_session):
    """Client on the session-bootstrapped scene with the **current**
    PIE state — whatever the prior test left it in. Use for tests that
    drive the PIE lifecycle (sim.start / sim.stop) themselves. Each
    such test is responsible for leaving PIE on when it exits so the
    next pie_client test doesn't have to pay the recompile cost.
    """
    state: SceneBootstrap = _live_session
    client = _make_client()
    try:
        client.connect()
        yield client
    finally:
        try:
            client.close()
        except Exception:
            pass
