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

"""End-to-end golden tests against a live editor.

L1 (wire smoke): exercises every namespace + the PIE lifecycle, no
numerical assertions. Catches RPC-shape and dispatcher regressions.

L2 (numerical golden): replays a seeded ctrl schedule on the golden
scene in direct mode and asserts qpos/qvel match a checked-in
trajectory bit-identically (modulo float-precision noise).

Both fixtures require ``URLAB_LIVE=1`` + a running editor on the host
and port configured in conftest.

Regenerate the trajectory after a deliberate physics change:

    URLAB_LIVE=1 pytest tests/integration/test_live_golden.py::test_numerical_golden \
        --regenerate-golden
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).resolve().parent / "_golden"
GOLDEN_TRAJECTORY = GOLDEN_DIR / "trajectory.npz"

SEED = 42
N_STEPS = 200
CTRL_SCALE = 0.4


# ---------------------------------------------------------------------------
# L1 — wire smoke
# ---------------------------------------------------------------------------


def _only_articulation(client):
    """Golden scene has exactly one articulation; lookup by name is
    fragile (URLab derives the name from UE actor naming + MJCF body
    name). Iterate instead."""
    arts = list(client.articulations.values())
    assert len(arts) == 1, f"expected 1 articulation, got {len(arts)}: {list(client.articulations.keys())}"
    return arts[0]


def test_wire_smoke(golden_session):
    """One pass through every namespace on the live golden session."""
    from urlab_client import (
        ActorBounds,
        ActorHierarchyNode,
        CameraPose,
        ContactsResult,
        SceneSnapshot,
        URLabRPCError,
        URLabSpawnHandle,
    )

    client = golden_session

    # Handshake state
    assert client.session_id, "no session id after discover"
    assert client.manager_present, "manager should be present after PIE start"
    assert client.articulations, "golden scene should expose an articulation"

    art = _only_articulation(client)
    assert len(art.actuators) == 2, f"expected 2 actuators, got {list(art.actuators.keys())}"
    assert len(art.sensors) == 2, f"expected 2 sensors, got {list(art.sensors.keys())}"

    # sim namespace
    status = client.sim.status()
    assert status.state.value == "ready"

    # outliner namespace — list_blueprints + list_actors round-trip
    blueprints = client.outliner.list_blueprints()
    assert isinstance(blueprints, list)
    actors = client.outliner.list_actors()
    assert isinstance(actors, list)
    assert actors, "outliner should report at least the imported actors"

    # outliner namespace — find / bounds
    arts = client.outliner.find_actors(class_filter="AMjArticulation")
    assert any(a.actor_id == "golden_root" for a in arts), \
        f"find_actors should report golden_root, got {[a.actor_id for a in arts]}"
    bounds = client.outliner.get_actor_bounds("golden_root")
    assert isinstance(bounds, ActorBounds)
    assert bounds.actor_name, "get_actor_bounds should resolve a UE name"

    # scene namespace — snapshot / hierarchy / duplicate
    snap = client.scene.snapshot()
    assert isinstance(snap, SceneSnapshot)
    assert snap.actors, "snapshot should list at least the manager + articulation"
    tree = client.scene.actor_hierarchy("golden_root")
    assert isinstance(tree, ActorHierarchyNode)
    assert tree.name, "actor_hierarchy root should have a name"
    dup = client.scene.duplicate_actor("golden_root", "golden_root_dup")
    try:
        assert isinstance(dup, URLabSpawnHandle)
        assert dup.actor_id == "golden_root_dup"
    finally:
        # Always destroy the duplicate so subsequent tests see a clean scene.
        try:
            client.scene.remove_actor("golden_root_dup")
        except URLabRPCError:
            pass

    # runtime namespace — step + observation surface
    actuator_names = list(art.actuators.keys())
    art.set_ctrl({actuator_names[0]: 0.2, actuator_names[1]: -0.2})
    client.step(n_steps=1)
    assert art.qpos_array.size > 0
    assert art.qvel_array.size > 0

    # runtime namespace — contacts / mocap
    contacts = client.runtime.get_contacts(max_contacts=16)
    assert isinstance(contacts, ContactsResult)  # n_contacts may be 0; just shape-check
    # Mocap ops: golden scene has no mocap body — assert the typed error
    # comes back rather than crashing the dispatcher.
    try:
        client.runtime.read_mocap_pose("golden_root")
        pytest.fail("read_mocap_pose should reject non-mocap body")
    except URLabRPCError as exc:
        assert exc.code in ("not_mocap_body", "unknown_body"), \
            f"unexpected error code {exc.code!r}"

    # debug namespace — fire-and-forget; just verify wire success.
    client.debug.draw_marker((0.0, 0.0, 0.5), (1.0, 0.0, 0.0), ttl=0.0, label="smoke")
    client.debug.draw_line((0.0, 0.0, 0.0), (0.5, 0.0, 0.5), (0.0, 1.0, 0.0), ttl=0.0)
    client.debug.draw_box((0.0, 0.0, 0.25), (0.1, 0.1, 0.1), (0.0, 0.0, 1.0), ttl=0.0)
    client.debug.draw_axes((0.0, 0.0, 0.5), scale=0.15, ttl=0.0)
    client.debug.clear_markers()
    client.debug.set_overlay_text("URLab live smoke")
    client.debug.set_overlay_text("")  # clear

    # runtime.list_keyframes — golden scene may or may not have any;
    # we just shape-check that the reply decodes to a list.
    kfs = client.runtime.list_keyframes()
    assert isinstance(kfs, list)

    # viewport namespace — capture + restore the pose so the test
    # doesn't leave the user's editor pointing somewhere weird.
    original_pose = client.viewport.get_camera()
    assert isinstance(original_pose, CameraPose)
    moved = client.viewport.set_camera(
        (2.0, 2.0, 1.5), rotation_euler=(0.0, -30.0, 225.0), fov=60.0,
    )
    assert isinstance(moved, CameraPose)
    framed = client.viewport.frame_actor("golden_root")
    assert isinstance(framed, CameraPose)
    assert client.viewport.set_mode("wireframe") == "wireframe"
    assert client.viewport.set_mode("lit") == "lit"
    tracked_path = client.viewport.track_actor("golden_root", smoothing=0.0)
    assert tracked_path, "track_actor should return the UE actor path"
    assert client.viewport.untrack() is True
    assert client.viewport.untrack() is False  # idempotent
    # Restore original pose.
    client.viewport.set_camera(
        original_pose.location,
        rotation_euler=original_pose.rotation_euler,
        fov=original_pose.fov,
    )

    # reset round-trips
    client.reset()
    assert client.step_count == 0


# ---------------------------------------------------------------------------
# L2 — numerical golden
# ---------------------------------------------------------------------------


def _run_trajectory(client, n_steps: int, seed: int):
    """Walk a seeded ctrl schedule on the live golden session and
    return (qpos[n_steps, nq], qvel[n_steps, nv], sensor[n_steps, ns])."""
    art = _only_articulation(client)
    actuator_names = list(art.actuators.keys())
    n_act = len(actuator_names)
    rng = np.random.default_rng(seed)

    nq = art.qpos_array.size
    nv = art.qvel_array.size
    sensor_names = list(art.sensors.keys())
    ns = sum(art.sensors[n].dim for n in sensor_names)

    qpos = np.empty((n_steps, nq), dtype=np.float64)
    qvel = np.empty((n_steps, nv), dtype=np.float64)
    sens = np.empty((n_steps, ns), dtype=np.float64)

    client.reset()
    for i in range(n_steps):
        ctrl = rng.uniform(-CTRL_SCALE, CTRL_SCALE, size=n_act)
        art.set_ctrl(dict(zip(actuator_names, ctrl)))
        client.step(n_steps=1)
        qpos[i] = art.qpos_array
        qvel[i] = art.qvel_array
        offset = 0
        for n in sensor_names:
            sensor = art.sensors[n]
            width = sensor.dim
            sens[i, offset : offset + width] = sensor.latest
            offset += width
    return qpos, qvel, sens


def test_numerical_golden(golden_session, regenerate_golden):
    """Replay a deterministic trajectory; bit-compare against the golden."""
    client = golden_session
    qpos, qvel, sens = _run_trajectory(client, N_STEPS, SEED)

    if regenerate_golden:
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            GOLDEN_TRAJECTORY,
            qpos=qpos,
            qvel=qvel,
            sens=sens,
            seed=np.int64(SEED),
            n_steps=np.int64(N_STEPS),
        )
        pytest.skip(f"regenerated {GOLDEN_TRAJECTORY}")

    if not GOLDEN_TRAJECTORY.is_file():
        pytest.skip(
            f"golden trajectory missing at {GOLDEN_TRAJECTORY}; "
            f"run with --regenerate-golden to create it"
        )

    golden = np.load(GOLDEN_TRAJECTORY)
    assert int(golden["seed"]) == SEED, "golden seed drift"
    assert int(golden["n_steps"]) == N_STEPS, "golden length drift"

    # URLab is mj_step under the hood; trajectory should be byte-identical
    # to the captured reference. atol leaves headroom for any msgpack
    # float roundtripping introduced by the wire layer.
    np.testing.assert_allclose(qpos, golden["qpos"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(qvel, golden["qvel"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sens, golden["sens"], rtol=1e-9, atol=1e-11)
