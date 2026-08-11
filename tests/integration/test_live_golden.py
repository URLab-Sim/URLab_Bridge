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

L2 (numerical golden): the reference is stock MuJoCo, computed here,
not a recording. Two separate questions, since they fail for unrelated
reasons: does UE compile the model the MJCF describes (handshake model
vs ``mj_loadXML``, option block included), and does UE step it the way
MuJoCo would (one ctrl schedule, replayed on both).

Both fixtures require ``URLAB_LIVE=1`` + a running editor on the host
and port configured in conftest.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

GOLDEN_SCENE = Path(__file__).resolve().parent.parent / "fixtures" / "golden_scene.xml"

N_STEPS = 200
TIMESTEP = 0.002


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


def _ctrl_at(step: int) -> np.ndarray:
    """Two out-of-phase sinusoids. Smooth and closed-form, so the
    schedule is reproducible without leaning on a particular numpy RNG
    version, and so neither joint spends the run against its limit."""
    t = step * TIMESTEP
    return np.array([
        0.6 * np.sin(2.0 * np.pi * 0.5 * t),
        0.4 * np.sin(2.0 * np.pi * 0.8 * t + 1.0),
    ])


def _run_live(client, n_steps: int):
    """Walk the ctrl schedule on the live session and return
    (qpos[n_steps, nq], qvel[n_steps, nv], sensor[n_steps, ns])."""
    art = _only_articulation(client)
    actuator_names = list(art.actuators.keys())
    sensor_names = list(art.sensors.keys())

    qpos = np.empty((n_steps, art.qpos_array.size), dtype=np.float64)
    qvel = np.empty((n_steps, art.qvel_array.size), dtype=np.float64)
    sens = np.empty((n_steps, sum(art.sensors[n].dim for n in sensor_names)), dtype=np.float64)

    client.reset()
    for i in range(n_steps):
        art.set_ctrl(dict(zip(actuator_names, _ctrl_at(i))))
        client.step(n_steps=1)
        qpos[i] = art.qpos_array
        qvel[i] = art.qvel_array
        offset = 0
        for n in sensor_names:
            sensor = art.sensors[n]
            sens[i, offset : offset + sensor.dim] = sensor.latest
            offset += sensor.dim
    return qpos, qvel, sens


def _run_stock(model, actuator_names, sensor_names, n_steps: int):
    """The same schedule under ``mj_step``. Slots resolve by name, since
    UE namespaces every element with the actor prefix."""
    data = mujoco.MjData(model)

    def _slot(objtype, short):
        names = [mujoco.mj_id2name(model, objtype, i) for i in range(
            model.nu if objtype == mujoco.mjtObj.mjOBJ_ACTUATOR else model.nsensor)]
        hits = [i for i, name in enumerate(names) if name == short or name.endswith("_" + short)]
        assert len(hits) == 1, f"{short!r} matched {hits} in {names}"
        return hits[0]

    ctrl_slots = [_slot(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
    sensor_ids = [_slot(mujoco.mjtObj.mjOBJ_SENSOR, n) for n in sensor_names]

    qpos = np.empty((n_steps, model.nq), dtype=np.float64)
    qvel = np.empty((n_steps, model.nv), dtype=np.float64)
    sens = np.empty((n_steps, sum(model.sensor_dim[i] for i in sensor_ids)), dtype=np.float64)

    for i in range(n_steps):
        ctrl = _ctrl_at(i)
        for slot, actuator in enumerate(ctrl_slots):
            data.ctrl[actuator] = ctrl[slot]
        mujoco.mj_step(model, data)
        qpos[i] = data.qpos
        qvel[i] = data.qvel
        offset = 0
        for sensor in sensor_ids:
            width = model.sensor_dim[sensor]
            adr = model.sensor_adr[sensor]
            sens[i, offset : offset + width] = data.sensordata[adr : adr + width]
            offset += width
    return qpos, qvel, sens


def test_compiled_model_matches_stock(golden_session):
    """The model UE handed over is the model the MJCF describes.

    Reports against the field that carries the difference, rather than
    as a trajectory that drifts for no stated reason. The option block
    is included: it is scene-wide state the manager owns.
    """
    ue = golden_session.model
    stock = mujoco.MjModel.from_xml_path(str(GOLDEN_SCENE))

    for field in ("timestep", "integrator", "gravity", "solver", "iterations",
                  "ls_iterations", "tolerance", "cone", "jacobian", "impratio",
                  "disableflags", "enableflags", "wind", "density", "viscosity"):
        theirs, ours = getattr(stock.opt, field), getattr(ue.opt, field)
        assert np.array_equal(np.asarray(theirs), np.asarray(ours)), \
            f"mjOption.{field}: stock={theirs} ue={ours}"

    for field in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsensor"):
        assert getattr(stock, field) == getattr(ue, field), \
            f"model size {field}: stock={getattr(stock, field)} ue={getattr(ue, field)}"

    for field in ("body_mass", "body_inertia", "body_pos", "body_quat", "dof_damping",
                  "dof_armature", "jnt_range", "jnt_axis", "actuator_gainprm",
                  "actuator_biasprm", "actuator_gear", "geom_size", "geom_solref",
                  "geom_solimp", "geom_friction", "geom_margin"):
        np.testing.assert_allclose(
            getattr(ue, field), getattr(stock, field), rtol=1e-12, atol=1e-12,
            err_msg=f"mjModel.{field} diverged from stock",
        )


def test_numerical_golden(golden_session):
    """Step the live sim and stock MuJoCo through the same schedule."""
    client = golden_session
    art = _only_articulation(client)
    actuator_names = list(art.actuators.keys())
    sensor_names = list(art.sensors.keys())

    live = _run_live(client, N_STEPS)
    stock = _run_stock(
        mujoco.MjModel.from_xml_path(str(GOLDEN_SCENE)),
        actuator_names, sensor_names, N_STEPS,
    )

    # Same mj_step over the same model, so the slack is the wire round-trip
    # and a ULP of ordering; the scene is dissipative, so it stays a floor.
    for name, mine, theirs in zip(("qpos", "qvel", "sensor"), live, stock):
        np.testing.assert_allclose(
            mine, theirs, rtol=1e-9, atol=1e-11,
            err_msg=f"{name} diverged from stock MuJoCo",
        )
