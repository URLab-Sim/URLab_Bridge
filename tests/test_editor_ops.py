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

"""Wire tests for the editor-only RPCs.

These tests don't need a live editor — they queue scripted replies on
the mock server and verify the request shape + reply absorption. The
real factory-driven import is exercised on the UE side
(URLab.LevelOps.ImportXmlEndToEnd).
"""

from __future__ import annotations

import pytest

from urlab_client import URLabClient, URLabRPCError

from . import wire_replies as wr


def _make_client(port: int) -> URLabClient:
    return URLabClient(
        "tcp://127.0.0.1",
        step_mode="direct",
        step_port=port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )


def _open_session(client: URLabClient, mock_step_server, base_handshake):
    """Issue hello so subsequent ops have a session_id to send."""
    mock_step_server.replies.append(base_handshake)
    client.connect()


def test_editor_time_hello_succeeds_with_no_manager(mock_step_server, mujoco_mod):
    """Editor-time / pre-PIE handshake: server returns ``manager_present=false``
    with no mjb / no articulations. ``discover()`` must succeed and skip
    every PIE-only follow-up (auto-promote, streaming SUB startup)."""
    # Use step_mode='direct' to make sure we'd normally try to auto-promote.
    # The bridge must NOT enqueue a set_mode RPC when manager_present=false.
    client = URLabClient(
        "tcp://127.0.0.1",
        step_mode="direct",
        step_port=mock_step_server.port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=True,
    )
    try:
        mock_step_server.replies.append({
            "op": "hello_ok",
            "session_id": "editor-uuid",
            "urlab_version": "urlab/test",
            "mujoco_version": mujoco_mod.__version__,
            "manager_present": False,
            "entities": [],
        })
        client.connect()
    finally:
        client.close()

    assert client.session_id == "editor-uuid"
    assert client.manager_present is False
    assert client.articulations == {}
    assert client.articulations_by_id == {}
    # Only the hello request is recorded — meta is filtered out into
    # `received_meta`. No auto-promote set_mode since
    # manager_present=False short-circuits the rest of discover().
    ops = [r["op"] for r in mock_step_server.received]
    assert ops == ["hello"]
    assert len(mock_step_server.received_meta) == 1


def test_handshake_defaults_manager_present_true_for_legacy_servers(
    mock_step_server, base_handshake
):
    """Older server builds don't ship ``manager_present``; default to True
    so existing PIE-time clients keep working."""
    client = URLabClient(
        "tcp://127.0.0.1",
        step_mode="auto",
        step_port=mock_step_server.port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )
    try:
        # base_handshake has no manager_present field.
        assert "manager_present" not in base_handshake
        mock_step_server.replies.append(base_handshake)
        client.connect()
    finally:
        client.close()
    assert client.manager_present is True


def test_import_xml_request_shape_and_reply_unpack(
    mock_step_server, base_handshake, tmp_path
):
    from urlab_client import URLabBlueprint

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.import_xml_ok(
            blueprint_class_path="/Game/MuJoCoImports/foo.foo_C",
            blueprint_short_name="foo",
            imported_now=True,
        ))
        result = client.scene.import_xml(str(tmp_path / "foo.xml"), force_reimport=False)
    finally:
        client.close()

    assert isinstance(result, URLabBlueprint)
    assert result.class_path == "/Game/MuJoCoImports/foo.foo_C"
    assert result.short_name == "foo"
    assert result.imported_now is True

    req = mock_step_server.received[-1]
    assert req["op"] == "import_xml"
    assert req["path"] == str(tmp_path / "foo.xml")
    assert req["force_reimport"] is False


def test_import_xml_force_reimport_flag_is_forwarded(
    mock_step_server, base_handshake, tmp_path
):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.import_xml_ok(
            blueprint_class_path="/Game/MuJoCoImports/bar.bar_C",
            blueprint_short_name="bar",
            imported_now=True,
        ))
        client.scene.import_xml(str(tmp_path / "bar.xml"), force_reimport=True)
    finally:
        client.close()

    req = mock_step_server.received[-1]
    assert req["force_reimport"] is True


def test_import_xml_not_in_editor_raises(mock_step_server, base_handshake, tmp_path):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.error(
            "not_in_editor",
            "import_xml is editor-only and has no registered handler",
        ))
        with pytest.raises(URLabRPCError) as exc_info:
            client.scene.import_xml(str(tmp_path / "foo.xml"))
    finally:
        client.close()

    assert exc_info.value.code == "not_in_editor"


# ---------------------------------------------------------------------------
# create_level / load_level / save_level
# ---------------------------------------------------------------------------


def test_create_level_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.create_level_ok(level_path="/Game/Levels/myscene"))
        result = client.scene.create_level("myscene")
    finally:
        client.close()

    assert result is None
    req = mock_step_server.received[-1]
    assert req["op"] == "create_level"
    assert req["name"] == "myscene"


def test_load_level_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.load_level_ok(level_path="/Game/Levels/myscene"))
        result = client.scene.load_level("/Game/Levels/myscene")
    finally:
        client.close()

    assert result is None
    req = mock_step_server.received[-1]
    assert req["op"] == "load_level"
    assert req["level_path"] == "/Game/Levels/myscene"


def test_save_level_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.save_level_ok(level_path="/Game/Levels/myscene"))
        out = client.scene.save_level()
    finally:
        client.close()

    assert out is None
    req = mock_step_server.received[-1]
    assert req["op"] == "save_level"


# ---------------------------------------------------------------------------
# spawn_actor / destroy_actor
# ---------------------------------------------------------------------------


def test_spawn_actor_returns_editor_stub(mock_step_server, base_handshake):
    from urlab_client import URLabSpawnHandle

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.spawn_actor_ok(
            actor_id="robot_a",
            actor_name="go2_C_UAID_X_1",
            actor_path="/Game/Levels/myscene.myscene:PersistentLevel.go2_C_UAID_X_1",
            blueprint_class_path="/Game/MuJoCoImports/go2.go2_C",
            location=[0.5, 0.0, 0.5],
            rotation_quat=[0.0, 0.0, 0.0, 1.0],
            was_existing=False,
        ))
        ed = client.scene.spawn_actor(
            "/Game/MuJoCoImports/go2",
            "robot_a",
            location=(0.5, 0.0, 0.5),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
        )
    finally:
        client.close()

    assert isinstance(ed, URLabSpawnHandle)
    assert ed.actor_id == "robot_a"
    assert ed.actor_name == "go2_C_UAID_X_1"
    assert ed.blueprint_class_path == "/Game/MuJoCoImports/go2.go2_C"
    assert ed.location == (0.5, 0.0, 0.5)
    assert ed.rotation_quat == (0.0, 0.0, 0.0, 1.0)
    assert ed.was_existing is False

    req = mock_step_server.received[-1]
    assert req["op"] == "spawn_actor"
    assert req["blueprint"] == "/Game/MuJoCoImports/go2"
    assert req["actor_id"] == "robot_a"
    assert req["location"] == [0.5, 0.0, 0.5]
    assert req["rotation_quat"] == [0.0, 0.0, 0.0, 1.0]


def test_spawn_actor_rotation_euler_passthrough(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.spawn_actor_ok(
            actor_id="robot_a",
            actor_name="x", actor_path="y",
            blueprint_class_path="z",
        ))
        client.scene.spawn_actor("/Game/X", "robot_a", rotation_euler=(0.0, 0.0, 90.0))
    finally:
        client.close()

    req = mock_step_server.received[-1]
    assert "rotation_euler" in req
    assert req["rotation_euler"] == [0.0, 0.0, 90.0]
    assert "rotation_quat" not in req


def test_destroy_actor_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.remove_actor_ok(requires_pie_restart=False))
        client.scene.remove_actor("robot_a")
    finally:
        client.close()

    req = mock_step_server.received[-1]
    assert req["op"] == "remove_actor"
    assert req["target"] == "robot_a"
    assert req.get("target_by", "actor_id") == "actor_id"


def test_destroy_actor_requires_target():
    client = _make_client(0)
    with pytest.raises(TypeError):
        client.scene.remove_actor()  # type: ignore[call-arg]  # missing positional


def test_async_editor_job_polls_op_status(mock_step_server, base_handshake):
    """An async editor op returns op_started + job_id; the client polls
    op_status until done and returns the embedded result transparently."""
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.op_started(job_id="job_7"))
        mock_step_server.replies.append(wr.op_status_ok(job_id="job_7", state="running"))
        mock_step_server.replies.append(wr.op_status_ok(
            job_id="job_7", state="done",
            result=wr.import_xml_ok(
                blueprint_class_path="/Game/MuJoCoImports/x.x_C",
                blueprint_short_name="x", imported_now=True),
        ))
        bp = client.scene.import_xml("/tmp/x.xml")
    finally:
        client.close()
    assert bp.class_path == "/Game/MuJoCoImports/x.x_C"
    ops = [r["op"] for r in mock_step_server.received]
    assert "import_xml" in ops
    assert ops.count("op_status") == 2


def test_async_editor_job_failed_raises(mock_step_server, base_handshake):
    """A failed async job surfaces as URLabRPCError carrying the job error."""
    from urlab_client.errors import URLabRPCError

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.op_started(job_id="j"))
        mock_step_server.replies.append(wr.op_status_ok(
            job_id="j", state="failed",
            result={"op": "error", "code": "import_failed", "message": "boom"}))
        with pytest.raises(URLabRPCError):
            client.scene.import_xml("/tmp/x.xml")
    finally:
        client.close()


# ---------------------------------------------------------------------------
# begin_pie / stop_pie / pie_status
# ---------------------------------------------------------------------------


def test_begin_pie_ready_absorbs_embedded_handshake(
    mock_step_server, base_handshake
):
    """On state=ready, begin_pie's reply carries a handshake_payload
    and the client should re-absorb it so ``model`` and ``articulations``
    refresh against the PIE world without an extra round-trip."""
    import copy
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        assert "vx300s" in client.articulations
        assert "robot_a" not in client.articulations_by_id

        post_pie = copy.deepcopy(base_handshake)
        for art in post_pie["entities"]:
            if art["prefix"] == "vx300s":
                art["actor_id"] = "robot_a"

        mock_step_server.replies.append(wr.begin_pie_ok(
            state="ready",
            handshake_payload=post_pie,
        ))
        result = client.sim.start(level_path="/Game/Levels/myscene")
    finally:
        client.close()

    from urlab_client import PIEState, PIEStartResult
    assert isinstance(result, PIEStartResult)
    assert result.state == PIEState.READY
    assert result.is_ready
    assert "robot_a" in client.articulations_by_id
    assert client.articulations_by_id["robot_a"] is client.articulations["vx300s"]


def test_begin_pie_compile_failed_raises_by_default(
    mock_step_server, base_handshake
):
    """state=compile_failed raises URLabPIEError unless raise_on_failure=False."""
    from urlab_client import PIEState, URLabPIEError

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.begin_pie_ok(
            state="compile_failed",
            compile_error="joint name 'wrist' duplicated",
        ))
        with pytest.raises(URLabPIEError) as exc_info:
            client.sim.start()
    finally:
        client.close()

    assert exc_info.value.state == PIEState.COMPILE_FAILED
    assert "wrist" in exc_info.value.message


def test_begin_pie_compile_failed_returns_result_when_no_raise(
    mock_step_server, base_handshake
):
    """raise_on_failure=False returns the result so callers can inspect
    compile_error and decide what to do."""
    from urlab_client import PIEState

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.begin_pie_ok(
            state="compile_failed",
            compile_error="joint name 'wrist' duplicated",
        ))
        result = client.sim.start(raise_on_failure=False)
    finally:
        client.close()

    assert result.state == PIEState.COMPILE_FAILED
    assert "wrist" in result.compile_error
    assert not result.is_ready


def test_begin_pie_timeout(mock_step_server, base_handshake):
    from urlab_client import PIEState

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.begin_pie_ok(state="timeout"))
        result = client.sim.start(timeout_s=1.0, raise_on_failure=False)
    finally:
        client.close()

    assert result.state == PIEState.TIMEOUT
    req = mock_step_server.received[-1]
    assert req["timeout_s"] == 1.0


def test_stop_pie_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.stop_pie_ok())
        client.sim.stop()
    finally:
        client.close()
    req = mock_step_server.received[-1]
    assert req["op"] == "stop_pie"


def test_pie_status_ready_state(mock_step_server, base_handshake):
    from urlab_client import PIEState, PIEStatus

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.pie_status_ok(state="ready", sim_time=1.234))
        out = client.sim.status()
    finally:
        client.close()
    assert isinstance(out, PIEStatus)
    assert out.state == PIEState.READY
    assert out.sim_time == 1.234


def test_set_actor_transform_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.set_actor_transform_ok(
            target="go2_C_UAID_X_1",
            actor_name="go2_C_UAID_X_1",
            requires_pie_restart=False,
        ))
        client.scene.set_actor_transform(
            "robot_a",
            location=(2.0, 0.0, 0.5),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
        )
    finally:
        client.close()
    req = mock_step_server.received[-1]
    assert req["op"] == "set_actor_transform"
    assert req["target"] == "robot_a"
    assert req.get("target_by", "actor_id") == "actor_id"
    assert req["location"] == [2.0, 0.0, 0.5]
    assert req["rotation_quat"] == [0.0, 0.0, 0.0, 1.0]


def test_set_actor_transform_validation():
    client = _make_client(0)
    with pytest.raises(TypeError):
        client.scene.set_actor_transform()  # type: ignore[call-arg]  # missing target
    with pytest.raises(ValueError):
        client.scene.set_actor_transform(
            "robot_a",
            rotation_quat=(0, 0, 0, 1),
            rotation_euler=(0, 0, 90),
        )


def test_spawn_light_returns_editor_stub(mock_step_server, base_handshake):
    from urlab_client import URLabLightHandle

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.spawn_light_ok(
            actor_id="sun",
            actor_name="DirectionalLight_1",
            actor_path="/Game/Levels/myscene.myscene:PersistentLevel.DirectionalLight_1",
            light_class="directional",
            intensity=7500.0,
            color=[0.9, 0.9, 0.8],
            location=[0.0, 0.0, 5.0],
            kind="directional",
            rotation_euler=[0.0, -45.0, 0.0],
        ))
        ed = client.scene.spawn_light(
            "directional",
            actor_id="sun",
            location=(0.0, 0.0, 5.0),
            rotation_euler=(0.0, -45.0, 0.0),
            intensity=7500.0,
            color=(0.9, 0.9, 0.8),
        )
    finally:
        client.close()

    assert isinstance(ed, URLabLightHandle)
    assert ed.kind == "directional"
    assert ed.actor_id == "sun"
    assert ed.intensity == 7500.0
    assert ed.color == (0.9, 0.9, 0.8)

    req = mock_step_server.received[-1]
    assert req["op"] == "spawn_light"
    assert req["kind"] == "directional"
    assert req["intensity"] == 7500.0


# ---------------------------------------------------------------------------
# set_qpos (manager-required runtime RPC)
# ---------------------------------------------------------------------------


def test_set_qpos_default_target_is_actor_id(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.set_qpos_ok(
            target="robot_a",
            actor_id="robot_a",
            actor_name="go2_C_UAID_X_1",
            qpos=[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            free_base_shortcut=True,
        ))
        out = client.runtime.set_qpos("robot_a", [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    finally:
        client.close()

    assert out is None
    req = mock_step_server.received[-1]
    assert req["op"] == "set_qpos"
    assert req["target"] == "robot_a"
    assert req.get("target_by", "actor_id") == "actor_id"
    assert "actor_id" not in req
    assert req["qpos"] == [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]


def test_set_qpos_by_name_uses_target_by(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.set_qpos_ok(
            target="go2_C_UAID_X_1",
            actor_name="go2_C_UAID_X_1",
            qpos=[0.42],
            free_base_shortcut=False,
        ))
        client.runtime.set_qpos("go2_C_UAID_X_1", [0.42], by_name=True)
    finally:
        client.close()

    req = mock_step_server.received[-1]
    assert req["target"] == "go2_C_UAID_X_1"
    assert req["target_by"] == "actor_name"
    assert "actor_id" not in req
    assert "articulation" not in req


def test_set_qpos_coerces_qpos_to_floats(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.set_qpos_ok(
            target="robot_a",
            actor_id="robot_a",
            qpos=[1.0, 2.0, 3.0],
            free_base_shortcut=False,
        ))
        client.runtime.set_qpos("robot_a", (1, 2, 3))
    finally:
        client.close()
    req = mock_step_server.received[-1]
    assert req["qpos"] == [1.0, 2.0, 3.0]
    assert all(isinstance(x, float) for x in req["qpos"])


def test_set_qpos_unknown_articulation_raises(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.error("unknown_articulation", "robot_a"))
        with pytest.raises(URLabRPCError) as exc_info:
            client.runtime.set_qpos("robot_a", [0.0])
    finally:
        client.close()
    assert exc_info.value.code == "unknown_articulation"


# ---------------------------------------------------------------------------
# set_qpos in puppet mode must not be silently overwritten by the next
# puppet step.
# ---------------------------------------------------------------------------


def test_set_qpos_mirrors_into_client_data_for_puppet_mode(
    mock_step_server, base_handshake, mujoco_mod
):
    """In puppet mode the client owns client.data and ships qpos/qvel up
    on every step. Without local mirroring, calling set_qpos would write
    the server's qpos but leave client.data.qpos unchanged — so the very
    next step would ship the OLD qpos back up and overwrite the write.
    Verify the bridge mirrors the server's qpos echo into client.data so
    the next puppet step preserves it."""
    client = URLabClient(
        "tcp://127.0.0.1",
        step_mode="puppet",
        step_port=mock_step_server.port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )
    try:
        _open_session(client, mock_step_server, base_handshake)
        # vx300s has two hinge joints (waist + shoulder); each is 1 dof.
        art = client.articulations["vx300s"]
        joints = list(art.joints.values())
        assert len(joints) == 2, "vx300s should have two hinge joints"
        slot_starts = [int(j.qpos_offset) for j in joints]

        # Pre-set: zeroed at both slots.
        for s in slot_starts:
            client.data.qpos[s] = 0.0

        # Server echoes back the actual write — full per-articulation
        # qpos path (one float per joint, in registration order).
        new_qpos = [0.7, -0.3]
        mock_step_server.replies.append(wr.set_qpos_ok(
            target="vx300s",
            actor_id="vx300s",
            actor_name="vx300s_actor",
            qpos=new_qpos,
            free_base_shortcut=False,
        ))
        client.articulations_by_id["vx300s"] = art  # actor_id lookup

        client.runtime.set_qpos("vx300s", new_qpos)

        # Each joint's slot in client.data.qpos must reflect the echo —
        # otherwise a puppet step would ship zeros back to UE.
        for s, v in zip(slot_starts, new_qpos):
            assert client.data.qpos[s] == pytest.approx(v), \
                f"qpos[{s}] not mirrored: got {client.data.qpos[s]}, want {v}"
    finally:
        client.close()


def test_set_qpos_dim_mismatch_raises(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append({
            "op": "error",
            "code": "dim_mismatch",
            "message": "qpos length 3 != articulation qpos dim 1",
        })
        with pytest.raises(URLabRPCError) as exc_info:
            client.runtime.set_qpos("robot_a", [0.0, 0.0, 0.0])
    finally:
        client.close()
    assert exc_info.value.code == "dim_mismatch"


# ---------------------------------------------------------------------------
# Outliner / quick-convert (Phase A)
# ---------------------------------------------------------------------------


def test_list_actors_returns_actor_list(mock_step_server, base_handshake):
    from urlab_client import ActorInfo

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append({
            "op": "list_actors_ok",
            "actors": [
                {
                    "name": "MjArtA_C_UAID_X",
                    "class": "AMjArticulation",
                    "actor_id": "robot_a",
                    "is_articulation": True,
                    "is_static_mesh_actor": False,
                    "is_light": False,
                    "has_quick_convert": False,
                    "location": [0.0, 0.0, 0.5],
                    "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "name": "Cube_2",
                    "class": "StaticMeshActor",
                    "actor_id": "",
                    "is_articulation": False,
                    "is_static_mesh_actor": True,
                    "is_light": False,
                    "has_quick_convert": True,
                    "quick_convert": {
                        "static": True,
                        "complex_mesh": False,
                        "coacd_threshold": 0.05,
                        "driven_by_unreal": False,
                        "friction": [1.0, 1.0, 1.0],
                    },
                    "location": [1.0, 0.0, 0.0],
                    "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                },
            ],
            "in_pie": False,
        })
        actors = client.outliner.list_actors()
    finally:
        client.close()
    assert len(actors) == 2
    assert all(isinstance(a, ActorInfo) for a in actors)
    assert actors[0].actor_id == "robot_a"
    assert actors[0].is_articulation is True
    assert actors[1].has_quick_convert is True
    assert actors[1].static is True
    assert actors[1].friction == (1.0, 1.0, 1.0)


def test_select_actor_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append({
            "op": "select_actor_ok",
            "actor_name": "MjArtA_C_UAID_X",
        })
        client.outliner.select_actor("robot_a")
    finally:
        client.close()
    req = mock_step_server.received[-1]
    assert req["op"] == "select_actor"
    assert req["target"] == "robot_a"
    assert req.get("target_by", "actor_id") == "actor_id"
    assert "actor_id" not in req
    assert "actor_name" not in req


def test_select_actor_by_name(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append({
            "op": "select_actor_ok",
            "actor_name": "Cube_2",
        })
        client.outliner.select_actor("Cube_2", by_name=True)
    finally:
        client.close()
    req = mock_step_server.received[-1]
    assert req["target"] == "Cube_2"
    assert req["target_by"] == "actor_name"
    assert "actor_id" not in req
    assert "actor_name" not in req


def test_add_quick_convert_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append({
            "op": "add_quick_convert_ok",
            "actor_name": "Cube_2",
            "static": True,
            "complex_mesh": False,
            "coacd_threshold": 0.05,
            "driven_by_unreal": False,
            "requires_pie_restart": False,
        })
        client.outliner.add_quick_convert(
            "Cube_2", by_name=True, static=True, complex_mesh=False,
        )
    finally:
        client.close()
    req = mock_step_server.received[-1]
    assert req["op"] == "add_quick_convert"
    assert req["target"] == "Cube_2"
    assert req["target_by"] == "actor_name"
    assert req["static"] is True
    assert req["friction"] == [1.0, 1.0, 1.0]


def test_list_blueprints_returns_bp_list(mock_step_server, base_handshake):
    from urlab_client import BlueprintInfo

    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append({
            "op": "list_blueprints_ok",
            "blueprints": [
                {
                    "blueprint_class_path": "/Game/MuJoCoImports/foo.foo_C",
                    "blueprint_short_name": "foo",
                    "package_path": "/Game/MuJoCoImports/foo",
                },
                {
                    "blueprint_class_path": "/Game/MuJoCoImports/bar.bar_C",
                    "blueprint_short_name": "bar",
                    "package_path": "/Game/MuJoCoImports/bar",
                },
            ],
        })
        bps = client.outliner.list_blueprints()
    finally:
        client.close()
    assert len(bps) == 2
    assert all(isinstance(b, BlueprintInfo) for b in bps)
    assert bps[0].short_name == "foo"
    assert bps[1].class_path == "/Game/MuJoCoImports/bar.bar_C"


def test_remove_quick_convert_request_shape(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append({
            "op": "remove_quick_convert_ok",
            "actor_name": "Cube_2",
            "requires_pie_restart": False,
        })
        client.outliner.remove_quick_convert("Cube_2", by_name=True)
    finally:
        client.close()
    req = mock_step_server.received[-1]
    assert req["op"] == "remove_quick_convert"
    assert req["target"] == "Cube_2"
    assert req["target_by"] == "actor_name"


def test_editor_runtime_lookup(base_handshake):
    """``URLabSpawnHandle.runtime(client)`` resolves to the live
    URLabArticulation post-discover via ``articulations_by_id``."""
    import copy
    from urlab_client import URLabClient, URLabSpawnHandle

    h = copy.deepcopy(base_handshake)
    for art in h["entities"]:
        if art["prefix"] == "vx300s":
            art["actor_id"] = "robot_a"

    client = URLabClient(step_mode="direct")
    client._apply_handshake(h)

    ed = URLabSpawnHandle(
        actor_id="robot_a",
        actor_name="ignored",
        actor_path="ignored",
        blueprint_class_path="ignored",
    )
    assert ed.runtime(client) is client.articulations["vx300s"]
    # Unknown id resolves to None.
    ed_missing = URLabSpawnHandle(
        actor_id="nope",
        actor_name="", actor_path="", blueprint_class_path="",
    )
    assert ed_missing.runtime(client) is None


# ---------------------------------------------------------------------------
# Scene introspection: find_actors, get_actor_bounds, snapshot,
# duplicate_actor, actor_hierarchy.
# ---------------------------------------------------------------------------


def test_find_actors_sends_optional_filters(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.find_actors_ok(
            actors=[{
                "name": "Cube_0", "class": "AStaticMeshActor",
                "actor_id": "", "is_articulation": False,
                "has_quick_convert": False,
                "location": [0.0, 0.0, 0.0],
                "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                "label": "Cube_0",
                "is_static_mesh_actor": True,
                "is_light": False,
            }],
            in_pie=False,
        ))
        actors = client.outliner.find_actors(
            class_filter="AStaticMeshActor",
            tag="cube",
            name_prefix="Cube",
            in_pie=True,
        )
        assert len(actors) == 1
        assert actors[0].name == "Cube_0"
        # Verify the wire request carried all four filters.
        req = mock_step_server.received[-1]
        assert req["op"] == "find_actors"
        assert req["class_filter"] == "AStaticMeshActor"
        assert req["tag"] == "cube"
        assert req["name_prefix"] == "Cube"
        assert req["in_pie"] is True
    finally:
        client.close()


def test_get_actor_bounds_returns_typed_dataclass(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.get_actor_bounds_ok(
            actor_name="Robot",
            min=[-1.0, -2.0, 0.0],
            max=[1.0, 2.0, 1.0],
            center=[0.0, 0.0, 0.5],
            extents=[1.0, 2.0, 0.5],
        ))
        bounds = client.outliner.get_actor_bounds("robot")
        assert bounds.actor_name == "Robot"
        assert bounds.min == (-1.0, -2.0, 0.0)
        assert bounds.max == (1.0, 2.0, 1.0)
        assert bounds.center == (0.0, 0.0, 0.5)
        assert bounds.extents == (1.0, 2.0, 0.5)
        req = mock_step_server.received[-1]
        assert req["op"] == "get_actor_bounds"
        assert req["target"] == "robot"
        assert req["components_only"] is False
    finally:
        client.close()


def test_snapshot_unpacks_urlab_metadata(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.snapshot_ok(
            level_path="/Game/Levels/Test",
            in_pie=True,
            actors=[{
                "name": "vx300s",
                "class": "AMjArticulation",
                "actor_id": "robot_a",
                "label": "vx300s",
                "location": [0.0, 0.0, 0.0],
                "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                "tags": ["URLab.ActorId=robot_a"],
                "urlab": {
                    "mj_class": "articulation",
                    "joints": ["waist", "shoulder"],
                    "actuators": ["waist", "shoulder"],
                    "sensors": [],
                    "cameras": [],
                },
            }],
        ))
        snap = client.scene.snapshot()
        assert snap.level_path == "/Game/Levels/Test"
        assert snap.in_pie is True
        assert len(snap.actors) == 1
        a = snap.actors[0]
        assert a.name == "vx300s"
        assert a.actor_id == "robot_a"
        assert a.urlab["joints"] == ["waist", "shoulder"]
    finally:
        client.close()


def test_duplicate_actor_returns_spawn_handle(mock_step_server, base_handshake):
    from urlab_client import URLabSpawnHandle
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.duplicate_actor_ok(
            actor_id="dup",
            actor_name="vx300s_2",
            actor_path="/Game/Levels/Test.Test:PersistentLevel.vx300s_2",
            blueprint_class_path="/Game/MuJoCoImports/vx300s.vx300s_C",
        ))
        ed = client.scene.duplicate_actor(
            "vx300s_src", "dup", location=(2.0, 0.0, 0.0),
        )
        assert isinstance(ed, URLabSpawnHandle)
        assert ed.actor_id == "dup"
        assert ed.actor_name == "vx300s_2"
        req = mock_step_server.received[-1]
        assert req["op"] == "duplicate_actor"
        assert req["target"] == "vx300s_src"
        assert req["new_actor_id"] == "dup"
        assert req["location"] == [2.0, 0.0, 0.0]
    finally:
        client.close()


def test_actor_hierarchy_recurses(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.actor_hierarchy_ok(
            root={
                "name": "root",
                "class": "AMjArticulation",
                "location": [0.0, 0.0, 0.0],
                "children": [
                    {
                        "name": "child",
                        "class": "AStaticMeshActor",
                        "location": [1.0, 0.0, 0.0],
                        "children": [],
                    },
                ],
            },
        ))
        tree = client.scene.actor_hierarchy("root")
        assert tree.name == "root"
        assert tree.actor_class == "AMjArticulation"
        assert len(tree.children) == 1
        assert tree.children[0].name == "child"
        assert tree.children[0].location == (1.0, 0.0, 0.0)
        assert tree.children[0].children == []
    finally:
        client.close()


# ---------------------------------------------------------------------------
# MuJoCo runtime specifics: set_mocap_pose, read_mocap_pose, get_contacts.
# ---------------------------------------------------------------------------


def test_set_mocap_pose_requires_pos_or_quat(mock_step_server, base_handshake):
    from urlab_client import MocapPose
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        # No pos / no quat -> client-side ValueError before any RPC fires.
        try:
            client.runtime.set_mocap_pose("mocap_target")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

        mock_step_server.replies.append(wr.set_mocap_pose_ok(
            body="mocap_target",
            pos=[0.5, 0.2, 1.0],
            quat=[1.0, 0.0, 0.0, 0.0],
        ))
        result = client.runtime.set_mocap_pose(
            "mocap_target", pos=(0.5, 0.2, 1.0),
        )
        assert isinstance(result, MocapPose)
        assert result.body == "mocap_target"
        assert result.pos == (0.5, 0.2, 1.0)
        req = mock_step_server.received[-1]
        assert req["op"] == "set_mocap_pose"
        assert req["body"] == "mocap_target"
        assert req["pos"] == [0.5, 0.2, 1.0]
        assert "quat" not in req
    finally:
        client.close()


def test_read_mocap_pose_returns_typed_dataclass(mock_step_server, base_handshake):
    from urlab_client import MocapPose
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.read_mocap_pose_ok(
            body="mocap_target",
            pos=[0.1, 0.2, 0.3],
            quat=[0.7071, 0.0, 0.0, 0.7071],
        ))
        pose = client.runtime.read_mocap_pose("mocap_target")
        assert isinstance(pose, MocapPose)
        assert pose.body == "mocap_target"
        assert pose.pos == (0.1, 0.2, 0.3)
        assert pose.quat == (0.7071, 0.0, 0.0, 0.7071)
        req = mock_step_server.received[-1]
        assert req["op"] == "read_mocap_pose"
        assert req["body"] == "mocap_target"
    finally:
        client.close()


def test_get_contacts_unpacks_filter_and_results(mock_step_server, base_handshake):
    from urlab_client import Contact, ContactsResult
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.get_contacts_ok(
            n_contacts=2,
            truncated=False,
            contacts=[
                {
                    "geom1": "floor", "geom2": "foot_L",
                    "body1": "world", "body2": "robot_foot_L",
                    "pos": [0.1, 0.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    "dist": -0.001,
                    "force": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                },
                {
                    "geom1": "floor", "geom2": "foot_R",
                    "body1": "world", "body2": "robot_foot_R",
                    "pos": [-0.1, 0.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    "dist": -0.0005,
                    "force": [0.0, 0.0, 12.0, 0.0, 0.0, 0.0],
                },
            ],
        ))
        result = client.runtime.get_contacts(body1="world", max_contacts=8)
        assert isinstance(result, ContactsResult)
        assert result.n_contacts == 2
        assert result.truncated is False
        assert len(result.contacts) == 2
        c0 = result.contacts[0]
        assert isinstance(c0, Contact)
        assert c0.geom2 == "foot_L"
        assert c0.force == (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
        req = mock_step_server.received[-1]
        assert req["op"] == "get_contacts"
        assert req["max_contacts"] == 8
        assert req["filter"] == {"body1": "world"}
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Debug visualisation: draw_marker, draw_line, draw_box, clear_markers,
# set_overlay_text.
# ---------------------------------------------------------------------------


def test_draw_marker_optional_label_and_tag(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.draw_marker_ok())
        client.debug.draw_marker(
            (0.5, 0.0, 0.2), (1.0, 0.0, 0.0), ttl=2.5,
            label="grasp_target", tag="grasp",
        )
        req = mock_step_server.received[-1]
        assert req["op"] == "draw_marker"
        assert req["location"] == [0.5, 0.0, 0.2]
        assert req["color"] == [1.0, 0.0, 0.0]
        assert req["ttl"] == 2.5
        assert req["label"] == "grasp_target"
        assert req["tag"] == "grasp"

        # No label / no tag → those fields stay out of the wire payload.
        mock_step_server.replies.append(wr.draw_marker_ok())
        client.debug.draw_marker((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        req = mock_step_server.received[-1]
        assert "label" not in req
        assert "tag" not in req
        assert req["ttl"] == 0.0
    finally:
        client.close()


def test_draw_line_carries_thickness(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.draw_line_ok())
        client.debug.draw_line(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            ttl=-1.0, thickness=3.0,
        )
        req = mock_step_server.received[-1]
        assert req["op"] == "draw_line"
        assert req["from"] == [0.0, 0.0, 0.0]
        assert req["to"] == [1.0, 0.0, 0.0]
        assert req["ttl"] == -1.0
        assert req["thickness"] == 3.0
    finally:
        client.close()


def test_draw_box_includes_rotation_when_set(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.draw_box_ok())
        client.debug.draw_box(
            (0.0, 0.0, 0.5), (0.1, 0.1, 0.1), (1.0, 1.0, 0.0),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
        )
        req = mock_step_server.received[-1]
        assert req["op"] == "draw_box"
        assert req["half_extents"] == [0.1, 0.1, 0.1]
        assert req["rotation_quat"] == [0.0, 0.0, 0.0, 1.0]

        mock_step_server.replies.append(wr.draw_box_ok())
        client.debug.draw_box(
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
        )
        req = mock_step_server.received[-1]
        assert "rotation_quat" not in req
    finally:
        client.close()


def test_clear_markers_tag_optional(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.clear_markers_ok())
        client.debug.clear_markers()
        req = mock_step_server.received[-1]
        assert req["op"] == "clear_markers"
        assert "tag" not in req

        mock_step_server.replies.append(wr.clear_markers_ok())
        client.debug.clear_markers(tag="grasp")
        req = mock_step_server.received[-1]
        assert req["tag"] == "grasp"
    finally:
        client.close()


def test_set_overlay_text_passes_anchor(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.set_overlay_text_ok())
        client.debug.set_overlay_text("step 1/100", anchor="top_right")
        req = mock_step_server.received[-1]
        assert req["op"] == "set_overlay_text"
        assert req["text"] == "step 1/100"
        assert req["anchor"] == "top_right"
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Viewport: set_camera, get_camera, frame_actor, set_mode, track_actor,
# untrack. screenshot is intentionally absent (deferred to followups).
# ---------------------------------------------------------------------------


def test_set_camera_rejects_both_rotations(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        try:
            client.viewport.set_camera(
                (0.0, 0.0, 0.0),
                rotation_quat=(0.0, 0.0, 0.0, 1.0),
                rotation_euler=(0.0, 0.0, 0.0),
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        client.close()


def test_set_camera_round_trips_pose(mock_step_server, base_handshake):
    from urlab_client import CameraPose
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.set_camera_ok(
            location=[1.0, 2.0, 3.0],
            rotation_quat=[0.0, 0.0, 0.0, 1.0],
            rotation_euler=[0.0, 0.0, 90.0],
            fov=60.0,
        ))
        pose = client.viewport.set_camera(
            (1.0, 2.0, 3.0), rotation_euler=(0.0, 0.0, 90.0), fov=60.0,
        )
        assert isinstance(pose, CameraPose)
        assert pose.location == (1.0, 2.0, 3.0)
        assert pose.fov == 60.0
        req = mock_step_server.received[-1]
        assert req["op"] == "set_camera"
        assert req["location"] == [1.0, 2.0, 3.0]
        assert req["rotation_euler"] == [0.0, 0.0, 90.0]
        assert "rotation_quat" not in req
        assert req["fov"] == 60.0
    finally:
        client.close()


def test_get_camera_returns_typed_dataclass(mock_step_server, base_handshake):
    from urlab_client import CameraPose
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.get_camera_ok(
            location=[0.5, 0.0, 1.2],
            rotation_quat=[0.0, 0.0, 0.7071, 0.7071],
            rotation_euler=[0.0, 0.0, 90.0],
            fov=70.0,
        ))
        pose = client.viewport.get_camera()
        assert isinstance(pose, CameraPose)
        assert pose.location == (0.5, 0.0, 1.2)
        assert pose.fov == 70.0
        req = mock_step_server.received[-1]
        assert req["op"] == "get_camera"
    finally:
        client.close()


def test_frame_actor_passes_target_payload(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.frame_actor_ok())
        client.viewport.frame_actor("robot_a", by_name=True)
        req = mock_step_server.received[-1]
        assert req["op"] == "frame_actor"
        assert req["target"] == "robot_a"
        assert req["target_by"] == "actor_name"
    finally:
        client.close()


def test_viewport_set_mode_rejects_invalid_mode(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        try:
            client.viewport.set_mode("bogus")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

        mock_step_server.replies.append(wr.set_viewport_mode_ok(mode="wireframe"))
        mode = client.viewport.set_mode("wireframe")
        assert mode == "wireframe"
        req = mock_step_server.received[-1]
        assert req["op"] == "set_viewport_mode"
        assert req["mode"] == "wireframe"
    finally:
        client.close()


def test_track_actor_passes_offset_and_smoothing(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.track_actor_ok(
            tracked_actor_path="/Game/Levels/Test.PersistentLevel.robot",
        ))
        path = client.viewport.track_actor(
            "robot", offset=(0.0, -3.0, 1.5), smoothing=0.5,
        )
        assert path.endswith("robot")
        req = mock_step_server.received[-1]
        assert req["op"] == "track_actor"
        assert req["target"] == "robot"
        assert req["offset"] == [0.0, -3.0, 1.5]
        assert req["smoothing"] == 0.5
    finally:
        client.close()


def test_untrack_returns_was_tracking(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.untrack_ok(was_tracking=True))
        assert client.viewport.untrack() is True
        mock_step_server.replies.append(wr.untrack_ok(was_tracking=False))
        assert client.viewport.untrack() is False
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Extras: debug.draw_axes, runtime.list_keyframes, scene.spawn_grid.
# ---------------------------------------------------------------------------


def test_draw_arrow_carries_arrow_size_in_ue_cm(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.draw_arrow_ok())
        client.debug.draw_arrow(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            ttl=1.0, thickness=2.0, arrow_size=0.15,
        )
        req = mock_step_server.received[-1]
        assert req["op"] == "draw_arrow"
        assert req["from"] == [0.0, 0.0, 0.0]
        assert req["to"] == [1.0, 0.0, 0.0]
        assert req["thickness"] == 2.0
        # arrow_size goes onto the wire in MJ metres (plugin converts to UE cm).
        assert req["arrow_size"] == 0.15

        # arrow_size omitted -> field absent on wire (plugin auto-sizes).
        mock_step_server.replies.append(wr.draw_arrow_ok())
        client.debug.draw_arrow(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        )
        req = mock_step_server.received[-1]
        assert "arrow_size" not in req
    finally:
        client.close()


def test_draw_axes_rejects_both_rotations(mock_step_server, base_handshake):
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        try:
            client.debug.draw_axes(
                (0.0, 0.0, 0.0),
                rotation_quat=(0.0, 0.0, 0.0, 1.0),
                rotation_euler=(0.0, 0.0, 0.0),
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

        mock_step_server.replies.append(wr.draw_axes_ok())
        client.debug.draw_axes(
            (0.1, 0.2, 0.3), rotation_euler=(0.0, 0.0, 45.0),
            scale=0.5, ttl=2.0,
        )
        req = mock_step_server.received[-1]
        assert req["op"] == "draw_axes"
        assert req["location"] == [0.1, 0.2, 0.3]
        assert req["scale"] == 0.5
        assert req["ttl"] == 2.0
        assert req["rotation_euler"] == [0.0, 0.0, 45.0]
        assert "rotation_quat" not in req
    finally:
        client.close()


def test_list_keyframes_unpacks_per_keyframe_state(mock_step_server, base_handshake):
    from urlab_client import KeyframeInfo
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.append(wr.list_keyframes_ok(keyframes=[
            {
                "name": "home",
                "time": 0.0,
                "qpos": [0.0, 0.5, -0.3],
                "qvel": [0.0, 0.0, 0.0],
                "ctrl": [0.0, 0.0],
                "mocap_pos":  [],
                "mocap_quat": [],
            },
            {
                "name": "ready",
                "time": 0.5,
                "qpos": [0.1, 0.4, -0.2],
                "qvel": [],
                "ctrl": [],
                "mocap_pos":  [],
                "mocap_quat": [],
            },
        ]))
        kfs = client.runtime.list_keyframes()
        assert len(kfs) == 2
        assert all(isinstance(k, KeyframeInfo) for k in kfs)
        assert kfs[0].name == "home"
        assert kfs[0].qpos == [0.0, 0.5, -0.3]
        assert kfs[1].time == 0.5
        req = mock_step_server.received[-1]
        assert req["op"] == "list_keyframes"
    finally:
        client.close()


def test_spawn_grid_builds_handles_per_cell(mock_step_server, base_handshake):
    from urlab_client import URLabSpawnHandle
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        try:
            client.scene.spawn_grid("Robot.Robot_C", "grid", 0, 1)
            raise AssertionError("expected ValueError for count_x <= 0")
        except ValueError:
            pass

        mock_step_server.replies.append(wr.spawn_grid_ok(
            count=4,
            blueprint_class_path="/Game/MuJoCoImports/Robot.Robot_C",
            actors=[
                {"actor_id": "grid_0_0", "actor_name": "Robot_0_0",
                 "actor_path": "/Game/Levels/L.L:PersistentLevel.Robot_0_0",
                 "location": [0.0, 0.0, 0.0], "was_existing": False},
                {"actor_id": "grid_1_0", "actor_name": "Robot_1_0",
                 "actor_path": "/Game/Levels/L.L:PersistentLevel.Robot_1_0",
                 "location": [1.0, 0.0, 0.0], "was_existing": False},
                {"actor_id": "grid_0_1", "actor_name": "Robot_0_1",
                 "actor_path": "/Game/Levels/L.L:PersistentLevel.Robot_0_1",
                 "location": [0.0, 1.0, 0.0], "was_existing": False},
                {"actor_id": "grid_1_1", "actor_name": "Robot_1_1",
                 "actor_path": "/Game/Levels/L.L:PersistentLevel.Robot_1_1",
                 "location": [1.0, 1.0, 0.0], "was_existing": False},
            ],
            requires_pie_restart=False,
        ))
        handles = client.scene.spawn_grid(
            "Robot.Robot_C", "grid", 2, 2, spacing=(1.0, 1.0, 0.0),
        )
        assert set(handles.keys()) == {"grid_0_0", "grid_1_0", "grid_0_1", "grid_1_1"}
        assert all(isinstance(h, URLabSpawnHandle) for h in handles.values())
        assert handles["grid_1_1"].location == (1.0, 1.0, 0.0)
        # Every cell handle should have the blueprint class stamped in.
        assert all(
            h.blueprint_class_path == "/Game/MuJoCoImports/Robot.Robot_C"
            for h in handles.values()
        )

        req = mock_step_server.received[-1]
        assert req["op"] == "spawn_grid"
        assert req["count_x"] == 2 and req["count_y"] == 2
        assert req["base_actor_id"] == "grid"
        assert req["spacing"] == [1.0, 1.0, 0.0]
    finally:
        client.close()
