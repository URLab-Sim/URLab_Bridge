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

"""Tests for ``scene.apply_scene`` orchestration over the editor RPCs."""

from __future__ import annotations

import pytest

from urlab_client import URLabAsset, URLabClient, URLabRPCError, URLabSpawnHandle

from . import wire_replies as wr


def _make_client(port: int) -> URLabClient:
    return URLabClient(
        "tcp://127.0.0.1",
        step_mode="stepped",
        step_port=port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )


def _open_session(client, mock_step_server, base_handshake) -> None:
    mock_step_server.replies.append(base_handshake)
    client.connect()


def _spawn_reply(actor_id: str, *, actor_name: str | None = None) -> dict:
    return wr.spawn_actor_ok(
        actor_id=actor_id,
        actor_name=actor_name or f"{actor_id}_C_UAID_X",
        actor_path=f"/Game/Levels/myscene.myscene:PersistentLevel.{actor_id}",
        blueprint_class_path="/Game/MuJoCoImports/foo.foo_C",
    )


def test_apply_scene_load_succeeds_then_spawns(
    mock_step_server, base_handshake, tmp_path
):
    """Happy path: load_level succeeds, no create_level call."""
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.extend([
            wr.load_level_ok(level_path="/Game/Levels/myscene"),
            wr.import_xml_ok(blueprint_class_path="/Game/MuJoCoImports/foo.foo_C", blueprint_short_name="foo", imported_now=False),
            _spawn_reply("robot_a"),
            _spawn_reply("robot_b"),
            wr.save_level_ok(level_path="/Game/Levels/myscene"),
        ])
        out = client.scene.apply_scene(
                        "myscene",
            [
                URLabAsset("robot_a", str(xml), location=(0.0, 0.0, 0.5)),
                URLabAsset("robot_b", str(xml), location=(1.0, 0.0, 0.5)),
            ],
        )
    finally:
        client.close()

    ops = [r["op"] for r in mock_step_server.received]
    assert ops == ["hello", "load_level", "import_xml", "spawn_actor", "spawn_actor", "save_level"]
    assert set(out.keys()) == {"robot_a", "robot_b"}
    assert all(isinstance(v, URLabSpawnHandle) for v in out.values())


def test_apply_scene_falls_back_to_create_when_load_fails(
    mock_step_server, base_handshake, tmp_path
):
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.extend([
            wr.error("load_level_failed", "no such level"),
            wr.create_level_ok(level_path="/Game/Levels/myscene"),
            wr.import_xml_ok(blueprint_class_path="/Game/MuJoCoImports/foo.foo_C", blueprint_short_name="foo", imported_now=True),
            _spawn_reply("robot_a"),
            wr.save_level_ok(level_path="/Game/Levels/myscene"),
        ])
        client.scene.apply_scene("myscene", [URLabAsset("robot_a", str(xml))])
    finally:
        client.close()

    ops = [r["op"] for r in mock_step_server.received]
    assert ops == ["hello", "load_level", "create_level", "import_xml", "spawn_actor", "save_level"]


def test_apply_scene_dedupes_xml_imports(mock_step_server, base_handshake, tmp_path):
    """Two URLabAssets with the same xml path should call import_xml once."""
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.extend([
            wr.load_level_ok(level_path="/Game/Levels/myscene"),
            wr.import_xml_ok(blueprint_class_path="/Game/MuJoCoImports/foo.foo_C", blueprint_short_name="foo", imported_now=True),
            _spawn_reply("a"),
            _spawn_reply("b"),
            _spawn_reply("c"),
            wr.save_level_ok(level_path="/Game/Levels/myscene"),
        ])
        client.scene.apply_scene(
                        "myscene",
            [
                URLabAsset("a", str(xml)),
                URLabAsset("b", str(xml)),
                URLabAsset("c", str(xml)),
            ],
        )
    finally:
        client.close()

    ops = [r["op"] for r in mock_step_server.received]
    assert ops.count("import_xml") == 1
    assert ops.count("spawn_actor") == 3


def test_apply_scene_save_false_skips_save_level(
    mock_step_server, base_handshake, tmp_path
):
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.extend([
            wr.load_level_ok(level_path="/Game/Levels/myscene"),
            wr.import_xml_ok(blueprint_class_path="/Game/MuJoCoImports/foo.foo_C", blueprint_short_name="foo", imported_now=True),
            _spawn_reply("robot_a"),
        ])
        client.scene.apply_scene("myscene", [URLabAsset("robot_a", str(xml))], save=False)
    finally:
        client.close()
    ops = [r["op"] for r in mock_step_server.received]
    assert "save_level" not in ops


def test_apply_scene_propagates_spawn_failure(
    mock_step_server, base_handshake, tmp_path
):
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    client = _make_client(mock_step_server.port)
    try:
        _open_session(client, mock_step_server, base_handshake)
        mock_step_server.replies.extend([
            wr.load_level_ok(level_path="/Game/Levels/myscene"),
            wr.import_xml_ok(blueprint_class_path="/Game/MuJoCoImports/foo.foo_C", blueprint_short_name="foo", imported_now=True),
            wr.error("spawn_failed", "bad pose"),
        ])
        with pytest.raises(URLabRPCError) as exc_info:
            client.scene.apply_scene("myscene", [URLabAsset("robot_a", str(xml))])
    finally:
        client.close()
    assert exc_info.value.code == "spawn_failed"
