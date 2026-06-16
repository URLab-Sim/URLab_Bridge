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

"""End-to-end tests for the `meta` schema fetch + namespace synthesis
on URLabClient.

These tests drive a real ZMQ round-trip through the mock step server,
script explicit meta_ok replies, and verify both pre- and post-discover
behaviour of `client.scene.<op>` / `client.sim.<op>` etc."""

from __future__ import annotations

import pytest

from urlab_client import URLabClient, URLabRPCError


def _make_client(port: int) -> URLabClient:
    return URLabClient(
        "tcp://127.0.0.1",
        step_mode="auto",
        step_port=port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )


def test_discover_fetches_meta_and_populates_ops(mock_step_server, base_handshake):
    """Hello+meta land on the mock; meta returns an op table; client
    stores it on `_ops_meta` keyed by op name."""
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append({
        "op": "meta_ok",
        "ops": [
            {"name": "spawn_actor",  "category": "editor_only",      "namespace": "scene"},
            {"name": "remove_actor","category": "editor_only",      "namespace": "scene"},
            {"name": "step",         "category": "manager_required", "namespace": "sim"},
            {"name": "begin_pie",    "category": "editor_only",      "namespace": "runtime"},
        ],
    })
    client = _make_client(mock_step_server.port)
    try:
        client.connect()
    finally:
        client.close()

    assert set(client._ops_meta.keys()) == {
        "spawn_actor", "remove_actor", "step", "begin_pie",
    }
    assert client._ops_meta["spawn_actor"]["namespace"] == "scene"
    assert client._ops_meta["step"]["category"] == "manager_required"


def test_discover_tolerates_missing_meta(mock_step_server, base_handshake):
    """A server that doesn't implement `meta` replies `unknown_op`.
    The client must accept that gracefully and leave `_ops_meta`
    empty — hand-written wrappers still work."""
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append({
        "op": "error",
        "code": "unknown_op",
        "message": "Unknown op 'meta'",
    })
    client = _make_client(mock_step_server.port)
    try:
        client.connect()
    finally:
        client.close()
    assert client._ops_meta == {}


def test_namespace_synthesizes_unknown_op(mock_step_server, base_handshake):
    """A server-only op (no hand-written wrapper on the bridge) becomes
    callable via `client.<namespace>.<op>(**kwargs)` because meta carries
    the schema. Verifies the synthesized callable hits the wire with
    `op=<name>` and the supplied kwargs as payload."""
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append({
        "op": "meta_ok",
        "ops": [
            {"name": "future_scene_op", "category": "editor_only", "namespace": "scene"},
        ],
    })
    # Reply for the synthesized op call.
    mock_step_server.replies.append({"op": "future_scene_op_ok", "result": 42})

    client = _make_client(mock_step_server.port)
    try:
        client.connect()
        result = client.scene.future_scene_op(some_arg="hi", n=7)
    finally:
        client.close()

    assert result["op"] == "future_scene_op_ok"
    assert result["result"] == 42
    sent = mock_step_server.received[-1]
    assert sent["op"] == "future_scene_op"
    assert sent["some_arg"] == "hi"
    assert sent["n"] == 7
    assert sent["session_id"] == base_handshake["session_id"]


def test_namespace_proxies_to_handwritten_method(mock_step_server, base_handshake):
    """When a concrete namespace class (e.g. `_SceneNamespace`) defines a
    method by name (e.g. `import_xml`), the explicit method lookup wins
    over `__getattr__` synthesis — so custom marshalling, dataclass
    unwrapping, target_payload helpers, etc. all stay in effect when
    callers reach the op via the namespace surface."""
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append({
        "op": "meta_ok",
        "ops": [
            {"name": "import_xml", "category": "editor_only", "namespace": "scene"},
        ],
    })
    mock_step_server.replies.append({"op": "import_xml_ok", "session_id": base_handshake["session_id"]})

    client = _make_client(mock_step_server.port)
    try:
        client.connect()
        client.scene.import_xml("/tmp/whatever.xml")
    finally:
        client.close()

    sent = mock_step_server.received[-1]
    assert sent["op"] == "import_xml"
    assert sent["path"] == "/tmp/whatever.xml"
    # Hand-written wrapper passes through the path field — synthesised
    # version would have used `path=` kwarg-only and emitted exactly that.


def test_namespace_raises_for_op_in_other_namespace(mock_step_server, base_handshake):
    """`client.scene.step` must raise — `step` lives in the `sim` namespace
    per meta, so the scene proxy refuses to call it. Prevents accidental
    cross-namespace leakage when someone confuses op categories."""
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append({
        "op": "meta_ok",
        "ops": [
            {"name": "step", "category": "manager_required", "namespace": "sim"},
        ],
    })
    client = _make_client(mock_step_server.port)
    try:
        client.connect()
        with pytest.raises(AttributeError):
            client.scene.step
    finally:
        client.close()


def test_sim_namespace_renames_pie_jargon(mock_step_server, base_handshake):
    """client.sim.start / sim.stop / sim.status are the canonical
    Python surface for what the wire calls begin_pie / stop_pie /
    pie_status. Verify each delegates to the right wire op."""
    from urlab_client import PIEState

    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append({
        "op": "begin_pie_ok", "state": "compile_failed",
        "compile_error": "synthetic test", "handshake_payload": None,
    })
    mock_step_server.replies.append({"op": "stop_pie_ok"})
    mock_step_server.replies.append({
        "op": "pie_status_ok", "state": "off", "compile_error": "",
    })

    client = _make_client(mock_step_server.port)
    try:
        client.connect()
        result_start = client.sim.start(raise_on_failure=False)
        client.sim.stop()
        result_status = client.sim.status()
    finally:
        client.close()

    ops = [r["op"] for r in mock_step_server.received]
    # hello + begin_pie + stop_pie + pie_status (meta is filtered out)
    assert ops == ["hello", "begin_pie", "stop_pie", "pie_status"]
    assert result_start.state == PIEState.COMPILE_FAILED
    assert result_status.state == PIEState.OFF


def test_namespace_dir_lists_namespace_ops_only(mock_step_server, base_handshake):
    """`dir(client.scene)` should expose only ops in the `scene` namespace
    so IDE introspection / tab-completion stays scoped."""
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append({
        "op": "meta_ok",
        "ops": [
            {"name": "spawn_actor", "category": "editor_only",      "namespace": "scene"},
            {"name": "step",        "category": "manager_required", "namespace": "sim"},
            {"name": "begin_pie",   "category": "editor_only",      "namespace": "runtime"},
        ],
    })
    client = _make_client(mock_step_server.port)
    try:
        client.connect()
        scene_attrs = dir(client.scene)
    finally:
        client.close()
    assert "spawn_actor" in scene_attrs
    assert "step" not in scene_attrs
    assert "begin_pie" not in scene_attrs
