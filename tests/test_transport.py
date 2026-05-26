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

"""End-to-end wire-level tests using the `mock_step_server` fixture.

Each test:
  1. Queues a reply on the mock server.
  2. Calls a URLabClient method.
  3. Asserts on both the reply handling AND the request the server saw.
"""

from __future__ import annotations

import numpy as np
import pytest

from urlab_client import URLabClient, URLabRPCError
from urlab_client.enums import StepMode

from . import wire_replies as wr


def _make_client(port: int, *, step_mode: str = "direct") -> URLabClient:
    return URLabClient(
        f"tcp://127.0.0.1",
        step_mode=step_mode,
        step_port=port,
        rcv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )


def test_hello_round_trip(mock_step_server, base_handshake):
    mock_step_server.replies.append(base_handshake)
    client = _make_client(mock_step_server.port)
    try:
        client.discover()
    finally:
        client.close()
    assert client.session_id == base_handshake["session_id"]
    assert "vx300s" in client.articulations
    assert len(mock_step_server.received) == 1
    req = mock_step_server.received[0]
    assert req["op"] == "hello"
    assert req["observations"] == "standard"
    # discover() also fetches the op schema via `meta`. That request
    # lands on the mock but is recorded separately so app-op tests can
    # use `received[]` without shifted indices.
    assert len(mock_step_server.received_meta) == 1


def test_error_reply_raises_rpcerror(mock_step_server):
    mock_step_server.replies.append(
        {"op": "error", "code": "version_mismatch", "message": "nope"}
    )
    client = _make_client(mock_step_server.port)
    try:
        with pytest.raises(URLabRPCError) as excinfo:
            client.discover()
        assert excinfo.value.code == "version_mismatch"
    finally:
        client.close()


def test_direct_step_sends_ctrl_and_absorbs_reply(mock_step_server, base_handshake):
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append(wr.step_ok(
        time=0.01, step=1,
        per_articulation={
            "vx300s": wr.per_articulation_block(
                qpos=[0.1, 0.2], qvel=[0.0, 0.0], ctrl=[0.5, 0.0],
                sensors={"waist_pos": [0.1], "tip_pos": [0.1, 0.0, 0.05]},
            ),
            "go2": wr.per_articulation_block(qpos=[0.0], qvel=[0.0], ctrl=[0.0]),
        },
    ))
    client = _make_client(mock_step_server.port, step_mode="direct")
    try:
        client.discover()
        client.articulations["vx300s"].set_ctrl({"waist": 0.5})
        reply = client.step(n_steps=5)
    finally:
        client.close()

    assert reply["op"] == "step_ok"
    assert client.sim_time == pytest.approx(0.01)
    assert client.step_count == 1
    vx = client.articulations["vx300s"]
    assert vx.sensors["waist_pos"].latest[0] == pytest.approx(0.1)
    assert vx.sensors["tip_pos"].latest.shape == (3,)

    # Request shape check
    assert mock_step_server.received[1]["op"] == "step"
    assert mock_step_server.received[1]["n_steps"] == 5
    per = mock_step_server.received[1]["per_articulation"]
    assert per["vx300s"]["ctrl"][0] == pytest.approx(0.5)


def test_twist_and_clocks_roundtrip(mock_step_server, base_handshake):
    """Per-articulation twist (ROS Twist-shape) + top-level sim_time / wall_time
    populate URLabArticulation.{twist_linear,twist_angular,actions,twist} and
    URLabClient.{sim_time_*,wall_time_*,recv_wall_time_ns}."""
    import numpy as np

    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append(wr.step_ok(
        time=0.02, step=2,
        sim_time={"sec": 0,         "nsec": 20_000_000},
        wall_time={"sec": 1745678912, "nsec": 345678901},
        per_articulation={
            "vx300s": wr.per_articulation_block(
                qpos=[0.0, 0.0], qvel=[0.0, 0.0], ctrl=[0.0, 0.0],
                twist={"linear":  [0.5, -0.25, 0.0],
                       "angular": [0.0, 0.0, 0.7]},
                actions=5,
            ),
            # no twist block -- twist fields stay zero
            "go2": wr.per_articulation_block(qpos=[0.0], qvel=[0.0], ctrl=[0.0]),
        },
    ))
    client = _make_client(mock_step_server.port, step_mode="direct")
    try:
        client.discover()
        client.step(n_steps=1)
    finally:
        client.close()

    # Top-level clocks (ROS-Time aligned)
    assert client.sim_time_sec == 0
    assert client.sim_time_nsec == 20_000_000
    assert client.wall_time_sec == 1745678912
    assert client.wall_time_nsec == 345678901
    # recv stamp is bridge-local time.time_ns(); just verify it was set non-zero
    assert client.recv_wall_time_ns > 0

    # Per-articulation twist + actions
    vx = client.articulations["vx300s"]
    assert np.allclose(vx.twist_linear,  [0.5, -0.25, 0.0])
    assert np.allclose(vx.twist_angular, [0.0,  0.0, 0.7])
    assert vx.actions == 5
    # Convenience 6-vec
    assert np.allclose(vx.twist, [0.5, -0.25, 0.0, 0.0, 0.0, 0.7])

    # Articulation without a twist block stays zero
    go2 = client.articulations["go2"]
    assert np.allclose(go2.twist_linear, [0.0, 0.0, 0.0])
    assert np.allclose(go2.twist_angular, [0.0, 0.0, 0.0])
    assert go2.actions == 0


def test_clocks_default_zero_before_first_reply(mock_step_server, base_handshake):
    """Clock fields stay at constructor defaults until a reply absorbs them.
    Older UE servers that don't emit the clock blocks shouldn't break the
    bridge -- just leave the fields at their previous value."""
    mock_step_server.replies.append(base_handshake)
    client = _make_client(mock_step_server.port)
    try:
        client.discover()
    finally:
        client.close()
    # base_handshake doesn't include clock blocks -> zeros after discover.
    assert client.sim_time_sec == 0
    assert client.sim_time_nsec == 0
    assert client.wall_time_sec == 0
    assert client.wall_time_nsec == 0


def test_reset_round_trip(mock_step_server, base_handshake):
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append(wr.step_ok())
    client = _make_client(mock_step_server.port)
    try:
        client.discover()
        client.reset(keyframe_name="home", seed=42)
    finally:
        client.close()
    req = mock_step_server.received[1]
    assert req["op"] == "reset"
    assert req["keyframe_name"] == "home"
    assert req["seed"] == 42


def test_set_mode_rpc(mock_step_server, base_handshake):
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append(wr.set_mode_ok(
        previous_mode="direct", current_mode="puppet"
    ))
    client = _make_client(mock_step_server.port)
    try:
        client.discover()
        current = client.runtime.set_mode("puppet")
    finally:
        client.close()
    assert current is StepMode.PUPPET
    assert client.step_mode is StepMode.PUPPET


def test_configure_controller_rpc(mock_step_server, base_handshake):
    mock_step_server.replies.append(base_handshake)
    mock_step_server.replies.append(wr.configure_controller_ok(
        articulation="vx300s",
        params={
            "kp": {"waist": 320.0, "shoulder": 280.0},
            "kv": {"waist": 20.0, "shoulder": 18.0},
            "torque_limit": {"waist": 35.0, "shoulder": 45.0},
            "default_kp": 100.0,
            "default_kv": 6.0,
            "default_torque_limit": 200.0,
        },
    ))
    client = _make_client(mock_step_server.port)
    try:
        client.discover()
        result = client.articulations["vx300s"].controller.set_gains(
            kp={"waist": 320.0}
        )
    finally:
        client.close()
    assert result["kp"]["waist"] == 320.0
    assert result["default_kv"] == 6.0

    sent = mock_step_server.received[1]
    assert sent["op"] == "configure_controller"
    assert sent["articulation"] == "vx300s"
    assert sent["params"]["kp"]["waist"] == 320.0


def test_recording_start_stop_save(mock_step_server, base_handshake):
    from pathlib import Path

    from urlab_client import (
        RecordingHandle,
        RecordingSummary,
    )

    mock_step_server.replies.extend([
        base_handshake,
        wr.recording_start_ok(name="ep_1", max_duration_s=3.4e38),
        wr.recording_stop_ok(name="ep_1", frame_count=100, sim_duration_s=1.0),
        wr.recording_save_ok(absolute_path="C:/tmp/ep_1.json"),
    ])
    client = _make_client(mock_step_server.port)
    try:
        client.discover()
        handle = client.recording.start()
        assert isinstance(handle, RecordingHandle)
        assert handle.name == "ep_1"
        assert client.recording.is_active
        summary = client.recording.stop()
        assert isinstance(summary, RecordingSummary)
        assert summary.frame_count == 100
        assert summary.sim_duration_s == 1.0
        assert not client.recording.is_active
        assert client.recording.frame_count == 100
        path = client.recording.save()
    finally:
        client.close()
    assert isinstance(path, Path)
    assert path == Path("C:/tmp/ep_1.json")
    assert client.recording.last_saved_path == path


def test_replay_play_shortcut(mock_step_server, base_handshake):
    from urlab_client import ReplaySession

    mock_step_server.replies.extend([
        base_handshake,
        wr.replay_load_ok(name="ep_1"),
        wr.replay_set_active_ok(),
        wr.replay_start_ok(active_session="ep_1", total_frames=50),
    ])
    client = _make_client(mock_step_server.port)
    try:
        client.discover()
        session = client.replay.play("C:/tmp/ep_1.json")
    finally:
        client.close()
    assert isinstance(session, ReplaySession)
    assert session.name == "ep_1"
    assert client.replay.active_session == "ep_1"
    ops = [r["op"] for r in mock_step_server.received]
    assert ops == ["hello", "replay_load", "replay_set_active", "replay_start"]


def test_puppet_step_n_steps_calls_mj_step(
    mock_step_server, base_handshake, mujoco_mod
):
    """n_steps > 0 in puppet mode should advance client.data via mj_step."""
    mock_step_server.replies.extend([
        base_handshake,
        wr.step_ok(time=0.005, step=5),
    ])
    client = _make_client(mock_step_server.port, step_mode="puppet")
    try:
        client.discover()
        # Drive a non-trivial state so mj_step has something to integrate
        client.data.qvel[:] = 0.5
        t0 = client.data.time
        client.step(n_steps=3)
        assert client.data.time > t0
    finally:
        client.close()
    sent = mock_step_server.received[1]
    assert sent["op"] == "step"
    assert sent["mode"] == "puppet"
    assert sent["n_steps"] == 3
    assert "qpos" in sent and "qvel" in sent


def test_puppet_step_zero_does_not_call_mj_step(
    mock_step_server, base_handshake, mujoco_mod
):
    """n_steps == 0 is the MJX / manual-state escape hatch: no mj_step,
    just ship whatever is already in client.data."""
    mock_step_server.replies.extend([
        base_handshake,
        wr.step_ok(),
    ])
    client = _make_client(mock_step_server.port, step_mode="puppet")
    try:
        client.discover()
        # Set a recognisable state and confirm it survives (mj_step would integrate)
        client.data.qpos[0] = 1.234
        t_before = client.data.time
        client.step(n_steps=0)
        assert client.data.qpos[0] == pytest.approx(1.234)
        assert client.data.time == t_before
    finally:
        client.close()
    sent = mock_step_server.received[1]
    assert sent["n_steps"] == 0
    assert sent["qpos"][0] == pytest.approx(1.234)


def test_step_reply_mirrors_state_into_local_mjdata_direct_mode(
    mock_step_server, base_handshake, mujoco_mod
):
    mock_step_server.replies.extend([
        base_handshake,
        wr.step_ok(
            time=0.5, step=25,
            per_articulation={
                "vx300s": wr.per_articulation_block(qpos=[0.7, -0.3], qvel=[0.0, 0.0], ctrl=[0.0, 0.0]),
                "go2":    wr.per_articulation_block(qpos=[0.1], qvel=[0.0], ctrl=[0.0]),
            },
        ),
    ])
    client = _make_client(mock_step_server.port, step_mode="direct")
    try:
        client.discover()
        client.step(n_steps=1)
        # The local mirror should now hold the UE-authoritative qpos for
        # each articulation's joints at their qpos_offset.
        vx = client.articulations["vx300s"]
        waist = vx.joints["waist"]
        shoulder = vx.joints["shoulder"]
        assert client.data.qpos[waist.qpos_offset] == pytest.approx(0.7)
        assert client.data.qpos[shoulder.qpos_offset] == pytest.approx(-0.3)
        assert client.data.time == pytest.approx(0.5)
    finally:
        client.close()


def test_set_sim_options_returns_typed_simoptions(mock_step_server, base_handshake):
    """runtime.set_sim_options returns a SimOptions dataclass — not a
    raw dict — and the echoed timestep mirrors into client.model.opt."""
    from urlab_client.results import SimOptions

    mock_step_server.replies.extend([
        base_handshake,
        wr.set_sim_options_ok(options={
            "timestep": 0.002,
            "iterations": 100,
            "gravity": [0.0, 0.0, -9.81],
            "integrator": "implicitfast",
            "solver": "newton",
        }),
    ])
    client = _make_client(mock_step_server.port, step_mode="direct")
    try:
        client.discover()
        result = client.runtime.set_sim_options(timestep=0.002)
        assert isinstance(result, SimOptions)
        assert result.timestep == pytest.approx(0.002)
        assert result.iterations == 100
        assert result.gravity == (0.0, 0.0, -9.81)
        assert result.integrator == "implicitfast"
        assert result.solver == "newton"

        # Local mirror reflects the new timestep too.
        assert client.model.opt.timestep == pytest.approx(0.002)
    finally:
        client.close()


def test_set_sim_options_rejects_empty_call(mock_step_server, base_handshake):
    """set_sim_options() with no fields raises rather than sending an
    empty payload to the server."""
    mock_step_server.replies.append(base_handshake)
    client = _make_client(mock_step_server.port, step_mode="direct")
    try:
        client.discover()
        with pytest.raises(ValueError, match="at least one field"):
            client.runtime.set_sim_options()
    finally:
        client.close()
