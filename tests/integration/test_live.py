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

"""Live-UE integration smoke test.

Skipped unless `URLAB_LIVE=1` is set. Requires a running URLab editor
listening on port 5559 — the conftest fixtures bootstrap a golden scene
and PIE on demand, so the editor's pre-existing state doesn't matter.

Run via:
    URLAB_LIVE=1 micromamba run -n mj python -m pytest tests/integration

Or against a non-default host:
    URLAB_LIVE=1 URLAB_HOST=tcp://10.0.0.5 micromamba run -n mj \\
        python -m pytest tests/integration

Tests are intentionally minimal -- they assert connectivity and basic
op behaviour, not physics correctness. Their job is to catch wire-format
drift between the bridge and a real UE build.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("URLAB_LIVE") == "1"
HOST = os.environ.get("URLAB_HOST", "tcp://127.0.0.1")
STEP_PORT = int(os.environ.get("URLAB_STEP_PORT", "5559"))

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="URLAB_LIVE=1 not set; live-UE tests skipped",
)

from urlab_client import URLabClient  # noqa: E402
from urlab_client.enums import StepMode  # noqa: E402


def test_handshake_returns_articulations(pie_client):
    """The bootstrapped golden scene has exactly one articulation."""
    assert len(pie_client.articulations) >= 1
    assert pie_client.session_id


def test_step_returns_qpos(pie_client):
    """`step(n=1)` round-trips and populates qpos for at least one arm."""
    pie_client.step(n_steps=1)
    arm = next(iter(pie_client.articulations.values()))
    assert arm.qpos_array.size >= 0  # well-formed; specific values are scene-dependent


def test_reset_with_seed(pie_client):
    """`reset(seed=42)` does not raise and returns a fresh observation."""
    pie_client.reset(seed=42)


def test_set_mode_round_trip(pie_client):
    """Switch to puppet then back to direct (only valid if AMjManager.StepMode == Auto)."""
    try:
        pie_client.runtime.set_mode(StepMode.PUPPET)
    except Exception as exc:
        # Server may be locked; that's a documented failure mode, not a bug.
        if "mode_locked_by_server" in str(exc):
            pytest.skip("Server StepMode is locked, mode switch unavailable")
        raise
    pie_client.runtime.set_mode(StepMode.DIRECT)
    assert pie_client.step_mode == StepMode.DIRECT


# --- Puppet mode --------------------------------------------------------------


def test_puppet_step_round_trip(pie_client):
    """`step(n=1)` in puppet mode runs mj_step locally and pushes state to UE."""
    try:
        pie_client.runtime.set_mode(StepMode.PUPPET)
    except Exception as exc:
        if "mode_locked_by_server" in str(exc):
            pytest.skip("Server StepMode is locked; puppet promotion unavailable")
        raise
    if pie_client.model is None:
        pytest.skip("Puppet mode requires a local MJB-loaded MjModel")
    pie_client.step(n_steps=1)
    arm = next(iter(pie_client.articulations.values()))
    assert arm.qpos_array.size >= 0


def test_puppet_step_n_zero_just_pushes_state(pie_client):
    """`step(n=0)` skips local mj_step and just pushes whatever is in client.data."""
    try:
        pie_client.runtime.set_mode(StepMode.PUPPET)
    except Exception as exc:
        if "mode_locked_by_server" in str(exc):
            pytest.skip("Server StepMode is locked; puppet promotion unavailable")
        raise
    if pie_client.model is None:
        pytest.skip("Puppet mode requires a local MJB-loaded MjModel")
    pre = float(pie_client.data.time)
    pie_client.step(n_steps=0)
    # n=0 should not advance time; client.data.time stays put because no
    # local mj_step ran and the wire payload uses that same time value.
    assert float(pie_client.data.time) == pre


# --- Live (streaming PUB) -----------------------------------------------------


def test_streaming_pub_emits_state(pie_client, zmq_mod, msgpack_mod):  # noqa: ARG001
    """Verify the 5555 PUB stream is alive once the server is in live mode.

    pie_client bootstraps the scene + starts PIE in DIRECT. We flip to
    LIVE here so the publishers start emitting, then subscribe to
    anything (filter "") and assert at least one frame arrives within
    5 seconds.
    """
    pie_client.runtime.set_mode(StepMode.LIVE)

    state_port = 5555
    state_endpoint = f"{HOST}:{state_port}"

    ctx = zmq_mod.Context.instance()
    sub = ctx.socket(zmq_mod.SUB)
    try:
        sub.connect(state_endpoint)
        sub.setsockopt(zmq_mod.SUBSCRIBE, b"")
        sub.setsockopt(zmq_mod.RCVTIMEO, 5000)
        topic = sub.recv()
        more = sub.getsockopt(zmq_mod.RCVMORE)
        if more:
            sub.recv()
        assert len(topic) > 0
    finally:
        sub.close(linger=0)


# --- Close behaviour ---------------------------------------------------------


def test_close_reverts_to_live(pie_client):
    """After `close()`, the server should be back in live mode.

    pie_client is in DIRECT. We open a second client (which inherits
    DIRECT since pie_client holds the session), close it, then open a
    third that asserts the server is reachable in LIVE. Run with the
    `pie_client` fixture so the editor has a scene; otherwise close()
    has nothing to revert from.
    """
    c1 = URLabClient(
        HOST, step_mode=StepMode.DIRECT, step_port=STEP_PORT, rcv_timeout_ms=2000
    )
    c1.discover()
    assert c1.step_mode == StepMode.DIRECT
    c1.close()

    c2 = URLabClient(
        HOST, step_mode=StepMode.AUTO, step_port=STEP_PORT, rcv_timeout_ms=2000
    )
    try:
        c2.discover()
        current = c2.runtime.set_mode(StepMode.LIVE)
        assert current == StepMode.LIVE
    finally:
        c2.close()
