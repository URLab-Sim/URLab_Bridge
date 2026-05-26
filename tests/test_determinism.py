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

"""Determinism guarantees for the bridge wire path.

What this verifies:
  - `URLabClient.reset(seed=42)` ships the seed bit-identically over the
    wire on every call.
  - msgpack round-trip of qpos / qvel / sensor arrays is bit-exact (the
    bridge's local mirror sees the same bytes the server emitted).
  - `URLabEnv.reset(seed=...)` plumbs the seed straight through to the
    underlying `URLabClient.reset`.

What this does NOT verify (that's the live-UE integration test's job):
  - Physics-side determinism inside UE for a given seed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from urlab_client import URLabClient  # noqa: E402
from urlab_policy.adapters.robojudo.env import URLabEnv  # noqa: E402


def _client(port: int) -> URLabClient:
    return URLabClient(
        "tcp://127.0.0.1",
        step_mode="direct",
        step_port=port,
        rcv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )


def _step_reply(qpos):
    return {
        "op": "step_ok",
        "time": 0.01,
        "step": 1,
        "per_articulation": {
            "vx300s": {
                "qpos": list(qpos),
                "qvel": [0.0, 0.0],
                "ctrl": [0.0, 0.0],
                "act": [],
                "sensors": {},
            },
            "go2": {
                "qpos": [0.0],
                "qvel": [0.0],
                "ctrl": [0.0],
                "act": [],
                "sensors": {},
            },
        },
    }


def test_reset_seed_round_trip_bit_identical(mock_step_server, base_handshake):
    """Seed value lands on the wire exactly as passed."""
    mock_step_server.replies.extend([base_handshake, _step_reply([0.0, 0.0])])
    client = _client(mock_step_server.port)
    try:
        client.discover()
        client.reset(seed=12345)
        req = mock_step_server.received[-1]
        assert req["op"] == "reset"
        assert req["seed"] == 12345
        assert isinstance(req["seed"], int)
    finally:
        client.close()


def test_seed_replays_produce_identical_wire_payloads(mock_step_server, base_handshake):
    """Two separate reset(seed=X) calls produce identical request bytes."""
    mock_step_server.replies.extend(
        [base_handshake, _step_reply([0.0, 0.0]), _step_reply([0.0, 0.0])]
    )
    client = _client(mock_step_server.port)
    try:
        client.discover()
        client.reset(seed=42)
        client.reset(seed=42)
        # Compare the two reset requests
        resets = [r for r in mock_step_server.received if r["op"] == "reset"]
        assert len(resets) == 2
        assert resets[0] == resets[1]
    finally:
        client.close()


def test_msgpack_qpos_is_bit_exact(mock_step_server, base_handshake):
    """A weird-precision qpos round-trips bit-for-bit."""
    weird = [1.234567890123456e-05, -9.876543210987654e10]
    mock_step_server.replies.extend([base_handshake, _step_reply(weird)])
    client = _client(mock_step_server.port)
    try:
        client.discover()
        client.step()
        arm = client.articulations["vx300s"]
        # Bit-exact comparison via .view to bytes
        expected = np.asarray(weird, dtype=np.float64)
        np.testing.assert_array_equal(arm.qpos_array, expected)
        # And the actual bytes match
        assert arm.qpos_array.tobytes() == expected.tobytes()
    finally:
        client.close()


def test_env_reset_passes_seed_to_client(mock_step_server, base_handshake):
    """URLabEnv.reset(seed=...) goes straight through."""
    mock_step_server.replies.extend([base_handshake, _step_reply([0.0, 0.0])])
    client = _client(mock_step_server.port)
    try:
        client.discover()
        env = URLabEnv(client, space_mode="flat")
        env.reset(seed=7777)
        req = mock_step_server.received[-1]
        assert req["op"] == "reset"
        assert req["seed"] == 7777
    finally:
        client.close()
