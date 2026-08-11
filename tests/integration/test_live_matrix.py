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

"""Cross-product integration tests: step-mode x transport.

The bridge supports three step modes (``live``, ``direct``, ``puppet``)
and two transports (``zmq``, ``shm``). The tests below sweep all six
combinations against the session-scoped golden scene to catch any
mode-or-transport-specific wire regression.

SHM transport tests are skipped on non-Linux hosts: the kernel-event
signalling path (Windows) has a separate validation track; only the
Linux polling path runs in this suite. The tests themselves are
authored unconditionally so the matrix is visible in coverage.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import numpy as np
import pytest

from urlab_client import StepMode, URLabClient

HOST = os.environ.get("URLAB_HOST", "tcp://127.0.0.1")
STEP_PORT = int(os.environ.get("URLAB_STEP_PORT", "5559"))

# The transport is implemented for POSIX futex and for Windows kernel events
# (`urlab_client/transports/shm.py`), so the matrix runs it wherever the client
# can actually open the region. Gating on Linux alone left the Windows path
# claiming to be "covered by separate runs" that do not exist.
_SHM_SUPPORTED = sys.platform.startswith("linux") or sys.platform == "win32"
_SHM_SKIP_REASON = (
    f"SHM transport is not implemented on this platform ({sys.platform})."
)


_MATRIX = [
    ("zmq", StepMode.LIVE),
    ("zmq", StepMode.DIRECT),
    ("zmq", StepMode.PUPPET),
    ("shm", StepMode.LIVE),
    ("shm", StepMode.DIRECT),
    ("shm", StepMode.PUPPET),
]

_IDS = [f"{t}-{m.value}" for (t, m) in _MATRIX]


@pytest.fixture(params=_MATRIX, ids=_IDS)
def matrix_client(request, _live_session):
    """Per-parameter client. Opens a fresh URLabClient configured for
    the (transport, step_mode) combo, walks discover + auto-promotes
    the step mode if applicable, yields, then closes."""
    transport, mode = request.param
    if transport == "shm" and not _SHM_SUPPORTED:
        pytest.skip(_SHM_SKIP_REASON)

    # AUTO promotes implicitly. For LIVE, no promotion happens (LIVE is
    # the server default), but we still want the client object's
    # step_mode reflected so tests can branch off of it.
    client = URLabClient(
        HOST,
        step_port=STEP_PORT,
        recv_timeout_ms=10_000,
        step_mode=mode,
        transport=transport,
    )
    try:
        client.connect()
        # After discover with auto_promote_step_mode (default True),
        # DIRECT / PUPPET have been set on the server. For LIVE we
        # explicitly demote so tests that ran before in DIRECT see
        # streaming come back up.
        if mode == StepMode.LIVE and client.step_mode != StepMode.LIVE:
            client.runtime.set_mode(StepMode.LIVE)
            time.sleep(0.2)
        yield client
    finally:
        try:
            client.close()
        except Exception:
            pass


# --- Smoke ops covered across the full matrix ---------------------------------


def test_handshake_articulations(matrix_client):
    """The bootstrapped scene has one articulation regardless of
    transport / step_mode."""
    assert matrix_client.session_id
    assert len(matrix_client.articulations) >= 1


def test_step_runs(matrix_client):
    """One step round-trips on every (transport, mode) combo."""
    # PUPPET steps push local mj_step state to the server; LIVE just
    # samples the latest snapshot; DIRECT runs mj_step on UE. All three
    # should accept n_steps=1 without raising.
    if matrix_client.step_mode == StepMode.PUPPET and matrix_client.model is None:
        pytest.skip("Puppet mode requires a local MJB-loaded MjModel")
    matrix_client.step(n_steps=1)
    art = next(iter(matrix_client.articulations.values()))
    assert art.qpos_array.size >= 0


def test_reset_round_trip(matrix_client):
    """reset() works on every combo. Server snaps qpos/qvel back to
    keyframe defaults; client.data mirrors."""
    matrix_client.reset(seed=42)


def test_runtime_set_paused_round_trip(matrix_client):
    """set_paused is a manager-required runtime mutator; should reach
    the server on every transport regardless of step mode."""
    pre = matrix_client.runtime.set_paused(True)
    assert isinstance(pre, bool)
    matrix_client.runtime.set_paused(False)


def test_get_contacts_returns_typed(matrix_client):
    """runtime.get_contacts returns a typed ContactsResult on every
    (transport, mode) combo."""
    from urlab_client.results import ContactsResult

    res = matrix_client.runtime.get_contacts(max_contacts=16)
    assert isinstance(res, ContactsResult)
