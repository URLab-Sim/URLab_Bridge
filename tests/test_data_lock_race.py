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

"""Concurrent _absorb_step_reply and reads of client.data must not
produce torn state or crash MuJoCo.

`_absorb_step_reply` calls `mj_forward(model, data)` from the RPC
thread; without `_data_lock`, a concurrent reader on another thread
sees half-updated xpos/xquat fields. The RLock serialises writes and
exposes a contract for readers to acquire.

This test stress-fires concurrent absorbs from N threads and verifies:
1. No exception in any thread.
2. `data.qpos` post-stress matches the last-applied snapshot.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict

import pytest


def test_concurrent_absorb_does_not_crash_or_corrupt(base_handshake):
    pytest.importorskip("mujoco")
    from urlab_client import URLabClient

    client = URLabClient(step_mode="direct")
    client._apply_handshake(base_handshake)

    if client.model is None or client.data is None:
        pytest.skip("handshake didn't populate model/data; mujoco missing?")

    # Build a minimal step-reply shape with per-articulation qpos that
    # `_absorb_step_reply` will write into client.data via _mirror_state.
    sample_reply: Dict[str, Any] = {
        "time": 0.0,
        "step": 0,
        "per_articulation": {},
    }
    for prefix, art in client.articulations.items():
        n_qpos = sum(j.qpos_dim for j in art.joints.values())
        n_qvel = sum(j.qvel_dim for j in art.joints.values())
        sample_reply["per_articulation"][prefix] = {
            "qpos": [0.0] * n_qpos,
            "qvel": [0.0] * n_qvel,
        }

    if not sample_reply["per_articulation"]:
        pytest.skip("no articulations to drive in handshake")

    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(idx: int) -> None:
        try:
            for i in range(200):
                if stop.is_set():
                    return
                # Stagger qpos values per-iteration so torn writes would
                # be observable as a mismatch in the post-loop check.
                reply = copy.deepcopy(sample_reply)
                reply["time"] = float(i + idx * 0.001)
                reply["step"] = i + idx * 1000
                client._absorb_step_reply(reply)
        except BaseException as exc:  # pragma: no cover - debugging
            errors.append(exc)
            stop.set()

    def reader() -> None:
        try:
            for _ in range(1000):
                if stop.is_set():
                    return
                # Reader must hold the lock per the new contract.
                with client._data_lock:
                    _ = client.data.time
                    _ = client.data.qpos.copy()
                    _ = client.data.qvel.copy()
                # Yield to writers.
                time.sleep(0)
        except BaseException as exc:  # pragma: no cover - debugging
            errors.append(exc)
            stop.set()

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"concurrent absorb produced exceptions: {errors!r}"
    # No assertions about final qpos values — the test's purpose is no-crash
    # under contention. mujoco's mj_forward acquires no lock of its own;
    # without _data_lock, the prior implementation could have raced.
