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

"""In-process launcher for mjlab's Unitree G1 tracking task.

Mirrors ``scripts/run_mjlab_demo.py``. Reuses the dashboard's
``URLabClient`` (mjlab's :class:`MjlabRunner` accepts an existing
client, no second connection). Reads the checkpoint + optional motion
file from a fixed demo dir; if either is missing the launcher logs a
clear path-not-found hint and bails before starting the worker thread.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

from .._helpers import push_sim_dt
from ..log import log
from ..state import STATE
from ..tabs.policy import LAUNCHERS


_DEFAULT_DEMO_DIR = r"C:/Users/jonat/Documents/mjlab_demo"
_DEFAULT_FREQ_HZ = 50.0


def _mjlab_launcher(entry: dict, step_mode_str: str, art_prefix: Optional[str]) -> threading.Thread:
    # Heavy deps deferred so the UI imports cleanly without mjlab / torch.
    from urlab_policy.adapters.mjlab import MjlabRunner

    client = STATE.client
    if client is None:
        raise RuntimeError("not connected")

    if step_mode_str != "stepped" or client.step_mode.value != "stepped":
        client.runtime.set_mode("stepped")
        log("mjlab: switched step mode to stepped")

    if not art_prefix:
        arts = list(client.articulations)
        if len(arts) != 1:
            raise RuntimeError(f"need articulation prefix (have: {arts})")
        art_prefix = arts[0]

    task_id = entry.get("task_id", "Mjlab-Tracking-Flat-Unitree-G1")

    checkpoint = os.path.join(_DEFAULT_DEMO_DIR, "ckpt.pt")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"mjlab checkpoint not found at {checkpoint}. "
            f"Place a tracked-policy ckpt.pt under {_DEFAULT_DEMO_DIR}/ or run "
            f"scripts/run_mjlab_demo.py --demo-dir <other> from the CLI."
        )

    candidate_motion = os.path.join(_DEFAULT_DEMO_DIR, "motion.npz")
    motion = candidate_motion if os.path.isfile(candidate_motion) else None
    if motion is None:
        log(f"mjlab: no motion.npz at {_DEFAULT_DEMO_DIR}, running without motion "
            f"(fine for velocity tasks, mandatory for tracking)")

    runner = MjlabRunner(
        client,
        task_id=task_id,
        checkpoint=checkpoint,
        motion_file=motion,
        articulation_prefix=art_prefix,
        device="cpu",
    )
    log(f"mjlab: runner ready task={task_id} art={art_prefix} "
        f"ckpt={os.path.basename(checkpoint)} sim_dt={runner.dt:.5f} "
        f"decim={runner.cfg_decimation}")

    # Last write to opt.timestep before the step loop -- UE recompile
    # reverts to the XML value.
    push_sim_dt(client, runner.dt, label="mjlab")

    pr = STATE.policy_run
    pr.stop_flag.clear()
    pr.step_count = 0
    pr.last_error = ""
    pr.started_at = time.monotonic()
    loop_dt = 1.0 / _DEFAULT_FREQ_HZ

    def _loop() -> None:
        try:
            while not pr.stop_flag.is_set():
                t0 = time.perf_counter()
                runner.step()
                pr.step_count += 1
                sleep_s = loop_dt - (time.perf_counter() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except Exception as exc:
            pr.last_error = f"{type(exc).__name__}: {exc}"
            log(f"mjlab crashed: {pr.last_error}", error=True)

    thread = threading.Thread(target=_loop, name="MjlabPolicy", daemon=True)
    thread.start()
    return thread


LAUNCHERS["mjlab_g1_tracking"] = _mjlab_launcher
