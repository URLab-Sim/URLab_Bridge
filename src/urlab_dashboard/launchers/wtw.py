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

"""Walk-These-Ways launcher (Go2 12-DoF).

Mirrors what ``scripts/run_wtw_demo.py`` does, minus the connection
plumbing — the UI already owns the ``URLabClient``. Step mode is forced
to ``direct`` (the script's setting) and ``sim_dt=0.002`` is pushed
before stepping so ``NativePolicyRunner`` infers the right decimation.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .._helpers import push_sim_dt
from ..log import log
from ..state import STATE
from ..tabs.policy import LAUNCHERS


_DEFAULT_FREQ_HZ = 50.0
_DEFAULT_SIM_DT = 0.002


def _wtw_launcher(entry: dict, step_mode_str: str, art_prefix: Optional[str]) -> threading.Thread:
    # Heavy deps deferred so the UI imports cleanly without RoboJuDo / torch.
    from urlab_policy.configs.go2.policy.go2_wtw_policy_cfg import Go2WtwPolicyCfg
    from urlab_policy.native_runner import NativePolicyRunner
    from urlab_policy.policies.wtw_policy import WalkTheseWaysPolicy

    client = STATE.client
    if client is None:
        raise RuntimeError("not connected")

    # WTW is a stepped policy; force direct mode if the user picked something else.
    if step_mode_str != "direct" or client.step_mode.value != "direct":
        try:
            client.runtime.set_mode("direct")
            log("WTW: switched step mode to direct")
        except Exception as exc:
            raise RuntimeError(f"failed to switch to direct mode: {exc}")

    if art_prefix is None or art_prefix not in client.articulations:
        arts = list(client.articulations)
        if len(arts) != 1:
            raise RuntimeError(
                f"need an articulation prefix (have: {arts})"
            )
        art_prefix = arts[0]
    art = client.articulations[art_prefix]

    cfg = Go2WtwPolicyCfg()
    policy = WalkTheseWaysPolicy(cfg_policy=cfg, device="cpu")
    runner = NativePolicyRunner(client=client, art=art, policy=policy, push_gains=True)

    # Last write to opt.timestep before the step loop -- UE recompile
    # reverts to the XML value.
    push_sim_dt(client, _DEFAULT_SIM_DT, label="WTW")

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
            log(f"WTW crashed: {pr.last_error}", error=True)
        # NOTE: do not call runner.close() — that closes the client and
        # the UI is still using it.

    thread = threading.Thread(target=_loop, name="WTWPolicy", daemon=True)
    thread.start()
    return thread


LAUNCHERS["go2_wtw"] = _wtw_launcher
