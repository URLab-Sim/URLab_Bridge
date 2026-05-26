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

"""In-process launchers for the RoboJuDo G1 policy entries.

Each entry below maps to a RoboJuDo pipeline cfg name (registered via
``@cfg_registry.register`` in ``robojudo.config.g1.g1_cfg`` /
``g1_loco_mimic_cfg``). The launcher reuses the dashboard's existing
``URLabClient`` via ``URLabRoboJuDoEnvCfg.existing_client`` -- UE's
bridge dispatcher tracks one active session, so a second client would
session-expire the first.

Pipeline cfg names taken from ``RoboJuDo/robojudo/config/g1/g1_cfg.py``
and ``g1_loco_mimic_cfg.py``. If a name turns out to be wrong, the
launcher's exception bubbles up to the Policy tab's status line, and
the canonical CLI script (``scripts/run_*.py``) is still available.

PHC-required policies (beyondmimic_dance, h2h, amo, twist_tracker)
need the RoboJuDo PHC submodule installed; missing PHC produces the
real ImportError at launch time.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

from .._helpers import push_sim_dt
from .._policy_control import (
    build_handle_from_pipeline,
    swap_keyboard_ctrl_with_ui,
)
from ..log import log
from ..state import STATE
from ..tabs.policy import LAUNCHERS


_DEFAULT_FREQ_HZ = 50.0
_DEFAULT_SIM_DT = 0.002
_DEFAULT_DECIMATION = 10


# Maps registry key -> (RoboJuDo pipeline cfg name, optional policy override).
#
# Pipeline cfg names live in RoboJuDo/robojudo/config/g1/. Each is registered
# via @cfg_registry.register and held as a dataclass with a `policy` field
# (single-policy pipelines) or `loco_policy` + `mimic_policies` (locomimic
# pipelines).
#
# `policy_override` is None when the pipeline's default policy already matches
# the registry entry; otherwise it's the import path to the policy cfg class
# that should replace the pipeline's default `policy`.
CFG_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "unitree_12dof": {
        "cfg": "g1",                              # G1UnitreePolicyCfg is the default
        "policy_override": None,
        "dofs_29": False,
    },
    "unitree_wo_gait": {
        "cfg": "g1",                              # same pipeline, different policy
        "policy_override": "robojudo.config.g1.policy.g1_unitree_policy_cfg.G1UnitreeWoGaitPolicyCfg",
        "dofs_29": True,
    },
    "smooth": {
        "cfg": "g1",
        "policy_override": "robojudo.config.g1.policy.g1_smooth_policy_cfg.G1SmoothPolicyCfg",
        "dofs_29": True,
    },
    "beyondmimic_dance": {
        "cfg": "g1_locomimic_beyondmimic",        # loco-mimic pipeline (per scripts/run_beyondmimic.py)
        "policy_override": None,
        "dofs_29": True,
    },
    "h2h": {
        "cfg": "g1_h2h",
        "policy_override": None,
        "dofs_29": True,
    },
    "amo": {
        "cfg": "g1_asap",                         # AMO ships through the ASAP pipeline
        "policy_override": None,
        "dofs_29": True,
    },
    "twist_tracker": {
        "cfg": "g1_twist",
        "policy_override": None,
        "dofs_29": False,
    },
}


def _import_class(dotted_path: str) -> Any:
    import importlib
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _make_launcher(key: str, cfg_name: str, policy_override: Optional[str],
                   dofs_29: bool) -> Callable:
    def _launcher(entry: dict, step_mode_str: str, art_prefix: Optional[str]) -> threading.Thread:
        # Heavy deps deferred so the UI imports cleanly without RoboJuDo / torch.
        from robojudo.config import cfg_registry
        # Trigger pipeline cfg registration. These modules' decorators
        # populate cfg_registry at import time.
        import robojudo.config.g1.g1_cfg  # noqa: F401
        import robojudo.config.g1.g1_loco_mimic_cfg  # noqa: F401
        from robojudo.environment import env_registry  # noqa: F401
        from robojudo.pipeline import pipeline_registry
        # Importing URLab's env subclass triggers @env_registry.register so
        # the pipeline can find it by env_type. Also lets us swap in our
        # env cfg below.
        from urlab_policy.adapters.robojudo import (  # noqa: F401
            G1URLabRoboJuDoEnvCfg,
            G1_29URLabRoboJuDoEnvCfg,
            URLabRoboJuDoEnv,
        )

        client = STATE.client
        if client is None:
            raise RuntimeError("not connected")

        # RoboJuDo pipelines step at fixed Hz; coerce the bridge into direct
        # mode so the policy's expected decimation lines up. set_sim_options
        # is pushed AFTER pipeline construction below.
        if step_mode_str != "direct" or client.step_mode.value != "direct":
            client.runtime.set_mode("direct")
            log(f"{key}: switched step mode to direct")

        # Resolve articulation prefix (auto when there's only one).
        if not art_prefix:
            arts = list(client.articulations)
            if len(arts) != 1:
                raise RuntimeError(f"need articulation prefix (have: {arts})")
            art_prefix = arts[0]

        # Build the env cfg, reusing the dashboard's URLabClient.
        EnvCfgCls = G1_29URLabRoboJuDoEnvCfg if dofs_29 else G1URLabRoboJuDoEnvCfg
        env_cfg = EnvCfgCls(
            step_mode="direct",
            sim_dt=_DEFAULT_SIM_DT,
            sim_decimation=_DEFAULT_DECIMATION,
            articulation_prefix=art_prefix,
            observation_level="full",
            existing_client=client,
        )

        # Get the RoboJuDo pipeline cfg and swap its env to our adapter.
        pipeline_cfg = cfg_registry.get(cfg_name)()
        pipeline_cfg.env = env_cfg

        # Optional: override the pipeline's default `policy` to match the
        # registry entry's policy_cfg class.
        if policy_override:
            override_cls = _import_class(policy_override)
            pipeline_cfg.policy = override_cls()
            log(f"{key}: overrode pipeline policy with {policy_override.rsplit('.', 1)[-1]}")

        # Swap KeyboardCtrlCfg -> URLabUiCtrlCfg before construction;
        # the Policy tab renders the triggers as buttons.
        commands = swap_keyboard_ctrl_with_ui(pipeline_cfg)

        pipeline_cls = pipeline_registry.get(pipeline_cfg.pipeline_type)
        pipeline = pipeline_cls(cfg=pipeline_cfg)

        STATE.policy_run.control_handle = build_handle_from_pipeline(
            pipeline, commands,
        )

        # Last write to opt.timestep before the step loop -- UE recompile
        # reverts to the XML value.
        push_sim_dt(client, _DEFAULT_SIM_DT, label=key)

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
                    pipeline.step()
                    pr.step_count += 1
                    sleep_s = loop_dt - (time.perf_counter() - t0)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            except Exception as exc:
                pr.last_error = f"{type(exc).__name__}: {exc}"
                log(f"{key} crashed: {pr.last_error}", error=True)
            finally:
                # Do NOT call pipeline.env.shutdown() unconditionally --
                # our env honours `existing_client` and won't close the
                # client, but other resources (PHC handles, etc.) get
                # released here.
                try:
                    pipeline.env.shutdown()
                except Exception:
                    pass

        thread = threading.Thread(target=_loop, name=f"{key}Policy", daemon=True)
        thread.start()
        return thread

    return _launcher


for _key, _meta in CFG_MAP.items():
    LAUNCHERS[_key] = _make_launcher(
        _key, _meta["cfg"], _meta["policy_override"], bool(_meta["dofs_29"]),
    )
