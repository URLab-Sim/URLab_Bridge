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

"""UI-driven replacement for RoboJuDo's KeyboardCtrl.

`URLabUiCtrl` exposes the same controller protocol (event queue +
triggers dict + process_triggers) but is fed by clicks on dashboard
buttons -- no OS keyboard hook. `swap_keyboard_ctrl_with_ui` rewrites
a pipeline cfg in-place before construction so the pipeline never
sees the original KeyboardCtrlCfg.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, List, Optional


# Friendly labels per command. SHUTDOWN is omitted on purpose -- a
# one-click kill button next to "Play dance" is a footgun.
_COMMAND_LABELS = {
    "[POLICY_LOCO]":           "Locomotion mode",
    "[POLICY_MIMIC]":          "Play current motion",
    "[POLICY_SWITCH],NEXT":    "Next motion",
    "[POLICY_SWITCH],LAST":    "Previous motion",
    "[SIM_REBORN]":            "Sim reborn",
    "[MOTION_RESET]":          "Reset motion",
}

_EXCLUDED_COMMANDS = {"[SHUTDOWN]"}


# Module imports cleanly without RoboJuDo; swap helper just returns []
# when the controller classes can't be defined.
try:
    from robojudo.controller import Controller, ctrl_registry  # type: ignore
    from robojudo.controller.ctrl_cfgs import (  # type: ignore
        CtrlCfg, JoystickCtrlCfg, KeyboardCtrlCfg,
    )

    class URLabUiCtrlCfg(CtrlCfg):
        ctrl_type: str = "URLabUiCtrl"
        triggers: dict = {}

    @ctrl_registry.register
    class URLabUiCtrl(Controller):
        """KeyboardCtrl-shaped controller, queue-fed by dashboard buttons."""

        cfg_ctrl: "URLabUiCtrlCfg"

        def __init__(self, cfg_ctrl: "URLabUiCtrlCfg", env=None, **kwargs):
            super().__init__(cfg_ctrl=cfg_ctrl, env=env, **kwargs)
            self.event_queue: Queue = Queue(maxsize=100)

        def reset(self):
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except Empty:
                    break

        def get_data(self):
            events = []
            while not self.event_queue.empty():
                try:
                    events.append(self.event_queue.get_nowait())
                except Empty:
                    break
            return {"keyboard_event": events}

        def process_triggers(self, ctrl_data):
            commands = []
            if not self.triggers:
                return ctrl_data, commands
            for event in list(ctrl_data["keyboard_event"]):
                if event.get("type") == "keyboard" and not event.get("pressed"):
                    cmd = self.triggers.get(event["name"])
                    if cmd is not None:
                        commands.append(cmd)
                        ctrl_data["keyboard_event"].remove(event)
            return ctrl_data, commands

    @ctrl_registry.register
    class URLabTwistCtrl(Controller):
        """Feeds ``env.twist_cmd`` as a JoystickCtrl-shaped payload so loco
        policies (AMO, ASAP, Smooth, Unitree) consume UE's UMjTwistController
        without needing a real joystick."""

        def __init__(self, cfg_ctrl, env=None, **kwargs):
            super().__init__(cfg_ctrl=cfg_ctrl, env=env, **kwargs)

        def reset(self):
            pass

        def get_data(self):
            twist = getattr(self.env, "twist_cmd", None) if self.env is not None else None
            if twist is None or len(twist) < 3:
                twist = (0.0, 0.0, 0.0)
            # JoystickCtrl shape; AMO maps LeftY->vx, LeftX->vy, RightX->yaw.
            return {
                "axes": {
                    "LeftY":  float(twist[0]),
                    "LeftX":  float(twist[1]),
                    "RightX": float(twist[2]),
                    "RightY": 0.0,
                },
                "button_event": [],
            }

    # Any JoystickCtrlCfg in a pipeline cfg now instantiates URLabTwistCtrl
    # instead of the pygame-backed native one -- so UE's twist reaches loco
    # policies via the same code path a real joystick would use.
    ctrl_registry.registered_modules["JoystickCtrl"] = URLabTwistCtrl

    _HAS_ROBOJUDO = True

except ImportError:
    _HAS_ROBOJUDO = False
    URLabUiCtrlCfg = None  # type: ignore[assignment]
    URLabUiCtrl = None  # type: ignore[assignment]
    URLabTwistCtrl = None  # type: ignore[assignment]
    KeyboardCtrlCfg = None  # type: ignore[assignment]
    JoystickCtrlCfg = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public surface: ControlSpec + Handle + swap helper
# ---------------------------------------------------------------------------


@dataclass
class CommandSpec:
    """One clickable command extracted from a controller's trigger map."""

    label: str
    key: str
    command_str: str


@dataclass
class PolicyControlHandle:
    """Bridge between the Policy tab's buttons and a running pipeline's
    URLabUiCtrl. Thread-safe: ``queue.Queue.put`` from the dpg main
    thread, queue.get from the policy worker thread."""

    commands: List[CommandSpec] = field(default_factory=list)
    _ctrl: Any = None  # URLabUiCtrl instance

    def fire(self, key: str) -> None:
        if self._ctrl is None:
            return
        self._ctrl.event_queue.put({
            "type": "keyboard",
            "name": key,
            "pressed": False,
            "timestamp": time.time(),
        })


def _label_for(command_str: str, key: str) -> str:
    label = _COMMAND_LABELS.get(command_str)
    if label is not None:
        return label
    return f"{command_str.strip('[]')} ({key})"


def swap_keyboard_ctrl_with_ui(pipeline_cfg: Any) -> List[CommandSpec]:
    """Rewrite ``pipeline_cfg.ctrl`` in place: swap every KeyboardCtrlCfg
    for a URLabUiCtrlCfg (preserving its triggers) and ensure a
    JoystickCtrlCfg is present so UE's twist reaches loco policies via
    the URLabTwistCtrl shim. Returns the discovered command list for the
    Policy tab's button row. No-op (returns []) without RoboJuDo.

    Call BEFORE constructing the pipeline -- the pipeline reads
    cfg.ctrl in its __init__ to build the CtrlManager.
    """
    if not _HAS_ROBOJUDO:
        return []
    ctrls = getattr(pipeline_cfg, "ctrl", None) or []
    new_ctrls = []
    commands: List[CommandSpec] = []
    for cfg in ctrls:
        if isinstance(cfg, KeyboardCtrlCfg):
            triggers = dict(cfg.triggers)
            triggers.update(getattr(cfg, "triggers_extra", {}) or {})
            new_ctrls.append(URLabUiCtrlCfg(triggers=triggers))
            for key, command_str in triggers.items():
                if command_str in _EXCLUDED_COMMANDS:
                    continue
                commands.append(CommandSpec(
                    label=_label_for(command_str, key),
                    key=key,
                    command_str=command_str,
                ))
        else:
            new_ctrls.append(cfg)
    if not any(getattr(c, "ctrl_type", "") == "JoystickCtrl" for c in new_ctrls):
        new_ctrls.append(JoystickCtrlCfg())
    pipeline_cfg.ctrl = new_ctrls
    return commands


def build_handle_from_pipeline(
    pipeline: Any, commands: List[CommandSpec]
) -> Optional[PolicyControlHandle]:
    """Resolve the live URLabUiCtrl instance on a built pipeline and
    bundle it with the discovered command list into a handle."""
    if not commands:
        return None
    ctrl_manager = getattr(pipeline, "ctrl_manager", None)
    controllers = getattr(ctrl_manager, "controllers", None) if ctrl_manager else None
    if not controllers:
        return None
    entry = controllers.get("URLabUiCtrl") if hasattr(controllers, "get") else None
    if entry is None:
        return None
    inst = getattr(entry, "inst", None)
    if inst is None and hasattr(entry, "get"):
        inst = entry.get("inst")
    if inst is None:
        return None
    return PolicyControlHandle(commands=commands, _ctrl=inst)
