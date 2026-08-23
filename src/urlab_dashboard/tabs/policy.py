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

"""Policy tab: pick from policy_registry, view metadata, launch/stop.

Status: scaffold. The dropdown / metadata / mode-constraint UX is wired,
but in-process Launch is per-policy work — see ``LAUNCHERS`` below.
Adding a policy: write a launcher function that spawns a worker thread
running the policy against ``STATE.client`` and register it in LAUNCHERS.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import dearpygui.dearpygui as dpg

from ..log import log
from ..state import STATE
from urlab_policy.registry import (
    POLICIES,
    check_step_mode_compatible,
    get_required_step_mode,
)


# Map policy key -> launcher callable. Each launcher takes (entry_dict,
# step_mode_str, articulation_prefix_or_none) and returns a started
# threading.Thread. Set STATE.policy_run.thread / .name / .stop_flag /
# .started_at on the way out.
LAUNCHERS: Dict[str, Callable] = {}


def _selected_policy_key() -> Optional[str]:
    if not dpg.does_item_exist("policy_combo"):
        return None
    label = dpg.get_value("policy_combo")
    for key, entry in POLICIES.items():
        if entry["label"] == label:
            return key
    return None


def _refresh_metadata(_s=None, _a=None) -> None:
    key = _selected_policy_key()
    if key is None:
        dpg.set_value("policy_meta", "")
        return
    entry = POLICIES[key]
    req = get_required_step_mode(entry)
    req_str = ", ".join(m.value for m in req) if req else "any"
    lines = [
        f"key:        {key}",
        f"label:      {entry['label']}",
        f"desc:       {entry.get('desc', '')}",
        f"dofs:       {entry.get('dofs', '?')}",
        f"xml:        {entry.get('xml', '?')}",
        f"ctrl_type:  {entry.get('ctrl_type', '?')}",
        f"req mode:   {req_str}",
        f"policy_cfg: {entry.get('policy_cfg', '?')}",
        f"env_cfg:    {entry.get('env_cfg', '?')}",
    ]
    if entry.get("requires_phc"):
        lines.append("requires_phc: True (RoboJuDo PHC submodule)")
    dpg.set_value("policy_meta", "\n".join(lines))


def on_launch(_s=None, _a=None) -> None:
    if not STATE.is_connected():
        log("launch: not connected", error=True); return
    if STATE.policy_run.thread is not None and STATE.policy_run.thread.is_alive():
        log(f"launch: '{STATE.policy_run.name}' already running", error=True); return

    key = _selected_policy_key()
    if key is None:
        log("launch: pick a policy", error=True); return
    entry = POLICIES[key]

    mode_str = dpg.get_value("policy_mode_combo") or STATE.client.step_mode.value
    try:
        check_step_mode_compatible(entry, mode_str)
    except ValueError as exc:
        log(f"launch: {exc}", error=True); return

    art = dpg.get_value("policy_art_combo") or None

    launcher = LAUNCHERS.get(key)
    if launcher is None:
        log(
            f"launch: '{key}' has no in-process launcher yet — run the "
            f"matching scripts/run_*.py script for now. See ui/README.md "
            f"to wire it in.", error=True,
        )
        return

    try:
        thread = launcher(entry, mode_str, art)
    except Exception as exc:
        log(f"launch failed: {type(exc).__name__}: {exc}", error=True); return

    STATE.policy_run.name = key
    STATE.policy_run.thread = thread
    STATE.policy_run.last_error = ""
    log(f"launched '{key}' in mode '{mode_str}' on art '{art or '(default)'}'")


def on_stop(_s=None, _a=None) -> None:
    pr = STATE.policy_run
    if pr.thread is None or not pr.thread.is_alive():
        log("stop: nothing running"); return
    pr.stop_flag.set()
    log(f"stop signalled for '{pr.name}'")


_last_art_keys: tuple = ()


def refresh_articulations() -> None:
    """Repopulate the Articulation dropdown from the current client.

    Idempotent; also called from tick() so the dropdown picks up new
    articulations after Begin PIE / spawn_actor without the user having
    to disconnect+reconnect."""
    global _last_art_keys
    if not dpg.does_item_exist("policy_art_combo"):
        return
    items = list(STATE.client.articulations.keys()) if STATE.is_connected() else []
    new_keys = tuple(items)
    if new_keys == _last_art_keys:
        return
    _last_art_keys = new_keys
    current = dpg.get_value("policy_art_combo") or ""
    dpg.configure_item("policy_art_combo", items=items)
    # Preserve the current selection if it's still valid; otherwise
    # default to the first item.
    if current in items:
        dpg.set_value("policy_art_combo", current)
    elif items:
        dpg.set_value("policy_art_combo", items[0])
    else:
        dpg.set_value("policy_art_combo", "")


def build(parent: str) -> None:
    # Importing the launchers package triggers each submodule's
    # registration into LAUNCHERS. Done lazily on tab build so the UI
    # imports cleanly even if some launchers' deps aren't installed.
    from .. import launchers  # noqa: F401

    labels = [e["label"] for e in POLICIES.values()]
    with dpg.group(parent=parent):
        dpg.add_text("Policy")
        dpg.add_combo(labels, label="", tag="policy_combo",
                      width=400, callback=_refresh_metadata,
                      default_value=labels[0] if labels else "")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_combo(["auto", "freerun", "stepped", "statepushed"],
                          label="Step mode", tag="policy_mode_combo",
                          default_value="stepped", width=140)
            dpg.add_combo([], label="Articulation",
                          tag="policy_art_combo", width=240)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Launch", callback=on_launch)
            dpg.add_button(label="Stop", callback=on_stop)
            dpg.add_text("", tag="policy_status_label")

        dpg.add_separator()
        dpg.add_input_text(tag="policy_meta", multiline=True, readonly=True,
                           width=680, height=200,
                           default_value="(pick a policy)")

        # Per-pipeline control surface. Populated by tick() when a
        # launched policy exposes a control_handle (e.g. RoboJuDo's
        # loco-mimic pipeline). Stays hidden otherwise.
        with dpg.group(tag="policy_control_section", show=False):
            dpg.add_separator()
            dpg.add_text("Pipeline controls", color=(110, 185, 255))
            dpg.add_group(tag="policy_control_buttons", horizontal=True)

    if labels:
        _refresh_metadata()


_rendered_handle_id: int = 0


def _make_control_callback(key: str):
    def _cb(_s=None, _a=None) -> None:
        handle = STATE.policy_run.control_handle
        if handle is None:
            return
        handle.fire(key)
        log(f"{STATE.policy_run.name}: fired {key!r}")
    return _cb


def _refresh_control_section() -> None:
    """Sync the control-buttons section with the current handle. Idempotent
    -- only rebuilds when the handle object changes (by id)."""
    global _rendered_handle_id
    handle = STATE.policy_run.control_handle
    if not dpg.does_item_exist("policy_control_section"):
        return
    target_id = id(handle) if handle is not None else 0
    if target_id == _rendered_handle_id:
        return
    _rendered_handle_id = target_id

    dpg.delete_item("policy_control_buttons", children_only=True)
    if handle is None or not handle.commands:
        dpg.configure_item("policy_control_section", show=False)
        return
    dpg.configure_item("policy_control_section", show=True)
    for spec in handle.commands:
        dpg.add_button(
            label=spec.label,
            parent="policy_control_buttons",
            callback=_make_control_callback(spec.key),
        )


def tick() -> None:
    if not dpg.does_item_exist("policy_status_label"):
        return
    # Pick up articulations that appeared after connect — e.g. user
    # clicked Begin PIE or spawned an actor since the last refresh.
    refresh_articulations()
    pr = STATE.policy_run
    alive = pr.thread is not None and pr.thread.is_alive()
    if not alive and pr.thread is not None and not pr.stop_flag.is_set():
        # Thread exited on its own.
        STATE.policy_run.thread = None
        STATE.policy_run.control_handle = None
    if not alive and pr.control_handle is not None:
        # Stop was signalled but the handle was never cleared.
        STATE.policy_run.control_handle = None
    _refresh_control_section()
    msg = ""
    if alive:
        msg = f"running '{pr.name}'  steps={pr.step_count}"
        if pr.last_error:
            msg += f"  last_error={pr.last_error}"
    elif pr.last_error:
        msg = f"last error from '{pr.name}': {pr.last_error}"
    dpg.set_value("policy_status_label", msg)
