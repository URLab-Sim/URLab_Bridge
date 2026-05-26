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

"""Runtime tab: live mutators while the sim is up.

Surfaces `client.runtime.*` ops + the articulation inspector. PIE
control / pause / step-mode / sim-speed live in the always-visible
sim toolbar (see app.py); this tab is for the mutators that target a
selected articulation: set_qpos, set_twist, set_control_source.
"""

from __future__ import annotations

from typing import Any

import dearpygui.dearpygui as dpg

from ..log import log
from ..state import STATE
from urlab_client import URLabRPCError
from .._helpers import parse_floats as _parse_floats


# ---------------------------------------------------------------------------
# Articulation selection / inspector
# ---------------------------------------------------------------------------


def on_articulation_selected(_s, app_data) -> None:
    STATE.selected_articulation = app_data
    refresh_articulation_info()


def refresh_articulations_dropdown() -> None:
    if not STATE.is_connected():
        return
    if not dpg.does_item_exist("runtime_art_combo"):
        return
    names = list(STATE.client.articulations.keys())
    dpg.configure_item("runtime_art_combo", items=names)
    if names:
        dpg.set_value("runtime_art_combo", names[0])
        STATE.selected_articulation = names[0]
        refresh_articulation_info()


def refresh_articulation_info() -> None:
    if not dpg.does_item_exist("runtime_art_info"):
        return
    if not STATE.is_connected() or not STATE.selected_articulation:
        dpg.set_value("runtime_art_info", "(connect + pick an articulation)")
        return
    art = STATE.client.articulations.get(STATE.selected_articulation)
    if art is None:
        dpg.set_value("runtime_art_info", "(articulation not found)")
        return

    lines = []
    lines.append(f"prefix={art.prefix}  actor_id={art.actor_id or '—'}")
    lines.append(f"actuators ({len(art.actuators)}):")
    last_applied = art.last_applied_ctrl
    for name, a in art.actuators.items():
        t = getattr(a, "type", "—")
        applied = (
            f"{last_applied[a._local_index]:+.4f}"
            if 0 <= a._local_index < len(last_applied) else "?"
        )
        lines.append(f"  {name:<20} type={t:<11}  sent={a.value:+.4f}  applied={applied}")

    qpos, qvel = art.qpos_array, art.qvel_array
    lines.append("")
    lines.append(f"joints ({len(art.joints)}):")
    qoff = voff = 0
    for name, j in art.joints.items():
        qslice = qpos[qoff: qoff + j.qpos_dim]
        vslice = qvel[voff: voff + j.qvel_dim]
        qstr = ", ".join(f"{v:+.4f}" for v in qslice) if qslice.size else "?"
        vstr = ", ".join(f"{v:+.4f}" for v in vslice) if vslice.size else "?"
        lines.append(f"  {name:<20} qpos=[{qstr}]  qvel=[{vstr}]")
        qoff += j.qpos_dim; voff += j.qvel_dim

    lines.append("")
    lines.append(f"sensors ({len(art.sensors)}):")
    for name, s in art.sensors.items():
        latest = "n/a" if s.latest is None else (
            f"[{', '.join(f'{v:+.3f}' for v in s.latest[:3])}"
            f"{', ...' if s.latest.size > 3 else ''}]"
        )
        lines.append(f"  {name:<20} dim={s.dim}  latest={latest}")

    lines.append("")
    lines.append(f"bodies ({len(art.bodies)}):")
    for name, b in art.bodies.items():
        xp = "?" if b.xpos is None else f"[{b.xpos[0]:+.3f}, {b.xpos[1]:+.3f}, {b.xpos[2]:+.3f}]"
        lines.append(f"  {name:<20} id={b.id:3d}  xpos={xp}")

    dpg.set_value("runtime_art_info", "\n".join(lines))


# ---------------------------------------------------------------------------
# Mutator callbacks
# ---------------------------------------------------------------------------


def _need_connection() -> bool:
    if not STATE.is_connected():
        log("not connected", error=True)
        return False
    return True


def on_set_qpos(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    target = (dpg.get_value("runtime_qpos_target") or "").strip()
    if not target:
        log("set_qpos: target required", error=True); return
    by_name = bool(dpg.get_value("runtime_qpos_by_name"))
    qpos_str = dpg.get_value("runtime_qpos_values") or ""
    try:
        qpos = [float(t) for t in qpos_str.replace(",", " ").split() if t]
    except (ValueError, TypeError) as exc:
        log(f"set_qpos: bad qpos input ({exc})", error=True); return
    if not qpos:
        log("set_qpos: qpos required", error=True); return
    try:
        STATE.client.runtime.set_qpos(target, qpos, by_name=by_name)
        log(f"set_qpos({target!r}, len={len(qpos)}) -> ok")
    except URLabRPCError as exc:
        log(f"set_qpos failed [{exc.code}]: {exc.message}", error=True)


def on_set_twist(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    art = STATE.selected_articulation
    if not art:
        log("set_twist: pick an articulation first", error=True); return
    vx = float(dpg.get_value("runtime_twist_vx") or 0.0)
    vy = float(dpg.get_value("runtime_twist_vy") or 0.0)
    yaw = float(dpg.get_value("runtime_twist_yaw") or 0.0)
    try:
        STATE.client.runtime.set_twist(art, linear=(vx, vy, 0.0), angular=(0.0, 0.0, yaw))
        log(f"set_twist({art}) vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}")
    except URLabRPCError as exc:
        log(f"set_twist failed [{exc.code}]: {exc.message}", error=True)


def on_set_control_source(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    src = (dpg.get_value("runtime_ctrlsrc_combo") or "zmq").lower()
    try:
        STATE.client.runtime.set_control_source(src)
        log(f"set_control_source(global) -> {src!r}")
    except URLabRPCError as exc:
        log(f"set_control_source failed [{exc.code}]: {exc.message}", error=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def build(parent: str) -> None:
    """Add the Runtime tab body under ``parent``."""
    with dpg.group(parent=parent):
        # ── Articulation picker (drives the inspector + set_twist) ────
        with dpg.group(horizontal=True):
            dpg.add_combo([], label="Articulation",
                          tag="runtime_art_combo", width=240,
                          callback=on_articulation_selected)
            dpg.add_button(label="Refresh",
                           callback=lambda: refresh_articulations_dropdown())
        dpg.add_separator()

        # ── set_qpos ───────────────────────────────────────────────────
        dpg.add_text("set_qpos", color=(110, 185, 255))
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="runtime_qpos_target",
                               width=200,
                               hint="actor_id (or actor name)")
            dpg.add_checkbox(label="by name",
                             tag="runtime_qpos_by_name",
                             default_value=False)
        dpg.add_input_text(tag="runtime_qpos_values",
                           width=520,
                           hint="space-separated floats; len 7 = free-base shortcut")
        btn = dpg.add_button(label="Apply qpos", callback=on_set_qpos)
        dpg.bind_item_theme(btn, "urlab_theme_primary_button")

        dpg.add_separator()

        # ── set_twist ──────────────────────────────────────────────────
        dpg.add_text("set_twist (selected articulation)", color=(110, 185, 255))
        with dpg.group(horizontal=True):
            dpg.add_input_float(label="vx",  tag="runtime_twist_vx",  default_value=0.0, width=90)
            dpg.add_input_float(label="vy",  tag="runtime_twist_vy",  default_value=0.0, width=90)
            dpg.add_input_float(label="yaw", tag="runtime_twist_yaw", default_value=0.0, width=90)
            dpg.add_button(label="Send twist", callback=on_set_twist)

        dpg.add_separator()

        # ── set_control_source ─────────────────────────────────────────
        dpg.add_text("set_control_source", color=(110, 185, 255))
        with dpg.group(horizontal=True):
            dpg.add_combo(["zmq", "ui"],
                          tag="runtime_ctrlsrc_combo",
                          default_value="zmq", width=140,
                          label="(global)")
            dpg.add_button(label="Apply", callback=on_set_control_source)

        dpg.add_separator()

        # ── Articulation inspector ─────────────────────────────────────
        dpg.add_text("Articulation inspector", color=(110, 185, 255))
        dpg.add_input_text(tag="runtime_art_info",
                           multiline=True, readonly=True,
                           width=1140, height=320,
                           default_value="(connect + pick an articulation)")


def tick() -> None:
    """Per-frame work for the Runtime tab."""
    if not STATE.is_connected():
        return
    refresh_articulation_info()
