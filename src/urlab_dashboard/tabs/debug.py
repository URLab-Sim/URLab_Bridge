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

"""Debug tab: developer-only step/reset + an embedded render viewer.

Live mutators (set_qpos / set_twist / set_control_source) and the
articulation inspector live on the Runtime tab. PIE control / pause /
sim_speed / set_mode live in the always-visible sim toolbar.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from ..log import log
from ..renderer import EmbeddedRenderer, VIEWER_H, VIEWER_W
from ..state import STATE
from urlab_client import URLabRPCError


# Single Renderer instance per process. The shell owns / disposes it via
# release() on disconnect.
RENDERER = EmbeddedRenderer()


def on_step(_s=None, _a=None) -> None:
    if not STATE.is_connected():
        log("not connected", error=True); return
    n = int(dpg.get_value("debug_n_steps") or 1)
    try:
        STATE.client.step(n_steps=n)
        log(f"step(n={n}) sim_time={STATE.client.sim_time:.4f} step={STATE.client.step_count}")
    except URLabRPCError as exc:
        log(f"step failed [{exc.code}]: {exc.message}", error=True)


def on_reset(_s=None, _a=None) -> None:
    if not STATE.is_connected():
        log("not connected", error=True); return
    seed_str = dpg.get_value("debug_seed") or ""
    seed = int(seed_str) if seed_str.strip() else None
    try:
        STATE.client.reset(seed=seed)
        log(f"reset(seed={seed}) sim_time={STATE.client.sim_time:.4f}")
    except URLabRPCError as exc:
        log(f"reset failed [{exc.code}]: {exc.message}", error=True)


def on_render_now(_s=None, _a=None) -> None:
    STATE.render_request = True


def build(parent: str) -> None:
    """Add the Debug tab body under ``parent``."""
    with dpg.group(parent=parent):
        dpg.add_text("Manual step / reset", color=(110, 185, 255))
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="n_steps", tag="debug_n_steps",
                              default_value=1, width=120)
            dpg.add_button(label="Step", callback=on_step)
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="seed", tag="debug_seed",
                               default_value="", width=120,
                               hint="blank = no seed")
            dpg.add_button(label="Reset", callback=on_reset)

        dpg.add_separator()

        dpg.add_text("Render viewer", color=(110, 185, 255))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Render now", callback=on_render_now)
        dpg.add_image("mj_render_texture", width=VIEWER_W, height=VIEWER_H)


def tick() -> None:
    if not STATE.is_connected():
        return
    err = RENDERER.render_into_texture(
        STATE.client, STATE.selected_articulation, "mj_render_texture",
    )
    if err:
        log(err, error=True)
