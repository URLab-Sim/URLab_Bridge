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

"""URLab UI shell. Owns the dpg context, viewport, texture registry, and
per-frame render loop; tabs plug in via build/tick functions under
``urlab_dashboard.tabs``."""

from __future__ import annotations

import argparse
import sys
import threading
import time

# Renderer module imports mujoco BEFORE dearpygui to keep GL warmup ordering.
from .renderer import VIEWER_H, VIEWER_W

import dearpygui.dearpygui as dpg

from .log import log
from .state import STATE
from .tabs import cameras as tab_cameras
from .tabs import debug as tab_debug
from .tabs import policy as tab_policy
from .tabs import recording as tab_recording
from .tabs import runtime as tab_runtime
from .tabs import scene as tab_scene
from urlab_client import URLabClient, URLabRPCError, StepMode


# ---------------------------------------------------------------------------
# Connection callbacks (top-bar)
# ---------------------------------------------------------------------------


def on_connect(_s=None, _a=None) -> None:
    """Hand the connect to a worker and return.

    Every RPC in `connect()` runs inline, and the last of them starts a
    camera's broadcast and opens a SUB socket per camera -- seconds of work
    against a real robot. On the callback thread that is the UI thread, so the
    window stops answering for the duration and looks hung.
    """
    if STATE.is_connected():
        log("already connected"); return
    if STATE.connecting:
        log("already connecting"); return
    host = dpg.get_value("host_input") or STATE.host
    port = int(dpg.get_value("port_input") or STATE.step_port)
    mode_str = dpg.get_value("connect_mode_combo") or "auto"
    STATE.connecting = True
    log(f"connecting to {host}:{port} ...")
    threading.Thread(target=_connect_worker, args=(host, port, mode_str),
                     daemon=True).start()


def _connect_worker(host: str, port: int, mode_str: str) -> None:
    try:
        STATE.client = URLabClient(host, step_mode=mode_str, step_port=port,
                                    recv_timeout_ms=5000)
        STATE.client.connect()
        STATE.host, STATE.step_port = host, port
        log(f"connected to {host}:{port} session={STATE.client.session_id} "
            f"mode={STATE.client.step_mode.value} "
            f"manager_present={STATE.client.manager_present}")
        tab_runtime.refresh_articulations_dropdown()
        tab_runtime.refresh_articulation_info()
        tab_policy.refresh_articulations()
        _claim_control(STATE.client)
        STATE.cameras_dirty = True
        STATE.render_request = True
    except Exception as exc:
        msg = str(exc)
        # zmq Again (EAGAIN): server never replied. Bridge not running,
        # wrong port, or editor holds an old build.
        import zmq  # local: dashboard may not always carry it transitively
        if isinstance(exc, zmq.Again) or "temporarily unavailable" in msg.lower():
            hint = (
                f"connect timeout on {host}:{port} -- check that:\n"
                f"  1. The UE editor is running and the URLab toolbar pill is GREEN\n"
                f"  2. The port matches (toolbar tooltip / Config/LocalUnrealRoboticsLab.ini [BridgeServer]/StepPort)\n"
                f"  3. The editor was restarted after the last C++ rebuild"
            )
            log(hint, error=True)
        else:
            log(f"connect failed: {exc}", error=True)
        # Close a partial client so its ZMQ context + sockets tear down
        # deterministically. Skipping this leaves an orphan context that
        # atexit later terms while threads or sockets are still alive,
        # which trips libzmq's signaler assertion (WSAECONNRESET) on
        # Windows.
        if STATE.client is not None:
            try:
                STATE.client.close()
            except Exception:
                pass
        STATE.client = None
    finally:
        STATE.connecting = False
    _refresh_status()


def on_disconnect(_s=None, _a=None) -> None:
    if not STATE.is_connected():
        log("already disconnected"); return
    try:
        STATE.client.close(); log("disconnected (server reverted to live)")
    except Exception as exc:
        log(f"disconnect error: {exc}", error=True)
    STATE.client = None
    STATE.selected_articulation = None
    STATE.pie_state = None
    tab_debug.RENDERER.release()
    tab_cameras.release_textures()
    tab_runtime.refresh_articulation_info()
    tab_policy.refresh_articulations()
    _refresh_status()


def _set_pill(tag: str, text: str, color: tuple) -> None:
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, text)
        dpg.configure_item(tag, color=color)


# Pill colors. Keep dim (muted) for "no signal", green for healthy,
# yellow for transient, red for error / disconnected.
_DIM    = (155, 165, 180)
_GREEN  = ( 80, 200, 130)
_YELLOW = (240, 200,  80)
_RED    = (220,  90,  90)
_BLUE   = (110, 185, 255)


def _pie_state_pill_color(state: str) -> tuple:
    if state == "ready":          return _GREEN
    if state == "compiling":      return _YELLOW
    if state == "compile_failed": return _RED
    if state == "timeout":        return _RED
    return _DIM  # "off" or unknown


def _refresh_status() -> None:
    """Populate the status-pill row(s). Runs on the render-loop poll
    every 0.5 s (background thread); only reads STATE, no mutations."""
    if not dpg.does_item_exist("pill_conn"):
        return  # UI hasn't been built yet

    if not STATE.is_connected():
        _set_pill("pill_conn",    "● disconnected",      _RED)
        _set_pill("pill_session", "session: —",           _DIM)
        _set_pill("pill_manager", "manager: —",           _DIM)
        _set_pill("pill_pie",     "PIE: —",               _DIM)
        _set_pill("pill_mode",    "mode: —",              _DIM)
        _set_pill("pill_simtime", "sim_time: —",          _DIM)
        _set_pill("pill_step",    "step: —",              _DIM)
        _set_pill("pill_arts",    "arts: —",              _DIM)
        _set_pill("pill_ents",    "entities: —",          _DIM)
        _set_pill("pill_speed",   "speed: —",             _DIM)
        return

    c = STATE.client
    sid = (c.session_id or "")[:8]
    _set_pill("pill_conn",    "● connected",          _GREEN)
    _set_pill("pill_session", f"session: {sid}",       _DIM)
    _set_pill("pill_manager", "manager: ✓" if c.manager_present else "manager: ✗",
              _GREEN if c.manager_present else _RED)
    pie = STATE.pie_state or "—"
    _set_pill("pill_pie",     f"PIE: {pie}", _pie_state_pill_color(pie))
    _set_pill("pill_mode",    f"mode: {c.step_mode.value}",   _BLUE)
    _set_pill("pill_simtime", f"sim_time: {c.sim_time:.4f}s", _DIM)
    _set_pill("pill_step",    f"step: {c.step_count}",        _DIM)
    _set_pill("pill_arts",    f"arts: {len(c.articulations)}", _DIM)
    _set_pill("pill_ents",    f"entities: {len(c.entities)}", _DIM)


# ---------------------------------------------------------------------------
# Sim toolbar callbacks (hoisted from Scene + Debug; affect global state).
# ---------------------------------------------------------------------------


def _need_connection() -> bool:
    if not STATE.is_connected():
        log("not connected", error=True)
        return False
    return True


def on_begin_pie(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    level = (dpg.get_value("toolbar_pie_level") or "").strip() or None
    timeout_s = float(dpg.get_value("toolbar_pie_timeout") or 30.0)
    try:
        reply = STATE.client.sim.start(level, timeout_s=timeout_s)
        STATE.pie_state = reply.state.value
        compile_error = reply.compile_error or ""
        log(f"sim.start -> state={STATE.pie_state} compile_error={compile_error!r}")
        # PIE just (re)built the scene: articulations + cameras came into being
        # via the absorbed handshake (client.sim.start now also starts their
        # streams). Refresh the UI so the camera/runtime tabs pick them up.
        if reply.is_ready:
            tab_runtime.refresh_articulations_dropdown()
            tab_runtime.refresh_articulation_info()
            tab_policy.refresh_articulations()
            tab_cameras.ensure_textures()
            STATE.render_request = True
    except URLabRPCError as exc:
        log(f"sim.start failed [{exc.code}]: {exc.message}", error=True)


def on_stop_pie(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    try:
        STATE.client.sim.stop()
        STATE.pie_state = "off"
        log("sim.stop ok")
    except URLabRPCError as exc:
        log(f"sim.stop failed [{exc.code}]: {exc.message}", error=True)


def on_play(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    try:
        new = STATE.client.runtime.set_paused(False)
        log(f"set_paused -> {new}")
    except URLabRPCError as exc:
        log(f"set_paused failed [{exc.code}]: {exc.message}", error=True)


def on_pause(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    try:
        new = STATE.client.runtime.set_paused(True)
        log(f"set_paused -> {new}")
    except URLabRPCError as exc:
        log(f"set_paused failed [{exc.code}]: {exc.message}", error=True)


def on_set_mode(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    mode_str = dpg.get_value("toolbar_mode_combo") or "live"
    try:
        new = STATE.client.runtime.set_mode(mode_str)
        log(f"set_mode -> {new.value}")
    except URLabRPCError as exc:
        log(f"set_mode failed [{exc.code}]: {exc.message}", error=True)


def on_set_sim_speed(_s=None, _a=None) -> None:
    if not _need_connection():
        return
    pct = float(dpg.get_value("toolbar_speed_slider") or 100.0)
    try:
        eff = STATE.client.runtime.set_sim_speed(pct)
        log(f"set_sim_speed({pct:.1f}) -> {eff:.1f}%")
    except URLabRPCError as exc:
        log(f"set_sim_speed failed [{exc.code}]: {exc.message}", error=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


MODE_OPTIONS = [m.value for m in StepMode]


def _install_theme() -> None:
    """Apply a dark blue-gray theme with rounded corners + breathing-room
    paddings. Bound to the root window so every tab inherits it."""
    with dpg.theme(tag="urlab_theme"):
        with dpg.theme_component(dpg.mvAll):
            # Spacing & rounding.
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,      6,  4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,       8,  6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing,  6,  4)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,     4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,      4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding,       4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,    4)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)

            # Palette: cool dark gray with cyan accent.
            BG       = (28, 30, 36, 255)
            BG2      = (38, 41, 48, 255)
            BG3      = (50, 54, 62, 255)
            BORDER   = (60, 65, 75, 255)
            TEXT     = (220, 224, 232, 255)
            DIM      = (155, 165, 180, 255)
            ACCENT   = (88, 166, 255, 255)
            ACCENT_H = (110, 185, 255, 255)
            HEADER   = (45, 55, 75, 255)
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,   BG)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,    BG2)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg,    BG2)
            dpg.add_theme_color(dpg.mvThemeCol_Border,     BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,            BG3)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,     (62, 68, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,      (72, 78, 92, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,            BG2)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,      HEADER)
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg,          BG2)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,        BG)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,      BG3)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_Button,             BG3)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,      (72, 102, 150, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,       ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_Header,             HEADER)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,      (62, 78, 110, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,       (72, 92, 130, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Tab,                BG2)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered,         (72, 92, 130, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive,          HEADER)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused,       BG2)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, BG3)
            dpg.add_theme_color(dpg.mvThemeCol_Separator,          BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered,   ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive,    ACCENT_H)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark,          ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,         ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,   ACCENT_H)
            dpg.add_theme_color(dpg.mvThemeCol_Text,               TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,       DIM)

    # Distinct theme for "primary" action buttons (Connect, Apply, etc.)
    with dpg.theme(tag="urlab_theme_primary_button"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (62, 120, 200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 150, 230, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (110, 170, 255, 255))

    # Theme for "danger" buttons (Disconnect, Destroy, Stop PIE).
    with dpg.theme(tag="urlab_theme_danger_button"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (160, 60, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (190, 80, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (220, 100, 100, 255))


def build_ui(initial_host: str, initial_port: int) -> None:
    dpg.create_context()
    dpg.create_viewport(title="URLab Bridge UI", width=1280, height=900)

    _install_theme()

    init_pixels = [0.18, 0.18, 0.20, 1.0] * (VIEWER_W * VIEWER_H)
    with dpg.texture_registry(tag="texture_registry"):
        dpg.add_dynamic_texture(VIEWER_W, VIEWER_H, init_pixels,
                                tag="mj_render_texture")

    with dpg.window(label="URLab", tag="root", width=1280, height=900,
                    no_title_bar=True, no_resize=False, no_move=True):
        # ── 1. Connection row ──────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_text("URLab Bridge", color=(110, 185, 255))
            dpg.add_text("|")
            dpg.add_input_text(tag="host_input",
                               default_value=initial_host, width=180,
                               hint="host (tcp://...)")
            dpg.add_input_int(tag="port_input",
                              default_value=initial_port, width=90,
                              step=0)
            dpg.add_combo(MODE_OPTIONS, tag="connect_mode_combo",
                          default_value="auto", width=130)
            connect_btn = dpg.add_button(label="Connect", callback=on_connect)
            dpg.bind_item_theme(connect_btn, "urlab_theme_primary_button")
            disc_btn = dpg.add_button(label="Disconnect", callback=on_disconnect)
            dpg.bind_item_theme(disc_btn, "urlab_theme_danger_button")

        # ── 2. Status pills (2 rows) ───────────────────────────────────
        # Row 1: connection / session / manager / PIE / mode
        with dpg.group(horizontal=True):
            dpg.add_text("● disconnected", tag="pill_conn",    color=_RED)
            dpg.add_text("|", color=_DIM)
            dpg.add_text("session: —",     tag="pill_session", color=_DIM)
            dpg.add_text("|", color=_DIM)
            dpg.add_text("manager: —",     tag="pill_manager", color=_DIM)
            dpg.add_text("|", color=_DIM)
            dpg.add_text("PIE: —",         tag="pill_pie",     color=_DIM)
            dpg.add_text("|", color=_DIM)
            dpg.add_text("mode: —",        tag="pill_mode",    color=_DIM)
        # Row 2: sim_time / step / arts / entities
        with dpg.group(horizontal=True):
            dpg.add_text("sim_time: —",    tag="pill_simtime", color=_DIM)
            dpg.add_text("|", color=_DIM)
            dpg.add_text("step: —",        tag="pill_step",    color=_DIM)
            dpg.add_text("|", color=_DIM)
            dpg.add_text("arts: —",        tag="pill_arts",    color=_DIM)
            dpg.add_text("|", color=_DIM)
            dpg.add_text("entities: —",    tag="pill_ents",    color=_DIM)
        dpg.add_separator()

        # ── 3. Sim toolbar ─────────────────────────────────────────────
        # Always visible because every tab depends on PIE / pause / mode.
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="toolbar_pie_level", width=240,
                               hint="level path (blank = current)")
            dpg.add_input_float(tag="toolbar_pie_timeout",
                                default_value=30.0, width=90, step=0.0,
                                format="%.1f s")
            begin = dpg.add_button(label="Begin PIE", callback=on_begin_pie)
            dpg.bind_item_theme(begin, "urlab_theme_primary_button")
            stop = dpg.add_button(label="Stop PIE", callback=on_stop_pie)
            dpg.bind_item_theme(stop, "urlab_theme_danger_button")
            dpg.add_text(" │ ", color=_DIM)
            dpg.add_button(label="Play",  callback=on_play)
            dpg.add_button(label="Pause", callback=on_pause)
            dpg.add_text(" │ ", color=_DIM)
            dpg.add_slider_float(tag="toolbar_speed_slider",
                                 default_value=100.0, min_value=5.0,
                                 max_value=100.0, width=180,
                                 callback=on_set_sim_speed,
                                 format="%.0f%%")
            dpg.add_text(" │ ", color=_DIM)
            dpg.add_combo(["live", "direct", "puppet"],
                          tag="toolbar_mode_combo",
                          default_value="direct", width=130)
            dpg.add_button(label="Set mode", callback=on_set_mode)
        dpg.add_separator()

        # ── 4. Tabs ────────────────────────────────────────────────────
        # Order reflects the namespace surface: scene (authoring) →
        # runtime (live mutators) → diagnostics tail.
        with dpg.tab_bar(tag="tabs"):
            with dpg.tab(label="Scene", tag="tab_scene"):
                tab_scene.build("tab_scene")
            with dpg.tab(label="Runtime", tag="tab_runtime"):
                tab_runtime.build("tab_runtime")
            with dpg.tab(label="Cameras", tag="tab_cameras"):
                tab_cameras.build("tab_cameras")
            with dpg.tab(label="Recording", tag="tab_recording"):
                tab_recording.build("tab_recording")
            with dpg.tab(label="Policy", tag="tab_policy"):
                tab_policy.build("tab_policy")
            with dpg.tab(label="Debug", tag="tab_debug"):
                tab_debug.build("tab_debug")

        # ── 5. Log pane ────────────────────────────────────────────────
        dpg.add_separator()
        dpg.add_text("Log")
        dpg.add_input_text(tag="log_panel", multiline=True, readonly=True,
                           width=1060, height=180)

    dpg.bind_theme("urlab_theme")
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)


# ---------------------------------------------------------------------------
# Background poll
# ---------------------------------------------------------------------------


def _claim_control(client) -> None:
    """Take the control lease on every articulation.

    A step request carries ctrl, and the server refuses ctrl from a client that
    holds no claim, so without this the Step button answers `not_control_owner`
    and nothing moves.

    Forced, because the lease is indefinite and is not released by a client that
    dies holding it: an unclean exit would otherwise lock this UI out of its own
    scene with no way back short of restarting PIE. The steal is logged, so a
    policy that loses control says so rather than quietly stopping.
    """
    for prefix in list(getattr(client, "articulations", {}) or {}):
        try:
            client.runtime.claim_control(prefix, force=True)
        except URLabRPCError as exc:
            log(f"claim_control({prefix}) failed [{exc.code}]: {exc.message}",
                error=True)
        else:
            log(f"control claimed on {prefix}")


def _note_stall(label: str, started: float, budget_ms: float) -> None:
    """Report a main-loop stage that held the UI longer than its budget.

    The window belongs to this thread, so anything slow here is a UI that has
    stopped answering; naming the stage is the difference between "the bridge
    froze" and a line number.
    """
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if elapsed_ms >= budget_ms:
        log(f"UI stall: {label} held the main loop for {elapsed_ms:.0f}ms")


def _status_poll_loop() -> None:
    """Refresh the status line every 0.5s. Renderer is NOT touched here:
    GL context is owned by the main thread."""
    while not STATE.stop_poll.is_set():
        try:
            _refresh_status()
        except Exception:
            pass
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="URLab unified bridge UI")
    parser.add_argument("--host", default="tcp://127.0.0.1",
                        help="ZMQ address (default: tcp://127.0.0.1)")
    parser.add_argument("--port", type=int, default=5559,
                        help="step server port (default: 5559)")
    parser.add_argument("--render-fps", type=float, default=15.0,
                        help="continuous render rate when connected (default 15)")
    parser.add_argument("--stall-ms", type=float, default=250.0,
                        help="log any UI stage that blocks the main loop for "
                             "longer than this (default 250)")
    parser.add_argument("--autoconnect", action="store_true",
                        help="connect on startup instead of waiting for the button")
    args = parser.parse_args()

    build_ui(args.host, args.port)

    poll = threading.Thread(target=_status_poll_loop, daemon=True)
    poll.start()

    if args.autoconnect:
        on_connect()

    render_interval = 1.0 / max(args.render_fps, 1.0)
    last_render = 0.0
    try:
        while dpg.is_dearpygui_running():
            now = time.monotonic()
            render_due = (now - last_render) >= render_interval or STATE.render_request
            if STATE.is_connected() and render_due:
                # Free mode pulls latest snapshot before tick so the
                # inspector / viewer see live UE state without an RPC.
                if STATE.client.step_mode.value == "live":
                    snap = STATE.client._latest_state_snapshot
                    if snap is not None:
                        try:
                            STATE.client._absorb_step_reply(snap)
                        except Exception:
                            pass
                # tick() implementations are individually robust to a
                # missing client.model (renderer no-ops, inspector still
                # reads art.qpos_array which exists post-discover).
                # Wrap each tab's tick: an exception in one must not
                # kill the whole UI loop. Background ops can race PIE
                # transitions / server restarts and surface transient
                # transport errors that should just retry next frame.
                if STATE.cameras_dirty:
                    STATE.cameras_dirty = False
                    try:
                        tab_cameras.ensure_textures()
                    except Exception as exc:
                        log(f"camera window setup failed: {exc}", error=True)

                for name, fn in (("scene",   tab_scene.tick),
                                 ("runtime", tab_runtime.tick),
                                 ("cameras", tab_cameras.tick),
                                 ("policy",  tab_policy.tick),
                                 ("debug",   tab_debug.tick)):
                    stage = time.perf_counter()
                    try:
                        fn()
                    except Exception as exc:
                        log(f"{name}.tick error ({type(exc).__name__}): {exc}",
                            error=True)
                    _note_stall(f"{name}.tick", stage, args.stall_ms)
                last_render = now
                STATE.render_request = False
            present = time.perf_counter()
            dpg.render_dearpygui_frame()
            _note_stall("present", present, args.stall_ms)
    finally:
        STATE.stop_poll.set()
        if STATE.is_connected():
            try:
                STATE.client.close()
            except Exception:
                pass
        dpg.destroy_context()


if __name__ == "__main__":
    main()
