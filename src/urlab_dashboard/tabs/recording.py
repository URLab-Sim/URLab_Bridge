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

"""Recording / replay tab. Thin wrapper over client.recording / client.replay."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from ..log import log
from ..state import STATE
from urlab_client import URLabRPCError


def on_recording_start(_s=None, _a=None) -> None:
    if not STATE.is_connected(): return
    name = dpg.get_value("recording_name_input") or None
    try:
        handle = STATE.client.recording.start(name=name)
        log(f"recording_start -> {handle.name!r}")
    except URLabRPCError as exc:
        log(f"recording_start failed [{exc.code}]: {exc.message}", error=True)


def on_recording_stop(_s=None, _a=None) -> None:
    if not STATE.is_connected(): return
    try:
        summary = STATE.client.recording.stop()
        log(f"recording_stop frames={summary.frame_count} "
            f"duration={summary.sim_duration_s:.4f}s")
    except URLabRPCError as exc:
        log(f"recording_stop failed [{exc.code}]: {exc.message}", error=True)


def on_recording_save(_s=None, _a=None) -> None:
    if not STATE.is_connected(): return
    path = dpg.get_value("recording_path_input") or None
    try:
        actual = STATE.client.recording.save(path=path)
        log(f"recording_save -> {actual}")
    except URLabRPCError as exc:
        log(f"recording_save failed [{exc.code}]: {exc.message}", error=True)


def on_replay_play(_s=None, _a=None) -> None:
    if not STATE.is_connected(): return
    target = dpg.get_value("replay_target_input")
    if not target:
        log("replay_play: provide a path or session name", error=True); return
    try:
        session = STATE.client.replay.play(target)
        log(f"replay_play -> active session {session.name!r}")
    except URLabRPCError as exc:
        log(f"replay_play failed [{exc.code}]: {exc.message}", error=True)


def on_replay_stop(_s=None, _a=None) -> None:
    if not STATE.is_connected(): return
    try:
        STATE.client.replay.stop(); log("replay_stop ok")
    except URLabRPCError as exc:
        log(f"replay_stop failed [{exc.code}]: {exc.message}", error=True)


def build(parent: str) -> None:
    with dpg.group(parent=parent):
        dpg.add_text("Recording")
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="name", tag="recording_name_input",
                               hint="auto-name if empty", width=220)
            dpg.add_button(label="Start", callback=on_recording_start)
            dpg.add_button(label="Stop", callback=on_recording_stop)
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="save path", tag="recording_path_input",
                               hint="blank = <Project>/Saved/URLab/Replays/<name>.json",
                               width=400)
            dpg.add_button(label="Save", callback=on_recording_save)

        dpg.add_separator()
        dpg.add_text("Replay")
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="path or session", tag="replay_target_input",
                               hint="absolute path or loaded session name", width=400)
            dpg.add_button(label="Play", callback=on_replay_play)
            dpg.add_button(label="Stop", callback=on_replay_stop)


def tick() -> None:
    pass
