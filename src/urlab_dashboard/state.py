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

"""Shared app state for the URLab UI.

Single ``AppState`` instance is exposed as ``STATE``. Tabs and helpers
import it; nothing in this module knows about dearpygui or mujoco.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from urlab_client import ActorInfo, BlueprintInfo, URLabClient


@dataclass
class PolicyRun:
    """Live state of a launched policy worker."""

    name: str = ""
    thread: Optional[threading.Thread] = None
    stop_flag: threading.Event = field(default_factory=threading.Event)
    step_count: int = 0
    last_error: str = ""
    started_at: float = 0.0
    # Per-pipeline control surface (set by launchers that swap in a
    # UI-driven controller; None for adapters that don't need one).
    control_handle: Any = None


@dataclass
class SceneTabState:
    """Scene-tab UI state. Wrapped here so a future worker thread can
    take a single lock around the whole bag instead of needing five
    separate ones. Reads/writes today happen on the dpg main thread, so
    the lock is defensive — held as a sentinel for the next change that
    wires worker-thread updates."""

    actors: List[ActorInfo] = field(default_factory=list)
    selected: Optional[ActorInfo] = None
    blueprints: List[BlueprintInfo] = field(default_factory=list)
    apply_scene_rows: List[Dict[str, Any]] = field(default_factory=list)
    next_row_id: int = 0
    last_poll_monotonic: float = 0.0
    poll_interval_s: float = 0.75
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class AppState:
    client: Optional[URLabClient] = None
    host: str = "tcp://127.0.0.1"
    step_port: int = 5559

    # User selections
    selected_articulation: Optional[str] = None

    # Latest known PIE state (off/compiling/compile_failed/timeout/ready).
    # Set by sim.start / sim.stop callbacks. Drives the PIE pill colour
    # in the always-visible status row.
    pie_state: Optional[str] = None

    # Raised off-thread when the live camera set may have changed. The windows
    # are OpenCV's, and it pumps them from the main loop, so they are opened
    # there too rather than wherever the discovery happened to land.
    cameras_dirty: bool = False

    # Set while the connect worker is in flight, so a second click on Connect
    # does not start a second handshake -- each one rotates the server session
    # and evicts the other.
    connecting: bool = False

    # Render flags
    render_request: bool = False

    # Open cv2 camera windows. Key = "<prefix>/<cam>" or "global/<cam>".
    # Value = {"wh": (w, h), "last_count": int}. (Named cam_textures for
    # historical app.py call-site compatibility; the live feeds render in
    # OpenCV windows, not dpg textures -- see tabs/cameras.py.)
    cam_textures: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Policy worker state
    policy_run: PolicyRun = field(default_factory=PolicyRun)

    # Scene-tab state (was 5 module-level globals in ui/tabs/scene.py)
    scene_tab: SceneTabState = field(default_factory=SceneTabState)

    # Background poller
    stop_poll: threading.Event = field(default_factory=threading.Event)

    def is_connected(self) -> bool:
        return self.client is not None and self.client.session_id is not None

    def status_line(self) -> str:
        if not self.is_connected():
            return "disconnected"
        c = self.client
        return (
            f"connected | mode={c.step_mode.value} | "
            f"sim_time={c.sim_time:.4f} | step={c.step_count} | "
            f"articulations={len(c.articulations)} | "
            f"entities={len(c.entities)}"
        )


STATE = AppState()
