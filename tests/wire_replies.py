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

"""Factory functions for the `*_ok` and error replies the UE side emits.

Centralises the dict literals that mirror server reply shapes so a
regression in any reply field propagates to its factory here, then to
every test that uses it — fix once, not in N test files.

These mirror the on-wire shape exactly. Field names match the UE
handler's `Reply->SetStringField(...)` calls; types are JSON-compatible
(strings, ints, floats, lists, dicts, None).

Coverage: every editor-only op + the runtime control ops the editor
tests poke at. Free-form per-op kwargs go through and are appended
unchanged so callers can override fields per-test."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Generic shapes
# ---------------------------------------------------------------------------


def error(code: str, message: str = "", *, op: str = "error") -> Dict[str, Any]:
    """An error reply. Any op handler can emit this in place of `<op>_ok`."""
    return {"op": op, "code": code, "message": message}


# ---------------------------------------------------------------------------
# scene namespace — level / asset / spawn
# ---------------------------------------------------------------------------


def import_xml_ok(*, blueprint_class_path: str = "",
                  blueprint_short_name: str = "",
                  imported_now: bool = True,
                  **extra: Any) -> Dict[str, Any]:
    return {
        "op": "import_xml_ok",
        "blueprint_class_path": blueprint_class_path,
        "blueprint_short_name": blueprint_short_name,
        "imported_now": imported_now,
        **extra,
    }


def op_started(*, job_id: str = "job_1", **extra: Any) -> Dict[str, Any]:
    """Async editor-op kickoff reply: the op runs on a game-thread job and the
    client polls op_status(job_id)."""
    return {"op": "op_started", "job_id": job_id, "state": "running", **extra}


def op_status_ok(*, job_id: str = "job_1", state: str = "running",
                 result: Optional[Dict[str, Any]] = None,
                 progress: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """op_status poll reply. ``state`` in {running, done, failed}; ``result`` is
    the original op reply once terminal."""
    out: Dict[str, Any] = {"op": "op_status_ok", "job_id": job_id, "state": state, **extra}
    if progress is not None:
        out["progress"] = progress
    if result is not None:
        out["result"] = result
    return out


def create_level_ok(*, level_path: str, **extra: Any) -> Dict[str, Any]:
    return {"op": "create_level_ok", "level_path": level_path, **extra}


def load_level_ok(*, level_path: str, **extra: Any) -> Dict[str, Any]:
    return {"op": "load_level_ok", "level_path": level_path, **extra}


def save_level_ok(*, level_path: str, **extra: Any) -> Dict[str, Any]:
    return {"op": "save_level_ok", "level_path": level_path, **extra}


def spawn_actor_ok(*, actor_id: str = "",
                   actor_name: str = "",
                   actor_path: str = "",
                   blueprint_class_path: str = "",
                   location: Optional[List[float]] = None,
                   rotation_quat: Optional[List[float]] = None,
                   requires_pie_restart: bool = False,
                   was_existing: bool = False,
                   **extra: Any) -> Dict[str, Any]:
    return {
        "op": "spawn_actor_ok",
        "actor_id": actor_id,
        "actor_name": actor_name,
        "actor_path": actor_path,
        "blueprint_class_path": blueprint_class_path,
        "location": list(location or [0.0, 0.0, 0.0]),
        "rotation_quat": list(rotation_quat or [0.0, 0.0, 0.0, 1.0]),
        "requires_pie_restart": requires_pie_restart,
        "was_existing": was_existing,
        **extra,
    }


def spawn_light_ok(*, actor_id: str = "",
                   actor_name: str = "",
                   actor_path: str = "",
                   light_class: str = "",
                   location: Optional[List[float]] = None,
                   rotation_quat: Optional[List[float]] = None,
                   intensity: float = 0.0,
                   color: Optional[List[float]] = None,
                   **extra: Any) -> Dict[str, Any]:
    return {
        "op": "spawn_light_ok",
        "actor_id": actor_id,
        "actor_name": actor_name,
        "actor_path": actor_path,
        "light_class": light_class,
        "location": list(location or [0.0, 0.0, 0.0]),
        "rotation_quat": list(rotation_quat or [0.0, 0.0, 0.0, 1.0]),
        "intensity": float(intensity),
        "color": list(color or [1.0, 1.0, 1.0]),
        **extra,
    }


def remove_actor_ok(*, target: str = "",
                     requires_pie_restart: bool = False,
                     **extra: Any) -> Dict[str, Any]:
    return {"op": "remove_actor_ok", "target": target,
            "requires_pie_restart": requires_pie_restart, **extra}


def set_actor_transform_ok(*, target: str = "", **extra: Any) -> Dict[str, Any]:
    return {"op": "set_actor_transform_ok", "target": target, **extra}


# ---------------------------------------------------------------------------
# runtime namespace — PIE control
# ---------------------------------------------------------------------------


def begin_pie_ok(*, state: str,
                 compile_error: str = "",
                 handshake_payload: Optional[Dict[str, Any]] = None,
                 **extra: Any) -> Dict[str, Any]:
    """PIE begin reply. `state` is one of: ready, compile_failed, timeout.

    On `state="ready"` callers should pass a `handshake_payload` so the
    bridge can re-absorb the new world. compile_failed / timeout replies
    omit the payload."""
    out: Dict[str, Any] = {
        "op": "begin_pie_ok",
        "state": state,
        "compile_error": compile_error,
    }
    if handshake_payload is not None:
        out["handshake_payload"] = handshake_payload
    out.update(extra)
    return out


def stop_pie_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "stop_pie_ok", **extra}


def pie_status_ok(*, state: str,
                  compile_error: str = "",
                  sim_time: Optional[float] = None,
                  **extra: Any) -> Dict[str, Any]:
    """PIE status reply. `state` is one of: off, compiling, compile_failed, ready."""
    out: Dict[str, Any] = {
        "op": "pie_status_ok",
        "state": state,
        "compile_error": compile_error,
    }
    if sim_time is not None:
        out["sim_time"] = sim_time
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# outliner namespace — list / select / quick-convert
# ---------------------------------------------------------------------------


def list_actors_ok(*, actors: Optional[List[Dict[str, Any]]] = None,
                   **extra: Any) -> Dict[str, Any]:
    return {"op": "list_actors_ok", "actors": list(actors or []), **extra}


def list_blueprints_ok(*, blueprints: Optional[List[Dict[str, Any]]] = None,
                       **extra: Any) -> Dict[str, Any]:
    return {"op": "list_blueprints_ok",
            "blueprints": list(blueprints or []),
            **extra}


def select_actor_ok(*, target: str = "", **extra: Any) -> Dict[str, Any]:
    return {"op": "select_actor_ok", "target": target, **extra}


def add_quick_convert_ok(*, target: str = "", **extra: Any) -> Dict[str, Any]:
    return {"op": "add_quick_convert_ok", "target": target, **extra}


def remove_quick_convert_ok(*, target: str = "", **extra: Any) -> Dict[str, Any]:
    return {"op": "remove_quick_convert_ok", "target": target, **extra}


def find_actors_ok(*, actors: Optional[List[Dict[str, Any]]] = None,
                   in_pie: bool = False,
                   **extra: Any) -> Dict[str, Any]:
    return {"op": "find_actors_ok",
            "actors": list(actors or []),
            "in_pie": bool(in_pie),
            **extra}


def get_actor_bounds_ok(*, actor_name: str = "",
                        min: Optional[List[float]] = None,
                        max: Optional[List[float]] = None,
                        center: Optional[List[float]] = None,
                        extents: Optional[List[float]] = None,
                        **extra: Any) -> Dict[str, Any]:
    return {"op": "get_actor_bounds_ok",
            "actor_name": actor_name,
            "min": list(min or [0.0, 0.0, 0.0]),
            "max": list(max or [0.0, 0.0, 0.0]),
            "center": list(center or [0.0, 0.0, 0.0]),
            "extents": list(extents or [0.0, 0.0, 0.0]),
            **extra}


def snapshot_ok(*, actors: Optional[List[Dict[str, Any]]] = None,
                in_pie: bool = False,
                level_path: str = "",
                **extra: Any) -> Dict[str, Any]:
    return {"op": "snapshot_ok",
            "actors": list(actors or []),
            "in_pie": bool(in_pie),
            "level_path": level_path,
            **extra}


def duplicate_actor_ok(*, actor_id: str = "",
                       actor_name: str = "",
                       actor_path: str = "",
                       blueprint_class_path: str = "",
                       **extra: Any) -> Dict[str, Any]:
    return {"op": "duplicate_actor_ok",
            "actor_id": actor_id,
            "actor_name": actor_name,
            "actor_path": actor_path,
            "blueprint_class_path": blueprint_class_path,
            **extra}


def actor_hierarchy_ok(*, root: Optional[Dict[str, Any]] = None,
                       **extra: Any) -> Dict[str, Any]:
    return {"op": "actor_hierarchy_ok",
            "root": root or {"name": "", "class": "", "location": [0, 0, 0], "children": []},
            **extra}


# ---------------------------------------------------------------------------
# sim namespace — runtime control replies (small set used by editor tests)
# ---------------------------------------------------------------------------


def step_ok(*, time: float = 0.0, step: int = 0,
            per_articulation: Optional[Dict[str, Any]] = None,
            sim_time: Optional[Dict[str, int]] = None,
            wall_time: Optional[Dict[str, int]] = None,
            **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "op": "step_ok",
        "time": float(time),
        "step": int(step),
        "per_articulation": dict(per_articulation or {}),
    }
    if sim_time is not None:
        out["sim_time"] = dict(sim_time)
    if wall_time is not None:
        out["wall_time"] = dict(wall_time)
    out.update(extra)
    return out


def per_articulation_block(*, qpos: Optional[List[float]] = None,
                           qvel: Optional[List[float]] = None,
                           ctrl: Optional[List[float]] = None,
                           act: Optional[List[float]] = None,
                           sensors: Optional[Dict[str, List[float]]] = None,
                           **extra: Any) -> Dict[str, Any]:
    """One articulation's slice of a step_ok per_articulation map."""
    out: Dict[str, Any] = {
        "qpos": list(qpos or []),
        "qvel": list(qvel or []),
        "ctrl": list(ctrl or []),
        "act": list(act or []),
        "sensors": dict(sensors or {}),
    }
    out.update(extra)
    return out


def reset_ok(*, time: float = 0.0, step: int = 0,
             **extra: Any) -> Dict[str, Any]:
    return {"op": "reset_ok", "time": float(time), "step": int(step), **extra}


def set_qpos_ok(*, target: str = "", **extra: Any) -> Dict[str, Any]:
    return {"op": "set_qpos_ok", "target": target, **extra}


def set_mocap_pose_ok(*, body: str = "",
                      pos: Optional[List[float]] = None,
                      quat: Optional[List[float]] = None,
                      **extra: Any) -> Dict[str, Any]:
    return {"op": "set_mocap_pose_ok",
            "body": body,
            "pos": list(pos or [0.0, 0.0, 0.0]),
            "quat": list(quat or [1.0, 0.0, 0.0, 0.0]),
            **extra}


def read_mocap_pose_ok(*, body: str = "",
                       pos: Optional[List[float]] = None,
                       quat: Optional[List[float]] = None,
                       **extra: Any) -> Dict[str, Any]:
    return {"op": "read_mocap_pose_ok",
            "body": body,
            "pos": list(pos or [0.0, 0.0, 0.0]),
            "quat": list(quat or [1.0, 0.0, 0.0, 0.0]),
            **extra}


def get_contacts_ok(*, n_contacts: int = 0,
                    truncated: bool = False,
                    contacts: Optional[List[Dict[str, Any]]] = None,
                    **extra: Any) -> Dict[str, Any]:
    return {"op": "get_contacts_ok",
            "n_contacts": int(n_contacts),
            "truncated": bool(truncated),
            "contacts": list(contacts or []),
            **extra}


def draw_marker_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "draw_marker_ok", **extra}


def draw_line_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "draw_line_ok", **extra}


def draw_box_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "draw_box_ok", **extra}


def draw_axes_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "draw_axes_ok", **extra}


def draw_arrow_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "draw_arrow_ok", **extra}


def clear_markers_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "clear_markers_ok", **extra}


def set_overlay_text_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "set_overlay_text_ok", **extra}


def _camera_pose_fields(*, location: Optional[List[float]] = None,
                        rotation_quat: Optional[List[float]] = None,
                        rotation_euler: Optional[List[float]] = None,
                        fov: float = 90.0) -> Dict[str, Any]:
    return {
        "location":       list(location       or [0.0, 0.0, 0.0]),
        "rotation_quat":  list(rotation_quat  or [0.0, 0.0, 0.0, 1.0]),
        "rotation_euler": list(rotation_euler or [0.0, 0.0, 0.0]),
        "fov":            float(fov),
    }


def set_camera_ok(**fields: Any) -> Dict[str, Any]:
    return {"op": "set_camera_ok", **_camera_pose_fields(**fields)}


def get_camera_ok(**fields: Any) -> Dict[str, Any]:
    return {"op": "get_camera_ok", **_camera_pose_fields(**fields)}


def frame_actor_ok(**fields: Any) -> Dict[str, Any]:
    return {"op": "frame_actor_ok", **_camera_pose_fields(**fields)}


def set_viewport_mode_ok(*, mode: str = "lit", **extra: Any) -> Dict[str, Any]:
    # Plugin replies with op="set_mode_ok" for set_viewport_mode (shared op name).
    return {"op": "set_mode_ok", "mode": mode, **extra}


def track_actor_ok(*, tracked_actor_path: str = "", **extra: Any) -> Dict[str, Any]:
    return {"op": "track_actor_ok",
            "tracked_actor_path": tracked_actor_path,
            **extra}


def untrack_ok(*, was_tracking: bool = False, **extra: Any) -> Dict[str, Any]:
    return {"op": "untrack_ok", "was_tracking": bool(was_tracking), **extra}


def list_keyframes_ok(*, keyframes: Optional[List[Dict[str, Any]]] = None,
                      **extra: Any) -> Dict[str, Any]:
    return {"op": "list_keyframes_ok",
            "keyframes": list(keyframes or []),
            **extra}


def spawn_grid_ok(*, count: int = 0,
                  blueprint_class_path: str = "",
                  actors: Optional[List[Dict[str, Any]]] = None,
                  requires_pie_restart: bool = False,
                  **extra: Any) -> Dict[str, Any]:
    return {"op": "spawn_grid_ok",
            "count": int(count),
            "blueprint_class_path": blueprint_class_path,
            "actors": list(actors or []),
            "requires_pie_restart": bool(requires_pie_restart),
            **extra}


def set_mode_ok(*, previous_mode: str = "live",
                current_mode: str,
                **extra: Any) -> Dict[str, Any]:
    return {"op": "set_mode_ok", "previous_mode": previous_mode,
            "current_mode": current_mode, **extra}


def configure_controller_ok(*, articulation: str = "",
                            **extra: Any) -> Dict[str, Any]:
    return {"op": "configure_controller_ok", "articulation": articulation, **extra}


def set_camera_delay_ok(*, cameras: Optional[Dict[str, Any]] = None,
                        **extra: Any) -> Dict[str, Any]:
    """Echo of the per-camera latency / capture-rate config the server applied
    (delay_s / jitter_s / clock / on_state_change / max_fps), keyed by canonical
    camera name."""
    return {"op": "set_camera_delay_ok", "cameras": dict(cameras or {}), **extra}


def set_sim_options_ok(*, options: Optional[Dict[str, Any]] = None,
                       **extra: Any) -> Dict[str, Any]:
    """Echo of mjOption fields written by set_sim_options. The server
    returns the resulting m_model->opt.* values (not just what the
    caller asked to set), but the test factory only fills the fields
    you pass in."""
    return {"op": "set_sim_options_ok", "options": dict(options or {}), **extra}


# ---------------------------------------------------------------------------
# recording / replay namespaces
# ---------------------------------------------------------------------------


def recording_start_ok(*, name: str = "",
                       max_duration_s: float = 0.0,
                       **extra: Any) -> Dict[str, Any]:
    return {"op": "recording_start_ok", "name": name,
            "max_duration_s": max_duration_s, **extra}


def recording_stop_ok(*, frame_count: int = 0,
                      sim_duration_s: float = 0.0,
                      **extra: Any) -> Dict[str, Any]:
    return {"op": "recording_stop_ok", "frame_count": frame_count,
            "sim_duration_s": sim_duration_s, **extra}


def recording_save_ok(*, absolute_path: str = "",
                      **extra: Any) -> Dict[str, Any]:
    return {"op": "recording_save_ok", "absolute_path": absolute_path, **extra}


def recording_clear_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "recording_clear_ok", **extra}


def replay_load_ok(*, name: str = "", **extra: Any) -> Dict[str, Any]:
    return {"op": "replay_load_ok", "name": name, **extra}


def replay_list_sessions_ok(*, sessions: Optional[List[str]] = None,
                            **extra: Any) -> Dict[str, Any]:
    return {"op": "replay_list_sessions_ok",
            "sessions": list(sessions or []), **extra}


def replay_set_active_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "replay_set_active_ok", **extra}


def replay_start_ok(*, active_session: str = "",
                    total_frames: int = 0,
                    **extra: Any) -> Dict[str, Any]:
    return {"op": "replay_start_ok", "active_session": active_session,
            "total_frames": total_frames, **extra}


def replay_stop_ok(**extra: Any) -> Dict[str, Any]:
    return {"op": "replay_stop_ok", **extra}
