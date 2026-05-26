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

"""Scene tab: editor-only world authoring.

Drives `client.scene.*` ops (Level / Import / Spawn / Edit-time
transform / Quick Component) plus the world Outliner. PIE control
lives in the always-visible sim toolbar (app.py); set_qpos / set_twist
live on the Runtime tab.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg

from ..log import log
from ..state import STATE
from urlab_client import URLabAsset, URLabRPCError
from .._helpers import parse_floats as _parse_floats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_client() -> bool:
    if not STATE.is_connected():
        log("not connected", error=True)
        return False
    return True


def _show_reply(label: str, reply: Any) -> None:
    if dpg.does_item_exist("scene_last_reply"):
        try:
            text = json.dumps(reply, indent=2, default=str)
        except Exception:
            text = repr(reply)
        dpg.set_value("scene_last_reply", f"{label}\n{text}")


# ---------------------------------------------------------------------------
# Level ops
# ---------------------------------------------------------------------------


def on_create_level(_s=None, _a=None) -> None:
    if not _require_client(): return
    name = (dpg.get_value("scene_level_name") or "").strip()
    if not name:
        log("create_level: enter a level name first", error=True); return
    try:
        STATE.client.scene.create_level(name)
        log(f"create_level({name!r}) -> ok")
        _show_reply("create_level", {"name": name})
    except URLabRPCError as exc:
        log(f"create_level failed [{exc.code}]: {exc.message}", error=True)


def on_load_level(_s=None, _a=None) -> None:
    if not _require_client(): return
    name = (dpg.get_value("scene_level_name") or "").strip()
    if not name:
        log("load_level: enter a level name or path first", error=True); return
    try:
        STATE.client.scene.load_level(name)
        log(f"load_level({name!r}) -> ok")
        _show_reply("load_level", {"name_or_path": name})
    except URLabRPCError as exc:
        log(f"load_level failed [{exc.code}]: {exc.message}", error=True)


def on_save_level(_s=None, _a=None) -> None:
    if not _require_client(): return
    try:
        STATE.client.scene.save_level()
        log("save_level -> ok")
        _show_reply("save_level", "ok")
    except URLabRPCError as exc:
        log(f"save_level failed [{exc.code}]: {exc.message}", error=True)


# ---------------------------------------------------------------------------
# Import XML
# ---------------------------------------------------------------------------


def on_discover_blueprints(_s=None, _a=None) -> None:
    """Populate the spawn-actor dropdown with every BP in
    /Game/MuJoCoImports. One-time discovery -- the user clicks Refresh
    when they re-run import_xml on a new asset."""
    if not _require_client(): return
    try:
        bps = STATE.client.outliner.list_blueprints()
    except URLabRPCError as exc:
        log(f"list_blueprints [{exc.code}]: {exc.message}", error=True)
        return
    STATE.scene_tab.blueprints = bps
    short_names = [b.short_name for b in bps]
    if dpg.does_item_exist("spawn_bp_combo"):
        dpg.configure_item("spawn_bp_combo", items=short_names)
    log(f"list_blueprints -> {len(bps)} BP(s) under /Game/MuJoCoImports")


def on_blueprint_picked(_sender, app_data, _user_data) -> None:
    """Combo callback: copy the selected BP's class path into the
    spawn-actor blueprint text field."""
    short = str(app_data or "")
    match = next((b for b in STATE.scene_tab.blueprints
                  if b.short_name == short), None)
    if match and dpg.does_item_exist("scene_spawn_blueprint"):
        dpg.set_value("scene_spawn_blueprint", match.class_path)


def on_xml_picked(_sender, app_data) -> None:
    """File-dialog callback for the Import XML section."""
    path = (app_data or {}).get("file_path_name", "")
    if not path:
        return
    dpg.set_value("scene_import_path", path)


def on_import_xml(_s=None, _a=None) -> None:
    if not _require_client(): return
    path = (dpg.get_value("scene_import_path") or "").strip()
    if not path:
        log("import_xml: pick an .xml first", error=True); return
    force = bool(dpg.get_value("scene_import_force"))
    try:
        bp_obj = STATE.client.scene.import_xml(path, force_reimport=force)
        log(
            f"import_xml({path!r}, force={force}) -> "
            f"{bp_obj.class_path}  imported_now={bp_obj.imported_now}"
        )
        if bp_obj.class_path:
            dpg.set_value("scene_spawn_blueprint", bp_obj.class_path)  # auto-fill spawn
        _show_reply("import_xml", bp_obj.__dict__)
    except URLabRPCError as exc:
        log(f"import_xml failed [{exc.code}]: {exc.message}", error=True)


# ---------------------------------------------------------------------------
# Spawn / destroy / transform
# ---------------------------------------------------------------------------


def on_spawn_actor(_s=None, _a=None) -> None:
    if not _require_client(): return
    bp = (dpg.get_value("scene_spawn_blueprint") or "").strip()
    aid = (dpg.get_value("scene_spawn_actor_id") or "").strip()
    if not bp:
        log("spawn_actor: blueprint path empty (run import_xml first)", error=True); return
    if not aid:
        log("spawn_actor: actor_id empty", error=True); return
    loc = _parse_floats(dpg.get_value("scene_spawn_location"), 3, [0.0, 0.0, 0.0])
    rot_quat_text = (dpg.get_value("scene_spawn_rot_quat") or "").strip()
    rot_eul_text  = (dpg.get_value("scene_spawn_rot_euler") or "").strip()
    rot_quat = _parse_floats(rot_quat_text, 4, [0.0, 0.0, 0.0, 1.0]) if rot_quat_text else None
    rot_eul  = _parse_floats(rot_eul_text,  3, [0.0, 0.0, 0.0])      if rot_eul_text  else None
    scale = _parse_floats(dpg.get_value("scene_spawn_scale"), 3, [1.0, 1.0, 1.0])
    try:
        ed = STATE.client.scene.spawn_actor(
            blueprint=bp, actor_id=aid,
            location=loc,
            rotation_quat=rot_quat, rotation_euler=rot_eul,
            scale=scale,
        )
        log(
            f"spawn_actor({aid!r}) -> {ed.actor_name} "
            f"(was_existing={ed.was_existing}, "
            f"requires_pie_restart={ed.requires_pie_restart})"
        )
        _show_reply("spawn_actor", ed.__dict__)
    except URLabRPCError as exc:
        log(f"spawn_actor failed [{exc.code}]: {exc.message}", error=True)


def on_spawn_light(_s=None, _a=None) -> None:
    if not _require_client(): return
    kind = (dpg.get_value("scene_light_kind") or "directional").strip()
    aid = (dpg.get_value("scene_light_actor_id") or "").strip()
    loc = _parse_floats(dpg.get_value("scene_light_location"), 3, [0.0, 0.0, 0.0])
    eul = _parse_floats(dpg.get_value("scene_light_euler"),    3, [0.0, 0.0, 0.0])
    intensity = float(dpg.get_value("scene_light_intensity") or 5000.0)
    color = _parse_floats(dpg.get_value("scene_light_color"), 3, [1.0, 1.0, 1.0])
    try:
        ed = STATE.client.scene.spawn_light(
            kind=kind, actor_id=aid,
            location=loc, rotation_euler=eul,
            intensity=intensity, color=color,
        )
        log(f"spawn_light({kind}, {aid!r}) -> {ed.actor_name}")
        _show_reply("spawn_light", ed.__dict__)
    except URLabRPCError as exc:
        log(f"spawn_light failed [{exc.code}]: {exc.message}", error=True)


def on_destroy_actor(_s=None, _a=None) -> None:
    if not _require_client(): return
    aid = (dpg.get_value("scene_destroy_actor_id") or "").strip()
    if not aid:
        log("destroy_actor: actor_id empty", error=True); return
    try:
        STATE.client.scene.destroy_actor(aid)
        log(f"destroy_actor({aid!r}) -> ok")
        _show_reply("destroy_actor", {"target": aid})
    except URLabRPCError as exc:
        log(f"destroy_actor failed [{exc.code}]: {exc.message}", error=True)


def on_set_actor_transform(_s=None, _a=None) -> None:
    if not _require_client(): return
    aid = (dpg.get_value("scene_xform_actor_id") or "").strip()
    if not aid:
        log("set_actor_transform: actor_id empty", error=True); return
    loc_text = (dpg.get_value("scene_xform_location") or "").strip()
    rot_quat_text = (dpg.get_value("scene_xform_rot_quat") or "").strip()
    rot_eul_text  = (dpg.get_value("scene_xform_rot_euler") or "").strip()
    if not loc_text and not rot_quat_text and not rot_eul_text:
        log("set_actor_transform: provide at least one of location / rotation", error=True); return
    loc = _parse_floats(loc_text, 3, [0.0, 0.0, 0.0]) if loc_text else None
    rot_quat = _parse_floats(rot_quat_text, 4, [0.0, 0.0, 0.0, 1.0]) if rot_quat_text else None
    rot_eul  = _parse_floats(rot_eul_text,  3, [0.0, 0.0, 0.0])      if rot_eul_text  else None
    try:
        STATE.client.scene.set_actor_transform(
            aid, location=loc, rotation_quat=rot_quat, rotation_euler=rot_eul,
        )
        log(f"set_actor_transform({aid!r}) -> ok")
        _show_reply("set_actor_transform", {
            "target": aid,
            "location": loc, "rotation_quat": rot_quat, "rotation_euler": rot_eul,
        })
    except URLabRPCError as exc:
        log(f"set_actor_transform failed [{exc.code}]: {exc.message}", error=True)


# ---------------------------------------------------------------------------
# PIE control
# ---------------------------------------------------------------------------


# PIE control (begin_pie / stop_pie / pie_status) lives in the
# always-visible sim toolbar (app.py). set_qpos / set_twist /
# set_control_source live on the Runtime tab.


# ---------------------------------------------------------------------------
# Apply scene (bulk)
# ---------------------------------------------------------------------------


def _row_tag(rid: int, field: str) -> str:
    return f"scene_row_{rid}_{field}"


def _redraw_apply_scene_rows() -> None:
    """Rebuild the rows region from the in-memory STATE.scene_tab.apply_scene_rows list."""
    if not dpg.does_item_exist("scene_apply_rows"):
        return
    dpg.delete_item("scene_apply_rows", children_only=True)
    for row in STATE.scene_tab.apply_scene_rows:
        rid = row["rid"]
        with dpg.group(horizontal=True, parent="scene_apply_rows"):
            dpg.add_input_text(default_value=row["actor_id"],
                               tag=_row_tag(rid, "aid"), width=120, hint="actor_id")
            dpg.add_input_text(default_value=row["xml"],
                               tag=_row_tag(rid, "xml"), width=380, hint="xml path")
            dpg.add_button(label="...", callback=_make_pick_xml_for_row(rid))
            dpg.add_input_text(default_value=row["location"],
                               tag=_row_tag(rid, "loc"), width=140, hint="x y z")
            dpg.add_input_text(default_value=row["rotation_euler"],
                               tag=_row_tag(rid, "eul"), width=140, hint="rx ry rz (deg)")
            dpg.add_button(label="-", callback=_make_remove_row(rid))


def _make_pick_xml_for_row(rid: int):
    def _cb(_s=None, _a=None) -> None:
        # File dialog for THIS row's xml field. Reuse a single shared dialog.
        dpg.configure_item(
            "scene_apply_xml_dialog",
            user_data=rid,
            show=True,
        )
    return _cb


def _on_apply_xml_picked(sender, app_data, user_data) -> None:
    rid = user_data
    path = (app_data or {}).get("file_path_name", "")
    if not path or rid is None:
        return
    tag = _row_tag(rid, "xml")
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, path)


def _make_remove_row(rid: int):
    def _cb(_s=None, _a=None) -> None:
        STATE.scene_tab.apply_scene_rows = [r for r in STATE.scene_tab.apply_scene_rows if r["rid"] != rid]
        _redraw_apply_scene_rows()
    return _cb


def on_apply_scene_add_row(_s=None, _a=None) -> None:
    rid = STATE.scene_tab.next_row_id
    STATE.scene_tab.next_row_id += 1
    STATE.scene_tab.apply_scene_rows.append({
        "rid": rid,
        "actor_id": f"robot_{len(STATE.scene_tab.apply_scene_rows) + 1}",
        "xml": "",
        "location": "0 0 0",
        "rotation_euler": "",
    })
    _redraw_apply_scene_rows()


def on_apply_scene_clear_rows(_s=None, _a=None) -> None:
    STATE.scene_tab.apply_scene_rows.clear()
    _redraw_apply_scene_rows()


# ---------------------------------------------------------------------------
# Outliner
# ---------------------------------------------------------------------------


def _actor_target(actor: Any) -> tuple[str, bool]:
    """Pick the best handle for the :class:`ActorInfo`: actor_id if
    present (wire arg), else fall back to actor name with by_name=True."""
    aid = (actor.actor_id or "").strip()
    if aid:
        return aid, False
    return str(actor.name or ""), True


def _refresh_outliner_list() -> None:
    """Rebuild the actor list region from STATE.scene_tab.actors."""
    if not dpg.does_item_exist("outliner_rows"):
        return
    dpg.delete_item("outliner_rows", children_only=True)
    sel_name = STATE.scene_tab.selected.name if STATE.scene_tab.selected else None
    for actor in STATE.scene_tab.actors:
        name = actor.name
        # Prefer the user-visible "Actor Label" (what UE's outliner
        # shows). Falls back to the raw UE actor name when label is
        # empty (which can happen pre-PIE for some default actors).
        label_str = actor.label or name
        aid = actor.actor_id or ""
        flags = []
        if actor.is_articulation:       flags.append("art")
        if actor.has_quick_convert:     flags.append("qc")
        if actor.is_light:              flags.append("light")
        if actor.is_static_mesh_actor:  flags.append("smesh")
        flag_str = f"  [{','.join(flags)}]" if flags else ""
        # Format: "<actor label> [flags]  id=<actor_id>"
        row_label = f"{label_str}{flag_str}"
        if aid:
            row_label = f"{label_str}{flag_str}  id={aid}"
        dpg.add_selectable(
            label=row_label,
            default_value=(name == sel_name),
            parent="outliner_rows",
            user_data=actor,
            callback=_on_outliner_row_clicked,
        )


def _refresh_outliner_inspector() -> None:
    if not dpg.does_item_exist("outliner_inspector"):
        return
    if STATE.scene_tab.selected is None:
        dpg.set_value("outliner_inspector", "(no actor selected)")
        return
    a = STATE.scene_tab.selected
    lines = [
        f"label:          {a.label or '(none)'}",
        f"name:           {a.name}",
        f"class:          {a.actor_class}",
        f"actor_id:       {a.actor_id or '(none)'}",
        f"location:       {a.location}",
        f"rotation_quat:  {a.rotation_quat}",
        f"is_articulation: {a.is_articulation}",
        f"is_light:        {a.is_light}",
        f"is_static_mesh:  {a.is_static_mesh_actor}",
        f"has_quick_convert: {a.has_quick_convert}",
    ]
    if a.has_quick_convert:
        lines.append("quick_convert:")
        for k, v in (
            ("static",           a.static),
            ("complex_mesh",     a.complex_mesh),
            ("coacd_threshold",  a.coacd_threshold),
            ("driven_by_unreal", a.driven_by_unreal),
            ("friction",         a.friction),
        ):
            if v is not None:
                lines.append(f"  {k}: {v}")
    dpg.set_value("outliner_inspector", "\n".join(lines))


def _propagate_selection_to_sections() -> None:
    """When the outliner selection changes, push the target into the
    destroy / transform / qpos sections so the user doesn't have to
    retype it. Also (re)configures the Quick Component section: hides
    it for AMjArticulation actors (those have their own physics setup),
    and pre-fills the controls with the actor's existing
    UMjQuickConvertComponent settings when present."""
    target = ""
    by_name_qpos = False
    sel = STATE.scene_tab.selected
    if sel is not None:
        target, by_name_qpos = _actor_target(sel)

    for tag in (
        "scene_destroy_actor_id",
        "scene_xform_actor_id",
        "runtime_qpos_target",   # Cross-tab: outliner selection drives the Runtime tab's qpos.
    ):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, target)
    if dpg.does_item_exist("runtime_qpos_by_name"):
        dpg.set_value("runtime_qpos_by_name", by_name_qpos)

    # Quick Component section visibility + pre-fill.
    show_qc = sel is not None and not sel.is_articulation
    if dpg.does_item_exist("qc_section"):
        dpg.configure_item("qc_section", show=show_qc)
    if dpg.does_item_exist("qc_status"):
        if sel is None:
            dpg.set_value("qc_status", "(no actor selected)")
        elif sel.is_articulation:
            dpg.set_value("qc_status",
                "selection is AMjArticulation -- quick convert disabled")
        elif sel.has_quick_convert:
            dpg.set_value("qc_status",
                f"actor has quick_convert: static={sel.static} "
                f"complex_mesh={sel.complex_mesh} "
                f"driven_by_unreal={sel.driven_by_unreal}")
        else:
            dpg.set_value("qc_status", "no quick_convert on this actor")

    # Pre-fill checkboxes / fields from current settings if any.
    pairs = [
        ("qc_static",            bool(sel.static)            if sel and sel.static            is not None else False),
        ("qc_complex_mesh",      bool(sel.complex_mesh)      if sel and sel.complex_mesh      is not None else False),
        ("qc_driven_by_unreal",  bool(sel.driven_by_unreal)  if sel and sel.driven_by_unreal  is not None else False),
        ("qc_coacd_threshold",   float(sel.coacd_threshold)  if sel and sel.coacd_threshold   is not None else 0.05),
    ]
    for tag, val in pairs:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)
    if dpg.does_item_exist("qc_friction"):
        f = sel.friction if (sel and sel.friction) else (1.0, 1.0, 1.0)
        dpg.set_value("qc_friction", " ".join(f"{v:g}" for v in f))


def _on_outliner_row_clicked(_sender, _value, user_data) -> None:
    # user_data is the ActorInfo instance pinned to this row. Keep the
    # reference (no need to copy a frozen-style dataclass).
    STATE.scene_tab.selected = user_data
    _refresh_outliner_list()  # update selection highlight
    _refresh_outliner_inspector()
    _propagate_selection_to_sections()
    # Auto-highlight in UE editor viewport too.
    if STATE.is_connected():
        target, by_name = _actor_target(STATE.scene_tab.selected)
        if target:
            try:
                STATE.client.outliner.select_actor(target, by_name=by_name)
            except URLabRPCError as exc:
                log(f"select_actor [{exc.code}]: {exc.message}", error=True)


def _refresh_outliner_now() -> None:
    """Synchronous list_actors call + UI refresh.

    Tolerates any transport / RPC error: the auto-poll fires from
    ``tick()`` every frame, so one failure (e.g. server restart, PIE
    transition, transient timeout) must NOT crash the UI loop.
    """
    if not STATE.is_connected():
        return
    try:
        actors = STATE.client.outliner.list_actors()
    except URLabRPCError as exc:
        log(f"list_actors [{exc.code}]: {exc.message}", error=True)
        return
    except Exception as exc:
        # ZMQ EAGAIN (recv timeout), socket reset, etc. The transport
        # already drops + recreates its REQ socket on error so the next
        # poll has a fresh socket; just swallow this one and retry.
        log(f"list_actors transport error ({type(exc).__name__}): {exc}",
            error=True)
        return
    STATE.scene_tab.actors = actors
    # Re-resolve the current selection by name so we keep showing fresh
    # transform / quick_convert state for the same actor.
    if STATE.scene_tab.selected is not None:
        sel_name = STATE.scene_tab.selected.name
        match = next((a for a in actors if a.name == sel_name), None)
        STATE.scene_tab.selected = match  # may become None if the actor was destroyed
    _refresh_outliner_list()
    _refresh_outliner_inspector()
    _propagate_selection_to_sections()
    if dpg.does_item_exist("outliner_status"):
        dpg.set_value("outliner_status",
            f"{len(actors)} actor(s) listed")


def on_outliner_refresh(_s=None, _a=None) -> None:
    if not _require_client(): return
    _refresh_outliner_now()


def on_outliner_highlight_in_editor(_s=None, _a=None) -> None:
    if not _require_client(): return
    sel = STATE.scene_tab.selected
    if sel is None:
        log("outliner: pick a row first", error=True); return
    target, by_name = _actor_target(sel)
    if not target:
        log("outliner: selected actor has no name/actor_id", error=True); return
    try:
        STATE.client.outliner.select_actor(target, by_name=by_name)
        log(f"select_actor({target!r}, by_name={by_name}) -> ok")
    except URLabRPCError as exc:
        log(f"select_actor [{exc.code}]: {exc.message}", error=True)


def on_outliner_destroy_selected(_s=None, _a=None) -> None:
    if not _require_client(): return
    sel = STATE.scene_tab.selected
    if sel is None:
        log("outliner: pick a row first", error=True); return
    target, by_name = _actor_target(sel)
    try:
        STATE.client.scene.destroy_actor(target, by_name=by_name)
        log(f"destroy_actor({target!r}, by_name={by_name}) -> ok")
        _refresh_outliner_now()
    except URLabRPCError as exc:
        log(f"destroy_actor [{exc.code}]: {exc.message}", error=True)


def on_outliner_use_for_xform(_s=None, _a=None) -> None:
    sel = STATE.scene_tab.selected
    if sel is None:
        log("outliner: pick a row first", error=True); return
    target, _ = _actor_target(sel)
    if dpg.does_item_exist("scene_xform_actor_id"):
        dpg.set_value("scene_xform_actor_id", target)
    log(f"transform target = {target!r}")


def on_outliner_use_for_qpos(_s=None, _a=None) -> None:
    """Outliner -> Runtime tab: copy selection's actor_id (or actor
    name) into the Runtime tab's qpos target field. The user still
    needs to switch to the Runtime tab to enter qpos values + click
    Apply."""
    sel = STATE.scene_tab.selected
    if sel is None:
        log("outliner: pick a row first", error=True); return
    target, by_name = _actor_target(sel)
    if dpg.does_item_exist("runtime_qpos_target"):
        dpg.set_value("runtime_qpos_target", target)
    if dpg.does_item_exist("runtime_qpos_by_name"):
        dpg.set_value("runtime_qpos_by_name", by_name)
    log(f"outliner -> Runtime tab: qpos target set to {target!r}")
    log(f"qpos target = {target!r} (by_name={by_name})")


def on_quick_convert_apply(_s=None, _a=None) -> None:
    if not _require_client(): return
    sel = STATE.scene_tab.selected
    if sel is None:
        log("quick_convert: pick a row first", error=True); return
    target, by_name = _actor_target(sel)
    payload = dict(
        static=bool(dpg.get_value("qc_static")),
        complex_mesh=bool(dpg.get_value("qc_complex_mesh")),
        coacd_threshold=float(dpg.get_value("qc_coacd_threshold") or 0.05),
        driven_by_unreal=bool(dpg.get_value("qc_driven_by_unreal")),
        friction=_parse_floats(dpg.get_value("qc_friction"), 3, [1.0, 1.0, 1.0]),
    )
    try:
        STATE.client.outliner.add_quick_convert(target, by_name=by_name, **payload)
        log(f"add_quick_convert({target!r}) -> ok")
        _show_reply("add_quick_convert", {"target": target, **payload})
        _refresh_outliner_now()
    except URLabRPCError as exc:
        log(f"add_quick_convert [{exc.code}]: {exc.message}", error=True)


def _nudge_selected(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
    if not _require_client(): return
    sel = STATE.scene_tab.selected
    if sel is None:
        log("nudge: pick a row first", error=True); return
    target, by_name = _actor_target(sel)
    if not target:
        log("nudge: selected actor has no name/actor_id", error=True); return
    cur = sel.location or (0.0, 0.0, 0.0)
    new_loc = (
        float(cur[0]) + dx,
        float(cur[1]) + dy,
        float(cur[2]) + dz,
    )
    try:
        STATE.client.scene.set_actor_transform(target, by_name=by_name, location=new_loc)
        log(f"nudge {target!r}  delta=({dx:+g}, {dy:+g}, {dz:+g})  -> {new_loc}")
        _refresh_outliner_now()
    except URLabRPCError as exc:
        log(f"nudge [{exc.code}]: {exc.message}", error=True)


def on_nudge_x_neg(_s=None, _a=None) -> None:
    step = float(dpg.get_value("outliner_nudge_step") or 0.1)
    _nudge_selected(dx=-step)


def on_nudge_x_pos(_s=None, _a=None) -> None:
    step = float(dpg.get_value("outliner_nudge_step") or 0.1)
    _nudge_selected(dx=+step)


def on_nudge_y_neg(_s=None, _a=None) -> None:
    step = float(dpg.get_value("outliner_nudge_step") or 0.1)
    _nudge_selected(dy=-step)


def on_nudge_y_pos(_s=None, _a=None) -> None:
    step = float(dpg.get_value("outliner_nudge_step") or 0.1)
    _nudge_selected(dy=+step)


def on_nudge_z_neg(_s=None, _a=None) -> None:
    step = float(dpg.get_value("outliner_nudge_step") or 0.1)
    _nudge_selected(dz=-step)


def on_nudge_z_pos(_s=None, _a=None) -> None:
    step = float(dpg.get_value("outliner_nudge_step") or 0.1)
    _nudge_selected(dz=+step)


def on_quick_convert_remove(_s=None, _a=None) -> None:
    if not _require_client(): return
    sel = STATE.scene_tab.selected
    if sel is None:
        log("quick_convert: pick a row first", error=True); return
    target, by_name = _actor_target(sel)
    try:
        STATE.client.outliner.remove_quick_convert(target, by_name=by_name)
        log(f"remove_quick_convert({target!r}) -> ok")
        _show_reply("remove_quick_convert", {"target": target})
        _refresh_outliner_now()
    except URLabRPCError as exc:
        log(f"remove_quick_convert [{exc.code}]: {exc.message}", error=True)


def on_apply_scene(_s=None, _a=None) -> None:
    if not _require_client(): return
    level = (dpg.get_value("scene_apply_level") or "").strip()
    if not level:
        log("apply_scene: enter a level name first", error=True); return
    if not STATE.scene_tab.apply_scene_rows:
        log("apply_scene: add at least one asset row first", error=True); return

    specs: List[URLabAsset] = []
    for row in STATE.scene_tab.apply_scene_rows:
        rid = row["rid"]
        aid = (dpg.get_value(_row_tag(rid, "aid")) or "").strip()
        xml = (dpg.get_value(_row_tag(rid, "xml")) or "").strip()
        if not aid or not xml:
            log(f"apply_scene: row {rid} missing actor_id or xml", error=True); return
        loc = _parse_floats(dpg.get_value(_row_tag(rid, "loc")), 3, [0.0, 0.0, 0.0])
        eul_text = (dpg.get_value(_row_tag(rid, "eul")) or "").strip()
        eul = _parse_floats(eul_text, 3, [0.0, 0.0, 0.0]) if eul_text else None
        specs.append(URLabAsset(
            actor_id=aid, xml=xml,
            location=tuple(loc),
            rotation_euler=tuple(eul) if eul else None,
        ))
    save = bool(dpg.get_value("scene_apply_save"))
    try:
        result = STATE.client.scene.apply_scene(level, specs, save=save)
        log(f"apply_scene({level!r}) -> {len(result)} asset(s) spawned")
        _show_reply("apply_scene", {k: v.__dict__ for k, v in result.items()})
    except URLabRPCError as exc:
        log(f"apply_scene failed [{exc.code}]: {exc.message}", error=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _bind(tag: str, theme: str) -> None:
    """Helper: bind a recently-created widget to a theme tag."""
    if dpg.does_item_exist(tag):
        dpg.bind_item_theme(tag, theme)


def _section_header(label: str) -> None:
    """Render a styled section header. dearpygui 1.x doesn't have
    add_separator_text, so we hand-roll one: a separator above, then the
    label in the accent color (slightly larger feel via spacing)."""
    dpg.add_spacer(height=4)
    dpg.add_separator()
    dpg.add_text(label, color=(110, 185, 255))


def build(parent: str) -> None:
    """Add the Scene tab body under ``parent`` (a dpg tab tag).

    Layout (top → bottom):
      1. World Outliner (centerpiece): scrollable list of actors with
         inspector, nudge, and selection-targeted action buttons. The
         Quick Component sub-panel auto-shows when the selection is a
         regular UE actor (not an AMjArticulation).
      2. Asset workflow: Level | Import XML | Spawn actor / Spawn light.
      3. Edit-time transform / destroy.
      4. Apply scene (collapsing — bulk authoring).
      5. Last reply panel.
    """

    # File dialogs — invisible, shown on demand from button callbacks.
    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=on_xml_picked, tag="scene_xml_dialog",
                         width=720, height=440):
        dpg.add_file_extension(".xml")
        dpg.add_file_extension(".*")

    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=_on_apply_xml_picked,
                         tag="scene_apply_xml_dialog",
                         width=720, height=440):
        dpg.add_file_extension(".xml")
        dpg.add_file_extension(".*")

    with dpg.group(parent=parent):
        # ── 1. World Outliner — the centerpiece ──────────────────────────
        _section_header("World outliner")
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Auto-refresh",
                             tag="outliner_auto_refresh",
                             default_value=True)
            dpg.add_button(label="Refresh now", callback=on_outliner_refresh)
            dpg.add_text("(disconnected)", tag="outliner_status",
                         color=(155, 165, 180))

        with dpg.group(horizontal=True):
            # ─ Left: scrollable list of actor rows.
            with dpg.child_window(width=560, height=300,
                                  tag="outliner_rows_window",
                                  border=True):
                with dpg.group(tag="outliner_rows"):
                    pass
            # ─ Right: inspector + actions for the selected row.
            with dpg.child_window(width=580, height=300, border=True):
                dpg.add_text("Selected actor", color=(110, 185, 255))
                dpg.add_input_text(tag="outliner_inspector",
                                   multiline=True, readonly=True,
                                   width=550, height=176,
                                   default_value="(no actor selected)")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Highlight in editor",
                                   callback=on_outliner_highlight_in_editor)
                    dpg.add_button(label="-> transform",
                                   callback=on_outliner_use_for_xform)
                    dpg.add_button(label="-> set_qpos",
                                   callback=on_outliner_use_for_qpos)
                    btn = dpg.add_button(label="Destroy",
                                         callback=on_outliner_destroy_selected)
                    dpg.bind_item_theme(btn, "urlab_theme_danger_button")

        # ─ Nudge controls for the selected actor.
        with dpg.group(horizontal=True):
            dpg.add_text("Nudge:", color=(155, 165, 180))
            dpg.add_input_float(tag="outliner_nudge_step",
                                default_value=0.1, width=90, step=0.0,
                                format="%.3f m")
            dpg.add_button(label="-X", callback=on_nudge_x_neg, width=40)
            dpg.add_button(label="+X", callback=on_nudge_x_pos, width=40)
            dpg.add_button(label="-Y", callback=on_nudge_y_neg, width=40)
            dpg.add_button(label="+Y", callback=on_nudge_y_pos, width=40)
            dpg.add_button(label="-Z", callback=on_nudge_z_neg, width=40)
            dpg.add_button(label="+Z", callback=on_nudge_z_pos, width=40)

        # ── 3. Quick Component (selection-aware) ─────────────────────────
        with dpg.group(tag="qc_section", show=False):
            _section_header("URLab quick component")
            dpg.add_text("(no actor selected)", tag="qc_status",
                         color=(155, 165, 180))
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="static",
                                 tag="qc_static", default_value=False)
                dpg.add_checkbox(label="complex mesh (CoACD)",
                                 tag="qc_complex_mesh", default_value=False)
                dpg.add_checkbox(label="driven by Unreal (mocap)",
                                 tag="qc_driven_by_unreal",
                                 default_value=False)
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="coacd threshold",
                                    tag="qc_coacd_threshold",
                                    default_value=0.05, width=140)
                dpg.add_input_text(label="friction",
                                   tag="qc_friction",
                                   default_value="1 1 1", width=160,
                                   hint="slide tor roll")
            with dpg.group(horizontal=True):
                btn = dpg.add_button(label="Apply quick component",
                                     callback=on_quick_convert_apply)
                dpg.bind_item_theme(btn, "urlab_theme_primary_button")
                btn = dpg.add_button(label="Remove",
                                     callback=on_quick_convert_remove)
                dpg.bind_item_theme(btn, "urlab_theme_danger_button")

        # ── 4. Asset workflow (Level / Import / Spawn / Edit) ────────────
        _section_header("Level")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="scene_level_name",
                               default_value="myscene", width=320,
                               hint="name or /Game/... path")
            dpg.add_button(label="Create", callback=on_create_level)
            dpg.add_button(label="Load",   callback=on_load_level)
            dpg.add_button(label="Save",   callback=on_save_level)

        _section_header("Import MJCF")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="scene_import_path",
                               width=480, hint="absolute path to .xml")
            dpg.add_button(label="Browse...",
                           callback=lambda: dpg.show_item("scene_xml_dialog"))
            dpg.add_checkbox(label="force reimport",
                             tag="scene_import_force", default_value=False)
            btn = dpg.add_button(label="Import", callback=on_import_xml)
            dpg.bind_item_theme(btn, "urlab_theme_primary_button")

        # Spawn + Light side by side.
        _section_header("Spawn")
        with dpg.group(horizontal=True):
            with dpg.child_window(width=560, height=210, border=True):
                dpg.add_text("Actor (from Blueprint)", color=(110, 185, 255))
                with dpg.group(horizontal=True):
                    dpg.add_combo([], tag="spawn_bp_combo", width=300,
                                  callback=on_blueprint_picked,
                                  default_value="",
                                  no_preview=False)
                    dpg.add_button(label="Refresh BPs",
                                   callback=on_discover_blueprints)
                dpg.add_input_text(tag="scene_spawn_blueprint", width=540,
                                   hint="BP class path (auto-filled by Import / Refresh)")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="scene_spawn_actor_id",
                                       width=160, hint="actor_id (e.g. robot_a)")
                    dpg.add_input_text(tag="scene_spawn_location",
                                       default_value="0 0 0",
                                       width=140, hint="x y z (m)")
                    dpg.add_input_text(tag="scene_spawn_scale",
                                       default_value="1 1 1",
                                       width=110, hint="scale")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="scene_spawn_rot_quat",
                                       width=180, hint="quat x y z w")
                    dpg.add_input_text(tag="scene_spawn_rot_euler",
                                       width=180, hint="euler rx ry rz (deg)")
                btn = dpg.add_button(label="Spawn actor",
                                     callback=on_spawn_actor)
                dpg.bind_item_theme(btn, "urlab_theme_primary_button")

            with dpg.child_window(width=560, height=180, border=True):
                dpg.add_text("Light", color=(110, 185, 255))
                with dpg.group(horizontal=True):
                    dpg.add_combo(["directional", "point", "spot"],
                                  tag="scene_light_kind",
                                  default_value="directional", width=140)
                    dpg.add_input_text(tag="scene_light_actor_id",
                                       width=160, hint="actor_id (e.g. sun)")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="scene_light_location",
                                       default_value="0 0 5", width=140,
                                       hint="x y z (m)")
                    dpg.add_input_text(tag="scene_light_euler",
                                       default_value="0 -45 0", width=140,
                                       hint="rx ry rz (deg)")
                with dpg.group(horizontal=True):
                    dpg.add_input_float(tag="scene_light_intensity",
                                        default_value=5000.0, width=120,
                                        step=0.0, format="%.0f cd")
                    dpg.add_input_text(tag="scene_light_color",
                                       default_value="1 1 1", width=140,
                                       hint="r g b (0..1)")
                btn = dpg.add_button(label="Spawn light",
                                     callback=on_spawn_light)
                dpg.bind_item_theme(btn, "urlab_theme_primary_button")

        # Manual transform/destroy (the outliner's "->" buttons fill these).
        _section_header("Edit-time transform / destroy")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="scene_xform_actor_id",
                               width=180, hint="actor_id")
            dpg.add_input_text(tag="scene_xform_location",
                               width=140, hint="x y z (m)")
            dpg.add_input_text(tag="scene_xform_rot_quat",
                               width=160, hint="quat x y z w")
            dpg.add_input_text(tag="scene_xform_rot_euler",
                               width=180, hint="euler rx ry rz (deg)")
            dpg.add_button(label="Apply transform",
                           callback=on_set_actor_transform)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="scene_destroy_actor_id",
                               width=180, hint="actor_id (destroy)")
            btn = dpg.add_button(label="Destroy actor",
                                 callback=on_destroy_actor)
            dpg.bind_item_theme(btn, "urlab_theme_danger_button")

        # set_qpos / set_twist / set_control_source live on the Runtime
        # tab. The outliner's "→ set_qpos" button forwards selection
        # there.

        # ── Last reply panel ─────────────────────────────────────────────
        _section_header("Last reply")
        dpg.add_input_text(tag="scene_last_reply",
                           multiline=True, readonly=True,
                           width=1140, height=120,
                           default_value="(connect + run an op)")


def tick() -> None:
    """Per-frame work for the Scene tab. Auto-polls list_actors at
    ``STATE.scene_tab.poll_interval_s`` cadence when the user hasn't disabled it
    via the checkbox. Other sections are event-driven.

    The auto-poll is *gated on the Scene tab being the active tab* --
    every tab's tick() runs every frame, and a hidden Scene tab firing
    `list_actors` continuously is wasteful, drowns the editor log, and
    competes with the actively-running tab (Policy, Cameras, etc.) for
    the bridge's REQ socket. The user's outliner state is stale only
    while the Scene tab isn't visible; the moment they switch back, the
    next tick will refresh.
    """
    if not STATE.is_connected():
        return
    if not dpg.does_item_exist("outliner_auto_refresh"):
        return
    if not dpg.get_value("outliner_auto_refresh"):
        return
    # Only poll when the Scene tab is the active tab. dpg's tab_bar
    # get_value returns the item *id* (int) of the active tab, not its
    # alias -- resolve to the alias via get_item_alias for the compare.
    if dpg.does_item_exist("tabs"):
        active_id = dpg.get_value("tabs")
        if active_id:
            active_alias = (
                active_id if isinstance(active_id, str)
                else (dpg.get_item_alias(active_id) or "")
            )
            if active_alias and active_alias != "tab_scene":
                return
    now = time.monotonic()
    if (now - STATE.scene_tab.last_poll_monotonic) < STATE.scene_tab.poll_interval_s:
        return
    STATE.scene_tab.last_poll_monotonic = now
    _refresh_outliner_now()
