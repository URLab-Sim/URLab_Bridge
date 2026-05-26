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

"""Cameras tab: live UE-rendered camera streams (Real / Seg / Depth)."""

from __future__ import annotations

from typing import Dict, Optional

import dearpygui.dearpygui as dpg
import numpy as np

from ..state import STATE
from urlab_client import URLabCameraView
from urlab_client.enums import CameraMode


_CAMERA_VIEWER_MAX = 320


def _collect_views() -> Dict[str, URLabCameraView]:
    out: Dict[str, URLabCameraView] = {}
    if not STATE.is_connected():
        return out
    for art in STATE.client.articulations.values():
        for cam_name, view in art.cameras.items():
            out[f"{art.prefix}/{cam_name}"] = view
    for cam_name, view in STATE.client.global_cameras.items():
        out[f"global/{cam_name}"] = view
    return out


def ensure_textures() -> None:
    if not STATE.is_connected() or not dpg.does_item_exist("texture_registry"):
        return
    views = _collect_views()
    # Drop stale.
    for key in list(STATE.cam_textures.keys()):
        meta = STATE.cam_textures[key]
        new_view = views.get(key)
        if new_view is None or new_view.resolution != meta["wh"]:
            for tag in (meta["tex"], meta["img"], meta["row"], meta["lbl"]):
                if dpg.does_item_exist(tag):
                    dpg.delete_item(tag)
            STATE.cam_textures.pop(key, None)
    # Create new.
    for key, view in views.items():
        if key in STATE.cam_textures:
            continue
        w, h = view.resolution
        if w <= 0 or h <= 0:
            continue
        tex = f"cam_tex::{key}"
        img = f"cam_img::{key}"
        row = f"cam_row::{key}"
        lbl = f"cam_lbl::{key}"
        init = [0.1, 0.1, 0.1, 1.0] * (w * h)
        dpg.add_dynamic_texture(w, h, init, parent="texture_registry", tag=tex)
        scale = min(1.0, _CAMERA_VIEWER_MAX / max(w, h))
        if dpg.does_item_exist("camera_streams_panel"):
            with dpg.group(horizontal=True, parent="camera_streams_panel", tag=row):
                dpg.add_image(tex, width=int(w * scale),
                              height=int(h * scale), tag=img)
                mode_str = view.mode.value if hasattr(view.mode, "value") else str(view.mode)
                dpg.add_text(f"{key}\n{w}x{h} {mode_str}", tag=lbl)
        STATE.cam_textures[key] = {
            "tex": tex, "img": img, "row": row, "lbl": lbl, "wh": (w, h),
        }


def release_textures() -> None:
    for meta in STATE.cam_textures.values():
        for tag in (meta["tex"], meta["img"], meta["row"], meta["lbl"]):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
    STATE.cam_textures.clear()


def _frame_to_rgba_float(view: URLabCameraView) -> Optional[np.ndarray]:
    frame = view.latest_frame
    if frame is None:
        return None
    w, h = view.resolution
    if view.mode == CameraMode.DEPTH:
        # 5/95 percentile normalisation so foreground spans the visible
        # gradient instead of being crushed by the skybox max.
        d = np.asarray(frame, dtype=np.float32)
        if d.shape != (h, w):
            return None
        finite = np.isfinite(d)
        if int(finite.sum()) < 16:
            d_norm = np.zeros_like(d)
        else:
            p_low, p_high = np.percentile(d[finite], (5.0, 95.0))
            span = max(float(p_high - p_low), 1e-3)
            d_norm = np.clip((d - p_low) / span, 0.0, 1.0)
            d_norm[~finite] = 0.0
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = d_norm
        rgba[..., 3] = 1.0
        return rgba
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape != (h, w, 4):
        return None
    rgba_u8 = arr if view.mode == CameraMode.REAL else arr[..., [2, 1, 0, 3]]
    return rgba_u8.astype(np.float32) / 255.0


def build(parent: str) -> None:
    with dpg.group(parent=parent):
        dpg.add_text(
            "Live UE-rendered cameras. Real / Seg ship BGRA8; Depth is "
            "float32 normalized per-frame for display.",
            color=(140, 140, 140),
        )
        dpg.add_group(tag="camera_streams_panel")


def tick() -> None:
    if not STATE.is_connected():
        return
    for key, meta in STATE.cam_textures.items():
        view = None
        if "/" in key:
            owner, cam = key.split("/", 1)
            if owner == "global":
                view = STATE.client.global_cameras.get(cam)
            else:
                art = STATE.client.articulations.get(owner)
                if art is not None:
                    view = art.cameras.get(cam)
        if view is None:
            continue
        rgba = _frame_to_rgba_float(view)
        if rgba is None:
            continue
        if dpg.does_item_exist(meta["tex"]):
            dpg.set_value(meta["tex"], rgba.ravel())
