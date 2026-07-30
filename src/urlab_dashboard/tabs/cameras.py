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

"""Cameras tab: live UE-rendered camera streams (Real / Seg / Depth).

The feeds render in native OpenCV (cv2) windows, NOT dearpygui textures.
Measurement showed the pipeline delivers frames at ~100ms content age (UE
capture -> client has it) even under the full multi-camera load, yet dpg's
texture/present path lagged the on-screen image by ~3s. cv2.imshow displays
the identical streams in real time, so the live feeds pop out into their own
windows while dpg keeps the control panel. See scripts/diag_content_latency.py
and scripts/diag_cv_view.py for the diagnosis.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from ..state import STATE
from urlab_client import URLabCameraView
from urlab_client.enums import CameraMode


_CAMERA_VIEWER_MAX = 480  # cv2 window edge, px


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


def _enabled() -> bool:
    if dpg.does_item_exist("cam_cv2_enabled"):
        return bool(dpg.get_value("cam_cv2_enabled"))
    return True


def ensure_textures() -> None:
    """Reconcile the set of open cv2 windows with the live cameras. Named
    here (rather than ensure_windows) to preserve the app.py call contract."""
    if not STATE.is_connected():
        return
    if not _enabled():
        release_textures()
        return
    views = _collect_views()
    # Drop windows for cameras that vanished or changed resolution.
    for key in list(STATE.cam_textures.keys()):
        new_view = views.get(key)
        if new_view is None or new_view.resolution != STATE.cam_textures[key]["wh"]:
            _destroy_window(key)
    # Open windows for new cameras.
    for key, view in views.items():
        if key in STATE.cam_textures:
            continue
        w, h = view.resolution
        if w <= 0 or h <= 0:
            continue
        scale = min(1.0, _CAMERA_VIEWER_MAX / max(w, h))
        cv2.namedWindow(key, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(key, int(w * scale), int(h * scale))
        STATE.cam_textures[key] = {"wh": (w, h), "last_count": -1}
    _refresh_panel_list()


def _destroy_window(key: str) -> None:
    try:
        cv2.destroyWindow(key)
    except Exception:
        pass
    STATE.cam_textures.pop(key, None)


def release_textures() -> None:
    for key in list(STATE.cam_textures.keys()):
        _destroy_window(key)
    STATE.cam_textures.clear()
    # cv2 needs a GUI pump to actually tear the windows down.
    try:
        cv2.waitKey(1)
    except Exception:
        pass
    _refresh_panel_list()


def _frame_to_bgr(view: URLabCameraView) -> Optional[np.ndarray]:
    """Convert the view's latest frame to a contiguous uint8 BGR image for
    cv2.imshow, or None if there is no usable frame."""
    frame = view.latest_frame
    if frame is None:
        return None
    w, h = view.resolution
    if view.mode == CameraMode.DEPTH:
        d = np.asarray(frame, dtype=np.float32)
        if d.shape != (h, w):
            return None
        finite = np.isfinite(d)
        if int(finite.sum()) < 16:
            gray = np.zeros((h, w), dtype=np.uint8)
        else:
            p_low, p_high = np.percentile(d[finite], (5.0, 95.0))
            span = max(float(p_high - p_low), 1e-3)
            norm = np.clip((d - p_low) / span, 0.0, 1.0)
            norm[~finite] = 0.0
            gray = (norm * 255.0).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape != (h, w, 4):
        return None
    # cv2 wants BGR. Real mode ships RGBA -> swap R/B; seg/instance ship BGRA
    # -> already BGR order, just drop alpha.
    if view.mode == CameraMode.REAL:
        bgr = arr[..., [2, 1, 0]]
    else:
        bgr = arr[..., [0, 1, 2]]
    return np.ascontiguousarray(bgr)


def _refresh_panel_list() -> None:
    if not dpg.does_item_exist("camera_streams_list"):
        return
    keys = sorted(STATE.cam_textures.keys())
    if keys:
        txt = "Open windows:\n  " + "\n  ".join(keys)
    elif not STATE.is_connected():
        txt = "(not connected)"
    elif not _enabled():
        txt = "(windows disabled)"
    else:
        txt = "(no cameras streaming)"
    dpg.set_value("camera_streams_list", txt)


def _on_toggle(_s=None, _a=None) -> None:
    if _enabled():
        ensure_textures()
    else:
        release_textures()


def build(parent: str) -> None:
    with dpg.group(parent=parent):
        dpg.add_text(
            "Live UE-rendered cameras render in separate OpenCV windows for "
            "real-time display (dearpygui's texture present path lags video "
            "by seconds). Real / Seg ship BGRA8; Depth is float32 normalized "
            "per-frame for display.",
            color=(140, 140, 140), wrap=900,
        )
        dpg.add_checkbox(label="Show camera windows (OpenCV)",
                         tag="cam_cv2_enabled", default_value=True,
                         callback=_on_toggle)
        dpg.add_separator()
        dpg.add_text("(not connected)", tag="camera_streams_list")


def tick() -> None:
    if not STATE.is_connected() or not _enabled() or not STATE.cam_textures:
        return
    for key, meta in STATE.cam_textures.items():
        view = None
        owner, _, cam = key.partition("/")
        if owner == "global":
            view = STATE.client.global_cameras.get(cam)
        else:
            art = STATE.client.articulations.get(owner)
            if art is not None:
                view = art.cameras.get(cam)
        if view is None:
            continue
        # Skip cameras with no new frame -- the window still shows the last one.
        if view.frame_count == meta["last_count"]:
            continue
        bgr = _frame_to_bgr(view)
        if bgr is None:
            continue
        cv2.imshow(key, bgr)
        meta["last_count"] = view.frame_count
    # Single GUI pump per tick drives every window's repaint.
    cv2.waitKey(1)
