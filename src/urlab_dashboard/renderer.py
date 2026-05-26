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

"""Embedded mujoco offscreen renderer.

Bound to the dpg dynamic_texture ``mj_render_texture``. The Renderer runs
on whichever thread calls ``render_into_texture`` — which must be the dpg
main thread on Windows because the GL context is bound to the thread that
constructed the Renderer (and dpg's texture-set call lives on the same
thread).

Known issues to be fixed in a follow-up: rendering setup ordering vs
dpg's GLFW init, and the inspector not refreshing in lock-step with the
Renderer. The class boundary makes either fix local.
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional

# Mujoco import + GL warmup must happen BEFORE dearpygui's GLFW init.
# Both the import order and the warmup are load-bearing on Windows.
try:
    import mujoco
    import numpy as np
    HAS_MUJOCO = True
except Exception:
    HAS_MUJOCO = False

_GL_WARMUP_OK = False
if HAS_MUJOCO:
    try:
        _m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><geom type='sphere' size='0.1'/>"
            "</body></worldbody></mujoco>"
        )
        _r = mujoco.Renderer(_m, height=4, width=4)
        _r.close()
        del _r, _m
        _GL_WARMUP_OK = True
    except Exception as exc:
        sys.stderr.write(
            f"[ui.renderer] mujoco GL warmup failed: {type(exc).__name__}: {exc}\n"
            "[ui.renderer] Embedded renderer will be disabled.\n"
        )
        sys.stderr.flush()

VIEWER_W = 320
VIEWER_H = 240


class EmbeddedRenderer:
    """Wraps a mujoco offscreen Renderer + free camera + dpg texture write."""

    def __init__(self):
        self._renderer = None  # type: Optional["mujoco.Renderer"]
        self._camera = None    # type: Optional["mujoco.MjvCamera"]
        self._failed = False
        # Last articulation we re-fitted the camera to, so distance is
        # only recomputed on selection change instead of every frame.
        self._fitted_for: Optional[str] = None

    @property
    def available(self) -> bool:
        return HAS_MUJOCO and _GL_WARMUP_OK

    @property
    def failed(self) -> bool:
        return self._failed

    def ensure(self, client) -> bool:
        """Build the Renderer if we don't have one and a model is available.
        Latches failure: a single failed construction disables the renderer
        for the session (no per-frame retry spam)."""
        if not self.available:
            return False
        if client is None or client.model is None:
            return False
        if self._renderer is not None:
            return True
        if self._failed:
            return False
        try:
            self._renderer = mujoco.Renderer(
                client.model, height=VIEWER_H, width=VIEWER_W
            )
        except Exception as exc:
            self._failed = True
            self._renderer = None
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(
                f"\n[ui.renderer] Renderer init failed: {type(exc).__name__}: {exc}\n"
            )
            sys.stderr.flush()
            return False
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(client.model, self._camera)
        return True

    def release(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        self._renderer = None
        self._camera = None
        self._failed = False  # allow retry after reconnect
        self._fitted_for = None

    def fit_to_articulation(self, client, art_prefix: Optional[str]) -> None:
        """Set camera distance from the articulation's own body extents,
        not the whole scene's stat.extent (which gets dominated by ground
        planes / large props on UE-compiled MJB). Idempotent across
        repeated calls for the same prefix."""
        if self._camera is None or client is None:
            return
        if art_prefix is None or art_prefix not in client.articulations:
            return
        if self._fitted_for == art_prefix:
            return
        art = client.articulations[art_prefix]
        if not art.bodies or client.data is None:
            return
        # Collect the live xpos for every body in the articulation.
        # client.data.xpos is current after _mirror_state_into_data's
        # mj_forward, so this measures the articulation as it stands now.
        positions = []
        for body in art.bodies.values():
            if body.id < 0:
                continue
            try:
                positions.append(np.asarray(client.data.xpos[body.id], dtype=np.float64))
            except Exception:
                continue
        if not positions:
            return
        arr = np.stack(positions, axis=0)
        extent = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
        # Distance ~3x extent gives a comfortable framing with default FOV.
        # Floor at 0.5m so a single-body / point-mass articulation isn't
        # nose-pressed against the camera.
        self._camera.distance = max(extent * 3.0, 0.5)
        self._fitted_for = art_prefix

    def lookat_for(self, client, art_prefix: Optional[str]):
        """Camera tracking target. Prefers the articulation's live root
        body world position (`art.root_pos_w` reads `client.data.xpos`,
        which `_mirror_state_into_data` keeps current via `mj_forward`).
        Falls back to per-body xpos (only populated at obs level full)
        and finally the static model pose."""
        if client is None or art_prefix is None or art_prefix not in client.articulations:
            return None
        art = client.articulations[art_prefix]
        try:
            root = art.root_pos_w
            if np.any(root):
                return root
        except Exception:
            pass
        for body in art.bodies.values():
            if body.xpos is not None:
                return body.xpos
        if client.model is not None and art.bodies:
            first = next(iter(art.bodies.values()))
            try:
                return client.model.body_pos[first.id]
            except Exception:
                return None
        return None

    def render_into_texture(self, client, art_prefix: Optional[str], texture_tag: str) -> Optional[str]:
        """Render one frame into the dpg dynamic texture. Returns an error
        string on failure (caller logs it), or None on success."""
        import dearpygui.dearpygui as dpg
        if not self.ensure(client):
            return None
        try:
            # model.stat.extent gets dominated by ground planes on
            # UE-compiled MJBs, so refit once per articulation selection.
            self.fit_to_articulation(client, art_prefix)
            lookat = self.lookat_for(client, art_prefix)
            if lookat is not None:
                self._camera.lookat[0] = float(lookat[0])
                self._camera.lookat[1] = float(lookat[1])
                self._camera.lookat[2] = float(lookat[2])
            self._renderer.update_scene(client.data, camera=self._camera)
            rgb = self._renderer.render()  # (H, W, 3) uint8
        except Exception as exc:
            return f"render failed: {exc}"

        rgba = np.empty((VIEWER_H, VIEWER_W, 4), dtype=np.float32)
        rgba[..., :3] = rgb.astype(np.float32) / 255.0
        rgba[..., 3] = 1.0
        if dpg.does_item_exist(texture_tag):
            dpg.set_value(texture_tag, rgba.ravel())
        return None
