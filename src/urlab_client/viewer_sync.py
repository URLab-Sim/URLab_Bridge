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

"""Resolve a MuJoCo abstract camera into the render server's user-camera pose.

Every MuJoCo viewer -- the classic ``mujoco.viewer`` passive window, the new
``mujoco.experimental.studio`` native app, and its web viewer -- drives the same
abstraction: an :class:`mujoco.MjvCamera` (a *free* camera the user flies, or one
*tracking* a body, or a *fixed* model camera). The render server's user camera,
by contrast, wants a concrete world eye pose: ``(pos, fwd, up)``.

:func:`free_camera_pose` bridges the two with pure camera math -- it imports no
viewer framework, so the same helper feeds any of those viewers via a thin shim
you write on top. The one convenience that *does* know a specific viewer,
:func:`pose_from_passive`, is a three-line reader for the classic passive handle
and is kept separate so the core stays viewer-agnostic.

The returned triple drops straight into
:meth:`urlab_client.RenderClient.render`'s ``user_pose`` argument::

    scene = viewer_sync.make_scene(model)                 # once
    pose = viewer_sync.free_camera_pose(model, data, handle.cam, scene)
    frames = rc.render_mjdata(model, data, cameras=["user"], user_pose=pose)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["make_scene", "free_camera_pose", "pose_from_passive"]

Pose = Tuple[np.ndarray, np.ndarray, np.ndarray]


def make_scene(model, maxgeom: Optional[int] = None):
    """A reusable :class:`mujoco.MjvScene` for :func:`free_camera_pose`.

    ``mjv_updateScene`` populates the scene's geoms and *raises* if they exceed
    ``maxgeom``, so size it for the model (plus headroom for MuJoCo's decor geoms).
    Build one and pass it every frame -- allocating per call is wasteful.
    """
    import mujoco  # lazy: only when the viewer bridge is actually used

    if maxgeom is None:
        # Model geoms + generous headroom for decor (contacts, frames, lights...).
        maxgeom = max(1000, 2 * int(model.ngeom) + 1000)
    return mujoco.MjvScene(model, maxgeom=maxgeom)


def free_camera_pose(model, data, cam, scene=None, opt=None) -> Pose:
    """``(pos, fwd, up)`` world eye pose for an :class:`mujoco.MjvCamera`.

    Resolves the abstract camera (free / tracking / fixed) to a concrete world
    eye exactly as a MuJoCo viewer does each frame: ``mjv_updateScene`` fills the
    two stereo GL cameras, whose midpoint is the cyclopean eye. Returns three
    ``float64`` ``(3,)`` arrays -- eye position, unit view direction, unit up.

    Pass a persistent ``scene`` (from :func:`make_scene`) to avoid per-call
    allocation; ``opt`` defaults to a plain :class:`mujoco.MjvOption`.
    """
    import mujoco  # lazy

    scn = scene if scene is not None else make_scene(model)
    options = opt if opt is not None else mujoco.MjvOption()
    mujoco.mjv_updateScene(
        model, data, options, None, cam,
        int(mujoco.mjtCatBit.mjCAT_ALL), scn,
    )
    left, right = scn.camera[0], scn.camera[1]
    pos = 0.5 * (np.asarray(left.pos, np.float64) + np.asarray(right.pos, np.float64))
    fwd = _unit(np.asarray(left.forward, np.float64)
                + np.asarray(right.forward, np.float64))
    up = _unit(np.asarray(left.up, np.float64) + np.asarray(right.up, np.float64))
    return pos, fwd, up


def pose_from_passive(handle, scene=None) -> Pose:
    """``(pos, fwd, up)`` for a ``mujoco.viewer.launch_passive`` handle's camera.

    Reads ``handle.cam`` under the handle's lock (the render thread mutates it as
    the user flies), then resolves it via :func:`free_camera_pose`. Convenience
    only -- studio / web viewers hand you an ``MjvCamera`` directly, so call
    :func:`free_camera_pose` for those.
    """
    with handle.lock():
        model, data, cam = handle.m, handle.d, handle.cam
        return free_camera_pose(model, data, cam, scene=scene, opt=handle.opt)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
