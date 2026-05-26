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

"""Camera capture against a live UE editor.

Covers: include_cameras=True returns frames, frame dtype matches
camera mode, art.cameras[name].latest_frame is populated.

The session-bootstrapped golden scene has a ``<camera name="head">``
mounted on link1, so ``pie_client`` is enough. Tests use the per-
camera ``"sync"`` flag so they don't race the render thread on the
first step.
"""

from __future__ import annotations

import numpy as np
import pytest


def _articulation_with_camera(client):
    for art in client.articulations.values():
        if art.cameras:
            return art
    pytest.fail("golden scene should expose at least one camera-bearing articulation")


def _capture_one_frame(client, cam_name: str) -> None:
    """Take a few warmup steps so UE has time to render at least one
    camera frame, then a sync step so the reply waits for a fresh
    capture instead of returning the latest-cached (which is None on
    the first step)."""
    for _ in range(3):
        client.step(n_steps=1, include_cameras={cam_name: "latest"})
    client.step(n_steps=1, include_cameras={cam_name: "sync"})


def test_include_cameras_populates_latest_frame(pie_client):
    art = _articulation_with_camera(pie_client)
    cam_name = next(iter(art.cameras))
    _capture_one_frame(pie_client, cam_name)
    cam = art.cameras[cam_name]
    assert cam.latest_frame is not None
    assert isinstance(cam.latest_frame, np.ndarray)
    assert cam.frame_count >= 1


def test_camera_dtype_matches_mode(pie_client):
    art = _articulation_with_camera(pie_client)
    cam_name = next(iter(art.cameras))
    _capture_one_frame(pie_client, cam_name)
    cam = art.cameras[cam_name]
    if cam.mode.value == "depth":
        assert cam.latest_frame.dtype == np.float32
        assert cam.latest_frame.ndim == 2
    else:
        # real / semantic / instance — uint8 with 4 channels.
        assert cam.latest_frame.dtype == np.uint8
        assert cam.latest_frame.shape[-1] == 4
