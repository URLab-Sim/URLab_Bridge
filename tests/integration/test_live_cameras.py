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

The golden scene mounts a ``<camera name="head">`` on link1, so
``pie_client`` is enough.

A frame reaches ``latest_frame`` two ways, tested separately:
``camera_query="sync"`` renders inside the step, and the async streams
tick only after ``warmup_cameras`` enables broadcast.

The policy is ``camera_query``. ``include_cameras`` only names cameras;
its mapping values are discarded, so ``{cam: "sync"}`` reads the
streams instead.
"""

from __future__ import annotations

import numpy as np
import pytest


def _articulation_with_camera(client):
    for art in client.articulations.values():
        if art.cameras:
            return art
    pytest.fail("golden scene should expose at least one camera-bearing articulation")


def _sole_camera(client):
    art = _articulation_with_camera(client)
    return art, next(iter(art.cameras))


def test_sync_capture_populates_latest_frame(pie_client):
    art, cam_name = _sole_camera(pie_client)
    pie_client.step(n_steps=1, include_cameras=[cam_name], camera_query="sync")
    cam = art.cameras[cam_name]
    assert cam.latest_frame is not None
    assert isinstance(cam.latest_frame, np.ndarray)
    assert cam.frame_count >= 1


def test_streamed_capture_populates_latest_frame(pie_client):
    art, cam_name = _sole_camera(pie_client)
    assert pie_client.warmup_cameras([cam_name], timeout_s=15.0) == [cam_name]
    cam = art.cameras[cam_name]
    assert cam.latest_frame is not None
    delivered = cam.frame_count
    # The stream keeps feeding the same view, so a "latest" step reads a
    # cached frame rather than going back to the server for one.
    pie_client.step(n_steps=1, include_cameras=[cam_name], camera_query="latest")
    assert cam.latest_frame is not None
    assert cam.frame_count >= delivered


def test_camera_dtype_matches_mode(pie_client):
    art, cam_name = _sole_camera(pie_client)
    pie_client.step(n_steps=1, include_cameras=[cam_name], camera_query="sync")
    cam = art.cameras[cam_name]
    if cam.mode.value == "depth":
        assert cam.latest_frame.dtype == np.float32
        assert cam.latest_frame.ndim == 2
    else:
        # real / semantic / instance — uint8 with 4 channels.
        assert cam.latest_frame.dtype == np.uint8
        assert cam.latest_frame.shape[-1] == 4


def test_frame_matches_declared_resolution(pie_client):
    """A frame that decodes at the wrong size is dropped silently by
    ``_decode_camera_frame``, so assert the shape the view advertises."""
    art, cam_name = _sole_camera(pie_client)
    pie_client.step(n_steps=1, include_cameras=[cam_name], camera_query="sync")
    cam = art.cameras[cam_name]
    width, height = cam.resolution
    assert cam.latest_frame.shape[:2] == (height, width)
