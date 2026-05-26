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

"""Recording + replay lifecycle against a real UE editor.

Covers the typed returns: RecordingHandle, RecordingSummary, Path,
ReplaySession, ReplayStatus."""

from __future__ import annotations

from pathlib import Path

import pytest

from urlab_client.results import (
    RecordingHandle,
    RecordingSummary,
    ReplaySession,
    ReplayStatus,
)


def test_recording_lifecycle(pie_client, tmp_path):
    """start → step a few frames → stop → save returns a Path.

    The server uses a single rolling "live" recording buffer regardless
    of the ``name=`` parameter (the wire reply's ``name`` field carries
    the server-side label, not the requested one), so we just shape-
    check the handle rather than asserting equality."""
    handle = pie_client.recording.start(name="urlab_test_rec")
    assert isinstance(handle, RecordingHandle)
    assert handle.name, "recording_start_ok must echo a session name"

    for _ in range(5):
        pie_client.step(n_steps=1)

    summary = pie_client.recording.stop()
    assert isinstance(summary, RecordingSummary)
    assert summary.frame_count >= 1
    assert summary.sim_duration_s >= 0.0

    saved = pie_client.recording.save()
    assert isinstance(saved, Path)
    assert saved.exists()

    pie_client.recording.clear()


def test_replay_lifecycle(pie_client, tmp_path):
    """Record briefly, save, load by path, start playback."""
    pie_client.recording.start(name="urlab_test_replay")
    for _ in range(5):
        pie_client.step(n_steps=1)
    pie_client.recording.stop()
    saved = pie_client.recording.save()
    pie_client.recording.clear()

    session = pie_client.replay.load(str(saved))
    assert isinstance(session, ReplaySession)
    assert session.name

    pie_client.replay.set_active(session.name)
    status = pie_client.replay.start()
    assert isinstance(status, ReplayStatus)
    assert status.active_session == session.name

    pie_client.replay.stop()
