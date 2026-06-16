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

"""`client.recording` — :class:`URLabRecordingAPI` instance assigned by
the client. Result dataclasses live in :mod:`urlab_client.results`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ..results import RecordingHandle, RecordingSummary

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient


class URLabRecordingAPI:
    """`client.recording.*` namespace. Delegates to step-server RPCs.

    Result dataclasses (:class:`RecordingHandle`, :class:`RecordingSummary`)
    live in :mod:`urlab_client.results`.
    """

    def __init__(self, client: "URLabClient"):
        self._client = client
        self.is_active: bool = False
        self.frame_count: int = 0
        self.sim_duration: float = 0.0
        self.last_saved_path: Optional[Path] = None

    def start(
        self,
        name: Optional[str] = None,
        max_duration_s: Optional[float] = None,
    ) -> RecordingHandle:
        reply = self._client._rpc(
            "recording_start",
            {
                "name": name,
                "max_duration_s": max_duration_s,
            },
            expected_op="recording_start_ok",
        )
        self.is_active = True
        max_d = reply.get("max_duration_s")
        return RecordingHandle(
            name=str(reply.get("name", "") or ""),
            max_duration_s=float(max_d) if isinstance(max_d, (int, float)) else None,
        )

    def stop(self) -> RecordingSummary:
        reply = self._client._rpc(
            "recording_stop",
            {},
            expected_op="recording_stop_ok",
        )
        self.is_active = False
        self.frame_count = int(reply.get("frame_count", 0) or 0)
        self.sim_duration = float(reply.get("sim_duration_s", 0.0) or 0.0)
        return RecordingSummary(
            frame_count=self.frame_count,
            sim_duration_s=self.sim_duration,
        )

    def save(self, path: Optional[str] = None) -> Path:
        reply = self._client._rpc(
            "recording_save",
            {"path": path},
            expected_op="recording_save_ok",
        )
        abs_path = Path(str(reply.get("absolute_path", "") or ""))
        self.last_saved_path = abs_path
        return abs_path

    def clear_buffer(self) -> None:
        self._client._rpc(
            "recording_clear",
            {},
            expected_op="recording_clear_ok",
        )
        self.frame_count = 0
        self.sim_duration = 0.0
