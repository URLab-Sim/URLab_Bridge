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

"""`client.replay` — :class:`URLabReplayAPI` instance assigned by the
client. Result dataclasses live in :mod:`urlab_client.results`."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from ..results import ReplaySession, ReplayStatus

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient


class URLabReplayAPI:
    """`client.replay.*` namespace. Delegates to step-server RPCs.

    Result dataclasses (:class:`ReplaySession`, :class:`ReplayStatus`)
    live in :mod:`urlab_client.results`.
    """

    def __init__(self, client: "URLabClient"):
        self._client = client
        self._loaded_sessions: List[str] = []
        self.active_session: Optional[str] = None

    def load(self, path: str) -> ReplaySession:
        reply = self._client._rpc(
            "replay_load",
            {"path": path},
            expected_op="replay_load_ok",
        )
        name = str(reply.get("name", "") or "")
        if name and name not in self._loaded_sessions:
            self._loaded_sessions.append(name)
        total = reply.get("total_frames")
        return ReplaySession(
            name=name,
            total_frames=int(total) if isinstance(total, (int, float)) else 0,
            source_path=Path(str(path)) if path else None,
        )

    def list_sessions(self) -> List[str]:
        reply = self._client._rpc(
            "replay_list_sessions",
            {},
            expected_op="replay_list_sessions_ok",
        )
        sessions = [str(s) for s in reply.get("sessions", [])]
        self._loaded_sessions = sessions
        return list(sessions)

    def set_active(self, name: str) -> None:
        self._client._rpc(
            "replay_set_active",
            {"name": name},
            expected_op="replay_set_active_ok",
        )
        self.active_session = name

    def start(self) -> ReplayStatus:
        reply = self._client._rpc(
            "replay_start",
            {},
            expected_op="replay_start_ok",
        )
        active = str(reply.get("active_session", "") or "")
        if active:
            self.active_session = active
        total = reply.get("total_frames")
        return ReplayStatus(
            active_session=active,
            total_frames=int(total) if isinstance(total, (int, float)) else 0,
        )

    def stop(self) -> None:
        self._client._rpc(
            "replay_stop",
            {},
            expected_op="replay_stop_ok",
        )

    def play(self, path_or_name: str, *, loop: bool = False) -> ReplaySession:
        """Convenience: resolve against loaded sessions, else filesystem,
        then set_active + start. Mirrors the plan's three-call sequence."""
        if path_or_name in self._loaded_sessions:
            session = ReplaySession(name=path_or_name)
        else:
            session = self.load(path_or_name)
        self.set_active(session.name)
        if loop:
            # `loop` is metadata for the UE side when it lands; ship it as a
            # hint on replay_start so future UE code can honour it. Current
            # UE REP handler ignores unknown keys.
            self._client._rpc(
                "replay_start",
                {"loop": True},
                expected_op="replay_start_ok",
            )
        else:
            self.start()
        return session
