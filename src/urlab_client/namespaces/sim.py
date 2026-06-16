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

"""`client.sim.*` — PIE lifecycle: start / stop / status."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import _RpcNamespace
from ..errors import URLabPIEError
from ..results import PIEStartResult, PIEState, PIEStatus

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient

logger = logging.getLogger(__name__)


class _SimNamespace(_RpcNamespace):
    """`client.sim.*` — exposes the PIE trio (begin_pie / stop_pie /
    pie_status) as start / stop / status. Wire ops keep their original
    names; only the Python surface is renamed.

    Result types: :class:`PIEState`, :class:`PIEStartResult`,
    :class:`PIEStatus` live in ``urlab_client.results``.
    """

    def __init__(self, client: "URLabClient"):
        super().__init__(client, "sim")

    def start(
        self,
        level_path: Optional[str] = None,
        *,
        timeout_s: float = 30.0,
        raise_on_failure: bool = True,
    ) -> PIEStartResult:
        """Start PIE; auto-absorb the embedded handshake on success.

        Returns a :class:`PIEStartResult`. With ``raise_on_failure=True``
        (default) anything other than ``READY`` raises :class:`URLabPIEError`
        — typical app code can ignore the return value and trust the
        client's state. Pass ``raise_on_failure=False`` to inspect
        ``compile_error`` / ``state`` and decide what to do.

        Fast path: if ``level_path`` is omitted and PIE is already
        running with a ready manager, the server short-circuits to
        ``state=READY`` without issuing a ``RequestPlaySession`` round
        trip (which would briefly tear down the PIE world and risk
        tripping the UE blueprint-recompile wedge). Pass an explicit
        ``level_path`` to force a level switch + restart.
        """
        payload: Dict[str, Any] = {"timeout_s": float(timeout_s)}
        if level_path is not None:
            payload["level_path"] = str(level_path)
        reply = self._client._rpc(
            "begin_pie", payload,
            expected_op="begin_pie_ok",
            recv_timeout_ms=int((timeout_s + 5.0) * 1000),
        )
        # Coerce wire string to enum; tolerate unknowns (server may
        # introduce new lifecycle states ahead of the bridge).
        state_str = str(reply.get("state", "") or "")
        try:
            state = PIEState(state_str)
        except ValueError:
            state = PIEState.OFF  # safe fallback; surfaces as not-ready
        hs = reply.get("handshake_payload")
        if isinstance(hs, dict) and state == PIEState.READY:
            try:
                self._client._apply_handshake(hs)
                # PIE start is the moment the scene's cameras come into being.
                # connect() starts the streaming subs for whatever cameras exist
                # at handshake time; sim.start absorbs a FRESH handshake (the
                # PIE cameras) so it must (re)start the subs too, or the cameras
                # are discovered but never stream. Idempotent.
                self._client._start_streaming_subs()
            except Exception as exc:
                logger.warning(
                    "sim.start: handshake absorption failed: %s -- "
                    "client.model may be stale; call client.refresh() to refresh",
                    exc,
                )
        result = PIEStartResult(
            state=state,
            compile_error=str(reply.get("compile_error", "") or ""),
            handshake_payload=hs if isinstance(hs, dict) else None,
        )
        if raise_on_failure and state != PIEState.READY:
            raise URLabPIEError(
                code=f"pie_{state.value}",
                message=(
                    result.compile_error
                    or f"PIE start ended in state {state.value!r}"
                ),
                state=state,
            )
        return result

    def stop(self) -> None:
        """End the editor's current PIE session."""
        self._client._rpc("stop_pie", {}, expected_op="stop_pie_ok")
        # PIE end un-registers the manager server-side. Drop the cached
        # flag so subsequent client.close() / namespace calls don't try
        # manager-required ops against a stale True.
        self._client.manager_present = False

    def status(self) -> PIEStatus:
        """Cheap query: PIE state, compile error, latest sim_time."""
        reply = self._client._rpc(
            "pie_status", {}, expected_op="pie_status_ok",
        )
        state_str = str(reply.get("state", "") or "")
        try:
            state = PIEState(state_str)
        except ValueError:
            state = PIEState.OFF
        sim_time = reply.get("sim_time")
        return PIEStatus(
            state=state,
            compile_error=str(reply.get("compile_error", "") or ""),
            sim_time=float(sim_time) if isinstance(sim_time, (int, float)) else None,
        )
