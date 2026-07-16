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

"""Exception types raised by the URLab remote-stepping client.

`URLabRPCError`        -- generic RPC failure (server returned ``op=error``).
`URLabPIEError`        -- PIE start ended in a non-ready state.
`URLabVersionMismatch` -- MuJoCo version mismatch between client and server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .results import PIEState


class URLabRPCError(RuntimeError):
    def __init__(self, code: str, message: str, *, op: Optional[str] = None):
        super().__init__(f"[{code}] {message} (op={op!r})")
        self.code = code
        self.message = message
        self.op = op


class URLabPIEError(URLabRPCError):
    """Raised by :meth:`URLabClient.sim.start` when PIE start ends in a
    non-ready state (compile_failed / timeout / off / etc.) and the
    caller didn't pass ``raise_on_failure=False``.

    Carries the typed :class:`PIEState` alongside the usual error code
    + message so callers can branch without string-matching.
    """

    def __init__(self, *, code: str, message: str, state: "PIEState"):
        super().__init__(code, message, op="begin_pie")
        self.state = state


class URLabTimeoutError(URLabRPCError, TimeoutError):
    """Raised by the client-side await/readiness layer when a wait exceeds its
    deadline. Also a ``TimeoutError`` so ``except TimeoutError`` works.

    ``server_alive`` reflects whether the state stream looked fresh at timeout:
    ``True`` -> server alive but the op is slow; ``False`` -> server appears
    silent/hung; ``None`` -> no liveness signal available (e.g. not in PIE).
    """

    def __init__(
        self,
        description: str,
        *,
        waited_s: float,
        server_alive: Optional[bool] = None,
        op: Optional[str] = None,
    ):
        if server_alive is True:
            tail = " (server alive but slow)"
        elif server_alive is False:
            tail = " (server appears silent/hung)"
        else:
            tail = ""
        super().__init__(
            "timeout", f"{description}: not ready after {waited_s:.1f}s{tail}", op=op
        )
        self.description = description
        self.waited_s = waited_s
        self.server_alive = server_alive


class URLabVersionMismatch(RuntimeError):
    """No longer raised: a MuJoCo version skew logs a warning and the
    client builds its local model from the compiled XML instead of the
    version-locked MJB. Kept exported so existing `except` clauses don't
    break."""
