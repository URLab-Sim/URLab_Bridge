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


class URLabVersionMismatch(RuntimeError):
    pass
