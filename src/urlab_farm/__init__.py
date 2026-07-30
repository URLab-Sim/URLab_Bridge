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

"""URLab render-farm launcher.

Spawn and manage a pool of Unreal editor render instances. CLI: ``urlab-farm``
(runnable as ``python -m urlab_farm``) with ``up`` / ``down`` / ``ps``
subcommands. Client-side pool discovery + leasing lives in
:class:`urlab_client.URLabPool`.
"""

from __future__ import annotations

from .launcher import (
    LaunchedInstance,
    down,
    find_editor,
    find_project,
    isolate_project,
    kill_pid,
    ps,
    up,
)

__all__ = [
    "LaunchedInstance",
    "down",
    "find_editor",
    "find_project",
    "isolate_project",
    "kill_pid",
    "ps",
    "up",
]
