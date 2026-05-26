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

"""Pure-dict helpers for RPC payload construction. No transport, no UE coupling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def pose_payload(
    *,
    location: Optional[Sequence[float]] = None,
    rotation_quat: Optional[Sequence[float]] = None,
    rotation_euler: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """`location` (3, MJ metres), `rotation_quat` (xyzw) or
    `rotation_euler` (degrees), `scale` (3). Server picks quat when both
    rotations are present. Each field included only when non-None."""
    out: Dict[str, Any] = {}
    if location is not None:
        out["location"] = [float(x) for x in location]
    if rotation_quat is not None:
        out["rotation_quat"] = [float(x) for x in rotation_quat]
    if rotation_euler is not None:
        out["rotation_euler"] = [float(x) for x in rotation_euler]
    if scale is not None:
        out["scale"] = [float(x) for x in scale]
    return out


def target_payload(target: str, *, by_name: bool = False) -> Dict[str, Any]:
    """`target` + optional `target_by` discriminator (default actor_id)."""
    out: Dict[str, Any] = {"target": str(target)}
    if by_name:
        out["target_by"] = "actor_name"
    return out
