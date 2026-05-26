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

"""`client.outliner.*` — world-state introspection + quick-convert edits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .base import _RpcNamespace
from .._op_helpers import target_payload
from ..results import (
    ActorBounds,
    ActorInfo,
    BlueprintInfo,
    _actor_bounds_from_wire,
    _actor_info_from_wire,
    _blueprint_info_from_wire,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient


class _OutlinerNamespace(_RpcNamespace):
    """`client.outliner.*` — world-state introspection + quick-convert edits.

    Result types: :class:`ActorInfo`, :class:`BlueprintInfo` live in
    ``urlab_client.results``.
    """

    def __init__(self, client: "URLabClient"):
        super().__init__(client, "outliner")

    def list_actors(self) -> List[ActorInfo]:
        reply = self._client._rpc("list_actors", {}, expected_op="list_actors_ok")
        return [_actor_info_from_wire(a) for a in (reply.get("actors") or [])]

    def find_actors(
        self,
        *,
        class_filter: Optional[str] = None,
        tag: Optional[str] = None,
        name_prefix: Optional[str] = None,
        in_pie: bool = False,
    ) -> List[ActorInfo]:
        """Filtered actor enumeration. All filters optional; AND across
        set filters. By default searches the editor world; pass
        ``in_pie=True`` to search the active PIE world (falls back to
        the editor world if PIE isn't running)."""
        payload: Dict[str, Any] = {"in_pie": bool(in_pie)}
        if class_filter is not None:
            payload["class_filter"] = str(class_filter)
        if tag is not None:
            payload["tag"] = str(tag)
        if name_prefix is not None:
            payload["name_prefix"] = str(name_prefix)
        reply = self._client._rpc("find_actors", payload, expected_op="find_actors_ok")
        return [_actor_info_from_wire(a) for a in (reply.get("actors") or [])]

    def get_actor_bounds(
        self,
        target: str,
        *,
        by_name: bool = False,
        components_only: bool = False,
    ) -> ActorBounds:
        """AABB of an actor's components in MJ metres."""
        payload: Dict[str, Any] = {
            **target_payload(target, by_name=by_name),
            "components_only": bool(components_only),
        }
        reply = self._client._rpc(
            "get_actor_bounds", payload, expected_op="get_actor_bounds_ok",
        )
        return _actor_bounds_from_wire(reply)

    def list_blueprints(self) -> List[BlueprintInfo]:
        reply = self._client._rpc(
            "list_blueprints", {}, expected_op="list_blueprints_ok",
        )
        return [_blueprint_info_from_wire(b) for b in (reply.get("blueprints") or [])]

    def select_actor(self, target: str, *, by_name: bool = False) -> None:
        self._client._rpc(
            "select_actor", target_payload(target, by_name=by_name),
            expected_op="select_actor_ok",
        )

    def add_quick_convert(
        self,
        target: str,
        *,
        by_name: bool = False,
        static: bool = False,
        complex_mesh: bool = False,
        coacd_threshold: float = 0.05,
        driven_by_unreal: bool = False,
        friction: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        payload: Dict[str, Any] = {
            **target_payload(target, by_name=by_name),
            "static": bool(static),
            "complex_mesh": bool(complex_mesh),
            "coacd_threshold": float(coacd_threshold),
            "driven_by_unreal": bool(driven_by_unreal),
            "friction": [float(x) for x in friction],
        }
        self._client._rpc(
            "add_quick_convert", payload,
            expected_op="add_quick_convert_ok",
        )

    def remove_quick_convert(self, target: str, *, by_name: bool = False) -> None:
        self._client._rpc(
            "remove_quick_convert", target_payload(target, by_name=by_name),
            expected_op="remove_quick_convert_ok",
        )
