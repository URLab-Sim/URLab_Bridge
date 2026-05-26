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

"""`client.viewport.*` — perspective viewport camera control."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

from .base import _RpcNamespace
from .._op_helpers import target_payload
from ..results import CameraPose, _camera_pose_from_wire

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient


class _ViewportNamespace(_RpcNamespace):
    """`client.viewport.*` — perspective viewport camera control.

    All ops operate on the editor's most-recently-focused perspective
    viewport. Editor-only (no live PIE viewport binding — use UE's
    play-window camera spectator instead while PIE is running).

    Positions on the wire are in MJ metres. ``rotation_quat`` is xyzw
    (UE FQuat order). ``screenshot`` is not implemented in v1; see
    ``docs/plan_followups.md``.
    """

    def __init__(self, client: "URLabClient"):
        super().__init__(client, "viewport")

    def set_camera(
        self,
        location: Sequence[float],
        *,
        rotation_quat: Optional[Sequence[float]] = None,
        rotation_euler: Optional[Sequence[float]] = None,
        fov: Optional[float] = None,
    ) -> CameraPose:
        """Move the perspective viewport camera. ``location`` in MJ
        metres. Pass at most one of ``rotation_quat`` (xyzw) or
        ``rotation_euler`` (roll, pitch, yaw degrees). ``fov`` is
        horizontal degrees. Returns the resolved camera pose."""
        if rotation_quat is not None and rotation_euler is not None:
            raise ValueError("pass at most one of rotation_quat / rotation_euler")
        payload: Dict[str, Any] = {"location": [float(x) for x in location]}
        if rotation_quat is not None:
            payload["rotation_quat"] = [float(x) for x in rotation_quat]
        elif rotation_euler is not None:
            payload["rotation_euler"] = [float(x) for x in rotation_euler]
        if fov is not None:
            payload["fov"] = float(fov)
        reply = self._client._rpc(
            "set_camera", payload, expected_op="set_camera_ok",
        )
        return _camera_pose_from_wire(reply)

    def get_camera(self) -> CameraPose:
        reply = self._client._rpc(
            "get_camera", {}, expected_op="get_camera_ok",
        )
        return _camera_pose_from_wire(reply)

    def frame_actor(self, target: str, *, by_name: bool = False) -> CameraPose:
        """Reframe the viewport on an actor's bounds. Returns the new
        camera pose chosen by the editor."""
        reply = self._client._rpc(
            "frame_actor", target_payload(target, by_name=by_name),
            expected_op="frame_actor_ok",
        )
        return _camera_pose_from_wire(reply)

    def set_mode(self, mode: str) -> str:
        """One of ``"lit"``, ``"unlit"``, ``"wireframe"``, ``"collision"``,
        ``"reflections"``. Returns the resolved mode."""
        valid = ("lit", "unlit", "wireframe", "collision", "reflections")
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got {mode!r}")
        reply = self._client._rpc(
            "set_viewport_mode", {"mode": str(mode)},
            expected_op="set_mode_ok",
        )
        return str(reply.get("mode", mode))

    def track_actor(
        self,
        target: str,
        *,
        by_name: bool = False,
        offset: Optional[Sequence[float]] = None,
        smoothing: float = 0.0,
    ) -> str:
        """Install a per-tick callback that lerps the viewport camera
        toward ``actor.location + actor.rotation * offset``. Returns
        the tracked actor's UE path. ``offset`` is in MJ metres
        (default ``(0, -2, 1)`` — 2m behind, 1m above). ``smoothing``
        in ``[0, 1)`` — 0 snaps, 0.95 barely moves. Last writer wins;
        call :meth:`untrack` to clear.
        """
        payload: Dict[str, Any] = {
            **target_payload(target, by_name=by_name),
            "smoothing": float(smoothing),
        }
        if offset is not None:
            payload["offset"] = [float(x) for x in offset]
        reply = self._client._rpc(
            "track_actor", payload, expected_op="track_actor_ok",
        )
        return str(reply.get("tracked_actor_path", ""))

    def untrack(self) -> bool:
        """Stop any active :meth:`track_actor` callback. Idempotent;
        returns ``True`` if tracking was active, ``False`` otherwise."""
        reply = self._client._rpc(
            "untrack", {}, expected_op="untrack_ok",
        )
        return bool(reply.get("was_tracking", False))
