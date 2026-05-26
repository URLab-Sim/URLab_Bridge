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

"""`client.debug.*` — UE DrawDebug* primitives (editor + PIE)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

from .base import _RpcNamespace

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient


class _DebugNamespace(_RpcNamespace):
    """`client.debug.*` — UE DrawDebug* primitives.

    Wire convention: positions in MJ metres, colors ``[r, g, b]`` in
    ``[0, 1]``, ``ttl`` in seconds (``0`` = single frame, ``-1`` =
    persistent until :meth:`clear_markers`).

    Works in editor or PIE; the plugin picks the PIE world if running,
    else the editor world. All methods return ``None`` (fire-and-forget
    acks).
    """

    def __init__(self, client: "URLabClient"):
        super().__init__(client, "debug")

    def draw_marker(
        self,
        location: Sequence[float],
        color: Sequence[float],
        *,
        ttl: float = 0.0,
        label: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "location": [float(x) for x in location],
            "color":    [float(x) for x in color],
            "ttl":      float(ttl),
        }
        if label is not None:
            payload["label"] = str(label)
        if tag is not None:
            payload["tag"] = str(tag)
        self._client._rpc(
            "draw_marker", payload, expected_op="draw_marker_ok",
        )

    def draw_line(
        self,
        from_: Sequence[float],
        to: Sequence[float],
        color: Sequence[float],
        *,
        ttl: float = 0.0,
        thickness: float = 1.0,
        tag: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "from":      [float(x) for x in from_],
            "to":        [float(x) for x in to],
            "color":     [float(x) for x in color],
            "ttl":       float(ttl),
            "thickness": float(thickness),
        }
        if tag is not None:
            payload["tag"] = str(tag)
        self._client._rpc(
            "draw_line", payload, expected_op="draw_line_ok",
        )

    def draw_box(
        self,
        center: Sequence[float],
        half_extents: Sequence[float],
        color: Sequence[float],
        *,
        rotation_quat: Optional[Sequence[float]] = None,
        ttl: float = 0.0,
        tag: Optional[str] = None,
    ) -> None:
        """``rotation_quat`` is xyzw (UE FQuat convention)."""
        payload: Dict[str, Any] = {
            "center":       [float(x) for x in center],
            "half_extents": [float(x) for x in half_extents],
            "color":        [float(x) for x in color],
            "ttl":          float(ttl),
        }
        if rotation_quat is not None:
            payload["rotation_quat"] = [float(x) for x in rotation_quat]
        if tag is not None:
            payload["tag"] = str(tag)
        self._client._rpc(
            "draw_box", payload, expected_op="draw_box_ok",
        )

    def draw_arrow(
        self,
        from_: Sequence[float],
        to: Sequence[float],
        color: Sequence[float],
        *,
        ttl: float = 0.0,
        thickness: float = 1.0,
        arrow_size: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Draw a directional arrow from ``from_`` to ``to`` (MJ metres).
        ``arrow_size`` is the head length in MJ metres; default is 20%
        of the shaft length."""
        payload: Dict[str, Any] = {
            "from":      [float(x) for x in from_],
            "to":        [float(x) for x in to],
            "color":     [float(x) for x in color],
            "ttl":       float(ttl),
            "thickness": float(thickness),
        }
        if arrow_size is not None:
            payload["arrow_size"] = float(arrow_size)
        if tag is not None:
            payload["tag"] = str(tag)
        self._client._rpc(
            "draw_arrow", payload, expected_op="draw_arrow_ok",
        )

    def draw_axes(
        self,
        location: Sequence[float],
        *,
        rotation_quat: Optional[Sequence[float]] = None,
        rotation_euler: Optional[Sequence[float]] = None,
        scale: float = 0.2,
        ttl: float = 0.0,
        tag: Optional[str] = None,
    ) -> None:
        """Draw an RGB coordinate triad (X=red, Y=green, Z=blue) at
        ``location`` (MJ metres). ``scale`` is per-axis arrow length in
        MJ metres. Pass at most one of ``rotation_quat`` (xyzw) or
        ``rotation_euler`` (roll, pitch, yaw degrees) to orient the
        frame."""
        if rotation_quat is not None and rotation_euler is not None:
            raise ValueError("pass at most one of rotation_quat / rotation_euler")
        payload: Dict[str, Any] = {
            "location": [float(x) for x in location],
            "scale":    float(scale),
            "ttl":      float(ttl),
        }
        if rotation_quat is not None:
            payload["rotation_quat"] = [float(x) for x in rotation_quat]
        elif rotation_euler is not None:
            payload["rotation_euler"] = [float(x) for x in rotation_euler]
        if tag is not None:
            payload["tag"] = str(tag)
        self._client._rpc(
            "draw_axes", payload, expected_op="draw_axes_ok",
        )

    def clear_markers(self, *, tag: Optional[str] = None) -> None:
        """Clear all bridge-drawn debug primitives in the world. ``tag``
        is accepted for forward compatibility but currently ignored —
        UE's debug-drawing system has no per-tag removal, so v1 always
        performs a full flush."""
        payload: Dict[str, Any] = {}
        if tag is not None:
            payload["tag"] = str(tag)
        self._client._rpc(
            "clear_markers", payload, expected_op="clear_markers_ok",
        )

    def set_overlay_text(self, text: str, *, anchor: str = "top_left") -> None:
        """Set the in-viewport on-screen debug text. Empty string clears
        the message. ``anchor`` is accepted but currently ignored — v1
        uses UE's fixed top-of-viewport position."""
        self._client._rpc(
            "set_overlay_text",
            {"text": str(text), "anchor": str(anchor)},
            expected_op="set_overlay_text_ok",
        )
