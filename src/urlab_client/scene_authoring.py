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

"""Spawn/light handles + reply decoders + asset spec.

This module has zero `URLabClient` dependency, so it can be imported
from anywhere without the apply_scene-style cycle shim. The
orchestration helper (`apply_scene`) lives in
`urlab_client.namespaces.scene`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .articulation import URLabArticulation
    from .client import URLabClient


@dataclass
class URLabSpawnHandle:
    """Edit-time handle returned by ``client.spawn_actor`` for articulation
    blueprints.

    Holds identity + UE-side info only; physics fields (joints, actuators,
    sensors) aren't known until PIE runs and ``client.connect()``
    materialises a :class:`URLabArticulation` against the same
    ``actor_id``.
    """

    actor_id: str
    actor_name: str
    actor_path: str
    blueprint_class_path: str
    location: tuple = (0.0, 0.0, 0.0)
    rotation_quat: tuple = (0.0, 0.0, 0.0, 1.0)
    requires_pie_restart: bool = False
    was_existing: bool = False

    def runtime(self, client: "URLabClient") -> Optional["URLabArticulation"]:
        """Return the post-PIE :class:`URLabArticulation` for this id, or
        ``None`` if PIE hasn't started yet (or the actor was removed)."""
        return client.articulations_by_id.get(self.actor_id)


@dataclass
class URLabLightHandle:
    """Edit-time handle for a spawned light. No runtime equivalent —
    lights aren't physics, just UE actors."""

    actor_id: str
    actor_name: str
    actor_path: str
    kind: str
    location: tuple = (0.0, 0.0, 0.0)
    rotation_euler: tuple = (0.0, 0.0, 0.0)
    intensity: float = 5000.0
    color: tuple = (1.0, 1.0, 1.0)
    requires_pie_restart: bool = False


@dataclass
class URLabBlueprint:
    """Result of ``client.scene.import_xml``. Carries the UE-side BP
    class path so it can be passed straight to ``spawn_actor``, plus
    metadata about whether the import factory actually ran this call
    (vs. returning an already-imported BP).

    Pass instances directly to ``client.scene.spawn_actor(blueprint=...)``
    to skip the manual ``class_path`` shuffle.
    """

    class_path: str
    short_name: str
    imported_now: bool


def _editor_from_spawn_reply(reply: dict) -> URLabSpawnHandle:
    return URLabSpawnHandle(
        actor_id=str(reply.get("actor_id", "") or ""),
        actor_name=str(reply.get("actor_name", "") or ""),
        actor_path=str(reply.get("actor_path", "") or ""),
        blueprint_class_path=str(reply.get("blueprint_class_path", "") or ""),
        location=tuple(reply.get("location", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)),
        rotation_quat=tuple(
            reply.get("rotation_quat", (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0)
        ),
        requires_pie_restart=bool(reply.get("requires_pie_restart", False)),
        was_existing=bool(reply.get("was_existing", False)),
    )


def _light_from_spawn_reply(reply: dict) -> URLabLightHandle:
    return URLabLightHandle(
        actor_id=str(reply.get("actor_id", "") or ""),
        actor_name=str(reply.get("actor_name", "") or ""),
        actor_path=str(reply.get("actor_path", "") or ""),
        kind=str(reply.get("kind", "directional") or "directional"),
        location=tuple(reply.get("location", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)),
        rotation_euler=tuple(
            reply.get("rotation_euler", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
        ),
        intensity=float(reply.get("intensity", 5000.0) or 5000.0),
        color=tuple(reply.get("color", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)),
        requires_pie_restart=bool(reply.get("requires_pie_restart", False)),
    )


def _blueprint_from_import_reply(reply: dict) -> URLabBlueprint:
    return URLabBlueprint(
        class_path=str(reply.get("blueprint_class_path", "") or ""),
        short_name=str(reply.get("blueprint_short_name", "") or ""),
        imported_now=bool(reply.get("imported_now", False)),
    )


@dataclass
class URLabAsset:
    """One asset to materialise via :meth:`client.scene.apply_scene`.

    ``actor_id`` is the bridge-owned handle; ``xml`` is the absolute path
    to the MJCF file. Pose is optional and defaults to origin / identity.
    """

    actor_id: str
    xml: str
    location: Sequence[float] = (0.0, 0.0, 0.0)
    rotation_quat: Optional[Sequence[float]] = None
    rotation_euler: Optional[Sequence[float]] = None
    scale: Sequence[float] = (1.0, 1.0, 1.0)
