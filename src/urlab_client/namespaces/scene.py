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

"""`client.scene.*` — every editor-only op that adds, removes, or edits
world-state assets."""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, TYPE_CHECKING, Optional, Union

from .base import _RpcNamespace
from ..errors import URLabRPCError
from .._op_helpers import pose_payload, target_payload
from ..results import (
    ActorHierarchyNode,
    SceneSnapshot,
    _hierarchy_from_wire,
    _scene_snapshot_from_wire,
)
from ..scene_authoring import (
    URLabAsset,
    URLabBlueprint,
    URLabLightHandle,
    URLabSpawnHandle,
    _blueprint_from_import_reply,
    _editor_from_spawn_reply,
    _light_from_spawn_reply,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient

logger = logging.getLogger(__name__)


class _SceneNamespace(_RpcNamespace):
    """`client.scene.*` — every editor-only op that adds, removes, or
    edits world-state assets. Methods carry custom marshalling that the
    generic RPC synthesizer can't (pose payload encoding, dataclass
    unwrapping, target/target_by helpers).

    Result types: :class:`URLabBlueprint`, :class:`URLabSpawnHandle`,
    :class:`URLabLightHandle`, :class:`URLabAsset` live in
    ``urlab_client.scene_authoring``.
    """

    def __init__(self, client: "URLabClient"):
        super().__init__(client, "scene")

    def create_level(self, name: str, *, force_overwrite: bool = False) -> None:
        """Create an empty level at ``/Game/Levels/<name>``. Editor-only.

        ``force_overwrite=True`` deletes any existing asset at the target
        path first (switches the editor off it if currently loaded).
        Default ``False`` keeps the safe behavior — fails on collision.

        The new level's ``WorldSettings.DefaultGameMode`` is overridden
        to :class:`AGameModeBase` so subsequent ``sim.start`` doesn't
        transitively load the project-default game mode (and any broken
        blueprint references it pulls in -- e.g. Marketplace mannequin
        AnimBPs that fail to compile and pop a modal during PIE start).
        """
        self._client._run_editor_job(
            "create_level",
            {"name": str(name), "force_overwrite": bool(force_overwrite)},
            expected_op="create_level_ok",
        )

    def current_level(self) -> str:
        """Return the package path of the editor's currently-loaded
        level (e.g. ``/Game/Levels/MyLevel``). Editor-only."""
        reply = self._client._run_editor_job(
            "current_level", {}, expected_op="current_level_ok",
        )
        return str(reply.get("level_path", ""))

    def destroy_asset(self, asset_path: str) -> bool:
        """Force-delete a project asset by object path (e.g.
        ``/Game/Levels/Foo.Foo`` or ``/Game/MuJoCoImports/bar.bar``).
        Idempotent — returns ``True`` if the asset existed, ``False`` if
        already absent. Switches the editor off it first if it's a UWorld
        currently loaded. Editor-only.
        """
        reply = self._client._run_editor_job(
            "destroy_asset", {"asset_path": str(asset_path)},
            expected_op="destroy_asset_ok",
        )
        return bool(reply.get("was_found", False))

    def snapshot(self) -> SceneSnapshot:
        """JSON snapshot of every URLab actor in the current level, with
        articulation metadata (joint / actuator / sensor / camera names).
        Heavier than :meth:`URLabClient.outliner.list_actors`.
        Returns from the PIE world if PIE is running, else the editor world.
        """
        reply = self._client._run_editor_job("snapshot", {}, expected_op="snapshot_ok")
        return _scene_snapshot_from_wire(reply)

    def duplicate_actor(
        self,
        target: str,
        new_actor_id: str,
        *,
        by_name: bool = False,
        location: Optional[Sequence[float]] = None,
    ) -> URLabSpawnHandle:
        """Spawn a copy of an existing actor with a fresh ``actor_id``.
        Default placement offsets +1m on X from the source; pass
        ``location`` (MJ metres) to override."""
        payload: Dict[str, Any] = {
            **target_payload(target, by_name=by_name),
            "new_actor_id": str(new_actor_id),
        }
        if location is not None:
            payload["location"] = [float(x) for x in location]
        reply = self._client._run_editor_job(
            "duplicate_actor", payload, expected_op="duplicate_actor_ok"
        )
        return _editor_from_spawn_reply(reply)

    def actor_hierarchy(
        self, target: str, *, by_name: bool = False
    ) -> ActorHierarchyNode:
        """Return the recursive attachment tree rooted at ``target``."""
        payload = target_payload(target, by_name=by_name)
        reply = self._client._run_editor_job(
            "actor_hierarchy", payload, expected_op="actor_hierarchy_ok"
        )
        return _hierarchy_from_wire(reply.get("root") or {})

    def ensure_manager(self) -> bool:
        """Ensure an AAMjManager actor exists in the current level; spawn
        one at origin if not. Required before sim.start in freshly created
        levels (an empty level has no manager, so PIE has no physics
        engine and begin_pie times out). Returns ``True`` if a manager
        was already present, ``False`` if one was spawned. Editor-only.
        """
        reply = self._client._run_editor_job(
            "ensure_manager", {}, expected_op="ensure_manager_ok"
        )
        return bool(reply.get("was_existing", False))

    def load_level(self, name_or_path: str) -> None:
        """Load an existing level into the editor. Editor-only."""
        self._client._run_editor_job(
            "load_level", {"level_path": str(name_or_path)},
            expected_op="load_level_ok",
        )

    def save_level(self) -> None:
        """Save the editor's currently-loaded level. Editor-only."""
        self._client._run_editor_job("save_level", {}, expected_op="save_level_ok")

    def import_xml(
        self, path: str, *, force_reimport: bool = False
    ) -> URLabBlueprint:
        """Drive UE's MJCF import factory programmatically. Editor-only.

        Returns a :class:`URLabBlueprint` you can pass straight into
        :meth:`spawn_actor` (skip the manual class-path shuffle).

        ``force_reimport=True`` destroys any existing BP at the target
        path before re-importing. Without that explicit destroy, the
        factory's ``FKismetEditorUtilities::CreateBlueprint`` hits a
        duplicate-name assertion and crashes the editor. The new BP
        lands at the same class path (the BP name is derived from the
        XML file stem), so old :class:`URLabBlueprint` handles remain
        valid after a re-import.
        """
        # ``path`` is resolved on the SERVER host: the UE import factory opens
        # it from the editor machine's filesystem, so this only works when the
        # bridge and UE share a filesystem (local / same-host). A planned
        # network-model feature will let a remote client ship the MJCF XML plus
        # its VFS asset bytes over the RPC instead of a path, so any render
        # server can be driven remotely; that upload is owned by a separate
        # design pass and is intentionally NOT built here.
        reply = self._client._run_editor_job(
            "import_xml",
            {"path": str(path), "force_reimport": bool(force_reimport)},
            expected_op="import_xml_ok",
        )
        return _blueprint_from_import_reply(reply)

    def spawn_actor(
        self,
        blueprint: Union[URLabBlueprint, str],
        actor_id: str,
        *,
        location: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_quat: Optional[Sequence[float]] = None,
        rotation_euler: Optional[Sequence[float]] = None,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> URLabSpawnHandle:
        """Spawn a Blueprint-driven actor. Returns a URLabSpawnHandle.

        ``blueprint`` accepts a :class:`URLabBlueprint` (e.g. the return
        of :meth:`import_xml`) or a raw class-path string. Idempotent on
        ``actor_id``: a second spawn with the same id updates the
        existing actor in place; the reply's ``was_existing`` flag tells
        you which path ran.
        """
        if isinstance(blueprint, URLabBlueprint):
            bp_str = blueprint.class_path
        else:
            bp_str = str(blueprint)

        payload: Dict[str, Any] = {
            "blueprint": bp_str,
            "actor_id": str(actor_id),
            **pose_payload(
                location=location,
                rotation_quat=rotation_quat,
                rotation_euler=rotation_euler if rotation_quat is None else None,
                scale=scale,
            ),
        }
        reply = self._client._run_editor_job("spawn_actor", payload, expected_op="spawn_actor_ok")
        return _editor_from_spawn_reply(reply)

    def spawn_grid(
        self,
        blueprint: Union[URLabBlueprint, str],
        base_actor_id: str,
        count_x: int,
        count_y: int,
        *,
        spacing: Sequence[float] = (1.0, 1.0, 0.0),
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_quat: Optional[Sequence[float]] = None,
        rotation_euler: Optional[Sequence[float]] = None,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, URLabSpawnHandle]:
        """Spawn an ``count_x * count_y`` grid of duplicates of
        ``blueprint``. Cells are at ``origin + (i*spacing.x, j*spacing.y,
        0)`` in MJ metres; per-cell actor ids are
        ``f"{base_actor_id}_{i}_{j}"``. Server-side capped at 1024 cells;
        chunk larger grids on the client.

        Idempotent per actor id (re-running with the same base updates
        each cell in place). Returns ``{actor_id: URLabSpawnHandle}``.
        Editor-only.
        """
        if isinstance(blueprint, URLabBlueprint):
            bp_str = blueprint.class_path
        else:
            bp_str = str(blueprint)
        if count_x <= 0 or count_y <= 0:
            raise ValueError("spawn_grid requires count_x > 0 and count_y > 0")
        if rotation_quat is not None and rotation_euler is not None:
            raise ValueError("pass at most one of rotation_quat / rotation_euler")

        payload: Dict[str, Any] = {
            "blueprint":     bp_str,
            "base_actor_id": str(base_actor_id),
            "count_x":       int(count_x),
            "count_y":       int(count_y),
            "spacing":       [float(x) for x in spacing],
            "origin":        [float(x) for x in origin],
            "scale":         [float(x) for x in scale],
        }
        if rotation_quat is not None:
            payload["rotation_quat"] = [float(x) for x in rotation_quat]
        elif rotation_euler is not None:
            payload["rotation_euler"] = [float(x) for x in rotation_euler]

        reply = self._client._run_editor_job(
            "spawn_grid", payload, expected_op="spawn_grid_ok",
        )
        bp_class = str(reply.get("blueprint_class_path", "") or bp_str)
        handles: Dict[str, URLabSpawnHandle] = {}
        for a in (reply.get("actors") or []):
            # The per-cell entry carries actor_id + location + was_existing;
            # synthesise a full spawn handle by stamping in the shared
            # blueprint_class_path so each handle is self-describing.
            entry = dict(a)
            entry.setdefault("blueprint_class_path", bp_class)
            handle = _editor_from_spawn_reply(entry)
            handles[handle.actor_id] = handle
        return handles

    def spawn_light(
        self,
        kind: str = "directional",
        *,
        actor_id: str = "",
        location: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_euler: Sequence[float] = (0.0, 0.0, 0.0),
        intensity: float = 5000.0,
        color: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> URLabLightHandle:
        """Spawn a directional / point / spot light. Returns URLabLightHandle."""
        payload: Dict[str, Any] = {
            "kind": str(kind),
            "actor_id": str(actor_id),
            **pose_payload(location=location, rotation_euler=rotation_euler),
            "intensity": float(intensity),
            "color":     [float(x) for x in color],
        }
        reply = self._client._run_editor_job("spawn_light", payload, expected_op="spawn_light_ok")
        return _light_from_spawn_reply(reply)

    def remove_actor(self, target: str, *, by_name: bool = False) -> None:
        """Destroy an actor by actor id (default) or actor name."""
        self._client._run_editor_job(
            "remove_actor", target_payload(target, by_name=by_name),
            expected_op="remove_actor_ok",
        )

    def set_actor_transform(
        self,
        target: str,
        *,
        by_name: bool = False,
        location: Optional[Sequence[float]] = None,
        rotation_quat: Optional[Sequence[float]] = None,
        rotation_euler: Optional[Sequence[float]] = None,
    ) -> None:
        """Edit-time SetActorTransform on a previously-spawned actor."""
        if rotation_quat is not None and rotation_euler is not None:
            raise ValueError("pass at most one of rotation_quat / rotation_euler")
        payload: Dict[str, Any] = {
            **target_payload(target, by_name=by_name),
            **pose_payload(
                location=location,
                rotation_quat=rotation_quat,
                rotation_euler=rotation_euler if rotation_quat is None else None,
            ),
        }
        self._client._run_editor_job(
            "set_actor_transform", payload,
            expected_op="set_actor_transform_ok",
        )

    def apply_scene(
        self,
        level_name: str,
        assets: Sequence[URLabAsset],
        *,
        save: bool = True,
    ) -> Dict[str, URLabSpawnHandle]:
        """Compose the editor primitives into a scene-build flow.

        1. Try ``load_level(level_name)`` — fall through to ``create_level``
           on any RPC error (level doesn't exist yet).
        2. ``import_xml`` each unique asset path; cache the resulting
           :class:`URLabBlueprint`.
        3. ``spawn_actor`` once per ``URLabAsset`` (idempotent on ``actor_id``).
        4. ``save_level`` when ``save=True`` (default).

        Returns ``{actor_id: URLabSpawnHandle}``. ``spawn_actor`` is now
        idempotent on ``actor_id``: a second call with the same id will
        update the existing actor in place rather than producing a
        duplicate; the handle's ``was_existing`` flag distinguishes the
        two paths.
        """
        try:
            self.load_level(level_name)
        except URLabRPCError as exc:
            logger.debug(
                "scene.apply_scene: load_level(%r) failed (%s); falling back to create_level",
                level_name, exc.code,
            )
            self.create_level(level_name)

        blueprints: Dict[str, URLabBlueprint] = {}
        for spec in assets:
            if spec.xml in blueprints:
                continue
            blueprints[spec.xml] = self.import_xml(spec.xml)

        spawned: Dict[str, URLabSpawnHandle] = {}
        for spec in assets:
            bp = blueprints[spec.xml]
            ed = self.spawn_actor(
                blueprint=bp,
                actor_id=spec.actor_id,
                location=spec.location,
                rotation_quat=spec.rotation_quat,
                rotation_euler=spec.rotation_euler,
                scale=spec.scale,
            )
            spawned[spec.actor_id] = ed

        if save:
            self.save_level()
        return spawned
