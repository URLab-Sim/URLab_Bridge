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

"""Shared ``state/full`` snapshot reader for the streaming policy tools.

The per-component raw-binary ZMQ topics (``<art>/joint/<name>``,
``<art>/base_state/<name>``, ``<art>/sensor/<name>``, ``<art>/twist``,
``scene/<name>/xpos`` ...) were removed. Every streaming consumer now
subscribes to the single canonical ``state/full`` msgpack snapshot and
indexes into its ``arts`` / ``scene`` blocks.

Snapshot layout (published every physics step on the state PUB socket, at
observation level "standard")::

    {
      "op": "state_full", "time": float, "step": int,
      "sim_time": {"sec", "nsec"}, "wall_time": {"sec", "nsec"},
      "arts": {"<art>": {"qpos": [...], "qvel": [...],
                          "ctrl": [...], "act": [...],
                          "sensors": {"<part>": [...]},
                          "twist": {"linear": [3], "angular": [3]},
                          "actions": int}},
      "scene": {"<name>": {"xpos": [3], "xquat": [4],
                            "qpos": [7]?, "qvel": [6]?}},
    }

``qpos`` / ``qvel`` are concatenated per articulation in joint discovery
order (the same order the info-socket ``actuator_list`` and MJB walk use);
a leading free joint occupies ``qpos[0:7]`` / ``qvel[0:6]``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import zmq

logger = logging.getLogger(__name__)

try:  # pragma: no cover - trivial import guard
    import msgpack  # type: ignore
except ImportError:  # pragma: no cover
    msgpack = None  # noqa: N816

STATE_TOPIC = b"state/full"


class StateStream:
    """Drain-to-latest subscriber for the ``state/full`` snapshot.

    Owns a ZMQ SUB socket on the state endpoint and keeps only the freshest
    decoded snapshot. The caller pumps :meth:`drain` from its own loop.
    """

    def __init__(self, ctx: "zmq.Context", endpoint: str, *, rcvtimeo_ms: int = 100):
        if msgpack is None:  # pragma: no cover - dependency always present
            raise RuntimeError("msgpack is required to decode the state/full stream")
        self._sock = ctx.socket(zmq.SUB)
        self._sock.connect(endpoint)
        self._sock.setsockopt(zmq.SUBSCRIBE, STATE_TOPIC)
        self._sock.setsockopt(zmq.RCVTIMEO, rcvtimeo_ms)
        self._latest: Optional[Dict[str, Any]] = None

    def drain(self) -> Optional[Dict[str, Any]]:
        """Consume all pending snapshots, keep the newest. Returns the latest
        snapshot seen so far (or None if none has arrived)."""
        while True:
            try:
                self._sock.recv(flags=zmq.NOBLOCK)          # topic frame
                payload = self._sock.recv(flags=zmq.NOBLOCK)  # msgpack snapshot
            except zmq.Again:
                break
            try:
                self._latest = msgpack.unpackb(
                    payload, raw=False, strict_map_key=False
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("state/full decode failed: %s", exc)
        return self._latest

    @property
    def latest(self) -> Optional[Dict[str, Any]]:
        return self._latest

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


# --- snapshot accessors ----------------------------------------------------


def art_names(snap: Optional[Mapping[str, Any]]) -> List[str]:
    """Articulation keys present in the snapshot."""
    if not snap:
        return []
    return list((snap.get("arts") or {}).keys())


def scene_names(snap: Optional[Mapping[str, Any]]) -> List[str]:
    """Scene-entity keys present in the snapshot."""
    if not snap:
        return []
    return list((snap.get("scene") or {}).keys())


def art_block(snap: Optional[Mapping[str, Any]], prefix: str) -> Optional[Mapping[str, Any]]:
    if not snap:
        return None
    return (snap.get("arts") or {}).get(prefix)


def scene_block(snap: Optional[Mapping[str, Any]], name: str) -> Optional[Mapping[str, Any]]:
    if not snap:
        return None
    return (snap.get("scene") or {}).get(name)


def art_qpos(snap: Optional[Mapping[str, Any]], prefix: str) -> np.ndarray:
    block = art_block(snap, prefix)
    if not block:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(block.get("qpos", []), dtype=np.float64)


def art_qvel(snap: Optional[Mapping[str, Any]], prefix: str) -> np.ndarray:
    block = art_block(snap, prefix)
    if not block:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(block.get("qvel", []), dtype=np.float64)


def art_sensors(snap: Optional[Mapping[str, Any]], prefix: str) -> Dict[str, np.ndarray]:
    block = art_block(snap, prefix)
    if not block:
        return {}
    return {
        name: np.asarray(vals, dtype=np.float64)
        for name, vals in (block.get("sensors") or {}).items()
    }


def art_twist(snap: Optional[Mapping[str, Any]], prefix: str) -> Optional[np.ndarray]:
    """Flat 6-vec ``[linear xyz, angular xyz]`` for a twist-controlled art, or
    None when the articulation has no twist controller."""
    block = art_block(snap, prefix)
    if not block:
        return None
    twist = block.get("twist")
    if not isinstance(twist, Mapping):
        return None
    lin = np.asarray(twist.get("linear", [0.0, 0.0, 0.0]), dtype=np.float64)
    ang = np.asarray(twist.get("angular", [0.0, 0.0, 0.0]), dtype=np.float64)
    return np.concatenate([lin[:3], ang[:3]])


def free_base_state(qpos: np.ndarray, qvel: np.ndarray):
    """Extract root pose / velocity from a leading free joint.

    Returns ``(pos[3], quat_xyzw[4], lin_vel[3], ang_vel[3])`` or None when
    the arrays are too short to hold a free joint. Quaternion is converted
    from MuJoCo ``(w, x, y, z)`` to ``(x, y, z, w)`` -- the convention the
    old ``base_state`` binary topic used. ``ang_vel`` is in the body frame
    (MuJoCo free-joint convention), matching the retired stream.
    """
    if qpos.size < 7 or qvel.size < 6:
        return None
    pos = qpos[0:3].copy()
    quat_xyzw = np.array([qpos[4], qpos[5], qpos[6], qpos[3]], dtype=np.float64)
    lin_vel = qvel[0:3].copy()
    ang_vel = qvel[3:6].copy()
    return pos, quat_xyzw, lin_vel, ang_vel
