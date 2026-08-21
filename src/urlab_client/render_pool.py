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

"""Render across a POOL of fast-path render-server instances in parallel.

One :class:`~urlab_client.RenderClient` drives one UE instance, which renders its
cameras sequentially. For many cameras that serialisation dominates latency, so
:class:`RenderPool` spreads the cameras across several instances and renders them
**concurrently** -- a near drop-in, parallel version of ``render_mjdata``.

An external orchestrator spins the instances up (across the network and/or
several on one host, each with a distinct ``-URLabDmEnvPort=``) and hands their
addresses to the pool as a **config file** or an explicit **endpoint list** (the
latter is what a ``--endpoints`` CLI flag feeds). The pool attaches to them,
broadcasts the model to all, and each frame splits the requested cameras evenly
across the pool -- **no manual per-instance assignment**.

Everything fans out concurrently: the model broadcast (``load_*``) and every
``render`` submit one task per instance and join, so all requests are in flight
at once (wall-clock ~= the slowest single instance, not the sum). This is safe
because each ``RenderClient`` owns an independent gRPC channel + lock.

Typical use::

    from urlab_client import RenderPool
    with RenderPool.from_config("pool.json") as pool:   # or from_endpoints([...])
        pool.load_xml("scene.xml")
        frames = pool.render_mjdata(model, data)        # all cameras, split N ways
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

from .render_client import (
    USER_CAMERA,
    CameraFrame,
    RenderClient,
    UserPose,
    poses_from_mjdata,
)

__all__ = ["RenderPool", "InstanceSpec", "RenderPoolError", "parse_endpoints"]

DEFAULT_GRPC_PORT = 50051
# Env var naming a JSON pool-config file, used when from_config() gets no path.
ENV_POOL_CONFIG = "URLAB_RENDER_POOL"

# What a single endpoint may be given as: "host:port" / "host" / (host, port) /
# InstanceSpec.
EndpointLike = Union[str, "InstanceSpec", Sequence]


class RenderPoolError(RuntimeError):
    """One or more instances failed during a pool operation.

    ``failures`` maps the failed endpoint string to the exception it raised.
    Because the transport auto-reconnects a dropped stream, most blips self-heal;
    a hard failure surfaces here so the caller can retry the frame.
    """

    def __init__(self, message: str, failures: Dict[str, BaseException]):
        super().__init__(message)
        self.failures = dict(failures)


@dataclass
class InstanceSpec:
    """One render-server endpoint. Endpoint only -- cameras are never pinned to an
    instance; the pool distributes them automatically."""

    host: str
    port: int = DEFAULT_GRPC_PORT

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"


def parse_endpoints(spec: Union[str, Sequence[EndpointLike]]) -> List[InstanceSpec]:
    """Normalise endpoints into ``List[InstanceSpec]``.

    Accepts a comma/space-separated string (``"h1:p1, h2:p2"`` -- what a
    ``--endpoints`` CLI flag passes), or a sequence mixing ``"host:port"``
    strings, ``(host, port)`` tuples, and :class:`InstanceSpec` objects. A bare
    host defaults to port :data:`DEFAULT_GRPC_PORT`.
    """
    if isinstance(spec, str):
        items: Sequence[EndpointLike] = [s for s in spec.replace(" ", ",").split(",") if s]
    else:
        items = list(spec)

    out: List[InstanceSpec] = []
    for it in items:
        if isinstance(it, InstanceSpec):
            out.append(it)
        elif isinstance(it, (tuple, list)):
            host, port = it
            out.append(InstanceSpec(str(host), int(port)))
        else:
            s = str(it).strip()
            if s.startswith("tcp://"):
                s = s[len("tcp://"):]
            if ":" in s:
                host, _, p = s.rpartition(":")
                out.append(InstanceSpec(host, int(p)))
            else:
                out.append(InstanceSpec(s, DEFAULT_GRPC_PORT))
    if not out:
        raise ValueError("no endpoints parsed")
    return out


def _load_config(path: Optional[str]) -> List[InstanceSpec]:
    """Read a JSON pool config: ``{"instances": [{"host":..,"port":..}, ...]}``
    (a bare top-level list is also accepted). ``path`` falls back to
    ``$URLAB_RENDER_POOL``."""
    path = path or os.environ.get(ENV_POOL_CONFIG)
    if not path:
        raise ValueError(
            f"no pool-config path given and ${ENV_POOL_CONFIG} is unset")
    with open(path) as f:
        cfg = json.load(f)
    insts = cfg.get("instances") if isinstance(cfg, dict) else cfg
    if not insts:
        raise ValueError(f"pool config {path!r} has no 'instances'")
    out: List[InstanceSpec] = []
    for e in insts:
        if isinstance(e, dict):
            out.append(InstanceSpec(str(e["host"]), int(e.get("port", DEFAULT_GRPC_PORT))))
        else:
            out.extend(parse_endpoints([e]))
    return out


class RenderPool:
    """A pool of fast-path render servers rendered in parallel.

    Mirrors :class:`~urlab_client.RenderClient`'s ``load_*`` / ``render`` /
    ``render_mjdata`` surface, fanning each call out across every instance.
    """

    def __init__(
        self,
        instances: Sequence[EndpointLike],
        *,
        recv_timeout_ms: int = 10_000,
    ) -> None:
        self._specs = parse_endpoints(list(instances))
        self._clients: List[RenderClient] = [
            RenderClient.grpc(s.host, s.port, recv_timeout_ms=recv_timeout_ms)
            for s in self._specs
        ]
        # One worker per instance so every request is genuinely in flight at once.
        self._exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._clients), thread_name_prefix="RenderPool"
        )
        self._cam_cache: Optional[List[str]] = None

    @classmethod
    def from_endpoints(cls, endpoints: Union[str, Sequence[EndpointLike]],
                       **kw) -> "RenderPool":
        """Build from an explicit endpoint list / string (CLI or code)."""
        return cls(parse_endpoints(endpoints), **kw)

    @classmethod
    def from_config(cls, path: Optional[str] = None, **kw) -> "RenderPool":
        """Build from a JSON pool-config file (``path`` or ``$URLAB_RENDER_POOL``)."""
        return cls(_load_config(path), **kw)

    @property
    def endpoints(self) -> List[str]:
        return [s.endpoint for s in self._specs]

    def __len__(self) -> int:
        return len(self._clients)

    # -- parallel fan-out core --------------------------------------------
    def _fan_out(self, fn, opname: str) -> List:
        """Run ``fn(client, index)`` on every instance concurrently, join, and
        aggregate failures into a single RenderPoolError."""
        futs = {self._exec.submit(fn, c, i): i for i, c in enumerate(self._clients)}
        results: List = [None] * len(self._clients)
        failures: Dict[str, BaseException] = {}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 - surfaced via RenderPoolError
                failures[self._specs[i].endpoint] = exc
        if failures:
            detail = ", ".join(
                f"{ep} ({type(e).__name__}: {e})" for ep, e in failures.items())
            raise RenderPoolError(
                f"{opname} failed on {len(failures)}/{len(self._clients)} "
                f"instance(s): {detail}", failures)
        return results

    # -- model broadcast (parallel to ALL instances) ----------------------
    def load_model(self, data, **kw) -> None:
        self._cam_cache = None
        self._fan_out(lambda c, _i: c.load_model(data, **kw), "load_model")

    def load_xml(self, xml, **kw) -> None:
        self._cam_cache = None
        self._fan_out(lambda c, _i: c.load_xml(xml, **kw), "load_xml")

    def load_mjz(self, mjz, **kw) -> None:
        self._cam_cache = None
        self._fan_out(lambda c, _i: c.load_mjz(mjz, **kw), "load_mjz")

    def load_mjb(self, mjb, **kw) -> None:
        self._cam_cache = None
        self._fan_out(lambda c, _i: c.load_mjb(mjb, **kw), "load_mjb")

    def camera_names(self) -> List[str]:
        """Camera names the servers report (cached; all instances share the
        model, so one probe is enough). Invalidated by any ``load_*``."""
        if self._cam_cache is None:
            self._cam_cache = self._clients[0].camera_names()
        return list(self._cam_cache)

    # -- camera distribution (automatic, per render) ----------------------
    def _distribute(self, cameras: Sequence[str]) -> List[List[str]]:
        """Split ``cameras`` evenly across the pool, round-robin. Any instance can
        render any camera (all hold the full model), so this needs no state."""
        buckets: List[List[str]] = [[] for _ in self._clients]
        for k, cam in enumerate(cameras):
            buckets[k % len(self._clients)].append(cam)
        return buckets

    # -- render (parallel fan-out + merge) --------------------------------
    def render(
        self,
        *,
        bxpos: Sequence[float],
        bxquat: Sequence[float],
        cxpos: Optional[Sequence[float]] = None,
        cxquat: Optional[Sequence[float]] = None,
        geom_pos: Optional[Sequence[float]] = None,
        geom_quat: Optional[Sequence[float]] = None,
        gxpos: Optional[Sequence[float]] = None,
        gxquat: Optional[Sequence[float]] = None,
        sim_time: float = 0.0,
        cameras: Optional[Sequence[str]] = None,
        user_pose: Optional[UserPose] = None,
        delay: float = 0.0,
        timeout_ms: int = 5000,
    ) -> Dict[str, CameraFrame]:
        """Push one pose set and render ``cameras`` (default: all) split across the
        pool in parallel; returns the merged ``{camera_name: CameraFrame}``.

        The full per-body / per-camera poses go to every instance (they each apply
        the whole state); only the camera *subset* differs. ``user_pose`` is sent
        only to the instance that draws :data:`USER_CAMERA` this frame. ``delay`` is
        in **seconds** (server-side latency-ring sampling).

        ``geom_pos``/``geom_quat`` (local geom offsets) and ``gxpos``/``gxquat``
        (world geom transforms) are the reset-time re-baseline fields. They are only
        forwarded to instances that render a camera this frame, so to guarantee
        **every** instance is re-baselined (including idle ones) use :meth:`reset`.
        """
        names = list(cameras) if cameras is not None else self.camera_names()
        buckets = self._distribute(names)

        def one(client: RenderClient, i: int) -> Dict[str, CameraFrame]:
            subset = buckets[i]
            if not subset:
                return {}
            up = user_pose if (user_pose is not None and USER_CAMERA in subset) else None
            return client.render(
                bxpos=bxpos, bxquat=bxquat, cxpos=cxpos, cxquat=cxquat,
                geom_pos=geom_pos, geom_quat=geom_quat, gxpos=gxpos, gxquat=gxquat,
                sim_time=sim_time, cameras=subset, user_pose=up,
                delay=delay, timeout_ms=timeout_ms,
            )

        merged: Dict[str, CameraFrame] = {}
        for r in self._fan_out(one, "render"):
            if r:
                merged.update(r)
        return merged

    def render_mjdata(
        self, model, data, *, cameras: Optional[Sequence[str]] = None,
        user_pose: Optional[UserPose] = None, delay: float = 0.0,
        sync_geoms: bool = False,
        timeout_ms: int = 5000,
    ) -> Dict[str, CameraFrame]:
        """Convenience: render across the pool straight from a mujoco
        ``(model, data)`` pair (poses are computed once and broadcast).

        ``sync_geoms`` adds the local geom offsets to the pose set. Note it only
        reaches instances that render a camera this frame; for a guaranteed
        all-instance re-baseline on episode reset, call :meth:`reset` instead.
        """
        return self.render(
            sim_time=float(data.time), cameras=cameras, user_pose=user_pose,
            delay=delay, timeout_ms=timeout_ms,
            **poses_from_mjdata(model, data, sync_geoms=sync_geoms),
        )

    # -- reset re-baseline (broadcast to ALL instances) -------------------
    def reset(self, model, *, timeout_ms: int = 5000) -> None:
        """Re-baseline every instance's geom offsets from ``model.geom_pos`` /
        ``model.geom_quat``. Call on episode reset / geom re-randomisation, then
        resume body-only per-step :meth:`render`.

        Unlike ``render_mjdata(sync_geoms=True)``, this reaches **every** instance
        (including any not drawing a camera next frame): each applies + persists
        the new offsets in-engine. The server cannot render zero cameras, so one
        probe camera is captured per instance and discarded.
        """
        import numpy as np  # lazy: only the reset path needs it

        geom_pos = np.asarray(model.geom_pos, np.float64).reshape(-1)
        geom_quat = np.asarray(model.geom_quat, np.float64).reshape(-1)
        names = self.camera_names()
        probe = [names[0]] if names else None  # None -> server renders all (no cameras case)

        # Empty bxpos/bxquat: the server skips body transforms (size mismatch) but
        # still writes the geom offsets into its model -- the same safe request
        # shape camera_names() uses.
        self._fan_out(
            lambda c, _i: c.render(
                bxpos=[], bxquat=[],
                geom_pos=geom_pos, geom_quat=geom_quat,
                cameras=probe, timeout_ms=timeout_ms,
            ),
            "reset",
        )

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        for c in self._clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        self._exec.shutdown(wait=False)

    def __enter__(self) -> "RenderPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
