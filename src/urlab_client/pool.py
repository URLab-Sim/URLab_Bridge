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

"""Render-farm pool discovery and cooperative leasing for :class:`URLabClient`.

An Unreal editor instance, on server start, writes a registry file
``{registry_dir}/{instance_id}_{pid}.json`` describing how to reach it (host,
step / state / camera ports, capabilities, busy flag). This module reads those
files to discover a same-host pool, or probes a static list of remote
``host:step_port`` endpoints over ZMQ ``hello`` for cross-machine pools.

``URLabPool.lease`` picks a free instance, connects a :class:`URLabClient`, and
claims it with the cooperative ``acquire_lease`` op so two pool clients never
grab the same process. The returned client releases the lease on ``close()`` /
context-manager exit.

The registry is file-based with no broker: it works before any client has
connected and needs nothing running but the editors themselves. See
``docs/plan_render_farm.md`` sections 3.3, 3.4 and 3.6.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from .client import URLabClient
from .errors import URLabRPCError

logger = logging.getLogger(__name__)

# Default port layout (must match the UE side's -URLabInstanceIndex derivation):
# step = base + i * stride + 0, state = +1, cam_base = +2.
DEFAULT_PORT_BASE = 5559
DEFAULT_PORT_STRIDE = 10

# A registry file whose mtime is older than this (seconds) is treated as dead
# even if the PID check is inconclusive. The editor refreshes its file's mtime
# on a server-tick heartbeat, so a live instance never looks this stale.
DEFAULT_REGISTRY_TTL_S = 30.0

# The registry role/capability string that marks an entry as a joinable fast-path
# owner (a producer a viewer/renderer can join). Matches the UE writer
# (InstanceRegistry.cpp) and the Python writer (fastpath_owner.py::_write_registry).
OWNER_ROLE = "fastpath_owner"


def is_owner_entry(data: Mapping[str, Any]) -> bool:
    """The one shared role/capability match for the role-filtered registry readers.

    An entry is an owner when its ``role`` is ``fastpath_owner`` *or* its
    ``capabilities`` list contains ``fastpath_owner``. This is the single rule
    used by every reader that filters by role (:func:`session.discover_owners`
    and the UE ``DiscoverDrivers``) so they agree on what counts as an owner.

    :func:`read_registry` stays role-agnostic (liveness/staleness only, the farm
    read); if a caller ever wants to role-filter that list it uses this predicate
    so the rule lives in one place.
    """
    role = str(data.get("role", ""))
    caps = [str(c) for c in (data.get("capabilities") or [])]
    return role == OWNER_ROLE or OWNER_ROLE in caps


def default_registry_dir() -> str:
    """Return the registry directory: ``$URLAB_REGISTRY_DIR`` if set, else the
    platform cache location (``%LOCALAPPDATA%/URLab/registry`` on Windows,
    ``$XDG_CACHE_HOME`` or ``~/.cache`` ``/URLab/registry`` elsewhere).

    Mirrors the UE-side default so a client with no configuration finds the
    editors launched by ``urlab-farm up`` on the same machine.
    """
    override = os.environ.get("URLAB_REGISTRY_DIR")
    if override:
        return override
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    return os.path.join(base, "URLab", "registry")


def pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently running.

    Cross-platform and dependency-free: Windows opens a limited-information
    handle and checks the exit code; POSIX uses ``os.kill(pid, 0)``. A PID we
    lack permission to signal is treated as alive (it exists).
    """
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                # Handle opened but status unreadable: assume alive.
                return True
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, just not ours to signal
        return True
    except OverflowError:  # pid too large to be a real pid_t -> not a process
        return False
    except OSError:
        return False
    return True


def _derive_ports(
    index: int, port_base: int = DEFAULT_PORT_BASE, port_stride: int = DEFAULT_PORT_STRIDE
) -> Tuple[int, int, int]:
    """(step, state, cam_base) ports for an instance index, matching the UE
    ``PortBase + i*PortStride + {0,1,2}`` derivation."""
    root = port_base + index * port_stride
    return root, root + 1, root + 2


def _parse_endpoint(endpoint: str) -> Tuple[str, int]:
    """Split ``host:port`` / ``tcp://host:port`` into ``(host, port)``. A bare
    ``host`` (no port) defaults to the standard step port."""
    text = endpoint.strip()
    if "://" not in text:
        text = "tcp://" + text
    parsed = urlparse(text)
    host = parsed.hostname or "localhost"
    port = parsed.port if parsed.port is not None else DEFAULT_PORT_BASE
    return host, int(port)


@dataclass
class InstanceInfo:
    """One discoverable render-farm instance.

    Built either from a registry file (``source='registry'``) or from a live
    ``hello`` probe of a static endpoint (``source='static'``). ``host`` is
    always a connectable host: for static discovery it is the endpoint the
    client actually reached, never the advertised bind wildcard.
    """

    instance_id: str
    index: int
    pid: int
    host: str
    step_port: int
    state_port: int
    cam_base_port: int
    manager_present: bool = False
    busy: bool = False
    urlab_version: str = ""
    capabilities: Tuple[str, ...] = ()
    registry_written_at: Optional[float] = None
    source: str = "registry"
    registry_path: Optional[str] = None
    registry_mtime: Optional[float] = None

    @property
    def address(self) -> str:
        """``tcp://host`` for :class:`URLabClient` construction."""
        return f"tcp://{self.host}"

    def is_alive(self) -> bool:
        """True if the PID is running. Only meaningful for same-host
        (registry) instances; static instances report ``pid`` as advertised
        and their liveness is really the successful probe."""
        return pid_alive(self.pid)

    def is_stale(self, ttl_s: float = DEFAULT_REGISTRY_TTL_S, now: Optional[float] = None) -> bool:
        """True if the registry file's mtime is older than ``ttl_s``. Always
        False for static instances (no file to age out)."""
        if self.registry_mtime is None:
            return False
        current = time.time() if now is None else now
        return (current - self.registry_mtime) > ttl_s

    def has_caps(self, require_caps: Optional[Iterable[str]]) -> bool:
        """True if this instance's capabilities are a superset of
        ``require_caps`` (or ``require_caps`` is empty / None)."""
        if not require_caps:
            return True
        return set(require_caps).issubset(set(self.capabilities))

    @classmethod
    def from_registry(
        cls,
        data: Mapping[str, Any],
        *,
        path: Optional[str] = None,
        mtime: Optional[float] = None,
    ) -> "InstanceInfo":
        """Build from a parsed registry-file dict."""
        index = int(data.get("index", 0) or 0)
        step, state, cam = _derive_ports(index)
        caps = data.get("capabilities") or []
        return cls(
            instance_id=str(data.get("instance_id", "")),
            index=index,
            pid=int(data.get("pid", 0) or 0),
            host=str(data.get("host", "") or "localhost"),
            step_port=int(data.get("step_port", step) or step),
            state_port=int(data.get("state_port", state) or state),
            cam_base_port=int(data.get("cam_base_port", cam) or cam),
            manager_present=bool(data.get("manager_present", False)),
            busy=bool(data.get("busy", False)),
            urlab_version=str(data.get("urlab_version", "") or ""),
            capabilities=tuple(str(c) for c in caps),
            registry_written_at=(
                float(data["registry_written_at"])
                if isinstance(data.get("registry_written_at"), (int, float))
                else None
            ),
            source="registry",
            registry_path=path,
            registry_mtime=mtime,
        )

    @classmethod
    def from_instance_block(
        cls,
        data: Mapping[str, Any],
        *,
        host: str,
        step_port: int,
    ) -> "InstanceInfo":
        """Build from a ``hello_ok`` ``instance`` sub-object probed over the
        network. ``host`` / ``step_port`` are the reachable endpoint the client
        used, which override any advertised bind wildcard in the block."""
        index = int(data.get("index", 0) or 0)
        _step, state, cam = _derive_ports(index)
        caps = data.get("capabilities") or []
        return cls(
            instance_id=str(data.get("instance_id", "")),
            index=index,
            pid=int(data.get("pid", 0) or 0),
            host=host,
            step_port=step_port,
            state_port=int(data.get("state_port", state) or state),
            cam_base_port=int(data.get("cam_base_port", cam) or cam),
            manager_present=bool(data.get("manager_present", False)),
            busy=bool(data.get("busy", False)),
            urlab_version=str(data.get("urlab_version", "") or ""),
            capabilities=tuple(str(c) for c in caps),
            registry_written_at=(
                float(data["registry_written_at"])
                if isinstance(data.get("registry_written_at"), (int, float))
                else None
            ),
            source="static",
        )


def read_registry(
    registry_dir: Optional[str] = None,
    *,
    include_dead: bool = False,
    include_stale: bool = False,
    ttl_s: float = DEFAULT_REGISTRY_TTL_S,
) -> List[InstanceInfo]:
    """Read every ``*.json`` registry file into an :class:`InstanceInfo` list.

    By default dead-PID and stale (mtime older than ``ttl_s``) entries are
    dropped. Pass ``include_dead`` / ``include_stale`` to keep them (``ps``
    wants the full picture; discovery does not). Unparseable files are skipped
    with a debug log rather than raising, so one corrupt file never breaks
    discovery.
    """
    directory = registry_dir or default_registry_dir()
    out: List[InstanceInfo] = []
    if not os.path.isdir(directory):
        return out
    now = time.time()
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            mtime = os.path.getmtime(path)
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.debug("skipping unreadable registry file %s: %s", path, exc)
            continue
        if not isinstance(data, Mapping):
            continue
        inst = InstanceInfo.from_registry(data, path=path, mtime=mtime)
        if not include_dead and not inst.is_alive():
            continue
        if not include_stale and inst.is_stale(ttl_s, now=now):
            continue
        out.append(inst)
    return out


class URLabPool:
    """Discovery + leasing over a set of render-farm instances.

    Three discovery modes match ``docs/plan_render_farm.md`` section 3.3:

    - Direct: construct :class:`URLabClient` yourself with a known
      ``host:step_port`` (unchanged; not this class).
    - Same host: :meth:`discover` reads the file registry.
    - Multi-machine: :meth:`discover_static` probes a static endpoint list.

    :meth:`lease` claims one free instance cooperatively and returns a
    connected client.
    """

    @staticmethod
    def discover(
        registry_dir: Optional[str] = None,
        *,
        require_caps: Optional[Iterable[str]] = None,
        ttl_s: float = DEFAULT_REGISTRY_TTL_S,
        include_busy: bool = False,
    ) -> List[InstanceInfo]:
        """Discover free same-host instances from the file registry.

        Drops entries whose PID is dead or whose file mtime is older than
        ``ttl_s``, keeps only ``not busy`` (unless ``include_busy``), and filters
        to instances whose capabilities are a superset of ``require_caps``.
        Manager presence is not required: an editor is discoverable pre-PIE.
        """
        alive = read_registry(registry_dir, ttl_s=ttl_s)
        out: List[InstanceInfo] = []
        for inst in alive:
            if inst.busy and not include_busy:
                continue
            if not inst.has_caps(require_caps):
                continue
            out.append(inst)
        return out

    @staticmethod
    def discover_static(
        endpoints: Sequence[str],
        *,
        require_caps: Optional[Iterable[str]] = None,
        include_busy: bool = False,
        timeout_s: float = 5.0,
    ) -> List[InstanceInfo]:
        """Discover instances across machines by probing a static endpoint list.

        The file registry is same-host only (local filesystem), so a
        multi-machine pool passes ``host:step_port`` strings; each is queried
        with a lightweight ZMQ ``hello`` and its ``instance`` block is read for
        busy / capabilities. Unreachable endpoints are skipped with a warning.
        """
        out: List[InstanceInfo] = []
        for endpoint in endpoints:
            inst = _probe_instance(endpoint, timeout_s=timeout_s)
            if inst is None:
                continue
            if inst.busy and not include_busy:
                continue
            if not inst.has_caps(require_caps):
                continue
            out.append(inst)
        return out

    @staticmethod
    def lease(
        candidates: Union[InstanceInfo, Iterable[InstanceInfo]],
        *,
        owner: Optional[str] = None,
        ttl_s: int = 60,
        transport: str = "auto",
        step_mode: str = "auto",
        observations: str = "standard",
        recv_timeout_ms: int = 5000,
    ) -> URLabClient:
        """Claim one free instance and return a connected, leased client.

        Tries each candidate in order: connects a :class:`URLabClient` (transport
        ``auto`` upgrades to SHM when co-located), sends ``acquire_lease``, and
        on a ``busy`` reply moves to the next. The winning client carries the
        server-issued ``lease_id`` and releases it on ``close()`` /
        context-manager exit. Raises :class:`RuntimeError` if every candidate is
        busy or unreachable.
        """
        if isinstance(candidates, InstanceInfo):
            candidates = [candidates]
        candidate_list = list(candidates)
        if not candidate_list:
            raise RuntimeError("URLabPool.lease: no candidate instances supplied")

        errors: List[str] = []
        for inst in candidate_list:
            client = URLabClient(
                inst.address,
                step_mode=step_mode,
                step_port=inst.step_port,
                state_port=inst.state_port,
                transport=transport,
                recv_timeout_ms=recv_timeout_ms,
            )
            try:
                client.connect(observations=observations)
            except Exception as exc:  # unreachable / handshake failure: try next
                errors.append(f"{inst.instance_id or inst.host}: connect failed ({exc})")
                _safe_close(client)
                continue
            payload: Dict[str, Any] = {"ttl_s": int(ttl_s)}
            if owner is not None:
                payload["owner"] = owner
            try:
                reply = client._rpc("acquire_lease", payload)
            except URLabRPCError as exc:
                if exc.code == "busy":
                    errors.append(f"{inst.instance_id or inst.host}: busy")
                    _safe_close(client)
                    continue
                _safe_close(client)
                raise
            client.lease_id = reply.get("lease_id")
            client._leased_instance = inst
            logger.info(
                "URLabPool.lease: claimed instance %s at %s:%d (lease_id=%s)",
                inst.instance_id or "?", inst.host, inst.step_port, client.lease_id,
            )
            return client

        raise RuntimeError(
            "URLabPool.lease: no free instance among "
            f"{len(candidate_list)} candidate(s): " + "; ".join(errors)
        )


def _probe_instance(endpoint: str, *, timeout_s: float = 5.0) -> Optional[InstanceInfo]:
    """Send one ``hello`` to ``endpoint`` and return its instance block, or None
    if the endpoint is unreachable / gives no instance data."""
    try:
        import msgpack  # type: ignore
        import zmq  # type: ignore
    except ImportError as exc:  # pragma: no cover - deployment without deps
        logger.warning("discover_static needs pyzmq + msgpack (%s)", exc)
        return None

    host, port = _parse_endpoint(endpoint)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(f"tcp://{host}:{port}")
        sock.send(
            msgpack.packb(
                {
                    "op": "hello",
                    "client_version": "urlab_farm/discover",
                    "encoding": "msgpack",
                },
                use_bin_type=True,
            )
        )
        raw = sock.recv()
        reply = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception as exc:
        logger.warning("discover_static: %s:%d unreachable (%s)", host, port, exc)
        return None
    finally:
        sock.close(linger=0)
        ctx.term()

    if not isinstance(reply, Mapping):
        return None
    block = reply.get("instance")
    if not isinstance(block, Mapping):
        # Older server without an instance block: synthesise a minimal entry
        # from the reachable endpoint so leasing can still target it.
        block = {
            "urlab_version": reply.get("urlab_version", ""),
            "manager_present": reply.get("manager_present", False),
        }
    return InstanceInfo.from_instance_block(block, host=host, step_port=port)


def _safe_close(client: URLabClient) -> None:
    try:
        client.close()
    except Exception:  # pragma: no cover - best-effort teardown
        pass


__all__ = [
    "DEFAULT_PORT_BASE",
    "DEFAULT_PORT_STRIDE",
    "DEFAULT_REGISTRY_TTL_S",
    "OWNER_ROLE",
    "InstanceInfo",
    "URLabPool",
    "default_registry_dir",
    "is_owner_entry",
    "pid_alive",
    "read_registry",
]
