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

"""Find and join live owner sessions -- the scriptable side of the peek.

An *owner* (a Python client or a UE instance) advertises itself in the shared
registry (role ``fastpath_owner``) with the transports a viewer can reach it on
(ZMQ bus/control and/or a gRPC endpoint), its scene, and capabilities. This
module discovers those, and joins one as a viewer (a mujoco/pystudio peek) or VR.

CLI (``python -m urlab_client.session``):

    session list [--endpoints h1:50051,h2:50051] [--registry DIR]
    session join <target> --model scene.xml --mode viewer [--transport grpc|zmq]
    session join <target> --model scene.xml --mode vr        # launches a UE viewer

``target`` is an instance id, a host, or a ``host:port`` endpoint. The rich GUI
server browser lives in the UE plugin (SMjServerBrowser); this is the headless
counterpart over the same registry.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .pool import (
    OWNER_ROLE,
    _normalize_time,
    default_registry_dir,
    is_owner_entry,
    pid_alive,
)

__all__ = ["OwnerInfo", "discover_owners", "format_table", "OWNER_ROLE"]

DEFAULT_GRPC_PORT = 50051


@dataclass
class OwnerInfo:
    """One discoverable owner session."""

    instance_id: str
    host: str
    scene: str = ""
    role: str = OWNER_ROLE
    caps: Tuple[str, ...] = ()
    transports: Tuple[str, ...] = ()
    bus: Optional[str] = None       # ZMQ viewer PUB (tcp://host:port)
    control: Optional[str] = None   # ZMQ control REP (tcp://host:port)
    grpc: Optional[str] = None      # host:port
    ngeom: int = 0
    pid: Optional[int] = None
    updated: Optional[float] = None  # epoch seconds (normalized)
    source: str = "registry"

    def endpoint_for(self, transport: str) -> Optional[str]:
        """The endpoint a viewer dials for the given transport."""
        if transport == "grpc":
            return self.grpc
        if transport == "zmq":
            return self.control  # perturb RPC target; the bus is separate
        return None

    def default_transport(self) -> str:
        """Prefer gRPC when advertised, else ZMQ."""
        if "grpc" in self.transports or self.grpc:
            return "grpc"
        return "zmq"


def _host_of(endpoint: Optional[str]) -> str:
    if not endpoint:
        return ""
    return endpoint.replace("tcp://", "", 1).split("/", 1)[0].split(":", 1)[0]


def _owner_from_entry(data: dict) -> Optional[OwnerInfo]:
    if not is_owner_entry(data):
        return None
    role = str(data.get("role", ""))
    caps = tuple(str(c) for c in (data.get("capabilities") or []))
    transports = data.get("transports")
    if not transports:
        transports = ["zmq"] + (["grpc"] if data.get("grpc") else [])
    host = str(data.get("host") or _host_of(data.get("bus") or data.get("control")))
    return OwnerInfo(
        instance_id=str(data.get("instance_id", "?")),
        host=host,
        scene=str(data.get("scene", "")),
        role=role or OWNER_ROLE,
        caps=caps,
        transports=tuple(str(t) for t in transports),
        bus=data.get("bus"),
        control=data.get("control"),
        grpc=data.get("grpc"),
        ngeom=int(data.get("ngeom", 0) or 0),
        pid=data.get("pid") if isinstance(data.get("pid"), int) else None,
        updated=_normalize_time(data.get("registry_written_at")),
        source="registry",
    )


def _owner_from_endpoint(ep: str) -> OwnerInfo:
    """An explicitly-given endpoint (assumed a gRPC host:port)."""
    s = ep.replace("tcp://", "", 1)
    host, _, port = s.rpartition(":")
    host = host or "127.0.0.1"
    port = port or str(DEFAULT_GRPC_PORT)
    return OwnerInfo(
        instance_id=f"{host}:{port}", host=host, scene="(explicit)",
        transports=("grpc",), grpc=f"{host}:{port}", source="endpoint",
    )


def _is_local_host(host: Optional[str]) -> bool:
    """True if ``host`` names this machine (so a registry pid can be liveness-checked)."""
    if not host:
        return True  # no host recorded -> assume it was written locally
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return True
    try:
        return host in (socket.gethostname(), socket.getfqdn())
    except OSError:
        return False


def discover_owners(
    registry_dir: Optional[str] = None,
    endpoints: Optional[Sequence[str]] = None,
    include_dead: bool = False,
) -> List[OwnerInfo]:
    """Owners from the registry directory plus any explicit ``endpoints``
    (``["host:port", ...]`` -- assumed gRPC). Registry entries first.

    An owner killed with SIGKILL never runs its ``close()`` and so leaves its
    registry file behind; by default those dead-pid ghosts are filtered out (and
    their local files opportunistically pruned) so ``list``/``join`` only show
    reachable owners. Pass ``include_dead=True`` to keep them (diagnostics)."""
    out: List[OwnerInfo] = []
    rdir = registry_dir or default_registry_dir()
    for path in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        owner = _owner_from_entry(data)
        if owner is None:
            continue
        # Drop (and prune) entries whose owning process is gone. A pid is only
        # meaningful on the host that wrote it, so the liveness check applies to
        # LOCAL owners only; a remote owner's pid is unknowable here, so keep it.
        if (not include_dead and owner.pid is not None
                and _is_local_host(owner.host) and not pid_alive(owner.pid)):
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        out.append(owner)
    for ep in (endpoints or []):
        ep = ep.strip()
        if ep:
            out.append(_owner_from_endpoint(ep))
    return out


def format_table(owners: Sequence[OwnerInfo]) -> str:
    """A compact list of sessions for the CLI."""
    if not owners:
        return "(no owner sessions found)"
    rows = [("INSTANCE", "HOST", "SCENE", "TRANSPORTS", "CAPS")]
    for o in owners:
        rows.append((
            o.instance_id[:24], o.host or "?", (o.scene or "-")[:20],
            ",".join(o.transports) or "-", ",".join(o.caps) or "-",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )


def _resolve(target: str, owners: Sequence[OwnerInfo]) -> Optional[OwnerInfo]:
    """Match a target (instance id / host / host:port) to a discovered owner, or
    treat a bare host:port as an explicit gRPC endpoint."""
    for o in owners:
        if target in (o.instance_id, o.host, o.grpc):
            return o
    if ":" in target or target.replace(".", "").isdigit():
        return _owner_from_endpoint(target)
    for o in owners:
        if o.host == target or o.instance_id.startswith(target):
            return o
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="urlab_client.session", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p):
        p.add_argument("--registry", default=None, help="registry dir (default: shared)")
        p.add_argument("--endpoints", default=None,
                       help="extra gRPC endpoints, comma-separated (h1:50051,h2:50051)")

    pl = sub.add_parser("list", help="list discoverable owner sessions")
    _common(pl)
    pl.add_argument("--all", action="store_true",
                    help="include dead-pid ghosts (default: hide + prune them)")

    pj = sub.add_parser("join", help="join an owner as a VR mirror")
    _common(pj)
    pj.add_argument("target", help="instance id / host / host:port")
    pj.add_argument("--model", required=True, help="scene xml/mjb the owner runs")
    # The old Python "viewer" (peek) mode was removed in Phase 3.3; a desktop mirror
    # is now an ordinary UE transform-mirror renderer. Only VR join remains.
    pj.add_argument("--mode", choices=["vr"], default="vr")
    pj.add_argument("--transport", choices=["zmq", "grpc"], default=None,
                    help="default: what the owner advertises")

    args = ap.parse_args(argv)
    eps = [e for e in (args.endpoints or "").split(",") if e]
    owners = discover_owners(args.registry, eps, include_dead=getattr(args, "all", False))

    if args.cmd == "list":
        print(format_table(owners))
        return 0

    owner = _resolve(args.target, owners)
    if owner is None:
        print(f"no owner matched {args.target!r}; try 'session list'")
        return 1

    return _join_vr(owner)


def _join_vr(owner: OwnerInfo) -> int:
    """Launch a UE viewer instance pointed at the owner's transform stream. The
    viewer boots as an ordinary transform-mirror renderer (Drive=stream, no engine,
    no mj_forward) with the free-fly drone / VR pawn (-URLabCaps=vr)."""
    src = owner.bus or owner.control
    if not src:
        print("owner advertises no ZMQ viewer bus for a UE viewer; a UE-gRPC "
              "viewer stream is pending (Phase 2 UE).")
        return 1
    ue = os.environ.get("URLAB_UE")
    proj = os.environ.get("URLAB_UPROJECT")
    if not ue or not proj:
        print("set URLAB_UE and URLAB_UPROJECT to launch a UE VR viewer. Command:")
        ue = ue or "<UnrealEditor>"
        proj = proj or "<project.uproject>"
    cmd = (f'{ue} {proj} /Game/FastPath/FastPathRender -game '
           f'-URLabDrive=stream:{src} -URLabCaps=vr -windowed')
    print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
