# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Session discovery tests: registry parsing (owner filter + int/ISO timestamp
normalization), explicit endpoints, target resolution, and the list table. Pure
Python -- no UE, no owner running."""
from __future__ import annotations

import json

from urlab_client.session import (
    OWNER_ROLE,
    discover_owners,
    format_table,
)
from urlab_client.session import _resolve  # noqa: PLC2701 - unit under test


def _write(tmp, name, data):
    (tmp / name).write_text(json.dumps(data))


def test_discover_owner_and_ignore_non_owners(tmp_path):
    _write(tmp_path, "fastpath_demo_1.json", {
        "instance_id": "demo", "role": OWNER_ROLE, "host": "10.0.0.1",
        "scene": "aloha", "capabilities": ["fastpath_owner", "view"],
        "transports": ["zmq", "grpc"], "bus": "tcp://10.0.0.1:5561",
        "control": "tcp://10.0.0.1:5571", "grpc": "10.0.0.1:50051",
        "pid": 123, "registry_written_at": 1700000000,
    })
    # a render server, not an owner -> ignored
    _write(tmp_path, "render_2.json", {
        "instance_id": "r", "role": "", "capabilities": ["render_sync"],
        "host": "10.0.0.2", "registry_written_at": "2026-08-20T10:00:00",
    })
    owners = discover_owners(registry_dir=str(tmp_path))
    assert len(owners) == 1
    o = owners[0]
    assert o.instance_id == "demo" and o.host == "10.0.0.1" and o.scene == "aloha"
    assert o.transports == ("zmq", "grpc") and o.grpc == "10.0.0.1:50051"
    assert o.default_transport() == "grpc"
    assert o.endpoint_for("grpc") == "10.0.0.1:50051"
    assert o.endpoint_for("zmq") == "tcp://10.0.0.1:5571"
    assert o.updated == 1700000000.0


def test_iso8601_timestamp_normalized(tmp_path):
    _write(tmp_path, "fastpath_ue_1.json", {
        "instance_id": "ue", "role": OWNER_ROLE, "host": "h",
        "registry_written_at": "2026-08-20T10:00:00",
    })
    o = discover_owners(registry_dir=str(tmp_path))[0]
    assert o.updated is not None and o.updated > 0
    # no explicit transports field -> inferred from grpc absence
    assert o.transports == ("zmq",)


def test_explicit_endpoints(tmp_path):
    owners = discover_owners(registry_dir=str(tmp_path), endpoints=["1.2.3.4:50052"])
    assert len(owners) == 1
    assert owners[0].grpc == "1.2.3.4:50052"
    assert owners[0].transports == ("grpc",)
    assert owners[0].default_transport() == "grpc"


def test_resolve_by_id_host_endpoint(tmp_path):
    _write(tmp_path, "fastpath_a_1.json", {
        "instance_id": "alpha", "role": OWNER_ROLE, "host": "10.0.0.9",
        "grpc": "10.0.0.9:50051", "transports": ["grpc"],
    })
    owners = discover_owners(registry_dir=str(tmp_path))
    assert _resolve("alpha", owners).instance_id == "alpha"
    assert _resolve("10.0.0.9", owners).instance_id == "alpha"
    # a bare host:port not in the registry becomes an explicit gRPC endpoint
    assert _resolve("5.6.7.8:50051", owners).grpc == "5.6.7.8:50051"


def test_ghost_pruning_local_only(tmp_path, monkeypatch):
    """Only a LOCAL-host entry whose pid is dead is dropped on discover (and its
    file opportunistically pruned). A remote dead entry is kept (its pid is
    unknowable here), and ``include_dead=True`` keeps everything and prunes
    nothing. Covers session.py's ``_is_local_host``-guarded prune path.

    pid_alive is patched to a pure predicate so the test is deterministic and
    doesn't depend on any real process id being (un)reachable."""
    from urlab_client import session as sess

    live_pid = 4242
    dead_pid = 4343
    monkeypatch.setattr(sess, "pid_alive", lambda pid: pid == live_pid)

    def _owner(instance_id, host, pid):
        return {
            "instance_id": instance_id, "role": OWNER_ROLE, "host": host,
            "grpc": f"{host}:50051", "transports": ["grpc"], "pid": pid,
            "registry_written_at": 1700000000,
        }

    _write(tmp_path, "fastpath_live_local.json", _owner("live-local", "127.0.0.1", live_pid))
    _write(tmp_path, "fastpath_dead_local.json", _owner("dead-local", "127.0.0.1", dead_pid))
    _write(tmp_path, "fastpath_dead_remote.json", _owner("dead-remote", "10.99.99.99", dead_pid))

    # include_dead: all three returned, nothing pruned.
    all_owners = sess.discover_owners(registry_dir=str(tmp_path), include_dead=True)
    assert {o.instance_id for o in all_owners} == {"live-local", "dead-local", "dead-remote"}
    assert (tmp_path / "fastpath_dead_local.json").exists()

    # default: dead-LOCAL dropped + its file unlinked; dead-REMOTE kept.
    owners = sess.discover_owners(registry_dir=str(tmp_path))
    assert {o.instance_id for o in owners} == {"live-local", "dead-remote"}
    assert not (tmp_path / "fastpath_dead_local.json").exists()
    assert (tmp_path / "fastpath_live_local.json").exists()
    assert (tmp_path / "fastpath_dead_remote.json").exists()


def test_format_table():
    assert "no owner" in format_table([])
    owners = discover_owners(endpoints=["9.9.9.9:50051"], registry_dir="/nonexistent")
    out = format_table(owners)
    assert "9.9.9.9" in out and "grpc" in out
