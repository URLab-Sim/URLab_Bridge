# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Discovery / registry tests for the features built since v0.6.0-beta.

Grounded in `audit_docs/0_render_source_of_truth.md` §12 ("Discovery /
registry -- one schema, one writer per repo, one filter") and
`audit_docs/cleanup_audit_addendum.md`:

  * §12's target is ONE entry schema, a superset of both writers:
    ``{instance_id, role, capabilities, control, bus, transports, grpc,
    step_port, state_port, cam_base_port, grpc_port, manager_present, busy,
    scene, ngeom, host, pid, heartbeat}`` -- everything a joiner needs to
    connect and display an owner. `pool.read_registry` is documented as
    deliberately role-agnostic/minimal (the "farm" read); `session.
    discover_owners` is the joiner-facing read that is supposed to expose the
    full schema.
  * addendum: "`registry_written_at` int-epoch (Python) vs ISO8601 (UE); the
    readers normalize int-epoch vs the UE writer's ISO-8601 string" --
    `session._normalize_time` does this; `pool.InstanceInfo.from_registry`
    (`pool.py:242-246`) only accepts a numeric value and silently drops an
    ISO-8601 timestamp to `None`.
  * addendum §C.2 ("systemic patterns"): "Same concept, N copies... endpoint/
    host parse (×5)... No single source of truth for addressing/parsing" --
    `pool._parse_endpoint`, `render_pool.parse_endpoints` and
    `session._owner_from_endpoint` now all route through the shared,
    scheme-agnostic `pool.parse_host_port` SSOT helper, so they agree on the
    parsed (host, port) for any endpoint scheme (tcp/grpc/shm/bare).
  * `capability_naming_proposal.md` §4.6: the registry `capabilities` key
    means two different things across writers -- UE's (non-broadcasting)
    list is protocol/transport features (`render_sync`, `shm_rpc`, ...);
    Python's is the peer-facing grant set (`stream_cameras`,
    `accept_input`). The shared `role`/`capabilities` owner filter
    (`pool.is_owner_entry`) must not be fooled by that meaning collision.

`test_capabilities_features.py` already covers capability-set round-tripping
through a Python-written registry entry end to end; this file does not repeat
that. `test_session.py`/`test_farm_pool.py` already cover dead-pid ghost
pruning and dead/stale filtering with hand-written registry dicts; this file
adds the same behaviours driven by a REAL `FastPathOwner`-written file (so a
regression in what the owner actually writes would be caught here too), plus
the schema-completeness and cross-module-parse-agreement gaps above that
neither file covers.

Hermetic: real ZMQ sockets on OS-assigned loopback ports, real registry files
under `tmp_path`, no live UE, no network peer.
"""
from __future__ import annotations

import json
import os
import socket
import time

import pytest

msgpack = pytest.importorskip("msgpack")
zmq = pytest.importorskip("zmq")

from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402
from urlab_client.pool import (  # noqa: E402
    InstanceInfo,
    _parse_endpoint,
    is_owner_entry,
    read_registry,
)
from urlab_client.render_pool import parse_endpoints  # noqa: E402
from urlab_client.session import (  # noqa: E402
    OWNER_ROLE,
    OwnerInfo,
    _owner_from_endpoint,
    discover_owners,
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_owner(tmp_path, **kw):
    kw.setdefault("scene", "widgetbot")
    kw.setdefault("control_port", _free_port())
    kw.setdefault("bus_port", _free_port())
    kw.setdefault("advertise_host", "127.0.0.1")
    kw.setdefault("registry_dir", str(tmp_path))
    return FastPathOwner(kw.pop("mjb_bytes", b""), **kw)


def _read_entry(owner) -> dict:
    with open(owner._registry_path) as fh:  # noqa: SLF001 - test introspection
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Registry round-trip: a real owner's entry, read back by each reader.
# --------------------------------------------------------------------------- #

def test_pool_read_registry_exposes_farm_fields_for_a_real_owner(tmp_path):
    """`pool.read_registry` is documented (§12) as the role-agnostic FARM read
    -- liveness/staleness only. It should still surface the fields a farm
    consumer (URLabPool.discover/lease) actually uses: host, step_port
    (the owner's control REP -- see `fastpath_owner.py::_write_registry`'s
    ``step_port: self._control_port`` comment), capabilities and busy."""
    owner = _make_owner(tmp_path, ngeom=7)
    try:
        found = [i for i in read_registry(str(tmp_path)) if i.instance_id == "widgetbot"]
        assert len(found) == 1
        inst = found[0]
        assert isinstance(inst, InstanceInfo)
        assert inst.host == "127.0.0.1"
        assert inst.step_port == owner._control_port  # noqa: SLF001
        assert inst.busy is False
        assert inst.manager_present is True
        assert "fastpath_owner" in inst.capabilities
    finally:
        owner.close()


def test_session_discover_owners_exposes_joiner_fields_for_a_real_owner(tmp_path):
    """EXPECTED FAIL (source-of-truth §12: the one entry schema explicitly
    lists ``ngeom`` alongside scene/host/role/caps as part of what a reader
    exposes, and `fastpath_owner.py::_write_registry` writes ``ngeom`` into
    every entry -- but `session.OwnerInfo` has no `ngeom` field at all, so a
    joiner discovering an owner via `discover_owners()` cannot learn its geom
    count for display, even though the writer put it on disk.
    """
    owner = _make_owner(tmp_path, ngeom=7, capabilities=("stream_cameras", "accept_input"))
    try:
        found = [o for o in discover_owners(registry_dir=str(tmp_path))
                 if o.instance_id == "widgetbot"]
        assert len(found) == 1
        o = found[0]
        assert isinstance(o, OwnerInfo)
        # These all round-trip correctly today.
        assert o.host == "127.0.0.1"
        assert o.control == owner.control_endpoint
        assert o.bus == owner.bus_endpoint
        assert o.scene == "widgetbot"
        assert o.role == OWNER_ROLE
        assert "stream_cameras" in o.caps and "accept_input" in o.caps
        # The gap: ngeom is on disk (assert to prove the writer side is fine)
        # but OwnerInfo drops it on the floor.
        assert _read_entry(owner)["ngeom"] == 7
        assert o.ngeom == 7  # AttributeError today: OwnerInfo has no 'ngeom'
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# Ghost-pruning by pid + TTL, driven by a REAL owner-written file (not a
# hand-crafted dict) -- covers both readers' documented liveness rules.
# --------------------------------------------------------------------------- #

def test_dead_pid_owner_entry_pruned_by_pool_and_session(tmp_path):
    """A SIGKILLed owner never runs `close()`, so its file survives with a now
    -dead pid. Mutate a real owner's on-disk pid to simulate exactly that
    (closing normally would delete the file, masking what we want to test)."""
    owner = _make_owner(tmp_path, instance_id="ghost")
    path = owner._registry_path  # noqa: SLF001
    try:
        entry = _read_entry(owner)
        entry["pid"] = 4_000_000_000  # not a real pid
        with open(path, "w") as fh:
            json.dump(entry, fh)

        alive = [i for i in read_registry(str(tmp_path)) if i.instance_id == "ghost"]
        assert alive == []

        owners = [o for o in discover_owners(registry_dir=str(tmp_path))
                  if o.instance_id == "ghost"]
        assert owners == []
        assert not os.path.exists(path)  # opportunistically pruned by discover_owners
    finally:
        if os.path.exists(path):
            os.remove(path)
        owner.close()


def test_stale_mtime_owner_entry_pruned_by_pool_read_registry(tmp_path):
    """An owner with a genuinely alive pid but a heartbeat file older than the
    TTL (the process wedged, or the heartbeat thread died) is still dropped by
    the farm read -- pid liveness alone is not enough (§12: "one shared filter
    predicate... pool.read_registry keeps its liveness rules")."""
    owner = _make_owner(tmp_path, instance_id="stale")
    try:
        path = owner._registry_path  # noqa: SLF001
        old = time.time() - 500.0
        os.utime(path, (old, old))

        fresh = [i for i in read_registry(str(tmp_path), ttl_s=30.0)
                 if i.instance_id == "stale"]
        assert fresh == []

        everything = [i for i in read_registry(str(tmp_path), ttl_s=30.0, include_stale=True)
                      if i.instance_id == "stale"]
        assert len(everything) == 1
        assert everything[0].is_stale(ttl_s=30.0) is True
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# registry_written_at: int-epoch (Python) vs ISO-8601 (UE) normalization.
# --------------------------------------------------------------------------- #

def test_registry_written_at_iso8601_normalizes_in_session_not_pool(tmp_path):
    """EXPECTED FAIL (addendum §12 code-today note: "int epoch here (pool.py
    only accepts a numeric value); the readers normalize int-epoch vs the UE
    writer's ISO-8601 string"). Feed the SAME UE-shaped entry
    (`registry_written_at` as `FDateTime::UtcNow().ToIso8601()` produces) to
    both readers: `session.discover_owners` normalizes it to an epoch float
    (already covered alone by `test_session.py::test_iso8601_timestamp_
    normalized`); `pool.read_registry`/`InstanceInfo.from_registry` does not
    -- it silently drops the UE timestamp to `None`, even though both readers
    are meant to agree per the one-schema target.
    """
    entry = {
        "instance_id": "ue-owner", "index": 0, "pid": os.getpid(), "host": "myhost",
        "role": OWNER_ROLE, "capabilities": ["fastpath_owner"],
        "manager_present": True, "busy": False,
        "registry_written_at": "2026-08-20T10:00:00",  # UE's ToIso8601() shape
    }
    path = os.path.join(str(tmp_path), f"ue-owner_{os.getpid()}.json")
    with open(path, "w") as fh:
        json.dump(entry, fh)

    owner_view = [o for o in discover_owners(registry_dir=str(tmp_path))
                  if o.instance_id == "ue-owner"][0]
    assert owner_view.updated is not None and owner_view.updated > 0  # session: OK

    pool_view = [i for i in read_registry(str(tmp_path)) if i.instance_id == "ue-owner"][0]
    assert pool_view.registry_written_at is not None, (
        "pool.InstanceInfo.from_registry dropped a UE-style ISO-8601 "
        "registry_written_at to None instead of normalizing it like "
        "session.discover_owners does"
    )


# --------------------------------------------------------------------------- #
# The shared role/capability filter against a UE-shaped entry: UE's
# `capabilities` key (protocol features when not broadcasting) must not be
# mistaken for the Python grant vocabulary by `is_owner_entry`.
# --------------------------------------------------------------------------- #

def test_is_owner_entry_not_fooled_by_ue_protocol_capabilities_list(tmp_path):
    """A non-broadcasting UE instance's `capabilities` (InstanceRegistry.cpp
    `Capabilities()`) is a PROTOCOL-FEATURE list (`render_sync`, `render_async`,
    `shm_rpc`, `model_upload`, `content_cache`) -- a different vocabulary than
    the Python owner's peer-facing GRANT list under the same JSON key
    (`capability_naming_proposal.md` §4.6). The shared filter must still tell
    them apart correctly: absent `fastpath_owner` in either `role` or
    `capabilities` -> not an owner; present -> an owner, regardless of what
    else rides in that list."""
    non_broadcasting_ue_entry = {
        "instance_id": "ue-server", "pid": 123, "host": "10.0.0.2",
        "capabilities": ["render_sync", "render_async", "shm_rpc",
                         "model_upload", "content_cache"],
        "transports": ["zmq"], "grpc": None,
    }
    assert is_owner_entry(non_broadcasting_ue_entry) is False

    broadcasting_ue_entry = dict(non_broadcasting_ue_entry)
    broadcasting_ue_entry["role"] = "fastpath_owner"
    broadcasting_ue_entry["capabilities"] = non_broadcasting_ue_entry["capabilities"] + [
        "fastpath_owner"
    ]
    broadcasting_ue_entry["scene"] = "ue-scene"
    broadcasting_ue_entry["control"] = "tcp://10.0.0.2:5559"
    broadcasting_ue_entry["bus"] = "tcp://10.0.0.2:5561"
    assert is_owner_entry(broadcasting_ue_entry) is True

    path = os.path.join(str(tmp_path), "ue-server_123.json")
    with open(path, "w") as fh:
        json.dump(broadcasting_ue_entry, fh)
    owners = discover_owners(registry_dir=str(tmp_path), include_dead=True)
    assert len(owners) == 1
    o = owners[0]
    assert o.role == "fastpath_owner"
    assert "fastpath_owner" in o.caps and "render_sync" in o.caps
    assert o.scene == "ue-scene"


# --------------------------------------------------------------------------- #
# Endpoint/host parsing agreement across the ×5-duplicated parse
# (addendum §C.2) -- pool / render_pool / session must resolve the SAME
# (host, port) for the same endpoint string.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("endpoint", ["host:6000", "tcp://host:6000"])
def test_endpoint_parse_agrees_for_bare_and_tcp_forms(endpoint):
    pool_host, pool_port = _parse_endpoint(endpoint)
    rp_spec = parse_endpoints([endpoint])[0]
    owner_info = _owner_from_endpoint(endpoint)
    owner_host, _, owner_port = owner_info.grpc.rpartition(":")

    assert pool_host == rp_spec.host == owner_host == "host"
    assert pool_port == rp_spec.port == int(owner_port) == 6000


def test_endpoint_parse_agrees_for_grpc_scheme():
    """Fixed (addendum §C.2 systemic pattern: "Same concept, N copies ...
    endpoint/host parse (×5) ... No single source of truth for
    addressing/parsing/decoding"). `pool._parse_endpoint`,
    `render_pool.parse_endpoints` and `session._owner_from_endpoint` all now
    route through the shared `pool.parse_host_port` (urlparse-based, scheme
    agnostic) helper, so a "grpc://" endpoint no longer leaks its scheme into
    the parsed host: every one of the ×5 parsers resolves the same
    (host, port) for the same endpoint string, regardless of scheme.
    """
    endpoint = "grpc://host:6000"
    pool_host, pool_port = _parse_endpoint(endpoint)
    rp_spec = parse_endpoints([endpoint])[0]
    owner_info = _owner_from_endpoint(endpoint)
    owner_host, _, owner_port = owner_info.grpc.rpartition(":")

    assert pool_host == rp_spec.host == owner_host, (
        f"host disagrees across parsers for {endpoint!r}: "
        f"pool={pool_host!r} render_pool={rp_spec.host!r} session={owner_host!r}"
    )
    assert pool_port == rp_spec.port == int(owner_port) == 6000
