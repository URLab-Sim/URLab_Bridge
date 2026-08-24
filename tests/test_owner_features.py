# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""FastPathOwner feature tests: hello schema, model hot-swap, the transform
cache a gRPC subscribe reads, registry bookkeeping, and the gRPC endpoint
lifecycle. Grounded in `audit_docs/0_render_source_of_truth.md` §5 (five
capabilities) and §11 (one hello schema on every transport face), and
`audit_docs/cleanup_audit_addendum.md` §B ("Bridge owner / farm / examples":
`start_grpc_server` returns a non-dialable `0.0.0.0:port`).

Perturbation-specific behaviour lives in `test_perturb_features.py`;
capability-advertisement behaviour lives in `test_capabilities_features.py`.
No GUI, no UE.
"""
from __future__ import annotations

import json
import os
import socket

import pytest

mujoco = pytest.importorskip("mujoco")

from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_owner(tmp_path, **kw):
    kw.setdefault("scene", "t")
    kw.setdefault("control_port", _free_port())
    kw.setdefault("bus_port", _free_port())
    kw.setdefault("advertise_host", "127.0.0.1")
    kw.setdefault("registry_dir", str(tmp_path))
    return FastPathOwner(kw.pop("mjb_bytes", b""), **kw)


def _read_registry(owner) -> dict:
    with open(owner._registry_path) as fh:  # noqa: SLF001 (test introspection)
        return json.load(fh)


# --------------------------------------------------------------------------- #
# hello schema (source-of-truth §11): one schema, model_format selects the
# shape -- mjb ships compiled bytes, xml/mjz ship source + a base64 asset map.
# --------------------------------------------------------------------------- #

def test_hello_reply_mjb_schema(tmp_path):
    owner = _make_owner(tmp_path, mjb_bytes=b"FAKEMJB", model_format="mjb", ngeom=5)
    try:
        reply = owner.hello_reply()
        assert reply["ok"] is True
        assert reply["scene"] == "t"
        assert reply["ngeom"] == 5
        assert reply["model_format"] == "mjb"
        assert reply["mjb"] == b"FAKEMJB"
        assert reply["bus"] == owner.bus_endpoint
        # generic aliases kept for non-UE consumers
        assert reply["model"] == b"FAKEMJB" and reply["format"] == "mjb"
        assert "xml" not in reply and "vfs_assets" not in reply
    finally:
        owner.close()


def test_hello_reply_xml_schema_carries_assets(tmp_path):
    xml_bytes = b"<mujoco/>"
    assets = {"cube.obj": b"OBJDATA"}
    owner = _make_owner(
        tmp_path, mjb_bytes=xml_bytes, model_format="xml", assets=assets, ngeom=2,
    )
    try:
        reply = owner.hello_reply()
        assert reply["model_format"] == "xml"
        assert reply["xml"] == "<mujoco/>"
        assert "mjb" not in reply
        assert reply["vfs_assets"] == {
            "cube.obj__b64__": "T0JKREFUQQ=="  # base64("OBJDATA")
        }
    finally:
        owner.close()


def test_hello_reply_bus_override_for_grpc_face(tmp_path):
    """The gRPC face passes an explicit ``bus=`` so a subscriber dials the gRPC
    endpoint instead of the ZMQ PUB (owner_server.py `_dispatch`)."""
    owner = _make_owner(tmp_path)
    try:
        reply = owner.hello_reply(bus="grpc://127.0.0.1:50051")
        assert reply["bus"] == "grpc://127.0.0.1:50051"
        assert reply["bus"] != owner.bus_endpoint
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# model hot-swap (update_model)
# --------------------------------------------------------------------------- #

def test_update_model_swaps_bytes_and_refreshes_registry(tmp_path):
    owner = _make_owner(tmp_path, mjb_bytes=b"OLD", ngeom=1)
    try:
        before = _read_registry(owner)
        assert before["ngeom"] == 1
        owner.update_model(b"NEW", ngeom=9)
        assert owner.model_bytes == b"NEW"
        assert owner.ngeom == 9
        after = _read_registry(owner)
        assert after["ngeom"] == 9
        # A renderer that discovers the owner AFTER the swap pulls the CURRENT
        # scene, not the one it booted on.
        assert owner.hello_reply()["mjb"] == b"NEW"
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# transform cache -- the payload a gRPC subscribe(format=render) stream reads
# (fastpath_owner.py `latest_transforms`, consumed by owner_server.py
# `_stream_render`).
# --------------------------------------------------------------------------- #

def test_publish_bodies_populates_latest_transforms_cache(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        assert owner.latest_transforms() is None
        owner.publish_bodies(3, bxpos=[1.0, 2.0, 3.0], bxquat=[1.0, 0.0, 0.0, 0.0])
        frame = owner.latest_transforms()
        assert frame is not None
        assert frame["f"] == 3
        assert frame["bxpos"] == [1.0, 2.0, 3.0]
        assert frame["bxquat"] == [1.0, 0.0, 0.0, 0.0]
    finally:
        owner.close()


def test_latest_transforms_returns_an_independent_copy(tmp_path):
    """`latest_transforms` docs promise a copy (thread-safe) -- mutating the
    returned dict must not corrupt the owner's cached frame."""
    owner = _make_owner(tmp_path)
    try:
        owner.publish_bodies(1, bxpos=[0.0, 0.0, 0.0], bxquat=[1.0, 0.0, 0.0, 0.0])
        snap = owner.latest_transforms()
        snap["bxpos"] = [999.0, 999.0, 999.0]
        fresh = owner.latest_transforms()
        assert fresh["bxpos"] == [0.0, 0.0, 0.0]
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# registry bookkeeping
# --------------------------------------------------------------------------- #

def test_registry_reflects_zmq_only_before_grpc(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        entry = _read_registry(owner)
        assert entry["transports"] == ["zmq"]
        assert entry["grpc"] is None
        assert entry["role"] == "fastpath_owner"
        assert entry["manager_present"] is True
    finally:
        owner.close()


def test_registry_gains_grpc_after_start_grpc_server(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        owner.start_grpc_server(port=_free_port())
        entry = _read_registry(owner)
        assert "grpc" in entry["transports"]
        assert entry["grpc"] == owner.grpc_endpoint
    finally:
        owner.close()


def test_close_removes_registry_file(tmp_path):
    owner = _make_owner(tmp_path)
    path = owner._registry_path  # noqa: SLF001
    assert os.path.exists(path)
    owner.close()
    assert not os.path.exists(path)


# --------------------------------------------------------------------------- #
# gRPC endpoint lifecycle
# --------------------------------------------------------------------------- #

def test_start_grpc_server_is_idempotent(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        port = _free_port()
        ep1 = owner.start_grpc_server(port=port)
        ep2 = owner.start_grpc_server(port=port)
        assert ep1 == ep2
        assert owner.grpc_endpoint == f"127.0.0.1:{port}"
    finally:
        owner.close()


def test_grpc_endpoint_property_is_dialable(tmp_path):
    """Contrast case for the bug below: the `.grpc_endpoint` PROPERTY (used to
    populate the registry and the `bus=` hello override) already advertises the
    resolved ``advertise_host``, not the bind address."""
    owner = _make_owner(tmp_path, advertise_host="10.0.0.5")
    try:
        port = _free_port()
        owner.start_grpc_server(port=port, bind="0.0.0.0")
        assert owner.grpc_endpoint == f"10.0.0.5:{port}"
        host = owner.grpc_endpoint.rsplit(":", 1)[0]
        assert host != "0.0.0.0"
    finally:
        owner.close()


def test_start_grpc_server_return_value_is_dialable(tmp_path):
    """EXPECTED FAIL (addendum §B, Bridge owner/farm/examples): 'start_grpc_server
    returns 0.0.0.0:port (non-dialable) while registry/property advertise
    host:port.' `OwnerGrpcServer.endpoint` is built from `bind` (default
    "0.0.0.0"), not from the owner's resolved `advertise_host`, so a caller that
    dials the RETURN VALUE of start_grpc_server() (rather than reading the
    `.grpc_endpoint` property afterwards) gets an address nothing outside this
    host can connect to. Intended behaviour: the returned endpoint must be the
    same dialable host:port the registry/property advertise.
    """
    owner = _make_owner(tmp_path, advertise_host="10.0.0.5")
    try:
        port = _free_port()
        returned = owner.start_grpc_server(port=port, bind="0.0.0.0")
        host = returned.rsplit(":", 1)[0]
        assert host != "0.0.0.0", (
            f"start_grpc_server() returned a non-dialable endpoint {returned!r}"
        )
        assert returned == owner.grpc_endpoint
    finally:
        owner.close()
