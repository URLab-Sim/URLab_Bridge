# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Capability-advertisement tests.

Grounded in `audit_docs/0_render_source_of_truth.md` §5: capabilities are
composable authorizations advertised in the handshake, each gating one thing.
A Python `FastPathOwner` advertises the two consumer-facing capabilities UE's
`EMjCapability` shares a vocabulary with -- `stream_cameras` (may subscribe to
this owner's view) and `accept_input` (may push perturbations back) -- plus
the fixed `fastpath_owner` role tag (`fastpath_owner.py:42-58`).

These tests pin the invariant that every surface an owner advertises through
(`.capabilities`, `hello_reply()["capabilities"]`, the registry entry, and a
viewer's `discover_owners()` read of that registry) agrees on EXACTLY the
capability set the owner was constructed with -- no more, no less, aliases
canonicalized. Also covers the `mujoco_version` handshake field on the
general session path (`client.py`/`enums.py`), using the existing
`conftest.py` fixtures (not modified here).

Perturbation ACCEPTANCE behaviour (the consumer side of `accept_input`) is
covered in `test_perturb_features.py`; owner networking/hello-schema
mechanics live in `test_owner_features.py`.
"""
from __future__ import annotations

import json
import socket

import pytest

mujoco = pytest.importorskip("mujoco")

from urlab_client.fastpath_owner import (  # noqa: E402
    CAP_ACCEPT_INPUT,
    CAP_STREAM_CAMERAS,
    FASTPATH_OWNER_CAP,
    FastPathOwner,
)
from urlab_client.session import discover_owners  # noqa: E402


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
    return FastPathOwner(b"", **kw)


# --------------------------------------------------------------------------- #
# advertised caps == granted caps, across every surface.
# --------------------------------------------------------------------------- #

def test_default_grants_both_capabilities(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        expected = {FASTPATH_OWNER_CAP, CAP_STREAM_CAMERAS, CAP_ACCEPT_INPUT}
        assert set(owner.capabilities) == expected
        assert owner.accepts_input is True
        assert owner.streams_view is True
    finally:
        owner.close()


def test_view_only_grant_omits_accept_input_everywhere(tmp_path):
    """Constructed with just `stream_cameras`: `accept_input` must be absent
    from EVERY surface that advertises capabilities, not just the property."""
    owner = _make_owner(tmp_path, capabilities=(CAP_STREAM_CAMERAS,))
    try:
        expected = {FASTPATH_OWNER_CAP, CAP_STREAM_CAMERAS}
        assert set(owner.capabilities) == expected
        assert owner.accepts_input is False
        assert owner.streams_view is True

        hello_caps = set(owner.hello_reply()["capabilities"])
        assert hello_caps == expected

        with open(owner._registry_path) as fh:  # noqa: SLF001
            registry_caps = set(json.load(fh)["capabilities"])
        assert registry_caps == expected
    finally:
        owner.close()


def test_input_only_grant_omits_stream_cameras_everywhere(tmp_path):
    owner = _make_owner(tmp_path, capabilities=(CAP_ACCEPT_INPUT,))
    try:
        expected = {FASTPATH_OWNER_CAP, CAP_ACCEPT_INPUT}
        assert set(owner.capabilities) == expected
        assert owner.accepts_input is True
        assert owner.streams_view is False
        assert set(owner.hello_reply()["capabilities"]) == expected
    finally:
        owner.close()


def test_no_extra_capabilities_are_advertised(tmp_path):
    """The advertised set is never a SUPERSET of what was granted (plus the
    fixed role tag) -- e.g. requesting only accept_input must not silently
    also grant stream_cameras."""
    owner = _make_owner(tmp_path, capabilities=(CAP_ACCEPT_INPUT,))
    try:
        assert CAP_STREAM_CAMERAS not in owner.capabilities
        assert CAP_STREAM_CAMERAS not in owner.hello_reply()["capabilities"]
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# alias canonicalization: legacy/enum-style spellings resolve to the ONE
# UE-shared wire vocabulary; the raw alias never leaks onto the wire.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("alias,canonical", [
    ("view", CAP_STREAM_CAMERAS),
    ("streamcameras", CAP_STREAM_CAMERAS),
    ("acceptinput", CAP_ACCEPT_INPUT),
])
def test_capability_aliases_canonicalize_on_the_wire(tmp_path, alias, canonical):
    owner = _make_owner(tmp_path, capabilities=(alias,))
    try:
        assert canonical in owner.capabilities
        assert alias not in owner.capabilities
        hello_caps = owner.hello_reply()["capabilities"]
        assert canonical in hello_caps
        assert alias not in hello_caps
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# a viewer's discover_owners() reads back EXACTLY the granted set through the
# real registry file (not a hand-written fixture dict -- an actual owner
# instance wrote this).
# --------------------------------------------------------------------------- #

def test_discover_owners_round_trips_granted_capabilities(tmp_path):
    owner = _make_owner(tmp_path, instance_id="capcheck",
                         capabilities=(CAP_STREAM_CAMERAS,))
    try:
        found = [o for o in discover_owners(registry_dir=str(tmp_path))
                 if o.instance_id == "capcheck"]
        assert len(found) == 1
        assert set(found[0].caps) == set(owner.capabilities)
        assert CAP_ACCEPT_INPUT not in found[0].caps
    finally:
        owner.close()


def test_discover_owners_round_trips_after_grpc_upgrade(tmp_path):
    """Granting nothing new -- just starting the gRPC face -- must not change
    the advertised CAPABILITY set (only `transports`/`grpc` grow)."""
    owner = _make_owner(tmp_path, instance_id="capcheck2")
    try:
        before = set(owner.capabilities)
        owner.start_grpc_server(port=_free_port())
        found = [o for o in discover_owners(registry_dir=str(tmp_path))
                 if o.instance_id == "capcheck2"][0]
        assert set(found.caps) == before
        assert "grpc" in found.transports
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# handshake fields: mujoco_version is part of the session handshake schema
# (client.py `_apply_handshake`) and must carry the ACTUAL installed
# mujoco version in the fixture the whole suite relies on -- not just survive
# a round-trip (already covered by test_handshake.py).
# --------------------------------------------------------------------------- #

def test_base_handshake_fixture_carries_real_mujoco_version(mujoco_mod, base_handshake):
    assert "mujoco_version" in base_handshake
    assert base_handshake["mujoco_version"] == mujoco_mod.__version__


def test_fastpath_hello_reply_always_carries_capabilities_field(tmp_path):
    """`capabilities` is a required field of the ONE fastpath_hello schema
    (source-of-truth §11) on every model_format -- mjb and xml/mjz alike."""
    for model_format in ("mjb", "xml"):
        owner = _make_owner(tmp_path, model_format=model_format,
                             instance_id=f"hello-{model_format}")
        try:
            reply = owner.hello_reply()
            assert "capabilities" in reply
            assert set(reply["capabilities"]) == set(owner.capabilities)
        finally:
            owner.close()
