# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Model-transport tests for the features built since v0.6.0-beta.

Grounded in `audit_docs/0_render_source_of_truth.md` §11 ("Model transport"):
three formats -- mjb (version-locked, `mj_loadModelBuffer` direct), xml+assets
and mjz (both version-independent, compiled in-engine from an `mjVFS`) -- all
normalized to an `mjModel` on the receiver, behind the ONE `fastpath_hello`
reply schema on every transport face.

`test_wire_schema_features.py::test_hello_schema_matches_across_zmq_and_grpc_faces`
already proves the ZMQ-REP and gRPC dispatch layers emit byte-identical hello
replies (mechanism coverage). This file adds the piece that was still
untested: that the CONTENT surviving that pipe is correct -- a real
xml+assets scene (MuJoCo's own `bunny.xml` flex fixture, not a placeholder
string) flattened by `_model_upload.flatten_model` recompiles to the same
topology after a full client-flatten -> owner-hello -> receiver-reconstruct
round trip, and that the mjb path carries bytes through unmodified.

Hermetic: real mujoco compiles, no live UE, no network (owner sockets bind to
OS-assigned loopback ports but nothing external connects to them).
"""
from __future__ import annotations

import base64
import os
import socket

import pytest

mujoco = pytest.importorskip("mujoco")

from urlab_client._model_upload import flatten_model  # noqa: E402
from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402

# The MuJoCo upstream flex fixture named in the task brief. Located relative
# to this test file (repo root -> sibling UnrealRoboticsLab checkout) rather
# than hardcoded to a homedir, so the test degrades to a clean skip on a
# machine with a different checkout layout instead of silently passing
# nothing or hard-failing on an unrelated path -- the exact hardcoded-fixture
# anti-pattern `cleanup_audit_addendum.md` §B flags in
# `MjRendererTests.cpp`/`MjAppearanceTests.cpp` ("skip on any other machine").
_BUNNY_XML = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "UnrealRoboticsLab", "third_party", "MuJoCo", "src", "model", "flex", "bunny.xml",
))


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


def _write_flattened(tmp_path, xml_text, asset_paths_or_bytes, *, from_disk: bool) -> str:
    """Materialise a flattened model + its bare-named assets under ``tmp_path``
    exactly as a receiver (server or renderer) would from the wire, and return
    the model.xml path."""
    for bare, blob in asset_paths_or_bytes.items():
        data = open(blob, "rb").read() if from_disk else blob
        (tmp_path / bare).write_bytes(data)
    model_path = tmp_path / "model.xml"
    model_path.write_text(xml_text)
    return str(model_path)


# --------------------------------------------------------------------------- #
# xml + assets: flatten_model on a real flex scene recompiles identically
# (source-of-truth §11: "xml + assets" -> mjSpec/mjVFS -> mj_compile in-engine,
# version-independent).
# --------------------------------------------------------------------------- #

def test_flatten_model_bunny_recompiles_same_topology(tmp_path):
    if not os.path.isfile(_BUNNY_XML):
        pytest.skip(f"bunny.xml fixture not found at {_BUNNY_XML!r} "
                    "(expected sibling UnrealRoboticsLab checkout)")

    original = mujoco.MjModel.from_xml_path(_BUNNY_XML)

    xml_text, asset_paths = flatten_model(_BUNNY_XML)

    # bunny.xml's only file= reference is the flexcomp mesh (<flexcomp file=
    # "bunny.obj">, not a <mesh>/<texture> tag) -- confirms flatten_model's
    # generic `el.get("file")` walk covers flexcomp too, not just mesh/texture.
    assert set(asset_paths) == {"bunny.obj"}
    assert asset_paths["bunny.obj"].endswith(os.path.join("asset", "bunny.obj"))

    recompile_dir = tmp_path / "recompiled"
    recompile_dir.mkdir()
    model_path = _write_flattened(recompile_dir, xml_text, asset_paths, from_disk=True)
    recompiled = mujoco.MjModel.from_xml_path(model_path)

    # The intended contract: normalizes to the SAME mjModel topology.
    assert recompiled.nbody == original.nbody
    assert recompiled.nflex == original.nflex
    assert recompiled.ngeom == original.ngeom
    assert recompiled.nq == original.nq

    # The <compiler> asset dirs are meaningless after flattening (bare names
    # resolve directly against the receiver's directory / VFS).
    assert "meshdir" not in xml_text and "texturedir" not in xml_text


# --------------------------------------------------------------------------- #
# mjb: version-locked, ships as opaque bytes through mj_loadModelBuffer with
# no in-engine recompilation -- the hello reply must carry them unmodified.
# --------------------------------------------------------------------------- #

def test_mjb_format_round_trips_through_owner_hello(tmp_path):
    mjcf = """
    <mujoco model="mjb_roundtrip">
      <worldbody>
        <body name="link" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator><motor name="m" joint="hinge" gear="1"/></actuator>
    </mujoco>
    """
    original = mujoco.MjModel.from_xml_string(mjcf)
    mjb_path = tmp_path / "orig.mjb"
    mujoco.mj_saveModel(original, str(mjb_path))
    mjb_bytes = mjb_path.read_bytes()

    owner = _make_owner(
        tmp_path, mjb_bytes=mjb_bytes, model_format="mjb", ngeom=original.ngeom,
    )
    try:
        reply = owner.hello_reply()
        assert reply["model_format"] == "mjb"
        # Bytes travel unmodified -- no re-encoding, no recompilation.
        assert reply["mjb"] == mjb_bytes

        reload_path = tmp_path / "reloaded.mjb"
        reload_path.write_bytes(reply["mjb"])
        reloaded = mujoco.MjModel.from_binary_path(str(reload_path))

        assert reloaded.nbody == original.nbody
        assert reloaded.ngeom == original.ngeom
        assert reloaded.nq == original.nq
        assert reloaded.nu == original.nu
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# End-to-end: a real flattened xml+assets scene served through the owner's
# hello reply, reconstructed by a receiver exactly as a renderer would
# (xml text + base64 vfs_assets under bare names), recompiles identically.
# --------------------------------------------------------------------------- #

def test_owner_hello_serves_flattened_bunny_end_to_end(tmp_path):
    if not os.path.isfile(_BUNNY_XML):
        pytest.skip(f"bunny.xml fixture not found at {_BUNNY_XML!r} "
                    "(expected sibling UnrealRoboticsLab checkout)")

    original = mujoco.MjModel.from_xml_path(_BUNNY_XML)
    xml_text, asset_paths = flatten_model(_BUNNY_XML)
    assets_bytes = {name: open(src, "rb").read() for name, src in asset_paths.items()}

    owner = _make_owner(
        tmp_path, mjb_bytes=xml_text.encode("utf-8"), model_format="xml",
        assets=assets_bytes, ngeom=original.ngeom, scene="bunny",
    )
    try:
        reply = owner.hello_reply()
        assert reply["model_format"] == "xml"
        assert reply["scene"] == "bunny"
        assert reply["xml"] == xml_text
        assert set(reply["vfs_assets"]) == {"bunny.obj__b64__"}

        # Reconstruct on the "receiver" side straight from the wire payload --
        # xml text + each vfs_asset base64-decoded under its bare name -- the
        # same shape MjRendererDriverClient::FetchModel consumes.
        receiver_dir = tmp_path / "receiver"
        receiver_dir.mkdir()
        for key, b64 in reply["vfs_assets"].items():
            bare = key[: -len("__b64__")]
            (receiver_dir / bare).write_bytes(base64.b64decode(b64))
        model_path = receiver_dir / "model.xml"
        model_path.write_text(reply["xml"])

        recompiled = mujoco.MjModel.from_xml_path(str(model_path))
        assert recompiled.nbody == original.nbody
        assert recompiled.nflex == original.nflex
        assert recompiled.ngeom == original.ngeom
    finally:
        owner.close()
