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

"""Unit tests for the client-side network model upload.

These need no live editor: the flatten / hash / chunk helpers are pure, and the
manifest -> chunk -> commit sequence is driven by monkeypatching ``_rpc`` with a
fake transport that records requests and returns scripted replies.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import pytest

from urlab_client import URLabClient, URLabRPCError
from urlab_client._model_upload import (
    flatten_model,
    is_bare_filename,
    iter_chunks,
    require_bare_filename,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# sha256 + chunking math
# ---------------------------------------------------------------------------


def test_sha256_hex_matches_hashlib():
    data = b"the quick brown fox"
    assert sha256_hex(data) == hashlib.sha256(data).hexdigest()


def test_chunking_10mib_at_4mib_yields_three_chunks():
    chunk_bytes = 4 * 1024 * 1024
    total = 10 * 1024 * 1024
    data = b"\x00" * total

    chunks = list(iter_chunks(data, chunk_bytes))
    assert len(chunks) == 3

    offsets = [off for off, _ in chunks]
    assert offsets == [0, 4 * 1024 * 1024, 8 * 1024 * 1024]

    sizes = [len(c) for _, c in chunks]
    assert sizes == [4 * 1024 * 1024, 4 * 1024 * 1024, 2 * 1024 * 1024]

    # Reassembly is exact and the last chunk completes the blob.
    assert sum(sizes) == total
    reassembled = b"".join(c for _, c in chunks)
    assert reassembled == data
    last_off, last_chunk = chunks[-1]
    assert last_off + len(last_chunk) == total


def test_chunking_small_blob_is_single_chunk():
    data = b"hello"
    chunks = list(iter_chunks(data, 4 * 1024 * 1024))
    assert chunks == [(0, b"hello")]


def test_chunking_empty_blob_yields_one_terminating_chunk():
    assert list(iter_chunks(b"", 1024)) == [(0, b"")]


def test_chunking_rejects_zero_chunk_bytes():
    with pytest.raises(ValueError):
        list(iter_chunks(b"abc", 0))


# ---------------------------------------------------------------------------
# bare-filename validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "../evil.obj",
        "sub/dir/mesh.obj",
        "sub\\dir\\mesh.obj",
        "/abs/path.obj",
        "\\leading.obj",
        "C:mesh.obj",
        "C:\\win\\mesh.obj",
        "..",
        ".",
        "",
    ],
)
def test_bare_filename_rejects_traversal(name):
    assert is_bare_filename(name) is False
    with pytest.raises(ValueError):
        require_bare_filename(name)


@pytest.mark.parametrize("name", ["mesh.obj", "base_link.stl", "wood.png"])
def test_bare_filename_accepts_plain_names(name):
    assert is_bare_filename(name) is True
    assert require_bare_filename(name) == name


# ---------------------------------------------------------------------------
# flatten: includes + subdir mesh path -> self-contained XML, bare refs
# ---------------------------------------------------------------------------


def _build_include_fixture(root_dir):
    """Write a small model split across an include with a subdir mesh path.

    Layout::

        root/
          scene.xml          <- includes robot.xml, has a texture in tex/
          robot.xml          <- declares a mesh under assets/
          assets/base.obj
          tex/wood.png
    """
    (root_dir / "assets").mkdir()
    (root_dir / "tex").mkdir()
    (root_dir / "assets" / "base.obj").write_text("o base\n")
    (root_dir / "tex" / "wood.png").write_bytes(b"\x89PNG\r\n")

    (root_dir / "robot.xml").write_text(
        """<mujoco model="robot">
  <asset>
    <mesh name="base" file="base.obj"/>
  </asset>
  <worldbody>
    <body name="link">
      <geom type="mesh" mesh="base"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    scene = root_dir / "scene.xml"
    scene.write_text(
        """<mujoco model="scene">
  <compiler meshdir="assets" texturedir="tex"/>
  <asset>
    <texture name="wood" type="2d" file="wood.png"/>
  </asset>
  <include file="robot.xml"/>
</mujoco>
"""
    )
    return scene


def test_flatten_resolves_includes_and_bare_refs(tmp_path):
    scene = _build_include_fixture(tmp_path)

    xml_text, asset_paths = flatten_model(str(scene))
    root = ET.fromstring(xml_text)

    # Self-contained: no <include> survives.
    assert root.find(".//include") is None
    assert "<include" not in xml_text

    # The included robot's mesh + body were spliced into the tree.
    mesh = root.find(".//mesh")
    assert mesh is not None
    assert root.find(".//body[@name='link']") is not None

    # Every file= reference is now a bare filename.
    for el in root.iter():
        fref = el.get("file")
        if fref is not None:
            assert "/" not in fref and "\\" not in fref, fref
    assert mesh.get("file") == "base.obj"
    tex = root.find(".//texture")
    assert tex.get("file") == "wood.png"

    # meshdir / texturedir stripped so bare names resolve against the VFS.
    compiler = root.find(".//compiler")
    assert compiler is not None
    assert "meshdir" not in compiler.attrib
    assert "texturedir" not in compiler.attrib

    # asset_paths maps bare names to the real on-disk source files.
    assert set(asset_paths) == {"base.obj", "wood.png"}
    assert asset_paths["base.obj"] == str(tmp_path / "assets" / "base.obj")
    assert asset_paths["wood.png"] == str(tmp_path / "tex" / "wood.png")


def test_flatten_from_string_without_assets():
    xml = '<mujoco model="m"><worldbody><body name="b"/></worldbody></mujoco>'
    xml_text, asset_paths = flatten_model(xml)
    assert asset_paths == {}
    assert "<body" in xml_text


# ---------------------------------------------------------------------------
# full manifest -> chunk -> commit sequence via a fake _rpc transport
# ---------------------------------------------------------------------------


class _FakeRpc:
    """Records every _rpc call and returns scripted replies by op name."""

    def __init__(self, replies):
        self.calls = []
        self._replies = replies

    def __call__(self, op, payload, *, expected_op=None, recv_timeout_ms=None):
        self.calls.append({"op": op, **dict(payload)})
        reply = self._replies[op]
        if callable(reply):
            reply = reply(payload)
        return dict(reply)


def _make_client():
    # step_port=0: never connects; we drive it purely through the fake _rpc.
    return URLabClient("tcp://127.0.0.1", step_mode="stepped", step_port=0)


def _commit_ok():
    return {
        "op": "upload_model_commit_ok",
        "imported": True,
        "nq": 3,
        "nv": 3,
        "nu": 1,
        "nbody": 2,
        "ngeom": 1,
        "mjb": b"\x00mjb",
        "warnings": [],
    }


def test_upload_model_full_sequence_sends_only_needed_blobs(tmp_path, monkeypatch):
    scene = _build_include_fixture(tmp_path)

    # Server needs the xml + only base.obj (wood.png is a cache hit).
    def _manifest(payload):
        return {
            "op": "upload_model_manifest_ok",
            "upload_id": "up-1",
            "need_xml": True,
            "need_assets": ["base.obj"],
            "max_asset_bytes": 16 * 1024 * 1024,
            "max_total_bytes": 64 * 1024 * 1024,
        }

    fake = _FakeRpc({
        "upload_model_manifest": _manifest,
        "upload_model_chunk": {"op": "upload_model_chunk_ok", "name": "", "received": 0, "complete": True},
        "upload_model_commit": _commit_ok(),
    })
    client = _make_client()
    monkeypatch.setattr(client, "_rpc", fake)

    result = client.upload_model(str(scene))

    ops = [c["op"] for c in fake.calls]
    assert ops[0] == "upload_model_manifest"
    assert ops[-1] == "upload_model_commit"
    assert ops.count("upload_model_commit") == 1

    # Manifest advertised both assets, hashed + sized.
    manifest_call = fake.calls[0]
    asset_names = {a["name"] for a in manifest_call["assets"]}
    assert asset_names == {"base.obj", "wood.png"}
    assert manifest_call["step_mode"] == "stepped"
    for a in manifest_call["assets"]:
        assert len(a["sha256"]) == 64
        assert a["size"] >= 0

    # Chunk sends: the xml (kind=xml) + only base.obj. wood.png was NOT sent.
    chunk_calls = [c for c in fake.calls if c["op"] == "upload_model_chunk"]
    kinds = [(c["kind"], c["name"]) for c in chunk_calls]
    assert ("xml", "model.xml") in kinds
    assert ("asset", "base.obj") in kinds
    assert not any(name == "wood.png" for _, name in kinds)

    # Each chunk carries offset/total/sha and the payload is real bytes.
    for c in chunk_calls:
        assert isinstance(c["data"], bytes)
        assert c["total"] >= len(c["data"])
        assert len(c["sha256"]) == 64

    assert result["imported"] is True
    assert result["mjb"] == b"\x00mjb"


def test_upload_model_cache_hit_sends_zero_asset_chunks(tmp_path, monkeypatch):
    scene = _build_include_fixture(tmp_path)

    def _manifest(payload):
        # Everything already cached: xml present, no assets needed.
        return {
            "op": "upload_model_manifest_ok",
            "upload_id": "up-2",
            "need_xml": False,
            "need_assets": [],
        }

    fake = _FakeRpc({
        "upload_model_manifest": _manifest,
        "upload_model_chunk": {"op": "upload_model_chunk_ok", "name": "", "received": 0, "complete": True},
        "upload_model_commit": _commit_ok(),
    })
    client = _make_client()
    monkeypatch.setattr(client, "_rpc", fake)

    client.upload_model(str(scene))

    # A full cache hit sends no chunks at all: manifest then commit.
    ops = [c["op"] for c in fake.calls]
    assert ops == ["upload_model_manifest", "upload_model_commit"]


def test_upload_model_chunk_offsets_are_correct(tmp_path, monkeypatch):
    # A large synthetic asset forces multi-chunk streaming; assert offsets.
    big = b"\x01" * (10 * 1024 * 1024)
    xml = '<mujoco model="m"><worldbody/></mujoco>'

    def _manifest(payload):
        return {
            "op": "upload_model_manifest_ok",
            "upload_id": "up-3",
            "need_xml": True,
            "need_assets": ["big.bin"],
        }

    fake = _FakeRpc({
        "upload_model_manifest": _manifest,
        "upload_model_chunk": {"op": "upload_model_chunk_ok", "name": "", "received": 0, "complete": True},
        "upload_model_commit": _commit_ok(),
    })
    client = _make_client()
    monkeypatch.setattr(client, "_rpc", fake)

    client.upload_model(xml, assets={"big.bin": big}, chunk_bytes=4 * 1024 * 1024)

    asset_chunks = [
        c for c in fake.calls
        if c["op"] == "upload_model_chunk" and c["name"] == "big.bin"
    ]
    assert [c["offset"] for c in asset_chunks] == [0, 4 * 1024 * 1024, 8 * 1024 * 1024]
    assert all(c["total"] == len(big) for c in asset_chunks)
    # Reassembling the streamed chunks reproduces the blob exactly.
    assert b"".join(c["data"] for c in asset_chunks) == big
    # The blob's advertised sha matches its content.
    assert asset_chunks[0]["sha256"] == sha256_hex(big)


def test_upload_model_rejects_non_bare_asset_key(monkeypatch):
    xml = '<mujoco model="m"><worldbody/></mujoco>'
    client = _make_client()
    # Should fail during client-side validation, before any _rpc fires.
    monkeypatch.setattr(
        client, "_rpc",
        lambda *a, **k: pytest.fail("_rpc must not be called on invalid input"),
    )
    with pytest.raises(ValueError):
        client.upload_model(xml, assets={"../escape.obj": b"x"})


def test_upload_model_rejects_oversize_total(tmp_path, monkeypatch):
    xml = '<mujoco model="m"><worldbody/></mujoco>'

    def _manifest(payload):
        return {
            "op": "upload_model_manifest_ok",
            "upload_id": "up-4",
            "need_xml": True,
            "need_assets": ["blob.bin"],
            "max_total_bytes": 1024,  # tiny cap
        }

    fake = _FakeRpc({
        "upload_model_manifest": _manifest,
        "upload_model_chunk": {"op": "upload_model_chunk_ok"},
        "upload_model_commit": _commit_ok(),
    })
    client = _make_client()
    monkeypatch.setattr(client, "_rpc", fake)

    with pytest.raises(ValueError):
        client.upload_model(xml, assets={"blob.bin": b"y" * 4096})
    # No chunk should have been streamed once the size check failed.
    assert not any(c["op"] == "upload_model_chunk" for c in fake.calls)


def test_upload_model_commit_error_raises(tmp_path, monkeypatch):
    xml = '<mujoco model="m"><worldbody/></mujoco>'

    def _manifest(payload):
        return {
            "op": "upload_model_manifest_ok",
            "upload_id": "up-5",
            "need_xml": True,
            "need_assets": [],
        }

    def _commit_err(payload):
        raise URLabRPCError("import_failed", "geom size must be positive", op="upload_model_commit")

    fake = _FakeRpc({
        "upload_model_manifest": _manifest,
        "upload_model_chunk": {"op": "upload_model_chunk_ok"},
        "upload_model_commit": _commit_err,
    })
    client = _make_client()
    monkeypatch.setattr(client, "_rpc", fake)

    with pytest.raises(URLabRPCError) as exc_info:
        client.upload_model(xml)
    assert exc_info.value.code == "import_failed"
