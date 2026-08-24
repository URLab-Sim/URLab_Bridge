# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Wire-schema features built since v0.6.0-beta (source-of-truth §8, §8.1, §11).

Covers:
  * The render tier carries per-body transforms ONLY -- qpos never rides the
    render bus (§8.1 / §8.4: "qpos never travels the render bus" / "the qpos
    render tier is REMOVED from the wire"). Unit test on the frame builder +
    a loopback owner -> ZMQ-subscriber integration.
  * `fastpath_hello` is the ONE reply schema on every transport face (§11) --
    the ZMQ REP face and the gRPC face must agree on every field except the
    `bus` scheme, which legitimately differs per face.
  * The `subscribe(format=render)` request grammar
    (`{format, contacts, overlay, maxcontacts}`, §8.2) round-trips through
    `FastPathOwner.parse_render_debug_caps` / `set_render_debug_caps`.

Hermetic: real ZMQ/gRPC sockets on loopback (127.0.0.1) with OS-assigned free
ports, real mujoco models, no external processes.
"""
from __future__ import annotations

import socket
import time

import msgpack
import pytest

mujoco = pytest.importorskip("mujoco")
zmq = pytest.importorskip("zmq")
grpc = pytest.importorskip("grpc")

from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402
from urlab_client.transports._dmenv import (  # noqa: E402
    dm_env_rpc_pb2,
    dm_env_rpc_pb2_grpc,
    urlab_dm_env_rpc_pb2 as upb,
)

# A scene with more than one body and free joints, so `data.qpos` is
# non-trivial (7 dof per free body) and clearly distinguishable from the
# per-body bxpos/bxquat transforms the render tier carries.
_SCENE_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="boxA" pos="0 0 0.5"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
    <body name="arm" pos="2 0 0.5">
      <joint name="h" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0.1 0 0"/>
    </body>
  </worldbody>
  <actuator><motor name="m" joint="h" gear="1"/></actuator>
</mujoco>
"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    data = mujoco.MjData(model)
    # Nudge qpos away from a trivial all-zero vector so "qpos absent" isn't
    # vacuously true because qpos happens to be zero.
    data.qpos[:] = [float(i) * 0.1 + 1.0 for i in range(model.nq)]
    mujoco.mj_forward(model, data)
    assert model.nq >= 8, "scene needs a real qpos vector (free + hinge)"
    return model, data


def _make_owner(tmp_path, **kw):
    kw.setdefault("scene", "wire_schema_test")
    kw.setdefault("control_port", _free_port())
    kw.setdefault("bus_port", _free_port())
    kw.setdefault("advertise_host", "127.0.0.1")
    kw.setdefault("registry_dir", str(tmp_path))
    return FastPathOwner(b"MODELBYTES", model_format="xml", **kw)


# --------------------------------------------------------------------------- #
# Render tier: transforms only, never qpos.
# --------------------------------------------------------------------------- #

def test_frame_builder_never_emits_qpos_key(scene):
    """Unit test on the frame builder: `_build_render_frame` only ever adds
    camera/user-camera keys on top of whatever transform payload it is given;
    it must never itself introduce a `qpos`/`qvel` key (source-of-truth §8.1:
    "One render bus, transforms only ... qpos never travels the render bus")."""
    owner = object.__new__(FastPathOwner)
    model, data = scene

    payload = {"f": 3, "bxpos": list(data.xpos.ravel()), "bxquat": list(data.xquat.ravel())}
    frame = owner._build_render_frame(payload, cxpos=None, cxquat=None, usercam=None)
    assert "qpos" not in frame and "qvel" not in frame
    assert set(frame) == {"f", "bxpos", "bxquat"}


def test_publish_mjdata_frame_carries_transforms_not_qpos(tmp_path, scene):
    """Even when publish_mjdata is handed a `data` whose qpos is fully
    populated (this fixture's qpos is non-zero), the assembled + cached render
    frame must carry only per-body transforms -- never qpos/qvel -- because
    the render bus is transforms-only (§8.1) even for the richest ("overlay")
    debug tier (§8.2), which explicitly excludes qpos too."""
    model, data = scene
    owner = _make_owner(tmp_path)
    try:
        # Ask for the full debug tier too -- if qpos were going to leak onto
        # the wire anywhere, the richest subscription is where it would show.
        owner.set_render_debug_caps({"contacts": True, "overlay": True, "maxcontacts": 0})
        owner.publish_mjdata(5, model, data)
        frame = owner.latest_transforms()
        assert frame is not None
        assert "qpos" not in frame and "qvel" not in frame
        assert "bxpos" in frame and "bxquat" in frame
        assert len(frame["bxpos"]) == 3 * model.nbody
        assert len(frame["bxquat"]) == 4 * model.nbody
    finally:
        owner.close()


def test_loopback_owner_to_zmq_subscriber_never_carries_qpos(tmp_path, scene):
    """Integration: a real ZMQ SUB subscribing the owner's `render` topic --
    exactly what a UE render-server mirror does -- must never see qpos on the
    wire, even for a frame published from live mjData with a populated qpos
    and the full debug tier requested."""
    model, data = scene
    owner = _make_owner(tmp_path)
    try:
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"render")
        sub.setsockopt(zmq.RCVTIMEO, 2000)
        sub.connect(owner.bus_endpoint)
        time.sleep(0.2)  # let the SUB connection + subscription land (slow joiner)

        owner.set_render_debug_caps({"contacts": True, "overlay": True, "maxcontacts": 0})
        deadline = time.time() + 2.0
        frame = None
        while time.time() < deadline:
            owner.publish_mjdata(9, model, data)
            try:
                topic, payload = sub.recv_multipart()
            except zmq.Again:
                continue
            frame = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            break
        sub.close(0)
        assert frame is not None, "never received a frame on the render topic"
        assert "qpos" not in frame and "qvel" not in frame
        assert "bxpos" in frame
        assert len(frame["bxpos"]) == 3 * model.nbody
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# Hello: one schema on every transport face (§11), already fixed -- assert it.
# --------------------------------------------------------------------------- #

def _zmq_hello(owner) -> dict:
    req = zmq.Context.instance().socket(zmq.REQ)
    # Short RCVTIMEO + a serve_pending()/recv() retry loop (matching
    # test_owner_stream.py's pattern): a long RCVTIMEO would block this
    # single-threaded poll-then-recv loop for the whole timeout on the very
    # first pass whenever the request hasn't landed by the first poll,
    # starving serve_pending() of the chance to answer it in time.
    req.setsockopt(zmq.RCVTIMEO, 200)
    req.setsockopt(zmq.SNDTIMEO, 500)
    req.connect(owner.control_endpoint)
    req.send(msgpack.packb({"op": "fastpath_hello"}, use_bin_type=True))
    rep = None
    for _ in range(100):
        owner.serve_pending()
        try:
            rep = msgpack.unpackb(req.recv(), raw=False, strict_map_key=False)
            break
        except zmq.error.Again:
            time.sleep(0.01)
    req.close(0)
    assert rep is not None, "no ZMQ fastpath_hello reply"
    return rep


def _grpc_hello(port: int) -> dict:
    def sub_pkt(op, req):
        p = upb.UrlabPacket(op=op, payload=msgpack.packb(req, use_bin_type=True), sequence_id=1)
        e = dm_env_rpc_pb2.EnvironmentRequest()
        e.extension.Pack(p)
        return e

    stub = dm_env_rpc_pb2_grpc.EnvironmentStub(grpc.insecure_channel(f"127.0.0.1:{port}"))
    for r in stub.Process(iter([sub_pkt("fastpath_hello", {})])):
        out = upb.UrlabPacket()
        r.extension.Unpack(out)
        return msgpack.unpackb(bytes(out.payload), raw=False, strict_map_key=False)
    raise AssertionError("no gRPC fastpath_hello reply")


def test_hello_schema_matches_across_zmq_and_grpc_faces(tmp_path):
    """`fastpath_hello` must be the ONE reply schema on every transport face
    (source-of-truth §11): every field is byte-for-byte identical whether
    fetched over the ZMQ control REP or the gRPC `Process` stream, except
    `bus`, which legitimately carries a different URI scheme per face
    (tcp:// for ZMQ, grpc:// for gRPC pointing at the same owner)."""
    grpc_port = _free_port()
    owner = _make_owner(tmp_path)
    try:
        owner.start_grpc_server(port=grpc_port)

        zmq_reply = _zmq_hello(owner)
        grpc_reply = _grpc_hello(grpc_port)

        # `bus` differs by design (scheme selects the transport, §9.1).
        assert zmq_reply["bus"].startswith("tcp://")
        assert grpc_reply["bus"].startswith("grpc://")

        for key in ("ok", "scene", "ngeom", "model_format", "format", "model"):
            assert zmq_reply[key] == grpc_reply[key], f"hello field {key!r} diverges across faces"
        assert sorted(zmq_reply["capabilities"]) == sorted(grpc_reply["capabilities"])
        assert zmq_reply.get("xml") == grpc_reply.get("xml")
        assert zmq_reply.get("vfs_assets") == grpc_reply.get("vfs_assets")
    finally:
        owner.close()


def test_hello_schema_has_no_qpos_render_tier_fields(tmp_path):
    """The gRPC `subscribe_viewer` / `subscribe(format=qpos)` alias was
    deleted (§8.4); `fastpath_hello` never advertises a qpos-tier bus field or
    schema marker."""
    owner = _make_owner(tmp_path)
    try:
        reply = owner.hello_reply()
        assert "qpos" not in reply
        assert reply.get("format") != "qpos" and reply.get("model_format") != "qpos"
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# Subscribe payload grammar: {format, contacts, overlay, maxcontacts}.
# --------------------------------------------------------------------------- #

def test_subscribe_grammar_round_trips_through_parse_and_set():
    """The `subscribe(format=render)` request grammar (§8.2) --
    `{format, contacts, overlay, maxcontacts}` -- round-trips through
    `parse_render_debug_caps` (pure parse) and `set_render_debug_caps`
    (records it on the owner). `format` is not a debug-tier field and must be
    ignored by the parser, not raise or leak into the parsed caps."""
    owner = object.__new__(FastPathOwner)
    owner._lock = __import__("threading").Lock()
    owner._render_debug_caps = {"contacts": False, "overlay": False, "max_contacts": 0}

    req = {"format": "render", "contacts": True, "overlay": True, "maxcontacts": 12}
    caps = FastPathOwner.parse_render_debug_caps(req)
    assert caps == {"contacts": True, "overlay": True, "max_contacts": 12}
    assert "format" not in caps

    owner.set_render_debug_caps(req)
    assert owner._render_debug_caps == caps


def test_subscribe_grammar_max_contacts_alias():
    """`max_contacts` (snake_case) is an accepted alias for `maxcontacts`
    (the wire spelling) -- both must parse to the same field."""
    a = FastPathOwner.parse_render_debug_caps({"contacts": True, "maxcontacts": 5})
    b = FastPathOwner.parse_render_debug_caps({"contacts": True, "max_contacts": 5})
    assert a == b == {"contacts": True, "overlay": False, "max_contacts": 5}


def test_subscribe_grammar_defaults_are_lean():
    """An empty/absent subscribe payload parses to the all-off lean default --
    a subscriber that sends no debug-tier keys at all must get nothing."""
    assert FastPathOwner.parse_render_debug_caps({}) == {
        "contacts": False, "overlay": False, "max_contacts": 0,
    }
    assert FastPathOwner.parse_render_debug_caps(None) == {
        "contacts": False, "overlay": False, "max_contacts": 0,
    }


def test_subscribe_grammar_malformed_maxcontacts_does_not_raise():
    """A malformed `maxcontacts` (not coercible to int) must not raise --
    it degrades to the cap-off value (0 == "no cap"/lean), matching the
    owner's "malformed wire value can't wedge the handler" invariant used
    elsewhere (e.g. `_vec3` padding for perturb vectors)."""
    caps = FastPathOwner.parse_render_debug_caps({"contacts": True, "maxcontacts": "not-a-number"})
    assert caps["max_contacts"] == 0
    assert caps["contacts"] is True


def test_grpc_subscribe_request_negotiates_caps_end_to_end(tmp_path, scene):
    """End-to-end: a real gRPC `subscribe(format=render, contacts=True,
    overlay=True, maxcontacts=N)` request must result in a streamed frame
    carrying exactly those debug-tier fields, sized/capped per §8.2."""
    model, data = scene
    grpc_port = _free_port()
    owner = _make_owner(tmp_path)
    try:
        owner.start_grpc_server(port=grpc_port)
        owner.publish_mjdata(1, model, data)

        def sub_pkt(op, req):
            p = upb.UrlabPacket(op=op, payload=msgpack.packb(req, use_bin_type=True), sequence_id=1)
            e = dm_env_rpc_pb2.EnvironmentRequest()
            e.extension.Pack(p)
            return e

        stub = dm_env_rpc_pb2_grpc.EnvironmentStub(grpc.insecure_channel(f"127.0.0.1:{grpc_port}"))
        req = sub_pkt("subscribe", {"format": "render", "contacts": True, "overlay": True, "maxcontacts": 2})

        got = None
        deadline = time.time() + 3.0
        for r in stub.Process(iter([req])):
            out = upb.UrlabPacket()
            r.extension.Unpack(out)
            fr = msgpack.unpackb(bytes(out.payload), raw=False, strict_map_key=False)
            if "bxpos" in fr:
                # Republish after negotiating so this frame is guaranteed to
                # have been built under the just-set caps.
                owner.publish_mjdata(2, model, data)
                got = owner.latest_transforms()
                break
            if time.time() > deadline:
                break
        assert got is not None
        assert "xfrc_applied" in got and len(got["xfrc_applied"]) == 6 * model.nbody
        assert "subtree_com" in got
    finally:
        owner.close()
