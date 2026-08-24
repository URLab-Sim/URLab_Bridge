# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Debug-tier (source-of-truth §8.2) subscription semantics, end to end over
real ZMQ/gRPC sockets against a live `FastPathOwner`.

The intended contract: `StreamContacts`/`StreamOverlay` are negotiated **per
subscription** on `subscribe(format=render)`. A subscriber that asks for
neither must pay ZERO extra bytes; a subscriber that asks for both must get
exactly those fields; and one subscriber's request must never change what any
OTHER subscriber (gRPC or the always-on ZMQ `render` topic) receives.

Cleanup audit addendum §A2 documents that this is currently broken: debug caps
are one field on the owner (`self._render_debug_caps`), so they are (a)
clobbered across subscribers -- the last debug subscribe wins for everyone,
(b) leaked onto the shared ZMQ `render` topic, which has no way to ask for the
debug tier at all, and (c) never reset when a debug subscriber disconnects.

Per the task brief: tests below assert the INTENDED behavior. Where §A2 makes
that fail today, the test is written anyway (not xfail) with a
`# EXPECTED FAIL (addendum §A2): ...` comment, so it goes green the moment the
per-subscription fix lands.
"""
from __future__ import annotations

import concurrent.futures
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

# Plane + an overlapping box (real contacts) plus an actuated hinge arm (nu>0,
# so the overlay bundle has a non-empty ctrl array) -- same shape of scene as
# the existing owner-side unit tests, kept local per the task's "no edits to
# existing test files" constraint.
_SCENE_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="boxA" pos="0 0 0.05"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
    <body name="arm" pos="2 0 0.5">
      <joint name="h" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0.1 0 0"/>
    </body>
  </worldbody>
  <actuator><motor name="m" joint="h" gear="1"/></actuator>
</mujoco>
"""

# The full set of debug-tier keys `_append_render_debug_fields` can add
# (source-of-truth §8.2 tables). "Zero extra bytes" for a lean subscriber
# means none of these appear on its frame.
_DEBUG_KEYS = {
    "contacts", "xfrc_applied", "subtree_com", "ctrl", "act", "wrap_xpos",
    "wrap_obj", "ten_wrapadr", "ten_wrapnum", "eq_active", "eq_anchor",
    "sensordata", "light_xpos", "light_xdir",
}

# The always-present render tier (§8.1): transforms + per-camera transforms
# (present, possibly empty, whenever publish_mjdata is used) + the optional
# user/free-camera pose. None of these are debug-tier fields.
_RENDER_KEYS = {"f", "t", "bxpos", "bxquat", "cxpos", "cxquat", "ucpos", "ucfwd", "ucup"}


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
    mujoco.mj_forward(model, data)
    assert data.ncon > 0, "scene must have a real contact"
    return model, data


def _make_owner(tmp_path, **kw):
    kw.setdefault("scene", "debug_tier_test")
    kw.setdefault("control_port", _free_port())
    kw.setdefault("bus_port", _free_port())
    kw.setdefault("advertise_host", "127.0.0.1")
    kw.setdefault("registry_dir", str(tmp_path))
    return FastPathOwner(b"MODELBYTES", model_format="xml", **kw)


def _sub_pkt(op: str, req: dict, seq: int = 1):
    p = upb.UrlabPacket(op=op, payload=msgpack.packb(req, use_bin_type=True), sequence_id=seq)
    e = dm_env_rpc_pb2.EnvironmentRequest()
    e.extension.Pack(p)
    return e


def _open_subscribe(port: int, req: dict):
    """Open a real gRPC `subscribe(format=render, ...)` call and return the
    (channel, call) pair. The call is a live server-stream iterator: each
    `next(call)` blocks until the owner publishes a NEW frame (different `f`).
    Ending the request iterator (as here) does not close the RPC -- the
    server keeps streaming; this mirrors `test_owner_stream.py`'s pattern."""
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = dm_env_rpc_pb2_grpc.EnvironmentStub(channel)
    call = stub.Process(iter([_sub_pkt("subscribe", req)]))
    return channel, call


def _next_frame(call, timeout: float = 3.0) -> dict:
    """Block for the next streamed frame, bounded by `timeout` (a bare
    `next()` would hang forever if the owner never republishes)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        resp = ex.submit(next, call).result(timeout=timeout)
    out = upb.UrlabPacket()
    resp.extension.Unpack(out)
    return msgpack.unpackb(bytes(out.payload), raw=False, strict_map_key=False)


def _subscribe_and_sync(port: int, req: dict):
    """Open a subscribe call and block until the OWNER has actually processed
    the request (`set_render_debug_caps` has run on the server thread).

    Necessary because `stub.Process(iter([...]))` is async: the server may not
    have parsed the subscribe request yet by the time this call returns, so a
    publish issued right after opening the call can race the caps update.
    `owner_server.py`'s `Process` calls `set_render_debug_caps` BEFORE it can
    yield anything at all (even the pre-existing cached frame), so receiving
    ANY frame back is proof the caps are now set -- deterministically
    resolving the race without a sleep-and-hope. The frame returned here may
    be a stale one built under the PREVIOUS caps; callers that care about the
    negotiated caps must publish again afterwards and call `_next_frame` once
    more.
    """
    channel, call = _open_subscribe(port, req)
    _next_frame(call)  # drains whatever is already cached; proves caps are set
    return channel, call


# --------------------------------------------------------------------------- #
# Baseline (correct today): a lone subscriber gets exactly what it asked for.
# --------------------------------------------------------------------------- #

def test_lone_debug_subscriber_gets_requested_fields(tmp_path, scene):
    """A subscriber that requests both StreamContacts and StreamOverlay, alone
    on the owner, must get those fields on its streamed frame."""
    model, data = scene
    port = _free_port()
    owner = _make_owner(tmp_path)
    channel = None
    try:
        owner.start_grpc_server(port=port)
        owner.publish_mjdata(1, model, data)
        channel, call = _subscribe_and_sync(
            port, {"format": "render", "contacts": True, "overlay": True, "maxcontacts": 0})
        owner.publish_mjdata(2, model, data)
        frame = _next_frame(call)
        assert "contacts" in frame and len(frame["contacts"]) == data.ncon
        assert "xfrc_applied" in frame and len(frame["xfrc_applied"]) == 6 * model.nbody
        assert "ctrl" in frame and len(frame["ctrl"]) == model.nu
    finally:
        if channel is not None:
            channel.close()
        owner.close()


def test_lone_lean_subscriber_pays_zero_extra_bytes(tmp_path, scene):
    """A subscriber that requests neither StreamContacts nor StreamOverlay,
    alone on the owner (nobody else has ever negotiated debug fields), must
    get a frame with ONLY the render-tier transform keys -- no debug key at
    all, i.e. genuinely zero extra bytes, not merely 'empty' debug arrays."""
    model, data = scene
    port = _free_port()
    owner = _make_owner(tmp_path)
    channel = None
    try:
        owner.start_grpc_server(port=port)
        owner.publish_mjdata(1, model, data)
        channel, call = _subscribe_and_sync(port, {"format": "render"})
        owner.publish_mjdata(2, model, data)
        frame = _next_frame(call)
        assert not (_DEBUG_KEYS & set(frame)), f"lean subscriber got debug keys: {_DEBUG_KEYS & set(frame)}"
        assert set(frame) <= _RENDER_KEYS, f"unexpected non-render-tier keys: {set(frame) - _RENDER_KEYS}"
    finally:
        if channel is not None:
            channel.close()
        owner.close()


# --------------------------------------------------------------------------- #
# Addendum §A2: per-subscriber isolation -- EXPECTED FAIL today.
# --------------------------------------------------------------------------- #

def test_debug_caps_are_per_subscriber_not_global(tmp_path, scene):
    """Two concurrent gRPC subscribers: B subscribes lean first, A then
    subscribes requesting the full debug tier. Correct/intended behavior: A's
    request must not change what B receives -- B must keep paying zero extra
    bytes on every subsequent frame, regardless of what any other subscriber
    negotiates.

    # EXPECTED FAIL (addendum §A2): `_render_debug_caps` is a single field on
    # the owner (fastpath_owner.py `self._render_debug_caps`, set from
    # `set_render_debug_caps` in `owner_server.py`'s `subscribe` handler), and
    # every subscriber's frame is read from the SAME cached
    # `self._latest_transforms` (`_send_transforms` appends the debug tier
    # once, before caching, then both the ZMQ publish and every gRPC
    # subscriber's `latest_transforms()` read that one dict). So the instant A
    # negotiates the debug tier, B's very next frame -- the same object A
    # gets -- also carries the debug tier. This assertion is the crisp,
    # deterministic form of "clobbered across subscribers": it must flip green
    # once caps are tracked per-subscription instead of per-owner.
    """
    model, data = scene
    port = _free_port()
    owner = _make_owner(tmp_path)
    channel_a = channel_b = None
    try:
        owner.start_grpc_server(port=port)
        owner.publish_mjdata(1, model, data)

        channel_b, call_b = _subscribe_and_sync(port, {"format": "render"})  # lean
        owner.publish_mjdata(2, model, data)
        frame_b1 = _next_frame(call_b)
        assert not (_DEBUG_KEYS & set(frame_b1)), "B should be lean before anyone negotiates debug"

        channel_a, call_a = _subscribe_and_sync(
            port, {"format": "render", "contacts": True, "overlay": True, "maxcontacts": 0})
        owner.publish_mjdata(3, model, data)
        frame_a1 = _next_frame(call_a)
        assert "xfrc_applied" in frame_a1, "A negotiated the debug tier and should have it"

        # B never asked for anything -- but gets the same frame A does.
        frame_b2 = _next_frame(call_b)
        assert not (_DEBUG_KEYS & set(frame_b2)), (
            "EXPECTED FAIL (addendum §A2): lean subscriber B received debug-tier "
            f"keys {_DEBUG_KEYS & set(frame_b2)} leaked from subscriber A's negotiation "
            "-- render-debug-caps is a per-owner global, not per-subscription"
        )
    finally:
        if channel_a is not None:
            channel_a.close()
        if channel_b is not None:
            channel_b.close()
        owner.close()


def test_debug_caps_reset_after_subscriber_disconnects(tmp_path, scene):
    """A debug subscriber negotiates the full tier, then disconnects. Correct/
    intended behavior: once the debug subscriber is gone, nobody has asked for
    the debug tier any more, so the owner must stop paying for it -- a fresh
    lean read (or the shared frame cache) must go back to zero extra bytes.

    # EXPECTED FAIL (addendum §A2): "never reset on disconnect" -- nothing in
    # `owner_server.py`'s `_stream_render` (or its caller) calls
    # `set_render_debug_caps` back to the lean default when `context.is_active()`
    # goes False and the generator returns. The debug tier stays on forever
    # (until some OTHER subscriber happens to negotiate different caps), so
    # this assertion must flip green once disconnect resets the caps.
    """
    model, data = scene
    port = _free_port()
    owner = _make_owner(tmp_path)
    channel = None
    try:
        owner.start_grpc_server(port=port)
        owner.publish_mjdata(1, model, data)

        channel, call = _subscribe_and_sync(
            port, {"format": "render", "contacts": True, "overlay": True, "maxcontacts": 0})
        owner.publish_mjdata(2, model, data)
        negotiated = _next_frame(call)
        assert "xfrc_applied" in negotiated, "sanity: the debug tier really was negotiated"

        # Disconnect the debug subscriber.
        call.cancel()
        channel.close()
        channel = None
        # Give the server-side stream loop time to observe context.is_active()
        # go False and (if it did the right thing) reset the caps.
        time.sleep(0.5)

        owner.publish_mjdata(3, model, data)
        after_disconnect = owner.latest_transforms()
        assert not (_DEBUG_KEYS & set(after_disconnect)), (
            "EXPECTED FAIL (addendum §A2): debug-tier keys "
            f"{_DEBUG_KEYS & set(after_disconnect)} are still being computed/sent "
            "after the only subscriber that asked for them disconnected -- caps "
            "are never reset on disconnect"
        )
    finally:
        if channel is not None:
            channel.close()
        owner.close()


def test_zmq_render_topic_does_not_leak_grpc_debug_tier(tmp_path, scene):
    """The ZMQ `render` topic is the always-on, capability-less bus (source-of-
    truth: ZMQ subscribe has no way to negotiate `StreamContacts`/
    `StreamOverlay` -- addendum §B confirms `SetRenderDebugCaps` is gRPC-only).
    A gRPC subscriber negotiating the full debug tier must NEVER cause a plain
    ZMQ mirror -- which never asked for anything and structurally cannot ask --
    to start receiving those fields.

    # EXPECTED FAIL (addendum §A2): the debug tier is appended to the frame
    # dict BEFORE it is either cached for gRPC or published on the ZMQ `render`
    # topic (`fastpath_owner.py` `_send_transforms`), from the one shared
    # `self._render_debug_caps`. There is no ZMQ-vs-gRPC distinction at that
    # point, so a gRPC negotiation leaks straight onto the ZMQ wire. Must flip
    # green once the ZMQ publish path is guaranteed to carry only the
    # capability-less render tier.
    """
    model, data = scene
    grpc_port = _free_port()
    owner = _make_owner(tmp_path)
    channel = None
    try:
        owner.start_grpc_server(port=grpc_port)

        zmq_sub = zmq.Context.instance().socket(zmq.SUB)
        zmq_sub.setsockopt(zmq.SUBSCRIBE, b"render")
        zmq_sub.setsockopt(zmq.RCVTIMEO, 2000)
        zmq_sub.connect(owner.bus_endpoint)
        time.sleep(0.2)  # slow-joiner: let the SUB subscription land

        # Seed a frame so `_subscribe_and_sync` has something cached to drain
        # while it waits for the owner to have processed the subscribe request.
        owner.publish_mjdata(1, model, data)

        # Negotiate the full debug tier over gRPC only.
        channel, call = _subscribe_and_sync(
            grpc_port, {"format": "render", "contacts": True, "overlay": True, "maxcontacts": 0})
        owner.publish_mjdata(2, model, data)
        negotiated = _next_frame(call)
        assert "xfrc_applied" in negotiated, "sanity: gRPC subscriber really negotiated the debug tier"

        # Read the SAME publish on the plain ZMQ render topic.
        owner.publish_mjdata(3, model, data)
        deadline = time.time() + 2.0
        zmq_frame = None
        while time.time() < deadline:
            try:
                _topic, payload = zmq_sub.recv_multipart()
            except zmq.Again:
                continue
            zmq_frame = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            if zmq_frame.get("f") == 3:
                break
        zmq_sub.close(0)
        assert zmq_frame is not None, "never received a frame on the ZMQ render topic"
        assert not (_DEBUG_KEYS & set(zmq_frame)), (
            "EXPECTED FAIL (addendum §A2): plain ZMQ render-topic frame carries "
            f"debug-tier keys {_DEBUG_KEYS & set(zmq_frame)} leaked from a gRPC-only "
            "negotiation -- ZMQ subscribers have no way to ask for these and must "
            "never receive them"
        )
    finally:
        if channel is not None:
            channel.close()
        owner.close()
