# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Peek viewer wire tests: viewer-frame decode, perturb request shape, and a real
ZMQ round-trip of FastPathOwner.publish_state on the "viewer" topic (the contract
UE's ViewerSubscribeTransport also speaks). No mujoco, no GUI."""
from __future__ import annotations

import socket
import time

import msgpack
import pytest
import zmq

from urlab_client.peek import VIEWER_TOPIC, decode_viewer_frame, perturb_request


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_decode_roundtrip():
    payload = msgpack.packb({"t": 1.5, "qpos": [1, 2, 3], "qvel": [0.1, 0.2]},
                            use_bin_type=True)
    t, qpos, qvel = decode_viewer_frame(payload)
    assert t == 1.5
    assert list(qpos) == [1, 2, 3]
    assert list(qvel) == [0.1, 0.2]


def test_decode_rejects_bad():
    assert decode_viewer_frame(msgpack.packb({"t": 0}, use_bin_type=True)) is None
    assert decode_viewer_frame(b"\xff\xff\xff") is None


def test_perturb_request_shape():
    assert perturb_request(3, [1, 2, 3], [4, 5, 6]) == {
        "op": "fastpath_perturb", "body": 3,
        "force": [1.0, 2.0, 3.0], "torque": [4.0, 5.0, 6.0],
    }


def test_owner_publish_state_roundtrip(tmp_path):
    """A FastPathOwner.publish_state frame is received + decoded on the viewer bus."""
    from urlab_client.fastpath_owner import FastPathOwner

    owner = FastPathOwner(
        b"", scene="t", control_port=_free_port(), bus_port=_free_port(),
        advertise_host="127.0.0.1", registry_dir=str(tmp_path),
    )
    try:
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, VIEWER_TOPIC)
        sub.setsockopt(zmq.RCVTIMEO, 200)
        sub.connect(owner.bus_endpoint)

        got = None
        for _ in range(100):  # PUB/SUB slow-joiner: publish until the SUB is attached
            owner.publish_state(2.0, [1.0, 2.0], [0.5])
            try:
                parts = sub.recv_multipart()
            except zmq.error.Again:
                time.sleep(0.02)
                continue
            assert parts[0] == VIEWER_TOPIC
            got = decode_viewer_frame(parts[-1])
            break
        sub.close(0)

        assert got is not None, "no viewer frame received"
        t, qpos, qvel = got
        assert t == 2.0
        assert list(qpos) == [1.0, 2.0]
        assert list(qvel) == [0.5]
    finally:
        owner.close()


def test_transport_viewer_stream_roundtrip(tmp_path):
    """ZmqTransport.start_viewer_stream receives frames an owner publish_states."""
    import queue

    from urlab_client.fastpath_owner import FastPathOwner
    from urlab_client.transports import make_transport

    ctrl_port, bus_port = _free_port(), _free_port()
    owner = FastPathOwner(
        b"", scene="t", control_port=ctrl_port, bus_port=bus_port,
        advertise_host="127.0.0.1", registry_dir=str(tmp_path),
    )
    t = make_transport("zmq", address="tcp://127.0.0.1", step_port=ctrl_port)
    got: "queue.Queue" = queue.Queue()
    try:
        t.start_viewer_stream(lambda f: got.put(f), endpoint=owner.bus_endpoint)
        frame = None
        for _ in range(100):  # slow-joiner: publish until the SUB attaches
            owner.publish_state(3.0, [1.0, 2.0], [0.5])
            try:
                frame = got.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        assert frame is not None, "no viewer frame delivered to the transport stream"
        assert list(frame["qpos"]) == [1.0, 2.0]
        assert frame["t"] == 3.0
    finally:
        t.close()
        owner.close()


def test_grpc_owner_stream_and_perturb(tmp_path):
    """The Python owner gRPC server: subscribe_viewer streams {t,qpos,qvel} and
    fastpath_perturb accumulates -- both driven through GrpcTransport."""
    import queue

    from urlab_client.fastpath_owner import FastPathOwner
    from urlab_client.peek import perturb_request
    from urlab_client.transports import make_transport

    grpc_port = _free_port()
    owner = FastPathOwner(
        b"", scene="t", control_port=_free_port(), bus_port=_free_port(),
        advertise_host="127.0.0.1", registry_dir=str(tmp_path),
    )
    owner.start_grpc_server(port=grpc_port)
    t = make_transport("grpc", address="127.0.0.1", step_port=grpc_port,
                       recv_timeout_ms=3000)
    got: "queue.Queue" = queue.Queue()
    try:
        t.start_viewer_stream(lambda f: got.put(f))
        frame = None
        for i in range(200):  # keep publishing new frames until one is streamed back
            owner.publish_state(5.0 + i * 0.01, [1.0, 2.0, 3.0], [0.0])
            try:
                frame = got.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        assert frame is not None, "no viewer frame streamed over gRPC"
        assert list(frame["qpos"]) == [1.0, 2.0, 3.0]

        reply = t.rpc(perturb_request(2, [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]),
                      recv_timeout_ms=3000)
        assert reply.get("ok") is True
        assert owner.drain_perturbations() == {2: [1.0, 0.0, 0.0, 0.0, 0.0, 0.5]}
    finally:
        t.close()
        owner.close()


def test_grpc_owner_transform_stream_and_hello(tmp_path):
    """Owner gRPC server, mirror path: subscribe(format=transforms) streams the
    per-body bxpos/bxquat payload under stream_cameras, fastpath_hello returns the
    model bytes + format, and a no-stream_cameras owner refuses the subscribe."""
    import grpc

    from urlab_client.fastpath_owner import FastPathOwner
    from urlab_client.transports._dmenv import (
        dm_env_rpc_pb2, dm_env_rpc_pb2_grpc, urlab_dm_env_rpc_pb2 as upb,
    )

    def sub_pkt(op, req):
        p = upb.UrlabPacket(op=op, payload=msgpack.packb(req, use_bin_type=True),
                            sequence_id=1)
        e = dm_env_rpc_pb2.EnvironmentRequest(); e.extension.Pack(p)
        return e

    def unwrap(resp):
        o = upb.UrlabPacket(); resp.extension.Unpack(o)
        return o.op, msgpack.unpackb(bytes(o.payload), raw=False, strict_map_key=False)

    port = _free_port()
    owner = FastPathOwner(
        b"MODELBYTES", scene="t", model_format="xml",
        control_port=_free_port(), bus_port=_free_port(),
        advertise_host="127.0.0.1", registry_dir=str(tmp_path))
    owner.start_grpc_server(port=port)
    stub = dm_env_rpc_pb2_grpc.EnvironmentStub(grpc.insecure_channel(f"127.0.0.1:{port}"))
    try:
        for r in stub.Process(iter([sub_pkt("fastpath_hello", {})])):
            _, hello = unwrap(r); break
        assert hello["format"] == "xml" and hello["model"] == b"MODELBYTES"
        assert "stream_cameras" in hello["capabilities"]

        owner.publish_bodies(7, bxpos=[1.0, 2.0, 3.0], bxquat=[1.0, 0.0, 0.0, 0.0])
        for r in stub.Process(iter([sub_pkt("subscribe", {"format": "transforms"})])):
            op, fr = unwrap(r); break
        assert op == "view_frame"
        assert list(fr["bxpos"]) == [1.0, 2.0, 3.0] and fr["f"] == 7
    finally:
        owner.close()

    port2 = _free_port()
    ro = FastPathOwner(
        b"", scene="ro", control_port=_free_port(), bus_port=_free_port(),
        advertise_host="127.0.0.1", registry_dir=str(tmp_path),
        capabilities=("accept_input",))  # no stream_cameras
    ro.start_grpc_server(port=port2)
    stub2 = dm_env_rpc_pb2_grpc.EnvironmentStub(grpc.insecure_channel(f"127.0.0.1:{port2}"))
    try:
        for r in stub2.Process(iter([sub_pkt("subscribe", {"format": "transforms"})])):
            _, reply = unwrap(r); break
        assert reply["ok"] is False and "stream_cameras" in reply["error"]
    finally:
        ro.close()


def test_readonly_owner_refuses_perturb(tmp_path):
    """An owner without the AcceptInput capability refuses fastpath_perturb."""
    from urlab_client.fastpath_owner import FastPathOwner

    owner = FastPathOwner(
        b"", scene="t", control_port=_free_port(), bus_port=_free_port(),
        advertise_host="127.0.0.1", registry_dir=str(tmp_path),
        capabilities=("view",),  # look, don't touch
    )
    try:
        assert owner.accepts_input is False
        assert "accept_input" not in owner.capabilities and "stream_cameras" in owner.capabilities
        req = zmq.Context.instance().socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 500)
        req.setsockopt(zmq.SNDTIMEO, 500)
        req.connect(owner.control_endpoint)
        req.send(msgpack.packb(perturb_request(2, [1, 0, 0], [0, 0, 0]),
                               use_bin_type=True))
        rep = None
        for _ in range(100):
            owner.serve_pending()
            try:
                rep = msgpack.unpackb(req.recv(), raw=False)
                break
            except zmq.error.Again:
                time.sleep(0.01)
        req.close(0)
        assert rep is not None and rep.get("ok") is False
        assert "accept_input" in rep.get("error", "")
        assert owner.drain_perturbations() == {}  # nothing accumulated
    finally:
        owner.close()


def test_owner_accepts_peek_perturb(tmp_path):
    """A peek's perturb_request over the control channel is accepted + drained --
    the push-back half of the loop, end to end at the wire level."""
    from urlab_client.fastpath_owner import FastPathOwner

    owner = FastPathOwner(
        b"", scene="t", control_port=_free_port(), bus_port=_free_port(),
        advertise_host="127.0.0.1", registry_dir=str(tmp_path),
    )
    try:
        req = zmq.Context.instance().socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 500)
        req.setsockopt(zmq.SNDTIMEO, 500)
        req.connect(owner.control_endpoint)
        req.send(msgpack.packb(perturb_request(2, [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]),
                               use_bin_type=True))

        rep = None
        for _ in range(100):
            owner.serve_pending()
            try:
                rep = msgpack.unpackb(req.recv(), raw=False)
                break
            except zmq.error.Again:
                time.sleep(0.01)
        req.close(0)

        assert rep is not None and rep.get("ok") is True
        assert owner.drain_perturbations() == {2: [1.0, 0.0, 0.0, 0.0, 0.0, 0.5]}
    finally:
        owner.close()
