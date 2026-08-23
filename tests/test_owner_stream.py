# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Owner stream + perturb wire tests: the surviving render tier (subscribe
(format=render) transform round-trip over gRPC) and the owner's fastpath_perturb
handling. The qpos render tier and the Python peek viewer were removed in Phase
3.2/3.3, so those tests (viewer-frame decode, publish_state / start_viewer_stream
round-trips, subscribe_viewer) are gone with them. No mujoco, no GUI."""
from __future__ import annotations

import socket
import time

import msgpack
import pytest
import zmq


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def perturb_request(body: int, force, torque) -> dict:
    """The exact-wrench perturb request the owner accepts over the control channel
    (formerly urlab_client.peek.perturb_request; inlined after peek's removal)."""
    return {
        "op": "fastpath_perturb", "body": int(body),
        "force": [float(x) for x in force], "torque": [float(x) for x in torque],
    }


def test_grpc_owner_transform_stream_and_hello(tmp_path):
    """Owner gRPC server, mirror path: subscribe(format=render) streams the
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
        for r in stub.Process(iter([sub_pkt("subscribe", {"format": "render"})])):
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
        for r in stub2.Process(iter([sub_pkt("subscribe", {"format": "render"})])):
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


def test_owner_accepts_perturb(tmp_path):
    """An exact-wrench perturb_request over the control channel is accepted +
    drained -- the push-back half of the loop, end to end at the wire level."""
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
