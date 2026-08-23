# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Control-in wire format (H4/H9 fold): the pure-ZMQ senders (robojudo, auto_grasp,
lerobot) must publish a msgpack ``{ids:[...], vals:[...]}`` map on the
``{prefix}/control `` topic -- the exact shape FURLabMsgpackUtil parses UE-side.

The legacy little-endian ``[i32 n][i32 id, f32 val]*`` binary format (parsed by the
now-removed UE unsafe-cast reader) must NOT be emitted. These tests pin the wire
bytes each sender puts on its PUB socket, without any live ZMQ: each sender is built
with ``object.__new__`` (skipping the ctx/connect) and handed a capturing socket.
"""
from __future__ import annotations

import msgpack
import numpy as np
import pytest
import zmq


class _CaptureSocket:
    """A stand-in PUB socket that records send_string / send calls verbatim."""

    def __init__(self):
        self.parts = []  # list of ("str"|"bin", payload, flags)

    def send_string(self, s, flags=0):
        self.parts.append(("str", s, flags))

    def send(self, data, flags=0):
        self.parts.append(("bin", data, flags))


def _assert_ids_vals_frame(sock, prefix, expect_ids, expect_vals):
    """Common assertions: topic frame + msgpack {ids,vals} body frame."""
    assert len(sock.parts) == 2, "must be exactly [topic][payload], no legacy frames"

    kind, topic, flags = sock.parts[0]
    assert kind == "str"
    assert topic == f"{prefix}/control "          # trailing space is part of the topic
    assert flags == zmq.SNDMORE                    # topic then body in one multipart msg

    kind, body, _flags = sock.parts[1]
    assert kind == "bin"
    decoded = msgpack.unpackb(body, raw=False)     # legacy binary would not unpack to a map
    assert isinstance(decoded, dict)
    assert set(decoded.keys()) == {"ids", "vals"}
    assert decoded["ids"] == expect_ids
    assert len(decoded["ids"]) == len(decoded["vals"])
    assert all(isinstance(v, float) for v in decoded["vals"])
    assert decoded["vals"] == pytest.approx(expect_vals)


def test_robojudo_send_control_packs_ids_vals():
    from urlab_policy.adapters.robojudo.env import ZmqLink

    link = object.__new__(ZmqLink)
    link.ctrl_pub = _CaptureSocket()
    link.send_control("robo", np.array([0.1, 0.2, 0.3]), actuator_ids=[5, 6, 7])

    _assert_ids_vals_frame(link.ctrl_pub, "robo", [5, 6, 7], [0.1, 0.2, 0.3])


def test_robojudo_send_control_ordinal_ids_when_unnamed():
    """No actuator_ids -> bare ordinal ids (the :5557-discovery-gone fallback)."""
    from urlab_policy.adapters.robojudo.env import ZmqLink

    link = object.__new__(ZmqLink)
    link.ctrl_pub = _CaptureSocket()
    link.send_control("robo", np.array([1.0, 2.0]))

    _assert_ids_vals_frame(link.ctrl_pub, "robo", [0, 1], [1.0, 2.0])


def test_auto_grasp_send_joints_packs_ids_vals():
    from urlab_policy.auto_grasp import ZMQInterface

    zi = object.__new__(ZMQInterface)
    zi.pub = _CaptureSocket()
    zi.send_joints("arm", np.array([0.5, -0.25]), actuator_ids=[3, 4])

    _assert_ids_vals_frame(zi.pub, "arm", [3, 4], [0.5, -0.25])


def test_lerobot_send_action_packs_ids_vals():
    from urlab_policy.lerobot_runner import LeRobotRunner

    runner = object.__new__(LeRobotRunner)
    runner._ctrl_pub = _CaptureSocket()
    runner._prefix = "lr"
    runner._actuator_ids = {"j0": 9, "j1": 10}
    runner._joint_ids = {}
    runner._send_action(np.array([0.7, 0.8]), ["j0", "j1"])

    _assert_ids_vals_frame(runner._ctrl_pub, "lr", [9, 10], [0.7, 0.8])
