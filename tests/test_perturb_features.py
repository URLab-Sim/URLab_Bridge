# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Perturbation wire-level tests.

Grounded in `audit_docs/0_render_source_of_truth.md` §10: perturbation is
exactly two ops -- **drag intent** `{select, active, localpos, refselpos}`
(the owner runs `mjv_applyPerturbForce`, a mass-scaled critically-damped
spring, so a dragged body settles instead of flying off) and **exact wrench**
`{body, force, torque}` (applied verbatim to `xfrc_applied`, no scaling).

`tests/test_perturb.py` already pins the spring math via DIRECT
`submit_perturb`/`apply_perturbations` calls. These tests instead go through
the real ZMQ `fastpath_perturb` wire op (`_handle_request`), covering the
full owner request path end to end, plus the two bug-revealing checks flagged
in `audit_docs/cleanup_audit_addendum.md` §B ("Bridge owner / farm /
examples"): `submit_perturb` writes the drag intent with **no lock** while
`apply_perturbations` reads it under one -- a "thread-safe" doc claim that
doesn't hold for the writer side.
"""
from __future__ import annotations

import socket
import threading
import time

import msgpack
import numpy as np
import pytest
import zmq

mujoco = pytest.importorskip("mujoco")

from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402

_CUBE_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="free_cube" pos="0 0 0.7">
      <freejoint/>
      <geom type="box" size="0.12 0.12 0.12" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
"""
_CUBE_BODY = 1  # world=0, free_cube=1


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_owner(tmp_path, **kw):
    kw.setdefault("scene", "cube")
    kw.setdefault("model_format", "xml")
    kw.setdefault("assets", {})
    kw.setdefault("ngeom", 2)
    kw.setdefault("control_port", _free_port())
    kw.setdefault("bus_port", _free_port())
    kw.setdefault("advertise_host", "127.0.0.1")
    kw.setdefault("registry_dir", str(tmp_path))
    return FastPathOwner(b"", **kw)


def _model_data():
    model = mujoco.MjModel.from_xml_string(_CUBE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _req_socket(endpoint: str):
    req = zmq.Context.instance().socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 1000)
    req.setsockopt(zmq.SNDTIMEO, 1000)
    req.connect(endpoint)
    return req


def _roundtrip(owner: FastPathOwner, req_sock, payload: dict) -> dict:
    """Send one wire request and pump `serve_pending` until the reply lands --
    the same pattern `test_owner_stream.py` uses for `fastpath_perturb`."""
    req_sock.send(msgpack.packb(payload, use_bin_type=True))
    for _ in range(200):
        owner.serve_pending()
        try:
            return msgpack.unpackb(req_sock.recv(), raw=False)
        except zmq.error.Again:
            time.sleep(0.005)
    raise AssertionError("owner never replied to fastpath_perturb")


def _intent_request(select: int, active: bool, localpos, refselpos) -> dict:
    return {
        "op": "fastpath_perturb", "select": int(select), "active": bool(active),
        "localpos": [float(x) for x in localpos],
        "refselpos": [float(x) for x in refselpos],
    }


def _wrench_request(body: int, force, torque) -> dict:
    return {
        "op": "fastpath_perturb", "body": int(body),
        "force": [float(x) for x in force], "torque": [float(x) for x in torque],
    }


# --------------------------------------------------------------------------- #
# drag intent, over the wire, settles instead of flying off (§10 + the
# mass-scaled critically-damped spring `apply_perturbations` runs).
# --------------------------------------------------------------------------- #

def test_drag_intent_over_wire_settles_on_target(tmp_path):
    # The owner HOLDS the drag intent across steps (see the release test), and
    # apply_perturbations re-runs mjv_applyPerturbForce from the current pose each
    # call -- so the intent is injected ONCE over the wire, then the spring is
    # driven locally. (Re-sending a wire roundtrip per physics step, as an earlier
    # draft did, made this ~600 blocking REQ/REP roundtrips and hung the suite.)
    owner = _make_owner(tmp_path)
    try:
        req_sock = _req_socket(owner.control_endpoint)
        model, data = _model_data()
        target = data.xpos[_CUBE_BODY].copy() + [0.3, 0.0, 0.0]
        try:
            rep = _roundtrip(
                owner, req_sock,
                _intent_request(_CUBE_BODY, True, [0, 0, 0], target))
            assert rep.get("ok") is True
            for _ in range(1500):  # pure local apply+step: fast, bounded
                data.xfrc_applied[:] = 0.0
                owner.apply_perturbations(model, data)
                mujoco.mj_step(model, data)
            # Settles near the drag target and never blows up -- a spring, not
            # an undamped/unscaled force that flies off.
            assert data.xpos[_CUBE_BODY][0] == pytest.approx(target[0], abs=0.1)
            assert np.all(np.abs(data.xpos[_CUBE_BODY]) < 5.0)
        finally:
            req_sock.close(0)
    finally:
        owner.close()


def test_drag_intent_release_over_wire_stops_pull(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        req_sock = _req_socket(owner.control_endpoint)
        model, data = _model_data()
        try:
            target = data.xpos[_CUBE_BODY].copy() + [0.2, 0, 0]
            rep = _roundtrip(owner, req_sock,
                             _intent_request(_CUBE_BODY, True, [0, 0, 0], target))
            assert rep.get("ok") is True
            data.xfrc_applied[:] = 0.0
            owner.apply_perturbations(model, data)
            assert np.linalg.norm(data.xfrc_applied[_CUBE_BODY][:3]) > 1.0

            rep = _roundtrip(owner, req_sock,
                             _intent_request(_CUBE_BODY, False, [0, 0, 0], [0, 0, 0]))
            assert rep.get("ok") is True
            data.xfrc_applied[:] = 0.0
            owner.apply_perturbations(model, data)
            assert np.linalg.norm(data.xfrc_applied[_CUBE_BODY]) == pytest.approx(0.0)
        finally:
            req_sock.close(0)
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# exact wrench, over the wire, applied verbatim (no spring, no scaling).
# --------------------------------------------------------------------------- #

def test_exact_wrench_over_wire_applied_verbatim(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        req_sock = _req_socket(owner.control_endpoint)
        model, data = _model_data()
        try:
            force, torque = [2.5, -1.0, 0.0], [0.0, 0.0, 0.75]
            rep = _roundtrip(owner, req_sock,
                             _wrench_request(_CUBE_BODY, force, torque))
            assert rep.get("ok") is True
            data.xfrc_applied[:] = 0.0
            owner.apply_perturbations(model, data)
            got = list(data.xfrc_applied[_CUBE_BODY])
            assert got == pytest.approx(force + torque)
        finally:
            req_sock.close(0)
    finally:
        owner.close()


def test_exact_wrench_accumulates_multiple_pushes_before_drain(tmp_path):
    """submit_perturb_force ACCUMULATES (fastpath_owner.py:412-424); two wire
    pushes to the same body before the next step sum, they don't overwrite."""
    owner = _make_owner(tmp_path)
    try:
        req_sock = _req_socket(owner.control_endpoint)
        model, data = _model_data()
        try:
            rep1 = _roundtrip(owner, req_sock,
                              _wrench_request(_CUBE_BODY, [1.0, 0, 0], [0, 0, 0]))
            rep2 = _roundtrip(owner, req_sock,
                              _wrench_request(_CUBE_BODY, [1.0, 0, 0], [0, 0, 0]))
            assert rep1.get("ok") is True and rep2.get("ok") is True
            data.xfrc_applied[:] = 0.0
            owner.apply_perturbations(model, data)
            assert list(data.xfrc_applied[_CUBE_BODY][:3]) == pytest.approx(
                [2.0, 0.0, 0.0])
        finally:
            req_sock.close(0)
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# accept_input capability gates acceptance (both shapes).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", ["intent", "wrench"])
def test_readonly_owner_refuses_wire_perturb(tmp_path, shape):
    owner = _make_owner(tmp_path, capabilities=("view",))  # no accept_input
    try:
        assert owner.accepts_input is False
        req_sock = _req_socket(owner.control_endpoint)
        model, data = _model_data()
        try:
            if shape == "intent":
                payload = _intent_request(_CUBE_BODY, True, [0, 0, 0], [0.5, 0, 0])
            else:
                payload = _wrench_request(_CUBE_BODY, [5.0, 0, 0], [0, 0, 0])
            rep = _roundtrip(owner, req_sock, payload)
            assert rep.get("ok") is False
            assert "accept_input" in rep.get("error", "")
            # Nothing was queued: applying perturbations is a no-op.
            data.xfrc_applied[:] = 0.0
            owner.apply_perturbations(model, data)
            assert np.linalg.norm(data.xfrc_applied[_CUBE_BODY]) == pytest.approx(0.0)
        finally:
            req_sock.close(0)
    finally:
        owner.close()


def test_input_granted_owner_accepts_wire_perturb(tmp_path):
    """Sanity converse of the refusal test: the default capability set (which
    includes accept_input) does accept the same request shape."""
    owner = _make_owner(tmp_path)  # default caps include accept_input
    try:
        assert owner.accepts_input is True
        req_sock = _req_socket(owner.control_endpoint)
        try:
            rep = _roundtrip(owner, req_sock,
                             _wrench_request(_CUBE_BODY, [1, 0, 0], [0, 0, 0]))
            assert rep.get("ok") is True
        finally:
            req_sock.close(0)
    finally:
        owner.close()


# --------------------------------------------------------------------------- #
# Bug-revealing (best-effort): concurrent submit_perturb under load.
#
# addendum §B: "submit_perturb (fastpath_owner.py:398) writes intent with no
# lock; reader locks; sibling submit_perturb_force locks. False 'thread-safe'
# doc." Intended behaviour is that concurrent drag-intent submissions (e.g. a
# ZMQ REP thread and a gRPC server thread both calling submit_perturb) never
# hand `apply_perturbations` a torn/self-inconsistent intent. Each writer
# tags every field of its intent with its own thread id so a reader can
# detect a torn read (fields from two different submissions mixed together).
# This is a best-effort stress test, not a guaranteed reproduction: a single
# `self._perturb_intent = {...}` attribute rebind happens to be atomic under
# CPython's GIL even without the documented lock, so this is expected to pass
# today -- it exists to catch a regression (e.g. in-place mutation of the
# intent dict) that WOULD be unsafe given the missing lock.
# --------------------------------------------------------------------------- #

def test_concurrent_submit_perturb_is_race_free(tmp_path):
    owner = _make_owner(tmp_path)
    try:
        stop = threading.Event()
        bad = []

        def writer(tag: float) -> None:
            n = 0
            while not stop.is_set():
                v = tag + (n % 7) * 1e-3  # small per-iteration jitter, same tag
                owner.submit_perturb(1, True, [v, v, v], [v, v, v])
                n += 1

        def reader() -> None:
            for _ in range(4000):
                with owner._lock:  # noqa: SLF001 -- mirrors apply_perturbations' read
                    intent = dict(owner._perturb_intent) if owner._perturb_intent else None
                if intent is not None:
                    vals = list(intent["localpos"]) + list(intent["refselpos"])
                    # All six components must come from the SAME writer call --
                    # i.e. all within one jitter step of each other. A torn
                    # read (fields from two different submit_perturb calls)
                    # would show up as widely mismatched components.
                    if max(vals) - min(vals) > 1e-6:
                        bad.append(vals)

        writers = [threading.Thread(target=writer, args=(float(i),)) for i in range(4)]
        for w in writers:
            w.start()
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        reader_thread.join()
        stop.set()
        for w in writers:
            w.join(timeout=2.0)

        assert bad == [], f"observed torn drag-intent reads: {bad[:5]}"
    finally:
        owner.close()
