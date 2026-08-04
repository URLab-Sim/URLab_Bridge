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

"""
Shared pytest fixtures for the URLab bridge test suite.

Key helper: `mock_step_server` spins up a ZMQ REP socket in a thread with
a scripted reply queue, letting every unit test exercise the wire
protocol without a live Unreal instance. The fixture also returns the
bytes of a tiny synthetic MJB (single-arm two-hinge pendulum with one
actuator + one sensor) so handshake / MJB-walk tests get a real
`mujoco.MjModel` round-trip.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Optional

import pytest

# Make the `src/` layout importable without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _require(module_name: str):
    """Skip a test cleanly if an optional import is missing."""
    return pytest.importorskip(module_name)


@pytest.fixture(scope="session")
def mujoco_mod():
    return _require("mujoco")


@pytest.fixture(scope="session")
def msgpack_mod():
    return _require("msgpack")


@pytest.fixture(scope="session")
def zmq_mod():
    return _require("zmq")


@pytest.fixture(scope="session")
def synthetic_mjb(mujoco_mod, tmp_path_factory) -> bytes:
    """Compile a minimal MJCF scene with two articulations and return the MJB."""
    mjcf = """
    <mujoco model="urlab_test">
      <worldbody>
        <body name="vx300s_waist_link" pos="0 0 0.1">
          <joint name="vx300s_waist" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="capsule" size="0.02 0.05" fromto="0 0 0 0.1 0 0"/>
          <body name="vx300s_shoulder_link" pos="0.1 0 0">
            <joint name="vx300s_shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="capsule" size="0.02 0.05" fromto="0 0 0 0.1 0 0"/>
            <site name="vx300s_tip" pos="0.1 0 0" size="0.005"/>
          </body>
        </body>
        <body name="go2_hip" pos="1 0 0.1">
          <joint name="go2_hip" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="box" size="0.05 0.02 0.02"/>
        </body>
      </worldbody>
      <actuator>
        <position name="vx300s_waist" joint="vx300s_waist" kp="100"/>
        <motor    name="vx300s_shoulder" joint="vx300s_shoulder" gear="10"/>
        <motor    name="go2_hip" joint="go2_hip" gear="5"/>
      </actuator>
      <sensor>
        <jointpos name="vx300s_waist_pos" joint="vx300s_waist"/>
        <framepos name="vx300s_tip_pos" objtype="site" objname="vx300s_tip"/>
      </sensor>
    </mujoco>
    """
    model = mujoco_mod.MjModel.from_xml_string(mjcf)
    path = tmp_path_factory.mktemp("mjb") / "urlab_test.mjb"
    mujoco_mod.mj_saveModel(model, str(path))
    return path.read_bytes()


@pytest.fixture(scope="session")
def base_handshake(mujoco_mod, synthetic_mjb) -> Dict[str, Any]:
    """A canned `hello_ok` reply using the synthetic MJB."""
    return {
        "op": "hello_ok",
        "session_id": "test-session-0",
        "urlab_version": "urlab/test",
        "mujoco_version": mujoco_mod.__version__,
        "mjb": synthetic_mjb,
        "articulations": [
            {
                "prefix": "vx300s",
                "default_control_mode": "ue_controller",
                "controller": {
                    "kind": "pd",
                    "params": {
                        "kp": {"waist": 300.0, "shoulder": 280.0},
                        "kv": {"waist": 20.0, "shoulder": 18.0},
                        "torque_limit": {"waist": 35.0, "shoulder": 45.0},
                        "default_kp": 100.0,
                        "default_kv": 5.0,
                        "default_torque_limit": 200.0,
                    },
                    "schema": {
                        "kp": {"type": "per_joint", "dtype": "float32", "min": 0},
                        "kv": {"type": "per_joint", "dtype": "float32", "min": 0},
                        "torque_limit": {
                            "type": "per_joint",
                            "dtype": "float32",
                            "min": 0,
                        },
                        "default_kp": {"type": "scalar", "dtype": "float32", "min": 0},
                        "default_kv": {"type": "scalar", "dtype": "float32", "min": 0},
                        "default_torque_limit": {
                            "type": "scalar",
                            "dtype": "float32",
                            "min": 0,
                        },
                    },
                },
                "actuator_types": {
                    "waist": "position",
                    "shoulder": "motor",
                },
                "camera_topics": {
                    "wrist": {
                        "mode": "real",
                        "resolution": [320, 240],
                        "fovy": 45.0,
                        "zmq_endpoint": None,
                        "zmq_topic": None,
                    }
                },
            },
            {
                "prefix": "go2",
                "default_control_mode": "raw",
                "actuator_types": {"hip": "motor"},
            },
        ],
        "global_cameras": {},
        "entities": {},
    }


class MockStepServer:
    """Minimal REP-socket server driven by a scripted reply queue.

    Tests push replies onto `.replies` before invoking a client RPC. The
    server's recv loop pops them in order. Any received request is
    appended to `.received` so tests can assert on the request shape.
    """

    def __init__(self, zmq_mod, msgpack_mod, port: Optional[int] = None):
        self._zmq = zmq_mod
        self._msgpack = msgpack_mod
        self._ctx = zmq_mod.Context()
        self._socket = self._ctx.socket(zmq_mod.REP)
        if port is None:
            # Let the OS assign a free port. A fixed counter-allocated band
            # is not safe here: the Windows dynamic port range starts at
            # 1024, so any outbound socket in the suite can transiently hold
            # a port we were about to bind, producing "Address in use".
            port = self._socket.bind_to_random_port("tcp://127.0.0.1")
        else:
            self._socket.bind(f"tcp://127.0.0.1:{port}")
        self._socket.setsockopt(zmq_mod.RCVTIMEO, 500)

        self.port = port
        self.replies: List[Dict[str, Any]] = []
        self.received: List[Dict[str, Any]] = []
        # Meta requests are recorded separately so they don't shift
        # `received[]` indices for tests that introspect specific
        # post-handshake ops. Empty for any test that doesn't introspect.
        self.received_meta: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._socket.recv()
            except self._zmq.Again:
                continue
            except self._zmq.ContextTerminated:  # pragma: no cover
                return
            except Exception:  # pragma: no cover
                return
            try:
                req = self._msgpack.unpackb(raw, raw=False)
            except Exception:
                req = {"_unparseable": True}
            # The bridge's discover() always follows hello with a meta
            # lookup. Tests target specific app ops, not the schema
            # fetch — meta is infrastructure, so the mock auto-responds
            # and skips recording the request entirely. This keeps
            # `received[]` indices test-author-friendly:
            # `received[0]==hello`, `received[1]==first-app-op`, etc.
            #
            # Tests that need to customise the meta payload can still
            # queue an explicit meta_ok reply at the head — that path
            # consumes the queued reply and records the request in
            # `received_meta` instead of `received`.
            is_meta = isinstance(req, dict) and req.get("op") == "meta"
            head_is_meta = (
                self.replies
                and isinstance(self.replies[0], dict)
                and self.replies[0].get("op") == "meta_ok"
            )
            if is_meta:
                self.received_meta.append(req)
                if head_is_meta:
                    reply = self.replies.pop(0)
                else:
                    reply = {"op": "meta_ok", "ops": []}
            else:
                self.received.append(req)
                if self.replies:
                    reply = self.replies.pop(0)
                else:
                    reply = {"op": "error", "code": "test_no_reply_queued", "message": ""}
            try:
                self._socket.send(self._msgpack.packb(reply, use_bin_type=True))
            except Exception:  # pragma: no cover
                return

    def close(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:  # pragma: no cover
            pass
        try:
            self._socket.close(linger=0)
        except Exception:  # pragma: no cover
            pass
        try:
            self._ctx.term()
        except Exception:  # pragma: no cover
            pass


@pytest.fixture
def mock_step_server(zmq_mod, msgpack_mod):
    """Yield a fresh MockStepServer on an OS-assigned loopback port."""
    srv = MockStepServer(zmq_mod, msgpack_mod)
    try:
        yield srv
    finally:
        srv.close()
