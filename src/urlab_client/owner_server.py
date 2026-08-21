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

"""A gRPC face for a Python fast-path owner.

A UE instance is already a gRPC server; a Python owner is normally a gRPC
*client*. This embeds a small ``dm_env_rpc`` gRPC server in a Python owner so
viewers can subscribe + perturb over gRPC too -- one transport everywhere, not
just ZMQ. It speaks the exact same envelope as the UE bridge: a ``UrlabPacket``
(``op``, ``payload`` msgpack, ``sequence_id``) inside ``EnvironmentRequest/
Response.extension``.

Ops served (peek/viewer/VR are capability *consumers*, not modes):
- ``fastpath_hello`` -- scene/ngeom, the advertised capabilities, and the model
  bytes + format (``xml``/``mjz``/``mjb``; xml/mjz are decoded in-engine, so a
  mirror needs no MJB version match).
- ``subscribe`` -- gated on the ``stream_cameras`` capability; a server-stream of
  the owner's view. ``format`` selects the payload: ``transforms`` (the true
  mirror -- per-body ``bxpos``/``bxquat``, consumer runs zero MuJoCo; op
  ``view_frame``) or ``qpos`` (``{t,qpos,qvel}``; op ``viewer_frame``).
  ``subscribe_viewer`` is the back-compat alias for ``format=qpos``.
- ``fastpath_perturb`` -- gated on the ``accept_input`` capability; into the
  owner's perturb queue.

Started via ``FastPathOwner.start_grpc_server()``. The same contract is served by
a UE owner, so a viewer is owner-agnostic.
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import msgpack


def _pb():
    """Lazy-load the generated dm_env modules (they pull in grpc)."""
    from .transports._dmenv import (  # noqa: PLC0415
        dm_env_rpc_pb2,
        dm_env_rpc_pb2_grpc,
        urlab_dm_env_rpc_pb2,
    )
    return dm_env_rpc_pb2, dm_env_rpc_pb2_grpc, urlab_dm_env_rpc_pb2


def _req_format(payload: bytes) -> str:
    """Read the requested view format from a subscribe payload. Defaults to
    'transforms' (the true mirror payload) -- 'qpos' only if explicitly asked."""
    try:
        req = msgpack.unpackb(bytes(payload), raw=False, strict_map_key=False) or {}
        return "qpos" if str(req.get("format", "")).lower() == "qpos" else "transforms"
    except Exception:  # noqa: BLE001
        return "transforms"


class _OwnerServicer:
    """dm_env EnvironmentServicer (duck-typed -- add_EnvironmentServicer_to_server
    only reads ``.Process``, so no base class import is needed)."""

    def __init__(self, owner):
        self._owner = owner
        self._stream_poll_s = 0.005  # ~200 Hz; only emits on a NEW frame

    def Process(self, request_iterator, context):
        dm_env_rpc_pb2, _grpc_mod, urlab_pb2 = _pb()
        for env_req in request_iterator:
            if not env_req.HasField("extension"):
                continue
            pkt = urlab_pb2.UrlabPacket()
            if not env_req.extension.Unpack(pkt):
                continue
            op = pkt.op
            if op in ("subscribe", "subscribe_viewer"):
                # Subscribing to the owner's view is gated on the stream_cameras
                # capability (peek/viewer/VR are consumers of that cap, not modes).
                if not self._owner.streams_view:
                    yield self._wrap(op, pkt.sequence_id, msgpack.packb(
                        {"ok": False, "error": "capability disabled: stream_cameras"},
                        use_bin_type=True))
                    return
                # Format: "transforms" (true mirror -- bxpos/bxquat, viewer runs no
                # MuJoCo) or "qpos" ({t,qpos,qvel}). subscribe_viewer == qpos.
                fmt = "qpos" if op == "subscribe_viewer" else _req_format(pkt.payload)
                # This stream is dedicated to the subscription; stream frames until
                # the client goes away, then end (don't read further requests).
                if fmt == "transforms":
                    yield from self._stream_transforms(context, pkt.sequence_id)
                else:
                    yield from self._stream_viewer(context, pkt.sequence_id)
                return
            reply = self._dispatch(op, bytes(pkt.payload))
            yield self._wrap(op, pkt.sequence_id, reply)

    def _dispatch(self, op: str, payload: bytes) -> bytes:
        try:
            req = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception:  # noqa: BLE001
            req = {}
        if op == "fastpath_perturb":
            if not self._owner.accepts_input:
                return msgpack.packb(
                    {"ok": False, "error": "capability disabled: accept_input"},
                    use_bin_type=True)
            self._owner.submit_perturb(
                req.get("body", -1), req.get("force", (0, 0, 0)),
                req.get("torque", (0, 0, 0)))
            return msgpack.packb({"ok": True}, use_bin_type=True)
        if op == "fastpath_hello":
            # Serve the model so a mirror can build geometry: bytes + format
            # ("xml"/"mjz"/"mjb"; xml/mjz are decoded in-engine, no MJB version
            # match needed), plus the advertised capabilities the consumer negotiates.
            return msgpack.packb(
                {"ok": True, "scene": self._owner.scene, "ngeom": self._owner.ngeom,
                 "capabilities": list(self._owner.capabilities),
                 "model": self._owner.model_bytes, "format": self._owner.model_format},
                use_bin_type=True)
        return msgpack.packb(
            {"ok": False, "error": f"unknown op {op!r}"}, use_bin_type=True)

    def _stream_viewer(self, context, seq: int):
        # qpos view (format=qpos / subscribe_viewer): consumer runs its own FK.
        last_t = None
        while context.is_active():
            st = self._owner.latest_state()
            if st is not None and st[0] != last_t:
                last_t = st[0]
                t, qpos, qvel = st
                payload = msgpack.packb(
                    {"t": t, "qpos": qpos, "qvel": qvel}, use_bin_type=True)
                yield self._wrap("viewer_frame", seq, payload)
            time.sleep(self._stream_poll_s)

    def _stream_transforms(self, context, seq: int):
        # Transform view (format=transforms): the true mirror payload -- per-body
        # bxpos/bxquat (+ optional camera transforms). The viewer applies them
        # directly and runs zero MuJoCo. Same frames a ZMQ 'geoms' subscriber gets.
        last_f = None
        while context.is_active():
            fr = self._owner.latest_transforms()
            if fr is not None and fr.get("f") != last_f:
                last_f = fr.get("f")
                yield self._wrap("view_frame", seq,
                                 msgpack.packb(fr, use_bin_type=True))
            time.sleep(self._stream_poll_s)

    def _wrap(self, op: str, seq: int, payload: bytes):
        dm_env_rpc_pb2, _grpc_mod, urlab_pb2 = _pb()
        out = urlab_pb2.UrlabPacket(op=op, payload=payload, sequence_id=seq)
        env = dm_env_rpc_pb2.EnvironmentResponse()
        env.extension.Pack(out)
        return env


class OwnerGrpcServer:
    """Runs an :class:`_OwnerServicer` for a FastPathOwner."""

    def __init__(self, owner, *, port: int = 50051, bind: str = "0.0.0.0"):
        self._owner = owner
        self._port = int(port)
        self._bind = bind
        self._server: Any = None
        self.endpoint = f"{bind}:{port}"

    def start(self) -> None:
        import grpc  # noqa: PLC0415

        _dm, dm_env_rpc_pb2_grpc, _u = _pb()
        self._server = grpc.server(
            concurrent.futures.ThreadPoolExecutor(max_workers=8),
            options=[("grpc.max_send_message_length", -1),
                     ("grpc.max_receive_message_length", -1)],
        )
        dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(
            _OwnerServicer(self._owner), self._server)
        self._server.add_insecure_port(f"{self._bind}:{self._port}")
        self._server.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(grace=0.5)
            self._server = None
