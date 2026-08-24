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
mirrors can subscribe + perturb over gRPC too -- one transport everywhere, not
just ZMQ. It speaks the exact same envelope as the UE bridge: a ``UrlabPacket``
(``op``, ``payload`` msgpack, ``sequence_id``) inside ``EnvironmentRequest/
Response.extension``.

Ops served (mirror/VR are capability *consumers*, not modes):
- ``fastpath_hello`` -- scene/ngeom, the advertised capabilities, and the model
  bytes + format (``xml``/``mjz``/``mjb``; xml/mjz are decoded in-engine, so a
  mirror needs no MJB version match).
- ``subscribe`` (``format=render``) -- gated on the ``stream_cameras`` capability;
  a server-stream of the render tier (the true mirror -- per-body
  ``bxpos``/``bxquat`` + optional debug fields, consumer runs zero MuJoCo; op
  ``view_frame``). ``render`` is the only subscription format; there is no
  ``subscribe_viewer`` / ``format=qpos`` tier.
- ``fastpath_perturb`` -- gated on the ``accept_input`` capability; into the
  owner's perturb queue.

Started via ``FastPathOwner.start_grpc_server()``. The same contract is served by
a UE owner, so a mirror is owner-agnostic.
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
            if op == "subscribe":
                # Subscribing to the owner's view is gated on the stream_cameras
                # capability (mirror/VR are consumers of that cap, not modes).
                if not self._owner.streams_view:
                    yield self._wrap(op, pkt.sequence_id, msgpack.packb(
                        {"ok": False, "error": "capability disabled: stream_cameras"},
                        use_bin_type=True))
                    return
                # Only the "render" tier is served (true mirror -- bxpos/bxquat +
                # optional debug, viewer runs no MuJoCo; there is no qpos tier).
                # This stream is dedicated to the subscription; stream frames
                # until the client goes away, then end (don't read further
                # requests).
                #
                # Parse the debug-tier caps off the subscribe request (§8.2:
                # {contacts, overlay, maxcontacts}). These are PER-SUBSCRIPTION:
                # this stream appends exactly the wanted debug fields for ITS OWN
                # frames only (count-capped, zero extra bytes when none asked) --
                # never for any other subscriber, and never onto the shared ZMQ
                # topic. Register the subscription so the owner's aggregate view
                # stays in sync, and unregister on stream end so a disconnect
                # resets the tier for everyone.
                try:
                    sub_req = msgpack.unpackb(
                        bytes(pkt.payload), raw=False, strict_map_key=False)
                except Exception:  # noqa: BLE001
                    sub_req = {}
                caps = self._owner.parse_render_debug_caps(sub_req)
                token = self._owner.register_render_subscriber(caps)
                try:
                    yield from self._stream_render(context, pkt.sequence_id, caps)
                finally:
                    self._owner.unregister_render_subscriber(token)
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
            # Interactive drag intent -> mjv spring; else a raw wrench -> exact force.
            if "refselpos" in req or "localpos" in req or "active" in req:
                self._owner.submit_perturb(
                    req.get("select", -1),
                    req.get("active", True),
                    req.get("localpos", (0, 0, 0)),
                    req.get("refselpos", (0, 0, 0)))
            else:
                self._owner.submit_perturb_force(
                    req.get("body", -1), req.get("force", (0, 0, 0)),
                    req.get("torque", (0, 0, 0)))
            return msgpack.packb({"ok": True}, use_bin_type=True)
        if op == "fastpath_hello":
            # Serve the model so a mirror can build geometry. The ONE hello
            # schema on every transport face (§11) is built by the owner itself
            # (FastPathOwner.hello_reply), so this gRPC face and the ZMQ REP face
            # can never diverge; only the bus differs -- for a gRPC owner the
            # mirror subscribes at grpc://<our gRPC endpoint>.
            ep = self._owner.grpc_endpoint
            return msgpack.packb(
                self._owner.hello_reply(bus=f"grpc://{ep}" if ep else ""),
                use_bin_type=True)
        return msgpack.packb(
            {"ok": False, "error": f"unknown op {op!r}"}, use_bin_type=True)

    def _stream_render(self, context, seq: int, caps=None):
        # Render tier (format=render): the true mirror payload -- per-body
        # bxpos/bxquat (+ optional camera transforms + the debug tier for THIS
        # subscription's negotiated `caps` only). The viewer applies them directly
        # and runs zero MuJoCo. A lean subscriber (caps off) gets exactly the
        # frames a ZMQ 'render' subscriber gets; a debug subscriber additionally
        # gets its own §8.2 fields -- without changing what any other subscriber,
        # or the ZMQ topic, receives.
        last_f = None
        while context.is_active():
            fr = self._owner.render_frame_for_caps(caps)
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
