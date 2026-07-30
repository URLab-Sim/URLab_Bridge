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

"""`urlab-test` — dump 10 s of the PUB/SUB `state/full` snapshot stream.

Diagnostic tool for the streaming protocol (state on port 5555). For the
modern remote-stepping handshake check, use `urlab-ping`."""

from __future__ import annotations

import argparse
import logging
import time

from .common import add_common_args, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="urlab-test",
        description="Dump the PUB/SUB state/full snapshot stream for 10 s.",
    )
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging()

    import msgpack
    import zmq

    logger.info(f"Testing {args.state_ep} (prefix: {args.prefix})")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.state_ep)
    sub.setsockopt(zmq.SUBSCRIBE, b"state/full")
    sub.setsockopt(zmq.RCVTIMEO, 5000)

    count = 0
    start = time.time()
    try:
        while time.time() - start < 10:
            try:
                sub.recv()  # topic frame ("state/full"), discard
                if not sub.getsockopt(zmq.RCVMORE):
                    continue
                snap = msgpack.unpackb(sub.recv(), raw=False, strict_map_key=False)
                arts = snap.get("arts") or {}
                for prefix, block in arts.items():
                    if args.prefix and prefix != args.prefix:
                        continue
                    qpos = block.get("qpos") or []
                    print(f"  [{prefix}] step:{snap.get('step')} "
                          f"qpos({len(qpos)}):"
                          f"{[round(float(x), 3) for x in qpos[:8]]}")
                count += 1
            except zmq.Again:
                if count == 0:
                    logger.warning("No data yet...")
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        ctx.term()
    logger.info(f"Received {count} snapshots in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
