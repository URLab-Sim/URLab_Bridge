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

"""`urlab-test` — dump 10 s of the legacy PUB/SUB state stream.

Diagnostic tool for the legacy streaming protocol (state on port 5555).
For the modern remote-stepping handshake check, use `urlab-ping`."""

from __future__ import annotations

import argparse
import logging
import struct
import time

from .common import add_common_args, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="urlab-test",
        description="Dump the legacy PUB/SUB state stream for 10 s.",
    )
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging()

    import zmq

    logger.info(f"Testing {args.state_ep} (prefix: {args.prefix})")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.state_ep)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 5000)

    count = 0
    start = time.time()
    try:
        while time.time() - start < 10:
            try:
                topic = sub.recv_string()
                if sub.getsockopt(zmq.RCVMORE):
                    payload = sub.recv()
                    if "/joint/" in topic and len(payload) == 16:
                        jid, p, v, _ = struct.unpack("<Ifff", payload)
                        print(f"  [{topic}] ID:{jid} Pos:{p:.3f} Vel:{v:.3f}")
                        count += 1
            except zmq.Again:
                if count == 0:
                    logger.warning("No data yet...")
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        ctx.term()
    logger.info(f"Received {count} messages in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
