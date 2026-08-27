#!/usr/bin/env python3
# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Show the live camera feeds a UE fast-path render server publishes.

A fast-path renderer launched with ``-URLabCaps=...,cameras`` (formerly
``-URLabFastCameras``) renders each MJB camera
and publishes its frames over ZMQ: camera ``i`` on ``tcp://<host>:<base+i>``, as
a 2-frame multipart message ``[topic ][40-byte meta + BGRA8 pixels]``. This
subscribes to the whole port range and shows each camera in its own OpenCV
window, labelled by the camera's canonical name (the ZMQ topic).

Run it alongside run_fastpath_demo.py (the owner) and the UE renderer:
    python scripts/show_fastpath_cameras.py --host 127.0.0.1 --base-port 5600
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("this viewer needs opencv-python (pip install opencv-python)")
import zmq

# 40-byte v2 header: magic, version, frame_id, sim_time, width, height, capture.
_META_V2 = struct.Struct("<IIQdIId")
_META_V1 = struct.Struct("<IIQdII")  # 32-byte legacy (no capture time)
_MAGIC = 0x314D4355  # 'UCM1'


def parse_frame(payload: bytes):
    """Return (name-agnostic) BGR image from a camera payload, or None."""
    if len(payload) < _META_V1.size:
        return None
    magic = struct.unpack_from("<I", payload, 0)[0]
    if magic != _MAGIC:
        return None
    ver = struct.unpack_from("<I", payload, 4)[0]
    meta = _META_V2 if ver >= 2 else _META_V1
    fields = meta.unpack_from(payload, 0)
    w, h = fields[4], fields[5]
    pixels = payload[meta.size :]
    if w <= 0 or h <= 0 or len(pixels) < w * h * 4:
        return None
    bgra = np.frombuffer(pixels[: w * h * 4], dtype=np.uint8).reshape((h, w, 4))
    # UE FColor is BGRA in memory; dropping alpha yields BGR, which is exactly
    # what OpenCV wants -- no channel swap needed.
    return bgra[:, :, :3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Show UE fast-path camera feeds")
    ap.add_argument("--host", default="127.0.0.1", help="renderer host")
    ap.add_argument("--base-port", type=int, default=5600, help="first camera port")
    ap.add_argument("--max-cams", type=int, default=12,
                    help="number of consecutive ports to subscribe to")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 4)
    sub.setsockopt(zmq.SUBSCRIBE, b"")  # all camera topics
    for i in range(args.max_cams):
        sub.connect(f"tcp://{args.host}:{args.base_port + i}")
    print(f"[cams] subscribing to {args.host}:{args.base_port}-"
          f"{args.base_port + args.max_cams - 1}")
    print("[cams] windows open as feeds arrive. Focus a window and press q to quit.")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    windows: set[str] = set()
    try:
        while True:
            # Drain everything queued this tick, keeping the newest per topic.
            latest: dict[str, bytes] = {}
            while dict(poller.poll(timeout=30)).get(sub) == zmq.POLLIN:
                topic = sub.recv().decode("utf-8", "replace").strip()
                payload = sub.recv()
                latest[topic] = payload
                # don't spin forever if a flood arrives; one pass is enough
                if len(latest) >= args.max_cams:
                    break
            for topic, payload in latest.items():
                img = parse_frame(payload)
                if img is None:
                    continue
                if topic not in windows:
                    cv2.namedWindow(topic, cv2.WINDOW_NORMAL)
                    windows.add(topic)
                cv2.imshow(topic, img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        sub.close(0)


if __name__ == "__main__":
    main()
