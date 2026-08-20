# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Render one MuJoCo scene across a POOL of render-server instances in parallel.

An external orchestrator launches the instances (across the network and/or
several on one host, each with a distinct ``-URLabDmEnvPort=``); this example
just attaches to them and fans the cameras out. Give it the pool either way:

    # explicit endpoints (what a CLI passes)
    uv run python examples/render_pool_example.py \
        --xml ../mujoco_menagerie/aloha/scene.xml \
        --endpoints 127.0.0.1:50051,127.0.0.1:50052

    # or a JSON config file: {"instances": [{"host":"h","port":50051}, ...]}
    uv run python examples/render_pool_example.py --xml scene.xml --pool-config pool.json

It loads the scene into every instance, renders all cameras split across the
pool, prints the per-instance split + the merged frames, and times the pool
render against a single instance for comparison.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from urlab_client import RenderClient, RenderPool, parse_endpoints


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True, help="MuJoCo scene to load (menagerie ok)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--endpoints", help="comma/space list, e.g. 'h1:50051,h2:50051'")
    src.add_argument("--pool-config", help="path to a JSON pool config")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pool = (RenderPool.from_endpoints(args.endpoints) if args.endpoints
            else RenderPool.from_config(args.pool_config))
    with pool:
        print(f"pool: {pool.endpoints}")
        pool.load_xml(args.xml)                 # broadcast to all instances at once
        cams = pool.camera_names()
        print(f"{len(cams)} cameras across {len(pool)} instance(s): {cams}")

        # Show the automatic split (same logic the pool uses each render).
        buckets = pool._distribute(cams)  # noqa: SLF001 - illustrative
        for ep, b in zip(pool.endpoints, buckets):
            print(f"  {ep}: {b}")

        pool_ms, single_ms = [], []
        for _ in range(args.frames):
            mujoco.mj_step(model, data)
            t0 = time.perf_counter()
            frames = pool.render_mjdata(model, data)          # all cams, parallel
            pool_ms.append((time.perf_counter() - t0) * 1000.0)
        assert set(frames) == set(cams), f"missing cameras: {set(cams) - set(frames)}"

        # Baseline: the same cameras from ONE instance (sequential internally).
        one = RenderClient.grpc(*_split(pool.endpoints[0]))
        try:
            for _ in range(args.frames):
                mujoco.mj_step(model, data)
                t0 = time.perf_counter()
                one.render_mjdata(model, data, cameras=cams)
                single_ms.append((time.perf_counter() - t0) * 1000.0)
        finally:
            one.close()

    n = len(pool.endpoints)
    print(f"\n{len(cams)} cameras, {args.frames} frames")
    print(f"  pool   ({n} inst): mean {np.mean(pool_ms):.1f} ms")
    print(f"  single (1 inst):   mean {np.mean(single_ms):.1f} ms")
    print(f"  speedup: {np.mean(single_ms) / max(1e-6, np.mean(pool_ms)):.2f}x")


def _split(ep: str):
    host, _, port = ep.rpartition(":")
    return host, int(port)


if __name__ == "__main__":
    main()
