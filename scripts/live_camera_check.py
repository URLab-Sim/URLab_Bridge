# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Live camera-feed proof / API lockdown for the async streaming redesign.

Drives a running URLab editor: imports a scene, spawns it, starts PIE, then
steps in the chosen mode and pulls camera frames off the async SHM/ZMQ stream.
The whole point is to (1) prove real pixels arrive and (2) print the EXACT
canonical camera names UE registers, so the names can be handed to another
machine/agent verbatim.

Prereq: a UE editor with the URLab plugin is running and listening on
``--host`` (default tcp://127.0.0.1, step port 5559).

Examples:
    uv run python scripts/live_camera_check.py \
        --xml /c/Users/jonat/Downloads/reaf_test/reaf_test/base_scene_ue.xml \
        --local-xml /c/Users/jonat/Downloads/reaf_test/reaf_test/base_scene.xml \
        --mode puppet --query latest --frames 60 --save-dir /tmp/urlab_cam_check

    # just enumerate cameras against whatever scene is already loaded:
    uv run python scripts/live_camera_check.py --names-only
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional


def _log(msg: str) -> None:
    print(msg, flush=True)


def enumerate_cameras(client) -> List[tuple]:
    """Return [(location, name, view), ...] across articulations + globals."""
    out = []
    for art in client.articulations.values():
        for name, view in art.cameras.items():
            out.append((f"art[{art.prefix}]", name, view))
    for name, view in client.global_cameras.items():
        out.append(("GLOBAL", name, view))
    return out


def print_cameras(client) -> List[str]:
    cams = enumerate_cameras(client)
    _log("=" * 70)
    _log(f"DISCOVERED CAMERAS ({len(cams)}):")
    for loc, name, v in cams:
        ep = getattr(v, "_zmq_endpoint", None)
        topic = getattr(v, "_zmq_topic", None)
        _log(f"  {loc:>16}  name={name!r}")
        _log(f"  {'':>16}  mode={v.mode} res={v.resolution} zmq_ep={ep} topic={topic!r}")
    _log("=" * 70)
    return [name for _, name, _ in cams]


def setup_scene_full(client, args) -> None:
    """Author a level + import + spawn the BP. Opt-in via --setup-scene.
    Mirrors the proven integration conftest. Most of the time you DON'T want
    this -- you author/light the scene by hand in the editor and just point
    this script at the BP that's already in the level."""
    if client.manager_present:
        _log("Existing manager present; stopping current sim first...")
        try:
            client.sim.stop()
            time.sleep(0.5)
            client.connect()
        except Exception as exc:  # noqa: BLE001
            _log(f"  sim.stop failed (continuing): {exc}")

    _log(f"Creating level {args.level_name!r}...")
    client.scene.create_level(args.level_name, force_overwrite=True)
    client.scene.ensure_manager()
    _log(f"Importing XML into UE: {args.xml}")
    bp = client.scene.import_xml(args.xml, force_reimport=True)
    _log(f"Spawning actor {args.actor_id!r} from {bp}")
    client.scene.spawn_actor(bp, actor_id=args.actor_id)
    client.scene.save_level()


def ensure_running(client, args) -> None:
    """Bring the editor to a PIE-on state for whatever scene is loaded.

    Default path: the BP (e.g. base_scene_ue0) is already placed + lit in the
    current level by hand. We do NOT create a level, import, or spawn -- we
    just start PIE if it isn't already running. Pass --setup-scene to do the
    full author/import/spawn flow instead.
    """
    if args.setup_scene:
        setup_scene_full(client, args)

    if not client.manager_present:
        _log("PIE not running; starting it on the current level...")
        result = client.sim.start(raise_on_failure=False, timeout_s=90.0)
        # NB: never print `result` directly -- its handshake_payload embeds the
        # full compiled mjb (tens-hundreds of MB for mesh scenes).
        _log(f"  PIE start: state={result.state} "
             f"compile_error={result.compile_error[:200]!r}")
        time.sleep(0.5)
    else:
        _log("Manager already present (PIE running); using current scene as-is.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    ap.add_argument("--step-port", type=int, default=5559)
    ap.add_argument("--transport", choices=["zmq", "shm"], default="zmq")
    ap.add_argument("--mode", choices=["puppet", "direct", "live"], default="puppet")
    ap.add_argument("--query", choices=["latest", "fresh"], default="latest")
    ap.add_argument("--camera-timeout", type=float, default=0.3)
    ap.add_argument("--rate", type=float, default=30.0,
                    help="target step rate in Hz for the measured loop (0 = unpaced)")
    ap.add_argument("--warmup-s", type=float, default=5.0,
                    help="seconds to wait for streams to warm up before measuring")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--cameras", default="all",
                    help="comma-separated canonical names, or 'all'")
    ap.add_argument("--setup-scene", action="store_true",
                    help="author level + import + spawn the BP (default: OFF; "
                         "use the BP already placed/lit in the current level)")
    ap.add_argument("--xml", default="",
                    help="UE-side XML to import (only with --setup-scene)")
    ap.add_argument("--local-xml", default="",
                    help="plain MJCF for the local MuJoCo model (puppet mode)")
    ap.add_argument("--level-name", default="URLabCamCheck")
    ap.add_argument("--actor-id", default="CamCheckScene")
    ap.add_argument("--names-only", action="store_true",
                    help="ensure PIE, then just print camera names and exit")
    ap.add_argument("--save-dir", default="/tmp/urlab_cam_check")
    args = ap.parse_args()

    from urlab_client import URLabClient, StepMode

    _log(f"Connecting to {args.host} (transport={args.transport})...")
    client = URLabClient(
        args.host,
        step_port=args.step_port,
        mujoco_version_check=False,
        recv_timeout_ms=120_000,
        transport=args.transport,
    )

    try:
        client.connect()
    except ValueError as exc:
        _log(f"  first discover ValueError (ok): {exc}")

    # Puppet needs a local model; load it and hand it to the client.
    model = data = None
    if args.mode == "puppet" and not args.names_only:
        if not args.local_xml:
            _log("ERROR: --mode puppet requires --local-xml (plain MJCF).")
            return 2
        import mujoco
        _log(f"Loading local MuJoCo model: {args.local_xml}")
        model = mujoco.MjModel.from_xml_path(args.local_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        client.model = model
        client.data = data

    ensure_running(client, args)

    # Re-discover now that PIE is on, so the client learns the cameras.
    # Skip MJB load to avoid any version-mismatch ValueError.
    client.local_model = False
    try:
        client.connect()
    except ValueError as exc:
        _log(f"  post-PIE discover ValueError (ok): {exc}")

    names = print_cameras(client)
    if args.names_only:
        return 0 if names else 1
    if not names:
        _log("FAIL: no cameras discovered. Check the scene has <camera> elements "
             "and that import/spawn succeeded.")
        return 1

    _log(f"Setting step mode -> {args.mode}")
    client.runtime.set_mode(args.mode)

    # Resolve requested cameras to the canonical names actually present.
    if args.cameras.strip().lower() == "all":
        want = names
    else:
        want = [c.strip() for c in args.cameras.split(",") if c.strip()]
        missing = [c for c in want if c not in names]
        if missing:
            _log(f"WARNING: requested cameras not found and will be dropped: {missing}")
            _log(f"         available: {names}")
        want = [c for c in want if c in names]
    if not want:
        _log("FAIL: no requested cameras matched the discovered set.")
        return 1

    # Map name -> view for direct cache reads.
    view_by_name = {name: v for _, name, v in enumerate_cameras(client)}

    os.makedirs(args.save_dir, exist_ok=True)

    # Warm up: ZMQ PUB/SUB is a slow joiner and the first render+readback takes
    # a moment, so the first frames after subscribing are dropped. warmup_cameras
    # is the locked-down "block until ready" setup: it (re)enables streaming and
    # waits for every requested camera's first frame, raising a loud error with
    # the missing names instead of leaving silent None images.
    _log(f"Warming up camera streams (block until ready, <= {args.warmup_s}s)...")
    try:
        ready = client.warmup_cameras(want, timeout_s=args.warmup_s)
        _log(f"  warm-up: {len(ready)}/{len(want)} cameras ready")
    except TimeoutError as exc:
        _log(f"  WARN: {exc}")

    _log(f"Stepping {args.frames} frames (query={args.query}, rate={args.rate}Hz)...")
    period = 1.0 / args.rate if args.rate > 0 else 0.0

    received: Dict[str, int] = {n: 0 for n in want}
    last_fid: Dict[str, Optional[int]] = {n: None for n in want}
    stale_count = 0
    saved: Dict[str, bool] = {n: False for n in want}
    t_start = time.time()

    for i in range(args.frames):
        if args.mode == "puppet":
            # Gentle motion so frames visibly change; tweak a couple of joints.
            import numpy as np
            data.qpos[:] = data.qpos  # keep current
            if model.nq > 0:
                data.qpos[0] = 0.3 * np.sin(i * 0.1)
            import mujoco
            mujoco.mj_forward(model, data)

        t0 = time.time()
        # step() is physics-only now; cameras are decoupled.
        reply = client.step(n_steps=0 if args.mode == "puppet" else 1)
        t1 = time.time()

        # Decoupled getter: ask for each camera's frame by canonical name.
        line = []
        for n in want:
            frame = client.get_camera(
                n, fresh=(args.query == "fresh"), timeout_s=args.camera_timeout
            )
            fid = view_by_name[n].frame_id
            if frame is not None:
                received[n] += 1
                last_fid[n] = fid
                if not saved[n]:
                    _save_frame(args.save_dir, n, frame)
                    saved[n] = True
            line.append(f"{n.split('/')[-1]}={'Y' if frame is not None else '.'}(fid={fid})")

        if i % 10 == 0 or i == args.frames - 1:
            _log(f"  frame {i:3d}: step={(t1-t0)*1000:6.1f}ms "
                 f"reply_fid={reply.get('frame_id')} | " + " ".join(line))

        if period:
            slack = period - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)

    dt = time.time() - t_start
    _log("=" * 70)
    _log("RESULT")
    _log(f"  {args.frames} steps in {dt:.2f}s -> {args.frames/dt:.1f} steps/s")
    ok = True
    for n in want:
        got = received[n]
        status = "OK" if got > 0 else "NO FRAMES"
        if got == 0:
            ok = False
        _log(f"  {status:>9}  {n}  frames={got}/{args.frames} last_frame_id={last_fid[n]}")
    _log("=" * 70)
    if ok:
        _log("PASS: live camera feeds confirmed. Saved first frames to "
             f"{args.save_dir}")
        _log("\nLocked-down camera API (canonical names below):")
        _log(f"  client.runtime.set_mode({args.mode!r})")
        _log("  client.warmup_cameras()                 # block until every camera streams")
        _log(f"  client.step(n_steps={'0' if args.mode=='puppet' else '1'})"
             "                       # physics only; cameras are decoupled")
        _log(f"  img = client.get_camera({want[0]!r})")
        _log("  #   add fresh=True to sync the frame to the latest step's state")
        _log(f"  client.camera_names() -> {want}")
    else:
        _log("FAIL: one or more cameras produced no frames. See diagnostics above.")
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass
    return 0 if ok else 1


def _save_frame(save_dir: str, name: str, frame) -> None:
    safe = name.replace("/", "_")
    path = os.path.join(save_dir, f"{safe}.png")
    try:
        import numpy as np
        arr = frame
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            # depth -> normalize for viewing
            a = arr.astype(np.float32)
            a = (a - a.min()) / (a.ptp() + 1e-9) * 255.0
            arr = a.astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(arr).save(path)
        except Exception:
            np.save(path.replace(".png", ".npy"), arr)
            path = path.replace(".png", ".npy")
        _log(f"    saved {path}  shape={frame.shape} dtype={frame.dtype}")
    except Exception as exc:  # noqa: BLE001
        _log(f"    save failed for {name}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
