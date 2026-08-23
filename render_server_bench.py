"""Render-server latency: puppet mode, Python owns mj_step, pushes fresh state,
and needs the rendered frame back. Measures the EXACT request->frame time two
ways, over zmq and shm:

  A) render:sync  -- push qpos + force an immediate render, frame returned IN the
     step reply (one round-trip). Large frame in reply may exceed the shm RPC
     region and fall back to zmq -- we detect + report that.
  B) fresh-stream -- push qpos (small reply), then get_camera(fresh) waits for the
     frame with frame_id >= this step, delivered over the dedicated camera ring.

Reports mean/p50/p95/max per (strategy, transport), plus decoded frame size and
whether the frame is confirmed fresh (frame_id matches the pushed step).
"""
from __future__ import annotations
import argparse
import base64
import statistics
import time
from typing import List

import mujoco
import numpy as np
from urlab_client import URLabClient

GOLDEN = "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/URLab_Bridge/tests/fixtures/golden_scene.xml"
GOLDEN_LEVEL = "/Game/Levels/URLabGoldenTestCamera"


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, int(round(p * (len(s) - 1))))
    return s[k]


def summarize(name: str, ms: List[float], extra: str = "") -> dict:
    row = {
        "name": name, "n": len(ms),
        "mean": statistics.mean(ms) if ms else float("nan"),
        "p50": pct(ms, 0.50), "p95": pct(ms, 0.95), "max": max(ms) if ms else float("nan"),
    }
    print(f"  {name:22} n={row['n']:3d}  mean={row['mean']:6.2f}ms  p50={row['p50']:6.2f}  "
          f"p95={row['p95']:6.2f}  max={row['max']:6.2f}  {extra}")
    return row


def run_transport(transport: str, cam: str, iters: int, warmup: int) -> List[dict]:
    print(f"\n########## transport={transport} ##########")
    c = URLabClient("tcp://127.0.0.1", step_port=5559, mujoco_version_check=False,
                    recv_timeout_ms=120_000, transport=transport)
    c.connect()
    # Bring up the golden camera level + PIE only when no manager is live yet.
    # Reloading the level while PIE is already running stops PIE, so an
    # already-running golden session is reused rather than restarted.
    if not c.manager_present:
        c.sim.start(level_path=GOLDEN_LEVEL, raise_on_failure=False, timeout_s=90.0)
        time.sleep(0.5)
    c.local_model = False
    c.connect()
    names = c.camera_names()
    cam = cam if cam in names else (names[0] if names else "")
    c.runtime.set_mode("statepushed")

    m = mujoco.MjModel.from_xml_path(GOLDEN)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def push_payload():
        return {"qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "time": float(d.time)}

    rows = []

    # ---- A: render:sync, streaming OFF (pure on-demand render server) ----
    c.runtime.set_camera_streaming({cam: {"zmq": False, "shm": False}})
    time.sleep(0.3)
    for _ in range(warmup):
        d.ctrl[:] = 0.02 * np.sin(np.arange(m.nu))
        mujoco.mj_step(m, d)
        c._rpc("step", {**push_payload(), "include_cameras": True,
                        "render": "sync", "camera_timeout_ms": 2000})

    # ---- A: render:sync, frame in reply ----
    a_ms: List[float] = []
    fresh_ok = 0
    frame_bytes = 0
    fell_back = False
    for i in range(iters):
        d.ctrl[:] = 0.02 * np.sin(np.arange(m.nu) + i * 0.05)
        mujoco.mj_step(m, d)
        t0 = time.perf_counter()
        reply = c._rpc("step", {**push_payload(), "include_cameras": True,
                                "render": "sync", "camera_timeout_ms": 2000})
        t1 = time.perf_counter()
        cams = reply.get("cameras") or {}
        info = cams.get(cam) or (next(iter(cams.values())) if cams else None)
        if info:
            step_fid = reply.get("frame_id")
            cam_fid = info.get("frame_id")
            if step_fid is not None and cam_fid is not None and cam_fid >= step_fid:
                fresh_ok += 1
            data = info.get("data")
            if data is None:
                px = info.get("pixels")
                data = base64.b64decode(px) if px else None
            if data is not None and frame_bytes == 0:
                frame_bytes = len(data)
        if reply.get("wrong_transport") or reply.get("reply_too_large"):
            fell_back = True
        a_ms.append((t1 - t0) * 1000.0)
    rows.append(summarize(f"A render:sync",
                          a_ms, f"fresh={fresh_ok}/{iters} frame~{frame_bytes//1024}KB"
                          + ("  [reply fell back to zmq]" if fell_back else "")))

    # ---- B: push + fresh get_camera over the stream (streaming ON) ----
    c.runtime.set_camera_streaming({cam: {"zmq": True, "shm": True}})
    time.sleep(0.5)
    for _ in range(warmup):
        d.ctrl[:] = 0.02 * np.sin(np.arange(m.nu))
        mujoco.mj_step(m, d)
        c._rpc("step", push_payload())
        c.get_camera(cam, fresh=True, timeout_s=2.0)
    b_ms: List[float] = []
    b_fresh = 0
    for i in range(iters):
        d.ctrl[:] = 0.02 * np.sin(np.arange(m.nu) + i * 0.07)
        mujoco.mj_step(m, d)
        t0 = time.perf_counter()
        reply = c._rpc("step", push_payload())
        step_fid = reply.get("frame_id")
        frame = c.get_camera(cam, fresh=True, timeout_s=2.0)
        t1 = time.perf_counter()
        if frame is not None:
            b_fresh += 1
        b_ms.append((t1 - t0) * 1000.0)
    rows.append(summarize(f"B push+fresh-stream", b_ms, f"got={b_fresh}/{iters}"))

    # ---- C: render:async (frame in reply, latest-completed, no wait) ----
    c.runtime.set_camera_streaming({cam: {"zmq": False, "shm": False}})
    time.sleep(0.3)
    for _ in range(warmup):
        d.ctrl[:] = 0.02 * np.sin(np.arange(m.nu))
        mujoco.mj_step(m, d)
        c._rpc("step", {**push_payload(), "include_cameras": True,
                        "render": "async", "camera_timeout_ms": 2000})
    cp_ms: List[float] = []
    cp_got = 0
    cp_fresh = 0
    cp_bytes = 0
    for i in range(iters):
        d.ctrl[:] = 0.02 * np.sin(np.arange(m.nu) + i * 0.09)
        mujoco.mj_step(m, d)
        t0 = time.perf_counter()
        reply = c._rpc("step", {**push_payload(), "include_cameras": True,
                                "render": "async", "camera_timeout_ms": 2000})
        t1 = time.perf_counter()
        cams = reply.get("cameras") or {}
        info = cams.get(cam) or (next(iter(cams.values())) if cams else None)
        if info:
            cp_got += 1
            step_fid = reply.get("frame_id")
            cam_fid = info.get("frame_id")
            if step_fid is not None and cam_fid is not None and cam_fid >= step_fid:
                cp_fresh += 1
            data = info.get("data")
            if data is not None and cp_bytes == 0:
                cp_bytes = len(data)
        cp_ms.append((t1 - t0) * 1000.0)
    rows.append(summarize(f"C render:async", cp_ms,
                          f"got={cp_got}/{iters} fresh={cp_fresh}/{iters} frame~{cp_bytes//1024}KB"))

    c.close()
    for r in rows:
        r["transport"] = transport
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--camera", default="head")
    ap.add_argument("--transports", default="zmq,shm")
    args = ap.parse_args()
    all_rows = []
    for t in [x.strip() for x in args.transports.split(",") if x.strip()]:
        all_rows.extend(run_transport(t, args.camera, args.iters, args.warmup))
    print("\n===== render-server request->frame latency (ms) =====")
    print(f"{'transport':10} {'strategy':22} {'mean':>7} {'p50':>7} {'p95':>7} {'max':>7}")
    for r in all_rows:
        print(f"{r['transport']:10} {r['name']:22} {r['mean']:7.2f} {r['p50']:7.2f} "
              f"{r['p95']:7.2f} {r['max']:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
