# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Integration tests for the packaged headless render server over gRPC (dm_env_rpc).

Requires the packaged Linux server (UrlLinux.sh) running with:
  -URLabDrive=push -URLabModel=<scene.mjb> -URLabCaps=serve,cameras -RenderOffScreen -nosplash
"""
from __future__ import annotations

import os
import time
from pathlib import Path
import pytest
import numpy as np
import mujoco

from urlab_client import URLabClient, RenderClient
from urlab_client.render_client import CameraFrame

GRPC_PORT = int(os.environ.get("URLAB_GRPC_PORT", "50051"))
GOLDEN_XML = Path(__file__).resolve().parent / "fixtures" / "golden_scene.xml"


def _render_server_available() -> bool:
    """True only if a packaged RENDER server (advertising fastpath_render) answers
    on GRPC_PORT. Skips this integration module otherwise -- e.g. nothing running,
    or a non-render fast-path OWNER squatting the port (which answers hello but has
    no render ops), so the suite stays green without the packaged server."""
    try:
        c = URLabClient("tcp://127.0.0.1", step_port=GRPC_PORT,
                        transport="grpc", recv_timeout_ms=1500)
        try:
            ops = c._rpc("meta", {}).get("ops", [])
            names = [op["name"] if isinstance(op, dict) else op for op in ops]
            return "fastpath_render" in names
        finally:
            c.close()
    except Exception:  # noqa: BLE001 -- unreachable/refused/wrong-server all skip
        return False


pytestmark = pytest.mark.skipif(
    not _render_server_available(),
    reason=f"packaged gRPC render server (fastpath_render) not reachable on :{GRPC_PORT}",
)


@pytest.fixture(scope="module")
def mj_model_data():
    model = mujoco.MjModel.from_xml_path(str(GOLDEN_XML))
    data = mujoco.MjData(model)
    return model, data


def test_grpc_hello_and_meta():
    """Verify raw RPC handshake and op discovery over dm_env_rpc / gRPC."""
    client = URLabClient("tcp://127.0.0.1", step_port=GRPC_PORT, transport="grpc", recv_timeout_ms=10_000)
    try:
        reply = client._rpc("hello", {"observations": "standard"})
        assert reply["op"] == "hello_ok"
        assert "session_id" in reply
        print(f"[gRPC] hello_ok session_id={reply['session_id']}")

        meta_reply = client._rpc("meta", {})
        assert meta_reply["op"] == "meta_ok"
        assert "ops" in meta_reply
        op_names = [op["name"] if isinstance(op, dict) else op for op in meta_reply["ops"]]
        assert "fastpath_render" in op_names
        print(f"[gRPC] meta_ok supported ops count={len(op_names)}")
    finally:
        client.close()


def test_grpc_camera_names():
    """Verify camera discovery via RenderClient over gRPC."""
    with RenderClient("tcp://127.0.0.1", step_port=GRPC_PORT, transport="grpc", recv_timeout_ms=10_000) as rc:
        names = rc.camera_names()
        assert len(names) > 0, "Expected at least one camera"
        print(f"[gRPC] camera_names: {names}")


def test_grpc_render_fresh_frame(mj_model_data):
    """Verify synchronous (delay=0) fastpath render over gRPC."""
    model, data = mj_model_data
    mujoco.mj_step(model, data)

    with RenderClient("tcp://127.0.0.1", step_port=GRPC_PORT, transport="grpc", recv_timeout_ms=10_000) as rc:
        cam_names = rc.camera_names()
        assert len(cam_names) > 0
        cam_name = cam_names[0]
        frames = rc.render_mjdata(model, data, cameras=[cam_name], delay=0)
        assert cam_name in frames
        frame = frames[cam_name]
        assert isinstance(frame, CameraFrame)
        assert frame.name == cam_name
        assert frame.width > 0 and frame.height > 0
        assert frame.dtype in ("bgra8", "float32")
        arr = frame.to_array()
        assert arr.shape == (frame.height, frame.width, 4)
        bgr = frame.to_bgr()
        assert bgr.shape == (frame.height, frame.width, 3)
        assert bgr.dtype == np.uint8
        # Ensure frame is rendered
        print(f"[gRPC] Rendered frame: {frame.width}x{frame.height} {frame.dtype}, bgr mean={bgr.mean():.2f}")


def test_grpc_render_continuous_stepping(mj_model_data):
    """Step physics and render consecutive frames over gRPC."""
    model, data = mj_model_data
    rts = []
    with RenderClient("tcp://127.0.0.1", step_port=GRPC_PORT, transport="grpc", recv_timeout_ms=10_000) as rc:
        cam_names = rc.camera_names()
        assert len(cam_names) > 0
        cam_name = cam_names[0]
        for i in range(25):
            data.ctrl[:] = 0.05 * np.sin(np.arange(model.nu) + i * 0.1)
            mujoco.mj_step(model, data)
            t0 = time.perf_counter()
            frames = rc.render_mjdata(model, data, cameras=[cam_name], delay=0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rts.append(elapsed_ms)
            assert cam_name in frames
            assert frames[cam_name].frame_id >= 0

        mean_ms = float(np.mean(rts))
        p50_ms = float(np.percentile(rts, 50))
        p95_ms = float(np.percentile(rts, 95))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        print(f"\n[gRPC Packaged Benchmark] 25 frames: mean={mean_ms:.2f}ms, p50={p50_ms:.2f}ms, p95={p95_ms:.2f}ms -> {fps:.1f} FPS")
        assert mean_ms < 100.0, f"Mean latency {mean_ms:.2f}ms exceeded 100ms threshold"


def test_grpc_hotswap_xml(mj_model_data):
    """Verify live model hot-swap over gRPC using raw XML (in-engine decode)."""
    model, data = mj_model_data
    with RenderClient("tcp://127.0.0.1", step_port=GRPC_PORT, transport="grpc", recv_timeout_ms=10_000) as rc:
        rc.load_xml(str(GOLDEN_XML))
        cam_names = rc.camera_names()
        assert len(cam_names) > 0
        cam_name = cam_names[0]
        mujoco.mj_step(model, data)
        frames = rc.render_mjdata(model, data, cameras=[cam_name], delay=0)
        assert cam_name in frames
        assert frames[cam_name].width > 0
        print(f"[gRPC] XML hot-swap successful: camera={cam_name}, frame={frames[cam_name].width}x{frames[cam_name].height}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
