# URLab render server — how it works (quickstart)

A headless Unreal Engine process that renders MuJoCo scenes photorealistically.
**Your Python process owns the physics** — it steps MuJoCo, pushes the resulting
state each frame, and gets camera images back. UE never simulates; it *mirrors*
your state and renders. Everything below goes through the high-level
`RenderClient` — never hand-roll op dicts, wire keys, or transports.

## 1. Boot the server (empty level, client-driven)

No model is needed at boot. `-URLabFastServe` stands up the bridge + a renderer
on an empty level; the client uploads the model over the wire and the server
(re)builds its renderer from it.

```
UnrealEditor URLabTest.uproject /Game/FastPath/FastPathRender -game \
  -URLabFastServe -URLabFastForcedOnly -URLabFastCameras \
  -RenderOffScreen -nosplash -unattended -stdout
```

Full flag reference: `UnrealRoboticsLab/docs/render_server_flags.md`.

## 2. Drive it from Python

```python
import mujoco
from urlab_bridge.urlab_client import RenderClient, USER_CAMERA, viewer_sync

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with RenderClient.grpc("127.0.0.1", 50051) as rc:
    rc.load_xml("scene.xml")          # server builds/rebuilds its renderer from this
    print(rc.camera_names())          # the model's cameras + "user"

    while running:
        mujoco.mj_step(model, data)                      # you own the physics
        frames = rc.render_mjdata(model, data, cameras=["cam0"])
        bgr = frames["cam0"].to_bgr()                    # HxWx3, ready for cv2
```

- `RenderClient.grpc(host, port)` — no transport strings; `port` is the UE
  ListenPort (50051). `RenderClient.zmq(...)` exists too.
- `load_xml` / `load_mjz` / `load_model` hot-swap the model; the server rebuilds
  its cameras. A path or raw bytes both work.
- `render_mjdata(model, data, cameras=[...])` pushes the body + camera world
  poses from `(model, data)` and returns `{name: CameraFrame}`. Name a subset of
  cameras — each one is a full scene capture, so fewer is cheaper.
- `CameraFrame`: `.to_bgr()` / `.to_rgb()` / `.to_array()`, plus `width`,
  `height`, `dtype`, `frame_id`, `sim_time`.

## 3. The `user` camera (your viewpoint)

Every scene also exposes a **`user` camera** you can point anywhere — typically
at a MuJoCo viewer's free/interactive camera, so UE renders exactly what you're
looking at. It is a *viewer* camera, not a model camera, and lives outside the
model's camera list.

```python
import mujoco.viewer
with mujoco.viewer.launch_passive(model, data) as v:
    scene = viewer_sync.make_scene(model)     # reuse across frames
    while v.is_running():
        mujoco.mj_step(model, data); v.sync()
        pose = viewer_sync.pose_from_passive(v, scene)   # (pos, fwd, up)
        frames = rc.render_mjdata(model, data,
                                  cameras=[USER_CAMERA], user_pose=pose)
        # e.g. paint it back into the viewer as picture-in-picture:
        # v.set_images((viewport_rect, frames[USER_CAMERA].to_rgb()))
```

`viewer_sync.free_camera_pose(model, data, cam)` turns *any* `mujoco.MjvCamera`
(free / tracking / fixed) into `(pos, fwd, up)`, so the same helper feeds the
classic passive viewer, `experimental.studio` native, or its web viewer.

Working end-to-end example: `examples/render_server_viewer.py`.

## 4. Two rendering modes (pick one per server instance)

Same cameras, same API — the difference is who paces capture and whether it
blocks. They **cannot share one instance** (a forced capture stalls the render
thread a smooth stream needs).

| | Forced | Viewer / streaming |
|---|---|---|
| Client call | `render(..., delay=0)` | `render(..., delay=N)` |
| Server boot | **with** `-URLabFastForcedOnly` | **without** it |
| Frame | exact-fresh for the pushed state | latest ring frame, a few substeps stale |
| Feel | deterministic, blocking, not smooth | server-paced, non-blocking, smooth |
| Use | eval / training data | live viewer, patching your view into an app |

## 5. Robustness (what the client handles for you)

The gRPC transport auto-reconnects on a dropped stream, validates that each
reply matches its request (`sequence_id`), keeps the connection alive between
renders, and **fails fast with a clear `ConnectionError`** if nothing is
listening (instead of an opaque hang). A single render that exceeds its
`timeout_ms` raises `URLabTimeoutError` — that's the caller's declared deadline;
catch it to skip one slow frame, the connection stays up.

## TL;DR for a downstream agent

1. Boot the server with `-URLabFastServe -URLabFastForcedOnly -URLabFastCameras`.
2. `with RenderClient.grpc(host, port) as rc: rc.load_xml(scene)`.
3. Each step: `rc.render_mjdata(model, data, cameras=[...])`; drive your own view
   with `user_pose=viewer_sync.free_camera_pose(...)` + `USER_CAMERA`.
4. Never touch transports or op dicts. Catch `URLabTimeoutError` per frame.
