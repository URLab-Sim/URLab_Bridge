# Render client quickstart — get camera renders from Python

Render photorealistic camera images from a MuJoCo scene without running Unreal
yourself. **Your Python process owns the physics** — it steps MuJoCo and pushes
each frame's state; a headless Unreal *render server* mirrors that state and
returns rendered camera images. UE never simulates.

Everything goes through the high-level `RenderClient` — never hand-roll op dicts,
wire keys, or transports.

> **Flags note (Aug 2026):** the render server's launch surface was rewritten by
> the "Render migration" (W18 / commit `2da60f6`). The old `-URLabFast*` flags were
> **deleted** and are now silently ignored — a boot with them binds no listener and
> the server never serves. This doc uses the current five-flag surface
> (`-URLabDrive` / `-URLabCaps` / `-URLabModel` / `-URLabScene` / `-URLabNet`); the
> canonical grammar lives in `Source/URLab/Public/MuJoCo/Fast/MjLauncherFlags.h`.

---

## 1. Start the render server (once)

Use the packaged/staged build. Boot it *client-driven* (no model at boot — the
client uploads one over the wire) with `-URLabDrive=await`:

```bash
cd /home/buzz/Documents/urlab_debug
URLabTest/Saved/StagedBuilds/Linux/URLabTest.sh /Game/FastPath/FastPathRender \
    -URLabDrive=await -URLabCaps=serve,cameras -URLabScene=cammax=0 \
    -RenderOffScreen -nosplash -unattended -stdout
```

| flag | why |
|------|-----|
| `-URLabDrive=await` | come up with **no** model; the gRPC/ZMQ listeners bind immediately and wait for the client to `load_*`. (was `-URLabFastServe`) |
| `-URLabCaps=serve,cameras` | `serve` stands up the render bridge — **required on the `await` path**, serve is *not* implicit there (without it no manager is created and `:50051` never binds); `cameras` builds + streams the capturing cameras (without it `render()` returns nothing). (was `-URLabFastCameras`; serve used to be implicit) |
| `-URLabScene=cammax=0` | don't cap camera height (the default caps model cameras at 480px; the `user` cam is unaffected). (was `-URLabFastCamMaxHeight=0`) |
| `-RenderOffScreen` | headless. |
| `/Game/FastPath/FastPathRender` | the dedicated **lit** render map (SkyLight + reflections). An unlit map leaves metallics looking flat / dark. |

It listens on **gRPC 50051** and **ZMQ 5559** by default (force gRPC's port with
`-URLabNet=grpc=50051`). To run several servers on one host, give each a distinct
`-URLabNet=grpc=<n>,index=<n>`.

Ready when the log prints:
```
LogURLabDmEnvRpc: dm_env_rpc gRPC server listening on 0.0.0.0:50051
LogURLabNet: UURLabBridgeServer: control RPC transport 'dm_env_rpc' bound
```

### Two rendering modes — pick one per instance

Same cameras, same API. The difference is who paces capture and whether it blocks.
A single server does **one** mode — a forced capture stalls the render thread a
smooth stream needs, so they can't share an instance. The mode is the **Drive**:

| | **Forced** (eval) | **Async / streaming** (viewer) |
|---|---|---|
| Server Drive | `-URLabDrive=push` | `-URLabDrive=await` (or any non-`push`) |
| Cameras | capture **only** on a `render()` request | capture **continuously** into a ring |
| Client call | `render_mjdata(..., delay=0)` | `render_mjdata(..., delay=N)` |
| Frame | exact-fresh for the pushed state | latest ring frame, a few substeps stale |
| Feel | deterministic, blocking, not smooth | server-paced, non-blocking, smooth |
| Use for | eval / training data | live viewer, patching a view into an app |

The `await` boot above is **async-capable** (continuous capture) — you can still
pass `delay=0` for an exact-fresh forced frame, or `delay=N` for a smooth ring
frame. For a **forced-only** eval server (lowest latency, no ring), boot with
`-URLabDrive=push`; preloading the model with `-URLabModel=` is the usual pairing
(push serves `fastpath_render` and serve is implicit, but still list `cameras`):

```bash
URLabTest/Saved/StagedBuilds/Linux/URLabTest.sh /Game/FastPath/FastPathRender \
    -URLabDrive=push -URLabModel=/abs/path/scene.xml \
    -URLabCaps=serve,cameras -URLabScene=cammax=0 \
    -RenderOffScreen -nosplash -unattended -stdout
```

---

## 2. Drive it from Python

```python
import cv2, mujoco
from urlab_client import RenderClient

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with RenderClient.grpc("127.0.0.1", 50051) as rc:
    rc.load_xml("scene.xml")           # server compiles it with its OWN libmujoco
    print(rc.camera_names())           # the model's cameras + "user"

    for _ in range(300):
        mujoco.mj_step(model, data)                       # you own the physics
        frames = rc.render_mjdata(model, data, cameras=["cam0"])
        bgr = frames["cam0"].to_bgr()                     # HxWx3 uint8, ready for cv2
        cv2.imshow("cam0", bgr); cv2.waitKey(1)
```

The loop above is **forced** (`delay=0`, the default) — each `render_mjdata` returns
the exact frame for the state you just pushed, and blocks until it's captured.
Against an **async/streaming** server (`-URLabDrive=await`), pass `delay=N` to pull
the latest ring frame instead — non-blocking and smooth:

```python
    frames = rc.render_mjdata(model, data, cameras=["cam0"], delay=0.05)
```

> **Tip — hide the round-trip with a 1-step pipeline (DIY, not yet built-in).**
> If you can tolerate images being one physics step stale, you can get near-zero
> wait *while staying in forced mode* (deterministic, exact-fresh frames). Render
> the **previous** state on a background thread while the main thread steps the
> **next** one — the render RPC and `mj_step` both release the GIL, so they overlap,
> and each iteration costs `max(step, render)` instead of `step + render`:
>
> ```python
> import copy, concurrent.futures as cf
> pool = cf.ThreadPoolExecutor(max_workers=1)
> pending, prev = None, None                      # in-flight render, its state
> while running:
>     if pending is not None:
>         frames = pending.result()               # last frame (state one step ago)
>     prev = (model, copy.copy(data))             # snapshot the state to render
>     pending = pool.submit(rc.render_mjdata, *prev, cameras=["cam0"])
>     mujoco.mj_step(model, data)                 # step forward, overlapping the render
> ```
>
> This is just a client loop pattern — a small ring buffer of in-flight states — not
> a `RenderClient` feature. Distinct from server async mode (§ modes table): there
> the *server* paces capture and frames are several substeps stale; here *you* pace
> it, frames are exactly one step stale, and every frame is a real forced capture.

- **Load once, render many.** `load_xml(path)` inlines every `<include>` and ships
  the referenced meshes/textures for you. (`load_mjb(path)` for a version-matched
  compiled model; `load_mjz(path)` for a `.mjz` archive.)
- **Name your cameras.** `cameras=["cam0"]` is far cheaper than rendering all — each
  camera is a full scene capture. Omit `cameras` to render every camera.
- **`CameraFrame`** → `.to_bgr()` (OpenCV), `.to_rgb()`, or `.to_array()` (raw HxWx3).
- **Resolution** comes from the model: `<camera resolution="1280 960">`. Changing it
  is just a scene reload — no re-cook. (Server default caps model-camera height at
  480 unless it was launched with `-URLabScene=cammax=0`.)

---

## 3. Transports

The render client (getting images back) works over three transports:

| Transport | Connect | Use when |
|---|---|---|
| **gRPC** | `RenderClient.grpc("127.0.0.1", 50051)` | Default. Any host, single port. |
| **ZMQ** | `RenderClient.zmq("127.0.0.1", 5559)` | Any host; bare host or `tcp://…`. |
| **SHM** | `RenderClient("tcp://127.0.0.1", transport="shm")` | **Co-located only** (client + server same box) — lowest latency. |

> Note: this is the *capture* path (Python ← images). The separate **live UE
> mirror** path (a UE window that mirrors an owner's pose bus, `-URLabDrive=stream:`)
> supports **gRPC + ZMQ only, not SHM**.

---

## 4. A free / "user" camera (mirror any MuJoCo viewer)

Beyond the model's own cameras, the server has a returnable free camera named
`"user"`. Drive it with a `(pos, fwd, up)` eye pose — `viewer_sync` extracts that
from any MuJoCo viewer's camera, so a passive-viewer orbit drives the render:

```python
from urlab_client import viewer_sync, USER_CAMERA

with mujoco.viewer.launch_passive(model, data) as viewer:
    scene = viewer_sync.make_scene(model)                 # once
    while viewer.is_running():
        mujoco.mj_step(model, data); viewer.sync()
        pose = viewer_sync.pose_from_passive(viewer, scene=scene)
        frames = rc.render_mjdata(model, data, cameras=[USER_CAMERA], user_pose=pose)
        # frames["user"].to_bgr() is the UE render from the viewer's viewpoint
```

A headless variant (no GUI) that drives the `user` cam from a plain `MjvCamera`
lives in `smoke_usercam.py` at the repo root — a good end-to-end smoke test.

---

## Runnable examples

- `examples/render_client_example.py` — minimal render loop. NOTE: it does **not**
  call `load_xml`, so it assumes the server already holds a matching model (either
  a `-URLabModel=` boot, or a prior client `load_xml`). For an empty `await` server,
  call `rc.load_xml(...)` first (see `smoke_usercam.py`) or you'll get black frames.
- `examples/render_server_viewer.py` — passive viewer + the `"user"` camera painted
  as a picture-in-picture (the pattern in §4).
- `examples/render_pool_example.py` — many servers, one client (`RenderPool`).

## See also

- `docs/render_server_guide.md` — the fuller how-it-works guide.
- `../UnrealRoboticsLab/docs/render_server_flags.md` — per-flag reference (its
  `-URLabFast*` names are the **legacy lineage**; each maps to a current flag via
  `MjLauncherFlags.h`).
- `../UnrealRoboticsLab/docs/render_server_packaging.md` — how to cook the server.
