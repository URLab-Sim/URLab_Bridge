# URLab render server — how it works (quickstart)

A headless Unreal Engine process that renders MuJoCo scenes photorealistically.
**Your Python process owns the physics** — it steps MuJoCo, pushes the resulting
state each frame, and gets camera images back. UE never simulates; it *mirrors*
your state and renders. Everything below goes through the high-level
`RenderClient` — never hand-roll op dicts, wire keys, or transports.

## 1. Boot the server (empty level, client-driven)

No model is needed at boot. `-URLabDrive=await` (formerly `-URLabFastServe`)
stands up the bridge + a renderer on an empty level; the client uploads the
model over the wire and the server (re)builds its renderer from it. `serve` must
be in `-URLabCaps` on the await path or the server never binds `:50051`.

```
UnrealEditor URLabTest.uproject /Game/FastPath/FastPathRender -game \
  -URLabDrive=await -URLabCaps=serve,cameras \
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
| Server boot | `-URLabDrive=await` (delay 0 gives exact forced frames) | `-URLabDrive=await` (same boot; the delay picks the mode) |
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

## 6. Rendering across a pool (parallel, many instances)

One instance renders its cameras **sequentially**. To parallelize, run several
render-server instances and split the cameras across them with `RenderPool` —
same `load_*` / `render_mjdata` surface as `RenderClient`, fanned out.

**Try it in one command (local demo):**
```
bash examples/launch_local_pool.sh 3          # boots 3 local instances + renders across them
# point it at your build with URLAB_UE / URLAB_UPROJECT / URLAB_MAP env vars
```
That script is the local stand-in for the real setup: in production an **external
orchestrator** launches the instances (below) and the client just attaches.

An **external orchestrator** launches the instances (across the network and/or
several on one host) and hands the pool their addresses. Same-host instances each
need a distinct gRPC port via `-URLabNet=grpc=` (plus a distinct `index=` in the
same `-URLabNet` so their ZMQ ports don't collide):

```
# instance 0
... -URLabDrive=await -URLabCaps=serve,cameras \
    -URLabNet=index=0,grpc=50051
# instance 1
... -URLabDrive=await -URLabCaps=serve,cameras \
    -URLabNet=index=1,grpc=50052
```
Cross-host instances each just bind `:50051` on their own machine — no flag needed.

The pool takes its endpoints from a **JSON config file** or an **explicit list**
(what a `--endpoints` CLI flag passes):

```python
from urlab_client import RenderPool

# config file: {"instances": [{"host":"10.0.0.1","port":50051}, {"host":"10.0.0.2","port":50051}]}
with RenderPool.from_config("pool.json") as pool:          # or $URLAB_RENDER_POOL
    pool.load_xml("scene.xml")                             # broadcast to ALL, in parallel
    frames = pool.render_mjdata(model, data)              # all cameras, auto-split N ways

# or explicit endpoints (from a CLI --endpoints)
pool = RenderPool.from_endpoints("10.0.0.1:50051,10.0.0.2:50051")
```

Rules:
- **Cameras are distributed automatically** — each render splits the requested
  cameras evenly across the whole pool. There is no manual per-instance mapping.
- **Everything fans out concurrently** — both `load_*` and every `render` dispatch
  to all endpoints at once and join (wall-clock ≈ the slowest instance, not the
  sum). The pool never loops instances sequentially.
- If `USER_CAMERA` is requested, only the instance it lands on gets `user_pose`.
- A hard per-instance failure raises `RenderPoolError` (with `.failures` naming the
  endpoints); transient stream drops self-heal via the transport's reconnect.
- Speedup is real on multiple GPUs / uncapped hardware (the orchestrator's target);
  on a single shared GPU it helps partially (the GPU is the shared bottleneck).

## 7. Mirroring a live sim (smooth, async, interactive)

Sometimes you just want to *watch* a running sim — and maybe reach in and push
things — without disturbing the eval render pool. That's a **mirror**: a smooth,
async view fed by a separate channel.

The key idea: an **owner** (a Python client *or* a UE instance — whoever steps the
physics) broadcasts the **render tier** on a **`render` bus** (per-body
`bxpos`/`bxquat` transforms + optional debug fields — never qpos) and accepts
`fastpath_perturb` on a control channel. Any number of **read-only or interactive
mirrors** subscribe and render it directly (the mirror runs zero MuJoCo). This is a
*different channel* from the forced eval pool, so mirroring never stalls it — which
also answers the "delay vs no-delay" question: the pool is no-delay/forced (crisp
eval); the mirror is the smooth async bus (latest transforms, a few frames behind).

> The qpos render tier (the old `viewer` bus `{t, qpos, qvel}` + the Python
> `peek.py` viewer) was removed in Phase 3.2/3.3 — a desktop mirror is now an
> ordinary UE transform-mirror renderer that consumes the `render` tier.

**Owner side** — a Python owner broadcasts + accepts pushes each step:
```python
from urlab_client.fastpath_owner import FastPathOwner
owner = FastPathOwner(mjb_bytes, scene="demo", bus_port=5561, control_port=5571)
frame = 0
while running:
    data.xfrc_applied[:] = 0
    owner.apply_perturbations(model, data)                     # pushes from mirrors
    mujoco.mj_step(model, data)
    owner.serve_pending()                                      # answer hello/perturb
    owner.publish_mjdata(frame, model, data)                  # feed mirrors (render tier)
    frame += 1
```
(A UE Direct instance is the other kind of owner — launch it with
`-URLabBroadcastViewers=1`; a mirror attaches to either identically.)

**Mirror side** — a UE transform-mirror renderer subscribes the owner's `render`
tier; ctrl-drag pushes back over `fastpath_perturb`. The push force is MuJoCo's own
(`mjv_applyPerturbForce`, run owner-side from a drag *intent*), so a grab in one
mirror shows up in every mirror's view. A UE VR instance is the photoreal/
walk-around version of the same mirror (same bus, same perturb). Launch/join with
`python -m urlab_client.session join <id> --mode vr`.

Runnable demo: `examples/owner_viewer_franka.py` (a Python owner + a native
`mujoco.viewer` window; discoverable over gRPC for a UE mirror to join).

## TL;DR for a downstream agent

1. Boot the server with `-URLabDrive=await -URLabCaps=serve,cameras`.
2. `with RenderClient.grpc(host, port) as rc: rc.load_xml(scene)`.
3. Each step: `rc.render_mjdata(model, data, cameras=[...])`; drive your own view
   with `user_pose=viewer_sync.free_camera_pose(...)` + `USER_CAMERA`.
4. Never touch transports or op dicts. Catch `URLabTimeoutError` per frame.
