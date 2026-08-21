# RenderPool — parallel rendering across many render-server instances

`RenderPool` renders one MuJoCo state across a **pool** of fast-path render-server
instances at once. One `RenderClient` drives one UE instance, which renders its
cameras sequentially; for many cameras that serialisation dominates latency. The
pool spreads the cameras across several instances and renders them **concurrently**
— a near drop-in, parallel version of `RenderClient.render_mjdata`.

```python
from urlab_client import RenderPool

with RenderPool.from_config("pool.json") as pool:
    pool.load_xml("scene.xml")                 # broadcast the model to ALL instances
    frames = pool.render_mjdata(model, data)   # all cameras, split across the pool
    #   frames -> {camera_name: CameraFrame}
```

## 1. You hand it the instance IPs + ports

The pool does **not** launch instances — an external orchestrator spins them up
(across the network, and/or several on one host each with a distinct
`-URLabDmEnvPort=`) and hands their addresses to the pool. Three ways:

**Config file** (recommended for a farm):

```jsonc
// pool.json
{
  "instances": [
    { "host": "10.0.0.11", "port": 50051 },
    { "host": "10.0.0.12", "port": 50051 },
    { "host": "127.0.0.1",  "port": 50052 }   // a 2nd instance on this host
  ]
}
```
```python
RenderPool.from_config("pool.json")            # or $URLAB_RENDER_POOL for the path
```

**Endpoint string / list** (what a `--endpoints` CLI flag feeds):

```python
RenderPool.from_endpoints("10.0.0.11:50051, 10.0.0.12:50051")
RenderPool.from_endpoints(["10.0.0.11", ("10.0.0.12", 50051)])  # bare host -> :50051
```

A bare host defaults to port **50051** (the gRPC / dm_env_rpc port). The transport
is gRPC; each instance gets its own channel + lock, so requests are independent.

`pool.endpoints` lists the resolved `host:port` set; `len(pool)` is the instance
count.

## 2. It auto-distributes the cameras — no manual assignment

Every instance holds the **full model** (the pool broadcasts `load_*` to all), so
any instance can render any camera. Each `render` call:

1. Takes the requested cameras (default: all of them).
2. **Round-robins** them across the instances (`_distribute`) — even split, no
   state, no pinning. 4 cameras over 2 instances → `2 + 2`; 5 over 2 → `3 + 2`;
   2 over 3 → `1 + 1 + 0` (one idle that frame).
3. Sends each instance the **full pose set** but only its **camera subset**, all
   in flight at once (one worker thread per instance).
4. Merges the returned frames into one `{camera_name: CameraFrame}`.

Wall-clock ≈ the slowest single instance's share, not the sum. There is nothing to
tune per-instance — add or remove endpoints and the split adapts.

## 3. The render surface (mirrors `RenderClient`)

```python
pool.load_xml(xml)            # or load_mjb / load_mjz / load_model — broadcast to ALL
pool.camera_names()           # names the servers report (probed once, cached)

# render straight from a mujoco (model, data):
frames = pool.render_mjdata(
    model, data,
    cameras=["cam0", "cam1", "wrist"],   # default None = all cameras
    user_pose=(pos, fwd, up),            # optional free/user camera pose
    delay=0.0,                           # SECONDS of server-side latency (0 = fresh)
)

# or push an explicit pose set:
frames = pool.render(bxpos=..., bxquat=..., cameras=[...], delay=0.0)
```

- **`delay`** is in **seconds** (real clock-based latency-ring sampling on the
  server). `0.0` = fresh/blocking.
- **`user_pose`** is sent only to the one instance that draws the `USER_CAMERA`
  ("user") this frame — add `USER_CAMERA` to `cameras` to get its frame back.

## 4. `reset(model)` — re-baseline geoms on every instance

Per step you send only compact per-body transforms (`bxpos`/`bxquat`); geom
*local* offsets don't change while stepping, so they are never re-sent — that's the
bandwidth win. But on an **episode reset / geom re-randomisation** the offsets do
change, and **every** mirror needs the new baseline:

```python
pool.reset(model)     # broadcasts model.geom_pos / geom_quat to ALL instances
```

This is a dedicated broadcast, *not* piggybacked on a render — a normal `render`
skips instances with no camera that frame, so an idle instance would otherwise miss
the reset and place geoms against stale offsets on its next frame. `reset` fans out
to all (one probe capture per instance, discarded, since the server can't render
zero cameras). Call it once after a reset, then resume body-only `render`.

## 5. Failures

If one or more instances fail a call, the pool raises `RenderPoolError` after the
others complete; `err.failures` maps the failed `host:port` to its exception. The
gRPC transport auto-reconnects a dropped stream, so most blips self-heal; a hard
failure surfaces here so the caller can retry the frame.

## Typical loop

```python
from urlab_client import RenderPool
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with RenderPool.from_config("pool.json") as pool:
    pool.load_xml("scene.xml")            # same model every instance loads
    pool.reset(model)                     # baseline geoms across the pool

    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        frames = pool.render_mjdata(model, data)   # cameras split across the pool
        # ... consume frames[name].to_bgr() ...
        # on a new episode / geom randomisation: pool.reset(model)
```
