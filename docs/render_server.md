# Fast-path render server + `RenderClient`

Render MuJoCo camera images from a packaged Unreal render server, driven from Python.
Python owns the physics (a *puppet*); Unreal is a *mirror* that renders the pushed
state. This doc covers the whole loop: package the server, compile a model, run it, and
drive it with `RenderClient`.

## The pipeline

```
Python: mj_step ──poses──▶ ZMQ (step port 5559) ──▶ UE mirror applies poses
                                                     └▶ render camera(s)
        frames ◀── reply (BGRA8 per camera) ◀────────┘
```

Two render regimes (same `fastpath_render` op, chosen per call):

| `delay` | behaviour | use for |
|---------|-----------|---------|
| `0`     | **fresh** — block until the exact pushed state is rendered | eval / exact-state observations |
| `> 0`   | **stale** — return an `N`-substep-old frame from the server ring (no fresh-render wait, higher throughput) | training data, matched to real-camera latency (usually 1–2 frames) |

Fresh-blocking is inherently sub-real-time when render time ≥ timestep; for a smooth
real-time *viewer* use a 1-frame delay (show frame `N-1` while `N` renders) or render
locally with the MuJoCo viewer.

## 1. Package the render server (Unreal side)

Build a cooked Windows server from the plugin (see the plugin's
`docs/render_server_packaging.md` for the full recipe):

```powershell
RunUAT.bat BuildCookRun -project="url_proj.uproject" -noP4 -platform=Win64 `
  -clientconfig=Development -cook -build -stage -pak -iostore -prereqs -nodebuginfo
```

Output: `Saved/StagedBuilds/Windows/url_proj.exe`.

## 2. Compile a version-matched model (MJB)

The server loads a `.mjb` whose MuJoCo version matches Unreal's. Compile it with the
matching `mjbcompile` (run from the scene dir so mesh paths resolve):

```bash
mjbcompile scene.xml scene.mjb        # -> "ver=3011001" must match the UE MuJoCo
```

Camera resolution comes from the model's `<camera resolution="W H">`; change res by
editing the XML and recompiling — no recook needed.

## 3. Launch the server

```bash
url_proj.exe /Game/FirstPerson/Lvl_FirstPerson \
  -URLabDrive=push -URLabModel=scene.mjb -URLabCaps=serve,cameras \
  -URLabScene=cammax=0 -RenderOffScreen -nosplash -abslog=server.log
```

Key flags:

| flag | meaning |
|------|---------|
| `<map>` | **use a lit level** (SkyLight + reflection captures) or metallic surfaces look flat. `/Game/FirstPerson/Lvl_FirstPerson` is lit; `/Engine/Maps/Entry` is not. |
| `-URLabModel=<file>` | model to load (formerly `-URLabFastMjb`; format from the `.mjb`/`.xml`/`.mjz` extension) |
| `-URLabDrive=push` | forced/eval regime — the bridge serves `fastpath_render` (formerly `-URLabFastForcedOnly`) |
| `-URLabCaps=serve,cameras` | serve the render bridge and enable camera capture (`cameras` formerly `-URLabFastCameras`) |
| `-URLabScene=cammax=N` | camera height cap; `0` = honour the model resolution exactly (else clamps, default 480). Formerly `-URLabFastCamMaxHeight=N` |
| `-RenderOffScreen` | headless |

Do **not** force low scalability (`sg.*Quality 0`) for a fidelity/viewer run — it drops
screen-space reflections and post-processing and flattens metallics. Low scalability is
only for comparable perf benchmarks.

## 4. Drive it with `RenderClient`

```python
import mujoco
from urlab_bridge.urlab_client import RenderClient

model = mujoco.MjModel.from_xml_path("scene.xml")   # same model the server loaded
data = mujoco.MjData(model)

with RenderClient("tcp://127.0.0.1", step_port=5559) as rc:
    names = rc.camera_names()
    for _ in range(100):
        mujoco.mj_step(model, data)
        frames = rc.render_mjdata(model, data, cameras=[names[0]])  # fresh, 1 camera
        bgr = frames[names[0]].to_bgr()      # HxWx3 uint8, ready for cv2

    # stale-but-fast (training-style, ~2-frame real-camera latency):
    frames = rc.render_mjdata(model, data, delay=2)
```

API summary (`urlab_client.render_client`):

- `RenderClient(address, step_port=5559, transport="zmq"|"shm")`
- `.load_mjb(bytes|path)` — hot-swap the model (optional; `-URLabModel` preloads one)
- `.render(bxpos, bxquat, cxpos=, cxquat=, cameras=, delay=, sim_time=)` — raw poses
- `.render_mjdata(model, data, cameras=, delay=)` — convenience from a MuJoCo state
- `.camera_names()` — authoritative names from the server
- `CameraFrame.to_bgr() / .to_rgb() / .to_array()` — decode; `.width/.height/.dtype/.frame_id/.sim_time`

Rendering **one** named camera is far cheaper than all of them (each is a full scene
capture). See `examples/render_client_example.py` for a runnable version (live cv2
window, PNG save, fresh vs delayed).
