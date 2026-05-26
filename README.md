# urlab_bridge

Python middleware for [Unreal Robotics Lab (URLab)](https://github.com/URLab-Sim/UnrealRoboticsLab) -- connects neural-network policies to MuJoCo-in-Unreal simulations over ZeroMQ.

Run pretrained locomotion policies, visualize joint states and camera streams, or bridge everything to ROS 2 -- all from a single Python package.

## Installation

```bash
# Recommended (uv)
cd urlab_bridge
uv sync                          # core deps (ZMQ, NumPy, OpenCV, DearPyGui)

# To run policies (optional):
uv sync --extra policy            # + PyTorch, ONNX, etc.
uv pip install -e ./RoboJuDo     # policy framework (bundled submodule)
```

The dashboard (joints, sensors, cameras, actuator control) works without the policy extras.
RoboJuDo is only needed if you want to run neural-network policies.

Requires Python 3.11+.

## Quick Start

```bash
# Launch the dashboard (joint/sensor/camera viewer, actuator control, optional policy runner)
uv run urlab-ui

# Run a specific policy headless
uv run urlab-policy --policy unitree_12dof --prefix g1

# Test ZMQ connection
uv run urlab-ping --prefix g1
```

`uv run src/run.py --ui` / `--policy` / `--ping` works too as a thin
shim around the same console scripts.

## Setting up a robot

A bundled policy needs three things in place before it will move
anything:

1. **The right MJCF.** G1 XMLs ship in
   [`assets/robots/g1/`](assets/robots/g1/) alongside their mesh dirs
   (`meshes/`, `g1_mjlab_meshes/`). Go2 ships separately: clone
   [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
   and use `unitree_go2/go2.xml`. See the policy table below for the
   key-to-MJCF mapping.
2. **Imported in Unreal.** Drag the `.xml` into the UE Content
   Browser. URLab's importer runs the mesh-conversion pipeline
   (STL -> GLB) on first import and produces an Articulation Blueprint.
3. **A `UMjPDController` on the Articulation Blueprint.** Open the
   imported Blueprint, add a `MjPDController` component (Add Component
   in the Components panel), compile. The policies emit position
   targets at policy rate; the PD controller turns those into per-step
   joint torques. Without it the robot will not respond.

Then drop the Blueprint into a level, click Play in UE, and start the
dashboard or policy runner against the running editor.

The full step-by-step lives at
[Running a Bundled Policy](https://github.com/URLab-Sim/UnrealRoboticsLab/blob/main/docs/python/running_policies.md).

## Documentation

The Python API docs live in the main URLab plugin repository alongside
the rest of the URLab documentation:

- [Python Getting Started](https://github.com/URLab-Sim/UnrealRoboticsLab/blob/main/docs/python/getting_started.md) —
  guided walkthrough: connect, author a scene, run PIE, step the sim,
  send control, work with cameras, add a new policy.
- [Python API Reference](https://github.com/URLab-Sim/UnrealRoboticsLab/blob/main/docs/python/api.md) —
  full reference for `URLabClient` and the `scene` / `sim` / `runtime`
  / `outliner` / `debug` / `viewport` / `recording` / `replay`
  namespaces.
- [URLab docs index](https://github.com/URLab-Sim/UnrealRoboticsLab/blob/main/docs/index.md) —
  everything else: MJCF import, scene authoring, controllers, debug
  visualisation, recording / replay.

## Available Policies

| Key                | Robot   | DOF | MJCF                                                 | Description                              | PHC |
|--------------------|---------|-----|------------------------------------------------------|------------------------------------------|:---:|
| `unitree_12dof`    | G1      | 12  | `assets/robots/g1/g1_29dof_rev_1_0.xml`              | Basic walking, WASD twist control        |     |
| `unitree_wo_gait`  | G1      | 29  | `assets/robots/g1/g1_29dof_rev_1_0.xml`              | Full body walking without gait clock     |     |
| `smooth`           | G1      | 29  | `assets/robots/g1/g1_29dof_rev_1_0.xml`              | Smoother walking policy                  |     |
| `beyondmimic_dance`| G1      | 29  | `assets/robots/g1/g1_29dof_rev_1_0.xml`              | Motion imitation, dance                  |  Y  |
| `h2h`              | G1      | 21  | `assets/robots/g1/g1_29dof_rev_1_0.xml`              | Human motion retargeting                 |  Y  |
| `amo`              | G1      | 29  | `assets/robots/g1/g1_29dof_rev_1_0.xml`              | Adaptive motion optimization             |  Y  |
| `twist_tracker`    | G1      | 12  | `assets/robots/g1/g1_29dof_rev_1_0.xml`              | Motion tracker with twist                |  Y  |
| `go2_wtw`          | Go2     | 12  | `mujoco_menagerie/unitree_go2/go2.xml` (download)    | Walk-These-Ways rough-terrain locomotion |     |

Policies marked **PHC** require the [PHC submodule](https://github.com/ZhengyiLuo/PHC) installed inside RoboJuDo.

## ZMQ Protocol

URLab publishes binary-packed data over ZeroMQ PUB/SUB sockets. All topics are prefixed with the articulation name (e.g. `g1/`).

| Topic Pattern          | Direction       | Payload Format               |
|------------------------|-----------------|------------------------------|
| `{prefix}/joint/{id}`  | Unreal -> Python | `<Ifff` (ID, pos, vel, acc) |
| `{prefix}/sensor/{name}` | Unreal -> Python | `<I` ID + `<I` dim + `f`*N floats |
| `{prefix}/camera/{name}` | Unreal -> Python | Raw BGRA bytes (dedicated socket) |
| `{prefix}/control`     | Python -> Unreal | `<I` count + (`<If`)*N (ID, value) pairs |

- **State socket** (default `tcp://127.0.0.1:5555`): joints + sensors at up to 1000 Hz.
- **Control socket** (default `tcp://127.0.0.1:5556`): policy sends target positions.
- **Camera socket** (default `tcp://127.0.0.1:5558`): high-bandwidth image stream on a separate socket.

## ROS 2 Bridge

`urlab_tools.ros2_broadcaster` republishes ZMQ streams as standard ROS 2 topics (JointState, Image, Float64MultiArray). Requires a sourced ROS 2 workspace (Humble/Jazzy).

```bash
source /opt/ros/humble/setup.bash
uv run python -m urlab_tools.ros2_broadcaster
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).

Copyright 2026 Jonathan Embley-Riches.

## Related

This package is the Python companion to [Unreal Robotics Lab](https://github.com/URLab-Sim/UnrealRoboticsLab), an Unreal Engine plugin embedding MuJoCo physics for sim-to-real robotics research.
