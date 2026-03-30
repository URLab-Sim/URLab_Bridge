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
uv run src/run.py --ui

# Run a specific policy headless
uv run src/run.py --policy unitree_12dof --prefix g1

# Test ZMQ connection
uv run src/run.py --test --prefix g1

```

## Available Policies

| Key                | Robot   | DOF | Description                              | Requires PHC |
|--------------------|---------|-----|------------------------------------------|:------------:|
| `unitree_12dof`    | G1      | 12  | Basic walking -- WASD twist control      |              |
| `unitree_wo_gait`  | G1      | 29  | Full body walking without gait clock     |              |
| `smooth`           | G1      | 29  | Smoother walking policy                  |              |
| `beyondmimic_dance`| G1      | 29  | Motion imitation -- dance                |      Y       |
| `h2h`             | G1      | 21  | Human motion retargeting                 |      Y       |
| `amo`             | G1      | 29  | Adaptive motion optimization             |      Y       |
| `twist_tracker`   | G1      | 12  | Motion tracker with twist                |      Y       |
| `go2_wtw`         | Go2     | 12  | Walk-These-Ways rough-terrain locomotion |              |

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

`ros2_broadcaster.py` republishes ZMQ streams as standard ROS 2 topics (JointState, Image, Float64MultiArray). Requires a sourced ROS 2 workspace (Humble/Jazzy).

```bash
source /opt/ros/humble/setup.bash
uv run src/ros2_broadcaster.py
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).

Copyright 2026 Jonathan Embley-Riches.

## Related

This package is the Python companion to [Unreal Robotics Lab](https://github.com/URLab-Sim/UnrealRoboticsLab), an Unreal Engine plugin embedding MuJoCo physics for sim-to-real robotics research.
