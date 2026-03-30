# Walk-These-Ways Go2 Setup

## 1. Robot Model (MJCF XML)

The Go2 robot model comes from MuJoCo Menagerie (not walk-these-ways):

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
# Import into Unreal: drag mujoco_menagerie/unitree_go2/go2.xml into Content Browser
```

## 2. Policy Checkpoints

Place the following files in this directory:

- `body_latest.jit` — Main policy network
- `adaptation_module_latest.jit` — Observation history adaptation module

```bash
git clone https://github.com/Teddy-Liao/walk-these-ways-go2.git
cd walk-these-ways-go2

# Copy pretrained checkpoints
cp runs/gait-conditioned-agility/pretrain-go2/train/checkpoints/body_latest.jit <this_directory>/
cp runs/gait-conditioned-agility/pretrain-go2/train/checkpoints/adaptation_module_latest.jit <this_directory>/
```

## 3. Run

```bash
cd synthy_bridge
uv run src/run.py --policy go2_wtw --prefix go2
```

## Sources

- Robot XML: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2
- Policy: https://github.com/Teddy-Liao/walk-these-ways-go2
- Original paper: https://github.com/Improbable-AI/walk-these-ways (MIT License)
