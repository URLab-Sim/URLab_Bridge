#!/usr/bin/env bash
# URLab MJB fast-path demo: MuJoCo viewer + UE puppet + live camera feeds.
#
# Layout assumed (all siblings under one root):
#   <root>/UnrealRoboticsLab      the UE plugin (this repo's sibling)
#   <root>/URLab_Bridge           this repo
#   <root>/mujoco_menagerie       cloned menagerie scenes
#   <root>/mjb_test/mjbcompile    the xml->mjb tool (see docs/fast_path_render.md)
#
# Override any path via env: URLAB_ROOT, UE_EDITOR, UE_PROJECT, URLAB_MJBCOMPILE,
# URLAB_MJLIB. See UnrealRoboticsLab/docs/fast_path_render.md for full setup.
#
#   bash run_fastpath.sh [/path/to/scene.xml]
#
# Starts the MuJoCo owner (viewer + transform/camera broadcast), opens the Unreal
# editor as a fast-path renderer that auto-discovers the owner and renders its
# cameras, and opens an OpenCV window per camera. In UE, press PLAY to start the
# live puppet + camera streams. Close the editor to stop everything.
set -u

# Root = two levels up from this script (…/URLab_Bridge/scripts/run_fastpath.sh),
# unless URLAB_ROOT is set.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${URLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BRIDGE="$ROOT/URLab_Bridge"
UE="${UE_EDITOR:-$HOME/UnrealEngine/Engine/Binaries/Linux/UnrealEditor}"
UPROJ="${UE_PROJECT:-$ROOT/URLabTest/URLabTest.uproject}"
SCENE="${1:-$ROOT/mujoco_menagerie/aloha/scene.xml}"
BUS_PORT="${BUS_PORT:-5561}"
CAM_BASE="${CAM_BASE:-5600}"

export URLAB_ROOT="$ROOT"  # so run_fastpath_demo.py finds mjbcompile + libmujoco

for p in "$UE" "$UPROJ" "$SCENE"; do
  [ -e "$p" ] || { echo "[demo] missing: $p (set UE_EDITOR / UE_PROJECT, or pass a scene)"; exit 1; }
done
echo "[demo] root=$ROOT"
echo "[demo] scene=$SCENE"

# 1) Owner: MuJoCo viewer + transform/camera broadcaster; writes the MJB.
(
  cd "$BRIDGE" &&
  uv run --no-project --with mujoco --with pyzmq --with msgpack --with numpy --python 3.11 \
    python scripts/run_fastpath_demo.py "$SCENE" --port "$BUS_PORT"
) &
OWNER=$!
CAMVIEW=""
trap 'kill $OWNER 2>/dev/null; [ -n "$CAMVIEW" ] && kill $CAMVIEW 2>/dev/null' EXIT
echo "[demo] owner pid $OWNER (bus tcp://127.0.0.1:$BUS_PORT)"

# 2) Wait for the owner to advertise itself in the registry.
REGDIR="${URLAB_REGISTRY_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/URLab/registry}"
echo "[demo] waiting for the owner to advertise ($REGDIR)..."
for _ in $(seq 1 60); do ls "$REGDIR"/fastpath_*.json >/dev/null 2>&1 && break; sleep 1; done
ls "$REGDIR"/fastpath_*.json >/dev/null 2>&1 || { echo "[demo] owner never advertised"; exit 1; }
echo "[demo] owner advertised."

# 3) OpenCV camera viewer (background): windows appear when you press Play in UE.
(
  cd "$BRIDGE" &&
  uv run --no-project --with opencv-python --with pyzmq --with numpy --python 3.11 \
    python scripts/show_fastpath_cameras.py --host 127.0.0.1 --base-port "$CAM_BASE"
) &
CAMVIEW=$!
echo "[demo] camera viewer pid $CAMVIEW (feeds on $CAM_BASE+)"

# 4) UE editor fast-path renderer + render server (foreground).
echo "[demo] launching the Unreal editor renderer (auto-discovers the owner)..."
echo "[demo] >>> In UE, press PLAY to start the live puppet and camera feeds. <<<"
"$UE" "$UPROJ" -URLabFastDiscover -URLabFastCameras

echo "[demo] editor closed; stopping owner + camera viewer."
