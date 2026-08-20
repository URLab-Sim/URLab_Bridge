#!/usr/bin/env bash
# Handy one-command demo: launch a LOCAL pool of N render-server instances, write
# their pool config, and render a scene across them with render_pool_example.py.
#
# In production an external orchestrator launches the instances (often across the
# network) and writes the pool config; this script is the local stand-in so you
# can see the whole thing work on one machine.
#
# Usage:
#   bash examples/launch_local_pool.sh [N] [scene.xml]
#     N       number of instances (default 2)
#     scene   MuJoCo scene to render (default: aloha from mujoco_menagerie)
#
# Point it at your build with env vars (defaults shown):
#   URLAB_UE        /home/buzz/UnrealEngine/Engine/Binaries/Linux/UnrealEditor
#   URLAB_UPROJECT  /home/buzz/Documents/urlab_debug/URLabTest/URLabTest.uproject
#   URLAB_MAP       /Game/FastPath/FastPathRender
#   BASE_PORT       first gRPC port (default 50051); instance i uses BASE_PORT+i
set -uo pipefail

N="${1:-2}"
SCENE="${2:-/home/buzz/Documents/urlab_debug/mujoco_menagerie/aloha/scene.xml}"
UE="${URLAB_UE:-/home/buzz/UnrealEngine/Engine/Binaries/Linux/UnrealEditor}"
UPROJ="${URLAB_UPROJECT:-/home/buzz/Documents/urlab_debug/URLabTest/URLabTest.uproject}"
MAP="${URLAB_MAP:-/Game/FastPath/FastPathRender}"
BASE_PORT="${BASE_PORT:-50051}"
HERE="$(cd "$(dirname "$0")" && pwd)"          # examples/
BRIDGE="$(cd "$HERE/.." && pwd)"               # URLab_Bridge/
CFG="$(mktemp -t urlab_pool_XXXX.json)"
LOGDIR="$(mktemp -d -t urlab_pool_logs_XXXX)"

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -f "$CFG"; }
trap cleanup EXIT

echo "[pool] launching $N instance(s), gRPC ports ${BASE_PORT}..$((BASE_PORT+N-1))"
for i in $(seq 0 $((N-1))); do
  port=$((BASE_PORT + i))
  log="$LOGDIR/instance_$i.log"
  # Same-host instances need a distinct gRPC port (-URLabDmEnvPort) AND a distinct
  # -URLabInstanceIndex so their ZMQ ports don't collide either.
  "$UE" "$UPROJ" "$MAP" -game \
    -URLabFastServe -URLabFastForcedOnly -URLabFastCameras -URLabFastCamMaxHeight=0 \
    -URLabInstanceIndex="$i" -URLabDmEnvPort="$port" \
    -RenderOffScreen -nosplash -unattended -stdout \
    -abslog="$log" > "$log.stdout" 2>&1 &
  pids+=("$!")
  echo "  instance $i -> 127.0.0.1:$port (pid ${pids[-1]}, log $log)"
done

echo "[pool] waiting for all gRPC ports (UE cold-boot is slow)..."
for _ in $(seq 1 360); do
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "an instance exited early; see $LOGDIR"; exit 1; }; done
  ready=1
  for i in $(seq 0 $((N-1))); do
    ss -ltn 2>/dev/null | grep -q ":$((BASE_PORT+i)) " || { ready=0; break; }
  done
  [ "$ready" = 1 ] && break
  sleep 1
done
[ "${ready:-0}" = 1 ] || { echo "[pool] not all ports came up; see $LOGDIR"; exit 1; }

# Write the pool config the client reads (host:port per instance).
{
  echo '{ "instances": ['
  for i in $(seq 0 $((N-1))); do
    sep=","; [ "$i" -eq $((N-1)) ] && sep=""
    echo "  {\"host\": \"127.0.0.1\", \"port\": $((BASE_PORT+i))}$sep"
  done
  echo '] }'
} > "$CFG"
echo "[pool] all up. config: $CFG"

cd "$BRIDGE"
PYTHONPATH="$BRIDGE/src" MUJOCO_GL="${MUJOCO_GL:-egl}" \
uv run --no-project --with mujoco==3.11.0 --with numpy --with msgpack \
  --with grpcio --with protobuf --with googleapis-common-protos \
  --python 3.11 python "$HERE/render_pool_example.py" \
  --xml "$SCENE" --pool-config "$CFG"
