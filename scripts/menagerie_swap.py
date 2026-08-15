"""Interactive menagerie scene switcher for the UE fast-path render server.

Collates every ``scene*.xml`` under a mujoco_menagerie checkout. Launches the UE
fast-path viewer ONCE, then on each pick compiles the chosen scene to a
version-matched MJB (the UE MuJoCo is 3.11.x, so a stock-mujoco MJB would fail to
load) and ships the MJB **bytes over the wire** to the running viewer via the
``fastpath_load`` RPC. The viewer retires its current model and rebuilds from the
new one live -- no relaunch. Because the model travels over the network, the
renderer can be on a different machine (set URLAB_RENDER_HOST); this is a render
server, not a local file swap.

Run:  uv run python scripts/menagerie_swap.py     (needs the bridge zmq/msgpack deps)
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

from urlab_client.transports import make_transport

# --- config (all overridable by env) -------------------------------------- #
MENAGERIE = os.environ.get(
    "URLAB_MENAGERIE", r"C:/Users/jonat/Documents/GitHub/mujoco_menagerie")
MJBCOMPILE = os.environ.get(
    "URLAB_MJBCOMPILE", r"C:/Users/jonat/Documents/mjb_test/mjbcompile.exe")
UE_EXE = os.environ.get(
    "URLAB_UE_EXE",
    r"C:/Program Files/Epic Games/UE_5.7/Engine/Binaries/Win64/UnrealEditor.exe")
UPROJECT = os.environ.get(
    "URLAB_UPROJECT",
    r"C:/Users/jonat/Documents/Unreal Projects/url_proj/url_proj.uproject")
BOOT_MAP = os.environ.get("URLAB_BOOT_MAP", "/Engine/Maps/Entry")
OUT_DIR = os.environ.get(
    "URLAB_MJB_OUT",
    os.path.join(os.environ.get("TEMP", "."), "urlab_menagerie_mjb"))
# The renderer's RPC address. localhost by default; point at another box to drive
# a remote render server.
RENDER_HOST = os.environ.get("URLAB_RENDER_HOST", "tcp://localhost")
RENDER_PORT = int(os.environ.get("URLAB_RENDER_PORT", "5559"))
# Launch a local viewer (set 0 if a renderer is already running / remote).
LAUNCH_LOCAL = os.environ.get("URLAB_LAUNCH_LOCAL", "1") != "0"


def find_scenes() -> list[str]:
    return sorted(glob.glob(os.path.join(MENAGERIE, "*", "scene*.xml")))


def label(scene_xml: str) -> str:
    return f"{os.path.basename(os.path.dirname(scene_xml))}/{os.path.basename(scene_xml)}"


def compile_mjb(scene_xml: str) -> str:
    """scene.xml -> a version-matched .mjb via the fork's mjbcompile."""
    os.makedirs(OUT_DIR, exist_ok=True)
    robot = os.path.basename(os.path.dirname(scene_xml))
    stem = os.path.splitext(os.path.basename(scene_xml))[0]
    out = os.path.join(OUT_DIR, f"{robot}_{stem}.mjb")
    tool_dir = os.path.dirname(MJBCOMPILE)
    env = dict(os.environ, PATH=tool_dir + os.pathsep + os.environ.get("PATH", ""))
    subprocess.run([MJBCOMPILE, scene_xml, out], check=True, cwd=tool_dir, env=env)
    return out


_transport = None


def send_load(mjb_path: str) -> dict:
    """Ship the MJB bytes to the running renderer over the fastpath_load RPC."""
    global _transport
    if _transport is None:
        _transport = make_transport(
            "zmq", address=RENDER_HOST, step_port=RENDER_PORT, state_port=5555,
            recv_timeout_ms=30000)
    with open(mjb_path, "rb") as f:
        data = f.read()
    # msgpack packs `bytes` as a bin frame; UE reads it as base64 under mjb__b64__.
    return dict(_transport.rpc({"op": "fastpath_load", "mjb": data}))


def launch_viewer(initial_mjb: str) -> "subprocess.Popen | None":
    """Boot the viewer into an empty map with the first scene. The fast-path
    launcher builds it, spawns the light rig, and stands up the RPC-enabled Direct
    manager -- which subsequent live swaps talk to."""
    if not LAUNCH_LOCAL:
        return None
    return subprocess.Popen([
        UE_EXE, UPROJECT, BOOT_MAP, "-game", f"-URLabFastMjb={initial_mjb}",
        "-URLabFastDirect", "-windowed", "-resx=1280", "-resy=720",
    ])


def main() -> None:
    if not os.path.exists(MJBCOMPILE):
        sys.exit(f"mjbcompile not found at {MJBCOMPILE} (set URLAB_MJBCOMPILE)")
    scenes = find_scenes()
    if not scenes:
        sys.exit(f"no scene*.xml under {MENAGERIE} (set URLAB_MENAGERIE)")

    viewer = None
    launched = False
    try:
        while True:
            print()
            for i, s in enumerate(scenes):
                print(f"[{i:2}] {label(s)}")
            sel = input("\npick # to swap live (q to quit): ").strip().lower()
            if sel in ("q", "quit", "exit"):
                break
            if not sel.isdigit() or not (0 <= int(sel) < len(scenes)):
                print("  ? enter a listed number, or q")
                continue
            scene = scenes[int(sel)]
            print(f"  compiling {label(scene)} ...")
            try:
                mjb = compile_mjb(scene)
            except subprocess.CalledProcessError as e:
                print(f"  compile failed ({e}); some scenes need plugins/assets "
                      f"mjbcompile can't handle. Skipping.")
                continue

            if not launched and LAUNCH_LOCAL:
                # First pick: boot the viewer with this scene (brings up the manager
                # + RPC). Later picks swap live over the wire.
                print(f"  launching render-server viewer with {label(scene)} ...")
                viewer = launch_viewer(mjb)
                launched = True
                time.sleep(12)
                continue

            size = os.path.getsize(mjb)
            print(f"  shipping {size / 1e6:.1f} MB to the renderer for a live swap ...")
            try:
                reply = send_load(mjb)
                print(f"  swapped: {reply}")
            except Exception as e:  # noqa: BLE001 - report + keep the loop alive
                print(f"  load RPC failed ({e}); is the viewer up on "
                      f"{RENDER_HOST}:{RENDER_PORT}?")
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
    print("bye")


if __name__ == "__main__":
    main()
