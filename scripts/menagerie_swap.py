#!/usr/bin/env python3
# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Menagerie render-slave demo: drive a UE fast-path render slave from MuJoCo.

This is an *owner*: it holds a MuJoCo model, steps it with random control in
MuJoCo's own viewer, and streams the motion to a UE fast-path render slave that
mirrors it (puppet mode) with no physics of its own. It does everything the old
aloha demo did, plus three things:

* a **scene picker** over a mujoco_menagerie checkout -- pick a scene and it is
  compiled to a version-matched MJB (UE runs MuJoCo 3.11.x, so a stock-mujoco MJB
  would fail to load) and swapped into the running slave live, bytes over the
  wire via the ``fastpath_load`` RPC (no relaunch, works cross-machine).
* **camera feeds** -- the slave renders each MJB camera and publishes it; a
  companion OpenCV viewer window shows them.
* **copycat camera** -- the slave points its game viewport at MuJoCo's own
  free/user camera, so orbiting the MuJoCo viewer orbits the UE view in lockstep.

Run interactively (pick scenes by number):
    uv run python scripts/menagerie_swap.py

Run autonomously (cycle a few scenes, spin the copycat camera -- no keyboard):
    URLAB_SCENES=aloha,franka_emika_panda,unitree_g1 uv run python scripts/menagerie_swap.py

Drive a slave that is already running / on another box:
    URLAB_LAUNCH_LOCAL=0 URLAB_RENDER_HOST=tcp://<host> uv run python scripts/menagerie_swap.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:  # pragma: no cover
    sys.exit("this demo needs `mujoco` (pip install mujoco / uv add mujoco)")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402
from urlab_client.transports import make_transport  # noqa: E402

# --- config (all overridable by env) -------------------------------------- #
MENAGERIE = os.environ.get(
    "URLAB_MENAGERIE", r"C:/Users/jonat/Documents/GitHub/mujoco_menagerie")
MJBCOMPILE = os.environ.get(
    "URLAB_MJBCOMPILE", r"C:/Users/jonat/Documents/mjb_test/mjbcompile.exe")
UE_EXE = os.environ.get(
    "URLAB_UE_EXE",
    r"C:/Users/jonat/Documents/Unreal Projects/url_proj/Saved/StagedBuilds/"
    r"Windows/url_proj/Binaries/Win64/url_proj.exe")
BOOT_MAP = os.environ.get("URLAB_BOOT_MAP", "/Engine/Maps/Entry")
# Editor exe + project: used when a base level is chosen, so its (possibly
# uncooked) map + assets load through the editor's -game with no full cook.
UE_EDITOR = os.environ.get(
    "URLAB_UE_EDITOR",
    r"C:/Program Files/Epic Games/UE_5.7/Engine/Binaries/Win64/UnrealEditor.exe")
UPROJECT = os.environ.get(
    "URLAB_UPROJECT",
    r"C:/Users/jonat/Documents/Unreal Projects/url_proj/url_proj.uproject")
# Curated base level for the render slave (its own lights/props). If set, the
# slave boots this map with -URLabScene=base (default lights suppressed) and
# runs through the editor so an uncooked project map still loads.
BASE_MAP = os.environ.get("URLAB_BASE_MAP", "")
# World spot to drop the MJB at, "X,Y,Z" in UE cm (empty = world origin).
ORIGIN = os.environ.get("URLAB_ORIGIN", "")
OUT_DIR = os.environ.get(
    "URLAB_MJB_OUT",
    os.path.join(os.environ.get("TEMP", "."), "urlab_menagerie_mjb"))
# The slave's RPC address (fastpath_load). localhost by default; point at another
# box to drive a remote render slave.
RENDER_HOST = os.environ.get("URLAB_RENDER_HOST", "tcp://localhost")
RENDER_PORT = int(os.environ.get("URLAB_RENDER_PORT", "5559"))
# Transform bus the slave subscribes to (this owner binds it).
BUS_PORT = int(os.environ.get("URLAB_BUS_PORT", "5561"))
CAM_BASE_PORT = int(os.environ.get("URLAB_CAM_BASE_PORT", "5600"))
HZ = float(os.environ.get("URLAB_HZ", "60"))
SEED = int(os.environ.get("URLAB_SEED", "0"))
# Launch the slave locally (set 0 if one is already running / remote).
LAUNCH_LOCAL = os.environ.get("URLAB_LAUNCH_LOCAL", "1") != "0"
# Show the slave's camera feeds in an OpenCV window (needs opencv-python).
SHOW_CAMERAS = os.environ.get("URLAB_SHOW_CAMERAS", "1") != "0"
# Autonomous mode: comma-separated scene-name substrings to cycle through with no
# keyboard, spinning the copycat camera so the mirroring is visible unattended.
AUTO_SCENES = [s.strip() for s in os.environ.get("URLAB_SCENES", "").split(",") if s.strip()]
AUTO_DWELL_S = float(os.environ.get("URLAB_DWELL", "12"))
# GUI scene picker: a Tk window to manually choose which scene to load (random
# control stays on). Default when no URLAB_SCENES auto-cycle is given; force with
# URLAB_PICKER=1 even alongside a scene list.
PICKER = os.environ.get("URLAB_PICKER", "") not in ("", "0")


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


def _free_camera_pose(model, data, opt, scn, cam):
    """Resolve MuJoCo's free/user camera to a world (eye, forward, up) triple.

    Let MuJoCo do the azimuth/elevation/distance math: mjv_updateScene fills the
    scene's GL cameras with the exact world pose the viewer renders from. The two
    entries are the stereo pair; their midpoint is the mono eye.
    """
    mujoco.mjv_updateScene(
        model, data, opt, None, cam, int(mujoco.mjtCatBit.mjCAT_ALL), scn)
    c0, c1 = scn.camera[0], scn.camera[1]
    pos = (np.array(c0.pos) + np.array(c1.pos)) * 0.5
    fwd = (np.array(c0.forward) + np.array(c1.forward)) * 0.5
    up = (np.array(c0.up) + np.array(c1.up)) * 0.5
    return pos, fwd, up


class Driver:
    """Owns the MuJoCo sim + the bus to the render slave, across live swaps."""

    def __init__(self) -> None:
        self.owner: FastPathOwner | None = None
        self.load_rpc = None
        self.slave_proc: subprocess.Popen | None = None
        self.cam_proc: subprocess.Popen | None = None
        self.viewer = None
        self.model = None
        self.data = None
        self.opt = None
        self.scn = None
        self.rng = np.random.default_rng(SEED)
        self.frame = 0
        self.quat = np.zeros(4)

    # -- model / control ---------------------------------------------------- #
    def _adopt_model(self, scene_xml: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=max(1000, self.model.ngeom))
        lo = self.model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
        hi = self.model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
        unlim = ~self.model.actuator_ctrllimited.astype(bool)
        lo[unlim], hi[unlim] = -1.0, 1.0
        self._lo, self._hi = lo, hi
        self._nsub = max(1, round((1.0 / HZ) / self.model.opt.timestep))

    # -- slave lifecycle ---------------------------------------------------- #
    def _launch_slave(self, initial_mjb: str) -> None:
        if not LAUNCH_LOCAL:
            return
        bus = f"tcp://127.0.0.1:{BUS_PORT}"
        # serve is implicit for a bus mirror but harmless; keep it for clarity.
        caps = ["serve"]
        scene: list[str] = []
        if SHOW_CAMERAS:
            caps.append("cameras")
        if ORIGIN:
            # -URLabScene origin uses ';' separators (env value is "X,Y,Z").
            scene.append(f"origin={ORIGIN.replace(',', ';')}")
        if BASE_MAP:
            # Uncooked project base map: suppress the default light rig.
            scene.append("base")
        common = [
            "-game", f"-URLabDrive=stream:{bus}", f"-URLabModel={initial_mjb}",
            f"-URLabCaps={','.join(caps)}",
            "-windowed", "-resx=1280", "-resy=720",
        ]
        if scene:
            common.append(f"-URLabScene={','.join(scene)}")
        if BASE_MAP:
            # Uncooked project map: run through the editor's -game so its assets
            # load without a full cook.
            args = [UE_EDITOR, UPROJECT, BASE_MAP] + common
            print(f"[slave] editor -game base level {BASE_MAP} -> bus {bus}")
        else:
            args = [UE_EXE, BOOT_MAP] + common
            print(f"[slave] packaged puppet render slave -> bus {bus}")
        self.slave_proc = subprocess.Popen(args)

    def _launch_camera_viewer(self) -> None:
        if not SHOW_CAMERAS:
            return
        script = os.path.join(_HERE, "show_fastpath_cameras.py")
        print(f"[cams] opening camera-feed viewer on 127.0.0.1:{CAM_BASE_PORT}+")
        self.cam_proc = subprocess.Popen(
            [sys.executable, script, "--host", "127.0.0.1",
             "--base-port", str(CAM_BASE_PORT)])

    def _open_viewer(self) -> None:
        """(Re)bind the passive viewer to the current model/data. The passive
        viewer can't swap models in place, so a live scene change closes the old
        window and opens a fresh one on the new sim -- otherwise it would keep
        rendering the first scene while the slave moved on."""
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # -- scene load / swap -------------------------------------------------- #
    def load_scene(self, scene_xml: str) -> bool:
        """Compile + adopt a scene. First call boots the slave + owner + cameras;
        later calls swap the model into the running slave over the wire."""
        try:
            mjb = compile_mjb(scene_xml)
        except subprocess.CalledProcessError as e:
            print(f"  compile failed ({e}); some scenes need assets mjbcompile "
                  f"can't handle. Skipping.")
            return False
        with open(mjb, "rb") as fh:
            mjb_bytes = fh.read()
        self._adopt_model(scene_xml)

        if self.owner is None:
            # First scene: stand everything up. The owner's scene id becomes its
            # registry filename, so it must be slash-free -- use the robot dir name,
            # not label() ("robot/scene.xml"), which would break the registry write.
            robot = os.path.basename(os.path.dirname(scene_xml))
            self.owner = FastPathOwner(
                mjb_bytes, scene=robot, bus_port=BUS_PORT, ngeom=self.model.ngeom)
            self.load_rpc = make_transport(
                "zmq", address=RENDER_HOST, step_port=RENDER_PORT,
                state_port=5555, recv_timeout_ms=30000)
            self._launch_slave(mjb)
            self._launch_camera_viewer()
            self._open_viewer()
            # Only wait when we launched a LOCAL slave; in browser mode the owner has
            # no slave to wait for and must start serving fastpath_hello immediately.
            default_wait = "0" if not LAUNCH_LOCAL else ("45" if BASE_MAP else "12")
            boot_s = float(os.environ.get("URLAB_BOOT_WAIT", default_wait))
            if boot_s > 0:
                print(f"  waiting {boot_s:.0f}s for the render slave to boot ...")
                time.sleep(boot_s)
        else:
            # Keep the MJB served on fastpath_hello current, so a render slave that
            # joins LATER (via the server browser) pulls this scene, not the first one.
            self.owner.update_model(mjb_bytes, self.model.ngeom)
            # Live swap: ship the bytes to any already-connected slave; keep the bus,
            # and reopen the MuJoCo viewer on the new model so both sides match.
            print(f"  shipping {len(mjb_bytes) / 1e6:.1f} MB for a live swap ...")
            try:
                reply = dict(self.load_rpc.rpc({"op": "fastpath_load", "mjb": mjb_bytes}))
                print(f"  swapped: {reply}")
            except Exception as e:  # noqa: BLE001 - non-fatal: a slave can still join
                print(f"  (no connected slave to push to: {e}); "
                      f"streaming anyway for browser joins")
            self._open_viewer()
        print(f"  now streaming {label(scene_xml)}  "
              f"(nbody={self.model.nbody} ngeom={self.model.ngeom} nu={self.model.nu})")
        return True

    # -- streaming ---------------------------------------------------------- #
    def _step_once(self) -> None:
        self.owner.serve_pending()
        if self.model.nu:
            self.data.ctrl[:] = self.rng.uniform(self._lo, self._hi)
        # Apply any renderer-sent perturbations as transient body forces.
        perts = self.owner.drain_perturbations()
        self.data.xfrc_applied[:] = 0.0
        for body_id, ft in perts.items():
            if 0 <= body_id < self.model.nbody:
                self.data.xfrc_applied[body_id] = ft
        for _ in range(self._nsub):
            mujoco.mj_step(self.model, self.data)

        bxpos = np.asarray(self.data.xpos, dtype=np.float64).reshape(-1)
        bxquat = np.asarray(self.data.xquat, dtype=np.float64).reshape(-1)
        cxpos = cxquat = None
        if self.model.ncam:
            cxpos = np.asarray(self.data.cam_xpos, dtype=np.float64).reshape(-1)
            cx = np.asarray(self.data.cam_xmat, dtype=np.float64).reshape(self.model.ncam, 9)
            cxquat = np.empty(self.model.ncam * 4)
            for c in range(self.model.ncam):
                mujoco.mju_mat2Quat(self.quat, cx[c])
                cxquat[4 * c: 4 * c + 4] = self.quat

        # Copycat: mirror MuJoCo's free/user camera to the slave, so orbiting the
        # MuJoCo viewer with the mouse orbits the UE view in lockstep.
        usercam = None
        if self.viewer is not None:
            usercam = _free_camera_pose(
                self.model, self.data, self.opt, self.scn, self.viewer.cam)

        self.owner.publish_bodies(self.frame, bxpos, bxquat, cxpos, cxquat, usercam)
        if self.viewer is not None:
            self.viewer.sync()
        self.frame += 1

    def _deadline_stream(self, seconds: float) -> bool:
        """Stream for `seconds` wall-clock; return False if the viewer was closed."""
        dt = 1.0 / HZ
        end = time.time() + seconds
        while time.time() < end:
            if self.viewer is not None and not self.viewer.is_running():
                return False
            self._step_once()
            time.sleep(dt)
        return True

    def viewer_alive(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        if self.owner is not None:
            self.owner.close()
        for p in (self.cam_proc, self.slave_proc):
            if p is not None and p.poll() is None:
                p.terminate()


def run_auto(driver: Driver, scenes: list[str]) -> None:
    """Cycle the chosen scenes with no keyboard, spinning the copycat camera."""
    picks = []
    for want in AUTO_SCENES:
        match = next((s for s in scenes if want in label(s)), None)
        if match:
            picks.append(match)
        else:
            print(f"[auto] no scene matches '{want}'")
    if not picks:
        sys.exit("[auto] URLAB_SCENES matched no scenes")
    print(f"[auto] cycling {len(picks)} scenes, {AUTO_DWELL_S:.0f}s each; Ctrl-C to stop")
    idx = 0
    while driver.viewer_alive():
        scene = picks[idx % len(picks)]
        print(f"\n[auto] === {label(scene)} ===")
        if driver.load_scene(scene):
            if not driver._deadline_stream(AUTO_DWELL_S):
                break
        idx += 1


def run_gui(driver: Driver, scenes: list[str]) -> None:
    """A Tk window to manually pick which scene to load; random control stays on.
    The window is pumped from the stream loop (root.update) so MuJoCo's viewer, the
    picker, and the transform stream all run together on the main thread."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("[gui] tkinter unavailable; using the terminal picker instead")
        run_interactive(driver, scenes)
        return

    root = tk.Tk()
    root.title("URLab Owner  --  Scene Picker")
    tk.Label(root, text="Menagerie scenes (double-click or Load):").pack(
        anchor="w", padx=6, pady=(6, 0))
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=6)
    scroll = tk.Scrollbar(frame)
    scroll.pack(side="right", fill="y")
    listbox = tk.Listbox(frame, width=52, height=28, yscrollcommand=scroll.set)
    for s in scenes:
        listbox.insert(tk.END, label(s))
    listbox.pack(side="left", fill="both", expand=True)
    scroll.config(command=listbox.yview)
    listbox.selection_set(0)

    status = tk.StringVar(value="pick a scene")
    pending: dict = {"scene": scenes[0]}  # auto-load the first so something streams

    def do_load() -> None:
        sel = listbox.curselection()
        if sel:
            pending["scene"] = scenes[sel[0]]

    def do_browse() -> None:
        path = filedialog.askopenfilename(
            title="Pick a MuJoCo scene XML (any folder)",
            initialdir=MENAGERIE,
            filetypes=[("MuJoCo XML", "*.xml"), ("All files", "*.*")])
        if path:
            pending["scene"] = path

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=6, pady=(0, 4))
    tk.Button(btns, text="Load selected", command=do_load).pack(side="left")
    tk.Button(btns, text="Browse file...", command=do_browse).pack(side="left", padx=6)
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=6, pady=(0, 6))
    listbox.bind("<Double-Button-1>", lambda _e: do_load())
    alive = {"v": True}
    root.protocol("WM_DELETE_WINDOW", lambda: alive.update(v=False))
    print("[gui] scene picker open; the render slave mirrors your selection")

    dt = 1.0 / HZ
    try:
        while alive["v"] and driver.viewer_alive():
            if pending["scene"] is not None:
                scene = pending["scene"]
                pending["scene"] = None
                status.set(f"loading {label(scene)} ...")
                try:
                    root.update()
                except tk.TclError:
                    break
                if driver.load_scene(scene):
                    status.set(f"streaming {label(scene)}  (random control on)")
                else:
                    status.set(f"compile failed for {label(scene)} -- pick another")
            driver._step_once()
            try:
                root.update()
            except tk.TclError:
                break
            time.sleep(dt)
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def run_interactive(driver: Driver, scenes: list[str]) -> None:
    """Pick scenes by number; stream continuously between picks."""
    booted = False
    while driver.viewer_alive():
        print()
        for i, s in enumerate(scenes):
            print(f"[{i:2}] {label(s)}")
        sel = input("\npick # to load/swap live (q to quit): ").strip().lower()
        if sel in ("q", "quit", "exit"):
            break
        if not sel.isdigit() or not (0 <= int(sel) < len(scenes)):
            print("  ? enter a listed number, or q")
            continue
        if not driver.load_scene(scenes[int(sel)]):
            continue
        booted = True
        # Stream until the operator hits Enter to pick again (they orbit the
        # MuJoCo viewer meanwhile; the slave copycats the view).
        print("  streaming -- orbit the MuJoCo viewer to drive the UE view.")
        print("  press Enter here to pick another scene ...")
        _pump_until_enter(driver)
    if not booted:
        print("  (nothing streamed)")


def _pump_until_enter(driver: Driver) -> None:
    """Stream on this thread while a background thread waits for Enter."""
    import threading
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    dt = 1.0 / HZ
    while not stop.is_set() and driver.viewer_alive():
        driver._step_once()
        time.sleep(dt)


def main() -> None:
    if not os.path.exists(MJBCOMPILE):
        sys.exit(f"mjbcompile not found at {MJBCOMPILE} (set URLAB_MJBCOMPILE)")
    if LAUNCH_LOCAL and not os.path.exists(UE_EXE):
        sys.exit(f"render slave exe not found at {UE_EXE} (set URLAB_UE_EXE, or "
                 f"URLAB_LAUNCH_LOCAL=0 to drive a running slave)")
    scenes = find_scenes()
    if not scenes:
        sys.exit(f"no scene*.xml under {MENAGERIE} (set URLAB_MENAGERIE)")

    driver = Driver()
    try:
        if AUTO_SCENES and not PICKER:
            run_auto(driver, scenes)
        else:
            run_gui(driver, scenes)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        driver.close()
    print("bye")


if __name__ == "__main__":
    main()
