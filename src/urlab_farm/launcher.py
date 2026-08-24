# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render-farm launcher: spawn / stop / list Unreal editor instances.

Each instance is a standalone editor process launched with
``-URLabInstanceIndex=i`` so the UE side derives distinct step / state / camera
ports (``PortBase + i*PortStride + {0,1,2}``) and a per-PID SHM session, and
writes its registry file for :class:`urlab_client.URLabPool` discovery. This
module owns process spawning, cross-platform termination, and (optionally)
per-instance project isolation.

Project isolation (``--isolate-projects``, default OFF)
-------------------------------------------------------
Two editors can already share one project directory for pure rendering (no
asset writes), so isolation is off by default and stays simple.
When enabled, this uses the LIGHTEST correct approach: a per-instance working
directory holding a copy of the ``.uproject`` and ``Config/``, with ``Content/``
and ``Plugins/`` linked back to the source project via an OS junction (Windows
``mklink /J``) or symlink (POSIX ``os.symlink``), and fresh per-instance
``Saved/`` and ``Intermediate/`` directories.

Tradeoff: this isolates each instance's config, logs, DDC and intermediates, so
DirectoryWatcher / AssetRegistry churn from one editor no longer perturbs the
others. It does NOT yet isolate ``Content/`` writes: because ``Content/`` is
junctioned back to the source, an asset-writing import would still land in the
shared tree. That is acceptable for the render-only path, but a heavier
per-instance ``Content/`` copy (or the transient-runtime-import path) is
REQUIRED before farm model-upload starts writing ``.uasset`` files. See
``docs/plan_render_farm.md`` section 3.5.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from urlab_client.pool import (
    DEFAULT_PORT_BASE,
    DEFAULT_PORT_STRIDE,
    InstanceInfo,
    default_registry_dir,
    pid_alive,
    read_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class LaunchedInstance:
    """One spawned editor process and its assigned ports."""

    index: int
    pid: int
    step_port: int
    state_port: int
    cam_base_port: int
    log_path: str
    project_path: str


def default_log_dir() -> str:
    """Per-instance log directory, alongside the registry: ``URLAB_LOG_DIR`` if
    set, else the platform cache ``.../URLab/logs``."""
    override = os.environ.get("URLAB_LOG_DIR")
    if override:
        return override
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    return os.path.join(base, "URLab", "logs")


def _editor_binary_name() -> str:
    return "UnrealEditor.exe" if os.name == "nt" else "UnrealEditor"


def _engine_editor_from_root(root: str) -> Optional[Path]:
    platform_dir = "Win64" if os.name == "nt" else "Linux"
    candidate = (
        Path(root) / "Engine" / "Binaries" / platform_dir / _editor_binary_name()
    )
    return candidate if candidate.is_file() else None


def find_editor(explicit: Optional[str] = None) -> Path:
    """Resolve the UnrealEditor binary.

    Order: ``--editor`` path, then ``$UE_ROOT`` / ``$UNREAL_ENGINE`` engine
    roots, then a small set of platform default install locations. Raises
    :class:`FileNotFoundError` with actionable guidance if none is found.
    """
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        # Allow passing an engine root as --editor too.
        from_root = _engine_editor_from_root(explicit)
        if from_root is not None:
            return from_root
        raise FileNotFoundError(
            f"--editor {explicit!r} is not an UnrealEditor binary or engine root"
        )

    for env_var in ("UE_ROOT", "UNREAL_ENGINE"):
        root = os.environ.get(env_var)
        if root:
            found = _engine_editor_from_root(root)
            if found is not None:
                return found

    exe = _editor_binary_name()
    if os.name == "nt":
        defaults = [
            Path(r"C:\Program Files\Epic Games")
        ]
        for base in defaults:
            if base.is_dir():
                for child in sorted(base.glob("UE_*")):
                    found = _engine_editor_from_root(str(child))
                    if found is not None:
                        return found
    else:
        for base in (
            Path.home() / "UnrealEngine",
            Path("/opt/UnrealEngine"),
            Path("/usr/local/UnrealEngine"),
        ):
            found = _engine_editor_from_root(str(base))
            if found is not None:
                return found

    raise FileNotFoundError(
        f"could not locate {exe}. Pass --editor <path>, or set $UE_ROOT / "
        "$UNREAL_ENGINE to your engine root (the dir containing Engine/)."
    )


def find_project(explicit: Optional[str] = None) -> Path:
    """Resolve the ``.uproject`` to launch: ``--project`` if given, else the
    first ``.uproject`` found walking up from the current directory."""
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"--project {explicit!r} does not exist")
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        matches = sorted(directory.glob("*.uproject"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "no .uproject found walking up from the current directory; "
        "pass --project <path>"
    )


def _link_dir(src: Path, dst: Path) -> None:
    """Create a directory junction/symlink ``dst`` -> ``src`` (no copy).

    Windows uses ``mklink /J`` (a junction, which needs no admin rights);
    POSIX uses ``os.symlink``. Existing ``dst`` is left as-is.
    """
    if dst.exists() or dst.is_symlink():
        return
    if os.name == "nt":
        # cmd's mklink is a shell builtin, so it must run through cmd /c.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OSError(
                f"mklink /J {dst} -> {src} failed: {result.stderr.strip()}"
            )
    else:
        os.symlink(str(src), str(dst), target_is_directory=True)


def isolate_project(project: Path, index: int) -> Path:
    """Create a per-instance project working copy and return its ``.uproject``.

    See the module docstring for the isolation strategy and its tradeoff. The
    working copy lives under ``.../URLab/farm/instance_{index}/`` next to the
    registry. ``Content/`` and ``Plugins/`` are junctioned back to the source;
    ``Config/`` and the ``.uproject`` are copied; ``Saved/`` and
    ``Intermediate/`` are left fresh (per-instance).
    """
    src_root = project.parent.resolve()
    if os.name == "nt":
        cache = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    work_root = Path(cache) / "URLab" / "farm" / f"instance_{index}"
    work_root.mkdir(parents=True, exist_ok=True)

    # Copy the .uproject and Config/ (small, per-instance-mutable).
    dst_uproject = work_root / project.name
    shutil.copy2(project, dst_uproject)
    src_config = src_root / "Config"
    dst_config = work_root / "Config"
    if src_config.is_dir() and not dst_config.exists():
        shutil.copytree(src_config, dst_config)

    # Link the large, shared trees back to the source.
    for shared in ("Content", "Plugins"):
        src_dir = src_root / shared
        if src_dir.is_dir():
            _link_dir(src_dir, work_root / shared)

    # Saved/ + Intermediate/ stay per-instance: UE creates them on launch.
    return dst_uproject


def _spawn_process(cmd: List[str], log_path: str) -> subprocess.Popen:
    """Spawn a detached editor process with std streams discarded (UE logs to
    the ``-abslog`` file). Cross-platform detach: a new process group on
    Windows, a new session on POSIX, so terminating this launcher never signals
    the editors."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        start_new_session = True
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
        start_new_session=start_new_session,
    )


def build_launch_command(
    editor: Path,
    project: Path,
    index: int,
    *,
    port_base: int,
    port_stride: int,
    headless: bool,
    log_path: str,
) -> List[str]:
    """Assemble the editor command line for one instance."""
    cmd: List[str] = [
        str(editor),
        str(project),
        f"-URLabInstanceIndex={index}",
        f"-URLabPortBase={port_base}",
        f"-URLabPortStride={port_stride}",
        f"-abslog={log_path}",
        # Unattended + no splash so a farm of editors starts without prompts.
        "-unattended",
        "-nosplash",
        "-stdout",
        "-fullstdoutlogoutput",
    ]
    if headless:
        # Real GPU still required for readback -- RenderOffscreen (NOT nullrhi).
        cmd.append("-RenderOffscreen")
    return cmd


def up(
    count: int,
    *,
    port_base: int = DEFAULT_PORT_BASE,
    port_stride: int = DEFAULT_PORT_STRIDE,
    project: Optional[str] = None,
    editor: Optional[str] = None,
    headless: bool = False,
    isolate_projects: bool = False,
    log_dir: Optional[str] = None,
) -> List[LaunchedInstance]:
    """Spawn ``count`` editor instances and return their assigned ports.

    Raises :class:`FileNotFoundError` (before spawning anything) if the editor
    or project cannot be resolved.
    """
    if count < 1:
        raise ValueError(f"--count must be >= 1, got {count}")
    editor_path = find_editor(editor)
    project_path = find_project(project)
    logs_root = log_dir or default_log_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    launched: List[LaunchedInstance] = []
    for index in range(count):
        step = port_base + index * port_stride
        state = step + 1
        cam_base = step + 2
        inst_project = (
            isolate_project(project_path, index)
            if isolate_projects
            else project_path
        )
        log_path = os.path.join(logs_root, f"urlab_farm_i{index}_{stamp}.log")
        cmd = build_launch_command(
            editor_path,
            inst_project,
            index,
            port_base=port_base,
            port_stride=port_stride,
            headless=headless,
            log_path=log_path,
        )
        proc = _spawn_process(cmd, log_path)
        launched.append(
            LaunchedInstance(
                index=index,
                pid=proc.pid,
                step_port=step,
                state_port=state,
                cam_base_port=cam_base,
                log_path=log_path,
                project_path=str(inst_project),
            )
        )
        logger.info(
            "launched instance %d pid=%d step=%d state=%d cam_base=%d log=%s",
            index, proc.pid, step, state, cam_base, log_path,
        )
    return launched


def kill_pid(pid: int, *, timeout_s: float = 5.0) -> bool:
    """Terminate a process tree cross-platform. Returns True if the process is
    gone afterward. Windows uses ``taskkill /T /F`` (tree + force); POSIX sends
    SIGTERM, waits, then SIGKILL."""
    if not pid_alive(pid):
        return True
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
        )
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not pid_alive(pid):
                return True
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
    # Final confirmation.
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return not pid_alive(pid)


def down(registry_dir: Optional[str] = None) -> List[InstanceInfo]:
    """Terminate every instance in the registry and remove its file.

    Returns the list of instances that were acted on (including entries whose
    PID was already dead, whose stale file is cleaned up anyway)."""
    directory = registry_dir or default_registry_dir()
    instances = read_registry(directory, include_dead=True, include_stale=True)
    for inst in instances:
        if inst.pid > 0:
            killed = kill_pid(inst.pid)
            logger.info(
                "instance %s pid=%d terminated=%s",
                inst.instance_id or "?", inst.pid, killed,
            )
        if inst.registry_path and os.path.exists(inst.registry_path):
            try:
                os.remove(inst.registry_path)
            except OSError as exc:  # pragma: no cover - best-effort
                logger.warning("could not remove %s: %s", inst.registry_path, exc)
    return instances


def ps(registry_dir: Optional[str] = None) -> List[InstanceInfo]:
    """Return all registry entries (alive and dead) for display."""
    directory = registry_dir or default_registry_dir()
    return read_registry(directory, include_dead=True, include_stale=True)


__all__ = [
    "LaunchedInstance",
    "build_launch_command",
    "default_log_dir",
    "down",
    "find_editor",
    "find_project",
    "isolate_project",
    "kill_pid",
    "ps",
    "up",
]
