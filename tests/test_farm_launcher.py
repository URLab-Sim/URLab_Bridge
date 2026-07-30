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

"""Unit tests for the urlab_farm launcher: editor / project resolution and the
launch command line. No editor process is spawned."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from urlab_farm import launcher


def test_find_editor_explicit_file(tmp_path):
    exe = tmp_path / launcher._editor_binary_name()
    exe.write_text("stub")
    assert launcher.find_editor(str(exe)) == exe


def test_find_editor_from_env_root(tmp_path, monkeypatch):
    platform_dir = "Win64" if os.name == "nt" else "Linux"
    bindir = tmp_path / "Engine" / "Binaries" / platform_dir
    bindir.mkdir(parents=True)
    exe = bindir / launcher._editor_binary_name()
    exe.write_text("stub")
    monkeypatch.delenv("UNREAL_ENGINE", raising=False)
    monkeypatch.setenv("UE_ROOT", str(tmp_path))
    assert launcher.find_editor() == exe


def test_find_editor_missing_raises(monkeypatch):
    monkeypatch.delenv("UE_ROOT", raising=False)
    monkeypatch.delenv("UNREAL_ENGINE", raising=False)
    with pytest.raises(FileNotFoundError):
        launcher.find_editor("Z:/no/such/editor.exe")


def test_find_project_explicit(tmp_path):
    proj = tmp_path / "Demo.uproject"
    proj.write_text("{}")
    assert launcher.find_project(str(proj)) == proj


def test_find_project_walks_up(tmp_path, monkeypatch):
    proj = tmp_path / "Demo.uproject"
    proj.write_text("{}")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert launcher.find_project() == proj


def test_find_project_missing_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        launcher.find_project()


def test_build_launch_command_flags(tmp_path):
    editor = tmp_path / "UnrealEditor"
    project = tmp_path / "Demo.uproject"
    cmd = launcher.build_launch_command(
        editor, project, 3,
        port_base=5559, port_stride=10, headless=True,
        log_path=str(tmp_path / "i3.log"),
    )
    assert cmd[0] == str(editor)
    assert cmd[1] == str(project)
    assert "-URLabInstanceIndex=3" in cmd
    assert "-URLabPortBase=5559" in cmd
    assert "-URLabPortStride=10" in cmd
    assert any(a.startswith("-abslog=") for a in cmd)
    assert "-RenderOffscreen" in cmd
    assert "-nullrhi" not in cmd


def test_build_launch_command_not_headless(tmp_path):
    cmd = launcher.build_launch_command(
        Path("ed"), Path("p.uproject"), 0,
        port_base=5559, port_stride=10, headless=False,
        log_path="x.log",
    )
    assert "-RenderOffscreen" not in cmd


def test_up_count_validation():
    with pytest.raises(ValueError):
        launcher.up(0)


def test_kill_pid_already_dead():
    # A dead PID is trivially "gone".
    assert launcher.kill_pid(4_000_000_000) is True
