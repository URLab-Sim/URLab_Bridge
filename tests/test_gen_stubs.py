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

"""Tests for tools/gen_stubs.py.

Drives the generator in offline mode (--from-json) so we don't need a
live editor in CI. Verifies the rendered .pyi has the right shape:
namespace classes, URLabClient with namespace attributes, op stubs
under the right namespace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def gen_stubs_module():
    # Importable as a script via `python -m tools.gen_stubs` — pull it
    # in by file path so we don't rely on the runner's CWD.
    import importlib.util

    here = Path(__file__).resolve().parent
    src = here.parent / "tools" / "gen_stubs.py"
    spec = importlib.util.spec_from_file_location("urlab_gen_stubs", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _meta_payload() -> dict:
    return {
        "op": "meta_ok",
        "ops": [
            {"name": "spawn_actor",  "category": "editor_only",      "namespace": "scene",
             "required_fields": ["blueprint", "target"]},
            {"name": "destroy_actor","category": "editor_only",      "namespace": "scene"},
            {"name": "step",         "category": "manager_required", "namespace": "sim"},
            {"name": "begin_pie",    "category": "editor_only",      "namespace": "runtime"},
            {"name": "list_actors",  "category": "editor_only",      "namespace": "outliner"},
        ],
    }


def test_render_emits_namespace_classes(gen_stubs_module):
    out = gen_stubs_module.render(_meta_payload()["ops"])
    assert "class _Scene:" in out
    assert "class _Sim:" in out
    assert "class _Runtime:" in out
    assert "class _Outliner:" in out


def test_render_routes_op_to_correct_namespace(gen_stubs_module):
    out = gen_stubs_module.render(_meta_payload()["ops"])
    # spawn_actor lives in scene, NOT in sim
    scene_block = out.split("class _Scene:")[1].split("class ")[0]
    sim_block = out.split("class _Sim:")[1].split("class ")[0]
    assert "def spawn_actor(" in scene_block
    assert "def spawn_actor(" not in sim_block
    assert "def step(" in sim_block
    assert "def step(" not in scene_block


def test_render_required_fields_become_kwonly_params(gen_stubs_module):
    out = gen_stubs_module.render(_meta_payload()["ops"])
    # spawn_actor declared blueprint + target as required.
    scene_block = out.split("class _Scene:")[1].split("class ")[0]
    spawn_line = next(line for line in scene_block.splitlines() if "spawn_actor" in line)
    assert "blueprint: Any" in spawn_line
    assert "target: Any" in spawn_line
    assert "*" in spawn_line  # keyword-only marker
    assert "**kwargs: Any" in spawn_line


def test_render_urlab_client_exposes_namespace_attrs(gen_stubs_module):
    out = gen_stubs_module.render(_meta_payload()["ops"])
    client_block = out.split("class URLabClient:")[1]
    assert "scene: _Scene" in client_block
    assert "sim: _Sim" in client_block
    assert "runtime: _Runtime" in client_block
    assert "outliner: _Outliner" in client_block


def test_render_emits_reply_field_summary(gen_stubs_module):
    """reply_fields drive a per-op `# reply: {...}` summary so IDEs can
    show callers the reply schema without a cast."""
    ops = [{
        "name": "pie_status",
        "category": "editor_only",
        "namespace": "sim",
        "reply_fields": [
            "op:string",
            "state:string",
            "compile_error:string",
            "sim_time:float?",
        ],
    }]
    out = gen_stubs_module.render(ops)
    sim_block = out.split("class _Sim:")[1].split("class ")[0]
    assert "def pie_status(" in sim_block
    assert "# reply:" in sim_block
    assert "state: str" in sim_block
    assert "?sim_time: float" in sim_block  # the ? marks optional


def test_main_writes_output_file(gen_stubs_module, tmp_path: Path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_meta_payload()), encoding="utf-8")
    out = tmp_path / "out_stubs.pyi"
    rc = gen_stubs_module.main([
        "--from-json", str(schema),
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "AUTO-GENERATED" in text
    assert "class URLabClient:" in text
