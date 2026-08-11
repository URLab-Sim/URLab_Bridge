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

"""Unit tests for the render-farm pool: registry parsing, PID liveness,
capability filtering, discovery, and transport=auto locality detection. None
require a live editor."""

from __future__ import annotations

import json
import os
import socket
import time

import pytest

from urlab_client.pool import (
    DEFAULT_PORT_BASE,
    DEFAULT_PORT_STRIDE,
    InstanceInfo,
    URLabPool,
    _derive_ports,
    _parse_endpoint,
    default_registry_dir,
    pid_alive,
    read_registry,
)


# -- pid_alive --------------------------------------------------------------

def test_pid_alive_self():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_invalid():
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_pid_alive_dead_pid():
    # A very high PID is almost certainly not running.
    assert pid_alive(4_000_000_000) is False


# -- port + endpoint helpers ------------------------------------------------

def test_derive_ports_default():
    step, state, cam = _derive_ports(0)
    assert (step, state, cam) == (5559, 5560, 5561)
    step, state, cam = _derive_ports(2)
    assert (step, state, cam) == (5579, 5580, 5581)


def test_parse_endpoint_forms():
    assert _parse_endpoint("host:6000") == ("host", 6000)
    assert _parse_endpoint("tcp://1.2.3.4:5559") == ("1.2.3.4", 5559)
    assert _parse_endpoint("myhost") == ("myhost", DEFAULT_PORT_BASE)


def test_default_registry_dir_env_override(monkeypatch):
    monkeypatch.setenv("URLAB_REGISTRY_DIR", os.path.join("X", "reg"))
    assert default_registry_dir() == os.path.join("X", "reg")


# -- InstanceInfo -----------------------------------------------------------

def _reg_dict(**over):
    base = {
        "instance_id": "inst-a",
        "index": 1,
        "pid": os.getpid(),
        "host": "myhost",
        "step_port": 5569,
        "state_port": 5570,
        "cam_base_port": 5571,
        "manager_present": True,
        "busy": False,
        "urlab_version": "urlab/test",
        "capabilities": ["render_sync", "shm_rpc"],
        "registry_written_at": 123.0,
    }
    base.update(over)
    return base


def test_instance_from_registry_and_address():
    inst = InstanceInfo.from_registry(_reg_dict())
    assert inst.address == "tcp://myhost"
    assert inst.step_port == 5569
    assert inst.source == "registry"
    assert inst.capabilities == ("render_sync", "shm_rpc")


def test_instance_has_caps_superset():
    inst = InstanceInfo.from_registry(_reg_dict())
    assert inst.has_caps(None) is True
    assert inst.has_caps([]) is True
    assert inst.has_caps(["render_sync"]) is True
    assert inst.has_caps(["render_sync", "shm_rpc"]) is True
    assert inst.has_caps(["render_sync", "gpu_direct"]) is False


def test_instance_is_stale():
    inst = InstanceInfo.from_registry(_reg_dict(), mtime=time.time() - 100.0)
    assert inst.is_stale(ttl_s=30.0) is True
    assert inst.is_stale(ttl_s=200.0) is False
    # No mtime (static instance) is never stale.
    fresh = InstanceInfo.from_registry(_reg_dict())
    assert fresh.is_stale(ttl_s=1.0) is False


def test_instance_from_block_uses_reachable_host():
    block = _reg_dict(host="0.0.0.0")
    inst = InstanceInfo.from_instance_block(block, host="10.0.0.5", step_port=5559)
    assert inst.host == "10.0.0.5"
    assert inst.step_port == 5559
    assert inst.source == "static"


# -- read_registry / discover ----------------------------------------------

def _write_reg(directory, name, data, *, mtime=None):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_read_registry_filters_dead_and_stale(tmp_path):
    directory = str(tmp_path)
    _write_reg(directory, "inst-a_1.json", _reg_dict(instance_id="alive", pid=os.getpid()))
    _write_reg(
        directory,
        "inst-b_2.json",
        _reg_dict(instance_id="dead", pid=4_000_000_000, index=2),
    )
    old = time.time() - 500.0
    _write_reg(
        directory,
        "inst-c_3.json",
        _reg_dict(instance_id="stale", pid=os.getpid(), index=3),
        mtime=old,
    )

    alive = read_registry(directory, ttl_s=30.0)
    ids = {i.instance_id for i in alive}
    assert ids == {"alive"}

    full = read_registry(directory, include_dead=True, include_stale=True)
    assert {i.instance_id for i in full} == {"alive", "dead", "stale"}


def test_read_registry_skips_corrupt_file(tmp_path):
    directory = str(tmp_path)
    _write_reg(directory, "good_1.json", _reg_dict(instance_id="good", pid=os.getpid()))
    with open(os.path.join(directory, "bad_2.json"), "w", encoding="utf-8") as handle:
        handle.write("{not valid json")
    alive = read_registry(directory)
    assert {i.instance_id for i in alive} == {"good"}


def test_pool_discover_filters_busy_and_caps(tmp_path):
    directory = str(tmp_path)
    _write_reg(
        directory, "free_1.json",
        _reg_dict(instance_id="free", pid=os.getpid(), busy=False, index=1),
    )
    _write_reg(
        directory, "busy_2.json",
        _reg_dict(instance_id="busy", pid=os.getpid(), busy=True, index=2),
    )
    _write_reg(
        directory, "nocaps_3.json",
        _reg_dict(instance_id="nocaps", pid=os.getpid(), index=3,
                  capabilities=["render_sync"]),
    )

    free = URLabPool.discover(directory)
    assert {i.instance_id for i in free} == {"free", "nocaps"}

    with_busy = URLabPool.discover(directory, include_busy=True)
    assert {i.instance_id for i in with_busy} == {"free", "busy", "nocaps"}

    need_shm = URLabPool.discover(directory, require_caps=["shm_rpc"])
    assert {i.instance_id for i in need_shm} == {"free"}


def test_pool_lease_empty_candidates_raises():
    with pytest.raises(RuntimeError):
        URLabPool.lease([])


# -- transport=auto locality detection --------------------------------------

def _make_client(**kw):
    from urlab_client.client import URLabClient
    return URLabClient(transport="auto", **kw)


def test_locality_same_host_upgrades_to_shm(tmp_path):
    client = _make_client()
    client.instance = {"host": socket.gethostname()}
    client.shm_session_dir = str(tmp_path)  # a dir that exists locally
    client._detect_locality_transport()
    assert client._want_shm is True


def test_locality_remote_stays_zmq(tmp_path):
    client = _make_client()
    client.instance = {"host": "some-other-machine-99"}
    client.shm_session_dir = str(tmp_path)
    client._detect_locality_transport()
    assert client._want_shm is False


def test_locality_same_host_but_missing_shm_dir_stays_zmq():
    client = _make_client()
    client.instance = {"host": socket.gethostname()}
    client.shm_session_dir = os.path.join("Z", "does", "not", "exist")
    client._detect_locality_transport()
    assert client._want_shm is False


def test_locality_no_instance_block_stays_zmq(tmp_path):
    client = _make_client()
    client.instance = {}
    client.shm_session_dir = str(tmp_path)
    client._detect_locality_transport()
    assert client._want_shm is False


def test_explicit_transport_pref_not_auto():
    from urlab_client.client import URLabClient
    zmq_client = URLabClient(transport="zmq")
    assert zmq_client._transport_pref == "zmq"
    assert zmq_client._want_shm is False
    shm_client = URLabClient(transport="shm")
    assert shm_client._transport_pref == "shm"
    assert shm_client._want_shm is True


def test_unknown_transport_raises():
    from urlab_client.client import URLabClient
    with pytest.raises(ValueError):
        URLabClient(transport="carrier-pigeon")
