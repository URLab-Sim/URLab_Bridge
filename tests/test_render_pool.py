# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Unit tests for RenderPool: endpoint parsing, auto camera distribution, the
parallel fan-out + merge, user-cam ownership, and failure aggregation. Pure
Python with fake clients -- no UE server, no mujoco."""
from __future__ import annotations

import json

import pytest

from urlab_client import render_pool
from urlab_client.render_pool import (
    InstanceSpec,
    RenderPool,
    RenderPoolError,
    parse_endpoints,
)

CAMS = ["cam0", "cam1", "cam2", "user"]


class _FakeFrame:
    def __init__(self, name):
        self.name = name


class _FakeClient:
    def __init__(self, host, port, **kw):
        self.host, self.port = host, port
        self.loaded = []
        self.render_calls = []
        self.fail = False

    def camera_names(self):
        return list(CAMS)

    def load_xml(self, xml, **kw):
        self.loaded.append(xml)

    def render(self, *, cameras, user_pose=None, **kw):
        if self.fail:
            raise RuntimeError(f"boom@{self.host}:{self.port}")
        self.render_calls.append({"cameras": list(cameras), "user_pose": user_pose})
        return {c: _FakeFrame(c) for c in cameras}

    def close(self):
        pass


class _FakeRenderClient:
    made = []

    @classmethod
    def grpc(cls, host, port, **kw):
        c = _FakeClient(host, port, **kw)
        cls.made.append(c)
        return c


@pytest.fixture
def fake_clients(monkeypatch):
    _FakeRenderClient.made = []
    monkeypatch.setattr(render_pool, "RenderClient", _FakeRenderClient)
    return _FakeRenderClient


# -- endpoint parsing ------------------------------------------------------
def test_parse_endpoints_string():
    got = parse_endpoints("10.0.0.1:50051, 10.0.0.2:50052")
    assert got == [InstanceSpec("10.0.0.1", 50051), InstanceSpec("10.0.0.2", 50052)]


def test_parse_endpoints_mixed_and_defaults():
    got = parse_endpoints(["h1", ("h2", 5), InstanceSpec("h3", 7), "tcp://h4:9"])
    assert got == [InstanceSpec("h1", 50051), InstanceSpec("h2", 5),
                   InstanceSpec("h3", 7), InstanceSpec("h4", 9)]


def test_parse_endpoints_empty_raises():
    with pytest.raises(ValueError):
        parse_endpoints("")


def test_from_config_file_and_env(tmp_path, monkeypatch, fake_clients):
    cfg = tmp_path / "pool.json"
    cfg.write_text(json.dumps({"instances": [
        {"host": "a", "port": 1}, {"host": "b"}]}))
    pool = RenderPool.from_config(str(cfg))
    assert pool.endpoints == ["a:1", "b:50051"]
    # env fallback
    monkeypatch.setenv("URLAB_RENDER_POOL", str(cfg))
    assert RenderPool.from_config().endpoints == ["a:1", "b:50051"]


def test_from_config_missing_raises(monkeypatch, fake_clients):
    monkeypatch.delenv("URLAB_RENDER_POOL", raising=False)
    with pytest.raises(ValueError):
        RenderPool.from_config()


# -- automatic distribution ------------------------------------------------
@pytest.mark.parametrize("ncam,ninst,expect", [
    (4, 2, [2, 2]),      # exact
    (5, 2, [3, 2]),      # M > N, uneven
    (2, 3, [1, 1, 0]),   # M < N, one idle
    (3, 3, [1, 1, 1]),
])
def test_distribute_even(ncam, ninst, expect, fake_clients):
    pool = RenderPool([("h", 50051 + i) for i in range(ninst)])
    buckets = pool._distribute([f"c{k}" for k in range(ncam)])
    assert [len(b) for b in buckets] == expect
    # every camera assigned exactly once
    flat = [c for b in buckets for c in b]
    assert sorted(flat) == sorted(f"c{k}" for k in range(ncam))


# -- render fan-out + merge + user cam ------------------------------------
def test_render_merges_all_and_dispatches_to_all(fake_clients):
    pool = RenderPool([("h", 50051), ("h", 50052)])
    out = pool.render(bxpos=[0.0], bxquat=[1.0, 0, 0, 0], cameras=CAMS,
                      user_pose=([1, 2, 3], [0, 1, 0], [0, 0, 1]))
    # all requested cameras came back
    assert set(out) == set(CAMS)
    # both instances were actually called
    assert all(c.render_calls for c in _FakeRenderClient.made)


def test_user_pose_only_to_user_owner(fake_clients):
    pool = RenderPool([("h", 50051), ("h", 50052)])
    up = ([1, 2, 3], [0, 1, 0], [0, 0, 1])
    pool.render(bxpos=[0.0], bxquat=[1.0, 0, 0, 0], cameras=CAMS, user_pose=up)
    # exactly one instance -- the one whose bucket holds "user" -- got user_pose
    with_up = [c for c in _FakeRenderClient.made
               if c.render_calls and c.render_calls[-1]["user_pose"] is not None]
    assert len(with_up) == 1
    assert "user" in with_up[0].render_calls[-1]["cameras"]


def test_load_broadcasts_to_all(fake_clients):
    pool = RenderPool([("h", 50051), ("h", 50052), ("h", 50053)])
    pool.load_xml("scene.xml")
    assert all(c.loaded == ["scene.xml"] for c in _FakeRenderClient.made)


def test_failure_aggregates(fake_clients):
    pool = RenderPool([("h", 50051), ("h", 50052)])
    _FakeRenderClient.made[1].fail = True
    with pytest.raises(RenderPoolError) as ei:
        pool.render(bxpos=[0.0], bxquat=[1.0, 0, 0, 0], cameras=CAMS)
    assert "h:50052" in ei.value.failures
