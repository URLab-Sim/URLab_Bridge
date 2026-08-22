# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""FastPathOwner perturbation tests.

The mirror has no mjData, so it forwards a drag INTENT and the owner runs the real
MuJoCo spring (mjv_applyPerturbForce): mass-scaled + critically damped, exactly like
simulate's Ctrl-drag. These tests pin that behaviour so it can't silently regress
back to the old undamped/un-scaled force that flew off. A raw-wrench path is also
covered. No GUI.
"""
from __future__ import annotations

import socket

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

# A floor + a single 0.3 kg free-body cube (nbody=2: world + cube). Self-contained.
_CUBE_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="free_cube" pos="0 0 0.7">
      <freejoint/>
      <geom type="box" size="0.12 0.12 0.12" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
"""

_CUBE_BODY = 1          # body id of free_cube (world is 0)
_CUBE_MASS = 0.3
_STIFFNESS = 100.0      # MuJoCo default vis.map.stiffness


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_owner(tmp_path):
    from urlab_client.fastpath_owner import FastPathOwner

    return FastPathOwner(
        b"", scene="cube", model_format="xml", assets={}, ngeom=2,
        control_port=_free_port(), bus_port=_free_port(),
        advertise_host="127.0.0.1", registry_dir=str(tmp_path),
    )


def _model_data():
    model = mujoco.MjModel.from_xml_string(_CUBE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_drag_force_is_mass_scaled(tmp_path):
    """A drag pulls with force = stiffness * localmass * displacement (MuJoCo's own
    spring). For a free body localmass == body mass, so a 0.2 m pull on the 0.3 kg
    cube is ~6 N -- not the old ~100x-too-large value."""
    owner = _make_owner(tmp_path)
    try:
        model, data = _model_data()
        selpos0 = data.xpos[_CUBE_BODY].copy()
        owner.submit_perturb(_CUBE_BODY, True, [0, 0, 0], list(selpos0 + [0.2, 0, 0]))
        data.xfrc_applied[:] = 0.0
        owner.apply_perturbations(model, data)
        f = np.asarray(data.xfrc_applied[_CUBE_BODY][:3])
        expected = _STIFFNESS * _CUBE_MASS * 0.2  # 6.0 N
        assert np.linalg.norm(f) == pytest.approx(expected, rel=0.05)
        assert f[0] > 0 and abs(f[1]) < 1e-6 and abs(f[2]) < 1e-6  # pulls +x only
    finally:
        owner.close()


def test_drag_is_damped_and_settles(tmp_path):
    """Critical damping means the body converges on the target and stays bounded --
    it does NOT oscillate away ('flies off')."""
    owner = _make_owner(tmp_path)
    try:
        model, data = _model_data()
        target = data.xpos[_CUBE_BODY].copy() + [0.3, 0, 0]
        for _ in range(600):
            owner.submit_perturb(_CUBE_BODY, True, [0, 0, 0], list(target))
            data.xfrc_applied[:] = 0.0
            owner.apply_perturbations(model, data)
            mujoco.mj_step(model, data)
        # Settled near the target in x, and never blew up.
        assert data.xpos[_CUBE_BODY][0] == pytest.approx(target[0], abs=0.1)
        assert np.all(np.abs(data.xpos[_CUBE_BODY]) < 5.0)
    finally:
        owner.close()


def test_release_clears_wrench(tmp_path):
    """active=False stops the pull and zeroes the body's xfrc so it stops drifting."""
    owner = _make_owner(tmp_path)
    try:
        model, data = _model_data()
        owner.submit_perturb(_CUBE_BODY, True, [0, 0, 0],
                             list(data.xpos[_CUBE_BODY] + [0.2, 0, 0]))
        data.xfrc_applied[:] = 0.0
        owner.apply_perturbations(model, data)
        assert np.linalg.norm(data.xfrc_applied[_CUBE_BODY][:3]) > 1.0
        owner.submit_perturb(_CUBE_BODY, False, [0, 0, 0], [0, 0, 0])
        data.xfrc_applied[:] = 0.0
        owner.apply_perturbations(model, data)
        assert np.linalg.norm(data.xfrc_applied[_CUBE_BODY]) == pytest.approx(0.0)
    finally:
        owner.close()


def test_raw_wrench_path(tmp_path):
    """The legacy/programmatic raw wrench applies an EXACT force (no spring)."""
    owner = _make_owner(tmp_path)
    try:
        model, data = _model_data()
        owner.submit_perturb_force(_CUBE_BODY, [0, 0, 5.0], [0, 0, 0])
        data.xfrc_applied[:] = 0.0
        owner.apply_perturbations(model, data)
        assert list(data.xfrc_applied[_CUBE_BODY][:3]) == [0.0, 0.0, 5.0]
    finally:
        owner.close()
