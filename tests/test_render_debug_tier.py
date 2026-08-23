# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tiered render frame: the transform tier is always present, the debug tier
(source-of-truth 8.2) is capability-gated and count-capped.

`FastPathOwner._append_render_debug_fields` must:
  * add ZERO extra keys (and zero extra bytes) when a subscriber requests neither
    StreamContacts nor StreamOverlay -- a lean mirror pays nothing;
  * under StreamContacts, append the contact list capped at caps['max_contacts'],
    each entry carrying pos/frame/dist/force[6]/dim/g1/g2;
  * under StreamOverlay, append the derived-decor bundle, each array sized by its
    model dimension (xfrc_applied=6*nbody, subtree_com=3*nbody, ctrl=nu, ...).

Hermetic: builds a tiny mjModel/mjData with real contacts, calls the method on a
bare owner instance (it uses only its args, not owner state). No bus, no display.
"""
from __future__ import annotations

import msgpack
import pytest

mujoco = pytest.importorskip("mujoco")

from urlab_client.fastpath_owner import FastPathOwner  # noqa: E402

# Plane + two boxes penetrating it (>4 contacts) + an actuated hinge arm and a
# sensor, so nu>0 and nsensordata>0 exercise the overlay bundle's dimensions.
_SCENE_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="boxA" pos="0 0 0.05"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
    <body name="boxB" pos="0.5 0 0.05"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
    <body name="arm" pos="2 0 0.5">
      <joint name="h" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0.1 0 0"/>
    </body>
  </worldbody>
  <actuator><motor name="m" joint="h" gear="1"/></actuator>
  <sensor><jointpos name="hp" joint="h"/></sensor>
</mujoco>
"""

_RENDER_KEYS = {"cxpos", "cxquat", "ucpos", "ucfwd", "ucup", "bxpos", "bxquat"}
_DEBUG_KEYS = {
    "contacts", "xfrc_applied", "subtree_com", "ctrl", "act", "wrap_xpos",
    "wrap_obj", "ten_wrapadr", "ten_wrapnum", "eq_active", "eq_anchor",
    "sensordata", "light_xpos", "light_xdir",
}


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # populates data.ncon
    assert data.ncon > 4, "scene must have >4 contacts to exercise the cap"
    return model, data


def _owner():
    # The method uses only (frame, model, data, caps) -- never owner state -- so a
    # bare instance is enough and avoids opening sockets/registry in __init__.
    return object.__new__(FastPathOwner)


def _base_frame():
    # A minimal transform-tier frame (what _build_render_frame would have produced).
    return {"frame": 7, "bxpos": [0.0, 0.0, 0.0], "bxquat": [1.0, 0.0, 0.0, 0.0]}


def test_debug_fields_none_when_no_caps(scene):
    model, data = scene
    owner = _owner()

    # caps=None -> immediate return, frame untouched (byte-identical).
    frame = _base_frame()
    before = msgpack.packb(frame, use_bin_type=True)
    owner._append_render_debug_fields(frame, model, data, None)
    assert not (_DEBUG_KEYS & set(frame)), "no debug keys with caps=None"
    assert msgpack.packb(frame, use_bin_type=True) == before

    # caps present but both toggles off -> still zero extra bytes.
    frame2 = _base_frame()
    before2 = msgpack.packb(frame2, use_bin_type=True)
    owner._append_render_debug_fields(
        frame2, model, data,
        {"contacts": False, "overlay": False, "max_contacts": 0},
    )
    assert not (_DEBUG_KEYS & set(frame2))
    assert msgpack.packb(frame2, use_bin_type=True) == before2


def test_debug_contacts_capped_and_shaped(scene):
    model, data = scene
    owner = _owner()

    frame = _base_frame()
    owner._append_render_debug_fields(
        frame, model, data,
        {"contacts": True, "overlay": False, "max_contacts": 4},
    )
    # StreamContacts only -> contacts key present, capped at 4; no overlay keys.
    assert "contacts" in frame
    assert len(frame["contacts"]) == 4  # scene has >4, cap wins
    for c in frame["contacts"]:
        assert set(c) == {"pos", "frame", "dist", "force", "dim", "g1", "g2"}
        assert len(c["pos"]) == 3
        assert len(c["frame"]) == 9
        assert len(c["force"]) == 6
        assert isinstance(c["dim"], int)
    assert not ((_DEBUG_KEYS - {"contacts"}) & set(frame)), "no overlay bundle"

    # cap==0 (unset) means "no cap" -> full contact list.
    frame_all = _base_frame()
    owner._append_render_debug_fields(
        frame_all, model, data,
        {"contacts": True, "overlay": False, "max_contacts": 0},
    )
    assert len(frame_all["contacts"]) == int(data.ncon)


def test_debug_overlay_bundle_sized_by_model(scene):
    model, data = scene
    owner = _owner()

    frame = _base_frame()
    owner._append_render_debug_fields(
        frame, model, data,
        {"contacts": False, "overlay": True, "max_contacts": 0},
    )
    # StreamOverlay only -> the decor bundle, no contacts.
    assert "contacts" not in frame
    assert len(frame["xfrc_applied"]) == 6 * model.nbody
    assert len(frame["subtree_com"]) == 3 * model.nbody
    assert len(frame["ctrl"]) == model.nu            # nu == 1
    assert len(frame["sensordata"]) == model.nsensordata
    # na == 0 in this scene -> the act key is omitted (appended only when na>0).
    assert "act" not in frame
