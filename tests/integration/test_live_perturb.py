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

"""Viewer -> owner perturbation round-trip against a live editor.

Proves the "Mirror viewer sends interactive input back to the owner"
path end to end: a Mirror ``AMjRenderer`` forwards a viewer drag as an
``fastpath_perturb`` op over the wire, and the owner receives it,
gates it on the AcceptInput capability, and applies it via
``xfrc_applied`` on its next step.

The test emits ``fastpath_perturb`` directly over the step-port RPC
channel -- exactly the payload ``AMjRenderer::SendPerturbation``
builds -- so it exercises the receiver + apply half without needing a
second UE instance to play the Mirror. Two rollouts from the same
reset, identical (zero) ctrl, differ only by the forwarded wrench; the
perturbed body must move.

The golden scene is a fixed 2-DOF arm with no free-floating body
(its importer bundles every body into one articulation), so the target
is the deepest hinged link rather than a free base. A world-frame force
on that link still drives ``xfrc_applied`` and deflects the joints away
from the controller's hold, which is all the round-trip needs to prove.

Requires ``URLAB_LIVE=1`` + a running editor (the conftest fixtures
bootstrap the golden scene and PIE). Not run in CI; the coordinator
runs it live.
"""

from __future__ import annotations

import numpy as np
import pytest

# World-frame (MuJoCo) force forwarded on the wire, force-only, chosen so a
# component lies off the hinge axis (golden joints spin about +y) and produces a
# clear deflection without slamming both joints into their range limits.
PERTURB_FORCE = [80.0, 0.0, 80.0]
N_STEPS = 60
# Deflection threshold (rad): far above per-step integration noise, far below the
# arm's ~1.5 rad joint range, so the assert is unambiguous either way.
MOVE_THRESHOLD = 1e-2


def _only_articulation(client):
    arts = list(client.articulations.values())
    if len(arts) != 1:
        pytest.skip(f"expected 1 articulation, got {len(arts)}")
    return arts[0]


def _deepest_body_id(art) -> int:
    """The MuJoCo body id of the deepest hinged link -- the largest
    body id among the articulation's joints, which is the tip that gives
    a forwarded force the most leverage. Skips if no joint carries a
    usable body id."""
    body_ids = [int(j.body_id) for j in art.joints.values() if int(j.body_id) > 0]
    if not body_ids:
        pytest.skip("no joint exposes a positive body id to perturb")
    return max(body_ids)


def _perturb(client, body_id: int, force) -> dict:
    """Emit the exact op AMjRenderer::SendPerturbation sends: a
    short-lived, session-less fastpath_perturb with a MuJoCo-frame
    force + torque. Returns the reply dict (asserts the ok echo)."""
    reply = client._rpc(
        "fastpath_perturb",
        {"body": int(body_id), "force": list(force), "torque": [0.0, 0.0, 0.0]},
        expected_op="fastpath_perturb_ok",
    )
    assert int(reply.get("body", -1)) == int(body_id)
    return reply


def _rollout(client, n_steps: int, body_id, force):
    """Reset, then step n_steps. When body_id/force are given, re-forward
    the wrench each step (mirroring a Mirror sending every drag tick).
    Returns the final (qpos, qvel) snapshot."""
    art = _only_articulation(client)
    client.reset()
    for _ in range(n_steps):
        if body_id is not None:
            _perturb(client, body_id, force)
        client.step(n_steps=1)
    return art.qpos_array.copy(), art.qvel_array.copy()


def test_fastpath_perturb_moves_body(pie_client):
    """A forwarded fastpath_perturb (the wire form a Mirror viewer sends)
    is received by the owner and applied via xfrc_applied: the perturbed
    body's pose/velocity diverges from an identical no-perturb control
    rollout."""
    art = _only_articulation(pie_client)
    body_id = _deepest_body_id(art)

    # Clear any wrench a prior aborted run may have latched on this body, so the
    # control rollout is genuinely force-free.
    _perturb(pie_client, body_id, [0.0, 0.0, 0.0])

    try:
        # Control: no perturbation. Default ctrl (0) holds the arm at its reset
        # pose, so this rollout stays near the keyframe default.
        ctrl_qpos, ctrl_qvel = _rollout(pie_client, N_STEPS, None, None)

        # Perturbed: the same reset + ctrl, plus the forwarded world-frame force
        # re-sent each step.
        pert_qpos, pert_qvel = _rollout(pie_client, N_STEPS, body_id, PERTURB_FORCE)
    finally:
        # Release the latched wrench so later tests see a clean owner.
        _perturb(pie_client, body_id, [0.0, 0.0, 0.0])
        pie_client.reset()

    assert ctrl_qpos.size > 0, "articulation reported no qpos"

    qpos_delta = float(np.max(np.abs(pert_qpos - ctrl_qpos)))
    qvel_delta = float(np.max(np.abs(pert_qvel - ctrl_qvel)))
    assert qpos_delta > MOVE_THRESHOLD, (
        f"forwarded perturbation did not move the body: "
        f"max|dqpos|={qpos_delta:.3e} (threshold {MOVE_THRESHOLD:.0e}), "
        f"max|dqvel|={qvel_delta:.3e}"
    )
