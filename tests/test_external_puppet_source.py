from __future__ import annotations

import pytest

from urlab_client import URLabClient
from urlab_client.enums import StepMode

from . import wire_replies as wr


def _simulation(mujoco_mod):
    model = mujoco_mod.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    return model, mujoco_mod.MjData(model)


def test_attach_puppet_simulation_keeps_external_objects_by_identity(mujoco_mod):
    model, data = _simulation(mujoco_mod)
    client = URLabClient(step_mode="auto")

    client.attach_puppet_simulation(model, data)

    assert client.model is model
    assert client.data is data
    assert client.local_model is False
    assert client.step_mode is StepMode.STATEPUSHED


@pytest.mark.parametrize("step_mode", ["stepped", "freerun"])
def test_attach_puppet_simulation_rejects_non_puppet_mode(mujoco_mod, step_mode):
    model, data = _simulation(mujoco_mod)
    client = URLabClient(step_mode=step_mode)

    with pytest.raises(ValueError, match="statepushed"):
        client.attach_puppet_simulation(model, data)


def test_attach_puppet_simulation_rejects_connected_client(mujoco_mod):
    model, data = _simulation(mujoco_mod)
    client = URLabClient(step_mode="statepushed")
    client.session_id = "connected"

    with pytest.raises(RuntimeError, match="before connect"):
        client.attach_puppet_simulation(model, data)


def test_attach_puppet_simulation_rejects_data_from_another_model(mujoco_mod):
    model, _ = _simulation(mujoco_mod)
    other_model, other_data = _simulation(mujoco_mod)
    assert other_model is not model
    client = URLabClient(step_mode="statepushed")

    with pytest.raises(ValueError, match="same model"):
        client.attach_puppet_simulation(model, other_data)


def test_external_puppet_step_zero_pushes_without_advancing(
    mujoco_mod, mock_step_server, base_handshake
):
    model, data = _simulation(mujoco_mod)
    data.qpos[0] = 0.125
    data.qvel[0] = 0.5
    handshake = {
        **base_handshake,
        "mjb": None,
        "entities": [],
    }
    mock_step_server.replies.extend([handshake, wr.step_ok()])
    client = URLabClient(
        "tcp://127.0.0.1",
        step_mode="statepushed",
        step_port=mock_step_server.port,
        recv_timeout_ms=2000,
        auto_promote_step_mode=False,
    )
    client.attach_puppet_simulation(model, data)
    time_before = data.time

    try:
        client.connect()
        client.step(n_steps=0)
    finally:
        client.close()

    request = mock_step_server.received[1]
    assert request["mode"] == "statepushed"
    assert request["n_steps"] == 0
    assert request["qpos"][0] == pytest.approx(0.125)
    assert request["qvel"][0] == pytest.approx(0.5)
    assert data.time == time_before
