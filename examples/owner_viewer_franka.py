"""Fast-path OWNER with a native mujoco.viewer window.

Runs the franka scene in a regular ``mujoco.viewer`` passive window (the owner
watches its own sim) under random control, and is a discoverable gRPC owner:
publishes its transform view stream (``stream_cameras``), answers ``fastpath_hello``
with the model, and accepts perturbations (``accept_input``). A separate server
browser can then find it and join it with the UE mirror viewer.

    python examples/owner_viewer_franka.py           # a mujoco.viewer window opens
    # then, elsewhere:
    python -m urlab_client.session list              # see this owner
    python -m urlab_client.session join <id> --mode vr --caps stream_cameras,accept_input
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from urlab_client.fastpath_owner import FastPathOwner

DEFAULT_SCENE = "../mujoco_menagerie/franka_emika_panda/scene.xml"


def to_mjb(model) -> bytes:
    buf = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return buf.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=DEFAULT_SCENE)
    ap.add_argument("--grpc-port", type=int, default=50051)
    ap.add_argument("--scene", default="franka_emika_panda")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Serve the model as flattened XML + assets, NOT MJB: the UE mirror compiles it
    # with its own libmujoco, so there is no MJB version-match requirement (a 3.11.0
    # owner and a 3.11.1 renderer interoperate).
    from urlab_client._model_upload import flatten_model
    xml_text, asset_paths = flatten_model(args.xml)
    assets = {name: open(path, "rb").read() for name, path in asset_paths.items()}

    owner = FastPathOwner(
        xml_text.encode("utf-8"), scene=args.scene, model_format="xml", assets=assets,
        advertise_host="127.0.0.1", ngeom=int(model.ngeom),
    )
    ep = owner.start_grpc_server(port=args.grpc_port)
    print(f"[owner] live: gRPC {ep}  caps={owner.capabilities}")
    print("[owner] browse it:  python -m urlab_client.session list")

    frame = 0
    t0 = time.time()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                # mirrors push a drag INTENT (accept_input); the owner runs the
                # real mjv_applyPerturbForce -- mass-scaled + critically damped, like
                # simulate's Ctrl-drag. Zero first, then let it (re)write the wrench.
                data.xfrc_applied[:] = 0.0
                owner.apply_perturbations(model, data)
                # random control on the arm
                data.ctrl[:] = 0.6 * np.sin(np.arange(model.nu) * 0.7 + (time.time() - t0))
                mujoco.mj_step(model, data)
                owner.serve_pending()                          # zmq control (best-effort)
                owner.publish_mjdata(frame, model, data)       # transform stream (mirror)
                viewer.sync()
                frame += 1
                time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        pass
    finally:
        owner.close()
        print("\n[owner] stopped")


if __name__ == "__main__":
    main()
