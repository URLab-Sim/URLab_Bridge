"""Fast-path OWNER for external-control demos.

Steps a scene in-process and serves it over gRPC (+ZMQ) so a packaged UE mirror
can join and render it -- the "controlled externally" half of the demo. Random
control on any actuators; honors forwarded drag perturbations (accept_input);
publishes the render tier (+ debug tier when a subscriber negotiates
StreamContacts/StreamOverlay).

By default opens NO window (survives backgrounding). Pass --viewer to also open
MuJoCo's own native passive viewer, so you can watch the ground-truth MuJoCo
render side-by-side with the UE mirror (useful for validating any test scene).

    python examples/owner_headless.py --xml <scene.xml> --grpc-port 50051
    python examples/owner_headless.py --xml <scene.xml> --viewer   # + MuJoCo window
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from urlab_client import viewer_sync
from urlab_client._model_upload import flatten_model
from urlab_client.fastpath_owner import FastPathOwner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--grpc-port", type=int, default=50051)
    ap.add_argument("--scene", default="headless")
    ap.add_argument("--viewer", action="store_true",
                    help="also open MuJoCo's native passive viewer window")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    xml_text, asset_paths = flatten_model(args.xml)
    assets = {name: open(path, "rb").read() for name, path in asset_paths.items()}

    owner = FastPathOwner(
        xml_text.encode("utf-8"), scene=args.scene, model_format="xml", assets=assets,
        advertise_host="127.0.0.1", ngeom=int(model.ngeom),
    )
    ep = owner.start_grpc_server(port=args.grpc_port)
    tag = "owner" if args.viewer else "headless-owner"
    print(f"[{tag}] live: gRPC {ep}  scene={args.scene}  caps={owner.capabilities}"
          f"{'  + MuJoCo passive viewer' if args.viewer else ''}", flush=True)

    t0 = time.time()

    def step(frame: int, usercam=None, pert=None) -> None:
        data.xfrc_applied[:] = 0.0
        owner.apply_perturbations(model, data)          # drag intents forwarded from the UE mirror
        # Local ctrl-drag in the MuJoCo passive window: the viewer records it in
        # handle.perturb but (unlike simulate) does not apply it -- our step loop
        # owns physics, so we apply it here, exactly as simulate does.
        if pert is not None:
            mujoco.mjv_applyPerturbPose(model, data, pert, 0)   # mocap / kinematic drag
            mujoco.mjv_applyPerturbForce(model, data, pert)     # dynamic-body spring
        if model.nu:
            data.ctrl[:] = 0.6 * np.sin(np.arange(model.nu) * 0.7 + (time.time() - t0))
        mujoco.mj_step(model, data)
        owner.serve_pending()                            # zmq control (best-effort)
        # usercam (ucpos/ucfwd/ucup): the passive viewer's free-camera pose. The UE
        # mirror's copycat (ApplyUserCamera -> SetViewTarget) points its viewport at
        # the same eye, so orbiting the MuJoCo window orbits the UE view in lockstep.
        owner.publish_mjdata(frame, model, data, usercam=usercam)  # render tier (+ debug tier)
        time.sleep(model.opt.timestep)

    frame = 0
    try:
        if args.viewer:
            # `from ... import ... as` binds a NEW name -- a bare `import mujoco.viewer`
            # here would rebind `mujoco` as a function-local and shadow the module.
            from mujoco import viewer as mjviewer
            scn = viewer_sync.make_scene(model)          # reused each frame (no per-call alloc)
            with mjviewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    step(frame, viewer_sync.pose_from_passive(viewer, scene=scn),
                         viewer.perturb)                 # local ctrl-drag in this window
                    viewer.sync()                        # ground-truth MuJoCo render
                    frame += 1
        else:
            while True:
                step(frame)
                frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        owner.close()
        print(f"\n[{tag}] stopped", flush=True)


if __name__ == "__main__":
    main()
