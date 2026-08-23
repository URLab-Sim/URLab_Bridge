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

"""
Collision-aware pick-and-place for Franka Panda in URLab.

Uses the full exported MuJoCo scene for collision checking and
RRT-Connect motion planning. Discovers target/basket from ZMQ,
computes collision-free trajectories.

Usage:
    uv run src/urlab_policy/auto_grasp.py
    uv run src/urlab_policy/auto_grasp.py --scene path/to/scene_compiled.xml
    uv run src/urlab_policy/auto_grasp.py --loop
"""

import argparse
import json
import logging
import time
from pathlib import Path

import msgpack
import mujoco
import numpy as np
import zmq

from ._state_stream import StateStream, art_names, scene_names

logger = logging.getLogger(__name__)

# ─── Constants ───

DEFAULT_SCENE = str(
    Path.home() / "Documents" / "Unreal Projects" / "url_proj" / "Saved" / "URLab" / "scene_compiled.xml"
)
DEFAULT_FRANKA_MJCF = str(
    Path.home() / "Documents" / "mujoco_menagerie" / "franka_emika_panda" / "mjx_panda_ue.xml"
)
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
NUM_ARM_JOINTS = 7

# Natural reaching-down pose (gripper pointing down, arm forward)
READY_POSE = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8])


# ─── Collision-Aware IK & Planning ───


class ScenePlanner:
    """
    Motion planner using the full MuJoCo scene for collision checking
    and Jacobian-based IK with collision avoidance.
    """

    def __init__(self, scene_path: str | None = None, franka_mjcf: str = DEFAULT_FRANKA_MJCF):
        """
        Args:
            scene_path: Path to full scene XML/MJB. If None, uses Franka-only model.
            franka_mjcf: Fallback Franka-only MJCF for IK.
        """
        # Load Franka-only model for IK (always works)
        self.ik_model = mujoco.MjModel.from_xml_path(franka_mjcf)
        self.ik_data = mujoco.MjData(self.ik_model)
        self.ee_site_id = self.ik_model.site("gripper").id

        # Load full scene model for collision checking if available
        self.scene_model = None
        self.scene_data = None
        self._franka_geom_ids = set()
        self._target_geom_ids = set()

        if scene_path and Path(scene_path).exists():
            try:
                if scene_path.endswith(".mjb"):
                    self.scene_model = mujoco.MjModel.from_binary_path(scene_path)
                else:
                    self.scene_model = mujoco.MjModel.from_xml_path(scene_path)
                self.scene_data = mujoco.MjData(self.scene_model)
                self._identify_geom_groups()
                logger.info(f"Scene loaded: {self.scene_model.ngeom} geoms, {self.scene_model.nbody} bodies")
            except Exception as e:
                logger.warning(f"Could not load scene model: {e}")
                logger.warning("Falling back to Franka-only model (no collision checking)")

    def _identify_geom_groups(self):
        """Identify which geoms belong to the Franka arm vs environment."""
        if not self.scene_model:
            return
        for i in range(self.scene_model.ngeom):
            body_id = self.scene_model.geom(i).bodyid[0]
            body_name = self.scene_model.body(body_id).name.lower()
            if any(k in body_name for k in ["panda", "franka", "link", "hand", "finger"]):
                self._franka_geom_ids.add(i)
            if "target" in body_name:
                self._target_geom_ids.add(i)

    def check_collision(self, qpos: np.ndarray, ignore_target: bool = False) -> bool:
        """
        Check if the given arm configuration collides with the scene.
        Returns True if there IS a collision.
        """
        if not self.scene_model:
            return False  # No scene = no collision checking

        # Set arm joints in scene model (find matching joints)
        for i in range(min(NUM_ARM_JOINTS + 2, self.scene_model.nq)):
            if i < len(qpos):
                self.scene_data.qpos[i] = qpos[i]

        mujoco.mj_forward(self.scene_model, self.scene_data)

        # Check contacts
        for i in range(self.scene_data.ncon):
            con = self.scene_data.contact[i]
            g1, g2 = con.geom1, con.geom2

            # Skip self-collisions within the arm
            if g1 in self._franka_geom_ids and g2 in self._franka_geom_ids:
                continue

            # When carrying target, ignore target-arm contacts
            if ignore_target:
                if g1 in self._target_geom_ids or g2 in self._target_geom_ids:
                    continue

            # Any contact involving the arm and environment = collision
            if g1 in self._franka_geom_ids or g2 in self._franka_geom_ids:
                return True

        return False

    def solve_ik(self, target_pos: np.ndarray, seed_qpos: np.ndarray,
                 max_iter: int = 200, tol: float = 1e-3) -> np.ndarray | None:
        """
        Solve IK for target EE position, checking for collisions.
        Returns joint positions or None if no collision-free solution found.
        """
        self.ik_data.qpos[:] = 0
        nq = min(len(seed_qpos), self.ik_model.nq)
        self.ik_data.qpos[:nq] = seed_qpos[:nq]

        for _ in range(max_iter):
            mujoco.mj_forward(self.ik_model, self.ik_data)
            cur = self.ik_data.site_xpos[self.ee_site_id].copy()
            dx = target_pos - cur
            if np.linalg.norm(dx) < tol:
                break
            jacp = np.zeros((3, self.ik_model.nv))
            mujoco.mj_jacSite(self.ik_model, self.ik_data, jacp, None, self.ee_site_id)
            J = jacp[:, :NUM_ARM_JOINTS]
            JJT = J @ J.T + 1e-4 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, dx)
            dq = np.clip(dq, -0.05, 0.05)
            self.ik_data.qpos[:NUM_ARM_JOINTS] += dq

        solution = self.ik_data.qpos[:NUM_ARM_JOINTS].copy()

        # Verify we reached the target
        mujoco.mj_forward(self.ik_model, self.ik_data)
        achieved = self.ik_data.site_xpos[self.ee_site_id].copy()
        if np.linalg.norm(target_pos - achieved) > 0.01:
            logger.warning(f"  IK did not converge: err={np.linalg.norm(target_pos - achieved):.4f}m")
            return None

        # Check collisions
        if self.check_collision(solution):
            logger.warning(f"  IK solution collides — trying alternative seeds")
            # Try a few random perturbations
            for attempt in range(5):
                perturbed = seed_qpos.copy()
                perturbed[:NUM_ARM_JOINTS] += np.random.uniform(-0.3, 0.3, NUM_ARM_JOINTS)
                alt = self._solve_ik_no_check(target_pos, perturbed)
                if alt is not None and not self.check_collision(alt):
                    return alt
            logger.warning(f"  No collision-free IK solution found")

        return solution

    def _solve_ik_no_check(self, target_pos: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        """IK without collision check (helper for retries)."""
        self.ik_data.qpos[:] = 0
        nq = min(len(seed), self.ik_model.nq)
        self.ik_data.qpos[:nq] = seed[:nq]

        for _ in range(200):
            mujoco.mj_forward(self.ik_model, self.ik_data)
            cur = self.ik_data.site_xpos[self.ee_site_id].copy()
            dx = target_pos - cur
            if np.linalg.norm(dx) < 1e-3:
                break
            jacp = np.zeros((3, self.ik_model.nv))
            mujoco.mj_jacSite(self.ik_model, self.ik_data, jacp, None, self.ee_site_id)
            J = jacp[:, :NUM_ARM_JOINTS]
            JJT = J @ J.T + 1e-4 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, dx)
            dq = np.clip(dq, -0.05, 0.05)
            self.ik_data.qpos[:NUM_ARM_JOINTS] += dq

        achieved = self.ik_data.site_xpos[self.ee_site_id].copy()
        if np.linalg.norm(target_pos - achieved) > 0.01:
            return None
        return self.ik_data.qpos[:NUM_ARM_JOINTS].copy()

    def ee_pos(self, qpos: np.ndarray) -> np.ndarray:
        """Forward kinematics: return EE position."""
        self.ik_data.qpos[:min(len(qpos), self.ik_model.nq)] = qpos[:min(len(qpos), self.ik_model.nq)]
        mujoco.mj_forward(self.ik_model, self.ik_data)
        return self.ik_data.site_xpos[self.ee_site_id].copy()

    def plan_collision_free_path(self, start_q: np.ndarray, end_q: np.ndarray,
                                 ignore_target: bool = False,
                                 max_steps: int = 20) -> list[np.ndarray]:
        """
        Plan a collision-free path between two joint configurations.
        Uses linear interpolation with subdivision where collisions are detected.
        Falls back to direct interpolation if no scene model.
        """
        if not self.scene_model:
            return self._linear_path(start_q, end_q, max_steps)

        # First try direct path
        path = self._linear_path(start_q, end_q, max_steps)
        has_collision = False
        for q in path:
            if self.check_collision(q, ignore_target):
                has_collision = True
                break

        if not has_collision:
            return path

        # Collision detected — try going through a high waypoint
        logger.info("  Direct path has collision — routing through safe waypoint")
        ee_start = self.ee_pos(start_q)
        ee_end = self.ee_pos(end_q)

        # Safe waypoint: midpoint but higher up
        mid_pos = (ee_start + ee_end) / 2
        mid_pos[2] = max(ee_start[2], ee_end[2]) + 0.15  # go above both

        seed = READY_POSE.copy()
        q_mid = self.solve_ik(mid_pos, np.concatenate([seed, np.zeros(2)]))
        if q_mid is None:
            logger.warning("  Could not find safe waypoint — using direct path")
            return path

        # Two-segment path through waypoint
        path1 = self._linear_path(start_q, q_mid, max_steps // 2)
        path2 = self._linear_path(q_mid, end_q, max_steps // 2)
        return path1 + path2[1:]  # skip duplicate midpoint

    def _linear_path(self, start: np.ndarray, end: np.ndarray, steps: int) -> list[np.ndarray]:
        """Simple linear interpolation in joint space."""
        return [start + (end - start) * t for t in np.linspace(0, 1, steps)]


# ─── Smart Waypoint Generation ───


def compute_grasp_waypoints(planner: ScenePlanner, object_pos: np.ndarray,
                            basket_pos: np.ndarray, franka_base: np.ndarray,
                            current_qpos: np.ndarray) -> list[dict]:
    """
    Compute pick-and-place waypoints with smart positioning.

    - Approaches from directly above the object
    - Grasps at object height
    - Drops into basket from above (top opening, not center of mass)
    """
    obj_rel = object_pos - franka_base
    basket_rel = basket_pos - franka_base

    # The basket base_state gives its body origin which may be at the bottom.
    # We need to drop from ABOVE the basket opening.
    # Estimate basket top: base pos + some height (baskets are ~0.2-0.3m tall)
    basket_drop = basket_rel.copy()
    basket_drop[2] += 0.25  # drop point above basket opening

    # Waypoints relative to Franka base
    approach_height = 0.15
    grasp_offset = 0.01  # slightly above object center for top-grasp

    obj_above = obj_rel.copy()
    obj_above[2] += approach_height

    obj_grasp = obj_rel.copy()
    obj_grasp[2] += grasp_offset

    obj_lift = obj_rel.copy()
    obj_lift[2] += approach_height + 0.05  # lift a bit higher than approach

    logger.info(f"  Object (rel to base): {obj_rel}")
    logger.info(f"  Basket (rel to base): {basket_rel}")
    logger.info(f"  Basket drop point:    {basket_drop}")

    # Solve IK for each waypoint from a natural seed
    seed = np.concatenate([READY_POSE, current_qpos[NUM_ARM_JOINTS:]])

    q_above = planner.solve_ik(obj_above, seed)
    q_grasp = planner.solve_ik(obj_grasp, np.concatenate([q_above if q_above is not None else READY_POSE, seed[NUM_ARM_JOINTS:]]))
    q_lift = planner.solve_ik(obj_lift, np.concatenate([q_grasp if q_grasp is not None else READY_POSE, seed[NUM_ARM_JOINTS:]]))
    q_drop = planner.solve_ik(basket_drop, np.concatenate([q_lift if q_lift is not None else READY_POSE, seed[NUM_ARM_JOINTS:]]))

    for name, q, target in [("above", q_above, obj_above), ("grasp", q_grasp, obj_grasp),
                             ("lift", q_lift, obj_lift), ("drop", q_drop, basket_drop)]:
        if q is not None:
            achieved = planner.ee_pos(q)
            logger.info(f"  IK {name:>6s}: target={target}, err={np.linalg.norm(achieved - target):.4f}m")
        else:
            logger.error(f"  IK {name:>6s}: FAILED for target={target}")
            return []

    # Build trajectory with collision-free paths between waypoints
    trajectory = [
        {"joints": READY_POSE,  "gripper": GRIPPER_OPEN,   "steps": 60,  "label": "Move to ready pose",      "plan": True},
        {"joints": q_above,     "gripper": GRIPPER_OPEN,   "steps": 80,  "label": "Move above object",        "plan": True},
        {"joints": q_grasp,     "gripper": GRIPPER_OPEN,   "steps": 50,  "label": "Descend to object",        "plan": False},
        {"joints": q_grasp,     "gripper": GRIPPER_CLOSED, "steps": 40,  "label": "Close gripper (grasp)",    "plan": False},
        {"joints": q_lift,      "gripper": GRIPPER_CLOSED, "steps": 50,  "label": "Lift object",              "plan": False},
        {"joints": q_drop,      "gripper": GRIPPER_CLOSED, "steps": 80,  "label": "Move to basket",           "plan": True,  "ignore_target": True},
        {"joints": q_drop,      "gripper": GRIPPER_OPEN,   "steps": 40,  "label": "Release into basket",      "plan": False},
        {"joints": READY_POSE,  "gripper": GRIPPER_OPEN,   "steps": 60,  "label": "Return to ready pose",     "plan": True},
    ]

    return trajectory


# ─── ZMQ Interface ───


class ZMQInterface:
    """Handles all ZMQ communication with Unreal."""

    def __init__(self, state_ep: str, control_ep: str, info_ep: str):
        self.ctx = zmq.Context()

        # State arrives as the canonical `state/full` msgpack snapshot;
        # articulations under `arts`, dynamic scene props under `scene`.
        self.state = StateStream(self.ctx, state_ep)

        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(control_ep)

        # NOTE: the `:5557` `actuator_list` info broadcast was retired in 5.3.
        # This SUB now never receives anything -- recv() would just time out
        # (RCVTIMEO=2000). We keep the socket for call-site compatibility but do
        # not depend on it: send_joints() uses ordinal actuator ids (see
        # execute_trajectory's `list(range(...))`), so discovery failing here is
        # harmless. Full discovery migration (a replacement for the retired
        # broadcast) is a pending design decision.
        self.info_sub = self.ctx.socket(zmq.SUB)
        self.info_sub.connect(info_ep)
        self.info_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.info_sub.setsockopt(zmq.RCVTIMEO, 2000)

        time.sleep(0.3)
        # prefix -> {"qpos": np.ndarray, "base_pos": np.ndarray | None}
        self.articulations: dict[str, dict] = {}

    def discover(self, timeout: float = 5.0):
        """Listen to the state stream to discover all articulations + props."""
        logger.info("Discovering articulations...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._drain()
            prefixes = list(self.articulations.keys())
            has_franka = any(self._is_franka(p) for p in prefixes)
            has_target = any("target" in p.lower() for p in prefixes)
            has_basket = any("basket" in p.lower() for p in prefixes)
            if has_franka and has_target and has_basket:
                break
            time.sleep(0.05)

        for prefix, data in self.articulations.items():
            role = self._classify(prefix)
            pos = data.get("base_pos")
            pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]" if pos is not None else "no free joint"
            n_joints = len(data.get("qpos", []))
            logger.info(f"  [{role:>7s}] {prefix} — {n_joints} qpos, pos={pos_str}")

    def _drain(self):
        snap = self.state.drain()
        if snap is None:
            return
        # Articulations: qpos concatenated in joint discovery order. A free
        # base (if any) occupies qpos[0:7] -> use its position component.
        for prefix in art_names(snap):
            qpos = np.asarray(
                (snap["arts"][prefix].get("qpos") or []), dtype=np.float32
            )
            base_pos = np.array(qpos[0:3]) if qpos.size >= 7 else None
            self.articulations[prefix] = {"qpos": qpos, "base_pos": base_pos}
        # Dynamic scene props (target, basket): body-origin world pose.
        for name in scene_names(snap):
            block = snap["scene"][name]
            xpos = block.get("xpos")
            self.articulations[name] = {
                "qpos": np.zeros(0, dtype=np.float32),
                "base_pos": np.array(xpos, dtype=np.float64) if xpos else None,
            }

    def get_joint_positions(self, prefix: str) -> np.ndarray:
        self._drain()
        return np.asarray(
            self.articulations.get(prefix, {}).get("qpos", []), dtype=np.float32
        )

    def get_base_pos(self, prefix: str) -> np.ndarray | None:
        self._drain()
        return self.articulations.get(prefix, {}).get("base_pos")

    def send_joints(self, prefix: str, targets: np.ndarray, actuator_ids: list[int]):
        """Send joint targets on the `{prefix}/control ` topic.

        Control-in is a msgpack `{ids:[...], vals:[...]}` payload parsed
        UE-side by FURLabMsgpackUtil (5.3). The legacy little-endian
        `[i32 n][i32 id, f32 val]*` binary format is retired -- the UE
        unsafe-cast parser was removed, so emitting it now mismatches.
        """
        n = len(targets)
        ids = [int(actuator_ids[i]) if actuator_ids else i for i in range(n)]
        vals = [float(targets[i]) for i in range(n)]
        payload = msgpack.packb({"ids": ids, "vals": vals}, use_bin_type=True)
        self.pub.send_string(f"{prefix}/control ", zmq.SNDMORE)
        self.pub.send(payload)

    def find_by_role(self, role: str) -> str | None:
        for prefix in self.articulations:
            if self._classify(prefix) == role:
                return prefix
        return None

    def _classify(self, prefix: str) -> str:
        p = prefix.lower()
        if "target" in p:
            return "target"
        if "basket" in p or "bin" in p or "box" in p:
            return "basket"
        if self._is_franka(prefix):
            return "franka"
        return "other"

    def _is_franka(self, prefix: str) -> bool:
        p = prefix.lower()
        return "panda" in p or "franka" in p

    def close(self):
        self.state.close()
        self.pub.close()
        self.info_sub.close()
        self.ctx.term()


# ─── Trajectory Execution ───


def execute_trajectory(zmq_iface: ZMQInterface, planner: ScenePlanner,
                       franka_prefix: str, trajectory: list[dict],
                       current_qpos: np.ndarray):
    """Execute a pick-and-place trajectory with smooth interpolation."""

    actuator_ids = list(range(NUM_ARM_JOINTS + 1))  # [0..7]
    current_arm = current_qpos[:NUM_ARM_JOINTS].copy()
    current_gripper = GRIPPER_OPEN

    for seg in trajectory:
        label = seg["label"]
        target_arm = seg["joints"]
        target_gripper = seg["gripper"]
        steps = seg["steps"]
        use_planning = seg.get("plan", False)
        ignore_target = seg.get("ignore_target", False)

        logger.info(f"  >> {label}")

        if use_planning:
            # Use collision-free path planning
            path = planner.plan_collision_free_path(
                current_arm, target_arm,
                ignore_target=ignore_target,
                max_steps=steps,
            )
        else:
            # Direct interpolation (for short moves like descend/grasp)
            path = [current_arm + (target_arm - current_arm) * t
                    for t in np.linspace(0, 1, steps)]

        # Execute path with gripper interpolation
        for i, arm_q in enumerate(path):
            t = (i + 1) / len(path)
            t_smooth = t * t * (3 - 2 * t)  # ease in/out
            grip = current_gripper + (target_gripper - current_gripper) * t_smooth

            full_target = np.concatenate([arm_q, [grip]])
            zmq_iface.send_joints(franka_prefix, full_target, actuator_ids)
            time.sleep(0.02)  # 50 Hz

        current_arm = target_arm.copy()
        current_gripper = target_gripper


# ─── Main ───


def run_pick_and_place(zmq_iface: ZMQInterface, planner: ScenePlanner) -> bool:
    """Execute one pick-and-place cycle."""

    franka_prefix = zmq_iface.find_by_role("franka")
    target_prefix = zmq_iface.find_by_role("target")
    basket_prefix = zmq_iface.find_by_role("basket")

    if not franka_prefix:
        logger.error("No Franka found!")
        return False
    if not target_prefix:
        logger.error("No target object found! (name must contain 'target')")
        return False
    if not basket_prefix:
        logger.error("No basket found! (name must contain 'basket')")
        return False

    # Get current state
    franka_joints = zmq_iface.get_joint_positions(franka_prefix)
    franka_base = zmq_iface.get_base_pos(franka_prefix)
    object_pos = zmq_iface.get_base_pos(target_prefix)
    basket_pos = zmq_iface.get_base_pos(basket_prefix)

    if franka_base is None:
        logger.info("No Franka free joint — assuming base at origin [0, 0, 0]")
        franka_base = np.zeros(3)
    if object_pos is None:
        logger.error("No target base state received")
        return False
    if basket_pos is None:
        logger.error("No basket base state received")
        return False

    logger.info(f"Franka:  {franka_prefix} (base={franka_base})")
    logger.info(f"Target:  {target_prefix} (pos={object_pos})")
    logger.info(f"Basket:  {basket_prefix} (pos={basket_pos})")

    # Plan trajectory
    logger.info("Planning collision-aware trajectory...")
    trajectory = compute_grasp_waypoints(planner, object_pos, basket_pos, franka_base, franka_joints)

    if not trajectory:
        logger.error("Failed to plan trajectory!")
        return False

    # Execute
    logger.info("Executing trajectory...")
    execute_trajectory(zmq_iface, planner, franka_prefix, trajectory, franka_joints)
    logger.info("Pick and place complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Collision-aware pick-and-place for Franka in URLab")
    parser.add_argument("--state-ep", default="tcp://127.0.0.1:5555")
    parser.add_argument("--control-ep", default="tcp://127.0.0.1:5556")
    parser.add_argument("--info-ep", default="tcp://127.0.0.1:5557")
    parser.add_argument("--scene", default=None, help="Full scene XML/MJB for collision checking")
    parser.add_argument("--mjcf", default=DEFAULT_FRANKA_MJCF, help="Franka MJCF for IK")
    parser.add_argument("--loop", action="store_true", help="Keep picking up target repeatedly")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Initializing planner...")
    planner = ScenePlanner(scene_path=args.scene, franka_mjcf=args.mjcf)

    logger.info("Connecting to ZMQ...")
    zmq_iface = ZMQInterface(args.state_ep, args.control_ep, args.info_ep)
    zmq_iface.connect()

    try:
        while True:
            success = run_pick_and_place(zmq_iface, planner)
            if not args.loop:
                break
            if success:
                logger.info("Waiting 3s before next cycle...")
                time.sleep(3.0)
            else:
                logger.error("Failed — retrying in 2s...")
                time.sleep(2.0)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        zmq_iface.close()


if __name__ == "__main__":
    main()
