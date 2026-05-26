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

"""RoboJuDo controller that reads twist commands from the env's ZMQ
stream instead of keyboard/joystick.

Used when ``--twist-source zmq`` is passed. The accompanying registry
override (replacing ``robojudo.controller.JoystickCtrl`` with this class)
happens in :func:`register_controller` so import of this module has no
side effects -- the override only fires when the adapter package
explicitly registers it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from robojudo.controller import Controller, ctrl_registry
    from robojudo.controller.ctrl_cfgs import CtrlCfg
    HAS_ROBOJUDO = True
except ImportError:
    HAS_ROBOJUDO = False


if HAS_ROBOJUDO:

    class UnrealTwistCtrlCfg(CtrlCfg):
        # Masquerade as JoystickCtrl so ctrl_data_all is keyed correctly
        # for UnitreePolicy._get_commands which checks for "JoystickCtrl".
        ctrl_type: str = "JoystickCtrl"

    class UnrealTwistCtrl(Controller):
        """Reads twist commands from the env (populated via ZMQ from
        Unreal input)."""

        cfg_ctrl: UnrealTwistCtrlCfg

        def __init__(self, cfg_ctrl: UnrealTwistCtrlCfg, env=None, device="cpu"):
            super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
            self._env = env

        def reset(self):
            pass

        def get_data(self):
            """Return axes dict matching JoystickCtrl.get_data() format."""
            if self._env is None:
                return {
                    "axes": {"LeftX": 0, "LeftY": 0, "RightX": 0, "RightY": 0},
                    "button_event": [],
                }

            twist = self._env.twist_cmd  # [vx, vy, yaw_rate]

            if abs(twist[0]) > 0.01 or abs(twist[1]) > 0.01 or abs(twist[2]) > 0.01:
                logger.info(
                    f"Twist cmd: vx={twist[0]:.3f} vy={twist[1]:.3f} "
                    f"yaw={twist[2]:.3f}"
                )

            # Normalize to [-1, 1]; policy applies command_remap with max_cmd.
            max_vx = 0.8
            max_vy = 0.5
            max_yaw = 1.57

            return {
                "axes": {
                    "LeftX": float(twist[1] / max_vy) if max_vy > 0 else 0.0,
                    "LeftY": float(twist[0] / max_vx) if max_vx > 0 else 0.0,
                    "RightX": float(twist[2] / max_yaw) if max_yaw > 0 else 0.0,
                    "RightY": 0.0,
                },
                "button_event": [],
            }

    def register_controller() -> None:
        """Override RoboJuDo's ``JoystickCtrl`` with our UnrealTwistCtrl
        so ``ctrl_manager`` instantiates ours when it sees a JoystickCtrl
        cfg. Idempotent. Called once per process from the adapter
        ``__init__.py`` (no import-time side effects)."""
        import robojudo.controller as _ctrl

        _ctrl.JoystickCtrl = UnrealTwistCtrl
        logger.debug(
            "robojudo.controller.JoystickCtrl override -> UnrealTwistCtrl"
        )


else:

    UnrealTwistCtrl = None  # type: ignore[assignment]
    UnrealTwistCtrlCfg = None  # type: ignore[assignment]

    def register_controller() -> None:
        """No-op when RoboJuDo isn't installed."""
        return
