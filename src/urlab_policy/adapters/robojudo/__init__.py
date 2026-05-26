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

"""URLab <-> RoboJuDo adapter.

``URLabRoboJuDoEnv`` subclasses RoboJuDo's ``Environment`` and reads
state via ``URLabClient`` / ``URLabArticulation``. Importing registers
the ``UnrealTwistCtrl`` override; no monkey-patches.
"""

from __future__ import annotations

from .joint_specs import (
    G1_12DOF_JOINT_NAMES,
    G1_29DOF_JOINT_NAMES,
    GO2_12DOF_JOINT_NAMES,
)

# Re-export env classes only when RoboJuDo is importable, so a
# RoboJuDo-less install still gets a clean
# ``from urlab_policy.adapters.robojudo import G1_29DOF_JOINT_NAMES``.
try:
    from .joint_specs import G1_12DOF, G1_29DOF, GO2_12DOF  # noqa: F401
    from .env import (  # noqa: F401
        G1_29URLabRoboJuDoEnvCfg,
        G1URLabRoboJuDoEnvCfg,
        G1UnrealEnvCfg,
        G1_29UnrealEnvCfg,
        Go2URLabRoboJuDoEnvCfg,
        Go2UnrealEnvCfg,
        URLabRoboJuDoEnv,
        URLabRoboJuDoEnvCfg,
        UnrealEnv,
        UnrealEnvCfg,
        URLabEnv,
    )
    from .twist_ctrl import (  # noqa: F401
        UnrealTwistCtrl,
        UnrealTwistCtrlCfg,
        register_controller as _register_controller,
    )

    HAS_ROBOJUDO = True
    # Explicit registration: replaces RoboJuDo's JoystickCtrl with our
    # ZMQ-driven twist controller. Idempotent.
    _register_controller()
except ImportError:
    HAS_ROBOJUDO = False
