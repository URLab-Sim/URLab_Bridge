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

"""mjlab-bound policy registry entries.

The default entry points at mjlab's flat-terrain Unitree G1 tracking
task. The in-process launcher (``urlab_dashboard.launchers.mjlab_g1``)
mirrors ``scripts/run_mjlab_demo.py``. You still need a checkpoint on
disk -- the launcher infers ``DEFAULT_DEMO_DIR`` from that script and
logs a clear hint when none is found.
"""

from __future__ import annotations


POLICIES: dict[str, dict] = {
    "mjlab_g1_tracking": {
        "label": "mjlab Unitree G1 Tracking (flat)",
        "task_id": "Mjlab-Tracking-Flat-Unitree-G1",
        "policy_cfg": "mjlab.tasks.tracking.flat.g1",
        "env_cfg": "urlab_policy.adapters.mjlab.runtime.MjlabRunner",
        "robot": "g1_29dof",
        "dofs": 29,
        "xml": "g1_29dof",
        "desc": "mjlab motion-tracking policy. Checkpoint resolves from "
                "scripts/run_mjlab_demo.DEFAULT_DEMO_DIR.",
        "ctrl_type": "motion",
    },
}
