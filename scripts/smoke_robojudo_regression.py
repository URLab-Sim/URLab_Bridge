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

"""RoboJuDo regression smoke (offline, no editor required).

Verifies the bridge can:
1. Import the RoboJuDo adapter without errors.
2. Resolve every bundled policy entry's RobotSpec.
3. Look up every entry's policy_cfg / env_cfg dotted path (only when
   RoboJuDo is installed; skipped otherwise).

Does NOT spin up a live UE editor.

Run:
    micromamba run -n mj python -m scripts.smoke_robojudo_regression

Or via pytest:
    micromamba run -n mj python -m pytest scripts/smoke_robojudo_regression.py -v
"""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    print("RoboJuDo regression smoke (offline)")
    print("=" * 60)

    # 1. Adapter import.
    print("\n[1/3] Adapter import")
    import urlab_policy.adapters.robojudo as adapter
    print(f"      HAS_ROBOJUDO = {adapter.HAS_ROBOJUDO}")

    # 2. Registry merge + RobotSpec resolution.
    print("\n[2/3] POLICIES merge + RobotSpec resolution")
    from urlab_policy.registry import POLICIES
    from urlab_policy.adapters.robojudo.registry import robot_for

    if not POLICIES:
        print("      ERROR: POLICIES is empty (RoboJuDo adapter failed to merge?)")
        return 1
    print(f"      {len(POLICIES)} entries: {sorted(POLICIES)}")
    for name in sorted(POLICIES):
        entry = POLICIES[name]
        spec = robot_for(entry)
        if spec is None:
            print(f"      WARN: {name!r} has no RobotSpec")
        else:
            print(f"      {name:>20}: robot={spec.name:<10} num_dofs={spec.num_dofs:<3} "
                  f"xml={spec.xml_asset_key}")

    # 3. Dotted-path resolution (only if RoboJuDo + torch are present).
    print("\n[3/3] Dotted-path resolution")
    if not adapter.HAS_ROBOJUDO:
        print("      RoboJuDo not installed — skipping dotted-path checks.")
        print("      To run a full smoke, install RoboJuDo and re-run.")
        return 0

    failures = []
    for name, entry in sorted(POLICIES.items()):
        for key in ("policy_cfg", "env_cfg"):
            dotted = entry.get(key)
            if not dotted:
                continue
            mod_name, _, attr = dotted.rpartition(".")
            try:
                mod = importlib.import_module(mod_name)
                getattr(mod, attr)
            except Exception as exc:
                failures.append((name, key, dotted, str(exc)))
                print(f"      FAIL {name:<20} {key:<11} {dotted}: {exc}")

    if failures:
        print(f"\n{len(failures)} failures.")
        return 2

    print(f"      All {len(POLICIES)} entries resolve cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
