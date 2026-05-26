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

"""In-process policy launchers for the UI's Policy tab.

Each submodule here registers a callable in
``urlab_dashboard.tabs.policy.LAUNCHERS[<policy_key>]``. The Policy tab
imports this package once on tab build to trigger registration. Heavy
deps (RoboJuDo, torch, mlc) MUST be deferred to the launcher function
body so importing this package never breaks the UI.

To add a launcher: drop a file alongside ``wtw.py`` and import it from
this module's bottom block.
"""

from . import wtw  # noqa: F401  -- registers go2_wtw

# Importing each launcher submodule is best-effort: a missing heavy dep
# (RoboJuDo, mjlab, torch) on the *module import* path would otherwise
# break the entire policy tab. Each launcher already defers heavy deps to
# its launcher function body; this catch covers a misconfigured install.
try:
    from . import robojudo_g1  # noqa: F401  -- registers 7 G1 entries
except Exception as _exc:
    import logging
    logging.getLogger(__name__).warning(
        "launchers.robojudo_g1 not importable: %s -- "
        "G1 policy Launch buttons will say 'no in-process launcher'.",
        _exc,
    )

try:
    from . import mjlab_g1  # noqa: F401  -- registers mjlab_g1_tracking
except Exception as _exc:
    import logging
    logging.getLogger(__name__).warning(
        "launchers.mjlab_g1 not importable: %s -- "
        "mjlab Launch button will say 'no in-process launcher'.",
        _exc,
    )
