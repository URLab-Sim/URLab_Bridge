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

"""URLab adapters for third-party policy frameworks.

Each adapter is a self-contained subpackage:

  * :mod:`.robojudo` -- glue for the RoboJuDo policy ecosystem.
  * :mod:`.mjlab` -- mjlab eval-time runner + MJCF export tool.
  * :mod:`.lerobot` -- (placeholder) LeRobot policy support.

Adapters do not import each other at module level. Each one ships its
own :data:`registry.POLICIES` map (may be empty); the top-level
:data:`urlab_policy.registry.POLICIES` is the merged result of all
adapters that import successfully (guarded against missing optional
deps -- torch, mjlab, ...).
"""
