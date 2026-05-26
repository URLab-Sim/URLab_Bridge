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

"""mjlab integration adapter — bridges mjlab's eval-time policy stack
to URLab's transport / scene primitives. Top-level re-exports keep
``from urlab_policy.adapters.mjlab import MjlabRunner`` working."""

from __future__ import annotations

try:
    from .runtime import MjlabRunner, register_command_context  # noqa: F401
except ImportError:
    # mjlab / torch not installed -- the runtime module raises early.
    # Importing this package shouldn't blow up; callers that actually
    # need MjlabRunner will see the original ImportError when they try.
    pass

__all__ = ["MjlabRunner", "register_command_context"]
