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

"""Loaders that build a :class:`~urlab_policy.task_spec.TaskSpec` from a
description outside URLab itself.

Today there is one loader: :func:`yaml_to_taskspec` reads a static YAML
file. Adapter-specific loaders (e.g. the mjlab task-config loader) live
under their adapter package instead of here, so this package stays
framework-neutral.
"""

from __future__ import annotations

from .yaml import yaml_to_taskspec

__all__ = ["yaml_to_taskspec"]
