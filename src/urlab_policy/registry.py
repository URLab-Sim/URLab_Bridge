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

"""Registry of available URLab policies for the GUI dropdown.

The :data:`POLICIES` dict here is the merged result of contributions
from every ``urlab_policy.adapters.<adapter>.registry`` module that
imports cleanly. Adapters that fail to import (e.g. torch missing,
RoboJuDo missing, mjlab missing) silently drop their entries.

Each entry defines the import path, DOF requirement, and description.

Optional fields:
    "required_step_mode": one of "live" / "direct" / "puppet" / "auto"
        (or a tuple of those if either-works) for policies that require
        a specific URLab step mode. The launcher rejects mismatched
        modes with a clear error. Existing pre-step-server policies are
        live-streaming (twist over PUB/SUB); they don't need the field.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Optional, Tuple, Union

from urlab_client.enums import StepMode

logger = logging.getLogger(__name__)


# Type alias for the field. Accepts:
#   - None or omitted: policy works in any mode
#   - StepMode enum or its wire string: single required mode
#   - tuple of either: any-of
RequiredStepMode = Optional[
    Union[StepMode, str, Tuple[Union[StepMode, str], ...]]
]


def get_required_step_mode(entry: dict) -> Optional[Tuple[StepMode, ...]]:
    """Normalise an entry's required_step_mode field to a tuple of
    StepMode. Returns None if the entry doesn't declare a requirement."""
    raw = entry.get("required_step_mode")
    if raw is None:
        return None
    items = raw if isinstance(raw, tuple) else (raw,)
    out = []
    for item in items:
        if isinstance(item, StepMode):
            out.append(item)
        else:
            out.append(StepMode(item))
    return tuple(out)


def check_step_mode_compatible(entry: dict, mode: Union[StepMode, str]) -> None:
    """Raise ValueError if ``mode`` is not allowed for the policy entry."""
    requirement = get_required_step_mode(entry)
    if requirement is None:
        return
    mode_enum = mode if isinstance(mode, StepMode) else StepMode(mode)
    if mode_enum not in requirement:
        names = ", ".join(m.value for m in requirement)
        raise ValueError(
            f"Policy requires step_mode in ({names}); got {mode_enum.value}"
        )


def _merge_adapter_registries() -> Dict[str, dict]:
    """Try to import each adapter's registry module and merge their
    POLICIES dicts. Adapters that fail to import (torch missing,
    RoboJuDo missing, mjlab missing, ...) are skipped silently --
    their entries just don't show up in the merged map.

    Adapter names later in the list win on key collisions; today there
    are none, but if two adapters someday register the same key the
    later one (the more recent addition) takes precedence.
    """
    merged: Dict[str, dict] = {}
    for adapter_module in (
        "urlab_policy.adapters.robojudo.registry",
        "urlab_policy.adapters.mjlab.registry",
        "urlab_policy.adapters.lerobot.registry",
    ):
        try:
            mod = importlib.import_module(adapter_module)
        except Exception as exc:
            # Transitives can raise AttributeError at import time, not just
            # ImportError; skip the adapter and let the real traceback surface
            # when the user runs it directly.
            logger.warning(
                "registry: adapter %s failed to import -- skipping (%s: %s)",
                adapter_module, type(exc).__name__, exc,
            )
            continue
        adapter_policies = getattr(mod, "POLICIES", None)
        if not adapter_policies:
            continue
        for key, value in adapter_policies.items():
            if key in merged:
                logger.warning(
                    "registry: %r overridden by %s (was contributed by an "
                    "earlier adapter). Rename one of them to avoid the clash.",
                    key, adapter_module,
                )
            merged[key] = value
    return merged


POLICIES: Dict[str, dict] = _merge_adapter_registries()


def get_policy_labels():
    return {k: v["label"] for k, v in POLICIES.items()}


def import_class(dotted_path):
    """Import a class from a dotted path like 'module.submodule.ClassName'."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
