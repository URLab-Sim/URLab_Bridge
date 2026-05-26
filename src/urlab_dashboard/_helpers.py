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

"""Dashboard-internal helpers. Things that are UI-specific and don't
belong in the public `urlab_client` API surface."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def push_sim_dt(client: Any, dt: float, label: str) -> None:
    """``set_sim_options(timestep=dt)`` against UE, logged to the dpg panel.

    Call AFTER runner/pipeline construction and immediately before the
    step loop -- UE recompiles ``mjModel`` on every PIE start, resetting
    ``opt.timestep`` to the XML value, so this needs to be the last write.
    """
    from .log import log

    try:
        applied = client.runtime.set_sim_options(timestep=float(dt))
    except Exception as exc:
        log(f"{label}: set_sim_options(timestep={dt:.5f}) FAILED: "
            f"{type(exc).__name__}: {exc}", error=True)
        return
    ue_dt = float(getattr(applied, "timestep", 0.0) or 0.0)
    if abs(ue_dt - dt) > 1e-6:
        log(f"{label}: set_sim_options(timestep={dt:.5f}) -> UE reports "
            f"{ue_dt:.5f}s -- sim will run at {ue_dt/dt:.2f}x real-time", error=True)
        return
    log(f"{label}: pushed sim timestep={ue_dt:.5f}s to UE")


def parse_floats(text: str, count: int, default: List[float]) -> List[float]:
    """Parse ``"x, y, z"`` into a length-``count`` float list, padding
    from ``default`` for missing slots and falling back to ``default``
    entirely if any token isn't a valid float. Comma- or
    whitespace-separated.

    Used by the dashboard's text-input parsing (the user types a
    transform inline)."""
    if not text or not text.strip():
        return list(default)
    parts = [p for p in text.replace(",", " ").split() if p]
    out: List[float] = list(default)
    for i, part in enumerate(parts[:count]):
        try:
            out[i] = float(part)
        except ValueError:
            logger.debug("parse_floats: bad number %r in %r", part, text)
            return list(default)
    return out
