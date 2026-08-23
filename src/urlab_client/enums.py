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

"""
Python enum surface for the URLab remote-stepping client.

All wire-format mode / kind / option strings as `str, enum.Enum` mixins
so they round-trip JSON / msgpack as their string value without glue.
3.10-compatible (no StrEnum). Unknown values fall through `coerce` and
log a one-shot warning.
"""

from __future__ import annotations

import enum
import logging
from typing import TypeVar, Union

logger = logging.getLogger(__name__)


class StepMode(str, enum.Enum):
    """Producer step mode, mirroring UE's ``EMjStepMode``. The ``set_mode`` wire
    token is the member's lowercase name. ``auto`` is deliberately absent: it is a
    client-side policy (promote-on-connect), handled in ``URLabClient``, never a
    wire value."""

    FREERUN = "freerun"
    STEPPED = "stepped"
    STATEPUSHED = "statepushed"


class ActuatorType(str, enum.Enum):
    # `<general>` is MJCF's own actuator element, and the one every other kind
    # is shorthand for -- mjlab's models author it directly.
    GENERAL = "general"
    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    INT_VELOCITY = "intvelocity"
    DAMPER = "damper"
    CYLINDER = "cylinder"
    MUSCLE = "muscle"
    ADHESION = "adhesion"
    DC_MOTOR = "dcmotor"


class ControllerKind(str, enum.Enum):
    PD = "pd"
    PASSTHROUGH = "passthrough"


class LightKind(str, enum.Enum):
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"


class CameraMode(str, enum.Enum):
    REAL = "real"
    DEPTH = "depth"
    SEMANTIC = "semantic"
    INSTANCE = "instance"


class CameraTiming(str, enum.Enum):
    SYNC = "sync"
    LATEST = "latest"


class SpaceMode(str, enum.Enum):
    FLAT = "flat"
    DICT = "dict"


class ObservationLevel(str, enum.Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


E = TypeVar("E", bound=enum.Enum)


def coerce(enum_cls: type[E], value: Union[str, E, None], *, default: E | None = None) -> E:
    """Normalise to an `enum_cls` member. Accepts a member, a wire-string
    matching a value, or None (→ default or raise). Unknown strings raise
    ValueError after logging once."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"None is not a valid {enum_cls.__name__}")
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            logger.warning(
                "Unknown %s wire value %r (known: %s)",
                enum_cls.__name__,
                value,
                [m.value for m in enum_cls],
            )
            raise
    raise TypeError(
        f"Cannot coerce {type(value).__name__} to {enum_cls.__name__}"
    )


def wire(value: Union[str, enum.Enum]) -> str:
    """Return the wire-string for an enum member or pass through a string."""
    if isinstance(value, enum.Enum):
        return value.value
    return value
