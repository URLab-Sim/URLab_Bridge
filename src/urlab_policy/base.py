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

"""Abstract base classes for URLab-side policies, environments, and
controllers.

These are the contracts adapters wrap their backend's classes against.
URLab core (`NativePolicyRunner`, the registry, the dashboard launcher)
type-hints against these ABCs; concrete adapters
(`urlab_policy.adapters.robojudo`, `.lerobot`, `.mjlab`) provide
implementations that may inherit from a backend class while exposing
this surface.

URLab does not edit RoboJuDo. Instead, RoboJuDo's existing
``robojudo.policy.Policy`` happens to duck-type the contract here, so
RoboJuDo-shaped policies pass through `NativePolicyRunner` unchanged
even though they don't formally subclass `Policy`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Tuple


class Policy(ABC):
    """Minimal policy contract consumed by `NativePolicyRunner`.

    Mirrors what RoboJuDo's `Policy` already exposes — the runner reads
    these methods directly. Implementations either subclass `Policy`
    directly (URLab-bundled / custom policies) or duck-type it
    (RoboJuDo `Policy` subclasses).

    Required attributes:

    - ``cfg_obs_dof`` — dataclass with at least ``joint_names`` (sequence
      of strings naming the joints whose qpos/qvel feed observations).
    - ``cfg_action_dof`` — dataclass with at least ``joint_names``,
      ``stiffness``, ``damping``, ``default_pos`` (action-space joints
      and their PD gains, used when ``push_gains=True``).
    - ``freq`` (optional) — policy step rate in Hz, used by
      `NativePolicyRunner` to infer decimation when not passed
      explicitly.
    """

    cfg_obs_dof: Any
    cfg_action_dof: Any

    @abstractmethod
    def get_observation(
        self, env_data: Any, ctrl_data: Mapping[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build the observation vector from an `EnvData` namespace and
        a `ctrl_data` dict (joystick / twist input). Returns
        ``(obs, extras)`` — `extras` is forwarded to the runner's
        diagnostic surface and may be empty."""

    @abstractmethod
    def get_action(self, obs: Any) -> Any:
        """Compute the next action from `obs`. Returned shape must
        broadcast to ``len(cfg_action_dof.joint_names)`` after numpy
        conversion."""

    def post_step_callback(self) -> None:
        """Hook called after every URLab step. Override for gait-clock
        ticks, heading-alignment updates, etc. Default: no-op."""

    def reset(self) -> None:
        """Reset internal state (gait clocks, action history, motion
        playback indices). Called by `NativePolicyRunner.reset()`.
        Default: no-op."""


class PolicyEnv(ABC):
    """Gymnasium-compatible environment contract.

    Adapters wrap their backend's env in this surface so URLab's
    higher-level entry points (CLI launcher, dashboard policy tab) can
    consume any framework's env without backend-specific glue.
    """

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> Tuple[Any, Dict[str, Any]]:
        """Returns ``(obs, info)``."""

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Returns ``(obs, reward, terminated, truncated, info)``."""

    @abstractmethod
    def close(self) -> None:
        """Tear down resources (URLab session, ZMQ sockets, etc.)."""


class PolicyController(ABC):
    """Per-articulation controller contract. Adapters register a concrete
    controller via ``register_controller(...)`` from their package
    ``__init__.py``; no import-time monkey-patches."""

    @abstractmethod
    def update(self) -> None:
        """Called once per step before observation build. Pulls fresh
        input (twist / joystick / motion index) into internal state."""

    @abstractmethod
    def axes(self) -> Dict[str, float]:
        """Returns the joystick-shaped axes mapping (e.g.
        ``{"LeftX": 0.1, "LeftY": 0.0, "RightX": -0.3}``) the active
        policy reads via `ctrl_data["JoystickCtrl"]["axes"]`."""
