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

"""mjlab loader: convert an `mjlab.envs.ManagerBasedRlEnvCfg` into a
`TaskSpec` plus a freshly-built env shim against which the obs builders
will run.

The loader keeps `term.builder = term.func` -- i.e. the obs functions
in the emitted `TaskSpec` are mjlab's own. That way an mjlab task runs
through the agnostic `PolicyRunner` with byte-for-byte training-time
semantics, and we never reimplement obs functions whose subtle
conventions (frame, scale, sign) we'd risk getting wrong.

Custom command types map to a string registered in
`urlab_policy.command_sources`. Built-ins:

    - `mjlab.tasks.tracking.mdp.MotionCommandCfg`     -> `motion_file`
    - `mjlab.tasks.velocity.mdp.UniformVelocityCommandCfg` -> `urlab_twist`

Tasks with their own `CommandCfg` subclass can register a mapping via
`register_mjlab_command_mapping(MyCfg, "my_source")` before calling
`load_taskspec(...)`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...task_spec import (
    ActionSpec,
    CommandSpec,
    ObsGroupSpec,
    ObsTermSpec,
    TaskSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mjlab CommandCfg -> source-string mapping
# ---------------------------------------------------------------------------

_CMD_CFG_TO_SOURCE: Dict[type, "Tuple[str, callable]"] = {}


def register_mjlab_command_mapping(
    cfg_class: type,
    source: str,
    params_extractor: Optional[Any] = None,
) -> None:
    """Map an mjlab `CommandCfg` subclass to a `command_sources` source
    string. `params_extractor(cfg)` (optional) returns the dict of
    keyword args passed to the source factory."""
    _CMD_CFG_TO_SOURCE[cfg_class] = (source, params_extractor)


def _ensure_builtin_mappings() -> None:
    """Lazy registration of mjlab built-ins so loaders/mjlab.py imports
    cleanly even when those mjlab modules aren't present."""
    try:
        from mjlab.tasks.tracking.mdp import MotionCommandCfg  # type: ignore

        def _motion_params(cfg):
            return {
                "motion_file": cfg.motion_file,
                "body_names": list(cfg.body_names),
                "anchor_body_name": cfg.anchor_body_name,
            }
        register_mjlab_command_mapping(MotionCommandCfg, "motion_file", _motion_params)
    except ImportError:
        pass

    try:
        from mjlab.tasks.velocity.mdp.velocity_command import (  # type: ignore
            UniformVelocityCommandCfg,
        )
        register_mjlab_command_mapping(
            UniformVelocityCommandCfg, "urlab_twist", lambda cfg: {}
        )
    except ImportError:
        pass


def _resolve_cmd_source(cfg) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Walk MRO so subclasses pick up their parent's registration."""
    for cls in type(cfg).__mro__:
        entry = _CMD_CFG_TO_SOURCE.get(cls)
        if entry is not None:
            source, extractor = entry
            params = extractor(cfg) if extractor is not None else {}
            return source, params
    return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass
class MjlabLoadResult:
    spec: TaskSpec
    env_shim: Any  # the existing _EnvFacade — opaque to the runner


def load_taskspec(
    env_cfg, art, device: str = "cpu", scene_name: str = "robot"
) -> MjlabLoadResult:
    """Build a `TaskSpec` from an mjlab task config plus a freshly-built
    env shim (the existing `_EnvFacade`). The runner consumes the spec;
    obs term builders read off the shim."""
    _ensure_builtin_mappings()

    # Reuse the mature env shim from adapters/mjlab.py. The shim does
    # the heavy lifting (facades, sensor wiring, default joint pos
    # population). It still uses the legacy "env-shaped" surface that
    # mjlab obs functions read off (`scene[name].data.X`).
    #
    # `skip_command_contexts=True` + `defer_obs_probe=True`: don't have
    # the shim build legacy `_MotionContext` / `_TwistContext` (it'd
    # double-load motion files since `PolicyRunner` builds the canonical
    # set via `command_sources.build_command_source`). The obs probe is
    # deferred until after we plug those new contexts into the shim's
    # command_manager below.
    from .runtime import _EnvFacade  # type: ignore
    from ...command_sources import build_command_source

    env = _EnvFacade(
        client=None,  # unused by the shim's obs path
        art=art,
        env_cfg=env_cfg,
        device=device,
        scene_name=scene_name,
        skip_command_contexts=True,
        defer_obs_probe=True,
    )

    # ---- obs groups ----
    obs_groups: List[ObsGroupSpec] = []
    for group_name, group_cfg in env_cfg.observations.items():
        group_hist = getattr(group_cfg, "history_length", None)
        group_flatten = getattr(group_cfg, "flatten_history_dim", True)
        terms: List[ObsTermSpec] = []
        for term_name, term in group_cfg.terms.items():
            if group_hist is not None:
                hist_len = int(group_hist)
                flatten = bool(group_flatten)
            else:
                hist_len = int(getattr(term, "history_length", 0))
                flatten = bool(getattr(term, "flatten_history_dim", True))
            terms.append(ObsTermSpec(
                name=term_name,
                builder=term.func,            # mjlab's actual obs function
                params=dict(term.params),
                history_length=hist_len,
                flatten_history_dim=flatten,
                scale=term.scale,
                clip=term.clip,
            ))
        obs_groups.append(ObsGroupSpec(
            name=group_name,
            terms=terms,
            concatenate_dim=getattr(group_cfg, "concatenate_dim", -1),
        ))

    # ---- action spec ----
    if not env_cfg.actions:
        raise ValueError("env_cfg.actions is empty -- no action term to drive")
    action_cfg = next(iter(env_cfg.actions.values()))
    # Today we assume joint_position. Later: dispatch on the cfg class.
    action = ActionSpec(
        kind="joint_position",
        joint_patterns=list(action_cfg.actuator_names),
        scale=action_cfg.scale if isinstance(action_cfg.scale, dict)
              else float(action_cfg.scale),
        offset=getattr(action_cfg, "offset", 0.0),
        use_default_offset=bool(
            getattr(action_cfg, "use_default_offset", True)
        ),
    )

    # ---- commands ----
    commands: List[CommandSpec] = []
    for cname, ccfg in (getattr(env_cfg, "commands", None) or {}).items():
        resolved = _resolve_cmd_source(ccfg)
        if resolved is None:
            logger.warning(
                "command term %r has unmapped mjlab cfg type %s. Register "
                "via `urlab_policy.adapters.mjlab.loader.register_mjlab_command_mapping(...)`. "
                "Obs reading it will see zeros.",
                cname, type(ccfg).__name__,
            )
            continue
        source, params = resolved
        commands.append(CommandSpec(name=cname, source=source, params=params))

    # ---- init state ----
    init_state: Dict[str, float] = {}
    try:
        ip = getattr(env_cfg.scene.entities[scene_name].init_state, "joint_pos", {}) or {}
        init_state = {str(k): float(v) for k, v in ip.items()}
    except (AttributeError, KeyError, TypeError):
        pass

    # ---- timing ----
    physics_dt = float(getattr(env_cfg.sim, "mujoco", env_cfg.sim).timestep)
    decimation = int(env_cfg.decimation)
    episode_length_s = getattr(env_cfg, "episode_length_s", None)

    spec = TaskSpec(
        robot_name=scene_name,
        physics_dt=physics_dt,
        decimation=decimation,
        init_state=init_state,
        obs_groups=obs_groups,
        action=action,
        commands=commands,
        episode_length_s=float(episode_length_s) if episode_length_s is not None else None,
    )

    # Build command contexts via the canonical (string-keyed) registry
    # and plug them into the env shim's command_manager BEFORE running
    # the obs probe. This avoids the legacy `_EnvFacade` build path
    # double-loading motion files etc.: the shim has no legacy contexts
    # (we passed `skip_command_contexts=True`), so this is the single
    # set the runtime ever sees.
    shim_command_terms: Dict[str, Any] = {}
    entity = env.scene[scene_name]
    for cs in spec.commands:
        try:
            ctx = build_command_source(
                cs.source, art, env.scene, entity, device, **cs.params,
            )
        except Exception as exc:
            logger.warning(
                "command source %r (kind=%r) failed to build: %s -- "
                "obs reading it will see zeros",
                cs.name, cs.source, exc,
            )
            continue
        shim_command_terms[cs.name] = ctx
    # `_CommandShim._terms` is the lookup `mjlab.envs.mdp.observations.
    # generated_commands` and friends consult.
    env.command_manager._terms = shim_command_terms

    # Now safe to probe: every obs term that reads a command finds its
    # context in the shim.
    env.finalize_obs_probe()

    return MjlabLoadResult(spec=spec, env_shim=env)
