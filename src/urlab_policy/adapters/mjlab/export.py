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

"""Materialise mjlab's procedurally-built robot MJCF as a static XML.

mjlab's `EntityCfg` doesn't ship a complete MJCF -- the base XML in
`asset_zoo/robots/<robot>/xmls/<robot>.xml` carries geometry, joints,
sensors, and sites, but the actuators (and any collision overrides) are
added at construction time by walking `BuiltinPositionActuatorCfg`
regex patterns over the spec.

URLab imports MJCF at design time. To run a mjlab-trained policy in
URLab, the robot's MJCF must already include the same actuators mjlab
added at training time -- otherwise the policy's action schema doesn't
match. This script builds the entity, captures the resulting `MjSpec`,
and writes it back as XML at the requested output path.

Usage::

    # Export the standard G1 next to the original RoboJuDo MJCF so
    # URLab's importer can pick it up.
    python -m urlab_policy.adapters.mjlab.export \\
        --robot unitree_g1 \\
        --out RoboJuDo/assets/robots/g1/g1_mjlab.xml

By default the exporter copies mjlab's STL directory to
``<output_stem>_meshes/`` next to the output XML and rewrites
``meshdir`` to that relative name. URLab's MJCF importer joins the
XML's parent directory with ``meshdir`` and treats the result as the
mesh root, so a **relative** meshdir is required -- absolute paths get
double-concatenated to nonsense. The result is a portable two-file
bundle: move the XML + its sibling ``_meshes`` directory together.

Pass ``--no-copy-meshes`` to keep the spec's original ``meshdir`` and
manage the STL directory yourself; pass ``--meshdir-name`` to share
one mesh pool across several exported variants.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _g1_cfg():
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg

    return get_g1_robot_cfg()


def _go1_cfg():
    from mjlab.asset_zoo.robots.unitree_go1 import get_go1_robot_cfg  # type: ignore

    return get_go1_robot_cfg()


def _h1_cfg():
    """H1 entity cfg lives in the `mjlab-homierl` task package, not in
    mjlab core. The shim below makes the helper importable on mjlab 1.3
    (homierl pins to <1.3.0)."""
    from urlab_policy.adapters.robojudo import _compat as _homierl_compat  # noqa: F401  shim install
    from mjlab_homierl.robots.unitree_h1.h1_constants import get_h1_robot_cfg  # type: ignore

    return get_h1_robot_cfg()


# Map of "robot id" -> callable returning a fresh `EntityCfg`. Add new
# robots here as they're needed.
KNOWN_ROBOTS: dict[str, Callable[[], object]] = {
    "unitree_g1": _g1_cfg,
    "unitree_go1": _go1_cfg,
    "unitree_h1": _h1_cfg,
}


_MESHDIR_BY_ROBOT: dict[str, str] = {
    "unitree_g1": "mjlab/asset_zoo/robots/unitree_g1/xmls/assets",
    "unitree_go1": "mjlab/asset_zoo/robots/unitree_go1/xmls/assets",
    "unitree_h1": "mjlab_homierl/robots/unitree_h1/xmls/assets",
}


def _resolve_absolute_meshdir(robot_id: str) -> Path:
    """Find the installed STL directory for ``robot_id`` so the exported
    XML can carry an absolute ``meshdir``.

    Walks ``sys.path`` for the canonical sub-path; raises with a clear
    message if the assets aren't on the venv. Caller can pass
    ``--keep-relative-meshdir`` to bypass this if the standard layout
    doesn't apply (e.g. mjlab installed from a fork)."""
    import sys

    suffix = _MESHDIR_BY_ROBOT.get(robot_id)
    if not suffix:
        raise KeyError(f"no meshdir mapping for robot {robot_id!r}")
    for sp in sys.path:
        candidate = Path(sp) / suffix
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"meshdir for {robot_id!r} not found in any sys.path entry "
        f"(looked for {suffix!r}). Install the mjlab + robot extras "
        f"or pass --keep-relative-meshdir and place the assets manually."
    )


def export_robot_xml(
    robot_id: str,
    output_path: Path,
    *,
    copy_meshes: bool = True,
    meshdir_name: str | None = None,
) -> Path:
    """Build the mjlab `Entity` for `robot_id` and serialise its resolved
    `MjSpec` as XML at `output_path`.

    With ``copy_meshes=True`` (default) the exporter copies mjlab's STL
    directory to ``<output_path stem>_meshes/`` next to the output XML
    and rewrites ``meshdir`` to that relative name. URLab's MJCF importer
    joins the XML's parent directory with ``meshdir``, so a *relative*
    path is required -- absolute meshdirs end up double-concatenated and
    fail to resolve. The output bundle is then portable: move both the
    XML and its sibling ``_meshes`` directory together.

    With ``copy_meshes=False`` the spec's original ``meshdir`` (typically
    ``"assets/"``) is preserved verbatim. Caller must place the STL
    directory at that relative location.

    ``meshdir_name`` overrides the default ``<stem>_meshes`` directory
    name. Useful when several exported variants should share one mesh
    pool.

    Returns the absolute path of the written XML.
    """
    if robot_id not in KNOWN_ROBOTS:
        raise KeyError(
            f"unknown robot id {robot_id!r}; known: {sorted(KNOWN_ROBOTS)}"
        )
    cfg_factory = KNOWN_ROBOTS[robot_id]
    cfg = cfg_factory()

    from mjlab.entity.entity import Entity  # type: ignore

    entity = Entity(cfg)
    spec = entity.spec  # type: ignore[attr-defined]

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if copy_meshes:
        import shutil
        src_meshdir = _resolve_absolute_meshdir(robot_id)
        dst_name = meshdir_name or f"{output_path.stem}_meshes"
        dst_meshdir = output_path.parent / dst_name
        # dirs_exist_ok=True so re-exporting refreshes the bundle.
        shutil.copytree(src_meshdir, dst_meshdir, dirs_exist_ok=True)
        n_copied = sum(1 for _ in dst_meshdir.glob("*"))
        logger.info("copied %d mesh files: %s -> %s",
                    n_copied, src_meshdir, dst_meshdir)
        relative_meshdir: str | None = f"{dst_name}/"
    else:
        relative_meshdir = None

    # NB: don't mutate `spec.meshdir` before to_xml(). The serializer
    # re-validates the spec's mesh references against the meshdir at
    # serialize time, so a relative meshdir would have to resolve from
    # the spec's CWD (mjlab's install path) where our copy doesn't
    # exist. Leave the spec's original (working) meshdir alone and
    # rewrite the meshdir attribute of the serialized XML string
    # post-hoc -- URLab's importer only reads the final string, never
    # the spec's compile-time state.
    xml_str = spec.to_xml()

    if relative_meshdir is not None:
        import re
        new_xml, count = re.subn(
            r'meshdir="[^"]*"',
            f'meshdir="{relative_meshdir}"',
            xml_str,
            count=1,
        )
        if count == 0:
            # No <compiler meshdir="..."> in the output? Inject one.
            logger.warning("spec.to_xml() emitted no meshdir attribute; "
                           "injecting <compiler meshdir=%r/>", relative_meshdir)
            new_xml = re.sub(
                r"(<mujoco[^>]*>)",
                rf'\1\n  <compiler meshdir="{relative_meshdir}"/>',
                xml_str,
                count=1,
            )
        xml_str = new_xml

    output_path.write_text(xml_str, encoding="utf-8")
    logger.info("wrote %s (%d bytes)", output_path, len(xml_str))
    return output_path


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Export mjlab procedurally-built robot MJCF for URLab import.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--robot", required=True, choices=sorted(KNOWN_ROBOTS),
                        help="Robot id to export.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output XML path. By default mjlab's STL "
                             "directory is copied to '<stem>_meshes/' next to "
                             "the output XML and 'meshdir' is rewritten to "
                             "that relative name (portable bundle that "
                             "URLab can import).")
    parser.add_argument("--no-copy-meshes", action="store_true",
                        help="Don't copy the STL directory; preserve the "
                             "spec's original meshdir (typically 'assets/'). "
                             "Caller must then place the meshes at that "
                             "relative location next to the output XML.")
    parser.add_argument("--meshdir-name", default=None,
                        help="Override the default '<stem>_meshes' directory "
                             "name (useful when several exported variants "
                             "should share one mesh pool).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        out = export_robot_xml(
            args.robot, args.out,
            copy_meshes=not args.no_copy_meshes,
            meshdir_name=args.meshdir_name,
        )
    except Exception as exc:
        logger.error("export failed: %s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1
    print(f"OK: wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
