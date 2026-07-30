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

"""Client-side helpers for the network model upload (``URLabClient.upload_model``).

The upload contract is content-addressed: the client flattens the model into a
single self-contained MJCF with bare-filename asset references, hashes every
blob, and lets the server ask only for the blobs it does not already have in its
cache. This module holds the pure, transport-free pieces of that flow so they
can be unit-tested without a live server:

- :func:`flatten_model`   -- resolve ``<include>`` and rewrite asset paths to
                             bare filenames, producing one portable XML.
- :func:`sha256_hex`      -- content hash used as the cache key.
- :func:`iter_chunks`     -- the offset / slice math for chunked blob upload.
- :func:`is_bare_filename`/:func:`require_bare_filename` -- the traversal guard
                             the server also enforces.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, List, Optional, Tuple, Union

# Server-enforced ceiling on the number of assets in one upload; mirrored here
# so an over-large model fails fast client-side with a clear message instead of
# after a round-trip.
MAX_ASSETS = 4096

# Element tags whose ``file=`` reference is resolved against ``texturedir``;
# everything else with a ``file=`` (mesh / hfield / skin) uses ``meshdir``.
_TEXTURE_TAGS = frozenset({"texture"})

PathLike = Union[str, bytes, "os.PathLike[str]"]


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data`` (the blob cache key)."""
    return hashlib.sha256(data).hexdigest()


def is_bare_filename(name: str) -> bool:
    """True if ``name`` is a plain filename safe to use as a cache key.

    Rejects anything that could escape a directory or carry a rooted path:
    empty, ``.``/``..`` components, forward or back slashes, a leading
    separator, or a Windows drive letter (``C:...``).
    """
    if not name or name in (".", ".."):
        return False
    if "/" in name or "\\" in name:
        return False
    if ".." in name.split("/") or ".." in name.split("\\"):
        return False
    if name.startswith(("/", "\\")):
        return False
    # Windows drive-qualified path, e.g. "C:foo" or "C:\foo".
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        return False
    if os.path.isabs(name):
        return False
    return True


def require_bare_filename(name: str) -> str:
    """Return ``name`` unchanged, or raise ``ValueError`` if it is not bare."""
    if not is_bare_filename(name):
        raise ValueError(
            f"asset key {name!r} is not a bare filename; keys must have no "
            f"directory components, '..', leading separator or drive letter"
        )
    return name


def iter_chunks(data: bytes, chunk_bytes: int) -> Iterator[Tuple[int, bytes]]:
    """Yield ``(offset, chunk)`` pairs covering ``data`` in ``chunk_bytes`` slices.

    A zero-length blob yields exactly one ``(0, b"")`` pair so the server still
    receives a terminating chunk. ``chunk_bytes`` must be >= 1.
    """
    if chunk_bytes < 1:
        raise ValueError(f"chunk_bytes must be >= 1, got {chunk_bytes}")
    total = len(data)
    if total == 0:
        yield 0, b""
        return
    offset = 0
    while offset < total:
        chunk = data[offset : offset + chunk_bytes]
        yield offset, bytes(chunk)
        offset += len(chunk)


# ---------------------------------------------------------------------------
# MJCF flatten
# ---------------------------------------------------------------------------


def _effective_asset_dirs(root: ET.Element) -> Tuple[str, str]:
    """Compute the effective (meshdir, texturedir) from every ``<compiler>``.

    MuJoCo treats these as global settings relative to the main model
    directory; when several ``<compiler>`` elements set the same attribute the
    last one wins. ``assetdir`` is the fallback for whichever of meshdir /
    texturedir is unset.
    """
    meshdir: Optional[str] = None
    texturedir: Optional[str] = None
    assetdir: Optional[str] = None
    for compiler in root.iter("compiler"):
        if compiler.get("meshdir") is not None:
            meshdir = compiler.get("meshdir")
        if compiler.get("texturedir") is not None:
            texturedir = compiler.get("texturedir")
        if compiler.get("assetdir") is not None:
            assetdir = compiler.get("assetdir")
    mesh = meshdir if meshdir is not None else (assetdir or "")
    texture = texturedir if texturedir is not None else (assetdir or "")
    return mesh, texture


def _resolve_includes(elem: ET.Element, base_dir: str, stack: List[str]) -> None:
    """Splice every ``<include>`` descendant of ``elem`` in place.

    ``base_dir`` is the directory of the file that owns ``elem`` (include paths
    are resolved relative to the including file). Nested includes are resolved
    recursively; ``stack`` guards against include cycles.
    """
    new_children: List[ET.Element] = []
    for child in list(elem):
        if child.tag == "include":
            fname = child.get("file")
            if not fname:
                raise ValueError("<include> element is missing a 'file' attribute")
            inc_path = fname if os.path.isabs(fname) else os.path.join(base_dir, fname)
            inc_path = os.path.normpath(inc_path)
            if inc_path in stack:
                raise ValueError(f"include cycle detected at {inc_path!r}")
            if not os.path.isfile(inc_path):
                raise FileNotFoundError(
                    f"included MJCF file not found: {inc_path!r} "
                    f"(referenced as file={fname!r})"
                )
            inc_root = ET.parse(inc_path).getroot()
            inc_dir = os.path.dirname(inc_path)
            _resolve_includes(inc_root, inc_dir, stack + [inc_path])
            # The included file's root is <mujoco> or <mujocoinclude>; its
            # children are inserted at the include's position.
            new_children.extend(list(inc_root))
        else:
            _resolve_includes(child, base_dir, stack)
            new_children.append(child)
    elem[:] = new_children


def flatten_model(
    xml: PathLike,
    *,
    asset_root: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """Flatten an MJCF model into one self-contained XML with bare asset refs.

    Resolves every ``<include>`` / ``<mujocoinclude>`` into the tree so the
    result needs no companion files, and rewrites each ``file=`` reference plus
    ``meshdir`` / ``texturedir`` / ``assetdir`` so asset paths become bare
    filenames (the server's content-addressed cache keys on bare names).

    ``xml`` may be a filesystem path, an XML string, an XML ``bytes`` blob, or
    an ``os.PathLike``. ``asset_root`` overrides the base directory used to
    resolve includes and asset files; it defaults to the XML file's directory
    (for a path) or the current working directory (for string / bytes input).

    Returns ``(flattened_xml_text, asset_paths)`` where ``asset_paths`` maps
    each referenced bare filename to its resolved absolute source path on disk
    (used to read bytes when the caller does not supply them explicitly).

    Raises ``ValueError`` on a bad include, an unresolvable asset reference, or
    a bare-name collision between two distinct source files.
    """
    text, base_dir = _load_xml_text(xml, asset_root)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"could not parse MJCF XML: {exc}") from exc

    _resolve_includes(root, base_dir, [os.path.join(base_dir, "<main>")])

    mesh_dir, texture_dir = _effective_asset_dirs(root)

    asset_paths: Dict[str, str] = {}
    for el in root.iter():
        fref = el.get("file")
        if fref is None or el.tag in ("include", "compiler"):
            continue
        sub_dir = texture_dir if el.tag in _TEXTURE_TAGS else mesh_dir
        if os.path.isabs(fref):
            abspath = os.path.normpath(fref)
        else:
            abspath = os.path.normpath(os.path.join(base_dir, sub_dir, fref))
        bare = os.path.basename(fref.replace("\\", "/"))
        if not bare:
            raise ValueError(f"asset reference {fref!r} has no filename component")
        prior = asset_paths.get(bare)
        if prior is not None and prior != abspath:
            raise ValueError(
                f"asset filename collision: {bare!r} maps to both {prior!r} and "
                f"{abspath!r}. Bare-filename uploads require unique basenames."
            )
        asset_paths[bare] = abspath
        el.set("file", bare)

    # Drop the now-meaningless asset directories so the server resolves every
    # rewritten `file=` as a bare name against its VFS.
    for compiler in root.iter("compiler"):
        for attr in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(attr, None)

    return ET.tostring(root, encoding="unicode"), asset_paths


def _load_xml_text(xml: PathLike, asset_root: Optional[str]) -> Tuple[str, str]:
    """Return ``(xml_text, base_dir)`` for a path / str / bytes / PathLike input.

    ``base_dir`` is where includes and asset files resolve from: ``asset_root``
    when given, else the XML file's directory (path input) or the current
    working directory (in-memory input).
    """
    if isinstance(xml, (bytes, bytearray)):
        text = bytes(xml).decode("utf-8")
        return text, asset_root or os.getcwd()

    if hasattr(xml, "__fspath__"):
        path = os.fspath(xml)
    elif isinstance(xml, str):
        stripped = xml.lstrip()
        if stripped.startswith("<"):
            return xml, asset_root or os.getcwd()
        path = xml
    else:  # pragma: no cover - defensive
        raise TypeError(
            f"xml must be a path, str, bytes or os.PathLike, got {type(xml).__name__}"
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"MJCF file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text, asset_root or os.path.dirname(os.path.abspath(path))
