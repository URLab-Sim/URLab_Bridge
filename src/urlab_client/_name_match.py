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

"""Regex-based name matching helpers, mirrored from mjlab's
`utils.lab_api.string` so `urlab_policy.runner` stays usable when mjlab
isn't installed (e.g. a YAML-only deployment of a non-mjlab policy).
The semantics match mjlab byte-for-byte; we just don't import it."""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Sequence, Tuple, Union


def resolve_matching_names(
    keys: Union[str, Sequence[str]],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> Tuple[List[int], List[str]]:
    """Match each entry in `keys` (regex) against `list_of_strings` and
    return `(indices, names)` of matches. When `preserve_order=True`,
    the result is ordered by the query keys; otherwise by the natural
    order of `list_of_strings` (mjlab's default)."""
    if isinstance(keys, str):
        keys = [keys]

    index_list: List[int] = []
    names_list: List[str] = []
    key_idx_list: List[int] = []
    target_match: List[Union[str, None]] = [None] * len(list_of_strings)
    keys_match: List[List[str]] = [[] for _ in keys]

    for ti, target in enumerate(list_of_strings):
        for ki, rk in enumerate(keys):
            if re.fullmatch(rk, target):
                if target_match[ti]:
                    raise ValueError(
                        f"Multiple matches for {target!r}: "
                        f"{target_match[ti]!r} and {rk!r}"
                    )
                target_match[ti] = rk
                index_list.append(ti)
                names_list.append(target)
                key_idx_list.append(ki)
                keys_match[ki].append(target)

    if preserve_order:
        # Reorder so that all matches of keys[0] come first, then keys[1], etc.
        new_idx: List[int] = []
        new_names: List[str] = []
        for ki in range(len(keys)):
            for slot, kidx in enumerate(key_idx_list):
                if kidx == ki:
                    new_idx.append(index_list[slot])
                    new_names.append(names_list[slot])
        index_list = new_idx
        names_list = new_names

    if not all(keys_match):
        unmatched = [k for k, m in zip(keys, keys_match) if not m]
        raise ValueError(
            f"No matches for regex(es) {unmatched!r}. "
            f"Available: {list(list_of_strings)!r}"
        )
    return index_list, names_list


def resolve_matching_names_values(
    data: Mapping[str, float],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> Tuple[List[int], List[str], List[float]]:
    """Like `resolve_matching_names` but for `{regex: value}` dicts.
    Returns parallel `(indices, names, values)` lists. `preserve_order`
    forwards to `resolve_matching_names`."""
    index_list: List[int] = []
    names_list: List[str] = []
    values_list: List[float] = []
    for rk, val in data.items():
        ids, names = resolve_matching_names(
            rk, list_of_strings, preserve_order=preserve_order
        )
        for i, n in zip(ids, names):
            index_list.append(i)
            names_list.append(n)
            values_list.append(float(val))
    return index_list, names_list, values_list
