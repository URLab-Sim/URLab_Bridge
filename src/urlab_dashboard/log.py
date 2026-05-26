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

"""Append-only UI log panel. ``log_panel`` is the dpg tag the shell adds."""

from __future__ import annotations

import time

import dearpygui.dearpygui as dpg

_MAX_LINES = 200


def log(msg: str, *, error: bool = False) -> None:
    prefix = "[ERR] " if error else ""
    line = f"{time.strftime('%H:%M:%S')} {prefix}{msg}"
    if not dpg.does_item_exist("log_panel"):
        return
    existing = dpg.get_value("log_panel") or ""
    lines = (existing + "\n" + line).splitlines()[-_MAX_LINES:]
    dpg.set_value("log_panel", "\n".join(lines))
