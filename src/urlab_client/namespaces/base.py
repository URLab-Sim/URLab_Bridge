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

"""Base namespace class for `client.<ns>.<op>` proxies."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..client import URLabClient


class _RpcNamespace:
    """Base for `client.<namespace>.<op>`. Concrete subclasses hold
    hand-written methods; `__getattr__` synthesises generic RPC callables
    from server meta for ops added without a corresponding bridge release."""

    __slots__ = ("_client", "_namespace")

    def __init__(self, client: "URLabClient", namespace: str):
        self._client = client
        self._namespace = namespace

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        client = self._client
        decl = client._ops_meta.get(name)
        decl_ns = (decl or {}).get("namespace") or ""
        if decl is not None and decl_ns and decl_ns != self._namespace:
            raise AttributeError(
                f"{self._namespace!r} namespace has no op {name!r}; "
                f"it lives on `client.{decl_ns}` per meta"
            )

        if decl is not None and decl_ns == self._namespace:
            def _call(**payload) -> Dict[str, Any]:
                return client._rpc(name, payload, expected_op=f"{name}_ok")
            _call.__name__ = name
            _call.__qualname__ = f"{self._namespace}.{name}"
            return _call

        raise AttributeError(
            f"{self._namespace!r} namespace has no op {name!r}; "
            f"server hasn't advertised it (call connect() first), "
            f"and the namespace class has no hand-written method either"
        )

    def __dir__(self) -> List[str]:
        # Includes hand-written methods + meta-synthesised ops so
        # repl/IDE completion sees the full surface.
        names: set = {
            n for n in dir(type(self))
            if not n.startswith("_")
        }
        for op_name, decl in self._client._ops_meta.items():
            if decl.get("namespace") == self._namespace:
                names.add(op_name)
        return sorted(names)
