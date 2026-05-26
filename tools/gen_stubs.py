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

"""Generate `urlab_policy/_stubs.pyi` from a running URLab editor's
`meta` reply. The committed stubs make IDE autocomplete / mypy-style
type checking work for namespace-synthesised methods.

Two modes:

  Live: connect to an editor and call `meta` directly.

      python -m tools.gen_stubs --connect tcp://127.0.0.1:5559 \\
                                --output src/urlab_policy/_stubs.pyi

  Offline: read a saved schema JSON (useful for CI without a live
  editor; the format is a single object `{"ops": [...]}`).

      python -m tools.gen_stubs --from-json tests/schema_snapshot.json \\
                                --output src/urlab_policy/_stubs.pyi

CI re-runs this against a clean editor build and fails if the committed
stubs drift — that guarantees the bridge ↔ stubs contract stays in
lock-step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# Namespaces we know about and the matching `client.<name>` attribute on
# URLabClient. Ops with a namespace not in this list land on the top-
# level fallback (rendered as a generic `URLabClient.<op>` stub).
NAMESPACE_TO_PROXY = {
    "scene": "_Scene",
    "sim": "_Sim",
    "runtime": "_Runtime",
    "outliner": "_Outliner",
    "recording": "_Recording",
    "replay": "_Replay",
}


def fetch_via_zmq(endpoint: str) -> Dict[str, Any]:
    """Connect to a live editor over ZMQ REQ, send a `meta` request,
    return the decoded reply dict. msgpack on the wire."""
    import msgpack
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(endpoint)
    try:
        sock.send(msgpack.packb({"op": "meta"}, use_bin_type=True))
        raw = sock.recv()
        return msgpack.unpackb(raw, raw=False)
    finally:
        sock.close(linger=0)


_WIRE_TO_PY = {
    "string": "str",
    "int":    "int",
    "float":  "float",
    "bool":   "bool",
    "array":  "list",
    "object": "dict",
}


def _split_field(spec: str) -> tuple:
    """Parse a 'name:type' (optional ?-suffix) reply-field spec.

    Returns ``(name, py_type, optional)``. Falls back to ``Any`` if the
    type token isn't recognised (forward-compat for new wire types)."""
    name_part, _, type_part = spec.partition(":")
    optional = type_part.endswith("?")
    if optional:
        type_part = type_part[:-1]
    py = _WIRE_TO_PY.get(type_part, "Any")
    return name_part, py, optional


def render_op_signature(op: Dict[str, Any]) -> str:
    """Render one op into a typed signature line.

    Required fields (declared via `RequiredFields` on the UE side) become
    keyword-only required parameters. Reply fields drive the return type
    annotation: when present, the stub emits a TypedDict-style alias so
    callers see ``reply["state"]`` typed without a cast."""
    name = str(op["name"])
    required = [str(f) for f in op.get("required_fields") or []]
    reply = [str(f) for f in op.get("reply_fields") or []]

    # Param list.
    parts: List[str] = ["self"]
    if required:
        parts.append("*")
        for f in required:
            parts.append(f"{f}: Any")
        parts.append("**kwargs: Any")
    else:
        parts.append("**kwargs: Any")
    sig = ", ".join(parts)

    # Return type. With a known schema we'd normally emit a per-op
    # TypedDict, but those need to be top-level. Here we just emit
    # `Dict[str, Any]` and stash the schema in a comment so IDEs that
    # render docstrings show callers the field set. Callers who want
    # strict typing can read `_ops_meta[<op>]['reply_fields']`.
    sig_line = f"    def {name}(" + sig + ") -> Dict[str, Any]: ..."
    if not reply:
        return sig_line

    field_summary = ", ".join(
        ("?" if opt else "") + n + ": " + py
        for n, py, opt in (_split_field(f) for f in reply)
    )
    return sig_line + f"  # reply: {{{field_summary}}}"


def group_by_namespace(ops: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for op in ops:
        ns = (op.get("namespace") or "").strip()
        if not ns:
            ns = "_top"
        grouped.setdefault(ns, []).append(op)
    return grouped


def render(ops: List[Dict[str, Any]]) -> str:
    """Render the full .pyi text."""
    grouped = group_by_namespace(ops)

    lines: List[str] = [
        "# AUTO-GENERATED by tools/gen_stubs.py — do not edit by hand.",
        "# Regenerate after server schema changes:",
        "#",
        "#     python -m tools.gen_stubs --connect tcp://127.0.0.1:5559 \\",
        "#                               --output src/urlab_policy/_stubs.pyi",
        "",
        "from typing import Any, Dict",
        "",
    ]

    # Per-namespace classes. Sorted for stable diffs.
    proxy_classes: List[Tuple[str, str]] = []
    for ns_name in sorted(grouped):
        if ns_name == "_top":
            continue
        proxy = NAMESPACE_TO_PROXY.get(
            ns_name,
            f"_{ns_name.title().replace('_', '')}",
        )
        proxy_classes.append((ns_name, proxy))

    for ns_name, proxy in proxy_classes:
        lines.append(f"class {proxy}:")
        ns_ops = sorted(grouped[ns_name], key=lambda o: str(o["name"]))
        for op in ns_ops:
            lines.append(render_op_signature(op))
        lines.append("")

    # URLabClient surface: namespace attributes + top-level fallback ops.
    lines.append("class URLabClient:")
    for ns_name, proxy in proxy_classes:
        lines.append(f"    {ns_name}: {proxy}")
    if "_top" in grouped:
        lines.append("")
        for op in sorted(grouped["_top"], key=lambda o: str(o["name"])):
            lines.append(render_op_signature(op))
    if not proxy_classes and "_top" not in grouped:
        lines.append("    pass")
    lines.append("")

    return "\n".join(lines)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="urlab-policy gen-stubs")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--connect", metavar="ENDPOINT",
                     help="Live ZMQ endpoint (tcp://host:port).")
    src.add_argument("--from-json", metavar="PATH",
                     help="Read a saved meta reply from a JSON file.")
    parser.add_argument("--output", required=True, metavar="PATH",
                        help="Where to write the generated .pyi.")
    args = parser.parse_args(argv)

    if args.connect:
        reply = fetch_via_zmq(args.connect)
    else:
        with open(args.from_json, "r", encoding="utf-8") as f:
            reply = json.load(f)

    if not isinstance(reply, dict) or reply.get("op") not in ("meta_ok", None):
        sys.stderr.write(f"unexpected meta reply: {reply!r}\n")
        return 2

    ops = reply.get("ops") or []
    if not isinstance(ops, list):
        sys.stderr.write(f"meta.ops is not a list: {type(ops).__name__}\n")
        return 2

    out = render([o for o in ops if isinstance(o, dict) and o.get("name")])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    print(f"Wrote {out_path} ({len(ops)} ops).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
