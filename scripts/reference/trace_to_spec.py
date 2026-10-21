#!/usr/bin/env python
"""Turn a command trace (B1.2) into a markdown call graph: the draft of the orchestrator spec (B1.3).

    python scripts/reference/trace_to_spec.py TRACE [-o spec.md] [--title T]

TRACE is ``trace.jsonl`` (one record per sub-tool call, written by ``trace_shim.sh``) or the
``trace/`` directory of per-call ``*.json`` records.  Output sections:

1. summary per tool (calls, total seconds, failures);
2. the call sequence in start order (cwd, argv, exit, seconds);
3. file flow: for each file a call created/modified, which later calls take it as an argument
   (the dependency edges of the orchestrator);
4. flag usage per tool (distinct flags, with value examples) for the port notes (B3.1, B8.1).

Pure standard library; deterministic output (sorted by start time, then pid).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_records(src: Path) -> list[dict]:
    if src.is_dir():
        files = sorted(src.glob("*.json"))
        lines = [f.read_text() for f in files]
    else:
        lines = src.read_text().splitlines()
    recs = []
    for i, ln in enumerate(lines):
        ln = ln.strip()
        if not ln:
            continue
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError as e:
            print(f"warning: skipping unparsable record {i + 1}: {e}", file=sys.stderr)
    recs.sort(key=lambda r: (float(r.get("start", 0)), int(r.get("pid", 0))))
    return recs


def abspath(cwd: str, p: str) -> str:
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(cwd, p))


def arg_paths(rec: dict) -> list[str]:
    """Absolute paths for every non-flag argument (candidates for file-flow edges)."""
    return [abspath(rec["cwd"], a) for a in rec.get("argv", []) if a and not a.startswith("-")]


def md_code(s: str) -> str:
    return "`" + s.replace("`", "'") + "`"


def shorten(path: str, roots: list[str]) -> str:
    for r in roots:
        if r and path.startswith(r.rstrip("/") + "/"):
            return "…/" + path[len(r.rstrip("/")) + 1 :]
    return path


def flow_edges(recs: list[dict]) -> list[tuple[int, int, str]]:
    """(producer index, consumer index, path): a later call names a file/prefix an earlier one wrote."""
    produced: dict[str, int] = {}
    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for j, rec in enumerate(recs):
        args = arg_paths(rec)
        for a in args:
            for p, i in produced.items():
                if i != j and (p == a or p.startswith(a)) and (i, j, p) not in seen:
                    seen.add((i, j, p))
                    edges.append((i, j, p))
        for p in rec.get("created", []) + rec.get("modified", []):
            produced[p] = j
    return edges


def flag_usage(recs: list[dict]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        argv = rec.get("argv", [])
        for k, a in enumerate(argv):
            if a.startswith("-") and len(a) > 1 and not a[1:].replace(".", "").isdigit():
                vals = []
                for b in argv[k + 1 :]:
                    if b.startswith("-") and not b[1:].replace(".", "").isdigit():
                        break
                    vals.append(os.path.basename(b) if "/" in b else b)
                ex = " ".join(vals)
                if ex not in out[rec["tool"]][a]:
                    out[rec["tool"]][a].append(ex)
    return out


def render(recs: list[dict], title: str) -> str:
    cwds = sorted({r["cwd"] for r in recs})
    roots = [os.path.commonpath(cwds)] if cwds else []
    L: list[str] = [f"# {title}", ""]
    if not recs:
        return "\n".join(L + ["No calls recorded.", ""])
    t0 = float(recs[0]["start"])
    total = float(max(r["end"] for r in recs)) - t0
    L += [
        f"{len(recs)} sub-tool calls over {total:.1f} s; {sum(1 for r in recs if r['exit'] != 0)} "
        f"with non-zero exit. Paths under {md_code(roots[0])} are shortened to `…/`.",
        "",
        "## Tools",
        "",
        "| tool | calls | total s | failures |",
        "|---|---|---|---|",
    ]
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_tool[r["tool"]].append(r)
    for t in sorted(by_tool):
        rs = by_tool[t]
        L.append(
            f"| {t} | {len(rs)} | {sum(float(r['end']) - float(r['start']) for r in rs):.2f} "
            f"| {sum(1 for r in rs if r['exit'] != 0)} |"
        )
    L += ["", "## Call sequence", ""]
    for i, r in enumerate(recs):
        dt = float(r["end"]) - float(r["start"])
        L.append(f"### {i + 1}. {r['tool']}  (t+{float(r['start']) - t0:.1f} s, {dt:.2f} s, exit {r['exit']})")
        L.append("")
        L.append(f"- cwd: {md_code(shorten(r['cwd'], roots))}")
        L.append(f"- command: {md_code(' '.join([r['tool'], *r.get('argv', [])]))}")
        if r.get("env"):
            L.append("- env: " + ", ".join(f"{k}={v}" for k, v in sorted(r["env"].items())))
        for key, label in (("inputs", "existing path args"), ("created", "created"), ("modified", "modified"), ("deleted", "deleted")):
            if r.get(key):
                L.append(f"- {label}: " + ", ".join(md_code(shorten(p, roots)) for p in r[key]))
        L.append("")
    L += ["## File flow", "", "| producer | file | consumer |", "|---|---|---|"]
    edges = flow_edges(recs)
    for i, j, p in edges:
        L.append(f"| {i + 1}. {recs[i]['tool']} | {md_code(shorten(p, roots))} | {j + 1}. {recs[j]['tool']} |")
    if not edges:
        L.append("| – | – | – |")
    L += ["", "## Flags by tool", ""]
    for t, flags in sorted(flag_usage(recs).items()):
        L += [f"### {t}", "", "| flag | distinct values (basenames) |", "|---|---|"]
        for f, vals in sorted(flags.items()):
            shown = "; ".join(md_code(v) if v else "(none)" for v in vals[:6])
            more = f" … +{len(vals) - 6}" if len(vals) > 6 else ""
            L.append(f"| `{f}` | {shown}{more} |")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--title", default="Orchestrator call graph (draft)")
    a = ap.parse_args()
    md = render(load_records(a.trace), a.title)
    if a.out:
        a.out.write_text(md)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
