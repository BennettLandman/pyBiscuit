"""Run orchestration: planning, execution, checkpoints, ``--resume``, ``run.json``.

Layout under the output directory::

    run.json                        manifest (schema_version 1)
    stages/<name>/attempt-NN/       stage products + provenance.json

``run.json`` follows PyMaCRUISE's conventions where they apply: ``schema_version``
1, ``status``, ``stages{name: {attempt, status, ...}}`` with per-stage
``dependency_signature`` and hashed ``outputs``, a ``history`` list, a
``run_signature``, ``spatial_convention`` and ``final_outputs`` as a flat list of
file records. Output paths are stored relative to the output directory so a run
can be moved.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import Config
from .model import (
    InputFindings,
    PipelineError,
    ResumeError,
    Stage,
    StageBlocked,
    StageContext,
    sha256_file,
)
from .signature import components, digest

__all__ = ["SCHEMA_VERSION", "Run", "StagePlan", "format_plan", "order_stages"]

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def order_stages(stages: list[Stage], route: str) -> list[Stage]:
    """Stages of ``route`` in dependency order (declaration order must already be one)."""
    graph = [s for s in stages if route in s.routes]
    names = {s.name for s in graph}
    seen: set[str] = set()
    for s in graph:
        # references to stages that this route does not have (e.g. surfprep on the
        # FreeSurfer route) are dropped; everything else must be declared earlier
        missing = [u for u in s.upstream() if u in names and u not in seen]
        if missing:
            raise PipelineError(f"stage {s.name!r} depends on {missing}, not defined before it")
        if s.name in seen:
            raise PipelineError(f"duplicate stage {s.name!r}")
        seen.add(s.name)
    return graph


@dataclass
class StagePlan:
    stage: Stage
    signature: str
    components: dict[str, Any]
    state: str  # run | valid | blocked | stale | corrupt
    detail: list[str] = field(default_factory=list)
    attempt: int | None = None  # recorded attempt (valid / stale / corrupt) or next one (run)


def _diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    out = []
    for section in ("inputs", "params", "assets"):
        o, n = old.get(section, {}), new.get(section, {})
        for k in sorted(set(o) | set(n)):
            if o.get(k) != n.get(k):
                a, b = o.get(k, "<absent>"), n.get(k, "<absent>")
                if section != "params":
                    a, b = str(a)[:12], str(b)[:12]
                out.append(f"{section[:-1] if section != 'assets' else 'asset'} {k}: {a} -> {b}")
    for k in ("stage_version", "pybiscuit"):
        if old.get(k) != new.get(k):
            out.append(f"code {k}: {old.get(k)} -> {new.get(k)}")
    return out or ["signature differs (recorded components identical; stage graph changed)"]


class Run:
    def __init__(
        self,
        out: str | Path,
        findings: InputFindings,
        config: Config,
        stages: list[Stage],
        *,
        resume: bool = False,
    ):
        self.out = Path(out)
        self.findings = findings
        self.config = config
        self.route = findings.surf_recon
        self.stages = order_stages(stages, self.route)
        self.resume = resume
        self.state: dict[str, Any] | None = None
        run_json = self.out / "run.json"
        if run_json.exists():
            try:
                self.state = json.loads(run_json.read_text())
            except json.JSONDecodeError as exc:
                raise ResumeError(f"{run_json} is not valid JSON ({exc}); cannot resume") from exc

    # ------------------------------------------------------------------ planning
    def plan(self) -> list[StagePlan]:
        sigs: dict[str, str] = {}
        plans = []
        recorded = (self.state or {}).get("stages", {}) if self.resume else {}
        for st in self.stages:
            comp = components(st, self.findings, self.config, sigs)
            sig = digest(comp)
            sigs[st.name] = sig
            rec = recorded.get(st.name)
            if rec and rec.get("status") == "complete":
                if rec.get("dependency_signature") != sig:
                    plans.append(
                        StagePlan(st, sig, comp, "stale", _diff(rec.get("components", {}), comp),
                                  rec.get("attempt"))
                    )
                    continue
                bad = self._check_outputs(rec)
                if bad:
                    plans.append(StagePlan(st, sig, comp, "corrupt", bad, rec.get("attempt")))
                else:
                    plans.append(StagePlan(st, sig, comp, "valid", [], rec.get("attempt")))
            else:
                state = "blocked" if st.blocked else "run"
                plans.append(
                    StagePlan(st, sig, comp, state, [st.blocked] if st.blocked else [],
                              self._next_attempt(st.name))
                )
        return plans

    def _check_outputs(self, rec: dict[str, Any]) -> list[str]:
        bad = []
        for o in rec.get("outputs", []):
            p = self.out / o["path"]
            if not p.is_file():
                bad.append(f"output missing: {o['path']}")
            elif p.stat().st_size != o["bytes"] or sha256_file(p) != o["sha256"]:
                bad.append(f"output changed since it was recorded (SHA-256/size): {o['path']}")
        return bad

    def _next_attempt(self, name: str) -> int:
        d = self.out / "stages" / name
        nums = [
            int(p.name.split("-")[1])
            for p in (d.glob("attempt-*") if d.is_dir() else [])
            if p.name.split("-")[1:2] and p.name.split("-")[1].isdigit()
        ]
        return max(nums, default=0) + 1

    # ----------------------------------------------------------------- execution
    def execute(self, *, until: str | None = None) -> dict[str, Any]:
        """Run the graph (up to and including ``until``). Raises :class:`StageBlocked`
        at the first blocked stage and :class:`ResumeError` if ``--resume`` is refused;
        other exceptions from stage bodies propagate after being recorded."""
        self.findings.require_ok()
        if until is not None and until not in {s.name for s in self.stages}:
            raise PipelineError(f"unknown stage {until!r}; stages: {[s.name for s in self.stages]}")
        if self.resume:
            if self.state is None:
                raise ResumeError(f"--resume: no {self.out / 'run.json'} to resume from")
            self._check_resume_inputs()
        elif self.state is not None:
            raise ResumeError(
                f"{self.out} already contains a run (run.json). Use --resume to continue it, "
                "or choose an empty output directory."
            )
        plans = self.plan()
        refused = [p for p in plans if p.state in ("stale", "corrupt")]
        if refused:
            lines = []
            for p in refused:
                what = "outputs damaged"
                if p.state == "stale":
                    what = "inputs/params/assets/code changed"
                lines.append(f"stage {p.stage.name} (attempt {p.attempt:02d}): {what}")
                lines += [f"    {d}" for d in p.detail]
            raise ResumeError(
                "--resume refused; recorded state does not match this request:\n"
                + "\n".join(lines)
                + "\nUse a new output directory, or remove the stage's attempt directory and "
                "its entry in run.json to recompute it deliberately."
            )

        self.out.mkdir(parents=True, exist_ok=True)
        state = self._init_state(plans)
        state["status"] = "running"
        self._save(state)
        try:
            for p in plans:
                st = p.stage
                if p.state == "valid":
                    state["stages"][st.name]["status"] = "complete"  # carried over
                elif st.blocked:
                    state["stages"][st.name] = {
                        "status": "blocked", "reason": st.blocked, "plan_rows": list(st.plan_rows),
                        "dependency_signature": p.signature, "components": p.components,
                    }
                    state["status"] = "blocked"
                    self._save(state)
                    raise StageBlocked(f"stage '{st.name}' is blocked: {st.blocked}")
                else:
                    self._run_stage(state, p)
                if st.name == until:
                    break
            done = all(state["stages"].get(s.name, {}).get("status") == "complete"
                       for s in self.stages)
            state["status"] = "complete" if done else "partial"
            self._finalize(state)
            self._save(state)
        except StageBlocked:
            raise
        except BaseException as exc:
            state["status"] = "failed"
            state["error"] = f"{type(exc).__name__}: {exc}"
            self._save(state)
            raise
        return state

    def _check_resume_inputs(self) -> None:
        recorded = (self.state or {}).get("inputs", {}).get("products", {})
        old = {n: p["sha256"] for n, p in recorded.items()}
        new = self.findings.hashes()
        changed = [
            f"{n}: {str(old.get(n, '<absent>'))[:12]} -> {str(new.get(n, '<absent>'))[:12]}"
            for n in sorted(set(old) | set(new))
            if old.get(n) != new.get(n)
        ]
        if changed:
            raise ResumeError(
                "--resume refused; the input products differ from the ones this run started "
                "with:\n    " + "\n    ".join(changed)
            )

    def _init_state(self, plans: list[StagePlan]) -> dict[str, Any]:
        if self.state is not None and self.resume:
            state = self.state
            state.pop("error", None)
        else:
            state = {
                "schema_version": SCHEMA_VERSION,
                "tool": "pybiscuit",
                "pybiscuit_version": __version__,
                "spatial_convention": "RAS",
                "versions": _versions(),
                "inputs": self.findings.to_json(),
                "config": {
                    "values": self.config.to_dict(active_only=True),
                    "sources": list(self.config.sources),
                },
                "stages": {},
                "history": [],
                "started": _now(),
            }
        state["config"] = {
            "values": self.config.to_dict(active_only=True),
            "sources": list(self.config.sources),
        }
        state["pybiscuit_version"] = __version__
        state["assets"] = {
            f"{p.stage.name}:{k}": v for p in plans for k, v in p.components["assets"].items()
        }
        state["run_signature"] = digest(
            {"inputs": self.findings.hashes(), "stages": {p.stage.name: p.signature for p in plans}}
        )
        state["frames"] = self.findings.frames
        return state

    def _run_stage(self, state: dict[str, Any], p: StagePlan) -> None:
        st = p.stage
        if st.body is None:
            raise PipelineError(f"stage {st.name!r} has neither a body nor a blocked reason")
        attempt = self._next_attempt(st.name)
        adir = self.out / "stages" / st.name / f"attempt-{attempt:02d}"
        adir.mkdir(parents=True)
        rec: dict[str, Any] = {
            "attempt": attempt, "status": "running", "dependency_signature": p.signature,
            "components": p.components, "plan_rows": list(st.plan_rows),
        }
        state["stages"][st.name] = rec
        state["history"].append({"attempt": attempt, "stage": st.name, "status": "running"})
        self._save(state)
        ctx = StageContext(
            stage=st,
            attempt_dir=adir,
            inputs=self._resolve_inputs(st, state),
            params={k: self.config[k] for k in st.params},
            config=self.config,
            findings=self.findings,
        )
        t0 = time.monotonic()
        started = _now()
        try:
            st.body(ctx)
            missing = [o for o in st.outputs if not (adir / o).is_file()]
            if missing:
                raise PipelineError(f"stage {st.name} did not write declared outputs: {missing}")
            outputs = [
                {
                    "path": (adir / o).relative_to(self.out).as_posix(),
                    "sha256": sha256_file(adir / o),
                    "bytes": (adir / o).stat().st_size,
                }
                for o in st.outputs
            ]
        except BaseException as exc:
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            state["history"][-1]["status"] = "failed"
            (adir / "provenance.json").write_text(json.dumps(
                {"stage": st.name, "attempt": attempt, "status": "failed", "error": rec["error"],
                 "traceback": traceback.format_exc(), "dependency_signature": p.signature,
                 "started": started}, indent=2) + "\n")
            raise
        rec["status"] = "complete"
        rec["outputs"] = outputs
        rec["elapsed_s"] = round(time.monotonic() - t0, 3)
        state["history"][-1]["status"] = "complete"
        (adir / "provenance.json").write_text(json.dumps(
            {"stage": st.name, "attempt": attempt, "status": "complete",
             "dependency_signature": p.signature, "components": p.components,
             "plan_rows": list(st.plan_rows), "pybiscuit_version": __version__,
             "started": started, "finished": _now(), "outputs": outputs}, indent=2) + "\n")
        self._save(state)

    def _resolve_inputs(self, st: Stage, state: dict[str, Any]) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for name in st.inputs:
            if name == "*":
                continue
            if ":" in name:
                up, out = name.split(":", 1)
                if up not in state["stages"]:
                    continue
                rec = state["stages"][up]
                paths[name] = self.out / "stages" / up / f"attempt-{rec['attempt']:02d}" / out
            elif name in self.findings.products:
                paths[name] = Path(self.findings.products[name].path)
        return paths

    def _finalize(self, state: dict[str, Any]) -> None:
        consumed = {u for s in self.stages for u in s.upstream()}
        state["final_outputs"] = [
            o
            for s in self.stages
            if s.name not in consumed
            for o in state["stages"].get(s.name, {}).get("outputs", [])
        ]
        state["finished"] = _now()

    def _save(self, state: dict[str, Any]) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.out / "run.json.tmp"
        tmp.write_text(json.dumps(state, indent=2, sort_keys=False) + "\n")
        os.replace(tmp, self.out / "run.json")
        self.state = state


def _versions() -> dict[str, str]:
    import nibabel
    import numpy
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "nibabel": nibabel.__version__,
    }


def format_plan(plans: list[StagePlan]) -> str:
    """Human-readable stage plan (used by ``--dry-run``)."""
    lines = []
    for i, p in enumerate(plans, 1):
        st = p.stage
        tag = {"run": "will run", "valid": "up to date (resume skips)", "blocked": "BLOCKED",
               "stale": "STALE (resume refused)", "corrupt": "DAMAGED (resume refused)"}[p.state]
        lines.append(f"{i:2d}. {st.name:<10} [{', '.join(st.plan_rows)}] {tag}")
        lines.append(f"      signature {p.signature[:16]}  outputs: {', '.join(st.outputs) or '-'}")
        deps = st.upstream()
        if deps:
            lines.append(f"      needs: {', '.join(deps)}")
        if st.params:
            params = p.components["params"]
            lines.append("      params: " + ", ".join(f"{k}={params[k]}" for k in st.params))
        if st.assets:
            assets = p.components["assets"]
            lines.append("      assets: " + ", ".join(f"{k}={v[:12]}" for k, v in assets.items()))
        for d in p.detail:
            lines.append(f"      ! {d}")
    return "\n".join(lines)
