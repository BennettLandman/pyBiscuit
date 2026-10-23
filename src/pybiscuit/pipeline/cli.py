"""``pybiscuit run <input> <output> [--resume] [--dry-run]``.

Options named as in ``run_biscuit`` override the config (``--reg``, ``--parc``,
``--trace_r``, ...); ``--config`` reads an original ``biscuit.conf`` and
``--toml`` a TOML file. Exit codes: 0 ok / dry-run printed, 2 unusable input or
refused resume, 3 stage blocked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import __version__
from ..config import CLI_FLAGS, ConfigError, load_config
from .adapters import ADAPTERS, find_inputs
from .model import PipelineError, ResumeError, StageBlocked
from .runner import Run, format_plan
from .stages import STAGES

EXIT_ERROR = 2
EXIT_BLOCKED = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pybiscuit run",
        description="Run the BISCUIT pipeline on a PyMaCRUISE run, a MaCRUISE v3 output "
        "directory or a FreeSurfer 6 subject directory.",
    )
    p.add_argument("input", help="input directory (type is detected; see --input-type)")
    p.add_argument("output", help="output directory (created; holds run.json and stages/)")
    p.add_argument("--resume", action="store_true",
                   help="continue an earlier run in OUTPUT; refuses if signatures or hashes differ")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved config, input findings and stage plan; run nothing")
    p.add_argument("--input-type", choices=ADAPTERS, help="skip detection")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="accept a PyMaCRUISE run whose status is not 'complete'")
    p.add_argument("--until", metavar="STAGE", help="stop after this stage")
    p.add_argument("--config", metavar="BISCUIT.CONF", help="original biscuit.conf to read")
    p.add_argument("--toml", metavar="FILE", help="TOML configuration to read")
    g = p.add_argument_group("config overrides (names as in run_biscuit; beat --config/--toml)")
    for flag in CLI_FLAGS:
        g.add_argument(f"--{flag}", metavar="VALUE", help=f"overrides {CLI_FLAGS[flag]}")
    return p


def _config_from_args(args):
    cfg = load_config(conf=args.config, toml=args.toml,
                      overrides={f: getattr(args, f) for f in CLI_FLAGS})
    return cfg


def format_findings(f) -> str:
    lines = [f"adapter: {f.adapter}", f"root:    {f.root}", f"route:   {f.surf_recon}"]
    if f.source:
        for k, v in f.source.items():
            if v is not None and k != "stages":
                lines.append(f"  {k}: {v}")
        stages = f.source.get("stages")
        if stages:
            lines.append("  stages: " + ", ".join(
                f"{n}#{s['attempt']}:{s['status']}" for n, s in stages.items()))
    lines.append("products:")
    for n, pr in sorted(f.products.items()):
        req = "*" if n in f.required else " "
        lines.append(f"  {req} {n:<24} {pr.kind:<9} {pr.frame or '-':<6} "
                     f"{pr.sha256[:12]} {pr.bytes:>10,d} B  {pr.path}")
    if f.required:
        lines.append("  (* = required)")
    lines += [f"warning: {w}" for w in f.warnings]
    lines += [f"ERROR: {p}" for p in f.problems()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except StageBlocked as exc:
        print(f"pybiscuit run: {exc}", file=sys.stderr)
        print(f"run.json status 'blocked' written to {Path(args.output) / 'run.json'}",
              file=sys.stderr)
        return EXIT_BLOCKED
    except (PipelineError, ConfigError) as exc:
        kind = "resume refused" if isinstance(exc, ResumeError) else "error"
        print(f"pybiscuit run: {kind}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _run(args) -> int:
    cfg = _config_from_args(args)
    findings = find_inputs(args.input, args.input_type, cfg, allow_incomplete=args.allow_incomplete)
    if "surf_recon" in cfg.explicit and cfg["surf_recon"] != findings.surf_recon:
        raise ConfigError(
            f"surf_recon={cfg['surf_recon']} conflicts with the {findings.adapter} input "
            f"({findings.surf_recon} route)"
        )
    cfg = cfg.replace(surf_recon=findings.surf_recon)
    run = Run(args.output, findings, cfg, STAGES, resume=args.resume)

    if args.dry_run:
        print(f"pybiscuit {__version__} run --dry-run")
        print("\n== config ==  (sources: " + " < ".join(cfg.sources) + ")")
        for k, v in cfg.to_dict(active_only=True).items():
            print(f"  {k} = {v}")
        print("\n== input ==")
        print(format_findings(findings))
        print("\n== stage plan ==")
        state = (run.state or {}).get("status")
        mode = f"resume of existing run (status {state})" if args.resume and run.state else (
            "new run" if not run.state else f"existing run.json (status {state}); needs --resume")
        print(f"output: {args.output}  [{mode}]")
        print(format_plan(run.plan()))
        if not findings.ok:
            print("\ninput is not usable (see ERROR lines); a real run would be refused.",
                  file=sys.stderr)
            return EXIT_ERROR
        return 0

    state = run.execute(until=args.until)
    print(f"run {state['status']}: {Path(args.output) / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
