"""Cohort mosaic, summary distributions, and parity views (B12.6, docs/visualization-plan.md §4).

Provides:
- Statistical comparison between runs (vertex-wise differences, Bland-Altman, Dice per ROI).
- Intra-class correlation coefficients (ICC) across sessions/runs.
- Cohort-level distribution aggregation across subjects per ROI and measure.
- Outlier detection and QA summary.
- Standalone HTML cohort and parity report generation.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..stats.pipeline import MEASURES, StatsTable, read_stats_csv

# -----------------------------------------------------------------------------
# Statistical Metrics: Bland-Altman, ICC, Dice, Scalar comparison
# -----------------------------------------------------------------------------


def bland_altman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute Bland-Altman agreement statistics between x and y.

    Returns:
        dict with mean_diff (bias), std_diff, lower_loa (mean - 1.96*std),
        upper_loa (mean + 1.96*std), and max_abs_diff.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    diff = x - y
    valid = np.isfinite(diff)
    if not np.any(valid):
        return {
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "lower_loa": float("nan"),
            "upper_loa": float("nan"),
            "max_abs_diff": float("nan"),
        }
    d = diff[valid]
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "lower_loa": mean_d - 1.96 * std_d,
        "upper_loa": mean_d + 1.96 * std_d,
        "max_abs_diff": float(np.max(np.abs(d))),
    }


def compute_icc(ratings: np.ndarray, icc_type: str = "icc31") -> float:
    """Compute Intra-class Correlation Coefficient (ICC).

    Args:
        ratings: array of shape (n_targets, n_raters) with observations.
        icc_type: 'icc31' (two-way mixed, single score, consistency) or
                  'icc21' (two-way random, single score, absolute agreement), or
                  'icc11' (one-way random).

    Returns:
        ICC scalar float.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    if ratings.ndim != 2:
        raise ValueError(f"ratings must be 2D array, got shape {ratings.shape}")
    n, k = ratings.shape
    if n < 2 or k < 2:
        return float("nan")

    # Target means, rater means, grand mean
    mean_total = np.mean(ratings)
    mean_targets = np.mean(ratings, axis=1)
    mean_raters = np.mean(ratings, axis=0)

    # Sum of squares
    ss_total = np.sum((ratings - mean_total) ** 2)
    ss_rows = k * np.sum((mean_targets - mean_total) ** 2)
    ss_cols = n * np.sum((mean_raters - mean_total) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    if icc_type.lower() in ("icc31", "icc(3,1)"):
        # Consistency
        denom = ms_rows + (k - 1) * ms_error
        return float((ms_rows - ms_error) / denom) if denom > 0 else float("nan")
    elif icc_type.lower() in ("icc21", "icc(2,1)"):
        # Absolute agreement
        denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
        return float((ms_rows - ms_error) / denom) if denom > 0 else float("nan")
    elif icc_type.lower() in ("icc11", "icc(1,1)"):
        # One-way random
        ss_within = ss_total - ss_rows
        ms_within = ss_within / (n * (k - 1))
        denom = ms_rows + (k - 1) * ms_within
        return float((ms_rows - ms_within) / denom) if denom > 0 else float("nan")
    else:
        raise ValueError(f"Unknown icc_type: {icc_type!r}")


def roi_dice(labels1: np.ndarray, labels2: np.ndarray) -> dict[int, float]:
    """Compute per-ROI Dice similarity coefficient between two parcellations.

    Args:
        labels1, labels2: integer label arrays of identical shape.

    Returns:
        dict mapping roi_id -> Dice coefficient (0.0 to 1.0).
    """
    labels1 = np.asarray(labels1, dtype=np.int64).ravel()
    labels2 = np.asarray(labels2, dtype=np.int64).ravel()
    if labels1.shape != labels2.shape:
        raise ValueError(f"Shape mismatch: {labels1.shape} vs {labels2.shape}")

    unique_rois = np.union1d(labels1, labels2)
    dice_map: dict[int, float] = {}
    for roi in unique_rois:
        if roi == 0:  # background
            continue
        m1 = labels1 == roi
        m2 = labels2 == roi
        s1 = np.sum(m1)
        s2 = np.sum(m2)
        if s1 + s2 == 0:
            dice_map[int(roi)] = 1.0
        else:
            intersection = np.sum(m1 & m2)
            dice_map[int(roi)] = float(2.0 * intersection / (s1 + s2))
    return dice_map


def compare_scalar_maps(ours: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Compare two scalar arrays vertex-wise.

    Returns:
        dict with pearson_r, mae, rmse, max_abs, frac_exact, and bland-altman stats.
    """
    ours = np.asarray(ours, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if ours.shape != ref.shape:
        return {"shape_mismatch": (list(ours.shape), list(ref.shape))}

    valid = np.isfinite(ours) & np.isfinite(ref)
    if not np.any(valid):
        return {"valid_count": 0}

    o, r = ours[valid], ref[valid]
    diff = np.abs(o - r)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((o - r) ** 2)))
    max_abs = float(np.max(diff))
    r_std = float(np.std(r))
    o_std = float(np.std(o))
    corr = float(np.corrcoef(o, r)[0, 1]) if (r_std > 0 and o_std > 0) else 1.0
    frac_exact = float(np.mean(diff < 5e-7))

    ba = bland_altman(o, r)
    return {
        "n_vertices": int(len(o)),
        "pearson_r": corr,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "frac_exact_6dp": frac_exact,
        **ba,
    }


# -----------------------------------------------------------------------------
# Run / Parity Comparison
# -----------------------------------------------------------------------------


def compare_subject_runs(
    run_dir: Path | str,
    ref_dir: Path | str,
    protocols: tuple[str, ...] = ("braincolor", "dkt31"),
) -> dict[str, Any]:
    """Compare a pyBiscuit run directory against a reference run directory.

    Compares:
    - Stats tables per protocol.
    - Parcellations (Dice per ROI, overall concordance).
    - Per-vertex scalar measures (mid.H, mid.S, gray.SD, thickness, gray.LGI, mid.SCI).
    """
    run_path = Path(run_dir)
    ref_path = Path(ref_dir)

    result: dict[str, Any] = {
        "subject": run_path.name,
        "run_dir": str(run_path),
        "ref_dir": str(ref_path),
        "protocols": {},
        "scalars": {},
    }

    # 1. Compare Stats tables
    for proto in protocols:
        s_ours = None
        for cand in (
            run_path / "stats" / f"stats.{proto}.csv",
            run_path / "stat" / f"stats.{proto}.csv",
            run_path / f"stats.{proto}.csv",
        ):
            if cand.is_file():
                s_ours = cand
                break

        s_ref = None
        for cand in (
            ref_path / "stats" / f"stats.{proto}.csv",
            ref_path / "stat" / f"stats.{proto}.csv",
            ref_path / f"stats.{proto}.csv",
        ):
            if cand.is_file():
                s_ref = cand
                break

        if s_ours and s_ref:
            t_ours = read_stats_csv(s_ours)
            t_ref = read_stats_csv(s_ref)

            # Compare IDs and averages
            shared_ids = np.intersect1d(t_ours.ids, t_ref.ids)
            measure_diffs: dict[str, dict[str, float]] = {}
            for m_idx, m_name in enumerate(MEASURES):
                ours_vals = []
                ref_vals = []
                for roi_id in shared_ids:
                    idx_o = np.where(t_ours.ids == roi_id)[0][0]
                    idx_r = np.where(t_ref.ids == roi_id)[0][0]
                    ours_vals.append(t_ours.avg[idx_o, m_idx])
                    ref_vals.append(t_ref.avg[idx_r, m_idx])
                comp = compare_scalar_maps(np.array(ours_vals), np.array(ref_vals))
                measure_diffs[m_name] = {
                    "mae": comp.get("mae", float("nan")),
                    "max_abs": comp.get("max_abs", float("nan")),
                    "pearson_r": comp.get("pearson_r", float("nan")),
                }
            result["protocols"][proto] = {
                "shared_rois": len(shared_ids),
                "measure_stats": measure_diffs,
            }

        # Compare ROI label maps if available
        for hemi in ("lh", "rh"):
            p_ours = run_path / "roi" / f"{hemi}.parc.{proto}.txt"
            p_ref = ref_path / "roi" / f"{hemi}.parc.{proto}.txt"
            if p_ours.is_file() and p_ref.is_file():
                lbl_o = np.loadtxt(p_ours, dtype=int)
                lbl_r = np.loadtxt(p_ref, dtype=int)
                dices = roi_dice(lbl_o, lbl_r)
                vals = list(dices.values())
                acc = float(np.mean(lbl_o == lbl_r))
                key = f"{proto}_{hemi}"
                result["protocols"][key] = {
                    "accuracy": acc,
                    "median_dice": float(np.median(vals)) if vals else float("nan"),
                    "min_dice": float(np.min(vals)) if vals else float("nan"),
                    "mean_dice": float(np.mean(vals)) if vals else float("nan"),
                    "dices": dices,
                }

    # 2. Compare scalar maps
    scalar_names = [
        ("lh.mid.H.txt", "quant"),
        ("rh.mid.H.txt", "quant"),
        ("lh.mid.S.txt", "quant"),
        ("rh.mid.S.txt", "quant"),
        ("lh.gray.SD.txt", "quant"),
        ("rh.gray.SD.txt", "quant"),
        ("lh.thickness.txt", "quant"),
        ("rh.thickness.txt", "quant"),
        ("lh.gray.LGI.txt", "quant"),
        ("rh.gray.LGI.txt", "quant"),
        ("lh.mid.SCI.txt", "quant"),
        ("rh.mid.SCI.txt", "quant"),
    ]
    for filename, sub in scalar_names:
        f_ours = run_path / sub / filename
        f_ref = ref_path / sub / filename
        if f_ours.is_file() and f_ref.is_file():
            arr_o = np.loadtxt(f_ours, dtype=np.float64)
            arr_r = np.loadtxt(f_ref, dtype=np.float64)
            result["scalars"][filename] = compare_scalar_maps(arr_o, arr_r)

    return result


# -----------------------------------------------------------------------------
# Cohort Summary Aggregator
# -----------------------------------------------------------------------------


@dataclass
class SubjectSummary:
    subject_id: str
    run_dir: Path
    status: str = "ok"  # ok, incomplete, error
    missing_files: list[str] = field(default_factory=list)
    stats: dict[str, StatsTable] = field(default_factory=dict)


@dataclass
class CohortSummary:
    """Summary of an entire cohort of runs."""

    subjects: list[SubjectSummary] = field(default_factory=list)
    protocols: tuple[str, ...] = ("braincolor", "dkt31")

    @property
    def n_total(self) -> int:
        return len(self.subjects)

    @property
    def n_complete(self) -> int:
        return sum(1 for s in self.subjects if s.status == "ok")

    def aggregate_roi_distributions(self, protocol: str) -> dict[int, dict[str, Any]]:
        """Compute distribution statistics for each ROI in protocol across all subjects."""
        # Map roi_id -> {label, measure -> list of values across subjects}
        roi_data: dict[int, dict[str, Any]] = {}

        for subj in self.subjects:
            if protocol not in subj.stats:
                continue
            table = subj.stats[protocol]
            for i, roi_id in enumerate(table.ids):
                r_id = int(roi_id)
                if r_id not in roi_data:
                    roi_data[r_id] = {
                        "label": table.labels[i],
                        "measures": {m: [] for m in MEASURES},
                    }
                for m_idx, m_name in enumerate(MEASURES):
                    val = table.avg[i, m_idx]
                    if np.isfinite(val):
                        roi_data[r_id]["measures"][m_name].append(val)

        # Compute summary stats
        distributions: dict[int, dict[str, Any]] = {}
        for r_id, d in roi_data.items():
            dist_entry: dict[str, Any] = {
                "id": r_id,
                "label": d["label"],
                "measures": {},
            }
            for m_name, vals in d["measures"].items():
                if vals:
                    arr = np.array(vals, dtype=np.float64)
                    q25, med, q75 = np.percentile(arr, [25, 50, 75])
                    dist_entry["measures"][m_name] = {
                        "n": len(arr),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                        "median": float(med),
                        "mad": float(np.median(np.abs(arr - med))),
                        "iqr": float(q75 - q25),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
            distributions[r_id] = dist_entry

        return distributions

    def detect_outliers(
        self, protocol: str, z_thresh: float = 3.0, robust: bool = True
    ) -> list[dict[str, Any]]:
        """Identify subjects with ROI values beyond z_thresh.

        When robust=True, uses modified z-score: 0.6745 * |x - median| / MAD.
        Otherwise uses standard (x - mean) / std.
        """
        dists = self.aggregate_roi_distributions(protocol)
        outliers: list[dict[str, Any]] = []

        for subj in self.subjects:
            if protocol not in subj.stats:
                continue
            table = subj.stats[protocol]
            for i, roi_id in enumerate(table.ids):
                r_id = int(roi_id)
                if r_id not in dists:
                    continue
                for m_idx, m_name in enumerate(MEASURES):
                    if m_name not in dists[r_id]["measures"]:
                        continue
                    stat = dists[r_id]["measures"][m_name]
                    std = stat["std"]
                    val = table.avg[i, m_idx]
                    mad = stat.get("mad", 0.0)

                    # Standard z
                    z = (val - stat["mean"]) / std if std > 1e-12 else 0.0

                    # Modified robust z (Boris Iglewicz and David Hoaglin 1993)
                    robust_z = 0.6745 * (val - stat["median"]) / mad if mad > 1e-12 else z

                    score = robust_z if robust else z
                    if abs(score) >= z_thresh:
                        outliers.append(
                            {
                                "subject": subj.subject_id,
                                "protocol": protocol,
                                "roi_id": r_id,
                                "roi_label": dists[r_id]["label"],
                                "measure": m_name,
                                "value": float(val),
                                "mean": stat["mean"],
                                "std": std,
                                "median": stat["median"],
                                "z_score": float(z),
                                "robust_z": float(robust_z),
                            }
                        )
        return outliers


def build_cohort_summary(
    run_dirs: list[Path | str],
    protocols: tuple[str, ...] = ("braincolor", "dkt31"),
) -> CohortSummary:
    """Build a CohortSummary by scanning run directories."""
    summary = CohortSummary(protocols=protocols)
    for r in run_dirs:
        p = Path(r)
        subj_id = p.name
        subj = SubjectSummary(subject_id=subj_id, run_dir=p)

        # Check required components
        missing = []
        if not (p / "surf").is_dir():
            missing.append("surf")
        if not (p / "stats").is_dir() and not (p / "stat").is_dir():
            missing.append("stats")

        # Load stats
        for proto in protocols:
            s_file = None
            for cand in (
                p / "stats" / f"stats.{proto}.csv",
                p / "stat" / f"stats.{proto}.csv",
                p / f"stats.{proto}.csv",
            ):
                if cand.is_file():
                    s_file = cand
                    break
            if s_file is not None:
                try:
                    subj.stats[proto] = read_stats_csv(s_file)
                except Exception as e:
                    missing.append(f"stats.{proto}.csv ({e})")
            else:
                missing.append(f"stats.{proto}.csv")

        if missing:
            subj.status = "incomplete"
            subj.missing_files = missing
        else:
            subj.status = "ok"

        summary.subjects.append(subj)

    return summary


# -----------------------------------------------------------------------------
# HTML Report Rendering
# -----------------------------------------------------------------------------


def render_cohort_html(
    cohort: CohortSummary,
    parity_results: list[dict[str, Any]] | None = None,
    title: str = "pyBiscuit Cohort & Parity Report",
) -> str:
    """Generate a clean, self-contained HTML report with cohort mosaic and parity metrics."""
    outliers = []
    for proto in cohort.protocols:
        outliers.extend(cohort.detect_outliers(proto, z_thresh=3.0, robust=False))

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; margin: 30px; background: #fafafa; }",
        "h1, h2, h3 { color: #111; margin-bottom: 8px; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 24px; "
        "background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 4px; }",
        "th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #e1e4e8; "
        "font-size: 14px; }",
        "th { background: #f6f8fa; font-weight: 600; color: #24292e; }",
        "tr:hover { background: #f9f9f9; }",
        ".badge-ok { background: #28a745; color: white; padding: 2px 8px; "
        "border-radius: 12px; font-size: 12px; font-weight: bold; }",
        ".badge-warn { background: #ffc107; color: #212529; padding: 2px 8px; "
        "border-radius: 12px; font-size: 12px; font-weight: bold; }",
        ".card { background: #fff; padding: 20px; border-radius: 6px; "
        "box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p><strong>Cohort Size:</strong> {cohort.n_total} subjects "
        f"({cohort.n_complete} complete, {cohort.n_total - cohort.n_complete} incomplete)</p>",
        "<div class='card'>",
        "<h2>Subjects Overview</h2>",
        "<table>",
        "<tr><th>Subject ID</th><th>Status</th><th>Missing Items</th></tr>",
    ]

    for subj in cohort.subjects:
        badge = (
            "<span class='badge-ok'>OK</span>"
            if subj.status == "ok"
            else "<span class='badge-warn'>Incomplete</span>"
        )
        missing_str = ", ".join(subj.missing_files) if subj.missing_files else "None"
        html_parts.append(
            f"<tr><td><strong>{html.escape(subj.subject_id)}</strong></td>"
            f"<td>{badge}</td><td>{html.escape(missing_str)}</td></tr>"
        )
    html_parts.append("</table></div>")

    # Parity comparison section
    if parity_results:
        html_parts.append("<div class='card'>")
        html_parts.append("<h2>Parity Comparison vs Reference</h2>")
        for res in parity_results:
            html_parts.append(f"<h3>Subject: {html.escape(res['subject'])}</h3>")
            if res.get("protocols"):
                html_parts.append("<h4>Parcellation Agreement (Dice & Accuracy)</h4>")
                html_parts.append(
                    "<table><tr><th>Protocol / Hemisphere</th>"
                    "<th>Accuracy</th><th>Median Dice</th><th>Min Dice</th></tr>"
                )
                for k, v in res["protocols"].items():
                    if "accuracy" in v:
                        html_parts.append(
                            f"<tr><td>{html.escape(k)}</td><td>{v['accuracy']:.4f}</td>"
                            f"<td>{v['median_dice']:.4f}</td><td>{v['min_dice']:.4f}</td></tr>"
                        )
                html_parts.append("</table>")

            if res.get("scalars"):
                html_parts.append("<h4>Per-Vertex Scalar Map Agreement</h4>")
                html_parts.append(
                    "<table><tr><th>Measure File</th><th>Pearson r</th>"
                    "<th>MAE</th><th>Max Abs Diff</th><th>Bland-Altman Bias</th>"
                    "<th>95% LOA</th></tr>"
                )
                for filename, sc in res["scalars"].items():
                    if "pearson_r" in sc:
                        loa_str = f"[{sc['lower_loa']:.4f}, {sc['upper_loa']:.4f}]"
                        html_parts.append(
                            f"<tr><td>{html.escape(filename)}</td><td>{sc['pearson_r']:.6f}</td>"
                            f"<td>{sc['mae']:.5f}</td><td>{sc['max_abs']:.5f}</td>"
                            f"<td>{sc['mean_diff']:.5f}</td><td>{loa_str}</td></tr>"
                        )
                html_parts.append("</table>")
        html_parts.append("</div>")

    # Outlier section
    html_parts.append("<div class='card'>")
    html_parts.append("<h2>Outlier Flags (|z| &ge; 3.0)</h2>")
    if outliers:
        html_parts.append(
            "<table><tr><th>Subject</th><th>Protocol</th><th>ROI ID</th>"
            "<th>ROI Label</th><th>Measure</th><th>Value</th>"
            "<th>Mean &plusmn; Std</th><th>z-Score</th></tr>"
        )
        for out in outliers:
            html_parts.append(
                f"<tr><td>{html.escape(out['subject'])}</td>"
                f"<td>{html.escape(out['protocol'])}</td>"
                f"<td>{out['roi_id']}</td>"
                f"<td>{html.escape(out['roi_label'])}</td>"
                f"<td>{html.escape(out['measure'])}</td>"
                f"<td>{out['value']:.4f}</td>"
                f"<td>{out['mean']:.4f} &plusmn; {out['std']:.4f}</td>"
                f"<td>{out['z_score']:+.2f}</td></tr>"
            )
        html_parts.append("</table>")
    else:
        html_parts.append("<p>No outliers detected across protocols.</p>")
    html_parts.append("</div>")

    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts)


def generate_cohort_report(
    run_dirs: list[Path | str],
    out_dir: Path | str,
    ref_dirs: list[Path | str] | None = None,
    protocols: tuple[str, ...] = ("braincolor", "dkt31"),
) -> Path:
    """Generate cohort summary and parity report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort_summary(run_dirs, protocols=protocols)

    parity_results = []
    if ref_dirs:
        for run_p, ref_p in zip(run_dirs, ref_dirs, strict=False):
            parity_results.append(compare_subject_runs(run_p, ref_p, protocols=protocols))

    report_html = render_cohort_html(cohort, parity_results=parity_results)
    out_file = out_dir / "cohort_report.html"
    out_file.write_text(report_html, encoding="utf-8")

    # Also save structured JSON summary
    summary_data = {
        "n_total": cohort.n_total,
        "n_complete": cohort.n_complete,
        "subjects": [s.subject_id for s in cohort.subjects],
        "parity": parity_results,
    }
    summary_json = json.dumps(summary_data, indent=2)
    (out_dir / "cohort_summary.json").write_text(summary_json, encoding="utf-8")
    return out_file


def cohort_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for pybiscuit cohort."""
    import argparse

    p = argparse.ArgumentParser(
        prog="pybiscuit cohort",
        description="Generate cohort summary and parity report (B12.6).",
    )
    p.add_argument("runs", nargs="+", help="Run directories to aggregate")
    p.add_argument("-o", "--out", "--output", required=True, help="Output directory")
    p.add_argument(
        "--ref-runs",
        nargs="+",
        default=None,
        help="Paired reference run directories for parity comparison",
    )
    p.add_argument(
        "--protocols",
        nargs="+",
        default=["braincolor", "dkt31"],
        help="Protocols to analyze (default: braincolor dkt31)",
    )
    args = p.parse_args(argv)
    generate_cohort_report(
        args.runs,
        args.out,
        ref_dirs=args.ref_runs,
        protocols=tuple(args.protocols),
    )
    return 0
