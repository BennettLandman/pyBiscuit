"""Unit tests for cohort mosaic, Bland-Altman, ICC, Dice, and parity reports (B12.6)."""

from pathlib import Path

import numpy as np

from pybiscuit.cli import main as biscuit_main
from pybiscuit.stats import StatsTable, write_stats_csv
from pybiscuit.viz.cohort import (
    bland_altman,
    build_cohort_summary,
    compare_scalar_maps,
    compute_icc,
    generate_cohort_report,
    roi_dice,
)


def test_bland_altman():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    ba = bland_altman(x, y)
    diff = x - y
    assert np.isclose(ba["mean_diff"], np.mean(diff))
    assert np.isclose(ba["std_diff"], np.std(diff, ddof=1))
    assert np.isclose(ba["lower_loa"], ba["mean_diff"] - 1.96 * ba["std_diff"])
    assert np.isclose(ba["upper_loa"], ba["mean_diff"] + 1.96 * ba["std_diff"])
    assert np.isclose(ba["max_abs_diff"], np.max(np.abs(diff)))


def test_compute_icc():
    # Identical ratings -> ICC = 1.0
    r_perf = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    assert np.isclose(compute_icc(r_perf, "icc31"), 1.0)
    assert np.isclose(compute_icc(r_perf, "icc21"), 1.0)
    assert np.isclose(compute_icc(r_perf, "icc11"), 1.0)

    # Shrout & Fleiss (1979) classic 6 targets x 4 judges dataset
    sf_data = np.array(
        [
            [9.0, 2.0, 5.0, 8.0],
            [6.0, 1.0, 3.0, 2.0],
            [8.0, 4.0, 6.0, 8.0],
            [7.0, 1.0, 2.0, 6.0],
            [10.0, 5.0, 6.0, 9.0],
            [6.0, 2.0, 4.0, 7.0],
        ]
    )
    # Published values: ICC(1,1) ~ 0.166, ICC(2,1) ~ 0.290, ICC(3,1) ~ 0.715
    icc11 = compute_icc(sf_data, "icc11")
    icc21 = compute_icc(sf_data, "icc21")
    icc31 = compute_icc(sf_data, "icc31")

    assert np.isclose(icc11, 0.166, atol=0.01)
    assert np.isclose(icc21, 0.290, atol=0.01)
    assert np.isclose(icc31, 0.715, atol=0.01)


def test_roi_dice():
    l1 = np.array([1, 1, 2, 2, 0, 3])
    l2 = np.array([1, 1, 2, 0, 0, 3])
    d = roi_dice(l1, l2)
    assert d[1] == 1.0
    # ROI 2: l1 has 2, l2 has 1 -> 2 * 1 / (2 + 1) = 2/3
    assert np.isclose(d[2], 2.0 / 3.0)
    assert d[3] == 1.0


def test_compare_scalar_maps():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = compare_scalar_maps(a, b)
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["max_abs"] == 0.0
    assert res["pearson_r"] == 1.0
    assert res["frac_exact_6dp"] == 1.0


def _create_mock_run(run_dir: Path, subject_id: str, offset: float = 0.0) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "surf").mkdir()
    stats_dir = run_dir / "stats"
    stats_dir.mkdir()
    quant_dir = run_dir / "quant"
    quant_dir.mkdir()
    roi_dir = run_dir / "roi"
    roi_dir.mkdir()

    # Create dummy stats table
    n_rois = 5
    ids = np.arange(1, n_rois + 1, dtype=np.int64)
    labels = [f"ROI_{i}" for i in ids]
    nv = np.full(n_rois, 100, dtype=np.int64)
    avg = np.zeros((n_rois, 6), dtype=np.float64)
    stdev = np.ones((n_rois, 6), dtype=np.float64)
    for m in range(6):
        avg[:, m] = (m + 1.0) * 2.0 + offset

    table = StatsTable(
        ids=ids,
        labels=labels,
        n_vertices=nv,
        avg=avg,
        stdev=stdev,
        n_vertices_curve=nv // 2,
        avg_curve=avg * 0.9,
        stdev_curve=stdev,
    )
    write_stats_csv(stats_dir / "stats.braincolor.csv", table)
    write_stats_csv(stats_dir / "stats.dkt31.csv", table)

    # Create dummy scalar file
    scalars = np.linspace(1.0 + offset, 5.0 + offset, 100)
    np.savetxt(quant_dir / "lh.thickness.txt", scalars, fmt="%.6f")

    # Create dummy ROI label file
    labels_arr = np.repeat(np.arange(1, 6), 20)
    np.savetxt(roi_dir / "lh.parc.dkt31.txt", labels_arr, fmt="%d")
    return run_dir


def test_cohort_summary_and_outlier_detection(tmp_path):
    r1 = _create_mock_run(tmp_path / "sub-01", "sub-01", offset=0.0)
    r2 = _create_mock_run(tmp_path / "sub-02", "sub-02", offset=0.1)
    r3 = _create_mock_run(tmp_path / "sub-03", "sub-03", offset=0.05)
    # sub-04 is a large outlier
    r4 = _create_mock_run(tmp_path / "sub-04", "sub-04", offset=15.0)

    summary = build_cohort_summary([r1, r2, r3, r4])
    assert summary.n_total == 4
    assert summary.n_complete == 4

    dists = summary.aggregate_roi_distributions("braincolor")
    assert 1 in dists
    assert "Thickness" in dists[1]["measures"]

    outliers = summary.detect_outliers("braincolor", z_thresh=1.5)
    assert len(outliers) > 0
    assert any(o["subject"] == "sub-04" for o in outliers)


def test_generate_cohort_report_and_cli(tmp_path):
    r1 = _create_mock_run(tmp_path / "sub-01", "sub-01", offset=0.0)
    r2 = _create_mock_run(tmp_path / "sub-02", "sub-02", offset=0.05)
    ref1 = _create_mock_run(tmp_path / "ref-01", "ref-01", offset=0.0)

    out_dir = tmp_path / "report_out"
    report_file = generate_cohort_report(
        [r1, r2], out_dir, ref_dirs=[ref1, ref1], protocols=("braincolor", "dkt31")
    )
    assert report_file.is_file()
    assert (out_dir / "cohort_summary.json").is_file()
    html_content = report_file.read_text()
    assert "sub-01" in html_content
    assert "Parity Comparison vs Reference" in html_content

    # Test CLI via pybiscuit cohort
    cli_out = tmp_path / "cli_out"
    ret = biscuit_main(["cohort", str(r1), str(r2), "-o", str(cli_out)])
    assert ret == 0
    assert (cli_out / "cohort_report.html").is_file()

    # Test CLI via pybiscuit report --cohort
    rep_cli_out = tmp_path / "rep_cli_out"
    ret2 = biscuit_main(["report", "--cohort", str(r1), str(r2), "-o", str(rep_cli_out)])
    assert ret2 == 0
    assert (rep_cli_out / "cohort_report.html").is_file()
