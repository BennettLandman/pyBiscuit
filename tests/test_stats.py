"""Unit and reference tests for BISCUIT Phase 4 summary statistics (`B10.1`, spec §10)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pybiscuit.stats import (
    AGGREGATE_CORTEX_ID,
    AGGREGATE_HEMI_IDS,
    AGGREGATE_LOBE_IDS,
    CSV_COLUMNS,
    assign_curve_parcels,
    bary_interpolate,
    compute_roi_and_curve_stats,
    read_bary,
    read_protocol_labels,
    read_stats_csv,
    remesh_stat_maps,
    run_stats_pipeline,
    write_stats_csv,
)
from pybiscuit.stats.cli import main as stats_cli_main

OPEN = Path.home() / "dev" / "biscuit" / "open-examples"
OPT = Path.home() / "dev" / "biscuit" / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit"


def test_read_bary_deduplicates_junctions_and_sorts_lexicographically(tmp_path: Path) -> None:
    bary_txt = (
        "2\n"
        "\n"
        "3\n"
        "10 20 0.750000\n"
        "2 5 0.500000\n"
        "4 8 0.250000\n"
        "\n"
        "2\n"
        "2 5 0.500000\n"
        "1 3 0.125000\n"
    )
    p = tmp_path / "test.bary"
    p.write_text(bary_txt)

    v1_raw, v2_raw, w_raw = read_bary(p, unique=False)
    assert len(v1_raw) == 5

    v1, v2, w = read_bary(p, unique=True)
    assert v1.dtype == np.int64
    assert v2.dtype == np.int64
    assert w.dtype == np.float64
    # Lexicographic sort on (v1, v2, w) with duplicate (2, 5, 0.5) removed
    assert v1.tolist() == [1, 2, 4, 10]
    assert v2.tolist() == [3, 5, 8, 20]
    assert np.allclose(w, [0.125, 0.5, 0.25, 0.75])


def test_bary_interpolate_and_threshold_rule() -> None:
    parc = np.array([101, 102, 103, 104], dtype=np.int64)
    v1 = np.array([0, 0, 2], dtype=np.int64)
    v2 = np.array([1, 1, 3], dtype=np.int64)
    # w == 0.5 must pick v1 (parc[v1]); w < 0.5 must pick v2 (parc[v2])
    w = np.array([0.5, 0.4999999, 0.8], dtype=np.float64)
    pc = assign_curve_parcels(parc, v1, v2, w)
    assert pc.tolist() == [101, 102, 103]

    pmap = np.array([[10.0, 1.0], [20.0, 5.0], [30.0, 2.0], [40.0, 6.0]], dtype=np.float64)
    interp = bary_interpolate(pmap, v1, v2, w)
    assert np.allclose(interp[0], [15.0, 3.0])
    assert np.allclose(interp[2], [30.0 * 0.8 + 40.0 * 0.2, 2.0 * 0.8 + 6.0 * 0.2])


def test_hand_computed_roi_curve_and_lobe_aggregates(tmp_path: Path) -> None:
    """Verify N-1 std, empty ROI -> NaN, 1-vertex ROI -> std=NaN, unknown labels -> 0, and lobes."""
    # 3-column protocol (like braincolor):
    # ROI 101: lobe 1 (frontal), has 2 vertices in lh and 1 in rh -> total 3 vertices
    # ROI 102: lobe 2 (temporal), has 1 vertex in rh -> total 1 vertex (std = NaN)
    # ROI 103: lobe 3 (parietal), has 0 vertices -> empty ROI (mean = NaN, std = NaN)
    label_csv = tmp_path / "label.csv"
    label_csv.write_text(
        "Label,ID,Lobe\n"
        "Left Frontal ROI,101,1\n"
        "Right Temporal ROI,102,2\n"
        "Empty Parietal ROI,103,3\n"
    )

    # lh: 4 vertices with labels [101, 101, 0, 9999 (unlisted -> mapped to 0)]
    parc_lh = np.array([101, 101, 0, 9999], dtype=np.int64)
    # rh: 3 vertices with labels [101, 102, 0]
    parc_rh = np.array([101, 102, 0], dtype=np.int64)

    # 6 measures: [H, S, Depth, Thickness, SCI, LGI]
    # Make each measure easy to verify by hand: column m = base * (m + 1)
    base_lh = np.array([1.0, 2.0, 100.0, 200.0])[:, None]
    base_rh = np.array([4.0, 10.0, 300.0])[:, None]
    scales = np.arange(1, 7, dtype=np.float64)[None, :]
    props_lh = base_lh * scales
    props_rh = base_rh * scales

    # Curves:
    # lh has duplicate junction row (0, 1, 0.5) plus (2, 3, 0.75) [both unknown -> parcel 0]
    curve_lh = np.array(
        [
            [0, 1, 0.5],
            [0, 1, 0.5],  # duplicate junction row -> deduplicated
            [2, 3, 0.75],  # unknown parcel -> excluded from 20000..20012
        ],
        dtype=np.float64,
    )
    # rh has (0, 1, 0.5) -> w >= 0.5 picks v1=0 (label 101), value = 0.5*4 + 0.5*10 = 7.0
    # and (0, 1, 0.25) -> w < 0.5 picks v2=1 (label 102), value = 0.25*4 + 0.75*10 = 8.5
    curve_rh = np.array(
        [
            [0, 1, 0.5],
            [0, 1, 0.25],
        ],
        dtype=np.float64,
    )

    tbl = compute_roi_and_curve_stats(
        props_lh, props_rh, parc_lh, parc_rh, curve_lh, curve_rh, label_csv
    )

    # ID 0 (Unknown) must be dropped; expected IDs: 101, 102, 103, 20000, 20001, 20002, 20003..20012
    expected_ids = [101, 102, 103, 20000, 20001, 20002, *AGGREGATE_LOBE_IDS]
    assert tbl.ids.tolist() == expected_ids

    # Check ROI 101: vertices have base values [1.0, 2.0, 4.0] -> mean=7/3, std(ddof=1)=sqrt(7/3)
    r101 = tbl.row(101)
    assert r101["nVertices"] == 3
    assert np.isclose(r101["MeanCurvature"], 7.0 / 3.0)
    assert np.isclose(r101["MeanCurvature_stdev"], np.std([1.0, 2.0, 4.0], ddof=1))
    # Curve points for 101: lh (0,1,0.5) -> base 1.5; rh (0,1,0.5) -> base 7.0 -> 2 points
    assert r101["nVertices_curve"] == 2
    assert np.isclose(r101["MeanCurvature_curve"], (1.5 + 7.0) / 2.0)
    assert np.isclose(r101["MeanCurvature_stdev_curve"], np.std([1.5, 7.0], ddof=1))

    # Check ROI 102: single vertex (rh idx 1, base=10.0) -> mean=10*scale, std=NaN
    r102 = tbl.row(102)
    assert r102["nVertices"] == 1
    assert np.isclose(r102["MeanCurvature"], 10.0)
    assert np.isnan(r102["MeanCurvature_stdev"])
    # Single curve point for 102 (rh (0,1,0.25), base=8.5) -> mean=8.5, std=NaN
    assert r102["nVertices_curve"] == 1
    assert np.isclose(r102["MeanCurvature_curve"], 8.5)
    assert np.isnan(r102["MeanCurvature_stdev_curve"])

    # Check ROI 103: empty ROI -> nVertices=0, mean=NaN, std=NaN
    r103 = tbl.row(103)
    assert r103["nVertices"] == 0
    assert np.isnan(r103["MeanCurvature"])
    assert np.isnan(r103["MeanCurvature_stdev"])
    assert r103["nVertices_curve"] == 0
    assert np.isnan(r103["MeanCurvature_curve"])
    assert np.isnan(r103["MeanCurvature_stdev_curve"])

    # Check Whole cortex (20000): excludes unknown/unlisted vertices (100.0, 200.0, 300.0)
    # Valid bases: [1.0, 2.0, 4.0, 10.0] -> 4 vertices, 3 curve points [1.5, 7.0, 8.5]
    r20000 = tbl.row(AGGREGATE_CORTEX_ID)
    assert r20000["nVertices"] == 4
    assert np.isclose(r20000["MeanCurvature"], np.mean([1.0, 2.0, 4.0, 10.0]))
    assert np.isclose(r20000["MeanCurvature_stdev"], np.std([1.0, 2.0, 4.0, 10.0], ddof=1))
    assert r20000["nVertices_curve"] == 3
    assert np.isclose(r20000["MeanCurvature_curve"], np.mean([1.5, 7.0, 8.5]))

    # Check Left hemisphere (20001): [1.0, 2.0] and curve [1.5]
    r20001 = tbl.row(20001)
    assert r20001["nVertices"] == 2
    assert np.isclose(r20001["MeanCurvature"], 1.5)
    assert r20001["nVertices_curve"] == 1
    assert np.isnan(r20001["MeanCurvature_stdev_curve"])

    # Check Right hemisphere (20002): [4.0, 10.0] and curve [7.0, 8.5]
    r20002 = tbl.row(20002)
    assert r20002["nVertices"] == 2
    assert np.isclose(r20002["MeanCurvature"], 7.0)
    assert r20002["nVertices_curve"] == 2

    # Check Lobes:
    # 20003 = Left frontal lobe (lobe 1 in lh -> [1.0, 2.0])
    # 20004 = Right frontal lobe (lobe 1 in rh -> [4.0], rh vertex 0 has label 101 = lobe 1)
    # 20006 = Right temporal lobe (lobe 2 in rh -> [10.0])
    assert tbl.row(20003)["nVertices"] == 2
    assert np.isclose(tbl.row(20003)["MeanCurvature"], 1.5)
    assert tbl.row(20004)["nVertices"] == 1
    assert np.isclose(tbl.row(20004)["MeanCurvature"], 4.0)
    assert tbl.row(20005)["nVertices"] == 0
    assert tbl.row(20006)["nVertices"] == 1
    assert np.isclose(tbl.row(20006)["MeanCurvature"], 10.0)
    for empty_lobe in (20005, 20007, 20008, 20009, 20010, 20011, 20012):
        assert tbl.row(empty_lobe)["nVertices"] == 0
        assert np.isnan(tbl.row(empty_lobe)["MeanCurvature"])

    # CSV round-trip
    out_csv = tmp_path / "stats.test.csv"
    write_stats_csv(out_csv, tbl)
    rt = read_stats_csv(out_csv)
    assert rt.ids.tolist() == tbl.ids.tolist()
    assert rt.labels == tbl.labels
    assert np.allclose(rt.avg, tbl.avg, equal_nan=True)
    assert np.allclose(rt.stdev, tbl.stdev, equal_nan=True)
    assert np.allclose(rt.avg_curve, tbl.avg_curve, equal_nan=True)
    assert np.allclose(rt.stdev_curve, tbl.stdev_curve, equal_nan=True)


def test_two_column_protocol_omits_lobe_aggregates(tmp_path: Path) -> None:
    """Protocols without a 3rd lobe column (like dkt31) stop at 20002 (no 20003..20012)."""
    label_csv = tmp_path / "label.csv"
    label_csv.write_text("label,ID\nLeft cuneus,1005\nRight cuneus,2005\n")
    rows, has_lobe = read_protocol_labels(label_csv)
    assert not has_lobe
    assert len(rows) == 3 and rows[0].id == 0

    props_lh = np.ones((2, 6), dtype=np.float64)
    props_rh = np.ones((2, 6), dtype=np.float64) * 2.0
    parc_lh = np.array([1005, 1005], dtype=np.int64)
    parc_rh = np.array([2005, 0], dtype=np.int64)
    curve_lh = (np.array([0]), np.array([1]), np.array([0.5]))
    curve_rh = (np.array([0]), np.array([1]), np.array([0.75]))

    tbl = compute_roi_and_curve_stats(
        props_lh, props_rh, parc_lh, parc_rh, curve_lh, curve_rh, label_csv
    )
    assert tbl.ids.tolist() == [1005, 2005, 20000, *AGGREGATE_HEMI_IDS]
    for lobe_id in AGGREGATE_LOBE_IDS:
        assert lobe_id not in tbl.ids


def test_remesh_stat_maps_and_cli_synthetic(tmp_path: Path) -> None:
    """Test remesh_stat_maps and stats CLI on a tiny synthetic icosphere subject."""
    from pybiscuit.io import write_vtk
    from pybiscuit.sphere.icosphere import icosphere

    ico = icosphere(1)  # 42 vertices
    nv = ico.n_vertices
    subj = tmp_path / "subj"
    for d in ("quant", "roi", "curve", "reg", "stat"):
        (subj / d).mkdir(parents=True)
    uni = tmp_path / "sphere_uni.vtk"
    write_vtk(ico, uni)

    for h in ("lh", "rh"):
        write_vtk(ico, subj / "reg" / f"{h}.sphere.reg.vtk")
        for m_idx, fname in enumerate(
            ("mid.H", "mid.S", "gray.SD", "thickness", "mid.SCI", "gray.LGI")
        ):
            vals = np.linspace(1.0 + m_idx, 2.0 + m_idx, nv)
            (subj / "quant" / f"{h}.{fname}.txt").write_text(
                "".join(f"{v:f}\n" for v in vals)
            )
        parc = np.where(np.arange(nv) < nv // 2, 1005, 2005)
        (subj / "roi" / f"{h}.parc.dkt31.txt").write_text(
            "".join(f"{p}\n" for p in parc)
        )
        (subj / "curve" / f"{h}.gray.scurve.bary").write_text(
            "1\n2\n0 1 0.600000\n2 3 0.400000\n"
        )

    parc_root = tmp_path / "parc"
    (parc_root / "dkt31").mkdir(parents=True)
    (parc_root / "dkt31" / "label.csv").write_text(
        "label,ID\nLeft cuneus,1005\nRight cuneus,2005\n"
    )

    rc = stats_cli_main(
        [
            "--subject-dir",
            str(subj),
            "--parc-root",
            str(parc_root),
            "--uniform-sphere",
            str(uni),
            "--protocols",
            "dkt31",
        ]
    )
    assert rc == 0
    assert (subj / "stat" / "stats.dkt31.csv").is_file()
    for h in ("lh", "rh"):
        for stem in (
            "mid.reg.H",
            "mid.reg.S",
            "mid.reg.SCI",
            "gray.reg.LGI",
            "gray.reg.SD",
            "reg.thickness",
        ):
            p = subj / "stat" / f"{h}.{stem}.txt"
            assert p.is_file()
            arr = np.loadtxt(p)
            assert arr.shape == (nv,)

    # Also check legacy_names=False (scratch tmp4 naming)
    tmp4 = tmp_path / "tmp4"
    remesh_stat_maps(
        subj / "quant",
        subj / "reg",
        tmp4,
        uniform_sphere=uni,
        legacy_names=False,
    )
    for h in ("lh", "rh"):
        for m in ("H", "S", "SD", "CT", "LGI", "SCI"):
            assert (tmp4 / f"{h}.mid.reg.{m}.txt").is_file()


@pytest.mark.reference
@pytest.mark.skipif(
    not (OPEN.exists() and OPT.exists()),
    reason="open-examples or reference/ not available",
)
@pytest.mark.parametrize("subject", ["0050952", "0050961", "0050971"])
@pytest.mark.parametrize("protocol", ["braincolor", "dkt31"])
def test_stats_csv_matches_open_examples_image(
    tmp_path: Path, subject: str, protocol: str
) -> None:
    """Compare computed stats CSV against all 6 real biscuit_FC_v2.2 CSVs in open-examples."""
    sdir = OPEN / subject
    tables = run_stats_pipeline(
        sdir,
        stat_dir=tmp_path,
        parc_root=OPT / "parc",
        protocols=(protocol,),
        remesh_maps=False,
    )
    tbl = tables[protocol]
    ours_csv = tmp_path / f"stats.{protocol}.csv"
    ref_csv = sdir / "stat" / f"stats.{protocol}.csv"

    with open(ours_csv, newline="") as fh:
        ours_rows = list(csv.reader(fh))
    with open(ref_csv, newline="") as fh:
        ref_rows = list(csv.reader(fh))

    assert ours_rows[0] == list(CSV_COLUMNS)
    assert ref_rows[0] == list(CSV_COLUMNS)
    assert len(ours_rows) == len(ref_rows) == (112 if protocol == "braincolor" else 66)

    max_rel = 0.0
    max_abs = 0.0
    exact_cells = 0
    total_float_cells = 0
    roi_exact_cells = 0
    roi_total_cells = 0

    float_cols = list(range(3, 15)) + list(range(16, 28))
    for r_ours, r_ref in zip(ours_rows[1:], ref_rows[1:], strict=True):
        rid = int(r_ref[0])
        assert int(r_ours[0]) == rid
        assert r_ours[1] == r_ref[1]
        assert int(r_ours[2]) == int(r_ref[2])
        assert int(r_ours[15]) == int(r_ref[15])
        is_individual_roi = rid < 20000
        for c in float_cols:
            s_ours, s_ref = r_ours[c], r_ref[c]
            v_ours, v_ref = float(s_ours), float(s_ref)
            total_float_cells += 1
            if is_individual_roi:
                roi_total_cells += 1
            if s_ours == s_ref:
                exact_cells += 1
                if is_individual_roi:
                    roi_exact_cells += 1
            diff = abs(v_ours - v_ref)
            rel = diff / max(abs(v_ref), 1e-30)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, rel)

    assert len(tbl) == len(ref_rows) - 1
    assert max_rel <= 1e-12, f"{subject} {protocol}: max_rel={max_rel}"
    # Every individual ROI cell matches string-for-string at %.15g, and >= 98% overall
    assert roi_exact_cells == roi_total_cells
    assert exact_cells / total_float_cells >= 0.98


@pytest.mark.reference
@pytest.mark.skipif(
    not (OPEN.exists() and OPT.exists()),
    reason="open-examples or reference/ not available",
)
def test_remesh_stat_maps_matches_open_examples_bytes(tmp_path: Path) -> None:
    """Verify remesh_stat_maps reproduces all 12 stat/*.reg.*.txt files in 0050952 byte-for-byte."""
    sdir = OPEN / "0050952"
    uni = OPT / "sphere" / "sphere_327680.vtk"
    written = remesh_stat_maps(
        sdir / "quant",
        sdir / "reg",
        tmp_path,
        uniform_sphere=uni,
        legacy_names=True,
    )
    assert set(written) == {"lh", "rh"}
    for h in ("lh", "rh"):
        for dst in written[h].values():
            ref = sdir / "stat" / dst.name
            assert dst.read_bytes() == ref.read_bytes(), dst.name
