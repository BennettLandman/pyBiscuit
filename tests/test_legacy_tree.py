"""Unit and reference tests for the BISCUIT legacy output tree (`B10.4`, spec §13.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pybiscuit.pipeline.legacy import (
    discover_legacy_files,
    expected_legacy_files,
    legacy_source_candidates,
    populate_legacy_tree,
)

OPEN = Path.home() / "dev" / "biscuit" / "open-examples"


def test_expected_legacy_files_counts_and_irregularities() -> None:
    mc = expected_legacy_files(
        protocols=("braincolor", "dkt31"),
        reg_temp="mb101",
        route="macruise",
        include_merged_pdf=True,
    )
    # surf: 11, reg: 10, roi: 6, curve: 8, quant: 14, stat: 16, pdf: 5, PDF: 1 -> 71
    assert len(mc) == 71
    assert "surf/hemi_map.txt" in mc
    assert "PDF/biscuit.pdf" in mc
    assert "pdf/reg.mb101.pdf" in mc

    # Naming irregularities per spec §13.1
    for h in ("lh", "rh"):
        assert f"surf/{h}.gray.vtk" in mc
        assert f"surf/{h}.pial.vtk" not in mc
        assert f"reg/{h}.gray.reg.vtk" in mc
        assert f"quant/{h}.gray.SD.txt" in mc
        assert f"quant/{h}.gray.LGI.txt" in mc
        assert f"quant/{h}.mid.H.txt" in mc
        assert f"quant/{h}.mid.S.txt" in mc
        assert f"quant/{h}.mid.SCI.txt" in mc
        assert f"quant/{h}.mid.area.txt" in mc
        assert f"quant/{h}.thickness.txt" in mc
        assert f"stat/{h}.gray.reg.LGI.txt" in mc
        assert f"stat/{h}.gray.reg.SD.txt" in mc
        assert f"stat/{h}.mid.reg.H.txt" in mc
        assert f"stat/{h}.mid.reg.S.txt" in mc
        assert f"stat/{h}.mid.reg.SCI.txt" in mc
        assert f"stat/{h}.mid.reg.area.txt" in mc
        assert f"stat/{h}.reg.thickness.txt" in mc
        assert f"stat/{h}.mid.reg.CT.txt" not in mc

    fs_no_pdf = expected_legacy_files(
        protocols="braincolor,dkt31",
        reg_temp="oasis45",
        route="fs",
        include_merged_pdf=False,
    )
    assert len(fs_no_pdf) == 69
    assert "surf/hemi_map.txt" not in fs_no_pdf
    assert "PDF/biscuit.pdf" not in fs_no_pdf
    assert "pdf/reg.oasis45.pdf" in fs_no_pdf


def test_legacy_source_candidates_maps_scratch_names() -> None:
    assert "lh.pial.vtk" in legacy_source_candidates("surf/lh.gray.vtk")
    assert "lh.mid.CT.txt" in legacy_source_candidates("quant/lh.thickness.txt")
    assert "lh.mid.VertArea.txt" in legacy_source_candidates("quant/lh.mid.area.txt")
    assert "lh.mid.SD.txt" in legacy_source_candidates("quant/lh.gray.SD.txt")
    assert "lh.mid.LGI.txt" in legacy_source_candidates("quant/lh.gray.LGI.txt")
    assert "lh.mid.reg.VertArea.txt" in legacy_source_candidates("stat/lh.mid.reg.area.txt")
    assert "lh.mid.reg.LGI.txt" in legacy_source_candidates("stat/lh.gray.reg.LGI.txt")
    assert "lh.mid.reg.SD.txt" in legacy_source_candidates("stat/lh.gray.reg.SD.txt")
    assert "lh.mid.reg.CT.txt" in legacy_source_candidates("stat/lh.reg.thickness.txt")


def test_populate_and_discover_legacy_tree_from_scratch_names(tmp_path: Path) -> None:
    """Populate a full MaCRUISE legacy tree from scratch intermediate names (tmp1..tmp4)."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    # Write every file using its BISCUIT scratch intermediate name where irregular
    scratch_files = [
        "hemi_map.txt",
        "reg.mb101.pdf",
        "parc.braincolor.pdf",
        "parc.dkt31.pdf",
        "curve.pdf",
        "quant.pdf",
        "biscuit.pdf",
        "stats.braincolor.csv",
        "stats.dkt31.csv",
    ]
    for h in ("lh", "rh"):
        scratch_files.extend(
            [
                f"{h}.white.vtk",
                f"{h}.pial.vtk",  # -> surf/<h>.gray.vtk
                f"{h}.mid.vtk",
                f"{h}.inflated.vtk",
                f"{h}.sphere.vtk",
                f"{h}.white.reg.vtk",
                f"{h}.pial.reg.vtk",  # -> reg/<h>.gray.reg.vtk
                f"{h}.mid.reg.vtk",
                f"{h}.sphere.reg.vtk",
                f"{h}.coeff.txt",
                f"{h}.mask.txt",
                f"{h}.parc.braincolor.txt",
                f"{h}.parc.dkt31.txt",
                f"{h}.gray.scurve",
                f"{h}.gray.scurve.bary",
                f"{h}.white.gcurve",
                f"{h}.white.gcurve.bary",
                f"{h}.mid.CT.txt",  # -> quant/<h>.thickness.txt
                f"{h}.mid.VertArea.txt",  # -> quant/<h>.mid.area.txt
                f"{h}.mid.H.txt",
                f"{h}.mid.S.txt",
                f"{h}.mid.SD.txt",  # -> quant/<h>.gray.SD.txt
                f"{h}.mid.LGI.txt",  # -> quant/<h>.gray.LGI.txt
                f"{h}.mid.SCI.txt",
                f"{h}.mid.reg.VertArea.txt",  # -> stat/<h>.mid.reg.area.txt
                f"{h}.mid.reg.H.txt",
                f"{h}.mid.reg.S.txt",
                f"{h}.mid.reg.SCI.txt",
                f"{h}.mid.reg.LGI.txt",  # -> stat/<h>.gray.reg.LGI.txt
                f"{h}.mid.reg.SD.txt",  # -> stat/<h>.gray.reg.SD.txt
                f"{h}.mid.reg.CT.txt",  # -> stat/<h>.reg.thickness.txt
            ]
        )
    for name in scratch_files:
        (scratch / name).write_text(f"payload:{name}\n")

    legacy_dir = tmp_path / "legacy"
    populated = populate_legacy_tree(
        legacy_dir,
        scratch,
        protocols=("braincolor", "dkt31"),
        reg_temp="mb101",
        route="macruise",
        include_merged_pdf=True,
    )
    expected = expected_legacy_files(
        protocols=("braincolor", "dkt31"),
        reg_temp="mb101",
        route="macruise",
        include_merged_pdf=True,
    )
    assert set(populated) == expected
    discovered = discover_legacy_files(legacy_dir)
    assert discovered ^ expected == set()

    # Check content of renamed irregular files
    assert (legacy_dir / "surf/lh.gray.vtk").read_text() == "payload:lh.pial.vtk\n"
    assert (legacy_dir / "quant/lh.thickness.txt").read_text() == "payload:lh.mid.CT.txt\n"
    assert (legacy_dir / "stat/lh.reg.thickness.txt").read_text() == "payload:lh.mid.reg.CT.txt\n"
    assert (legacy_dir / "stat/lh.gray.reg.LGI.txt").read_text() == "payload:lh.mid.reg.LGI.txt\n"


def test_populate_legacy_tree_missing_raises_unless_partial(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="missing 71 expected legacy file"):
        populate_legacy_tree(out, empty)
    assert populate_legacy_tree(out, empty, allow_partial=True) == {}


@pytest.mark.reference
@pytest.mark.skipif(not OPEN.exists(), reason="open-examples not available")
@pytest.mark.parametrize("subject", ["0050952", "0050961", "0050971"])
def test_open_examples_matches_expected_legacy_tree(subject: str) -> None:
    """Verify zero file-set difference between expected_legacy_files and open-examples."""
    expected = expected_legacy_files(
        protocols=("braincolor", "dkt31"),
        reg_temp="mb101",
        route="fs",
        include_merged_pdf=False,
    )
    actual = discover_legacy_files(OPEN / subject)
    assert actual ^ expected == set()
