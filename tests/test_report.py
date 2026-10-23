"""Tests for QA PDF report generator (B12.3, spec §11)."""

from __future__ import annotations

import re
import zlib

import numpy as np

from pybiscuit.viz.pdf import PdfDocument
from pybiscuit.viz.report import (
    CURVE_HEIGHT,
    PAGE_WIDTH,
    PARC_HEIGHT,
    QUANT_HEIGHT,
    REG_HEIGHT,
    generate_report_pdfs,
    report_main,
)


def _extract_mediaboxes(pdf_data: bytes) -> list[tuple[float, float]]:
    boxes = []
    mb_pat = rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]"
    for m in re.finditer(mb_pat, pdf_data):
        w = float(m.group(3))
        h = float(m.group(4))
        boxes.append((w, h))
    return boxes


def _extract_decompressed_streams(pdf_data: bytes) -> list[str]:
    texts = []
    for m in re.finditer(rb"<<(.*?)>>\s*stream[\r\n]+(.*?)[\r\n]+endstream", pdf_data, re.DOTALL):
        dict_part = m.group(1)
        raw = m.group(2)
        if b"/FlateDecode" in dict_part:
            try:
                dec = zlib.decompress(raw)
                texts.append(dec.decode("latin1", errors="ignore"))
            except Exception:
                pass
        else:
            texts.append(raw.decode("latin1", errors="ignore"))
    return texts


def test_pdf_document_primitives(tmp_path):
    doc = PdfDocument()
    page = doc.add_page(612.0, 756.0)

    # Draw graphics
    page.draw_rect(7.2, 7.2, 597.6, 741.6, stroke=True, fill=False)
    page.draw_line(10.0, 10.0, 100.0, 100.0)

    # Draw text with links
    page.draw_text("Test Title", 306.0, 700.0, font="Helvetica-Bold", size=18.0, align="center")
    page.draw_text("Link Text", 50.0, 650.0, link="https://example.com/test")

    # Draw image
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 255
    page.draw_image(img, 50.0, 400.0, 100.0, 100.0)

    out_pdf = tmp_path / "primitive.pdf"
    doc.write(out_pdf)

    assert out_pdf.exists()
    data = out_pdf.read_bytes()
    assert data.startswith(b"%PDF-1.4")
    assert b"/MediaBox [0 0 612.00 756.00]" in data
    assert b"https://example.com/test" in data
    assert b"/XObject" in data

    # Verify decompressed content contains text
    streams = _extract_decompressed_streams(data)
    combined = " ".join(streams)
    assert "Test Title" in combined
    assert "Link Text" in combined


def test_generate_report_pdfs_geometry_and_pages(tmp_path):
    tile = np.full((100, 100, 3), 220, dtype=np.uint8)
    tiles = {
        "sph_l": tile,
        "reg_l": tile,
        "tmp_l": tile,
        "sph_r": tile,
        "reg_r": tile,
        "tmp_r": tile,
        "wm": tile,
        "gm": tile,
        "mid": tile,
        "inf": tile,
        "sulcus_mid": tile,
        "sulcus_inf": tile,
        "gyrus_mid": tile,
        "gyrus_inf": tile,
        "H": tile,
        "S": tile,
        "CT": tile,
        "SD": tile,
        "LGI": tile,
        "SCI": tile,
    }

    files = generate_report_pdfs(
        tmp_path,
        template="oasis45",
        protocols=["braincolor", "dkt31"],
        project="TestProject",
        subject="TestSubj",
        session="TestSess",
        version="2.2 (CBT_v1.3)",
        tiles=tiles,
    )

    # 1. reg.oasis45.pdf: 8.5 x 10.5 in = 612 x 756 pt
    reg_pdf = files["reg"]
    assert reg_pdf.exists()
    boxes = _extract_mediaboxes(reg_pdf.read_bytes())
    assert len(boxes) == 1
    assert np.isclose(boxes[0][0], PAGE_WIDTH)
    assert np.isclose(boxes[0][1], REG_HEIGHT)
    reg_text = " ".join(_extract_decompressed_streams(reg_pdf.read_bytes()))
    assert "HSD: Cortical Surface Registration" in reg_text

    # 2. parc.braincolor.pdf: 8.5 x 8.7 in = 612 x 626.4 pt
    parc_bc_pdf = files["parc_braincolor"]
    assert parc_bc_pdf.exists()
    boxes = _extract_mediaboxes(parc_bc_pdf.read_bytes())
    assert len(boxes) == 1
    assert np.isclose(boxes[0][0], PAGE_WIDTH)
    assert np.isclose(boxes[0][1], PARC_HEIGHT)
    parc_text = " ".join(_extract_decompressed_streams(parc_bc_pdf.read_bytes()))
    assert "Cortical Surface Parcellation" in parc_text

    # 3. parc.dkt31.pdf: 8.5 x 8.7 in = 612 x 626.4 pt
    parc_dkt_pdf = files["parc_dkt31"]
    assert parc_dkt_pdf.exists()
    boxes = _extract_mediaboxes(parc_dkt_pdf.read_bytes())
    assert len(boxes) == 1
    assert np.isclose(boxes[0][0], PAGE_WIDTH)
    assert np.isclose(boxes[0][1], PARC_HEIGHT)

    # 4. curve.pdf: 8.5 x 9.2 in = 612 x 662.4 pt
    curve_pdf = files["curve"]
    assert curve_pdf.exists()
    boxes = _extract_mediaboxes(curve_pdf.read_bytes())
    assert len(boxes) == 1
    assert np.isclose(boxes[0][0], PAGE_WIDTH)
    assert np.isclose(boxes[0][1], CURVE_HEIGHT)
    curve_text = " ".join(_extract_decompressed_streams(curve_pdf.read_bytes()))
    assert "TRACE: Curve Extraction" in curve_text

    # 5. quant.pdf: 8.5 x 11.7 in = 612 x 842.4 pt
    quant_pdf = files["quant"]
    assert quant_pdf.exists()
    boxes = _extract_mediaboxes(quant_pdf.read_bytes())
    assert len(boxes) == 1
    assert np.isclose(boxes[0][0], PAGE_WIDTH)
    assert np.isclose(boxes[0][1], QUANT_HEIGHT)
    quant_text = " ".join(_extract_decompressed_streams(quant_pdf.read_bytes()))
    assert "Cortical Surface Quantification" in quant_text

    # 6. Merged biscuit.pdf in PDF/ directory
    biscuit_pdf = files["biscuit"]
    assert biscuit_pdf.exists()
    assert biscuit_pdf.parent.name == "PDF"
    boxes = _extract_mediaboxes(biscuit_pdf.read_bytes())
    # 5 pages: reg, parc_braincolor, parc_dkt31, curve, quant
    assert len(boxes) == 5
    expected_heights = [REG_HEIGHT, PARC_HEIGHT, PARC_HEIGHT, CURVE_HEIGHT, QUANT_HEIGHT]
    for i, h in enumerate(expected_heights):
        assert np.isclose(boxes[i][0], PAGE_WIDTH)
        assert np.isclose(boxes[i][1], h)


def test_report_cli(tmp_path):
    out_dir = tmp_path / "cli_run"
    argv = [
        "-o",
        str(out_dir),
        "--template",
        "mb101",
        "--protocols",
        "braincolor",
        "--project",
        "MyProject",
        "--subject",
        "sub-100",
    ]
    ret = report_main(argv)
    assert ret == 0

    assert (out_dir / "pdf" / "reg.mb101.pdf").exists()
    assert (out_dir / "pdf" / "parc.braincolor.pdf").exists()
    assert (out_dir / "pdf" / "curve.pdf").exists()
    assert (out_dir / "pdf" / "quant.pdf").exists()
    biscuit_bytes = (out_dir / "PDF" / "biscuit.pdf").read_bytes()
    biscuit_streams = " ".join(_extract_decompressed_streams(biscuit_bytes))
    assert "MyProject" in biscuit_streams
    assert "sub-100" in biscuit_streams
