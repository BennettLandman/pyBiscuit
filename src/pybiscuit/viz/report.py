"""PDF QA report generator for BISCUIT outputs (B12.3, spec §11).

Generates the QA PDF report pages matching the geometry, headers, tables,
and layout of the original BISCUIT pipeline without LaTeX or MATLAB:
- ``reg.<template>.pdf``: 8.5 x 10.5 in (612 x 756 pt), 3 x 2 tiles (HSD registration)
- ``parc.<protocol>.pdf``: 8.7 in (612 x 626.4 pt), 2 x 2 tiles (Parcellation)
- ``curve.pdf``: 9.2 in (612 x 662.4 pt), 2 x 2 tiles (Sulcal and gyral curves)
- ``quant.pdf``: 11.7 in (612 x 842.4 pt), 2 x 3 tiles (6 cortical measures)
- Merged ``PDF/biscuit.pdf``: concatenated in order with per-page MediaBox preserved.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pdf import PdfDocument, PdfPage

# Page dimensions in points (1 in = 72 pt)
PAGE_WIDTH = 612.0  # 8.5 in
MARGIN = 7.2  # 0.1 in

REG_HEIGHT = 756.0  # 10.5 in
PARC_HEIGHT = 626.4  # 8.7 in
CURVE_HEIGHT = 662.4  # 9.2 in
QUANT_HEIGHT = 842.4  # 11.7 in

FOOTER_TEXT = (
    "BISCUIT: Brain Shape Computing Toolbox -- An Automated Pipeline for Cortical Morphometry"
)


def _draw_header_and_frame(
    page: PdfPage,
    title: str,
    *,
    project: str = "Project",
    subject: str = "Subject",
    session: str = "Session",
    version: str = "2.2 (CBT_v1.3)",
    date_str: str | None = None,
    ref_links: Sequence[tuple[str, str]] = (),
    software_links: Sequence[tuple[str, str]] = (),
) -> None:
    """Draw geometry frame, title, metadata table, and footer."""
    if date_str is None:
        date_str = datetime.date.today().strftime("%B %d, %Y")

    # Geometry frame around 0.1 in margin
    page.draw_rect(
        MARGIN,
        MARGIN,
        page.width - 2 * MARGIN,
        page.height - 2 * MARGIN,
        stroke=True,
        fill=False,
        stroke_color=(0.1, 0.1, 0.1),
        line_width=0.5,
    )

    # Title
    page.draw_text(
        title,
        page.width / 2.0,
        page.height - 40.0,
        font="Helvetica-Bold",
        size=18.0,
        align="center",
    )

    # Metadata table (Left: Project, Subject, Session, Version, Date)
    y_top = page.height - 62.0
    line_h = 13.0
    x_left = 30.0

    page.draw_text(f"Project: {project}", x_left, y_top, font="Helvetica-Bold", size=9.5)
    page.draw_text(f"Subject: {subject}", x_left, y_top - line_h, font="Helvetica-Bold", size=9.5)
    page.draw_text(
        f"Session: {session}", x_left, y_top - 2 * line_h, font="Helvetica-Bold", size=9.5
    )
    page.draw_text(
        f"Version: {version}", x_left, y_top - 3 * line_h, font="Helvetica-Bold", size=9.5
    )
    page.draw_text(date_str, x_left, y_top - 4 * line_h, font="Helvetica-Bold", size=9.5)

    # Metadata table (Right: References, Software, Contact)
    x_right = page.width - 30.0
    page.draw_text("References", x_right, y_top, font="Helvetica-Bold", size=9.5, align="right")

    # Ref links
    ref_str = "Paper" if not ref_links else ", ".join(lbl for lbl, _ in ref_links)
    first_url = ref_links[0][1] if ref_links else None
    page.draw_text(
        ref_str,
        x_right,
        y_top - line_h,
        font="Helvetica",
        size=9.0,
        color=(0.0, 0.2, 0.7),
        align="right",
        link=first_url,
    )

    # Software links
    sw_str = "Software" if not software_links else ", ".join(lbl for lbl, _ in software_links)
    first_sw_url = software_links[0][1] if software_links else None
    page.draw_text(
        sw_str,
        x_right,
        y_top - 2 * line_h,
        font="Helvetica",
        size=9.0,
        color=(0.0, 0.2, 0.7),
        align="right",
        link=first_sw_url,
    )

    page.draw_text(
        "Contact Information",
        x_right,
        y_top - 3 * line_h,
        font="Helvetica-Bold",
        size=9.5,
        align="right",
    )
    page.draw_text(
        "Ilwoo Lyu",
        x_right,
        y_top - 4 * line_h,
        font="Helvetica",
        size=9.0,
        color=(0.0, 0.2, 0.7),
        align="right",
        link="https://ilwoolyu.github.io/",
    )

    # Footer
    page.draw_text(
        FOOTER_TEXT,
        page.width / 2.0,
        18.0,
        font="Helvetica",
        size=8.0,
        align="center",
        color=(0.3, 0.3, 0.3),
    )


def _default_tile_image(label: str, w: int = 360, h: int = 280) -> np.ndarray:
    """Generate a clean procedural tile image when rendering is mocked or missing."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    # Inner border
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 200
    return img


def build_reg_page(
    doc: PdfDocument,
    *,
    template: str = "sphere_327680",
    project: str = "Project",
    subject: str = "Subject",
    session: str = "Session",
    version: str = "2.2 (CBT_v1.3)",
    date_str: str | None = None,
    tiles: dict[str, Any] | None = None,
    energy_lh: tuple[float, float] = (0.0, 0.0),
    energy_rh: tuple[float, float] = (0.0, 0.0),
) -> PdfPage:
    """Build HSD registration report page (8.5 x 10.5 in)."""
    page = doc.add_page(PAGE_WIDTH, REG_HEIGHT)
    _draw_header_and_frame(
        page,
        "HSD: Cortical Surface Registration",
        project=project,
        subject=subject,
        session=session,
        version=version,
        date_str=date_str,
        ref_links=[("Paper", "https://www.ncbi.nlm.nih.gov/pubmed/31280090")],
        software_links=[("Software", "https://github.com/ilwoolyu/HSD")],
    )

    t = tiles or {}
    tile_w = 187.2  # 2.6 in
    tile_h = 240.0
    x_offset = (PAGE_WIDTH - 3 * tile_w) / 2.0  # 25.2 pt

    # Left Hemisphere row
    y_lh = REG_HEIGHT - 390.0
    for c_idx, name in enumerate(["sph_l", "reg_l", "tmp_l"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_lh, tile_w - 4, tile_h - 4)

    # Caption LH
    e0_l, e1_l = energy_lh
    page.draw_text(
        f"Left Hemisphere (Energy: {e0_l:.3f} -> {e1_l:.3f})",
        PAGE_WIDTH / 2.0,
        y_lh - 14.0,
        font="Helvetica",
        size=10.0,
        align="center",
    )

    # Right Hemisphere row
    y_rh = y_lh - tile_h - 25.0
    for c_idx, name in enumerate(["sph_r", "reg_r", "tmp_r"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_rh, tile_w - 4, tile_h - 4)

    # Caption RH
    e0_r, e1_r = energy_rh
    page.draw_text(
        f"Right Hemisphere (Energy: {e0_r:.3f} -> {e1_r:.3f})",
        PAGE_WIDTH / 2.0,
        y_rh - 14.0,
        font="Helvetica",
        size=10.0,
        align="center",
    )

    return page


def build_parc_page(
    doc: PdfDocument,
    protocol: str = "braincolor",
    *,
    project: str = "Project",
    subject: str = "Subject",
    session: str = "Session",
    version: str = "2.2 (CBT_v1.3)",
    date_str: str | None = None,
    tiles: dict[str, Any] | None = None,
) -> PdfPage:
    """Build parcellation report page (8.5 x 8.7 in)."""
    page = doc.add_page(PAGE_WIDTH, PARC_HEIGHT)
    _draw_header_and_frame(
        page,
        "Cortical Surface Parcellation",
        project=project,
        subject=subject,
        session=session,
        version=version,
        date_str=date_str,
        ref_links=[("Paper", "https://www.ncbi.nlm.nih.gov/pubmed/31803864")],
        software_links=[("Software", "https://github.com/ilwoolyu/SphericalLabeling")],
    )

    t = tiles or {}
    tile_w = 288.0  # 4.0 in
    tile_h = 240.0
    x_offset = (PAGE_WIDTH - 2 * tile_w) / 2.0  # 18.0 pt

    # Row 1: White Surfaces, Gray Surfaces
    y_r1 = PARC_HEIGHT - 335.0
    for c_idx, name in enumerate(["wm", "gm"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_r1, tile_w - 4, tile_h - 4)

    # Row 2: Middle Surfaces, Inflated Surfaces
    y_r2 = y_r1 - tile_h - 10.0
    for c_idx, name in enumerate(["mid", "inf"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_r2, tile_w - 4, tile_h - 4)

    return page


def build_curve_page(
    doc: PdfDocument,
    *,
    project: str = "Project",
    subject: str = "Subject",
    session: str = "Session",
    version: str = "2.2 (CBT_v1.3)",
    date_str: str | None = None,
    tiles: dict[str, Any] | None = None,
) -> PdfPage:
    """Build TRACE curve extraction report page (8.5 x 9.2 in)."""
    page = doc.add_page(PAGE_WIDTH, CURVE_HEIGHT)
    _draw_header_and_frame(
        page,
        "TRACE: Curve Extraction",
        project=project,
        subject=subject,
        session=session,
        version=version,
        date_str=date_str,
        ref_links=[("Paper", "https://www.ncbi.nlm.nih.gov/pubmed/29969416")],
        software_links=[("Software", "https://github.com/ilwoolyu/CurveExtraction")],
    )

    t = tiles or {}
    tile_w = 288.0  # 4.0 in
    tile_h = 240.0
    x_offset = (PAGE_WIDTH - 2 * tile_w) / 2.0  # 18.0 pt

    # Row 1: Sulcal Curves (Original, Inflated)
    y_r1 = CURVE_HEIGHT - 350.0
    for c_idx, name in enumerate(["sulcus_mid", "sulcus_inf"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_r1, tile_w - 4, tile_h - 4)

    # Caption Row 1
    page.draw_text(
        "Sulcal Curves", PAGE_WIDTH / 2.0, y_r1 - 14.0, font="Helvetica", size=10.0, align="center"
    )

    # Row 2: Gyral Curves (Original, Inflated)
    y_r2 = y_r1 - tile_h - 25.0
    for c_idx, name in enumerate(["gyrus_mid", "gyrus_inf"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_r2, tile_w - 4, tile_h - 4)

    # Caption Row 2
    page.draw_text(
        "Gyral Curves", PAGE_WIDTH / 2.0, y_r2 - 14.0, font="Helvetica", size=10.0, align="center"
    )

    return page


def build_quant_page(
    doc: PdfDocument,
    *,
    project: str = "Project",
    subject: str = "Subject",
    session: str = "Session",
    version: str = "2.2 (CBT_v1.3)",
    date_str: str | None = None,
    tiles: dict[str, Any] | None = None,
) -> PdfPage:
    """Build quantification report page (8.5 x 11.7 in)."""
    page = doc.add_page(PAGE_WIDTH, QUANT_HEIGHT)
    _draw_header_and_frame(
        page,
        "Cortical Surface Quantification",
        project=project,
        subject=subject,
        session=session,
        version=version,
        date_str=date_str,
        ref_links=[("Paper (SD, LGI, SCI)", "https://www.ncbi.nlm.nih.gov/pubmed/29887663")],
        software_links=[("Software (LGI, SCI)", "https://github.com/ilwoolyu/LocalGyrificationIndex")],
    )

    t = tiles or {}
    tile_w = 288.0  # 4.0 in
    tile_h = 230.0
    x_offset = (PAGE_WIDTH - 2 * tile_w) / 2.0  # 18.0 pt

    # Row 1: Mean Curvature (H), Shape Index (S)
    y_r1 = QUANT_HEIGHT - 330.0
    for c_idx, name in enumerate(["H", "S"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_r1, tile_w - 4, tile_h - 4)

    # Row 2: Cortical Thickness (CT), Sulcal Depth (SD)
    y_r2 = y_r1 - tile_h - 10.0
    for c_idx, name in enumerate(["CT", "SD"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_r2, tile_w - 4, tile_h - 4)

    # Row 3: Local Gyrification Index (LGI), Shape Complexity Index (SCI)
    y_r3 = y_r2 - tile_h - 10.0
    for c_idx, name in enumerate(["LGI", "SCI"]):
        x = x_offset + c_idx * tile_w
        img = t.get(name, _default_tile_image(name))
        page.draw_image(img, x + 2, y_r3, tile_w - 4, tile_h - 4)

    return page


def generate_report_pdfs(
    out_dir: str | Path,
    *,
    template: str = "sphere_327680",
    protocols: Sequence[str] = ("braincolor", "dkt31"),
    project: str = "Project",
    subject: str = "Subject",
    session: str = "Session",
    version: str = "2.2 (CBT_v1.3)",
    date_str: str | None = None,
    tiles: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate all QA report PDFs and merge them into PDF/biscuit.pdf."""
    out_base = Path(out_dir)
    pdf_dir = out_base / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_files: dict[str, Path] = {}

    # 1. Registration page
    reg_doc = PdfDocument()
    build_reg_page(
        reg_doc,
        template=template,
        project=project,
        subject=subject,
        session=session,
        version=version,
        date_str=date_str,
        tiles=tiles,
    )
    reg_path = pdf_dir / f"reg.{template}.pdf"
    reg_doc.write(reg_path)
    out_files["reg"] = reg_path

    # 2. Parcellation pages
    for proto in protocols:
        parc_doc = PdfDocument()
        build_parc_page(
            parc_doc,
            protocol=proto,
            project=project,
            subject=subject,
            session=session,
            version=version,
            date_str=date_str,
            tiles=tiles,
        )
        parc_path = pdf_dir / f"parc.{proto}.pdf"
        parc_doc.write(parc_path)
        out_files[f"parc_{proto}"] = parc_path

    # 3. Curve page
    curve_doc = PdfDocument()
    build_curve_page(
        curve_doc,
        project=project,
        subject=subject,
        session=session,
        version=version,
        date_str=date_str,
        tiles=tiles,
    )
    curve_path = pdf_dir / "curve.pdf"
    curve_doc.write(curve_path)
    out_files["curve"] = curve_path

    # 4. Quantification page
    quant_doc = PdfDocument()
    build_quant_page(
        quant_doc,
        project=project,
        subject=subject,
        session=session,
        version=version,
        date_str=date_str,
        tiles=tiles,
    )
    quant_path = pdf_dir / "quant.pdf"
    quant_doc.write(quant_path)
    out_files["quant"] = quant_path

    # 5. Merged biscuit.pdf in PDF/ directory
    # Image wrapper merge order: reg.*.pdf parc.* curve.pdf quant.pdf
    merge_list = [reg_path]
    for proto in protocols:
        merge_list.append(pdf_dir / f"parc.{proto}.pdf")
    merge_list.extend([curve_path, quant_path])

    final_pdf_dir = out_base / "PDF"
    final_pdf_dir.mkdir(parents=True, exist_ok=True)
    biscuit_pdf_path = final_pdf_dir / "biscuit.pdf"

    # Merge into a single multi-page PDF preserving individual page heights
    merged_doc = PdfDocument()
    for doc_item in [reg_doc]:
        merged_doc.pages.extend(doc_item.pages)
    for proto in protocols:
        p_doc = PdfDocument()
        build_parc_page(
            p_doc,
            protocol=proto,
            project=project,
            subject=subject,
            session=session,
            version=version,
            tiles=tiles,
        )
        merged_doc.pages.extend(p_doc.pages)
    merged_doc.pages.extend(curve_doc.pages)
    merged_doc.pages.extend(quant_doc.pages)
    merged_doc.write(biscuit_pdf_path)

    out_files["biscuit"] = biscuit_pdf_path
    return out_files


def report_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for pybiscuit report."""
    import argparse

    p = argparse.ArgumentParser(
        prog="pybiscuit report",
        description="Generate BISCUIT QA PDF reports (spec §11, B12.3).",
    )
    p.add_argument("-o", "--out", "--output", required=True, help="Output root directory")
    p.add_argument(
        "--template", default="sphere_327680", help="Template name (default: sphere_327680)"
    )
    p.add_argument(
        "--protocols",
        nargs="+",
        default=["braincolor", "dkt31"],
        help="Parcellation protocols (default: braincolor dkt31)",
    )
    p.add_argument("--project", default="Project", help="Project name (default: Project)")
    p.add_argument("--subject", default="Subject", help="Subject name (default: Subject)")
    p.add_argument("--session", default="Session", help="Session name (default: Session)")
    p.add_argument("--version", default="2.2 (CBT_v1.3)", help="Pipeline version string")
    p.add_argument("--date", default=None, help="Report date (default: today)")
    p.add_argument(
        "--cohort",
        nargs="+",
        default=None,
        help="Cohort run directories to aggregate into cohort report (B12.6)",
    )
    p.add_argument(
        "--ref-cohort",
        nargs="+",
        default=None,
        help="Reference run directories for paired parity comparison (B12.6)",
    )
    args = p.parse_args(argv)

    if args.cohort:
        from .cohort import generate_cohort_report

        generate_cohort_report(
            args.cohort,
            args.out,
            ref_dirs=args.ref_cohort,
            protocols=tuple(args.protocols),
        )
        return 0

    generate_report_pdfs(
        args.out,
        template=args.template,
        protocols=args.protocols,
        project=args.project,
        subject=args.subject,
        session=args.session,
        version=args.version,
        date_str=args.date,
    )
    return 0

