"""Optional visualization (``pip install pybiscuit[viz]``).

Nothing outside this package imports it, and the pipeline never needs it. Importing
``pybiscuit.viz`` is light; PyVista/matplotlib load on first render.
"""

from __future__ import annotations

from .cameras import VIEWS, camera_for
from .cohort import (
    CohortSummary,
    bland_altman,
    build_cohort_summary,
    compare_scalar_maps,
    compare_subject_runs,
    compute_icc,
    generate_cohort_report,
    roi_dice,
)
from .colormaps import MEASURES, MeasureStyle, measure_style
from .pdf import PdfDocument, PdfPage
from .report import (
    build_curve_page,
    build_parc_page,
    build_quant_page,
    build_reg_page,
    generate_report_pdfs,
)

__all__ = [
    "CohortSummary",
    "MEASURES",
    "MeasureStyle",
    "PdfDocument",
    "PdfPage",
    "VIEWS",
    "bland_altman",
    "build_cohort_summary",
    "build_curve_page",
    "build_parc_page",
    "build_quant_page",
    "build_reg_page",
    "camera_for",
    "compare_scalar_maps",
    "compare_subject_runs",
    "compute_icc",
    "generate_cohort_report",
    "generate_report_pdfs",
    "measure_style",
    "roi_dice",
]
