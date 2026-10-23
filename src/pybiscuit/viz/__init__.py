"""Optional visualization (``pip install pybiscuit[viz]``).

Nothing outside this package imports it, and the pipeline never needs it. Importing
``pybiscuit.viz`` is light; PyVista/matplotlib load on first render.
"""

from __future__ import annotations

from .cameras import VIEWS, camera_for
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
    "MEASURES",
    "VIEWS",
    "MeasureStyle",
    "PdfDocument",
    "PdfPage",
    "build_curve_page",
    "build_parc_page",
    "build_quant_page",
    "build_reg_page",
    "camera_for",
    "generate_report_pdfs",
    "measure_style",
]
