"""Optional visualization (``pip install pybiscuit[viz]``).

Nothing outside this package imports it, and the pipeline never needs it. Importing
``pybiscuit.viz`` is light; PyVista/matplotlib load on first render.
"""

from __future__ import annotations

from .cameras import VIEWS, camera_for
from .colormaps import MEASURES, MeasureStyle, measure_style

__all__ = ["MEASURES", "VIEWS", "MeasureStyle", "camera_for", "measure_style"]
