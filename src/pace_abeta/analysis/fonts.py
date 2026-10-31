"""Portable sans-serif selection for the manuscript plotting style."""
from matplotlib import font_manager

try:
    font_manager.findfont("Arial", fallback_to_default=False)
    FONT_FAMILY = "Arial"
except ValueError:
    FONT_FAMILY = "DejaVu Sans"
