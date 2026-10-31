"""Generate manuscript result panels directly from numerical analysis outputs."""

from pathlib import Path
from io import BytesIO
import json
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation

from .pipeline import r_directory
from .plot_blocks import diagnosis_plot, ratio_plot, gap_plot, piecewise_plot
from .layout import LAYOUT, export, panel_letter
from .metrics import conditional_gap_tests, diagnosis_tests


def label_memory_panels(figure_dir):
    """Apply the same vector letters to R and Matplotlib panels."""
    width, height = LAYOUT["figure3"]["canvas_mm"]
    dpi = LAYOUT["png_dpi"]
    for letter in "cd":
        path = Path(figure_dir) / f"figure3{letter}"
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        panel_letter(fig, letter.upper(), width, height,
                     font_pt=LAYOUT["figure3"]["fonts_pt"]["panel_letter"])
        pdf, png = BytesIO(), BytesIO()
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(pdf, format="pdf", transparent=True, bbox_inches=None)
            fig.savefig(png, format="png", dpi=dpi, transparent=True, bbox_inches=None)
        plt.close(fig)
        pdf.seek(0); png.seek(0)
        with Image.open(path.with_suffix(".png")) as source, Image.open(png) as overlay:
            base = source.convert("RGBA")
            if base.size != overlay.size:
                raise ValueError("Panel and letter overlay must have identical pixel dimensions.")
            result = Image.alpha_composite(base, overlay.convert("RGBA"))
        page = PdfReader(path.with_suffix(".pdf")).pages[0]
        writer = PdfWriter()
        writer.add_page(page)
        writer.pages[0].merge_page(PdfReader(pdf).pages[0])
        with path.with_suffix(".pdf").open("wb") as stream:
            writer.write(stream)
        result.convert("RGB").save(path.with_suffix(".png"), dpi=(dpi, dpi))


def compose_panels(figure_dir):
    """Translate finished panels without resampling axes or cropping labels."""
    figures = Path(figure_dir)
    dpi = LAYOUT["png_dpi"]
    geometry = {}
    for number, panels, columns in [(2, "abc", 3), (3, "abcd", 2)]:
        pages = [PdfReader(figures / f"figure{number}{p}.pdf").pages[0] for p in panels]
        widths = [float(p.mediabox.width) for p in pages]
        heights = [float(p.mediabox.height) for p in pages]
        if max(heights) - min(heights) > 1e-6 or max(widths) - min(widths) > 1e-6:
            raise ValueError("Composite panels must have equal physical dimensions.")
        rows = len(panels) // columns
        gap_mm = LAYOUT["figure3"]["row_gap_mm"] if number == 3 else 0
        trim_mm = LAYOUT["figure3"].get("top_row_bottom_trim_mm", 0) if number == 3 else 0
        if not math.isfinite(trim_mm) or not 0 <= trim_mm < heights[0] * 25.4 / 72:
            raise ValueError("Top-row bottom trim must be finite, nonnegative, and smaller than the panel height.")
        gap = gap_mm * 72 / 25.4
        trim = trim_mm * 72 / 25.4
        canvas_width = columns * widths[0]
        canvas_height = rows * heights[0] + (rows - 1) * gap - trim
        images = [Image.open(figures / f"figure{number}{p}.png").convert("RGB") for p in panels]
        if len({im.size for im in images}) != 1:
            raise ValueError("PNG panel dimensions differ.")
        pw, ph = images[0].size
        gap_px = round(gap_mm * dpi / 25.4)
        trim_px = round(trim_mm * dpi / 25.4)
        # Check every pixel intersecting the physical band before cropping either format.
        check_px = math.ceil(trim_mm * dpi / 25.4)
        if check_px >= ph:
            raise ValueError("Top-row bottom trim must be smaller than the PNG panel height.")
        if check_px:
            for letter, im in zip(panels[:columns], images[:columns]):
                band = im.crop((0, ph - check_px, pw, ph))
                if any(channel != (255, 255) for channel in band.getextrema()):
                    raise ValueError(f"Figure {number}{letter} bottom trim contains nonblank pixels.")
        writer = PdfWriter()
        canvas = writer.add_blank_page(canvas_width, canvas_height)
        for index, page in enumerate(pages):
            row, col = divmod(index, columns)
            x = col * widths[0]
            y = (rows - 1 - row) * (heights[0] + gap)
            if row == 0 and trim:
                page.cropbox.bottom = float(page.mediabox.bottom) + trim
                y -= trim
            canvas.merge_transformed_page(page, Transformation().translate(x, y))
        with (figures / f"figure{number}.pdf").open("wb") as stream:
            writer.write(stream)
        image = Image.new("RGB", (columns * pw, rows * ph + (rows - 1) * gap_px - trim_px), "white")
        for index, im in enumerate(images):
            row, col = divmod(index, columns)
            y = row * (ph + gap_px) - (trim_px if row else 0)
            image.paste(im.crop((0, 0, pw, ph - trim_px)) if row == 0 and trim_px else im,
                        (col * pw, y))
            im.close()
        image.save(figures / f"figure{number}.png", dpi=(dpi, dpi))
        geometry[f"figure{number}"] = {"width_mm": canvas_width * 25.4 / 72,
                                       "height_mm": canvas_height * 25.4 / 72,
                                       "panels": list(panels), "columns": columns,
                                       "row_gap_mm": gap_mm,
                                       "top_row_bottom_trim_mm": trim_mm}
    return geometry


def render_figures(work_dir, rscript="Rscript"):
    work = Path(work_dir).resolve()
    data = work / "data" / "analysis"
    figures = work / "figures"
    report = work / "reports" / "analysis"
    figures.mkdir(exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(data / name, float_precision="round_trip")
    checks = {}
    diagnosis = read("diagnosis_visits.csv")
    diagnosis_result = diagnosis_tests(diagnosis)
    (report / "diagnosis_tests.json").write_text(json.dumps(diagnosis_result, indent=2, allow_nan=False) + "\n")
    checks["figure2a"] = export(diagnosis_plot(diagnosis, diagnosis_result["comparisons"]), "figure2a", figures)
    palette = {"SLOPE": "#B8E0D2", "Centiloids": "#A2D2FF", "dEBM": "#F7C58A", "DAPS": "#FFD1DC"}
    methods = [("SLOPE", "SLOPE"), ("Centiloids", "Centiloids"), ("dEBM", "dEBM_upstream"), ("DAPS", "PL_DAPS")]
    metrics = read("zero_threshold_metrics.csv").replace({"Method": {"PACE-Abeta": "DAPS"}}).set_index("Method").loc[list(palette)].reset_index()
    checks["figure2b"] = export(ratio_plot(metrics, palette), "figure2b", figures)
    pairs = read("adjacent_pairs.csv")
    gaps = []
    for label, key in methods:
        decline = pairs[f"{key}_rank_decline"]
        gaps.append(pd.DataFrame({"Method": label, "Violation_Gap": decline[decline > 0]}))
    gap_frame = pd.concat(gaps, ignore_index=True)
    gap_frame["Method"] = pd.Categorical(gap_frame.Method, categories=list(palette), ordered=True)
    gap_frame = gap_frame.sort_values("Method", kind="stable")
    gap_tests = conditional_gap_tests(pairs)
    (report / "conditional_gap_tests.json").write_text(json.dumps(gap_tests, indent=2) + "\n")
    annotations = [{**row, "reference": "DAPS"} for row in gap_tests["comparisons"]]
    checks["figure2c"] = export(gap_plot(gap_frame, list(palette), palette, annotations), "figure2c", figures)
    piecewise = json.loads((data / "piecewise.json").read_text())
    for panel, split, title in [("a", "train", "ADNI (Train)"), ("b", "test", "ADNI (Test)")]:
        fig = piecewise_plot(read(f"piecewise_{split}.csv"), title, manual_bp=piecewise["train"]["knot"])
        checks[f"figure3{panel}"] = export(fig, f"figure3{panel}", figures)
    environment = os.environ.copy()
    environment["PACE_PYTHON"] = sys.executable
    log_path = report / "memory_figure_render.log"
    try:
        with log_path.open("w") as log:
            subprocess.run([str(rscript), str(r_directory() / "figure4.R"), str(work)],
                           check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Memory-panel rendering failed; inspect {log_path}") from error
    label_memory_panels(figures)
    memory_geometry = json.loads((report / "memory_figure_geometry.json").read_text())
    checks.update(memory_geometry["panels"])
    geometry = compose_panels(figures)
    (report / "figure_geometry.json").write_text(json.dumps({"panels": checks, "composites": geometry}, indent=2) + "\n")
    return figures
