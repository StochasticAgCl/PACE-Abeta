import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from pace_abeta.analysis.figures import compose_panels, label_memory_panels
from pace_abeta.analysis.layout import LAYOUT, export, panel_letter, plot_signature


POINTS_PER_MM = 72 / 25.4


def _write_panel(path, label, canvas_mm, color, image_size=(21, 17), blank_bottom_px=0):
    """Write a known vector rectangle and an unscaled raster test pattern."""
    writer = PdfWriter()
    page = writer.add_blank_page(*(v * POINTS_PER_MM for v in canvas_mm))
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(("q 0.2 0.3 0.4 rg 11 40 23 29 re f Q\n"
                     f"BT /F1 9 Tf 1 0 0 1 3 32 Tm ({label}) Tj ET\n").encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.with_suffix(".pdf").open("wb") as output:
        writer.write(output)
    image = Image.new("RGB", image_size, color)
    image.putpixel((2, 3), (255, 255, 255))
    if blank_bottom_px:
        image.paste("white", (0, image.height - blank_bottom_px, image.width, image.height))
    image.save(path.with_suffix(".png"))


def _vector_rectangles(page):
    rectangles = []

    def visit(operator, operands, matrix, text_matrix):
        if operator != b"re" or len(operands) != 4:
            return
        x, y, width, height = map(float, operands)
        if (width, height) != (23, 29):
            return
        a, b, c, d, e, f = matrix
        rectangles.append((a * x + c * y + e, b * x + d * y + f,
                           a * width + c * height, b * width + d * height))

    page.extract_text(visitor_operand_before=visit)
    return rectangles


@pytest.fixture(params=[0, 7], ids=["untrimmed", "trimmed"])
def panels(tmp_path, monkeypatch, request):
    monkeypatch.setitem(LAYOUT["figure3"], "top_row_bottom_trim_mm", request.param)
    colors = [(25, 50, 75), (50, 100, 150), (100, 150, 200), (150, 200, 250)]
    for number, letters in [(2, "abc"), (3, "abcd")]:
        canvas_mm = LAYOUT[f"figure{number}"]["canvas_mm"]
        image_size = tuple(int(mm * LAYOUT["png_dpi"] / 25.4) for mm in canvas_mm)
        for index, letter in enumerate(letters):
            blank = math.ceil(request.param * LAYOUT["png_dpi"] / 25.4) if number == 3 and index < 2 else 0
            _write_panel(tmp_path / f"figure{number}{letter}", letter.upper(),
                         canvas_mm, colors[index], image_size, blank)
    return tmp_path


@pytest.mark.parametrize("name", ["figure2a", "figure3a", "figure3b"])
def test_export_preserves_statistics_and_declared_physical_axes(tmp_path, name):
    with plt.rc_context({"figure.dpi": 300, "savefig.bbox": "tight"}):
        figure, axes = plt.subplots(figsize=(5.2, 4.6))
        x = np.linspace(0, 1, 11)
        axes.scatter(x, x * 100, color="#9ECDF2", alpha=.55)
        axes.plot(x, x * 100, color="#E45756")
        axes.text(.4, 120, "p = 0.04")
        axes.set(xlabel="Score", ylabel="Burden", title="Example")
        axes.set(xlim=(-.05, 1.05), ylim=(-20, 160))
        axes.set_xticks([0, .5, 1])
        signature = plot_signature(figure)
        report = export(figure, name, tmp_path)
    expected = LAYOUT["figure2" if name.startswith("figure2") else "figure3"]
    np.testing.assert_allclose(report["axes_measured_mm"], expected["axes_rect_mm"],
                               rtol=0, atol=1e-10)
    assert plot_signature(figure) == signature
    page = PdfReader(tmp_path / f"{name}.pdf").pages[0]
    np.testing.assert_allclose([float(page.mediabox.width), float(page.mediabox.height)],
                               np.array(expected["canvas_mm"]) * POINTS_PER_MM,
                               rtol=0, atol=1e-7)
    with Image.open(tmp_path / f"{name}.png") as image:
        expected_pixels = tuple(int(mm / 25.4 * LAYOUT["png_dpi"])
                                for mm in expected["canvas_mm"])
        assert image.size == expected_pixels
        np.testing.assert_allclose(image.info["dpi"], [LAYOUT["png_dpi"]] * 2,
                                   rtol=0, atol=0.02)


def test_panel_letter_anchor_is_equal_in_physical_units():
    style = LAYOUT["panel_letter"]
    anchors = []
    for number in (2, 3):
        width, height = LAYOUT[f"figure{number}"]["canvas_mm"]
        figure = plt.figure(figsize=(width / 25.4, height / 25.4))
        panel_letter(figure, "A", width, height)
        text = figure.texts[0]
        x, y = text.get_position()
        anchors.append([x * width, (1 - y) * height])
        assert text.get_fontsize() == style["font_pt"]
        assert text.get_va() == "baseline"
        assert text.get_ha() == "left"
        plt.close(figure)
    np.testing.assert_allclose(anchors, [[style["x_mm"], style["baseline_from_top_mm"]]] * 2,
                               rtol=0, atol=1e-12)


def test_composition_preserves_vector_scale_and_orders_above_cd(panels):
    geometry = compose_panels(panels)
    for number, letters, columns in [(2, "abc", 3), (3, "abcd", 2)]:
        width, height = np.array(LAYOUT[f"figure{number}"]["canvas_mm"]) * POINTS_PER_MM
        gap = LAYOUT["figure3"]["row_gap_mm"] * POINTS_PER_MM if number == 3 else 0
        trim_mm = LAYOUT["figure3"]["top_row_bottom_trim_mm"] if number == 3 else 0
        trim = trim_mm * POINTS_PER_MM
        page = PdfReader(panels / f"figure{number}.pdf").pages[0]
        rows = len(letters) // columns
        expected_rectangles = []
        for index in range(len(letters)):
            row, col = divmod(index, columns)
            expected_rectangles.append([11 + col * width,
                                        40 + (rows - 1 - row) * (height + gap) - (trim if row == 0 else 0),
                                        23, 29])
        np.testing.assert_allclose(_vector_rectangles(page), expected_rectangles, rtol=0, atol=1e-6)
        operations = page.get_contents().operations
        clips = [list(map(float, operands)) for index, (operands, operator) in enumerate(operations[:-1])
                 if operator == b"re" and operations[index + 1][1] == b"W"]
        expected_clips = [[0, trim if index < columns else 0, width,
                           height - (trim if index < columns else 0)] for index in range(len(letters))]
        np.testing.assert_allclose(clips, expected_clips, rtol=0, atol=1e-6)
        assert "".join(page.extract_text().split()) == letters.upper()
        assert geometry[f"figure{number}"]["panels"] == list(letters)
        assert geometry[f"figure{number}"]["top_row_bottom_trim_mm"] == trim_mm
        np.testing.assert_allclose([float(page.mediabox.width), float(page.mediabox.height)],
                                   [columns * width, rows * height + (rows - 1) * gap - trim],
                                   rtol=0, atol=2e-6)


def test_composition_keeps_raster_pixels_and_row_gap(panels):
    compose_panels(panels)
    gap = round(LAYOUT["figure3"]["row_gap_mm"] * LAYOUT["png_dpi"] / 25.4)
    trim = round(LAYOUT["figure3"]["top_row_bottom_trim_mm"] * LAYOUT["png_dpi"] / 25.4)
    with Image.open(panels / "figure3a.png") as source:
        width, height = source.size
    top_height = height - trim
    with Image.open(panels / "figure3.png") as composite:
        assert composite.size == (2 * width, 2 * height + gap - trim)
        for index, letter in enumerate("abcd"):
            row, col = divmod(index, 2)
            y = row * (top_height + gap)
            kept_height = top_height if row == 0 else height
            crop = composite.crop((col * width, y, (col + 1) * width, y + kept_height))
            with Image.open(panels / f"figure3{letter}.png") as source:
                np.testing.assert_array_equal(np.asarray(crop), np.asarray(source)[:kept_height])
        np.testing.assert_array_equal(np.asarray(composite)[top_height:top_height + gap], 255)


@pytest.mark.parametrize("letter", "ab")
def test_composition_rejects_nonblank_trim(panels, monkeypatch, letter):
    monkeypatch.setitem(LAYOUT["figure3"], "top_row_bottom_trim_mm", 7)
    for panel in "ab":
        path = panels / f"figure3{panel}.png"
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.paste("white", (0, image.height - math.ceil(7 * LAYOUT["png_dpi"] / 25.4), image.width, image.height))
        if panel == letter:
            image.putpixel((0, image.height - 1), (254, 255, 255))
        image.save(path)
    with pytest.raises(ValueError, match=f"Figure 3{letter} bottom trim contains nonblank pixels"):
        compose_panels(panels)
    assert not (panels / "figure3.pdf").exists()
    assert not (panels / "figure3.png").exists()


@pytest.mark.parametrize("trim", [-1, float("nan"), float("inf"), "height", "beyond"])
def test_composition_rejects_invalid_trim(panels, monkeypatch, trim):
    if isinstance(trim, str):
        trim = LAYOUT["figure3"]["canvas_mm"][1] + (trim == "beyond")
    monkeypatch.setitem(LAYOUT["figure3"], "top_row_bottom_trim_mm", trim)
    with pytest.raises(ValueError, match="Top-row bottom trim must"):
        compose_panels(panels)


def test_composition_rejects_trim_larger_than_raster(panels, monkeypatch):
    monkeypatch.setitem(LAYOUT["figure3"], "top_row_bottom_trim_mm", 7)
    for letter in "abcd":
        Image.new("RGB", (21, 17), "white").save(panels / f"figure3{letter}.png")
    with pytest.raises(ValueError, match="smaller than the PNG panel height"):
        compose_panels(panels)


@pytest.mark.parametrize("dimension", ["width", "height"])
def test_composition_rejects_mismatched_pdf_panels(panels, dimension):
    path = panels / "figure3d.pdf"
    page = PdfReader(path).pages[0]
    if dimension == "width":
        page.mediabox.right += 1
    else:
        page.mediabox.top += 1
    writer = PdfWriter()
    writer.add_page(page)
    with path.open("wb") as output:
        writer.write(output)
    with pytest.raises(ValueError, match="equal physical dimensions"):
        compose_panels(panels)


def test_composition_rejects_mismatched_png_panels(panels):
    Image.new("RGB", (22, 17), "white").save(panels / "figure3d.png")
    with pytest.raises(ValueError, match="PNG panel dimensions differ"):
        compose_panels(panels)


def test_memory_letter_overlay_preserves_panel_geometry_and_content(tmp_path):
    canvas = LAYOUT["figure3"]["canvas_mm"]
    image_size = tuple(int(mm / 25.4 * LAYOUT["png_dpi"]) for mm in canvas)
    original_pixels = {}
    for letter in "cd":
        path = tmp_path / f"figure3{letter}"
        _write_panel(path, "DATA", canvas, (230, 240, 250), image_size)
        with Image.open(path.with_suffix(".png")) as image:
            original_pixels[letter] = np.asarray(image).copy()
    label_memory_panels(tmp_path)
    for letter in "cd":
        page = PdfReader(tmp_path / f"figure3{letter}.pdf").pages[0]
        assert "".join(page.extract_text().split()) == "DATA" + letter.upper()
        np.testing.assert_allclose(_vector_rectangles(page), [[11, 40, 23, 29]], rtol=0, atol=1e-7)
        np.testing.assert_allclose([float(page.mediabox.width), float(page.mediabox.height)],
                                   np.array(canvas) * POINTS_PER_MM, rtol=0, atol=1e-6)
        with Image.open(tmp_path / f"figure3{letter}.png") as image:
            # The letter occupies only the top margin; the plot body stays exact.
            first_plot_row = int(8 / 25.4 * LAYOUT["png_dpi"])
            np.testing.assert_array_equal(np.asarray(image)[first_plot_row:],
                                          original_pixels[letter][first_plot_row:])
