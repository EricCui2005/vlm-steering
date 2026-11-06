import math

import pytest
from PIL import ImageChops, Image

from steering.config import load_config
from steering.symbols import IMAGE_SIZE, SHAPES, render_symbol

BLACK = (0, 0, 0)


def symbol_box(shape: str) -> tuple[int, int, int, int]:
    image = render_symbol(shape, BLACK)
    background = Image.new("RGB", image.size, (255, 255, 255))
    return ImageChops.difference(image, background).point(lambda v: 255 if v > 32 else 0).getbbox()


def test_library_has_288_symbols():
    assert len(SHAPES) * len(load_config()["colors"]) == 288


@pytest.mark.parametrize("shape", [name for name in SHAPES if name != "small_x"])
def test_symbol_is_45_to_55_percent_of_the_canvas(shape):
    left, top, right, bottom = symbol_box(shape)
    extent = max(right - left, bottom - top) / IMAGE_SIZE
    assert 0.45 <= extent <= 0.55


@pytest.mark.parametrize("shape", SHAPES)
def test_symbol_is_centered(shape):
    left, top, right, bottom = symbol_box(shape)
    center = IMAGE_SIZE / 2
    tolerance = 0.03 * IMAGE_SIZE
    assert abs((left + right) / 2 - center) <= tolerance
    assert abs((top + bottom) / 2 - center) <= tolerance


def test_curved_strokes_have_no_seams():
    """Every pixel along the middle of the ring should be solid color."""
    image = render_symbol("open_circle_loop", BLACK)
    pixels_per_unit = IMAGE_SIZE * 0.5 / 2
    center = IMAGE_SIZE / 2
    for degrees in range(0, 360):
        radians = math.radians(degrees)
        x = center + 0.88 * pixels_per_unit * math.cos(radians)
        y = center + 0.88 * pixels_per_unit * math.sin(radians)
        assert image.getpixel((round(x), round(y))) == BLACK


def test_smoke_test_symbols_are_in_the_library():
    config = load_config()
    library = {f"{color}_{shape}" for color in config["colors"] for shape in SHAPES}
    assert set(config["smoke_test"]["symbols"]) <= library
