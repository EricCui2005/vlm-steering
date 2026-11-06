"""Draw the 288 symbol images: 24 shapes in 12 colors.

Shapes are described in simple coordinates: (0, 0) is the center of the symbol,
(1, 1) is its top-right corner, and y points up. The Canvas class converts those
coordinates to pixels. Each image is drawn at 4x size and then scaled down, which
gives clean anti-aliased edges.
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = 400
SUPERSAMPLING = 4
SYMBOL_WIDTH_FRACTION = 0.5
STROKE_WIDTH = 0.24
WHITE = (255, 255, 255)

Point = tuple[float, float]


class Canvas:
    """A white image that accepts drawing commands in symbol coordinates."""

    def __init__(self, color: tuple[int, int, int]):
        self.color = color
        size = IMAGE_SIZE * SUPERSAMPLING
        self.image = Image.new("RGB", (size, size), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.center = size / 2
        self.scale = size * SYMBOL_WIDTH_FRACTION / 2

    def to_pixels(self, point: Point) -> Point:
        x, y = point
        return (self.center + x * self.scale, self.center - y * self.scale)

    def polygon(self, points: list[Point], color=None) -> None:
        self.draw.polygon([self.to_pixels(p) for p in points], fill=color or self.color)

    def stroke(self, points: list[Point], width: float = STROKE_WIDTH, color=None) -> None:
        """Draw a thick line through the points, with rounded joints and flat ends."""
        pixels = [self.to_pixels(p) for p in points]
        self.draw.line(pixels, fill=color or self.color, width=round(width * self.scale))
        # A disc at every interior point covers the small gaps PIL leaves between segments.
        for point in points[1:-1]:
            self.disc(point, width / 2, color)

    def disc(self, center: Point, radius: float, color=None) -> None:
        x, y = self.to_pixels(center)
        r = radius * self.scale
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=color or self.color)

    def finished_image(self) -> Image.Image:
        return self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


# Geometry helpers


def point_on_circle(radius: float, degrees: float) -> Point:
    radians = math.radians(degrees)
    return (radius * math.cos(radians), radius * math.sin(radians))


def arc(radius: float, start_degrees: float, end_degrees: float, steps: int = 120) -> list[Point]:
    return [
        point_on_circle(radius, start_degrees + (end_degrees - start_degrees) * i / steps)
        for i in range(steps + 1)
    ]


def arrowhead(tip: Point, degrees: float, size: float = 0.5) -> list[Point]:
    """A filled triangle whose point is at `tip`, facing the given direction."""
    tip_x, tip_y = tip
    left_x, left_y = point_on_circle(size, degrees + 150)
    right_x, right_y = point_on_circle(size, degrees - 150)
    return [tip, (tip_x + left_x, tip_y + left_y), (tip_x + right_x, tip_y + right_y)]


def straight_arrow(canvas: Canvas, degrees: float) -> None:
    tip = point_on_circle(1, degrees)
    tail = point_on_circle(-1, degrees)
    neck = point_on_circle(0.6, degrees)
    canvas.stroke([tail, neck])
    canvas.polygon(arrowhead(tip, degrees))


def draw_x(canvas: Canvas, reach: float, width: float) -> None:
    canvas.stroke([(-reach, -reach), (reach, reach)], width)
    canvas.stroke([(-reach, reach), (reach, -reach)], width)


# Filled shapes


def circle(canvas: Canvas) -> None:
    canvas.disc((0, 0), 1)


def spiky_blob(canvas: Canvas) -> None:
    """An irregular star-burst. The fixed seed makes it identical in every color."""
    rng = random.Random(7)
    spike_count = 14
    points = []
    for i in range(spike_count * 2):
        is_spike_tip = i % 2 == 0
        radius = rng.uniform(0.9, 1.08) if is_spike_tip else rng.uniform(0.38, 0.55)
        points.append(point_on_circle(radius, i * 180 / spike_count))
    canvas.polygon(points)


def rounded_blob(canvas: Canvas) -> None:
    def radius(degrees: float) -> float:
        """Waves of different lengths, so the outline doesn't look like a star."""
        wobble = 0.09 * math.sin(math.radians(2 * degrees + 20))
        wobble += 0.08 * math.sin(math.radians(3 * degrees + 70))
        wobble += 0.05 * math.sin(math.radians(7 * degrees))
        return 0.88 + wobble

    canvas.polygon([point_on_circle(radius(d), d) for d in range(360)])


def plus(canvas: Canvas) -> None:
    half_width = 0.28
    canvas.polygon([(-1, -half_width), (1, -half_width), (1, half_width), (-1, half_width)])
    canvas.polygon([(-half_width, -1), (half_width, -1), (half_width, 1), (-half_width, 1)])


def star(canvas: Canvas) -> None:
    points = [
        point_on_circle(1.0 if i % 2 == 0 else 0.42, 90 + i * 36)
        for i in range(10)
    ]
    canvas.polygon(points)


def square(canvas: Canvas) -> None:
    canvas.polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])


def triangle(canvas: Canvas) -> None:
    canvas.polygon([(-1, -0.87), (1, -0.87), (0, 0.87)])


def heart(canvas: Canvas) -> None:
    """The standard parametric heart curve, scaled to fit the symbol width."""
    points = []
    for degrees in range(360):
        t = math.radians(degrees)
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        points.append((x / 16, (y + 2.5) / 16))
    canvas.polygon(points)


def caution_triangle(canvas: Canvas) -> None:
    triangle(canvas)
    canvas.polygon([(-0.08, -0.3), (0.08, -0.3), (0.12, 0.3), (-0.12, 0.3)], color=WHITE)
    canvas.disc((0, -0.53), 0.11, color=WHITE)


# Line shapes


def open_circle_loop(canvas: Canvas) -> None:
    canvas.stroke(arc(0.88, 0, 360))


def horizontal_line(canvas: Canvas) -> None:
    canvas.polygon([(-1, -0.12), (1, -0.12), (1, 0.12), (-1, 0.12)])


def vertical_line(canvas: Canvas) -> None:
    canvas.polygon([(-0.12, -1), (0.12, -1), (0.12, 1), (-0.12, 1)])


def small_x(canvas: Canvas) -> None:
    """Half the size of big_x. This is the one shape meant to break the size rule."""
    draw_x(canvas, reach=0.42, width=STROKE_WIDTH)


def big_x(canvas: Canvas) -> None:
    draw_x(canvas, reach=0.9, width=STROKE_WIDTH)


def checkmark(canvas: Canvas) -> None:
    canvas.stroke([(-0.92, 0.0), (-0.3, -0.62), (0.92, 0.72)])


def infinity_symbol(canvas: Canvas) -> None:
    """A lemniscate: a figure eight lying on its side."""
    points = []
    for degrees in range(361):
        t = math.radians(degrees)
        denominator = 1 + math.sin(t) ** 2
        points.append((
            0.9 * math.cos(t) / denominator,
            1.5 * math.sin(t) * math.cos(t) / denominator,
        ))
    canvas.stroke(points, width=0.19)


def spiral(canvas: Canvas) -> None:
    turns = 3
    steps = 360
    points = [
        point_on_circle(0.93 * i / steps, 360 * turns * i / steps)
        for i in range(steps + 1)
    ]
    canvas.stroke(points, width=0.15)
    canvas.disc(points[-1], 0.075)  # A rounded end.


# Arrows


def up_arrow(canvas: Canvas) -> None:
    straight_arrow(canvas, 90)


def down_arrow(canvas: Canvas) -> None:
    straight_arrow(canvas, 270)


def left_arrow(canvas: Canvas) -> None:
    straight_arrow(canvas, 180)


def right_arrow(canvas: Canvas) -> None:
    straight_arrow(canvas, 0)


def u_turn_arrow(canvas: Canvas) -> None:
    """Goes up on the left, curves over the top, and points down on the right."""
    radius = 0.55
    curve_x = -0.07
    curve_y = 0.3

    points = [(curve_x - radius, -1)]
    points += [(curve_x + x, curve_y + y) for x, y in arc(radius, 180, 0)]
    points += [(curve_x + radius, -0.4)]
    canvas.stroke(points)
    canvas.polygon(arrowhead((curve_x + radius, -0.75), 270))


def looping_arrow(canvas: Canvas) -> None:
    """A circle with a gap on the right, ending in an arrowhead that points into the gap."""
    radius = 0.82
    arc_end_degrees = 320
    heading_degrees = arc_end_degrees + 90  # Tangent to the circle, counterclockwise.

    curve = arc(radius, 60, arc_end_degrees)
    canvas.stroke(curve)
    canvas.disc(curve[0], STROKE_WIDTH / 2)  # A rounded tail.

    end_x, end_y = point_on_circle(radius, arc_end_degrees)
    step_x, step_y = point_on_circle(0.42, heading_degrees)
    canvas.polygon(arrowhead((end_x + step_x, end_y + step_y), heading_degrees))


def branching_arrow(canvas: Canvas) -> None:
    """A Y-shaped fork: one stem splitting into two arrows."""
    fork = (0, -0.1)
    canvas.stroke([(0, -1), fork])
    for sideways, degrees in ((-0.5, 135), (0.5, 45)):
        canvas.stroke([fork, (sideways * 1.44, 0.64)])
        canvas.polygon(arrowhead((sideways * 2, 0.92), degrees))


SHAPES = {
    "circle": circle,
    "spiky_blob": spiky_blob,
    "rounded_blob": rounded_blob,
    "plus": plus,
    "star": star,
    "square": square,
    "triangle": triangle,
    "open_circle_loop": open_circle_loop,
    "horizontal_line": horizontal_line,
    "vertical_line": vertical_line,
    "heart": heart,
    "small_x": small_x,
    "down_arrow": down_arrow,
    "u_turn_arrow": u_turn_arrow,
    "up_arrow": up_arrow,
    "right_arrow": right_arrow,
    "left_arrow": left_arrow,
    "looping_arrow": looping_arrow,
    "branching_arrow": branching_arrow,
    "caution_triangle": caution_triangle,
    "big_x": big_x,
    "checkmark": checkmark,
    "infinity_symbol": infinity_symbol,
    "spiral": spiral,
}


def library_names(colors: dict[str, list[int]]) -> list[str]:
    """Every symbol name, like red_heart, in a fixed order."""
    return [f"{color}_{shape}" for color in colors for shape in SHAPES]


def render_symbol(shape: str, color: tuple[int, int, int]) -> Image.Image:
    canvas = Canvas(color)
    SHAPES[shape](canvas)
    return canvas.finished_image()


def build_library(colors: dict[str, list[int]], output_dir: Path) -> None:
    """Save every color and shape combination as {color}_{shape}.png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for color_name, rgb in colors.items():
        for shape in SHAPES:
            image = render_symbol(shape, tuple(rgb))
            image.save(output_dir / f"{color_name}_{shape}.png")


def build_contact_sheet(colors: dict[str, list[int]], symbols_dir: Path, output_path: Path) -> None:
    """Lay all the symbols out in a grid: shapes as rows, colors as columns."""
    cell = 110
    label_width = 170
    header_height = 30
    width = label_width + cell * len(colors)
    height = header_height + cell * len(SHAPES)

    sheet = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)

    for column, color_name in enumerate(colors):
        draw.text((label_width + column * cell + 10, 6), color_name, fill="black", font=font)

    for row, shape in enumerate(SHAPES):
        top = header_height + row * cell
        draw.text((6, top + cell / 2 - 8), shape, fill="black", font=font)
        for column, color_name in enumerate(colors):
            symbol = Image.open(symbols_dir / f"{color_name}_{shape}.png").resize((cell - 6, cell - 6))
            sheet.paste(symbol, (label_width + column * cell + 3, top + 3))
            draw.rectangle(
                (label_width + column * cell, top, label_width + (column + 1) * cell, top + cell),
                outline=(225, 225, 225),
            )

    sheet.save(output_path)
