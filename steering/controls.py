"""Control images: Gaussian noise, a blank white image, and natural photos.

Every control is a 400 x 400 RGB image, the same size as the symbols.
"""

import numpy as np
from PIL import Image, ImageOps

from steering.symbols import IMAGE_SIZE

NOISE_MEAN = 128
NOISE_STD = 64


def gaussian_noise(seed: int) -> Image.Image:
    """Independent Gaussian noise in every pixel and channel, clipped to 0-255."""
    rng = np.random.default_rng(seed)
    pixels = rng.normal(NOISE_MEAN, NOISE_STD, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def blank_white() -> Image.Image:
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))


def square_photo(photo: Image.Image) -> Image.Image:
    """Crop the photo to a centered square and resize it to the symbol size."""
    return ImageOps.fit(photo.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
