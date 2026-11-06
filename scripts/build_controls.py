"""Build the control images under stimuli/controls/.

    uv run python scripts/build_controls.py PHOTO_DIR

PHOTO_DIR holds the downloaded photos named in PHOTOS below. They come from the
Flickr 1k test set on Hugging Face (nlphuji/flickr_1k_test_image_text_retrieval),
rows 0, 200, 400, 600 and 800. Their sources are recorded in stimuli/controls/sources.json.
"""

import json
import sys
from pathlib import Path

from PIL import Image

from steering.controls import blank_white, gaussian_noise, square_photo

OUTPUT_DIR = Path("stimuli/controls")
NOISE_SEEDS = [1, 2, 3, 4, 5]
PHOTOS = {
    "photo_1": ("photo_0.jpg", "1007129816.jpg", "Man with glasses and an orange hat"),
    "photo_2": ("photo_200.jpg", "2504764590.jpg", "Person throwing a stick into water for a dog"),
    "photo_3": ("photo_400.jpg", "3283368342.jpg", "Four people with three prize-winning dogs"),
    "photo_4": ("photo_600.jpg", "430173345.jpg", "Brown dog with a purple object in a field"),
    "photo_5": ("photo_800.jpg", "4923272678.jpg", "Man in a black hat riding a bicycle"),
}


def main(photo_dir: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    blank_white().save(OUTPUT_DIR / "blank_white.png")
    for index, seed in enumerate(NOISE_SEEDS, start=1):
        gaussian_noise(seed).save(OUTPUT_DIR / f"noise_{index}.png")

    sources = {}
    for name, (downloaded_file, flickr_file, description) in PHOTOS.items():
        square_photo(Image.open(photo_dir / downloaded_file)).save(OUTPUT_DIR / f"{name}.png")
        sources[name] = {
            "dataset": "nlphuji/flickr_1k_test_image_text_retrieval",
            "flickr_filename": flickr_file,
            "description": description,
        }
    (OUTPUT_DIR / "sources.json").write_text(json.dumps(sources, indent=2))
    print(f"Wrote 11 control images to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
