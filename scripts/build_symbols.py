"""Build the symbol library and its contact sheet under stimuli/."""

from pathlib import Path

from steering.config import load_config
from steering.symbols import SHAPES, build_contact_sheet, build_library

STIMULI_DIR = Path("stimuli")
SYMBOLS_DIR = STIMULI_DIR / "symbols"
CONTACT_SHEET = STIMULI_DIR / "contact_sheet.png"


def main() -> None:
    colors = load_config()["colors"]
    build_library(colors, SYMBOLS_DIR)
    build_contact_sheet(colors, SYMBOLS_DIR, CONTACT_SHEET)
    print(f"Wrote {len(colors) * len(SHAPES)} symbols to {SYMBOLS_DIR}/")
    print(f"Wrote contact sheet to {CONTACT_SHEET}")


if __name__ == "__main__":
    main()
