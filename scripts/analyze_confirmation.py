"""Analyze an n=100 run of pre-chosen symbols against symbol-free controls.

    PYTHONPATH=scripts uv run python scripts/analyze_confirmation.py MODEL [RUN]

RUN defaults to confirmation_inorder. For a choose_strategy run, where the model
only names its strategy, the script also prints the rates from writing the code.

Controls are the no-image baseline and images without a symbol (blank, noise,
photos). Each symbol is compared with the blank image and with the control that
comes closest to it in the symbol's own direction (the highest P(iterative) for a
symbol that raises it, the lowest for one that lowers it). Captions are listed for
comparison but aren't used as controls. The significance threshold is
Bonferroni-corrected for the number of symbols: 0.05 / 5 = 0.01 with five.
Writes results/{model}/{RUN}.csv and .png.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_sweep import count_labels
from steering.config import load_config
from steering.samples import BASELINE, CAPTION_PREFIX, CHOOSE_STRATEGY
from steering.stats import fisher_exact_two_sided, wilson_interval

CONFIRMATION_RUN = "confirmation_inorder"
TARGET = "iterative"
IMAGE_CONTROL_PREFIXES = ("blank_", "noise_", "photo_")


def is_control(condition: str) -> bool:
    return condition == BASELINE or condition.startswith(IMAGE_CONTROL_PREFIXES)


def main(model: str, run: str = CONFIRMATION_RUN) -> None:
    run_config = load_config()["runs"][run]
    conditions = run_config["conditions"]
    controls = [c for c in conditions if is_control(c)]
    captions = [c for c in conditions if c.startswith(CAPTION_PREFIX)]
    symbols = [c for c in conditions if c not in controls and c not in captions]
    alpha = 0.05 / len(symbols)
    print(f"alpha = {alpha}\n")

    path = Path(f"results/runs/{model}/{run}/generations.jsonl")
    counts = {key[0]: value for key, value in count_labels(path, ("condition",)).items()}
    by_order = count_labels(path, ("condition", "option_order"))
    blank = counts["blank_white"]

    def p_target(condition: str) -> float:
        return counts[condition].p(TARGET)

    rows = []
    for condition in symbols + captions + controls:
        c = counts[condition]
        low, high = wilson_interval(getattr(c, TARGET), c.valid)
        row = {"condition": condition, "n_valid": c.valid, "n_other": c.other,
               "p_iterative": c.p(TARGET), "ci_low": low, "ci_high": high}
        for order in ("recursion_first", "iteration_first"):
            row[f"p_{order}"] = by_order[(condition, order)].p(TARGET)
        if condition in symbols:
            raises = c.p(TARGET) >= blank.p(TARGET)
            closest = (max if raises else min)(controls, key=p_target)
            row["closest_control"] = closest
            for name, reference in [("blank", blank), ("closest_control", counts[closest])]:
                row[f"delta_vs_{name}"] = c.p(TARGET) - reference.p(TARGET)
                row[f"p_vs_{name}"] = fisher_exact_two_sided(
                    getattr(c, TARGET), c.valid - getattr(c, TARGET),
                    getattr(reference, TARGET), reference.valid - getattr(reference, TARGET))
            # Passing needs both differences significant and in the symbol's direction:
            # beating the closest control means going past it, not falling short of it.
            same_direction = (row["delta_vs_closest_control"] > 0) == raises
            row["passes"] = row["p_vs_blank"] < alpha and row["p_vs_closest_control"] < alpha and same_direction
        rows.append(row)

    print_table(rows)
    if run_config.get("task") == CHOOSE_STRATEGY:
        print_code_rates(model, rows)
    output_dir = Path(f"results/{model}")
    write_csv(rows, output_dir / f"{run}.csv")
    plot(rows, symbols, model, run, output_dir / f"{run}.png")


def print_code_rates(model: str, rows: list[dict]) -> None:
    """Put each condition's stated choice next to the code it wrote in the confirmation run."""
    path = Path(f"results/runs/{model}/{CONFIRMATION_RUN}/generations.jsonl")
    code_counts = {key[0]: value for key, value in count_labels(path, ("condition",)).items()}
    print(f"\n{'condition':22} {'stated':>6} {'code':>6}")
    for r in rows:
        code = code_counts[r["condition"]].p(TARGET) if r["condition"] in code_counts else float("nan")
        print(f"{r['condition']:22} {r['p_iterative']:6.2f} {code:6.2f}")


def print_table(rows: list[dict]) -> None:
    print(f"{'condition':32} {'p_iter':>6} {'95% CI':>13} {'rec1st':>6} {'it1st':>6} "
          f"{'Δblank':>7} {'p':>8}  {'closest control':>15} {'Δ':>6} {'p':>8}  pass")
    for r in rows:
        line = (f"{r['condition']:32} {r['p_iterative']:6.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] "
                f"{r['p_recursion_first']:6.2f} {r['p_iteration_first']:6.2f}")
        if "passes" in r:
            line += (f" {r['delta_vs_blank']:+7.2f} {r['p_vs_blank']:8.1e}  {r['closest_control']:>15} "
                     f"{r['delta_vs_closest_control']:+6.2f} {r['p_vs_closest_control']:8.1e}  {r['passes']}")
        print(line)


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], symbols: list[str], model: str, run: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, r in enumerate(rows):
        if r["condition"] in symbols:
            color = "#3b6fb6"
        elif r["condition"].startswith(CAPTION_PREFIX):
            color = "#dd8452"
        else:
            color = "#999999"
        ax.errorbar(i, r["p_iterative"], yerr=[[max(0, r["p_iterative"] - r["ci_low"])], [max(0, r["ci_high"] - r["p_iterative"])]],
                    fmt="o", color=color, capsize=3)
    ax.set_xticks(range(len(rows)), [r["condition"] for r in rows], rotation=45, ha="right")
    ax.set_ylabel("P(iterative), 95% Wilson CI")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model}: {run}, n=100 per condition\n(blue = symbols, orange = captions, grey = controls)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:])
