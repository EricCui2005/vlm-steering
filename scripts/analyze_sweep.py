"""Analyze one model's baselines, sanity check, and 288-symbol sweep, and draw the plots.

    uv run python scripts/analyze_sweep.py gemma4_e4b [PROBLEM]

PROBLEM defaults to inorder. It picks the runs phase3_sweep_{PROBLEM} and
controls_{PROBLEM}; output files are named sweep_{PROBLEM}_*.

Reads results/runs/{model}/*/generations.jsonl and writes CSVs and PNGs to
results/{model}/.

Steering goes toward whichever strategy the model rarely uses without an image:
toward recursion if the problem's baseline is mostly iterative, toward iteration if
it's mostly recursive. delta = P(target | symbol) - P(target | no image).
Responses labeled `other` are left out of both proportions.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steering.config import load_config
from steering.stats import benjamini_hochberg, fisher_exact_two_sided, wilson_interval
from steering.symbols import SHAPES

DEFAULT_PROBLEM = "inorder"
OTHER_RATE_WARNING = 0.15
FDR_Q = 0.05


class Counts:
    """Label counts for one condition."""

    def __init__(self):
        self.recursive = self.iterative = self.other = 0

    def add(self, label: str) -> None:
        setattr(self, label, getattr(self, label) + 1)

    @property
    def valid(self) -> int:
        return self.recursive + self.iterative

    def p(self, strategy: str) -> float:
        return getattr(self, strategy) / self.valid if self.valid else float("nan")

    @property
    def other_rate(self) -> float:
        total = self.valid + self.other
        return self.other / total if total else float("nan")


def count_labels(path: Path, group_by: tuple[str, ...]) -> dict[tuple, Counts]:
    counts = defaultdict(Counts)
    if not path.exists():
        return counts
    for line in path.open():
        record = json.loads(line)
        counts[tuple(record[field] for field in group_by)].add(record["label"])
    return counts


def main(model: str, problem: str = DEFAULT_PROBLEM) -> None:
    runs_dir = Path("results/runs") / model
    output_dir = Path("results") / model
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = list(load_config()["colors"])

    baselines = {
        problem: counts
        for (problem,), counts in count_labels(runs_dir / "phase1_baselines" / "generations.jsonl", ("problem_id",)).items()
    }
    write_baseline_table(baselines, output_dir)
    plot_baselines(baselines, model, output_dir)

    sanity = count_labels(runs_dir / "sanity_text_instruction" / "generations.jsonl", ("problem_id", "condition"))
    print_sanity(sanity)

    baseline = baselines[problem]
    target = "recursive" if baseline.p("recursive") <= 0.5 else "iterative"
    symbols = {
        name: counts
        for (name,), counts in count_labels(runs_dir / f"phase3_sweep_{problem}" / "generations.jsonl", ("condition",)).items()
    }
    if not symbols:
        print("No sweep results yet.")
        return

    rows = symbol_rows(symbols, baseline, target)
    write_symbol_csv(rows, output_dir, problem)
    by_name = {row["symbol"]: row for row in rows}
    deltas = np.array([[by_name[f"{c}_{s}"]["delta"] for c in colors] for s in SHAPES])
    plot_heatmap(deltas, colors, baseline, target, model, output_dir, problem)
    plot_marginals(deltas, colors, target, model, output_dir, problem)
    print_summary(rows, baseline, target, problem)
    print_option_order_effect(runs_dir, target, problem)

    controls = {
        name: counts
        for (name,), counts in count_labels(runs_dir / f"controls_{problem}" / "generations.jsonl", ("condition",)).items()
    }
    # Some problems only have the blank image as a control.
    if len(controls) > 1:
        control_rows = symbol_rows(controls, baseline, target)
        write_control_csv(control_rows, output_dir, problem)
        plot_controls(control_rows, by_name, rows, target, model, output_dir, problem)
    if "blank_white" in controls:
        compare_with_blank_image(symbols, controls["blank_white"], colors, target, model, output_dir, problem)


def write_baseline_table(baselines: dict[str, Counts], output_dir: Path) -> None:
    with (output_dir / "baselines.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["problem", "n_valid", "n_other", "p_recursive", "ci_low", "ci_high"])
        for problem, counts in baselines.items():
            low, high = wilson_interval(counts.recursive, counts.valid)
            writer.writerow([problem, counts.valid, counts.other, f"{counts.p('recursive'):.3f}", f"{low:.3f}", f"{high:.3f}"])
    print("Baselines, P(recursive):")
    for problem, counts in baselines.items():
        print(f"  {problem:<10} {counts.p('recursive'):.2f}  (n = {counts.valid}, other = {counts.other})")


def plot_baselines(baselines: dict[str, Counts], model: str, output_dir: Path) -> None:
    problems = list(baselines)
    values = [baselines[p].p("recursive") for p in problems]
    intervals = [wilson_interval(baselines[p].recursive, baselines[p].valid) for p in problems]
    errors = [[v - low for v, (low, _) in zip(values, intervals)], [high - v for v, (_, high) in zip(values, intervals)]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(problems, values, yerr=errors, capsize=4, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(recursive)")
    ax.set_title(f"{model}: no-image baseline (95% Wilson intervals)")
    for i, problem in enumerate(problems):
        ax.text(i, 0.97, f"n={baselines[problem].valid}", ha="center", va="top", color="gray", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "baselines.png", dpi=150)
    plt.close(fig)


def print_sanity(sanity: dict[tuple, Counts]) -> None:
    if not sanity:
        return
    print("Text-instruction check, P(recursive):")
    for (problem, condition), counts in sorted(sanity.items()):
        print(f"  {problem:<8} {condition:<28} {counts.p('recursive'):.2f}  (n = {counts.valid}, other = {counts.other})")


def symbol_rows(symbols: dict[str, Counts], baseline: Counts, target: str) -> list[dict]:
    other = "iterative" if target == "recursive" else "recursive"
    rows = []
    for name, counts in sorted(symbols.items()):
        low, high = wilson_interval(getattr(counts, target), counts.valid)
        p_value = fisher_exact_two_sided(
            getattr(counts, target), getattr(counts, other), getattr(baseline, target), getattr(baseline, other)
        )
        rows.append({
            "symbol": name,
            "n_valid": counts.valid,
            "n_other": counts.other,
            f"p_{target}": counts.p(target),
            "delta": counts.p(target) - baseline.p(target),
            "ci_low": low,
            "ci_high": high,
            "fisher_p": p_value,
        })

    survives = benjamini_hochberg([row["fisher_p"] for row in rows], q=FDR_Q)
    for row, passed in zip(rows, survives):
        row["survives_fdr"] = passed
    return rows


def write_symbol_csv(rows: list[dict], output_dir: Path, problem: str) -> None:
    with (output_dir / f"sweep_{problem}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: -r["delta"]):
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def plot_heatmap(deltas, colors, baseline: Counts, target: str, model: str, output_dir: Path, problem: str,
                 reference: str = "no-image baseline", suffix: str = "") -> None:
    """Shapes as rows, sorted by mean delta (strongest at the top); colors as columns."""
    order = np.argsort(-np.nanmean(deltas, axis=1))
    shapes = [list(SHAPES)[i] for i in order]
    sorted_deltas = deltas[order]

    fig, ax = plt.subplots(figsize=(10, 11))
    image = ax.imshow(sorted_deltas, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(colors)), colors, rotation=45, ha="right")
    ax.set_yticks(range(len(shapes)), shapes)
    for row in range(len(shapes)):
        for column in range(len(colors)):
            value = sorted_deltas[row, column]
            ax.text(column, row, f"{value:+.2f}".replace("+0.00", "0"), ha="center", va="center",
                    fontsize=6.5, color="white" if abs(value) > 0.6 else "black")
    fig.colorbar(image, ax=ax, label=f"Δ P({target}) vs. {reference}", shrink=0.6)
    ax.set_title(
        f"{model}, {problem}: Δ P({target}) vs. {reference} (n = 20 per symbol)\n"
        f"Reference P({target}) = {baseline.p(target):.2f} (n = {baseline.valid})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"sweep_{problem}_heatmap{suffix}.png", dpi=150)
    plt.close(fig)


def plot_marginals(deltas, colors, target: str, model: str, output_dir: Path, problem: str) -> None:
    shape_means = np.nanmean(deltas, axis=1)
    color_means = np.nanmean(deltas, axis=0)
    shape_order = np.argsort(shape_means)
    color_order = np.argsort(color_means)

    fig, (shape_ax, color_ax) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]})
    shape_ax.barh([list(SHAPES)[i] for i in shape_order], shape_means[shape_order], color="#4C72B0")
    shape_ax.set_xlabel(f"Mean Δ P({target}) over 12 colors")
    shape_ax.set_title(f"{model}, {problem}: shape marginals")
    color_ax.barh([colors[i] for i in color_order], color_means[color_order], color="#DD8452")
    color_ax.set_xlabel(f"Mean Δ P({target}) over 24 shapes")
    color_ax.set_title("Color marginals")
    for ax in (shape_ax, color_ax):
        ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / f"sweep_{problem}_marginals.png", dpi=150)
    plt.close(fig)


def print_summary(rows: list[dict], baseline: Counts, target: str, problem: str) -> None:
    surviving = [row for row in rows if row["survives_fdr"]]
    high_other = [row["symbol"] for row in rows if row["n_other"] / max(1, row["n_valid"] + row["n_other"]) > OTHER_RATE_WARNING]
    top = sorted(rows, key=lambda r: -abs(r["delta"]))[:10]

    print(f"Sweep on {problem}, steering toward {target}. Baseline P({target}) = {baseline.p(target):.2f}")
    print(f"  Symbols surviving Benjamini-Hochberg at q = {FDR_Q}: {len(surviving)} of {len(rows)}")
    print(f"  Symbols with |delta| >= 0.3: {sum(abs(r['delta']) >= 0.3 for r in rows)}")
    print(f"  Mean delta over all symbols: {np.mean([r['delta'] for r in rows]):+.3f}")
    print(f"  Symbols with more than {OTHER_RATE_WARNING:.0%} `other`: {len(high_other)}")
    print("  Largest |delta|:")
    for row in top:
        print(f"    {row['symbol']:<28} P = {row[f'p_{target}']:.2f}  delta = {row['delta']:+.2f}  p = {row['fisher_p']:.2g}")


def compare_with_blank_image(symbols, blank: Counts, colors, target, model, output_dir, problem) -> None:
    """Measure each symbol against the blank white image instead of the no-image prompt.

    The blank image uses the same prompt as the symbols ("Looking at the provided
    image..."), so this delta isolates what the symbol itself adds.
    """
    rows = symbol_rows(symbols, blank, target)
    by_name = {row["symbol"]: row for row in rows}
    deltas = np.array([[by_name[f"{c}_{s}"]["delta"] for c in colors] for s in SHAPES])

    plot_heatmap(deltas, colors, blank, target, model, output_dir, problem,
                 reference="blank white image", suffix="_vs_blank")
    surviving = [row for row in rows if row["survives_fdr"]]
    print(f"Symbols vs. the blank white image (P({target}) = {blank.p(target):.2f}, n = {blank.valid}):")
    print(f"  Surviving Benjamini-Hochberg at q = {FDR_Q}: {len(surviving)} of {len(rows)} "
          f"({sum(r['delta'] > 0 for r in surviving)} toward {target}, {sum(r['delta'] < 0 for r in surviving)} away)")
    print(f"  |delta| >= 0.3: {sum(abs(r['delta']) >= 0.3 for r in rows)}")
    for row in sorted(rows, key=lambda r: -abs(r["delta"]))[:8]:
        print(f"    {row['symbol']:<28} delta = {row['delta']:+.2f}  p = {row['fisher_p']:.2g}")


def write_control_csv(control_rows: list[dict], output_dir: Path, problem: str) -> None:
    with (output_dir / f"controls_{problem}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(control_rows[0]))
        writer.writeheader()
        for row in control_rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def plot_controls(control_rows, symbols_by_name, symbol_rows_all, target, model, output_dir, problem) -> None:
    """Each control's delta next to the symbols it should be compared with.

    Captions are paired with the same symbol shown as an image. The mean over all
    288 symbols is the reference for the blank, noise and photo controls.
    """
    by_name = {row["symbol"]: row for row in control_rows}
    mean_symbol_delta = float(np.mean([row["delta"] for row in symbol_rows_all]))

    def group_mean(prefix: str) -> float:
        return float(np.mean([row["delta"] for name, row in by_name.items() if name.startswith(prefix)]))

    print(f"Controls on {problem}, delta P({target}) vs. no image (n = 40 each):")
    print(f"  mean over all 288 symbols (n = 20 each)   {mean_symbol_delta:+.2f}")
    print(f"  blank_white                               {by_name['blank_white']['delta']:+.2f}")
    print(f"  noise, mean of 5                          {group_mean('noise_'):+.2f}  "
          f"(range {min(by_name[f'noise_{i}']['delta'] for i in range(1, 6)):+.2f} to {max(by_name[f'noise_{i}']['delta'] for i in range(1, 6)):+.2f})")
    print(f"  photos, mean of 5                         {group_mean('photo_'):+.2f}  "
          f"(range {min(by_name[f'photo_{i}']['delta'] for i in range(1, 6)):+.2f} to {max(by_name[f'photo_{i}']['delta'] for i in range(1, 6)):+.2f})")
    print("  captions vs. the same symbol as an image:")
    captions = sorted(name for name in by_name if name.startswith("caption_"))
    for name in captions:
        symbol = name.removeprefix("caption_")
        print(f"    {symbol:<22} caption {by_name[name]['delta']:+.2f}   image {symbols_by_name[symbol]['delta']:+.2f}")

    labels = ["all symbols\n(mean)", "blank white"]
    values = [mean_symbol_delta, by_name["blank_white"]["delta"]]
    colors = ["#4C72B0", "#999999"]
    for i in range(1, 6):
        labels.append(f"noise {i}"); values.append(by_name[f"noise_{i}"]["delta"]); colors.append("#BBBBBB")
    for i in range(1, 6):
        labels.append(f"photo {i}"); values.append(by_name[f"photo_{i}"]["delta"]); colors.append("#8C8C5A")

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [12, 8]})
    left.bar(range(len(values)), values, color=colors)
    left.set_xticks(range(len(values)), labels, rotation=45, ha="right")
    left.set_ylabel(f"Δ P({target}) vs. no-image baseline")
    left.set_title(f"{model}: images without a symbol")

    x = np.arange(len(captions))
    image_deltas = [symbols_by_name[name.removeprefix("caption_")]["delta"] for name in captions]
    caption_deltas = [by_name[name]["delta"] for name in captions]
    right.bar(x - 0.2, image_deltas, width=0.4, label="symbol as image (n = 20)", color="#4C72B0")
    right.bar(x + 0.2, caption_deltas, width=0.4, label="symbol as caption (n = 40)", color="#DD8452")
    right.set_xticks(x, [name.removeprefix("caption_") for name in captions], rotation=45, ha="right")
    right.set_title("Image vs. text caption")
    right.legend(fontsize=8)

    for ax in (left, right):
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / f"controls_{problem}.png", dpi=150)
    plt.close(fig)


def print_option_order_effect(runs_dir: Path, target: str, problem: str) -> None:
    """P(target) separately for each option order, for the baseline and the sweep."""
    print(f"  P({target}) by option order (which strategy the prompt lists first):")
    for run_name, problem_filter in (("phase1_baselines", problem), (f"phase3_sweep_{problem}", None)):
        path = runs_dir / run_name / "generations.jsonl"
        counts = count_labels(path, ("problem_id", "option_order"))
        for (problem_id, order), c in sorted(counts.items()):
            if problem_filter is None or problem_id == problem_filter:
                print(f"    {run_name:<22} {order:<16} {c.p(target):.3f}  (n = {c.valid})")


if __name__ == "__main__":
    main(*sys.argv[1:])
