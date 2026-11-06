#!/usr/bin/env bash
# Phase 0, from start to finish. This creates Modal Volumes and spends money
# (under $1), and it's safe to run again: finished generations are skipped.
set -euo pipefail

echo "== 1. Check Modal access =="
uv run modal profile current

echo "== 2. Create the Volumes if they don't exist =="
existing_volumes=$(uv run modal volume list --json)
for volume in hf-cache steering-results; do
    if ! grep -q "\"$volume\"" <<< "$existing_volumes"; then
        uv run modal volume create "$volume"
    fi
done

echo "== 3. Build the symbols and upload them =="
uv run python scripts/build_symbols.py
uv run modal volume put --force steering-results stimuli/symbols /stimuli/symbols

echo "== 4. Run the smoke test on a GPU =="
uv run modal run modal_app.py::smoke_test

echo "== 5. Download the results and write the summary =="
mkdir -p results/runs
uv run modal volume get --force steering-results /runs/phase0_smoke results/runs/
uv run python scripts/report_phase0.py
