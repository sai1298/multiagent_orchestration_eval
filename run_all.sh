#!/usr/bin/env bash
# Project 12 reproduction. Needs ANTHROPIC_API_KEY for real runs.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

echo "=== build frozen question set ==="
$PY eval/build_questions.py

echo "=== offline pipeline check (no API key) ==="
$PY -m eval.run_comparison --backend mock

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ANTHROPIC_API_KEY not set — skipping real runs."
  exit 0
fi

echo "=== real: single vs multi (cheap workers, strong planner+judge) ==="
$PY -m eval.run_comparison --backend anthropic \
    --worker-model claude-haiku-4-5 --planner-model claude-opus-4-8 \
    --judge-model claude-opus-4-8

echo "=== ablation: critic off ==="
$PY -m eval.run_comparison --backend anthropic --no-critic \
    --worker-model claude-haiku-4-5 --planner-model claude-opus-4-8

echo "=== ablation: strong workers ==="
$PY -m eval.run_comparison --backend anthropic \
    --worker-model claude-opus-4-8 --planner-model claude-opus-4-8

echo "IMPORTANT: replace eval/gold_labels.csv with REAL human labels, then re-run"
echo "to get a defensible Cohen's kappa. See README."
