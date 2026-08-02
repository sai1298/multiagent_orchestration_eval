"""Judge calibration — the crux of the project.

Computes judge-vs-human agreement per rubric axis: Cohen's kappa and raw
accuracy. The output tells you which axes the judge is trustworthy on; the
head-to-head comparison then relies on the judge ONLY for those axes.

`eval/gold_labels.csv` holds the human gold labels — the calibration ground
truth. Columns: question_id, answer_hash, coverage, grounding, structure,
no_halluc (each 0/1). **These must be real human labels for the number to mean
anything** — the file shipped here is a synthetic, clearly-marked placeholder so
the pipeline runs offline; replace it with your own labels before quoting kappa.
"""

from __future__ import annotations

import csv
import os

AXES = ["coverage", "grounding", "structure", "no_halluc"]


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two binary label sequences."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # marginal probabilities
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def load_gold(path: str) -> dict:
    gold = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["question_id"], row["answer_hash"])
            gold[key] = {ax: int(row[ax]) for ax in AXES}
    return gold


def calibrate(gold: dict, judge_scores: dict) -> dict:
    """gold and judge_scores keyed by (question_id, answer_hash) -> {axis: 0/1}."""
    keys = [k for k in gold if k in judge_scores]
    report = {"n": len(keys), "axes": {}}
    trustworthy = []
    for ax in AXES:
        h = [gold[k][ax] for k in keys]
        j = [judge_scores[k][ax] for k in keys]
        kappa = cohen_kappa(h, j)
        acc = sum(1 for x, y in zip(h, j) if x == y) / len(keys) if keys else float("nan")
        # convention: kappa >= 0.6 is "substantial" agreement -> trustworthy
        ok = kappa >= 0.6
        report["axes"][ax] = {"cohen_kappa": kappa, "accuracy": acc,
                              "trustworthy": ok}
        if ok:
            trustworthy.append(ax)
    report["trustworthy_axes"] = trustworthy
    return report


def answer_hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _write_placeholder_gold(path, rows):
    """Generate a clearly-marked synthetic gold set so the pipeline runs offline.
    Replace with real human labels."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "answer_hash", *AXES])
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    # quick self-test of the kappa function on a known case
    a = [1, 1, 0, 0, 1, 0, 1, 1]
    b = [1, 0, 0, 0, 1, 1, 1, 1]
    print("[calibration] kappa self-test:", round(cohen_kappa(a, b), 3))
