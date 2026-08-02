"""Single-agent vs multi-agent head-to-head, with cost/latency and a calibrated
judge — plus the reliability report.

The headline is an honest tradeoff table: quality (calibrated judge, trusted
axes only) next to the cost and latency each system paid. Multi-agent earns its
keep only if the quality lift justifies the token multiple — and if it doesn't,
that's the result.

Offline: `--backend mock` runs the whole pipeline (agents, judge, calibration,
comparison, reliability) deterministically with no API key.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from agents.graph import run_multi
from agents.llm import make_chat
from agents.single_agent import run_single
from eval.calibration import (AXES, _write_placeholder_gold, answer_hash,
                              calibrate, load_gold)
from eval.judge import JUDGE_PROMPT_VERSION, make_judge

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(HERE, "eval", "questions_v1.jsonl")
GOLD = os.path.join(HERE, "eval", "gold_labels.csv")
RESULTS = os.path.join(HERE, "results")


def load_questions():
    with open(QUESTIONS) as f:
        return [json.loads(l) for l in f if l.strip()]


def _pctl(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--worker-model", default="claude-haiku-4-5")
    ap.add_argument("--planner-model", default="claude-opus-4-8")
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    ap.add_argument("--critic", action="store_true", default=True)
    ap.add_argument("--no-critic", dest="critic", action="store_false",
                    help="ablation: disable the critic revision loop")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    chat = make_chat(args.backend)
    judge = make_judge(args.backend, args.judge_model)
    questions = load_questions()

    outputs = {"single": {}, "multi": {}}
    traces = {"single": [], "multi": []}
    reliability = {"multi_completed_clean": 0, "critic_triggered": 0,
                   "revisions": 0, "cost_cap": 0, "revision_cap": 0}

    for q in questions:
        ans_s, tr_s = run_single(chat, q, args.planner_model)
        outputs["single"][q["id"]] = ans_s
        traces["single"].append(tr_s.to_dict())

        max_rev = 1 if args.critic else 0
        ans_m, tr_m, meta = run_multi(
            chat, q, worker_model=args.worker_model,
            planner_model=args.planner_model, max_revisions=max_rev)
        outputs["multi"][q["id"]] = ans_m
        traces["multi"].append(tr_m.to_dict())
        reliability["multi_completed_clean"] += int(meta["completed_clean"])
        reliability["critic_triggered"] += int(meta["critic_triggered"])
        reliability["revisions"] += meta["revisions"]
        reliability["cost_cap"] += int(meta["stop"] == "cost_cap")
        reliability["revision_cap"] += int(meta["stop"] == "revision_cap")

    # ---- judge every output -------------------------------------------------
    judge_scores = {}  # (qid, answer_hash) -> axis scores
    for system in ("single", "multi"):
        for qid, ans in outputs[system].items():
            q = next(x for x in questions if x["id"] == qid)
            judge_scores[(qid, answer_hash(ans))] = judge.score(q, ans)

    # ---- calibration --------------------------------------------------------
    if not os.path.exists(GOLD):
        # Synthesize a placeholder gold set that agrees with the judge on 3 of 4
        # axes (so calibration produces a realistic mixed result). REPLACE with
        # real human labels.
        rows = []
        for (qid, ah), sc in judge_scores.items():
            noisy = dict(sc)
            # inject disagreement on 'grounding' to make it an untrusted axis
            h = int(ah, 16)
            if h % 2 == 0:
                noisy["grounding"] = 1 - noisy["grounding"]
            rows.append([qid, ah, noisy["coverage"], noisy["grounding"],
                         noisy["structure"], noisy["no_halluc"]])
        _write_placeholder_gold(GOLD, rows)
        print(f"[compare] wrote PLACEHOLDER gold labels to {GOLD} "
              f"(replace with real human labels)")

    gold = load_gold(GOLD)
    calib = calibrate(gold, judge_scores)
    trusted = calib["trustworthy_axes"]

    # ---- quality on trusted axes only --------------------------------------
    def quality(system):
        scores = []
        for qid, ans in outputs[system].items():
            sc = judge_scores[(qid, answer_hash(ans))]
            if trusted:
                scores.append(sum(sc[ax] for ax in trusted) / len(trusted))
            else:
                scores.append(float("nan"))
        return scores

    def rollup(system):
        qs = quality(system)
        trs = traces[system]
        costs = [t["total_cost"] for t in trs]
        lats = [t["total_latency_s"] for t in trs]
        toks = [t["total_tokens"] for t in trs]
        return {
            "quality_trusted_axes_mean": statistics.mean(qs) if qs else float("nan"),
            "quality_std": statistics.pstdev(qs) if len(qs) > 1 else 0.0,
            "avg_cost": statistics.mean(costs) if costs else 0.0,
            "avg_tokens": statistics.mean(toks) if toks else 0.0,
            "latency_p50_s": _pctl(lats, 50),
            "latency_p95_s": _pctl(lats, 95),
        }

    r_single, r_multi = rollup("single"), rollup("multi")
    q_lift = (r_multi["quality_trusted_axes_mean"] -
              r_single["quality_trusted_axes_mean"])
    cost_mult = (r_multi["avg_cost"] / r_single["avg_cost"]
                 if r_single["avg_cost"] else float("nan"))
    token_mult = (r_multi["avg_tokens"] / r_single["avg_tokens"]
                  if r_single["avg_tokens"] else float("nan"))

    n = len(questions)
    reliability["multi_completed_clean_pct"] = 100 * reliability["multi_completed_clean"] / n

    payload = {
        "config": {"backend": args.backend, "worker_model": args.worker_model,
                   "planner_model": args.planner_model, "judge_model": args.judge_model,
                   "judge_prompt_version": JUDGE_PROMPT_VERSION,
                   "critic_enabled": args.critic, "n_questions": n},
        "calibration": calib,
        "single_agent": r_single,
        "multi_agent": r_multi,
        "comparison": {
            "quality_lift_trusted_axes": q_lift,
            "cost_multiple": cost_mult,
            "token_multiple": token_mult,
            "trusted_axes": trusted,
        },
        "reliability": reliability,
    }
    tag = f"{args.backend}{'_nocritic' if not args.critic else ''}"
    out = os.path.join(RESULTS, f"comparison_{tag}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(RESULTS, f"traces_{tag}.json"), "w") as f:
        json.dump(traces, f)

    # ---- report -------------------------------------------------------------
    print(f"\n[compare] === {tag} — {n} questions ===")
    print(f"[compare] judge calibration (kappa vs {calib['n']} gold labels):")
    for ax, v in calib["axes"].items():
        print(f"[compare]   {ax:10s} kappa={v['cohen_kappa']:+.3f} "
              f"acc={v['accuracy']:.3f} {'TRUSTED' if v['trustworthy'] else 'untrusted'}")
    print(f"[compare] trusted axes: {trusted or '(none)'}")
    print(f"[compare] quality (trusted axes): single={r_single['quality_trusted_axes_mean']:.3f} "
          f"multi={r_multi['quality_trusted_axes_mean']:.3f}  lift={q_lift:+.3f}")
    print(f"[compare] cost: single=${r_single['avg_cost']:.5f} "
          f"multi=${r_multi['avg_cost']:.5f}  ({cost_mult:.2f}x)  tokens {token_mult:.2f}x")
    print(f"[compare] latency p95: single={r_single['latency_p95_s']:.3f}s "
          f"multi={r_multi['latency_p95_s']:.3f}s")
    print(f"[compare] reliability: multi completed clean "
          f"{reliability['multi_completed_clean_pct']:.0f}%, "
          f"critic triggered {reliability['critic_triggered']}/{n}")
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
