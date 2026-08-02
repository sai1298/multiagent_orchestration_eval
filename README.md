# Multi-Agent Orchestration + Observability + Calibrated LLM-as-Judge

A multi-agent system (planner/supervisor → specialist workers → critic) with
**per-agent cost/latency/token tracing**, evaluated by an **LLM-as-judge that is
calibrated against human gold labels**. The headline is an honest
**single-agent vs multi-agent** comparison that reports not just quality but the
cost and latency you paid for it — i.e. *when orchestration actually earns its
keep and when it just burns tokens*.

The rigor twist — **judge calibration** and a cost-aware quality tradeoff — is
what separates this from the CrewAI-demo projects that flood every portfolio.

> **Status:** complete and verified end-to-end in **offline mock mode** (agents,
> tracer, judge, calibration, single-vs-multi comparison, reliability — all
> runnable with no API key). **Real numbers need `ANTHROPIC_API_KEY`**; tables
> are marked `[fill after running]`. The **Cohen's kappa must be computed
> against real human labels** — the shipped `gold_labels.csv` is a
> clearly-marked synthetic placeholder (see below).

---

## What's built

| Component | File | What it does |
|---|---|---|
| Single-agent baseline | `agents/single_agent.py` | One strong model + scratchpad — the honest control |
| Multi-agent graph | `agents/graph.py` | planner → per-dimension workers → synth → critic (1 revision), with explicit deadlock/loop/cost caps |
| Observability | `observability/tracer.py` | Nested spans; token + cost + latency attributed **per agent role** |
| Judge | `eval/judge.py` | Rubric LLM-as-judge, 4 binary axes, structured JSON output, versioned prompt |
| Calibration | `eval/calibration.py` | Cohen's kappa + per-axis accuracy vs human gold |
| Comparison | `eval/run_comparison.py` | Head-to-head: quality (trusted axes) × cost × latency × reliability |
| Question set | `eval/questions_v1.jsonl` | 30 frozen comparison questions, each with 3 named dimensions |

Model policy (spec): **cheap workers** (`claude-haiku-4-5`), **strong
planner/critic/judge** (`claude-opus-4-8`). Real Anthropic model IDs; cost from
real token usage.

---

## Verified offline (mock backend, no API key)

```
$ python -m eval.run_comparison --backend mock
[compare] judge calibration (kappa vs 50 gold labels):
[compare]   coverage   kappa=+1.000 acc=1.000 TRUSTED
[compare]   grounding  kappa=+0.000 acc=0.680 untrusted     ← flagged, excluded
[compare]   structure  kappa=+1.000 acc=1.000 TRUSTED
[compare]   no_halluc  kappa=+1.000 acc=1.000 TRUSTED
[compare] trusted axes: ['coverage', 'structure', 'no_halluc']
[compare] cost: single=$0.00106 multi=$0.00379  (3.59x)  tokens 7.25x
[compare] reliability: multi completed clean 70%, critic triggered 9/30
```

The pipeline correctly **drops the `grounding` axis** (kappa below the 0.6
"substantial agreement" threshold) and reports quality only on the axes the
judge is trustworthy on — the whole point of calibration. It also surfaces a
real orchestration pathology in the mock (a deterministic critic looping on an
unchanged draft until the revision cap) — exactly the kind of reliability issue
the completion-rate metric exists to catch.

---

## The calibration step (the crux — read this)

`eval/calibration.py` computes judge↔human agreement per rubric axis. The
comparison then trusts the judge **only** on axes where kappa ≥ 0.6.

**You must hand-label a real gold set for the kappa to mean anything.** The
`gold_labels.csv` shipped/auto-generated here is a synthetic placeholder (it
deliberately disagrees with the judge on one axis so the pipeline demonstrates
an untrusted axis). Before quoting a kappa on a résumé:

1. Run a real pass to produce answers.
2. Hand-score ~60–100 of them yourself on the four axes.
3. Overwrite `gold_labels.csv` with your labels (`question_id, answer_hash,
   coverage, grounding, structure, no_halluc`).
4. Re-run — the reported kappa is now real.

A judge you haven't checked against human labels is not evidence, and the
project is built to make that check the centerpiece rather than skip it.

---

## Metrics (fill from a real run)

| | Single-agent | Multi-agent |
|---|---|---|
| Quality (calibrated judge, trusted axes) | `[fill]` | `[fill]` |
| Avg cost / question | `[fill]` | `[fill]` |
| Avg tokens / question | `[fill]` | `[fill]` |
| Latency p50 / p95 | `[fill]` | `[fill]` |

- **Quality lift** (multi − single, trusted axes): `[fill]`
- **Cost multiple** (multi / single): `[fill]`× — *is the lift worth it?*
- **Judge calibration**: Cohen's kappa per axis `[fill]`; trusted axes `[fill]`
- **Reliability**: `[fill]`% of multi-agent runs completed clean vs hit a
  deadlock / revision-cap / cost-cap
- **Ablations**: critic on/off; cheap vs strong workers — each with the quality
  delta *and* the cost delta.

Per-agent cost attribution (the observability story) is in
`results/traces_*.json` → `by_role`.

---

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
PY=.venv/bin/python ./run_all.sh
# then hand-label gold_labels.csv and re-run for a real kappa
```

Offline (no key): `.venv/bin/python -m eval.run_comparison --backend mock`.

---

## Résumé bullet (fill from your run)

> Designed a multi-agent system (planner/worker/critic) with full per-agent
> cost + latency tracing, and a calibrated LLM-as-judge evaluation (Cohen's
> kappa **[K]** vs **[N]** human-labeled outputs); showed multi-agent lifted
> quality **[X]%** but at **[Y]×** the token cost, quantifying exactly when
> orchestration is worth it over a single agent.

---

## Honest notes

- **The single-agent baseline is built to be strong**, not a strawman. If
  multi-agent doesn't beat it, that finding *is* the result.
- **Caps are enforced in code** (total steps, one revision, cost) and the
  harness counts how often each fires — orchestration graphs deadlock and loop.
- **Judge prompt is versioned** and model IDs + date are recorded; judge
  behavior drifts across model updates, so uncontrolled configs make the numbers
  meaningless.
- Framework note: the graph is hand-rolled (transparent, zero framework
  dependency) but mirrors a LangGraph supervisor pattern — the same nodes and
  edges, so the design maps directly onto LangGraph if a framework is required.
