"""Multi-agent orchestration: supervisor/planner -> specialist workers -> critic.

A from-scratch supervisor graph (mirrors a LangGraph supervisor pattern; kept
hand-rolled so the control flow is transparent and the project runs with no
framework dependency). The graph:

    planner  ── decomposes the question into per-dimension subtasks
       │
    workers  ── one worker per dimension, run independently (cheap model)
       │
    synth    ── planner-model merges worker outputs into a draft
       │
    critic   ── checks the draft; may trigger ONE revision loop
       │
    (revise) ── synth re-run with the critic's notes, then done

Explicit failure-mode handling (the spec calls these out):
  * worker returns empty          -> substituted with a "[no content]" marker,
                                     surfaced to the critic
  * infinite critic<->worker loop -> hard cap of 1 revision
  * total step / cost caps        -> enforced; `completed_clean` records whether
                                     the run finished normally vs hit a cap
"""

from __future__ import annotations

from observability.tracer import Trace

PLANNER_SYS = "You are a planning supervisor. Produce a short plan; do not answer."
WORKER_SYS = ("You are a specialist analyst. Answer ONLY the assigned sub-question "
              "for one dimension, concisely, citing a source as [source].")
SYNTH_SYS = ("You are a synthesis writer. Merge the worker findings into a single "
             "coherent answer: define terms, cover each dimension, cite sources, "
             "end with a synthesis. Use only the workers' content.")
CRITIC_SYS = ("You are a critic. Given a draft answer and the required dimensions, "
              "reply with exactly 'OK' if it covers every dimension with a source "
              "and a synthesis, otherwise reply 'REVISE:' followed by what's missing.")


def run_multi(chat, question: dict, *, worker_model: str, planner_model: str,
              max_revisions: int = 1, max_cost: float = 0.20) -> tuple[str, Trace, dict]:
    trace = Trace(question_id=question["id"], system="multi")
    meta = {"revisions": 0, "critic_triggered": False, "completed_clean": True,
            "empty_workers": 0, "stop": "end"}

    # 1) planner
    with trace.span("planner", "planner", planner_model) as s:
        plan, tin, tout = chat.complete(
            model=planner_model, system=PLANNER_SYS,
            prompt=f"Question: {question['question']}\nDimensions: "
                   f"{', '.join(question['dimensions'])}\nProduce a plan.",
            max_tokens=300)
        s.input_tokens, s.output_tokens = tin, tout

    # 2) workers — one per dimension
    worker_outputs = []
    for dim in question["dimensions"]:
        with trace.span(f"worker:{dim}", "worker", worker_model) as s:
            out, tin, tout = chat.complete(
                model=worker_model, system=WORKER_SYS,
                prompt=f"Question: {question['question']}\nYour dimension: {dim}\n"
                       f"Answer only for this dimension.", max_tokens=400)
            s.input_tokens, s.output_tokens = tin, tout
        if not out.strip():
            out = "[no content]"
            meta["empty_workers"] += 1
        worker_outputs.append(f"[{dim}] {out}")
        if trace.total_cost() > max_cost:
            meta["completed_clean"] = False
            meta["stop"] = "cost_cap"
            return "\n".join(worker_outputs), trace, meta

    # 3) synth
    draft = _synth(chat, question, worker_outputs, planner_model, trace, note="")

    # 4) critic (+ optional single revision)
    for _ in range(max_revisions + 1):
        with trace.span("critic", "critic", planner_model) as s:
            verdict, tin, tout = chat.complete(
                model=planner_model, system=CRITIC_SYS,
                prompt=f"Dimensions required: {', '.join(question['dimensions'])}\n\n"
                       f"Draft:\n{draft}", max_tokens=200)
            s.input_tokens, s.output_tokens = tin, tout
        if verdict.strip().upper().startswith("OK"):
            break
        # critic asked for a revision
        meta["critic_triggered"] = True
        meta["revisions"] += 1
        if meta["revisions"] > max_revisions:
            meta["completed_clean"] = False
            meta["stop"] = "revision_cap"
            break
        if trace.total_cost() > max_cost:
            meta["completed_clean"] = False
            meta["stop"] = "cost_cap"
            break
        draft = _synth(chat, question, worker_outputs, planner_model, trace,
                       note=verdict)

    return draft, trace, meta


def _synth(chat, question, worker_outputs, model, trace, note):
    with trace.span("synth", "planner", model) as s:
        prompt = (f"Question: {question['question']}\n"
                  f"Worker findings:\n" + "\n".join(worker_outputs))
        if note:
            prompt += f"\n\nRevise to address the critic: {note}"
        text, tin, tout = chat.complete(model=model, system=SYNTH_SYS,
                                        prompt=prompt, max_tokens=1200)
        s.input_tokens, s.output_tokens = tin, tout
    return text
