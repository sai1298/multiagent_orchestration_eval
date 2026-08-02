"""Single-agent baseline — the honest control the multi-agent system must beat.

Deliberately a *strong* single agent: one capable model, a scratchpad, and an
explicit instruction to cover every rubric axis. If multi-agent can't beat a
good single agent, that finding is the result — so the baseline is not a
strawman.
"""

from __future__ import annotations

from observability.tracer import Trace

SINGLE_SYSTEM = (
    "You are an expert research analyst. Answer the question thoroughly. "
    "You MUST: (1) define key terms, (2) compare across every named dimension, "
    "(3) cite sources inline as [source], (4) end with a synthesis. Do not "
    "hallucinate sources."
)


def run_single(chat, question: dict, model: str) -> tuple[str, Trace]:
    trace = Trace(question_id=question["id"], system="single")
    prompt = _format_question(question)
    with trace.span("single", "single", model) as s:
        text, tin, tout = chat.complete(model=model, system=SINGLE_SYSTEM,
                                        prompt=prompt, max_tokens=1200)
        s.input_tokens, s.output_tokens = tin, tout
    return text, trace


def _format_question(q: dict) -> str:
    dims = ", ".join(q["dimensions"])
    return (f"Question: {q['question']}\n"
            f"Compare specifically across these dimensions: {dims}.\n"
            f"Cite sources.")
