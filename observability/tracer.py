"""Minimal OpenTelemetry-style tracer for per-agent span attribution.

The observability story this project sells is being able to attribute tokens,
cost, and latency **per agent role** (planner / worker / critic), not just per
run. Langfuse / Phoenix would give a nicer UI, but the data model is the same:
nested spans, each carrying token + cost + latency attributes. This is that
data model, self-contained and exportable to JSON so it can be diffed and
charted without a hosted backend.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

PRICING = {  # $ per 1M tokens (input, output) — Anthropic, Jul 2026
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
    "mock": (0.0, 0.0),
}


def cost_usd(model, input_tokens, output_tokens):
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


@dataclass
class Span:
    name: str          # e.g. "planner", "worker:compare", "critic"
    role: str          # planner | worker | critic | single
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    question_id: str
    system: str        # "single" | "multi"
    spans: list = field(default_factory=list)

    @contextmanager
    def span(self, name: str, role: str, model: str):
        s = Span(name=name, role=role, model=model)
        t0 = time.perf_counter()
        try:
            yield s
        finally:
            s.latency_s = time.perf_counter() - t0
            s.cost = cost_usd(s.model, s.input_tokens, s.output_tokens)
            self.spans.append(s)

    # ---- roll-ups -----------------------------------------------------------
    def total_cost(self):
        return sum(s.cost for s in self.spans)

    def total_tokens(self):
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    def total_latency(self):
        return sum(s.latency_s for s in self.spans)

    def by_role(self):
        agg = {}
        for s in self.spans:
            r = agg.setdefault(s.role, {"cost": 0.0, "tokens": 0, "latency_s": 0.0,
                                        "calls": 0})
            r["cost"] += s.cost
            r["tokens"] += s.input_tokens + s.output_tokens
            r["latency_s"] += s.latency_s
            r["calls"] += 1
        return agg

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "system": self.system,
            "total_cost": self.total_cost(),
            "total_tokens": self.total_tokens(),
            "total_latency_s": self.total_latency(),
            "by_role": self.by_role(),
            "spans": [s.__dict__ for s in self.spans],
        }
