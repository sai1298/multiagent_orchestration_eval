"""Anthropic chat wrapper + deterministic mock, for the orchestration project.

Returns (text, input_tokens, output_tokens). The mock produces answers whose
*quality varies by design* — some cover all rubric axes, some miss coverage or
grounding — so the judge, the calibration, and the single-vs-multi comparison
all have real signal offline. Quality is keyed off the question id + role so runs
are reproducible.
"""

from __future__ import annotations

import hashlib
import os

DEFAULT_WORKER_MODEL = "claude-haiku-4-5"   # cheap workers (spec)
DEFAULT_PLANNER_MODEL = "claude-opus-4-8"    # strong planner/critic (spec)


class AnthropicChat:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def complete(self, *, model, system, prompt, max_tokens=1024, thinking=False):
        kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": prompt}])
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        r = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens


class MockChat:
    """Deterministic, offline. Answer completeness varies by a hash of the
    question so calibration/comparison have a realistic spread."""

    def complete(self, *, model, system, prompt, max_tokens=1024, thinking=False):
        h = int(hashlib.md5((prompt[:200]).encode()).hexdigest(), 16)
        sysl = system.lower()
        role = ("critic" if "critic" in sysl else
                "synth" if "synthesis" in sysl or "merge" in sysl else
                "planner" if "planning" in sysl or "plan" in sysl else
                "worker")

        if role == "planner":
            text = ("PLAN: 1) define terms 2) compare on the three dimensions "
                    "3) synthesize with sources.")
        elif role == "critic":
            # Critic asks for a revision on ~1 in 5 drafts, else approves.
            text = "REVISE: strengthen the synthesis." if h % 5 == 0 else "OK"
        else:
            # Worker/single answer — completeness varies.
            axes = ["Definition and background.",
                    "Dimension one comparison [source A].",
                    "Dimension two comparison [source B].",
                    "Dimension three comparison [source C].",
                    "Synthesis and recommendation."]
            keep = 3 + (h % 3)  # 3..5 axes covered
            text = " ".join(axes[:keep])
        # crude token accounting for the mock
        return text, len(prompt) // 4, len(text) // 4


def make_chat(backend: str):
    return MockChat() if backend == "mock" else AnthropicChat()
