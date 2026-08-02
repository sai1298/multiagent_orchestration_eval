"""Rubric-based LLM-as-judge with a versioned prompt and structured JSON scores.

Four binary rubric axes (0/1), scored independently so calibration can report
which axes the judge is trustworthy on:
  coverage   -- addresses every required dimension
  grounding  -- claims are attributed to sources, none fabricated
  structure  -- has definitions + per-dimension comparison + synthesis
  no_halluc  -- no invented facts/sources

The real judge forces structured output via output_config.format (json_schema).
The mock judge applies a deterministic rubric to the mock answers so the whole
calibration pipeline runs offline.
"""

from __future__ import annotations

import json

JUDGE_PROMPT_VERSION = "v1"

RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "coverage": {"type": "integer", "enum": [0, 1]},
        "grounding": {"type": "integer", "enum": [0, 1]},
        "structure": {"type": "integer", "enum": [0, 1]},
        "no_halluc": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["coverage", "grounding", "structure", "no_halluc"],
    "additionalProperties": False,
}

AXES = ["coverage", "grounding", "structure", "no_halluc"]

JUDGE_SYSTEM = (
    "You are a strict evaluation judge. Score the answer on four binary axes "
    "(0 or 1): coverage (addresses every required dimension), grounding (claims "
    "cite sources, none fabricated), structure (definitions + per-dimension "
    "comparison + synthesis), no_halluc (no invented facts). Return JSON only."
)


class AnthropicJudge:
    def __init__(self, model="claude-opus-4-8"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def score(self, question: dict, answer: str) -> dict:
        prompt = (f"Question: {question['question']}\n"
                  f"Required dimensions: {', '.join(question['dimensions'])}\n\n"
                  f"Answer to grade:\n{answer}\n\nScore all four axes.")
        r = self.client.messages.create(
            model=self.model, max_tokens=300, system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": RUBRIC_SCHEMA}},
        )
        text = next(b.text for b in r.content if b.type == "text")
        return json.loads(text)


class MockJudge:
    """Deterministic rubric applied to answer text — mirrors the mock answers so
    the calibration pipeline produces real agreement numbers offline."""

    model = "mock"

    def score(self, question: dict, answer: str) -> dict:
        a = answer.lower()
        dims = question["dimensions"]
        covered = sum(1 for d in dims if d.split()[0].lower() in a
                      or "dimension" in a)
        coverage = int(answer.count("[") >= len(dims) or "dimension three" in a)
        grounding = int("[source" in a or "[" in a)
        structure = int(("definition" in a or "background" in a) and
                        "synthesis" in a)
        no_halluc = int("fabricated" not in a)
        return {"coverage": coverage, "grounding": grounding,
                "structure": structure, "no_halluc": no_halluc}


def make_judge(backend: str, model="claude-opus-4-8"):
    return MockJudge() if backend == "mock" else AnthropicJudge(model)
