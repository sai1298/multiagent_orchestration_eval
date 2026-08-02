"""Build the frozen question set (research/answer-synthesis, comparison-style).

Each question names three explicit dimensions, so 'coverage' is objectively
gradeable and the task is one where decomposition *plausibly* helps (the
precondition for a fair single-vs-multi comparison).

30 questions is enough for a meaningful comparison while keeping a real API run
cheap; scale QUESTIONS up for a larger study. The set is frozen to a JSONL so
numbers are reproducible.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "eval", "questions_v1.jsonl")

TOPICS = [
    ("INT8 vs FP16 quantization", ["accuracy impact", "latency", "hardware support"]),
    ("Random Forest vs Gradient Boosting", ["bias-variance", "training speed", "interpretability"]),
    ("BM25 vs dense retrieval", ["recall", "precision", "cost"]),
    ("LSTM vs Transformer for time series", ["long-range dependencies", "training cost", "inference latency"]),
    ("structured vs unstructured pruning", ["hardware speedup", "accuracy recovery", "implementation effort"]),
    ("SMOTE vs class weighting", ["minority recall", "overfitting risk", "leakage risk"]),
    ("Docker vs bare-metal for ML serving", ["isolation", "startup latency", "resource overhead"]),
    ("batch vs streaming inference", ["throughput", "tail latency", "complexity"]),
    ("Isolation Forest vs LSTM autoencoder", ["detection quality", "training data needs", "latency"]),
    ("operator fusion vs kernel autotuning", ["speedup", "portability", "engineering cost"]),
    ("PTQ vs QAT", ["accuracy", "training cost", "deployment effort"]),
    ("round-robin vs least-loaded routing", ["tail latency", "fairness", "implementation cost"]),
    ("FAISS flat vs IVF index", ["recall", "query latency", "memory"]),
    ("cross-encoder vs bi-encoder reranking", ["accuracy", "latency", "scalability"]),
    ("RMSE vs C-MAPSS score for RUL", ["safety alignment", "interpretability", "sensitivity"]),
    ("feature selection vs regularization", ["overfitting control", "interpretability", "compute"]),
    ("data parallelism vs model parallelism", ["memory", "communication overhead", "scaling limit"]),
    ("A/B testing vs bandits", ["sample efficiency", "regret", "implementation"]),
    ("cosine vs euclidean similarity", ["scale sensitivity", "retrieval quality", "compute"]),
    ("early stopping vs fixed epochs", ["generalization", "compute", "reproducibility"]),
    ("greedy vs beam search decoding", ["quality", "latency", "diversity"]),
    ("static vs dynamic batching", ["throughput", "latency", "complexity"]),
    ("SHAP vs permutation importance", ["fidelity", "compute cost", "bias"]),
    ("z-score vs IQR outlier detection", ["robustness", "assumptions", "simplicity"]),
    ("mean vs median imputation", ["bias", "variance", "robustness to outliers"]),
    ("gRPC vs REST for inference", ["latency", "streaming support", "tooling"]),
    ("prompt caching vs fine-tuning", ["cost", "latency", "flexibility"]),
    ("CPU vs GPU quantized inference", ["throughput", "cost", "backend support"]),
    ("stratified vs random CV split", ["variance", "leakage risk", "small-data reliability"]),
    ("reciprocal rank fusion vs score fusion", ["robustness", "calibration need", "simplicity"]),
]


def main():
    qs = []
    for i, (topic, dims) in enumerate(TOPICS):
        qs.append({
            "id": f"q{i + 1:02d}",
            "question": f"Compare {topic} and explain the trade-offs.",
            "dimensions": dims,
        })
    with open(OUT, "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    print(f"[questions] wrote {len(qs)} questions to {OUT}")


if __name__ == "__main__":
    main()
