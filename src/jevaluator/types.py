"""Core data types shared by the offline and online evaluators."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# Accepted aliases for each RAGSample field, so datasets exported from common
# RAG frameworks load without renaming columns.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question", "query", "user_input", "input"),
    "answer": ("answer", "response", "output", "generation"),
    "contexts": ("contexts", "retrieved_contexts", "context", "documents", "passages"),
    "reference": ("reference", "ground_truth", "expected_answer", "gold_answer"),
    "id": ("id", "sample_id", "trace_id"),
}


@dataclass
class RAGSample:
    """One RAG interaction: the user question, retrieved contexts and generated answer.

    ``reference`` (a ground-truth answer) is optional; metrics that need it are
    skipped when it is missing, which is the normal case for online traffic.
    """

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    reference: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RAGSample":
        values: dict[str, Any] = {}
        used: set[str] = set()
        for target, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                if alias in data and data[alias] is not None:
                    values[target] = data[alias]
                    used.add(alias)
                    break
        missing = [f for f in ("question", "answer") if f not in values]
        if missing:
            raise ValueError(f"RAG sample is missing required field(s): {', '.join(missing)}")

        contexts = values.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        values["contexts"] = [str(c) for c in contexts]
        if "id" in values:
            values["id"] = str(values["id"])

        metadata = dict(data.get("metadata") or {})
        metadata.update({k: v for k, v in data.items() if k not in used and k != "metadata"})
        return cls(metadata=metadata, **values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JevScore:
    """A single Jev judgment: the probability that the evaluated criterion holds."""

    probability: float
    raw_probability: float
    rationale: str | None = None
    criterion: str | None = None

    @property
    def verdict(self) -> bool:
        return self.probability >= 0.5


@dataclass
class MetricResult:
    """The outcome of one metric on one sample.

    ``score`` is a probability in [0, 1], or ``None`` when the metric was
    skipped (missing inputs) or failed (``error`` is set).
    """

    name: str
    score: float | None
    details: dict[str, Any] = field(default_factory=dict)
    skipped_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.score is not None

    def passed(self, threshold: float = 0.5) -> bool | None:
        return None if self.score is None else self.score >= threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleResult:
    """All metric results for one sample."""

    sample: RAGSample
    metrics: dict[str, MetricResult]
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def scores(self) -> dict[str, float | None]:
        return {name: m.score for name, m in self.metrics.items()}

    def passed(self, thresholds: Mapping[str, float] | float = 0.5) -> bool:
        """True when every scored metric meets its threshold (skipped metrics are ignored)."""
        for name, result in self.metrics.items():
            if result.score is None:
                continue
            t = thresholds if isinstance(thresholds, (int, float)) else thresholds.get(name, 0.5)
            if result.score < t:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "scores": self.scores,
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }
