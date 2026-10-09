"""Metric interface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from ..jev import Jev
from ..types import MetricResult, RAGSample


class Metric(ABC):
    """A RAG quality metric scored by Jev.

    Subclasses set ``name`` and ``requires`` (the optional sample fields they
    need - ``"contexts"`` and/or ``"reference"``) and implement :meth:`_score`.
    """

    name: str = "metric"
    requires: tuple[str, ...] = ()

    def missing_fields(self, sample: RAGSample) -> list[str]:
        return [f for f in self.requires if not getattr(sample, f)]

    def is_applicable(self, sample: RAGSample) -> bool:
        return not self.missing_fields(sample)

    async def score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        missing = self.missing_fields(sample)
        if missing:
            return MetricResult(
                self.name, None, skipped_reason=f"sample has no {', '.join(missing)}"
            )
        try:
            return await self._score(sample, jev)
        except Exception as exc:  # one failing metric must not sink the whole sample
            return MetricResult(self.name, None, error=f"{type(exc).__name__}: {exc}")

    @abstractmethod
    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def split_sentences(text: str) -> list[str]:
    """Heuristic sentence/bullet splitter used when the judge cannot extract claims."""
    pieces: list[str] = []
    for line in text.splitlines():
        line = _BULLET_RE.sub("", line).strip()
        if line:
            pieces.extend(s.strip() for s in _SENTENCE_RE.split(line) if s.strip())
    return pieces


def parse_claim_lines(text: str) -> list[str]:
    if text.strip().upper() == "NONE":
        return []
    return [c for c in (_BULLET_RE.sub("", line).strip() for line in text.splitlines()) if c]


def aggregate(probabilities: list[float], method: str) -> float:
    if not probabilities:
        raise ValueError("nothing to aggregate")
    if method == "mean":
        return sum(probabilities) / len(probabilities)
    if method == "min":
        return min(probabilities)
    if method == "product":
        # P(all items hold), treating judgments as independent.
        out = 1.0
        for p in probabilities:
            out *= p
        return out
    raise ValueError(f"unknown aggregation {method!r}; use 'mean', 'min' or 'product'")


def judgment_detail(text_key: str, text: str, score: Any) -> dict[str, Any]:
    return {
        text_key: text,
        "probability": score.probability,
        "raw_probability": score.raw_probability,
        "rationale": score.rationale,
    }
