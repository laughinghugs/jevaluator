"""Shared helpers for scoring a sample with a set of metrics."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable, Mapping, Sequence

from .jev import Jev
from .metrics import Metric, get_metric
from .types import RAGSample, SampleResult

MetricSpec = Metric | str


def resolve_metrics(metrics: Iterable[MetricSpec]) -> list[Metric]:
    resolved = [get_metric(m) if isinstance(m, str) else m for m in metrics]
    names = [m.name for m in resolved]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate metric names: {', '.join(sorted(duplicates))}")
    return resolved


def to_sample(sample: RAGSample | Mapping[str, Any]) -> RAGSample:
    return sample if isinstance(sample, RAGSample) else RAGSample.from_dict(sample)


async def score_sample(sample: RAGSample, metrics: Sequence[Metric], jev: Jev) -> SampleResult:
    start = time.perf_counter()
    results = await asyncio.gather(*(m.score(sample, jev) for m in metrics))
    return SampleResult(
        sample=sample,
        metrics={r.name: r for r in results},
        latency_ms=(time.perf_counter() - start) * 1000,
    )
