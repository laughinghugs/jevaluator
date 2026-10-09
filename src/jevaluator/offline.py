"""Offline (batch) evaluation of a RAG system over a dataset."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .jev import Jev
from .metrics import default_metrics
from .runner import MetricSpec, resolve_metrics, score_sample, to_sample
from .types import RAGSample, SampleResult

# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def _parse_contexts_cell(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(c) for c in parsed]
        except json.JSONDecodeError:
            pass
    return [c.strip() for c in value.split("||") if c.strip()]


def load_dataset(path: str | Path) -> list[RAGSample]:
    """Load samples from ``.jsonl``, ``.json`` (a list of objects) or ``.csv``.

    In CSV files the contexts column holds either a JSON list or ``||``-separated
    passages. Common column aliases (``query``, ``response``, ``ground_truth``,
    ``retrieved_contexts``, ...) are accepted.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("samples") or rows.get("data") or []
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = []
            for row in csv.DictReader(f):
                for key in ("contexts", "retrieved_contexts", "context", "documents", "passages"):
                    if key in row and isinstance(row[key], str):
                        row[key] = _parse_contexts_cell(row[key])
                rows.append({k: v for k, v in row.items() if v not in ("", None)})
    else:
        raise ValueError(f"unsupported dataset format {suffix!r}; use .jsonl, .json or .csv")
    return [RAGSample.from_dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class MetricSummary:
    metric: str
    n: int
    n_skipped: int
    n_errors: int
    mean: float | None
    std: float | None
    median: float | None
    p10: float | None
    p90: float | None
    ci95_low: float | None
    ci95_high: float | None
    pass_rate: float | None
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if len(values) < 2:
        v = values[0] if values else float("nan")
        return v, v
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return _quantile(means, alpha), _quantile(means, 1 - alpha)


class EvaluationReport:
    """Per-sample results plus aggregate statistics for an offline run."""

    def __init__(
        self,
        results: list[SampleResult],
        metric_names: Sequence[str],
        thresholds: Mapping[str, float] | None = None,
        default_threshold: float = 0.5,
    ) -> None:
        self.results = results
        self.metric_names = list(metric_names)
        self.thresholds = dict(thresholds or {})
        self.default_threshold = default_threshold

    def threshold(self, metric: str) -> float:
        return self.thresholds.get(metric, self.default_threshold)

    def scores(self, metric: str) -> list[float]:
        return [
            r.metrics[metric].score
            for r in self.results
            if metric in r.metrics and r.metrics[metric].score is not None
        ]

    def summary(self) -> dict[str, MetricSummary]:
        out: dict[str, MetricSummary] = {}
        for name in self.metric_names:
            values = self.scores(name)
            metric_results = [r.metrics[name] for r in self.results if name in r.metrics]
            n_err = sum(1 for m in metric_results if m.error)
            n_skip = sum(1 for m in metric_results if m.skipped_reason)
            t = self.threshold(name)
            if values:
                s = sorted(values)
                lo, hi = bootstrap_ci(values)
                out[name] = MetricSummary(
                    metric=name,
                    n=len(values),
                    n_skipped=n_skip,
                    n_errors=n_err,
                    mean=statistics.fmean(values),
                    std=statistics.pstdev(values),
                    median=_quantile(s, 0.5),
                    p10=_quantile(s, 0.1),
                    p90=_quantile(s, 0.9),
                    ci95_low=lo,
                    ci95_high=hi,
                    pass_rate=sum(v >= t for v in values) / len(values),
                    threshold=t,
                )
            else:
                out[name] = MetricSummary(
                    name, 0, n_skip, n_err, None, None, None, None, None, None, None, None, t
                )
        return out

    def failures(self, metric: str, threshold: float | None = None) -> list[SampleResult]:
        """Samples scoring below ``threshold`` on ``metric``, worst first."""
        t = self.threshold(metric) if threshold is None else threshold
        failing = [
            r
            for r in self.results
            if metric in r.metrics and r.metrics[metric].score is not None and r.metrics[metric].score < t
        ]
        return sorted(failing, key=lambda r: r.metrics[metric].score)

    def compare(self, baseline: "EvaluationReport") -> dict[str, dict[str, float | None]]:
        """Mean score deltas versus a baseline run (e.g. the previous RAG version)."""
        mine, theirs = self.summary(), baseline.summary()
        out: dict[str, dict[str, float | None]] = {}
        for name in self.metric_names:
            a = mine[name].mean
            b = theirs[name].mean if name in theirs else None
            out[name] = {
                "current": a,
                "baseline": b,
                "delta": None if a is None or b is None else a - b,
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {k: v.to_dict() for k, v in self.summary().items()},
            "results": [r.to_dict() for r in self.results],
        }

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    def to_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "question", *self.metric_names])
            for r in self.results:
                writer.writerow(
                    [r.sample.id, r.sample.question]
                    + [
                        "" if r.metrics.get(n) is None or r.metrics[n].score is None else f"{r.metrics[n].score:.4f}"
                        for n in self.metric_names
                    ]
                )

    def format_table(self) -> str:
        def f(v: float | None) -> str:
            return "-" if v is None else f"{v:.3f}"

        header = f"{'metric':<20} {'n':>5} {'mean':>7} {'95% CI':>15} {'p10':>7} {'pass':>7} {'skip':>5} {'err':>4}"
        lines = [header, "-" * len(header)]
        for s in self.summary().values():
            ci = "-" if s.ci95_low is None else f"[{s.ci95_low:.3f},{s.ci95_high:.3f}]"
            lines.append(
                f"{s.metric:<20} {s.n:>5} {f(s.mean):>7} {ci:>15} {f(s.p10):>7} "
                f"{f(s.pass_rate):>7} {s.n_skipped:>5} {s.n_errors:>4}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"EvaluationReport(n_samples={len(self.results)}, metrics={self.metric_names})"


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class OfflineEvaluator:
    """Scores a whole dataset with Jev and produces an :class:`EvaluationReport`.

    >>> evaluator = OfflineEvaluator(jev, metrics=["faithfulness", "answer_relevance"])
    >>> report = evaluator.evaluate(load_dataset("eval.jsonl"))
    >>> print(report.format_table())
    """

    def __init__(
        self,
        jev: Jev,
        metrics: Iterable[MetricSpec] | None = None,
        *,
        max_concurrent_samples: int = 16,
        thresholds: Mapping[str, float] | None = None,
        default_threshold: float = 0.5,
    ) -> None:
        self.jev = jev
        self.metrics = resolve_metrics(metrics if metrics is not None else default_metrics())
        self.max_concurrent_samples = max_concurrent_samples
        self.thresholds = dict(thresholds or {})
        self.default_threshold = default_threshold

    async def aevaluate(
        self,
        dataset: Iterable[RAGSample | Mapping[str, Any]],
        on_result: Callable[[SampleResult], None] | None = None,
    ) -> EvaluationReport:
        samples = [to_sample(s) for s in dataset]
        sem = asyncio.Semaphore(self.max_concurrent_samples)

        async def run(sample: RAGSample) -> SampleResult:
            async with sem:
                result = await score_sample(sample, self.metrics, self.jev)
            if on_result:
                on_result(result)
            return result

        results = await asyncio.gather(*(run(s) for s in samples))
        return EvaluationReport(
            list(results),
            [m.name for m in self.metrics],
            self.thresholds,
            self.default_threshold,
        )

    def evaluate(
        self,
        dataset: Iterable[RAGSample | Mapping[str, Any]],
        on_result: Callable[[SampleResult], None] | None = None,
    ) -> EvaluationReport:
        """Synchronous wrapper around :meth:`aevaluate` (not for use inside a running loop)."""
        return asyncio.run(self.aevaluate(dataset, on_result))
