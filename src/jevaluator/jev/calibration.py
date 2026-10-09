"""Calibration of Jev probabilities against human labels.

Raw judge probabilities (especially verbalized ones) are often over-confident.
Collect a few hundred human-labelled judgments, fit a :class:`PlattCalibrator`
per criterion, and pass it to :class:`~jevaluator.jev.Jev` so every score is
mapped onto a calibrated probability. Use :func:`expected_calibration_error`
and :func:`brier_score` to check the result.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

_EPS = 1e-6


class Calibrator(Protocol):
    def __call__(self, probability: float) -> float: ...


def _logit(p: float) -> float:
    p = min(1 - _EPS, max(_EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


@dataclass
class PlattCalibrator:
    """Logistic recalibration in logit space: ``p' = sigmoid(a * logit(p) + b)``."""

    a: float = 1.0
    b: float = 0.0

    def __call__(self, probability: float) -> float:
        return _sigmoid(self.a * _logit(probability) + self.b)

    @classmethod
    def fit(
        cls,
        probabilities: Sequence[float],
        labels: Sequence[bool | int],
        iterations: int = 100,
        l2: float = 1e-3,
    ) -> "PlattCalibrator":
        """Fit by Newton's method on the (lightly L2-regularized) log loss."""
        if len(probabilities) != len(labels) or not probabilities:
            raise ValueError("probabilities and labels must be non-empty and of equal length")
        n_pos = sum(1 for y in labels if y)
        n_neg = len(labels) - n_pos
        # Platt's smoothed targets avoid infinite weights on separable data.
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1 / (n_neg + 2)
        xs = [_logit(p) for p in probabilities]
        ts = [t_pos if y else t_neg for y in labels]

        def loss(a: float, b: float) -> float:
            total = 0.5 * l2 * ((a - 1.0) ** 2 + b**2)
            for x, t in zip(xs, ts):
                z = a * x + b
                # log(1 + e^z) - t*z, computed stably
                total += (z if z > 0 else 0.0) + math.log1p(math.exp(-abs(z))) - t * z
            return total

        a, b = 1.0, 0.0
        current = loss(a, b)
        for _ in range(iterations):
            ga, gb = l2 * (a - 1.0), l2 * b
            haa, hab, hbb = l2, 0.0, l2
            for x, t in zip(xs, ts):
                p = _sigmoid(a * x + b)
                d = p - t
                w = p * (1 - p)
                ga += d * x
                gb += d
                haa += w * x * x
                hab += w * x
                hbb += w
            det = haa * hbb - hab * hab
            if abs(det) < 1e-12:
                break
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
            # Backtracking line search: undamped Newton can overshoot on logistic loss.
            step = 1.0
            while step > 1e-8:
                candidate = loss(a - step * da, b - step * db)
                if candidate <= current:
                    break
                step /= 2
            else:
                break
            a, b = a - step * da, b - step * db
            improvement = current - candidate
            current = candidate
            if improvement < 1e-12:
                break
        return cls(a=a, b=b)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "platt", "a": self.a, "b": self.b}))

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        data = json.loads(Path(path).read_text())
        return cls(a=data["a"], b=data["b"])


def brier_score(probabilities: Sequence[float], labels: Sequence[bool | int]) -> float:
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probabilities, labels)) / len(labels)


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[bool | int], n_bins: int = 10
) -> float:
    """Weighted mean |accuracy - confidence| over equal-width probability bins."""
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probabilities, labels):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, 1.0 if y else 0.0))
    total = len(probabilities)
    ece = 0.0
    for b in bins:
        if b:
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(y for _, y in b) / len(b)
            ece += len(b) / total * abs(acc - conf)
    return ece
