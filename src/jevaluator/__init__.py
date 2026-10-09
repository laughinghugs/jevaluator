"""jevaluator - offline and online RAG evaluation with Jev as the probabilistic judge."""

from .jev import (
    CallableBackend,
    ClaudeBackend,
    Jev,
    JevError,
    JevRefusal,
    LogprobBackend,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
)
from .metrics import (
    AnswerCorrectness,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
    JudgeMetric,
    Metric,
    default_metrics,
    reference_free_metrics,
)
from .offline import EvaluationReport, MetricSummary, OfflineEvaluator, load_dataset
from .online import (
    Alert,
    CallbackSink,
    JSONLSink,
    LoggingSink,
    OnlineEvaluator,
    ThresholdMonitor,
)
from .types import JevScore, MetricResult, RAGSample, SampleResult

__version__ = "0.1.0"

__all__ = [
    "Alert",
    "AnswerCorrectness",
    "AnswerRelevance",
    "CallableBackend",
    "CallbackSink",
    "ClaudeBackend",
    "ContextPrecision",
    "ContextRecall",
    "ContextRelevance",
    "EvaluationReport",
    "Faithfulness",
    "JSONLSink",
    "Jev",
    "JevError",
    "JevRefusal",
    "JevScore",
    "JudgeMetric",
    "LogprobBackend",
    "LoggingSink",
    "Metric",
    "MetricResult",
    "MetricSummary",
    "OfflineEvaluator",
    "OnlineEvaluator",
    "PlattCalibrator",
    "RAGSample",
    "SampleResult",
    "ThresholdMonitor",
    "brier_score",
    "default_metrics",
    "expected_calibration_error",
    "load_dataset",
    "reference_free_metrics",
]
