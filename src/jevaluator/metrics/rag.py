"""Built-in RAG metrics. Every score is a Jev probability in [0, 1]."""

from __future__ import annotations

import asyncio
from typing import Literal

from ..jev import Jev
from ..jev import prompts as P
from ..types import JevScore, MetricResult, RAGSample
from .base import Metric, aggregate, judgment_detail, parse_claim_lines, split_sentences

ClaimExtraction = Literal["auto", "llm", "sentences"]
Aggregation = Literal["mean", "min", "product"]


async def extract_claims(
    text: str, question: str, jev: Jev, mode: ClaimExtraction = "auto"
) -> tuple[list[str], str]:
    """Split ``text`` into atomic claims. Returns ``(claims, method_used)``.

    ``auto`` asks Jev to extract claims and falls back to sentence splitting if
    the backend cannot generate text.
    """
    if mode == "sentences":
        return split_sentences(text), "sentences"
    try:
        output = await jev.generate(P.CLAIM_EXTRACTION.format(question=question, answer=text))
    except NotImplementedError:
        if mode == "llm":
            raise
        return split_sentences(text), "sentences"
    return parse_claim_lines(output), "llm"


class Faithfulness(Metric):
    """Are the answer's claims supported by the retrieved context? (hallucination check)

    The answer is decomposed into atomic claims; Jev gives P(supported) for each,
    and the per-claim probabilities are aggregated (``mean`` by default; ``min``
    is stricter and ``product`` is P(every claim is supported)).
    """

    name = "faithfulness"
    requires = ("contexts",)

    def __init__(self, claim_extraction: ClaimExtraction = "auto", aggregation: Aggregation = "mean"):
        self.claim_extraction = claim_extraction
        self.aggregation = aggregation

    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        claims, method = await extract_claims(
            sample.answer, sample.question, jev, self.claim_extraction
        )
        if not claims:
            return MetricResult(
                self.name,
                None,
                details={"claims": [], "claim_extraction": method},
                skipped_reason="answer contains no factual claims",
            )
        context = P.format_contexts(sample.contexts)
        scores = await asyncio.gather(
            *(
                jev.judge(P.FAITHFULNESS_CLAIM.format(context=context, claim=c), self.name)
                for c in claims
            )
        )
        probs = [s.probability for s in scores]
        return MetricResult(
            self.name,
            aggregate(probs, self.aggregation),
            details={
                "claims": [judgment_detail("claim", c, s) for c, s in zip(claims, scores)],
                "claim_extraction": method,
                "aggregation": self.aggregation,
                "p_all_supported": aggregate(probs, "product"),
            },
        )


class AnswerRelevance(Metric):
    """Does the answer actually address the question?"""

    name = "answer_relevance"

    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        s = await jev.judge(
            P.ANSWER_RELEVANCE.format(question=sample.question, answer=sample.answer), self.name
        )
        return MetricResult(
            self.name, s.probability, details={"raw_probability": s.raw_probability, "rationale": s.rationale}
        )


class AnswerCorrectness(Metric):
    """Is the answer correct with respect to a ground-truth reference answer?"""

    name = "answer_correctness"
    requires = ("reference",)

    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        s = await jev.judge(
            P.ANSWER_CORRECTNESS.format(
                question=sample.question, reference=sample.reference, answer=sample.answer
            ),
            self.name,
        )
        return MetricResult(
            self.name, s.probability, details={"raw_probability": s.raw_probability, "rationale": s.rationale}
        )


async def _judge_chunks(sample: RAGSample, jev: Jev, criterion: str) -> list[JevScore]:
    # Both context metrics send identical questions, so Jev's cache answers the
    # second metric for free.
    return list(
        await asyncio.gather(
            *(
                jev.judge(P.CONTEXT_RELEVANCE.format(question=sample.question, passage=c), criterion)
                for c in sample.contexts
            )
        )
    )


def soft_average_precision(relevance: list[float]) -> float:
    """Average precision with probabilistic relevance labels (rank-aware)."""
    total = sum(relevance)
    if total == 0:
        return 0.0
    ap = cumulative = 0.0
    for k, r in enumerate(relevance, start=1):
        cumulative += r
        ap += (cumulative / k) * r
    return ap / total


class ContextRelevance(Metric):
    """Mean probability that each retrieved chunk is useful for the question."""

    name = "context_relevance"
    requires = ("contexts",)
    criterion = "context_relevance"

    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        scores = await _judge_chunks(sample, jev, self.criterion)
        probs = [s.probability for s in scores]
        return MetricResult(
            self.name,
            aggregate(probs, "mean"),
            details={
                "chunks": [judgment_detail("chunk_index", i, s) for i, s in enumerate(scores)],
                "average_precision": soft_average_precision(probs),
            },
        )


class ContextPrecision(ContextRelevance):
    """Rank-aware retrieval quality: are the relevant chunks ranked first?

    Uses the same per-chunk Jev judgments as :class:`ContextRelevance` and
    reports their soft average precision.
    """

    name = "context_precision"

    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        result = await super()._score(sample, jev)
        mean_relevance = result.score
        result.name = self.name
        result.score = result.details["average_precision"]
        result.details["mean_relevance"] = mean_relevance
        return result


class ContextRecall(Metric):
    """Does the retrieved context contain everything needed for the reference answer?"""

    name = "context_recall"
    requires = ("contexts", "reference")

    def __init__(self, claim_extraction: ClaimExtraction = "auto", aggregation: Aggregation = "mean"):
        self.claim_extraction = claim_extraction
        self.aggregation = aggregation

    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        statements, method = await extract_claims(
            sample.reference or "", sample.question, jev, self.claim_extraction
        )
        if not statements:
            return MetricResult(
                self.name, None, skipped_reason="reference contains no factual statements"
            )
        context = P.format_contexts(sample.contexts)
        scores = await asyncio.gather(
            *(
                jev.judge(
                    P.CONTEXT_RECALL_STATEMENT.format(context=context, statement=st), self.name
                )
                for st in statements
            )
        )
        return MetricResult(
            self.name,
            aggregate([s.probability for s in scores], self.aggregation),
            details={
                "statements": [judgment_detail("statement", st, s) for st, s in zip(statements, scores)],
                "claim_extraction": method,
                "aggregation": self.aggregation,
            },
        )


class JudgeMetric(Metric):
    """A custom single-question metric.

    ``template`` is a yes/no evaluation question that may reference
    ``{question}``, ``{answer}``, ``{contexts}`` and ``{reference}``::

        JudgeMetric(
            "conciseness",
            "Question: {question}\\nAnswer: {answer}\\n\\n"
            "Evaluation question: Is the answer concise, with no filler or repetition?",
        )
    """

    def __init__(self, name: str, template: str, requires: tuple[str, ...] = ()):
        self.name = name
        self.template = template
        fields = {f for f in ("contexts", "reference") if "{" + f + "}" in template}
        self.requires = tuple(dict.fromkeys((*requires, *sorted(fields))))

    async def _score(self, sample: RAGSample, jev: Jev) -> MetricResult:
        question = self.template.format(
            question=sample.question,
            answer=sample.answer,
            contexts=P.format_contexts(sample.contexts),
            reference=sample.reference or "",
        )
        s = await jev.judge(question, self.name)
        return MetricResult(
            self.name, s.probability, details={"raw_probability": s.raw_probability, "rationale": s.rationale}
        )


BUILTIN_METRICS: dict[str, type[Metric]] = {
    cls.name: cls
    for cls in (
        Faithfulness,
        AnswerRelevance,
        AnswerCorrectness,
        ContextRelevance,
        ContextPrecision,
        ContextRecall,
    )
}


def get_metric(name: str) -> Metric:
    try:
        return BUILTIN_METRICS[name]()
    except KeyError:
        raise ValueError(
            f"unknown metric {name!r}; available: {', '.join(sorted(BUILTIN_METRICS))}"
        ) from None


def default_metrics() -> list[Metric]:
    """All built-in metrics. Reference-based ones skip samples without a reference."""
    return [cls() for cls in BUILTIN_METRICS.values()]


def reference_free_metrics() -> list[Metric]:
    """Metrics that work on live traffic, where no ground-truth answer exists."""
    return [Faithfulness(), AnswerRelevance(), ContextRelevance()]
