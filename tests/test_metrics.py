import pytest

from jevaluator import (
    AnswerCorrectness,
    AnswerRelevance,
    CallableBackend,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
    Jev,
    JudgeMetric,
    RAGSample,
)
from jevaluator.metrics import split_sentences
from jevaluator.metrics.rag import soft_average_precision


async def test_faithfulness_separates_grounded_from_hallucinated(jev, good_sample, bad_sample):
    good = await Faithfulness().score(good_sample, jev)
    bad = await Faithfulness().score(bad_sample, jev)
    assert good.score > 0.9 and bad.score < 0.2
    assert good.details["claim_extraction"] == "llm"
    claim = good.details["claims"][0]
    assert set(claim) == {"claim", "probability", "raw_probability", "rationale"}


async def test_faithfulness_falls_back_to_sentences_without_generate(good_sample):
    jev = Jev(CallableBackend(lambda s, q: 0.8))
    result = await Faithfulness(aggregation="product").score(
        RAGSample("q", "First fact. Second fact.", ["ctx"]), jev
    )
    assert result.details["claim_extraction"] == "sentences"
    assert len(result.details["claims"]) == 2
    assert result.score == pytest.approx(0.64)


async def test_faithfulness_no_claims_is_skipped(good_sample):
    jev = Jev(CallableBackend(lambda s, q: 0.8, lambda s, p: "NONE"))
    result = await Faithfulness().score(good_sample, jev)
    assert result.score is None and "no factual claims" in result.skipped_reason


async def test_answer_relevance(jev, good_sample, bad_sample):
    assert (await AnswerRelevance().score(good_sample, jev)).score > 0.8
    assert (await AnswerRelevance().score(bad_sample, jev)).score < 0.2


async def test_answer_correctness(jev, good_sample, bad_sample):
    assert (await AnswerCorrectness().score(good_sample, jev)).score > 0.8
    assert (await AnswerCorrectness().score(bad_sample, jev)).score < 0.2


async def test_reference_metrics_skip_without_reference(jev, good_sample):
    good_sample.reference = None
    for metric in (AnswerCorrectness(), ContextRecall()):
        result = await metric.score(good_sample, jev)
        assert result.score is None and "reference" in result.skipped_reason


async def test_context_metrics_are_rank_aware(jev, good_sample, bad_sample):
    # Same chunks, relevant one ranked first vs second.
    rel_first = await ContextRelevance().score(good_sample, jev)
    rel_second = await ContextRelevance().score(bad_sample, jev)
    assert rel_first.score == pytest.approx(rel_second.score)

    prec_first = await ContextPrecision().score(good_sample, jev)
    prec_second = await ContextPrecision().score(bad_sample, jev)
    assert prec_first.score > prec_second.score
    assert prec_first.details["mean_relevance"] == pytest.approx(rel_first.score)


async def test_context_recall(jev, good_sample):
    good_sample.contexts = ["Bananas contain potassium."]
    low = await ContextRecall().score(good_sample, jev)
    good_sample.contexts = ["Paris is the capital city of France."]
    high = await ContextRecall().score(good_sample, jev)
    assert high.score > 0.8 > 0.2 > low.score


async def test_custom_judge_metric(good_sample):
    seen = []

    def judge(system, question):
        seen.append(question)
        return 0.7

    metric = JudgeMetric("grounded_tone", "Contexts:\n{contexts}\nAnswer: {answer}\nIs it polite?")
    assert metric.requires == ("contexts",)
    result = await metric.score(good_sample, Jev(CallableBackend(judge)))
    assert result.score == 0.7 and result.name == "grounded_tone"
    assert "[1] Paris is the capital" in seen[0]


async def test_metric_errors_are_captured(good_sample):
    def broken(system, question):
        raise RuntimeError("judge down")

    result = await AnswerRelevance().score(good_sample, Jev(CallableBackend(broken), retries=0))
    assert result.score is None and "judge down" in result.error


def test_soft_average_precision():
    assert soft_average_precision([1, 0, 0]) == pytest.approx(1.0)
    assert soft_average_precision([0, 0, 1]) == pytest.approx(1 / 3)
    assert soft_average_precision([0, 0]) == 0.0


def test_split_sentences():
    assert split_sentences("- One fact. Two facts!\n2) Third") == ["One fact.", "Two facts!", "Third"]
