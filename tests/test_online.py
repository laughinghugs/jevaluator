import asyncio
import json
import threading

import pytest

from jevaluator import (
    CallableBackend,
    Jev,
    JSONLSink,
    OnlineEvaluator,
    RAGSample,
    ThresholdMonitor,
)


async def test_inline_evaluate_uses_reference_free_metrics(jev, good_sample):
    evaluator = OnlineEvaluator(jev)
    result = await evaluator.evaluate(good_sample)
    assert set(result.scores) == {"faithfulness", "answer_relevance", "context_relevance"}
    assert result.passed({"faithfulness": 0.8, "answer_relevance": 0.8, "context_relevance": 0.4})
    assert not result.passed(0.9)
    assert evaluator.stats.evaluated == 1


async def test_background_submit_drains_into_sinks(jev, good_sample, bad_sample, tmp_path):
    received = []
    sink_path = tmp_path / "online.jsonl"
    async with OnlineEvaluator(jev, sinks=[JSONLSink(sink_path), received.append]) as evaluator:
        assert evaluator.submit(good_sample)
        assert evaluator.submit(bad_sample.to_dict())
    assert {r.sample.id for r in received} == {"good", "bad"}
    lines = [json.loads(l) for l in sink_path.read_text().splitlines()]
    assert len(lines) == 2
    assert evaluator.snapshot()["faithfulness"]["n"] == 2


async def test_sampling_and_queue_full(jev, good_sample):
    evaluator = OnlineEvaluator(jev, sample_rate=0.0)
    await evaluator.start()
    assert evaluator.submit(good_sample) is False
    assert evaluator.stats.sampled_out == 1
    await evaluator.stop()

    gate = asyncio.Event()

    async def slow(system, question):
        await gate.wait()
        return 0.9

    evaluator = OnlineEvaluator(
        Jev(CallableBackend(slow)), metrics=["answer_relevance"], workers=1, max_queue_size=1
    )
    await evaluator.start()
    evaluator.submit(good_sample)
    await asyncio.sleep(0.01)  # worker picks up the first sample and blocks
    assert evaluator.submit(good_sample)
    assert evaluator.submit(good_sample) is False
    assert evaluator.stats.dropped == 1
    gate.set()
    await evaluator.stop()
    assert evaluator.stats.evaluated == 2


async def test_submit_requires_running(jev, good_sample):
    with pytest.raises(RuntimeError):
        OnlineEvaluator(jev).submit(good_sample)


async def test_threshold_monitor_alerts_with_cooldown(jev, bad_sample, good_sample):
    alerts = []
    monitor = ThresholdMonitor("faithfulness", 0.5, alerts.append, window=3, min_samples=2, cooldown_s=3600)
    evaluator = OnlineEvaluator(jev, monitors=[monitor])
    await evaluator.evaluate(bad_sample)
    assert alerts == []
    await evaluator.evaluate(bad_sample)
    assert len(alerts) == 1 and alerts[0].metric == "faithfulness"
    await evaluator.evaluate(bad_sample)
    assert len(alerts) == 1  # cooldown


async def test_timeout_records_error(good_sample):
    async def hang(system, question):
        await asyncio.sleep(10)

    evaluator = OnlineEvaluator(Jev(CallableBackend(hang)), metrics=["answer_relevance"], timeout=0.05)
    result = await evaluator.evaluate(good_sample)
    assert result.metrics["answer_relevance"].error == "evaluation timed out"
    assert evaluator.stats.timeouts == 1


async def test_trace_decorator(jev):
    received = []
    evaluator = OnlineEvaluator(jev, metrics=["answer_relevance"], sinks=[received.append])

    @evaluator.trace()
    async def rag(question: str) -> dict:
        return {"answer": "Paris is the capital city of France.", "contexts": ["Paris..."]}

    await evaluator.start()
    assert (await rag("What is the capital city of France?"))["answer"].startswith("Paris")
    await evaluator.stop()
    assert received[0].sample.question == "What is the capital city of France?"


def test_background_thread_for_sync_apps(jev):
    received = []
    evaluator = OnlineEvaluator(jev, metrics=["answer_relevance"], sinks=[received.append])
    evaluator.start_background_thread()

    @evaluator.trace(lambda args, kwargs, answer: RAGSample(args[0], answer, ["ctx"]))
    def rag(question):
        return "Paris is the capital city of France."

    threads = [threading.Thread(target=rag, args=("What is the capital city of France?",)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    evaluator.shutdown(timeout=5)
    assert len(received) == 3
    assert not evaluator.running
