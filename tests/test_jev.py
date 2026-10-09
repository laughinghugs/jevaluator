import asyncio
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from jevaluator import (
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
from jevaluator.jev import yes_probability_from_logprobs


def test_yes_probability_from_logprobs_merges_variants():
    top = [
        {"token": "Yes", "logprob": math.log(0.5)},
        {"token": " yes", "logprob": math.log(0.1)},
        {"token": "No", "logprob": math.log(0.2)},
        {"token": "Maybe", "logprob": math.log(0.2)},
    ]
    assert yes_probability_from_logprobs(top) == pytest.approx(0.6 / 0.8)
    assert yes_probability_from_logprobs([{"token": "Hmm", "logprob": 0.0}]) is None


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append((dict(self.headers), body))
        if body.get("logprobs"):
            payload = {
                "choices": [
                    {
                        "message": {"content": "Yes"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "Yes",
                                    "logprob": math.log(0.7),
                                    "top_logprobs": [
                                        {"token": "Yes", "logprob": math.log(0.7)},
                                        {"token": "No", "logprob": math.log(0.3)},
                                    ],
                                }
                            ]
                        },
                    }
                ]
            }
        else:
            payload = {"choices": [{"message": {"content": "claim one\nclaim two"}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


async def test_logprob_backend_reads_probability(server):
    backend = LogprobBackend(server, "jev-judge", api_key="secret")
    p, rationale = await backend.yes_probability("sys", "Is it?")
    assert p == pytest.approx(0.7)
    assert rationale is None
    headers, body = _Handler.requests[0]
    assert headers["Authorization"] == "Bearer secret"
    assert body["model"] == "jev-judge"
    assert body["max_tokens"] == 1 and body["logprobs"] is True and body["temperature"] == 0
    assert body["messages"][1]["content"].endswith("Answer with exactly one word: Yes or No.")

    assert await backend.generate("sys", "extract") == "claim one\nclaim two"


class _FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **params):
        self.calls.append(params)
        return self.responses.pop(0)


def _response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
    )


def _claude(responses, **kwargs):
    messages = _FakeMessages(responses)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeBackend(client=client, **kwargs), messages


async def test_claude_backend_structured_probability():
    out = json.dumps({"rationale": "supported", "verdict": "yes", "probability": 0.8})
    backend, messages = _claude([_response(out)])
    p, rationale = await backend.yes_probability("sys", "Is it?")
    assert p == pytest.approx(0.8) and rationale == "supported"
    params = messages.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["fallbacks"] == "default"
    assert params["betas"] == ["server-side-fallback-2026-07-01"]


async def test_claude_backend_averages_samples_and_clamps():
    outs = [json.dumps({"rationale": "", "verdict": "yes", "probability": v}) for v in (1.4, 0.6)]
    backend, messages = _claude([_response(o) for o in outs], n_samples=2, fallbacks=False, effort="low")
    p, _ = await backend.yes_probability("sys", "Is it?")
    assert p == pytest.approx(0.8)
    assert "fallbacks" not in messages.calls[0]
    assert messages.calls[0]["output_config"]["effort"] == "low"


async def test_claude_backend_refusal_raises():
    backend, _ = _claude([_response("", stop_reason="refusal")])
    with pytest.raises(JevRefusal):
        await backend.yes_probability("sys", "Is it?")


async def test_jev_caches_and_dedupes_concurrent_calls():
    calls = 0

    async def judge(system, question):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return 0.9

    jev = Jev(CallableBackend(judge))
    scores = await asyncio.gather(*(jev.judge("same question") for _ in range(5)))
    assert calls == 1
    assert all(s.probability == 0.9 for s in scores)
    await jev.judge("same question")
    assert calls == 1


async def test_jev_retries_then_raises():
    attempts = 0

    def flaky(system, question):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("boom")
        return 0.4

    jev = Jev(CallableBackend(flaky), retries=2, retry_backoff=0)
    assert (await jev.judge("q")).probability == 0.4
    assert attempts == 3

    jev = Jev(CallableBackend(lambda s, q: (_ for _ in ()).throw(JevError("bad"))), retries=1, retry_backoff=0)
    with pytest.raises(JevError):
        await jev.judge("q")


async def test_jev_applies_calibrator_per_criterion():
    jev = Jev(CallableBackend(lambda s, q: 0.9), calibrators={"faithfulness": lambda p: p / 2})
    calibrated = await jev.judge("q", "faithfulness")
    other = await jev.judge("q", "answer_relevance")
    assert calibrated.probability == pytest.approx(0.45)
    assert calibrated.raw_probability == pytest.approx(0.9)
    assert other.probability == pytest.approx(0.9)


def test_jev_usable_across_event_loops():
    jev = Jev(CallableBackend(lambda s, q: 0.3))
    assert asyncio.run(jev.judge("a")).probability == 0.3
    assert asyncio.run(jev.judge("b")).probability == 0.3


def test_platt_calibration_fixes_overconfidence():
    # Judge says 0.95 but is right only 70% of the time, says 0.05 and is wrong 30%.
    probs = [0.95] * 100 + [0.05] * 100
    labels = [1] * 70 + [0] * 30 + [1] * 30 + [0] * 70
    cal = PlattCalibrator.fit(probs, labels)
    assert cal(0.95) == pytest.approx(0.7, abs=0.02)
    assert cal(0.05) == pytest.approx(0.3, abs=0.02)
    calibrated = [cal(p) for p in probs]
    assert brier_score(calibrated, labels) < brier_score(probs, labels)
    assert expected_calibration_error(calibrated, labels) < expected_calibration_error(probs, labels)


def test_platt_save_load(tmp_path):
    cal = PlattCalibrator(a=0.5, b=-0.2)
    cal.save(tmp_path / "c.json")
    assert PlattCalibrator.load(tmp_path / "c.json") == cal
