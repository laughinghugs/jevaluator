# jevaluator

Offline and online evaluation of RAG outputs, with **Jev** as the judge. Every metric
is a **probability** in `[0, 1]`: Jev answers a yes/no evaluation question
("Is this claim supported by the context?") and the score is P(Yes).

| Metric | Question Jev answers | Needs | Online |
|---|---|---|---|
| `faithfulness` | Is each claim in the answer supported by the retrieved context? (per-claim P, aggregated) | contexts | yes |
| `answer_relevance` | Does the answer address the question? | – | yes |
| `context_relevance` | Is each retrieved chunk useful for the question? (mean P) | contexts | yes |
| `context_precision` | Are the relevant chunks ranked first? (soft average precision over the same per-chunk P) | contexts | yes |
| `answer_correctness` | Is the answer correct w.r.t. the ground-truth reference? | reference | – |
| `context_recall` | Can each reference statement be attributed to the context? | contexts, reference | – |

A metric whose inputs are missing is reported as *skipped*, not as 0. Results keep the
per-claim / per-chunk probabilities and rationales so you can see *why* a score is low.
`JudgeMetric` adds your own yes/no criterion in one line.

## Install

```bash
pip install -e .            # core: no dependencies
pip install -e ".[claude]"  # + Anthropic SDK for the Claude backend
pip install -e ".[dev]"     # + pytest
```

## Jev backends

Jev itself is backend-agnostic (`jevaluator.jev.Jev`); it adds calibration, caching,
retries and a global concurrency limit on top of a backend:

* **`LogprobBackend`** (recommended) – Jev served behind any OpenAI-compatible
  `/chat/completions` endpoint that returns `logprobs` (vLLM, TGI, llama.cpp, ...).
  P(Yes) is read directly from the first-token distribution:
  `p(yes) / (p(yes) + p(no))`, summing casing/whitespace variants. One token per
  judgment, so it's fast and cheap.
* **`ClaudeBackend`** – Claude (`claude-opus-5-5` by default) via the Anthropic SDK with
  structured output. The Messages API does not expose log-probabilities, so the score is
  Claude's stated probability (set `n_samples>1` to average several judgments). Server-side
  refusal fallbacks (`fallbacks="default"`) are on by default; pass `fallbacks=False` to
  disable.
* **`CallableBackend`** – wrap any function `(system, question) -> probability`.

```python
from jevaluator import Jev, LogprobBackend

jev = Jev(LogprobBackend("http://localhost:8000/v1", model="jev-judge"), max_concurrency=16)
```

### Calibration

Probabilities are only useful if they are calibrated. Label a few hundred judgments,
fit a Platt calibrator per metric, and Jev applies it to every score (the raw value is
kept as `raw_probability`):

```python
from jevaluator import PlattCalibrator, expected_calibration_error

cal = PlattCalibrator.fit(raw_probs, human_labels)
print(expected_calibration_error([cal(p) for p in raw_probs], human_labels))
cal.save("faithfulness.cal.json")
jev = Jev(backend, calibrators={"faithfulness": cal})
```

## Offline evaluation

```python
from jevaluator import OfflineEvaluator, load_dataset

evaluator = OfflineEvaluator(jev, thresholds={"faithfulness": 0.7})
report = evaluator.evaluate(load_dataset("eval.jsonl"))   # or `await evaluator.aevaluate(...)`

print(report.format_table())          # mean, bootstrap 95% CI, p10, pass rate per metric
report.failures("faithfulness")       # worst samples first, with per-claim probabilities
report.compare(baseline_report)       # mean deltas vs. a previous RAG version
report.to_jsonl("results.jsonl"); report.to_csv("scores.csv")
```

Datasets can be `.jsonl`, `.json` or `.csv` with `question`, `answer`, `contexts` and
optional `reference` (aliases such as `query`, `response`, `retrieved_contexts`,
`ground_truth` work too). In CSV, `contexts` is a JSON list or `||`-separated.

### CLI / CI gate

```bash
jevaluator offline eval.jsonl --base-url http://localhost:8000/v1 --model jev-judge \
  --metrics faithfulness,answer_relevance,answer_correctness \
  --out results.jsonl --fail-under faithfulness=0.8   # exits 1 if the mean drops below
jevaluator offline eval.jsonl --backend claude --samples 3
```

## Online evaluation

Online evaluation defaults to the reference-free metrics. Scoring runs on background
workers so it never blocks the request path; results go to sinks and rolling-window
monitors.

```python
from jevaluator import OnlineEvaluator, JSONLSink, ThresholdMonitor

online = OnlineEvaluator(
    jev,
    sample_rate=0.1,                                   # score 10% of traffic
    sinks=[JSONLSink("jev_online.jsonl"), push_to_metrics],
    monitors=[ThresholdMonitor("faithfulness", 0.75, on_alert=page_oncall, window=200)],
)
await online.start()

@online.trace()                                        # submits every call
async def rag(question: str) -> dict:
    ...
    return {"answer": answer, "contexts": chunks}

online.submit(sample)                                  # or submit explicitly (non-blocking)
result = await online.evaluate(sample)                 # inline, e.g. to gate a response
online.snapshot()                                      # rolling means
online.stats                                           # submitted / sampled_out / dropped / errors
await online.stop()                                    # drains the queue
```

Sync apps (Flask, Django): call `online.start_background_thread()` once at startup;
`submit()` and `@trace` are then thread-safe. Call `online.shutdown()` on exit.

The queue is bounded (`max_queue_size`); when it is full, samples are dropped and counted
rather than slowing your app down. Each evaluation has a `timeout`.

## Layout

```
src/jevaluator/
  jev/          Jev judge, backends, prompts, calibration
  metrics/      metric interface + built-in RAG metrics
  offline.py    OfflineEvaluator, EvaluationReport, load_dataset
  online.py     OnlineEvaluator, sinks, ThresholdMonitor
  cli.py        `jevaluator offline ...`
examples/quickstart.py
```

## Tests

```bash
pytest
```
