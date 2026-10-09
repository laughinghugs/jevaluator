import csv
import json

import pytest

from jevaluator import OfflineEvaluator, RAGSample, load_dataset
from jevaluator import cli
from jevaluator.offline import bootstrap_ci


async def test_offline_report(jev, good_sample, bad_sample):
    no_ref = RAGSample("What is the capital city of France?", "Paris is the capital city.", ["Paris is the capital city of France."])
    evaluator = OfflineEvaluator(jev, thresholds={"faithfulness": 0.7})
    report = await evaluator.aevaluate([good_sample, bad_sample, no_ref])

    assert len(report.results) == 3
    summary = report.summary()
    assert set(summary) == {
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
        "context_relevance",
        "context_precision",
        "context_recall",
    }
    faith = summary["faithfulness"]
    assert faith.n == 3 and faith.threshold == 0.7
    assert faith.pass_rate == pytest.approx(2 / 3)
    assert faith.ci95_low <= faith.mean <= faith.ci95_high
    assert summary["answer_correctness"].n == 2 and summary["answer_correctness"].n_skipped == 1

    assert [r.sample.id for r in report.failures("faithfulness")] == ["bad"]
    assert "faithfulness" in report.format_table()


async def test_report_exports_and_compare(jev, good_sample, bad_sample, tmp_path):
    evaluator = OfflineEvaluator(jev, metrics=["faithfulness", "answer_relevance"])
    current = await evaluator.aevaluate([good_sample])
    baseline = await evaluator.aevaluate([bad_sample])

    delta = current.compare(baseline)
    assert delta["faithfulness"]["delta"] > 0.5

    current.to_jsonl(tmp_path / "r.jsonl")
    line = json.loads((tmp_path / "r.jsonl").read_text().splitlines()[0])
    assert line["sample"]["id"] == "good" and "faithfulness" in line["scores"]

    current.to_csv(tmp_path / "r.csv")
    rows = list(csv.reader((tmp_path / "r.csv").open()))
    assert rows[0] == ["id", "question", "faithfulness", "answer_relevance"]


def test_evaluate_sync_with_progress(jev, good_sample):
    seen = []
    report = OfflineEvaluator(jev, metrics=["answer_relevance"]).evaluate(
        [good_sample.to_dict()], on_result=seen.append
    )
    assert len(seen) == 1 and report.results[0].scores["answer_relevance"] > 0.8


def test_load_dataset_formats(tmp_path):
    (tmp_path / "d.jsonl").write_text(
        json.dumps({"query": "q1", "response": "a1", "retrieved_contexts": ["c1"], "ground_truth": "r1", "team": "x"})
        + "\n"
    )
    [s] = load_dataset(tmp_path / "d.jsonl")
    assert (s.question, s.answer, s.contexts, s.reference) == ("q1", "a1", ["c1"], "r1")
    assert s.metadata == {"team": "x"}

    (tmp_path / "d.json").write_text(json.dumps([{"question": "q", "answer": "a", "contexts": "one"}]))
    assert load_dataset(tmp_path / "d.json")[0].contexts == ["one"]

    with (tmp_path / "d.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer", "contexts", "reference"])
        w.writerow(["q", "a", "c1 || c2", ""])
        w.writerow(["q", "a", '["x", "y"]', "ref"])
    rows = load_dataset(tmp_path / "d.csv")
    assert rows[0].contexts == ["c1", "c2"] and rows[0].reference is None
    assert rows[1].contexts == ["x", "y"] and rows[1].reference == "ref"

    with pytest.raises(ValueError):
        load_dataset(tmp_path / "d.txt")


def test_missing_required_fields():
    with pytest.raises(ValueError, match="answer"):
        RAGSample.from_dict({"question": "q"})


def test_bootstrap_ci_is_deterministic():
    values = [0.1, 0.9, 0.5, 0.7]
    assert bootstrap_ci(values) == bootstrap_ci(values)
    assert bootstrap_ci([0.4]) == (0.4, 0.4)


def test_cli_fail_under(jev, tmp_path, monkeypatch, capsys):
    data = tmp_path / "d.jsonl"
    data.write_text(
        json.dumps(
            {
                "question": "What is the capital city of France?",
                "answer": "Berlin hosts famous Oktoberfest celebrations.",
                "contexts": ["Paris is the capital city of France."],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(cli, "build_jev", lambda args: jev)
    out = tmp_path / "out.jsonl"
    code = cli.main(
        ["offline", str(data), "--metrics", "faithfulness", "--out", str(out), "--fail-under", "faithfulness=0.8"]
    )
    assert code == 1
    assert out.exists()
    assert "FAIL: faithfulness" in capsys.readouterr().err
    assert cli.main(["offline", str(data), "--metrics", "faithfulness", "--fail-under", "faithfulness=0.01"]) == 0
