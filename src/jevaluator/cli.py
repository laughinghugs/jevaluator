"""Command-line entry point: ``jevaluator offline DATASET ...``."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from .jev import ClaudeBackend, Jev, LogprobBackend, PlattCalibrator
from .metrics import BUILTIN_METRICS
from .offline import OfflineEvaluator, load_dataset


def _parse_pairs(values: Sequence[str], flag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in values:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"{flag} expects metric=value, got {item!r}")
        out[name.strip()] = float(value)
    return out


def build_jev(args: argparse.Namespace) -> Jev:
    if args.backend == "logprob":
        if not args.base_url or not args.model:
            raise SystemExit("--backend logprob needs --base-url and --model")
        backend = LogprobBackend(
            args.base_url, args.model, api_key=os.environ.get(args.api_key_env) if args.api_key_env else None
        )
    else:
        backend = ClaudeBackend(model=args.model or "claude-opus-5-5", n_samples=args.samples)
    calibrators = {}
    for item in args.calibrator:
        name, _, path = item.partition("=")
        calibrators[name] = PlattCalibrator.load(path)
    return Jev(backend, calibrators=calibrators, max_concurrency=args.concurrency)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevaluator", description="Evaluate RAG outputs with Jev.")
    sub = parser.add_subparsers(dest="command", required=True)

    off = sub.add_parser("offline", help="score a dataset (.jsonl/.json/.csv)")
    off.add_argument("dataset")
    off.add_argument(
        "--metrics",
        default=",".join(BUILTIN_METRICS),
        help=f"comma-separated metric names (default: all of {', '.join(BUILTIN_METRICS)})",
    )
    off.add_argument("--backend", choices=["logprob", "claude"], default="logprob")
    off.add_argument("--base-url", help="OpenAI-compatible base URL serving Jev (logprob backend)")
    off.add_argument("--model", help="judge model name")
    off.add_argument("--api-key-env", default="JEV_API_KEY", help="env var holding the Jev API key")
    off.add_argument("--samples", type=int, default=1, help="judgments averaged per question (claude backend)")
    off.add_argument("--concurrency", type=int, default=8, help="max concurrent judge calls")
    off.add_argument("--calibrator", action="append", default=[], metavar="METRIC=PATH")
    off.add_argument("--threshold", action="append", default=[], metavar="METRIC=VALUE")
    off.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="exit 1 if the metric's mean is below VALUE (for CI gates)",
    )
    off.add_argument("--out", help="write per-sample results to this .jsonl file")
    off.add_argument("--csv", help="write a per-sample score table to this .csv file")

    args = parser.parse_args(argv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    evaluator = OfflineEvaluator(
        build_jev(args), metrics, thresholds=_parse_pairs(args.threshold, "--threshold")
    )
    dataset = load_dataset(args.dataset)
    done = 0

    def progress(_result) -> None:
        nonlocal done
        done += 1
        print(f"\rscored {done}/{len(dataset)}", end="", file=sys.stderr, flush=True)

    report = evaluator.evaluate(dataset, on_result=progress)
    print(file=sys.stderr)
    print(report.format_table())
    if args.out:
        report.to_jsonl(args.out)
    if args.csv:
        report.to_csv(args.csv)

    failed = False
    summary = report.summary()
    for name, minimum in _parse_pairs(args.fail_under, "--fail-under").items():
        mean = summary[name].mean if name in summary else None
        if mean is None or mean < minimum:
            print(f"FAIL: {name} mean {mean} < {minimum}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
