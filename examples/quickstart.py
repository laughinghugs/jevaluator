"""End-to-end tour of jevaluator: offline batch evaluation, then online monitoring.

Pick a Jev backend:

* Self-hosted Jev model behind an OpenAI-compatible server (vLLM, TGI, ...):
      JEV_BASE_URL=http://localhost:8000/v1 JEV_MODEL=jev-judge python examples/quickstart.py
* Claude (needs `pip install anthropic` and Anthropic credentials):
      JEV_BACKEND=claude python examples/quickstart.py
"""

import asyncio
import os

from jevaluator import (
    ClaudeBackend,
    Jev,
    LoggingSink,
    LogprobBackend,
    OfflineEvaluator,
    OnlineEvaluator,
    RAGSample,
    ThresholdMonitor,
)


def make_jev() -> Jev:
    if os.environ.get("JEV_BACKEND") == "claude":
        backend = ClaudeBackend()
    else:
        backend = LogprobBackend(
            base_url=os.environ.get("JEV_BASE_URL", "http://localhost:8000/v1"),
            model=os.environ.get("JEV_MODEL", "jev-judge"),
            api_key=os.environ.get("JEV_API_KEY"),
        )
    return Jev(backend, max_concurrency=8)


DATASET = [
    RAGSample(
        question="When was the Eiffel Tower completed?",
        answer="The Eiffel Tower was completed in 1889 for the World's Fair.",
        contexts=[
            "The Eiffel Tower was built for the 1889 Exposition Universelle (World's Fair).",
            "Construction finished in March 1889.",
        ],
        reference="It was completed in 1889.",
    ),
    RAGSample(
        question="When was the Eiffel Tower completed?",
        answer="It was completed in 1925 and was designed by Antoni Gaudi.",  # hallucinated
        contexts=["The Eiffel Tower was built for the 1889 Exposition Universelle (World's Fair)."],
        reference="It was completed in 1889.",
    ),
]


async def main() -> None:
    jev = make_jev()

    # ---- Offline: score a labelled dataset -------------------------------- #
    report = await OfflineEvaluator(jev, thresholds={"faithfulness": 0.7}).aevaluate(DATASET)
    print(report.format_table())
    for failure in report.failures("faithfulness"):
        print("\nUnfaithful answer:", failure.sample.answer)
        for claim in failure.metrics["faithfulness"].details["claims"]:
            print(f"  P(supported)={claim['probability']:.2f}  {claim['claim']}")

    # ---- Online: monitor live traffic -------------------------------------- #
    monitor = ThresholdMonitor(
        "faithfulness", threshold=0.7, on_alert=lambda a: print("ALERT:", a), min_samples=2
    )
    async with OnlineEvaluator(
        jev, sample_rate=1.0, sinks=[LoggingSink()], monitors=[monitor]
    ) as online:

        @online.trace()
        async def answer_question(question: str) -> dict:
            # ... your retriever + generator here ...
            sample = DATASET[1]
            return {"answer": sample.answer, "contexts": sample.contexts}

        for _ in range(2):
            await answer_question("When was the Eiffel Tower completed?")

        # Inline mode: gate a response on its score before returning it.
        result = await online.evaluate(DATASET[1])
        if not result.passed({"faithfulness": 0.7}):
            print("Blocked response, faithfulness =", result.scores["faithfulness"])

    print("Online stats:", online.stats.to_dict())
    print("Rolling:", online.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
