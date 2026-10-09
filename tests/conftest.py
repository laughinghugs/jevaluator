import re

import pytest

from jevaluator import CallableBackend, Jev, RAGSample


def _between(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.S)
    return m.group(1) if m else ""


_STOPWORDS = {"what", "which", "where", "when", "does", "that", "this", "with", "from"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in _STOPWORDS}


def keyword_judge(system: str, question: str) -> tuple[float, str]:
    """Deterministic stand-in for Jev: P(yes) from word overlap between the tagged sections."""
    if "<claim>" in question:
        target, evidence = _between(question, "claim"), _between(question, "context")
    elif "<statement>" in question:
        target, evidence = _between(question, "statement"), _between(question, "context")
    elif "<passage>" in question:
        target, evidence = _between(question, "question"), _between(question, "passage")
    elif "<reference>" in question:
        target, evidence = _between(question, "reference"), _between(question, "answer")
    elif "<answer>" in question:
        target, evidence = _between(question, "question"), _between(question, "answer")
    else:
        return 0.5, "no rule"
    t = _words(target)
    overlap = len(t & _words(evidence)) / len(t) if t else 0.0
    return 0.05 + 0.9 * overlap, f"overlap={overlap:.2f}"


def sentence_claims(system: str, prompt: str) -> str:
    answer = _between(prompt, "answer")
    return "\n".join(s.strip() for s in answer.split(".") if s.strip()) or "NONE"


@pytest.fixture
def jev() -> Jev:
    return Jev(CallableBackend(keyword_judge, sentence_claims), retry_backoff=0)


@pytest.fixture
def good_sample() -> RAGSample:
    return RAGSample(
        id="good",
        question="What is the capital city of France?",
        answer="Paris is the capital city of France.",
        contexts=["Paris is the capital city of France.", "Bananas contain potassium."],
        reference="The capital city of France is Paris.",
    )


@pytest.fixture
def bad_sample() -> RAGSample:
    return RAGSample(
        id="bad",
        question="What is the capital city of France?",
        answer="Berlin hosts famous Oktoberfest celebrations.",
        contexts=["Bananas contain potassium.", "Paris is the capital city of France."],
        reference="The capital city of France is Paris.",
    )
