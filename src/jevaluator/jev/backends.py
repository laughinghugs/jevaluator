"""Backends that turn a yes/no evaluation question into a probability.

A backend is anything with two async methods:

* ``yes_probability(system, question)`` -> ``(probability, rationale)``, where
  ``probability`` is P(the answer to ``question`` is "Yes").
* ``generate(system, prompt)`` -> ``str``, used for auxiliary steps such as
  splitting an answer into atomic claims. Backends that cannot generate text
  raise ``NotImplementedError`` and metrics fall back to heuristics.

Three backends ship with the package:

* :class:`LogprobBackend` - the preferred way to run Jev. Calls any
  OpenAI-compatible chat-completions server (vLLM, TGI, llama.cpp, ...) hosting
  the Jev judge model, and reads P(Yes) straight from the next-token
  distribution.
* :class:`ClaudeBackend` - uses Claude via the Anthropic SDK. The Messages API
  does not expose token log-probabilities, so the probability is the model's
  stated confidence (optionally averaged over several samples); fit a
  calibrator on labelled data if you need it to be well calibrated.
* :class:`CallableBackend` - wraps your own function, for any other judge.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import urllib.request
from typing import Any, Awaitable, Callable, Protocol, Union, runtime_checkable


class JevError(RuntimeError):
    """Raised when the judge cannot produce a usable judgment."""


class JevRefusal(JevError):
    """Raised when the judge model declines to answer. Not retried."""


@runtime_checkable
class JudgeBackend(Protocol):
    async def yes_probability(self, system: str, question: str) -> tuple[float, str | None]: ...

    async def generate(self, system: str, prompt: str) -> str: ...


def _clamp(p: float) -> float:
    if math.isnan(p):
        raise JevError("judge returned NaN probability")
    return min(1.0, max(0.0, float(p)))


# --------------------------------------------------------------------------- #
# Logprob backend (OpenAI-compatible chat completions)
# --------------------------------------------------------------------------- #

_YES_TOKENS = frozenset({"yes", "y", "true"})
_NO_TOKENS = frozenset({"no", "n", "false"})


def _normalize_token(token: str) -> str:
    # SentencePiece / BPE markers plus surrounding whitespace and punctuation.
    return token.replace("▁", "").replace("Ġ", "").strip().strip(".,:;!\"'").lower()


def yes_probability_from_logprobs(top_logprobs: list[dict[str, Any]]) -> float | None:
    """P(Yes) renormalized over the Yes/No mass in a top-k logprob list.

    Returns ``None`` if neither a Yes nor a No token appears in the list.
    """
    yes = no = 0.0
    for entry in top_logprobs:
        token = _normalize_token(str(entry.get("token", "")))
        mass = math.exp(float(entry["logprob"]))
        if token in _YES_TOKENS:
            yes += mass
        elif token in _NO_TOKENS:
            no += mass
    if yes + no == 0.0:
        return None
    return yes / (yes + no)


class LogprobBackend:
    """Jev served behind an OpenAI-compatible ``/chat/completions`` endpoint.

    The judge is asked to answer with a single word and the probability is read
    from the logprobs of the first generated token:
    ``P(Yes) = p(yes) / (p(yes) + p(no))`` summed over casing/whitespace variants.
    """

    answer_instruction = "Answer with exactly one word: Yes or No."

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        top_logprobs: int = 20,
        timeout: float = 60.0,
        generate_max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.generate_max_tokens = generate_max_tokens
        self.extra_body = extra_body or {}

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    async def _chat(self, system: str, user: str, **params: Any) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            **params,
            **self.extra_body,
        }
        return await asyncio.to_thread(self._post, body)

    async def yes_probability(self, system: str, question: str) -> tuple[float, str | None]:
        data = await self._chat(
            system,
            f"{question}\n\n{self.answer_instruction}",
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        try:
            choice = data["choices"][0]
            first = choice["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise JevError(f"response has no logprobs; is logprobs supported by the server? {data!r:.300}") from exc

        p = yes_probability_from_logprobs(first.get("top_logprobs") or [])
        if p is None:
            # Fall back to the sampled token alone.
            token = _normalize_token(first.get("token", ""))
            prob = math.exp(float(first.get("logprob", 0.0)))
            if token in _YES_TOKENS:
                p = prob
            elif token in _NO_TOKENS:
                p = 1.0 - prob
            else:
                raise JevError(f"judge answered {first.get('token')!r}, expected Yes or No")
        return _clamp(p), None

    async def generate(self, system: str, prompt: str) -> str:
        data = await self._chat(system, prompt, max_tokens=self.generate_max_tokens)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JevError(f"unexpected chat completion response: {data!r:.300}") from exc


# --------------------------------------------------------------------------- #
# Claude backend (Anthropic SDK)
# --------------------------------------------------------------------------- #

_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "verdict": {"type": "string", "enum": ["yes", "no"]},
        "probability": {"type": "number"},
    },
    "required": ["rationale", "verdict", "probability"],
    "additionalProperties": False,
}

_CLAUDE_INSTRUCTION = (
    "Respond with a JSON object containing: `rationale` (one or two sentences explaining "
    "your judgment), `verdict` (\"yes\" or \"no\"), and `probability` (a number between "
    "0 and 1: your probability that the correct answer to the evaluation question is Yes)."
)


class ClaudeBackend:
    """Jev powered by Claude through the Anthropic Python SDK.

    Requires ``pip install anthropic`` (or ``pip install jevaluator[claude]``).
    Credentials resolve the usual way (``ANTHROPIC_API_KEY``, ``ant auth login``, ...).

    Server-side refusal fallbacks are enabled by default (``fallbacks="default"``),
    so a request declined by a safety classifier is retried on a fallback model
    inside the same call. Pass ``fallbacks=False`` to turn this off.
    """

    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        client: Any = None,
        n_samples: int = 1,
        effort: str | None = None,
        max_tokens: int = 16000,
        fallbacks: bool = True,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError("ClaudeBackend requires `pip install anthropic`") from exc
            client = anthropic.AsyncAnthropic()
        self.client = client
        self.model = model
        self.n_samples = max(1, n_samples)
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks

    async def _create(self, system: str, user: str, output_format: dict[str, Any] | None) -> str:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict[str, Any] = {}
        if output_format is not None:
            output_config["format"] = output_format
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            params["output_config"] = output_config
        if self.fallbacks:
            params["betas"] = [self._FALLBACK_BETA]
            params["fallbacks"] = "default"

        response = await self.client.beta.messages.create(**params)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise JevRefusal(f"judge model declined to evaluate (category={category})")
        if response.stop_reason == "max_tokens":
            raise JevError("judge response hit max_tokens before finishing")
        return "".join(b.text for b in response.content if b.type == "text")

    async def _judge_once(self, system: str, question: str) -> tuple[float, str | None]:
        text = await self._create(
            system,
            f"{question}\n\n{_CLAUDE_INSTRUCTION}",
            {"type": "json_schema", "schema": _JUDGMENT_SCHEMA},
        )
        try:
            data = json.loads(text)
            return _clamp(float(data["probability"])), data.get("rationale")
        except (ValueError, KeyError, TypeError) as exc:
            raise JevError(f"could not parse judge output: {text!r:.300}") from exc

    async def yes_probability(self, system: str, question: str) -> tuple[float, str | None]:
        results = await asyncio.gather(
            *(self._judge_once(system, question) for _ in range(self.n_samples))
        )
        probability = sum(p for p, _ in results) / len(results)
        return probability, results[0][1]

    async def generate(self, system: str, prompt: str) -> str:
        return await self._create(system, prompt, None)


# --------------------------------------------------------------------------- #
# Callable backend
# --------------------------------------------------------------------------- #

JudgeFn = Callable[[str, str], Union[float, tuple[float, Union[str, None]], Awaitable[Any]]]
GenerateFn = Callable[[str, str], Union[str, Awaitable[str]]]


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class CallableBackend:
    """Adapts plain (sync or async) functions into a Jev backend.

    ``judge_fn(system, question)`` returns a probability or ``(probability, rationale)``;
    ``generate_fn(system, prompt)`` (optional) returns text.
    """

    def __init__(self, judge_fn: JudgeFn, generate_fn: GenerateFn | None = None) -> None:
        self.judge_fn = judge_fn
        self.generate_fn = generate_fn

    async def yes_probability(self, system: str, question: str) -> tuple[float, str | None]:
        result = await _maybe_await(self.judge_fn(system, question))
        if isinstance(result, tuple):
            return _clamp(result[0]), result[1]
        return _clamp(result), None

    async def generate(self, system: str, prompt: str) -> str:
        if self.generate_fn is None:
            raise NotImplementedError("this backend has no generate_fn")
        return await _maybe_await(self.generate_fn(system, prompt))
