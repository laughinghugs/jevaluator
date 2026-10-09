"""Jev: the probabilistic judge that scores every metric."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Mapping

from ..types import JevScore
from .backends import JevRefusal, JudgeBackend
from .calibration import Calibrator
from .prompts import JEV_SYSTEM

logger = logging.getLogger(__name__)


class Jev:
    """Asks a judge backend yes/no evaluation questions and returns probabilities.

    Responsibilities on top of the raw backend:

    * per-criterion calibration (``calibrators={"faithfulness": PlattCalibrator(...)}``),
    * bounded concurrency across all metrics and samples,
    * retries with exponential backoff,
    * an LRU cache so identical questions (e.g. the same chunk judged by two
      context metrics) are only asked once.
    """

    def __init__(
        self,
        backend: JudgeBackend,
        *,
        system_prompt: str = JEV_SYSTEM,
        calibrators: Mapping[str, Calibrator] | None = None,
        max_concurrency: int = 8,
        retries: int = 2,
        retry_backoff: float = 1.0,
        cache_size: int = 10_000,
    ) -> None:
        self.backend = backend
        self.system_prompt = system_prompt
        self.calibrators = dict(calibrators or {})
        self.max_concurrency = max_concurrency
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, str | None]] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None

    def _semaphore(self) -> asyncio.Semaphore:
        # Semaphores bind to an event loop; recreate when used from a new loop
        # (e.g. successive asyncio.run() calls from the sync helpers).
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrency)
            self._sem_loop = loop
            self._inflight.clear()
        return self._sem

    async def _call_with_retries(self, coro_fn, *args):
        attempt = 0
        while True:
            try:
                async with self._semaphore():
                    return await coro_fn(*args)
            except (JevRefusal, NotImplementedError):
                raise
            except Exception as exc:
                if attempt >= self.retries:
                    raise
                delay = self.retry_backoff * (2**attempt)
                logger.warning("Jev call failed (%s); retrying in %.1fs", exc, delay)
                attempt += 1
                await asyncio.sleep(delay)

    async def _raw(self, question: str) -> tuple[float, str | None]:
        key = hashlib.sha256(f"{self.system_prompt}\x00{question}".encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        self._semaphore()  # resets in-flight map if the loop changed
        if key in self._inflight:
            return await asyncio.shield(self._inflight[key])

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            result = await self._call_with_retries(
                self.backend.yes_probability, self.system_prompt, question
            )
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved so unawaited futures don't warn
            raise
        else:
            future.set_result(result)
            if self.cache_size > 0:
                self._cache[key] = result
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return result
        finally:
            self._inflight.pop(key, None)

    async def judge(self, question: str, criterion: str | None = None) -> JevScore:
        """Return P(Yes) for a yes/no evaluation ``question``.

        If a calibrator is registered for ``criterion`` it is applied to the
        backend's raw probability.
        """
        raw, rationale = await self._raw(question)
        calibrator = self.calibrators.get(criterion) if criterion else None
        probability = calibrator(raw) if calibrator else raw
        return JevScore(
            probability=probability, raw_probability=raw, rationale=rationale, criterion=criterion
        )

    async def generate(self, prompt: str) -> str:
        """Free-text generation for auxiliary steps (raises NotImplementedError if unsupported)."""
        return await self._call_with_retries(self.backend.generate, self.system_prompt, prompt)

    def clear_cache(self) -> None:
        self._cache.clear()
