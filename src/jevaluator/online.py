"""Online evaluation of live RAG traffic.

Two modes:

* **Inline** - ``await evaluator.evaluate(sample)`` scores one interaction and
  returns the result, e.g. to gate a response on its faithfulness score.
* **Background** - ``evaluator.submit(sample)`` samples and enqueues the
  interaction and returns immediately; worker tasks score it off the request
  path and push results to sinks and rolling-window monitors.

Sync applications (Flask, Django, scripts) can call
:meth:`OnlineEvaluator.start_background_thread`, which runs the workers on a
private event loop in a daemon thread; ``submit`` is then safe to call from any
thread.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .jev import Jev
from .metrics import reference_free_metrics
from .runner import MetricSpec, resolve_metrics, score_sample, to_sample
from .types import MetricResult, RAGSample, SampleResult

logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #


class Sink(Protocol):
    def emit(self, result: SampleResult) -> Any: ...


class JSONLSink:
    """Appends each result as a JSON line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _write(self, line: str) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def emit(self, result: SampleResult) -> None:
        await asyncio.to_thread(self._write, json.dumps(result.to_dict(), ensure_ascii=False))


class LoggingSink:
    """Logs a one-line score summary per result."""

    def __init__(self, logger_: logging.Logger | None = None, level: int = logging.INFO) -> None:
        self.logger = logger_ or logging.getLogger("jevaluator.online")
        self.level = level

    def emit(self, result: SampleResult) -> None:
        scores = " ".join(
            f"{k}={v:.3f}" for k, v in result.scores.items() if v is not None
        )
        self.logger.log(self.level, "jev sample=%s %s", result.sample.id, scores)


class CallbackSink:
    """Forwards results to any (sync or async) callable - metrics backends, queues, DBs."""

    def __init__(self, fn: Callable[[SampleResult], Any]) -> None:
        self.fn = fn

    async def emit(self, result: SampleResult) -> None:
        await _maybe_await(self.fn(result))


# --------------------------------------------------------------------------- #
# Monitors
# --------------------------------------------------------------------------- #


@dataclass
class Alert:
    metric: str
    rolling_mean: float
    threshold: float
    window_size: int
    sample_id: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThresholdMonitor:
    """Alerts when the rolling mean of a metric drops below a threshold.

    Fires ``on_alert(alert)`` at most once per ``cooldown_s`` seconds, and only
    once at least ``min_samples`` scores are in the window.
    """

    def __init__(
        self,
        metric: str,
        threshold: float,
        on_alert: Callable[[Alert], Any],
        *,
        window: int = 100,
        min_samples: int = 20,
        cooldown_s: float = 300.0,
    ) -> None:
        self.metric = metric
        self.threshold = threshold
        self.on_alert = on_alert
        self.window: deque[float] = deque(maxlen=window)
        self.min_samples = min_samples
        self.cooldown_s = cooldown_s
        self._last_alert = float("-inf")

    @property
    def rolling_mean(self) -> float | None:
        return statistics.fmean(self.window) if self.window else None

    async def observe(self, result: SampleResult) -> Alert | None:
        metric = result.metrics.get(self.metric)
        if metric is None or metric.score is None:
            return None
        self.window.append(metric.score)
        mean = self.rolling_mean
        now = time.monotonic()
        if (
            len(self.window) >= self.min_samples
            and mean is not None
            and mean < self.threshold
            and now - self._last_alert >= self.cooldown_s
        ):
            self._last_alert = now
            alert = Alert(self.metric, mean, self.threshold, len(self.window), result.sample.id)
            await _maybe_await(self.on_alert(alert))
            return alert
        return None


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


@dataclass
class OnlineStats:
    submitted: int = 0
    sampled_out: int = 0
    dropped: int = 0
    evaluated: int = 0
    errors: int = 0
    timeouts: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


_STOP = object()


class OnlineEvaluator:
    """Scores live RAG interactions with Jev.

    Defaults to the reference-free metrics (faithfulness, answer relevance and
    context relevance), since live traffic has no ground-truth answer.
    """

    def __init__(
        self,
        jev: Jev,
        metrics: Iterable[MetricSpec] | None = None,
        *,
        sample_rate: float = 1.0,
        workers: int = 4,
        max_queue_size: int = 1000,
        timeout: float | None = 120.0,
        sinks: Sequence[Sink | Callable[[SampleResult], Any]] = (),
        monitors: Sequence[ThresholdMonitor] = (),
        history_size: int = 1000,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        self.jev = jev
        self.metrics = resolve_metrics(metrics if metrics is not None else reference_free_metrics())
        self.sample_rate = sample_rate
        self.n_workers = workers
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        self.sinks = [s if hasattr(s, "emit") else CallbackSink(s) for s in sinks]
        self.monitors = list(monitors)
        self.history: deque[SampleResult] = deque(maxlen=history_size)
        self.stats = OnlineStats()
        self._rng = random.Random(seed)
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # -- inline -------------------------------------------------------------- #

    async def evaluate(self, sample: RAGSample | Mapping[str, Any]) -> SampleResult:
        """Score one interaction now, then forward it to sinks and monitors."""
        sample = to_sample(sample)
        try:
            if self.timeout is None:
                result = await score_sample(sample, self.metrics, self.jev)
            else:
                result = await asyncio.wait_for(
                    score_sample(sample, self.metrics, self.jev), self.timeout
                )
        except asyncio.TimeoutError:
            self.stats.timeouts += 1
            result = SampleResult(
                sample,
                {m.name: MetricResult(m.name, None, error="evaluation timed out") for m in self.metrics},
            )
        self.stats.evaluated += 1
        if any(m.error for m in result.metrics.values()):
            self.stats.errors += 1
        self.history.append(result)
        await self._dispatch(result)
        return result

    async def _dispatch(self, result: SampleResult) -> None:
        for sink in self.sinks:
            try:
                await _maybe_await(sink.emit(result))
            except Exception:
                logger.exception("jevaluator sink %r failed", sink)
        for monitor in self.monitors:
            try:
                await monitor.observe(result)
            except Exception:
                logger.exception("jevaluator monitor %r failed", monitor)

    # -- background ---------------------------------------------------------- #

    @property
    def running(self) -> bool:
        return bool(self._workers)

    async def start(self) -> None:
        """Start the background workers on the current event loop."""
        if self.running:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._workers = [
            asyncio.create_task(self._worker(), name=f"jevaluator-worker-{i}")
            for i in range(self.n_workers)
        ]

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                await self.evaluate(item)
            except Exception:
                self.stats.errors += 1
                logger.exception("jevaluator failed to evaluate a sample")
            finally:
                self._queue.task_done()

    def _should_sample(self) -> bool:
        return self.sample_rate >= 1.0 or self._rng.random() < self.sample_rate

    def _enqueue(self, sample: RAGSample) -> bool:
        assert self._queue is not None
        try:
            self._queue.put_nowait(sample)
            return True
        except asyncio.QueueFull:
            self.stats.dropped += 1
            return False

    def submit(self, sample: RAGSample | Mapping[str, Any]) -> bool:
        """Non-blocking: sample and enqueue an interaction for background scoring.

        Returns ``False`` if the interaction was sampled out or the queue was full
        (when called from another thread, queue-full drops are only counted in
        :attr:`stats`). Never raises on the request path for a full queue.
        """
        if not self.running or self._loop is None:
            raise RuntimeError("OnlineEvaluator is not running; call start() or start_background_thread()")
        self.stats.submitted += 1
        if not self._should_sample():
            self.stats.sampled_out += 1
            return False
        sample = to_sample(sample)
        try:
            in_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            in_loop = False
        if in_loop:
            return self._enqueue(sample)
        self._loop.call_soon_threadsafe(self._enqueue, sample)
        return True

    async def stop(self, drain: bool = True) -> None:
        """Stop the workers. With ``drain=True`` queued samples are scored first."""
        if not self.running or self._queue is None:
            return
        if drain:
            await self._queue.join()
            for _ in self._workers:
                await self._queue.put(_STOP)
            await asyncio.gather(*self._workers, return_exceptions=True)
        else:
            for w in self._workers:
                w.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def __aenter__(self) -> "OnlineEvaluator":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop(drain=True)

    # -- sync apps ----------------------------------------------------------- #

    def start_background_thread(self) -> None:
        """Run the workers on a private event loop in a daemon thread."""
        if self.running:
            return
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start())
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, name="jevaluator-online", daemon=True)
        self._thread.start()
        ready.wait()

    def shutdown(self, drain: bool = True, timeout: float | None = None) -> None:
        """Stop a background thread started with :meth:`start_background_thread`."""
        if self._thread is None or self._loop is None:
            return
        loop = self._loop
        asyncio.run_coroutine_threadsafe(self.stop(drain=drain), loop).result(timeout)
        loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout)
        loop.close()
        self._thread = None

    # -- instrumentation ----------------------------------------------------- #

    def trace(
        self,
        to_sample_fn: Callable[[tuple, dict, Any], RAGSample | Mapping[str, Any]] | None = None,
    ) -> Callable:
        """Decorator that submits every call of a RAG function for evaluation.

        By default the wrapped function must return a :class:`RAGSample` or a
        mapping with ``answer`` and ``contexts``; the question is taken from the
        result, the ``question`` keyword argument, or the first positional
        argument. Pass ``to_sample_fn(args, kwargs, result)`` for anything else.
        The function's return value is passed through unchanged.
        """

        def build(args: tuple, kwargs: dict, result: Any) -> RAGSample | Mapping[str, Any]:
            if to_sample_fn is not None:
                return to_sample_fn(args, kwargs, result)
            if isinstance(result, RAGSample):
                return result
            if isinstance(result, Mapping):
                data = dict(result)
                if not any(k in data for k in ("question", "query", "user_input", "input")):
                    data["question"] = kwargs.get("question", args[0] if args else "")
                return data
            raise TypeError(
                "trace() needs the function to return a RAGSample or mapping, or a to_sample_fn"
            )

        def submit_safely(args: tuple, kwargs: dict, result: Any) -> None:
            try:
                self.submit(build(args, kwargs, result))
            except Exception:
                logger.exception("jevaluator could not submit traced call")

        def decorator(fn: Callable) -> Callable:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    result = await fn(*args, **kwargs)
                    submit_safely(args, kwargs, result)
                    return result

                return async_wrapper

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)
                submit_safely(args, kwargs, result)
                return result

            return sync_wrapper

        return decorator

    def snapshot(self) -> dict[str, dict[str, float | int | None]]:
        """Rolling statistics over the most recent ``history_size`` results."""
        out: dict[str, dict[str, float | int | None]] = {}
        for m in self.metrics:
            values = [
                r.metrics[m.name].score
                for r in self.history
                if m.name in r.metrics and r.metrics[m.name].score is not None
            ]
            out[m.name] = {
                "n": len(values),
                "mean": statistics.fmean(values) if values else None,
                "min": min(values) if values else None,
            }
        return out
