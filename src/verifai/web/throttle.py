"""Limits for a server that strangers can reach.

Every check fetches from public APIs (Crossref, PubMed, Wikipedia, and
others) and may run a local model. Without limits one client can exhaust
the machine and get the server's polite-pool identity throttled upstream.

The limits live in this process. Several workers or serverless instances
each keep their own counts, so put a shared limit in front of those.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class Throttled(Exception):  # noqa: N818 - reads as a state, like TimeoutError
    """A check was refused. ``status`` is 429 for a rate limit, 503 when busy."""

    def __init__(self, status: int, retry_after: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = max(1, retry_after)
        self.message = message


class RateLimiter:
    """Sliding windows per client: so many checks a minute and an hour.

    A limit of 0 turns that window off. The table of clients is capped, and
    the least recently seen client is dropped first.
    """

    def __init__(
        self,
        *,
        per_minute: int,
        per_hour: int,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.windows = tuple(
            (limit, seconds) for limit, seconds in ((per_minute, 60), (per_hour, 3600)) if limit > 0
        )
        self.max_clients = max_clients
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.windows)

    def admit(self, client: str) -> None:
        """Record a check for ``client`` or raise :class:`Throttled`."""
        if not self.windows:
            return
        now = self._clock()
        longest = max(seconds for _, seconds in self.windows)
        with self._lock:
            hits = self._hits.pop(client, None) or deque()
            while hits and now - hits[0] >= longest:
                hits.popleft()
            for limit, seconds in self.windows:
                recent = [stamp for stamp in hits if now - stamp < seconds]
                if len(recent) >= limit:
                    self._hits[client] = hits
                    wait = math.ceil(seconds - (now - recent[-limit]))
                    raise Throttled(429, wait, _rate_message(limit, seconds, wait))
            hits.append(now)
            self._hits[client] = hits
            while len(self._hits) > self.max_clients:
                self._hits.popitem(last=False)


class CheckSlots:
    """At most ``size`` checks run at once. Others wait, then are refused."""

    def __init__(self, size: int, *, wait_seconds: float) -> None:
        self._slots = threading.BoundedSemaphore(size)
        self.wait_seconds = wait_seconds

    @contextmanager
    def hold(self) -> Iterator[None]:
        if not self._slots.acquire(timeout=self.wait_seconds):
            raise Throttled(
                503,
                math.ceil(self.wait_seconds) or 5,
                "The server is busy with other checks. Try again in a minute.",
            )
        try:
            yield
        finally:
            self._slots.release()


def _rate_message(limit: int, seconds: int, wait: int) -> str:
    window = "minute" if seconds == 60 else "hour"
    noun = "check" if limit == 1 else "checks"
    return (
        f"This server allows {limit} {noun} per {window} from one address. "
        f"Try again in {_duration(wait)}, or run VerifAI on your own machine."
    )


def _duration(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} seconds"
    return f"{math.ceil(seconds / 60)} minutes"
