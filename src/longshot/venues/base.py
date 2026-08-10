"""Venue client interface, HTTP helper, and registry.

All network access in longshot goes through :func:`http_get_json`:
urllib with a descriptive User-Agent, a 20s timeout, up to 3 retries with
exponential backoff (0.5s, 1.5s, 4.5s), and a polite sleep between
paginated calls. Failures raise :class:`VenueUnavailableError` with the
venue, the URL, and a remedy hint pointing at offline inputs.

Other 4xx responses abort immediately, but **429 is retried** on a longer
ladder (5s, 15s, 45s) and honours ``Retry-After``. It is the one client
error that means "try again", and treating it like a 404 turns an ordinary
rate limit into a failed run.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, Iterator

from ..types import MarketSeries

USER_AGENT = "longshot/0.1 (+https://github.com/)"
TIMEOUT_S = 20
BACKOFF_S = (0.5, 1.5, 4.5)
# 429 gets its own, longer ladder: the generic backoff is tuned for a flaky
# connection, while a rate limit needs the bucket to actually refill.
RATE_LIMIT_BACKOFF_S = (5.0, 15.0, 45.0)
TOO_MANY_REQUESTS = 429
POLITE_SLEEP_S = 0.25
REMEDY_HINT = (
    "this host may be blocked in your network; use --input "
    "data/bundled/manifold_resolved_sample.jsonl or fixtures/ instead"
)


class VenueUnavailableError(Exception):
    """Raised when a venue cannot be reached or serves unusable data.

    The message always names the venue and includes a remedy hint. Callers
    that know the failure is *not* a reachability problem pass their own
    ``remedy``: a budget guard tripping and a blocked host are different
    events, and reporting the first as the second sends the reader hunting
    a network fault that does not exist.
    """

    def __init__(
        self, venue: str, url: str, detail: str, remedy: str = REMEDY_HINT
    ) -> None:
        self.venue = venue
        self.url = url
        self.detail = detail
        self.remedy = remedy
        super().__init__(
            f"{venue} unavailable: {detail} (url: {url}). Remedy: {remedy}"
        )


def _retry_after(exc: Exception) -> float:
    """Seconds requested by a ``Retry-After`` header; 0 when absent or odd.

    Only the delta-seconds form is honoured. The HTTP-date form is legal but
    unused by the venues here, and guessing at clock skew to parse it would
    trade a known wait for an unknown one.
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        return 0.0
    try:
        return max(0.0, float(headers.get("Retry-After", "")))
    except (TypeError, ValueError):
        return 0.0


def http_get_json(
    url: str,
    *,
    venue: str,
    extra_headers: dict | None = None,
    timeout: float = TIMEOUT_S,
) -> object:
    """GET ``url`` and parse the JSON body, with retries and clear errors."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    last: Exception | None = None
    for attempt in range(len(BACKOFF_S)):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last = exc
            rate_limited = (
                isinstance(exc, urllib.error.HTTPError) and exc.code == TOO_MANY_REQUESTS
            )
            # 4xx responses are client errors and retrying will not help --
            # except 429, whose entire contract is "retry me later".
            if (
                isinstance(exc, urllib.error.HTTPError)
                and 400 <= exc.code < 500
                and not rate_limited
            ):
                break
            if attempt < len(BACKOFF_S) - 1:
                delay = BACKOFF_S[attempt]
                if rate_limited:
                    delay = max(RATE_LIMIT_BACKOFF_S[attempt], _retry_after(exc))
                time.sleep(delay)
    raise VenueUnavailableError(venue, url, str(last))


class VenueClient(ABC):
    """Read-only source of resolved markets with probability histories."""

    venue: str = ""

    @abstractmethod
    def fetch_resolved(
        self,
        max_markets: int = 250,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Iterator[MarketSeries]:
        """Yield resolved markets with probability histories.

        Implementations must raise :class:`VenueUnavailableError` when the
        venue cannot be reached, with a remedy hint in the message.
        """


VENUES: dict[str, type[VenueClient]] = {}


def register(cls: type[VenueClient]) -> type[VenueClient]:
    """Class decorator adding a venue client to the registry."""
    VENUES[cls.venue] = cls
    return cls


def get_client(venue: str, **kwargs) -> VenueClient:
    """Instantiate a registered venue client by name."""
    if venue not in VENUES:
        # Import lazily so all clients self-register on first use.
        from . import fixture, kalshi, manifold, polymarket  # noqa: F401
    if venue not in VENUES:
        raise ValueError(
            f"unknown venue {venue!r}; available: {sorted(VENUES)}"
        )
    return VENUES[venue](**kwargs)
