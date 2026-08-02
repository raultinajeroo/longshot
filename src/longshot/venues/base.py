"""Venue client interface, HTTP helper, and registry.

All network access in longshot goes through :func:`http_get_json`:
urllib with a descriptive User-Agent, a 20s timeout, up to 3 retries with
exponential backoff (0.5s, 1.5s, 4.5s), and a polite sleep between
paginated calls. Failures raise :class:`VenueUnavailableError` with the
venue, the URL, and a remedy hint pointing at offline inputs.
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
POLITE_SLEEP_S = 0.25
REMEDY_HINT = (
    "this host may be blocked in your network; use --input "
    "data/bundled/manifold_resolved_sample.jsonl or fixtures/ instead"
)


class VenueUnavailableError(Exception):
    """Raised when a venue cannot be reached or serves unusable data.

    The message always names the venue and includes a remedy hint.
    """

    def __init__(self, venue: str, url: str, detail: str) -> None:
        self.venue = venue
        self.url = url
        self.detail = detail
        super().__init__(
            f"{venue} unavailable: {detail} (url: {url}). Remedy: {REMEDY_HINT}"
        )


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
            # 4xx responses are client errors; retrying will not help.
            if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500:
                break
            if attempt < len(BACKOFF_S) - 1:
                time.sleep(BACKOFF_S[attempt])
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
