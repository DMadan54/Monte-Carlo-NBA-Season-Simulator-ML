"""
Shared nba_api configuration: browser-like headers, TLS impersonation,
and retry logic for stats.nba.com connection failures.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from requests.exceptions import ConnectionError, RequestException, Timeout

T = TypeVar("T")

NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_RETRIES = 5
BASE_DELAY_SEC = 2.0
REQUEST_TIMEOUT_SEC = 60

_configured = False


def configure_nba_api() -> None:
    """Apply headers and optional curl_cffi session patching once per process."""
    global _configured
    if _configured:
        return

    from nba_api.library.http import NBAHTTP
    from nba_api.stats.library.http import NBAStatsHTTP

    NBAStatsHTTP.headers = NBA_HEADERS.copy()

    try:
        from curl_cffi import requests as curl_requests

        session = curl_requests.Session(impersonate="chrome120")
        session.get("https://www.nba.com/stats/", timeout=REQUEST_TIMEOUT_SEC)

        if hasattr(NBAStatsHTTP, "get_session"):
            NBAStatsHTTP.get_session = lambda self: session  # type: ignore[method-assign]
        else:
            import requests as std_requests

            original_get = std_requests.get

            def patched_get(url, **kwargs):
                if "stats.nba.com" in str(url):
                    return session.get(url, **kwargs)
                return original_get(url, **kwargs)

            std_requests.get = patched_get

        print("Using curl_cffi browser impersonation for nba_api requests.")
    except ImportError:
        print(
            "curl_cffi not installed — using default requests with browser headers. "
            "If pulls keep failing, run: pip install curl_cffi"
        )

    _configured = True


def clear_nba_session() -> None:
    """Reset cached HTTP sessions after a failed request."""
    try:
        from nba_api.library.http import NBAHTTP

        if hasattr(NBAHTTP, "clear_session"):
            NBAHTTP.clear_session()
        else:
            NBAHTTP._session = None
            from nba_api.stats.library.http import NBAStatsHTTP

            NBAStatsHTTP._session = None
    except Exception:
        pass


def fetch_with_retry(fetch_fn: Callable[[], T], label: str = "request") -> T:
    """Run an nba_api call with exponential backoff on transient network errors."""
    configure_nba_api()
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_fn()
        except (ConnectionError, Timeout, RequestException) as err:
            last_err = err
            clear_nba_session()

            if attempt == MAX_RETRIES:
                break

            delay = BASE_DELAY_SEC * (2 ** (attempt - 1))
            print(f"  {label} failed (attempt {attempt}/{MAX_RETRIES}): {err}")
            print(f"  Retrying in {delay:.1f}s...")
            time.sleep(delay)

    assert last_err is not None
    raise last_err
