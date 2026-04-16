import logging
import os
import re
import time
from html.parser import HTMLParser
from typing import Dict, List
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_URLS = ["https://www.health.harvard.edu/topics/nutrition"]
MAX_CONTEXT_CHARS = 24_000
SOURCE_CACHE_TTL_SECONDS = 60 * 60

_cache: Dict[str, tuple[float, str]] = {}


class ReadableTextParser(HTMLParser):
    """Extract readable text while skipping scripts, styles, and navigation noise."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth:
            return

        text = " ".join(data.split())
        if len(text) >= 3:
            self._parts.append(text)

    def get_text(self) -> str:
        raw_text = "\n".join(self._parts)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        return raw_text.strip()


def get_source_urls() -> List[str]:
    raw_urls = os.getenv("SOURCE_URLS", "").strip()
    if not raw_urls:
        return DEFAULT_SOURCE_URLS

    return [url.strip() for url in raw_urls.split(",") if url.strip()]


def fetch_url_text(url: str, force_refresh: bool = False) -> str:
    cached = _cache.get(url)
    now = time.time()

    if cached and not force_refresh and now - cached[0] < SOURCE_CACHE_TTL_SECONDS:
        return cached[1]

    logger.info("Fetching context URL: %s", url)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RAGChatbot/1.0; +https://localhost)"
        },
    )

    with urlopen(request, timeout=20) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="ignore")

    parser = ReadableTextParser()
    parser.feed(html)
    text = parser.get_text()

    if not text:
        raise ValueError(f"No readable text found at {url}")

    _cache[url] = (now, text)
    return text


def get_url_context(force_refresh: bool = False) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []

    for url in get_source_urls():
        text = fetch_url_text(url, force_refresh=force_refresh)
        chunks.append(
            {
                "text": text[:MAX_CONTEXT_CHARS],
                "source": url,
            }
        )

    return chunks
