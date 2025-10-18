# apps/backend/ingest.py
"""
Generic ingestion pipeline for arbitrary URLs
Stores ingested documents into the Document ORM model
"""

import asyncio
import logging
from typing import List
import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from database import AsyncSessionLocal
from models import Document

logger = logging.getLogger(__name__)


# ----------------------------
# HTML Fetch + Parse
# ----------------------------

async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    """Fetch HTML content of a page."""
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"❌ Failed to fetch {url}: {e}")
        return ""


def parse_text(html: str) -> str:
    """Extract visible text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts and styles
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text


# ----------------------------
# DB Persistence
# ----------------------------

async def ingest_url(url: str) -> dict:
    """Fetch, parse, and store a single URL in DB."""
    async with httpx.AsyncClient() as client:
        html = await fetch_page(client, url)
        if not html:
            return {"url": url, "status": "failed_fetch"}

        content = parse_text(html)
        if not content:
            logger.warning(f"⚠️ No content extracted from {url}")
            return {"url": url, "status": "empty_content"}

        async with AsyncSessionLocal() as session:
            # Upsert (avoid duplicates by URL)
            stmt = sqlite_insert(Document).values(url=url, content=content)
            stmt = stmt.on_conflict_do_nothing(index_elements=["url"])
            await session.execute(stmt)
            await session.commit()

            logger.info(f"✅ Ingested URL: {url}")
            return {"url": url, "status": "ingested"}


async def ingest_urls(urls: List[str]) -> List[dict]:
    """Ingest multiple URLs concurrently."""
    tasks = [ingest_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results


# ----------------------------
# CLI Test
# ----------------------------
if __name__ == "__main__":
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]
    asyncio.run(ingest_urls(test_urls))
