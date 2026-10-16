"""Web fetch: pull a URL, extract clean text, cache on disk by URL hash.

- HTML → trafilatura.extract
- PDF (Content-Type or .pdf suffix) → pypdf text extraction
- Cap: 5 MB by default; 5 redirects max
- Cache: data/artifacts/<session>/papers/<sha1(url)>.json (survives resume)
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..config import Config
from ..ids import url_hash
from .base import ToolCtx, ToolResult


def _is_private_ip(host: str) -> bool:
    """Return True if `host` resolves to any private / loopback / link-local /
    reserved IP. Used to block SSRF against the metadata service and intranet
    targets even when the user-supplied URL passes the scheme check.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        # If we can't resolve, be conservative: refuse.
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


class WebFetchTool:
    name = "web_fetch"
    description = (
        "Fetch a URL and return its main text content (HTML → cleaned text; PDF → extracted text). "
        "Returns {url, title?, text, content_type, status, bytes}. "
        "Use after web_search / arxiv_search / pubmed_search to read the actual content."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."},
            "max_chars": {
                "type": "integer",
                "minimum": 200,
                "maximum": 200_000,
                "description": "Truncate text to this many characters (default 30000).",
            },
        },
        "required": ["url"],
    }

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        url = args.get("url", "").strip()
        max_chars = int(args.get("max_chars") or 30_000)
        if not url.startswith(("http://", "https://")):
            return ToolResult(is_error=True, error_message="URL must start with http(s)")

        # SSRF guard. We re-check after each hop via an httpx event_hook so a
        # redirect can't bounce us to 169.254.169.254 / 127.0.0.1 / RFC1918.
        host = urlsplit(url).hostname or ""
        if not host:
            return ToolResult(is_error=True, error_message="URL has no host")
        if await asyncio.to_thread(_is_private_ip, host):
            return ToolResult(
                is_error=True,
                error_message="URL resolves to a private/loopback address",
            )

        cached = await self._read_cache(ctx, url)
        if cached is not None:
            cached = self._truncate(cached, max_chars)
            return ToolResult(
                content=cached,
                duration_ms=int((time.monotonic() - t0) * 1000),
                result_bytes=len(json.dumps(cached)),
            )

        max_bytes = self._cfg.web_fetch.max_bytes

        async def _check_redirect(response: httpx.Response) -> None:
            loc = response.headers.get("location")
            if not loc:
                return
            next_url = httpx.URL(loc) if loc.startswith(("http://", "https://")) else response.url.join(loc)
            next_host = next_url.host
            if not next_host or await asyncio.to_thread(_is_private_ip, next_host):
                raise httpx.RequestError(
                    "redirect to private/loopback address blocked",
                    request=response.request,
                )

        try:
            # Stream so we can abort on size-overflow without buffering the
            # full body in memory.
            async with (
                httpx.AsyncClient(
                    timeout=self._cfg.web_fetch.timeout_seconds,
                    follow_redirects=True,
                    max_redirects=5,
                    headers={"User-Agent": self._cfg.web_fetch.user_agent},
                    event_hooks={"response": [_check_redirect]},
                ) as client,
                client.stream("GET", url) as r,
            ):
                if r.status_code >= 400:
                    return ToolResult(
                        is_error=True,
                        error_message=f"HTTP {r.status_code}",
                        content={"url": url, "status": r.status_code},
                    )
                # Reject upfront if server advertises a too-large body.
                cl = r.headers.get("content-length")
                if cl is not None:
                    try:
                        if int(cl) > max_bytes:
                            return ToolResult(
                                is_error=True,
                                error_message=f"response too large ({cl} bytes advertised)",
                            )
                    except ValueError:
                        pass

                chunks: list[bytes] = []
                total = 0
                async for chunk in r.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        return ToolResult(
                            is_error=True,
                            error_message=f"response too large (>{max_bytes} bytes, stopped streaming)",
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
                final_url = str(r.url)
                status = r.status_code
                headers = r.headers
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"fetch failed: {e}")

        ct = (headers.get("Content-Type") or "").lower()
        is_pdf = "application/pdf" in ct or url.lower().endswith(".pdf")
        try:
            if is_pdf:
                text = await asyncio.to_thread(_extract_pdf, body)
                title: str | None = None
            else:
                text, title = await asyncio.to_thread(
                    _extract_html, body.decode("utf-8", errors="replace"), url
                )
        except Exception as e:
            return ToolResult(
                is_error=True, error_message=f"extraction failed: {e}"
            )

        payload: dict[str, Any] = {
            "url": final_url,
            "title": title,
            "text": text,
            "content_type": ct,
            "status": status,
            "bytes": total,
        }
        await self._write_cache(ctx, url, payload)
        payload = self._truncate(payload, max_chars)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(json.dumps(payload)),
        )

    # ----------------------------- cache --------------------------------- #

    def _cache_path(self, ctx: ToolCtx, url: str) -> Path | None:
        if ctx.session_id is None:
            return None
        return (
            self._cfg.session_artifact_dir(ctx.session_id)
            / "papers"
            / f"{url_hash(url)}.json"
        )

    async def _read_cache(self, ctx: ToolCtx, url: str) -> dict[str, Any] | None:
        p = self._cache_path(ctx, url)
        if p is None or not p.exists():
            return None

        def _do() -> dict[str, Any]:
            return json.loads(p.read_text())

        return await asyncio.to_thread(_do)

    async def _write_cache(
        self, ctx: ToolCtx, url: str, payload: dict[str, Any]
    ) -> None:
        p = self._cache_path(ctx, url)
        if p is None:
            return

        def _do() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, default=str, ensure_ascii=False))
            tmp.replace(p)

        await asyncio.to_thread(_do)

    @staticmethod
    def _truncate(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
        text = payload.get("text") or ""
        if len(text) > max_chars:
            payload = {**payload, "text": text[:max_chars], "truncated": True}
        return payload


# --------------------------------------------------------------------------- #
# Extractors (sync; run via to_thread)


def _extract_html(html: str, url: str) -> tuple[str, str | None]:
    import trafilatura

    if "<article" in html[:5000].lower():
        text, title = _extract_jats(html)
        if text:
            return text, title

    extracted = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=True
    )
    title = None
    md = trafilatura.metadata.extract_metadata(html)
    if md:
        title = md.title
    return extracted or "", title


def _extract_jats(xml: str) -> tuple[str, str | None]:
    """Extract article prose from Europe PMC JATS XML without retaining markup noise."""
    from xml.etree import ElementTree as ET

    upper = xml.upper()
    if "<!ENTITY" in upper or re.search(r"<!DOCTYPE[^>]*\[", xml, flags=re.IGNORECASE | re.DOTALL):
        return "", None
    # PMC JATS declares its public DTD, but does not require it for parsing.
    # Remove the external declaration so ElementTree never attempts resolution.
    xml = re.sub(r"<!DOCTYPE[^>]*>", "", xml, count=1, flags=re.IGNORECASE | re.DOTALL)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return "", None

    def local_name(element) -> str:
        return element.tag.split("}")[-1]

    def normalized_text(element) -> str:
        return " ".join("".join(element.itertext()).split())

    title_element = root.find(".//{*}article-meta/{*}title-group/{*}article-title")
    if title_element is None:
        title_element = root.find(".//{*}article-title")
    title = normalized_text(title_element) if title_element is not None else None

    parts: list[str] = []
    ignored = {"ref-list", "back", "fig", "table-wrap", "supplementary-material", "fn-group"}

    def append_prose(element) -> None:
        for child in element:
            tag = local_name(child)
            if tag in ignored or tag == "title":
                continue
            if tag == "p":
                text = normalized_text(child)
                if text:
                    parts.append(text)
                continue
            if tag == "sec":
                heading = child.find("./{*}title")
                if heading is not None and normalized_text(heading):
                    parts.append(f"\n## {normalized_text(heading)}")
                append_prose(child)
                continue
            append_prose(child)

    abstract = root.find(".//{*}abstract")
    if abstract is not None:
        parts.append("## Abstract")
        append_prose(abstract)
    body = root.find(".//{*}body")
    if body is not None:
        append_prose(body)

    return "\n\n".join(parts), title


def _extract_pdf(data: bytes) -> str:
    import pypdf

    reader = pypdf.PdfReader(BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)
