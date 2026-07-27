"""deepweb — breadth of a multi-agent deep search, at retrieval cost.

The expensive part of "deep research" is reasoning, and a multi-agent skill
pays for it N times (one full LLM context per agent). But reasoning doesn't
need to fan out — ONE capable model already in context can synthesize. Only the
SEARCHING needs breadth, and searching the web provider is pure retrieval with
ZERO LLM tokens.

So deepweb takes several sub-queries, fires them all at the web provider, and
fuses the results (Reciprocal Rank Fusion + dedup by URL) into one compact,
budget-trimmed source set. A URL that surfaces across multiple sub-queries ranks
higher — cross-query agreement is a relevance signal. The caller (the in-context
model) does the single synthesis pass over the merged set.

No agents, no per-agent context, no LLM cost for the breadth.
"""
from __future__ import annotations

import re
import socket
import urllib.parse
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network

from .providers.web import WebProvider
from .redactor import redact

# R12 (Aether audit 2026-07-26) — SSRF guard.
# fetch_extract follows URLs that come out of a SEARCH ENGINE, so the target is
# attacker-influenceable: rank a page for a niche query and you choose what this
# fetches. Without a host check that reaches cloud metadata endpoints
# (169.254.169.254), loopback admin panels, and private LAN hosts. The YouTube
# provider already had an allowlist; this path had nothing.
# Ranges are identified via the stdlib's own classification flags
# (is_private / is_loopback / is_link_local / is_reserved) rather than a
# hand-written CIDR table. Two reasons: the stdlib list is maintained against
# the IANA special-purpose registry and cannot drift out of date here, and
# spelling RFC-1918 literals in shipped source trips this project's own
# personal-data scrub gate (tests/test_no_personal_leak.py), which is a rule
# worth keeping strict. Verified against the flags: 10/8, 172.16/12, 192.168/16,
# 127/8, 169.254/16, 0.0.0.0/8, 192.0.0.0/24, 198.18/15, ::1, fc00::/7 and
# fe80::/10 are all caught; public addresses are not.
#
# CGNAT is the one gap — the stdlib marks it neither private nor reserved — so
# it is named explicitly. It is not an RFC-1918 shape, so the scrub gate is fine.
_CGNAT = ip_network("100.64.0.0/10")
_MAX_REDIRECTS = 3


def _is_public_host(host: str) -> bool:
    """True only if every address `host` resolves to is publicly routable.

    Fail-CLOSED: an unresolvable host, or one with any private/loopback/
    link-local address, is refused. Checking every getaddrinfo result (not
    just the first) is what stops a dual-homed name from slipping through.
    """
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not infos:
        return False
    for res in infos:
        try:
            ip = ip_address(res[4][0])
        except ValueError:
            return False
        # An IPv4-mapped IPv6 address (the ::ffff: prefix form) must be judged
        # on the embedded v4 address, or a private target slips through as v6.
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
        if ip.version == 4 and ip in _CGNAT:
            return False
    return True


def _url_host_is_public(url: str) -> bool:
    try:
        return _is_public_host(urllib.parse.urlparse(url).hostname or "")
    except Exception:
        return False

_RRF_K = 60  # standard RRF damping; smooths rank contributions

_UA = ("Mozilla/5.0 (compatible; magpie-search/1.0; +deepweb)")
_DROP_TAGS = ("script", "style", "nav", "aside", "header", "footer", "form",
              "noscript", "svg", "button")


@dataclass
class DeepHit:
    url: str
    title: str
    snippet: str
    score: float = 0.0
    seen_in: list[str] = field(default_factory=list)  # which sub-queries surfaced it
    content: str = ""  # extracted page text, filled only when fetch=True

    def to_dict(self) -> dict:
        d = {"url": self.url, "title": self.title, "snippet": self.snippet,
             "score": round(self.score, 5), "n_queries": len(self.seen_in)}
        if self.content:
            d["content"] = self.content
        return d


def fetch_extract(url: str, *, max_chars: int = 1500, timeout: float = 8.0) -> str:
    """Fetch a URL and return its main text — redacted, whitespace-collapsed,
    truncated. Pure retrieval (no LLM). Fail-safe: '' on any error, never raises.

    Bounded for safety + tokens: http(s) only, response size capped, scripts/
    nav/boilerplate stripped, output truncated to max_chars."""
    if not url.lower().startswith(("http://", "https://")):
        return ""
    # R12: refuse non-public targets before any request is made.
    if not _url_host_is_public(url):
        return ""
    try:
        import httpx
        from bs4 import BeautifulSoup
    except Exception:
        return ""
    try:
        # R12: redirects are followed MANUALLY so every hop is re-validated —
        # `follow_redirects=True` would let a public URL 302 to
        # 169.254.169.254 with only the first host ever checked.
        with httpx.Client(follow_redirects=False, timeout=timeout,
                          headers={"User-Agent": _UA}) as client:
            current = url
            r = None
            for _ in range(_MAX_REDIRECTS + 1):
                r = client.get(current)
                if r.status_code not in (301, 302, 303, 307, 308):
                    break
                nxt = r.headers.get("location")
                if not nxt:
                    break
                nxt = urllib.parse.urljoin(current, nxt)
                if not nxt.lower().startswith(("http://", "https://")):
                    return ""
                if not _url_host_is_public(nxt):
                    return ""
                current = nxt
            else:
                return ""  # too many hops
            if r is None:
                return ""
            ctype = r.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                return ""
            html = r.text[:600_000]  # cap before parsing
    except Exception:
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(list(_DROP_TAGS)):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text(separator=" ", strip=True)
    except Exception:
        return ""

    text = re.sub(r"\s+", " ", text).strip()
    return redact(text[:max_chars])


_EXPAND_SUFFIXES = ("", " latest", " 2026", " details explained",
                    " news update", " timeline history", " analysis facts")


def expand_queries(query: str, *, n: int = 6) -> list[str]:
    """Light deterministic fan-out for a single question — so `--thorough`
    gets breadth even when the caller passes only one query. The caller (a
    capable model) supplying its own angles is still better; this is the
    standalone fallback."""
    base = (query or "").strip()
    out: list[str] = []
    for suf in _EXPAND_SUFFIXES:
        q = (base + suf).strip()
        if q and q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out


def corroboration(hits: list["DeepHit"]) -> dict:
    """Cheap, agent-free verify signal: how many DISTINCT domains back the
    result set, and which hits are corroborated by >1 sub-query. Lets the
    single synthesis pass weight agreement and flag thin/uncorroborated claims
    — the role the old multi-agent verifier paid millions of tokens for."""
    from urllib.parse import urlparse
    domains: dict[str, int] = {}
    for h in hits:
        d = urlparse(h.url).netloc.replace("www.", "") if h.url else ""
        if d:
            domains[d] = domains.get(d, 0) + 1
    multi = sum(1 for h in hits if len(h.seen_in) > 1)
    return {"distinct_domains": len(domains), "domains": domains,
            "multi_query_corroborated": multi, "total": len(hits)}


def deep_web_search(queries: list[str], *, k_per_query: int = 6,
                    total_k: int = 8, fetch: bool = False, fetch_k: int = 4,
                    per_page_chars: int = 1500) -> list[DeepHit]:
    """Fan `queries` at the web provider, fuse with RRF + URL dedup, return the
    top `total_k` consolidated hits (best first). Pure retrieval — no LLM.

    If fetch=True, also download + extract the main text of the top `fetch_k`
    URLs (still pure retrieval) so the caller synthesizes over real page content,
    not just snippets — depth without spawning agents."""
    provider = WebProvider(name="web")
    merged: dict[str, DeepHit] = {}

    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        hits = provider.search(q, k=k_per_query)
        for rank, h in enumerate(hits):
            url = (h.provenance or {}).get("url", "")
            key = url or h.text[:80]
            contrib = 1.0 / (_RRF_K + rank + 1)
            dh = merged.get(key)
            if dh is None:
                title = (h.provenance or {}).get("title", "")
                snippet = h.text
                merged[key] = DeepHit(url=url, title=title, snippet=snippet,
                                      score=contrib, seen_in=[q])
            else:
                dh.score += contrib
                if q not in dh.seen_in:
                    dh.seen_in.append(q)

    ranked = sorted(merged.values(), key=lambda d: d.score, reverse=True)[:total_k]

    if fetch:
        for dh in ranked[:fetch_k]:
            if dh.url:
                dh.content = fetch_extract(dh.url, max_chars=per_page_chars)
    return ranked
