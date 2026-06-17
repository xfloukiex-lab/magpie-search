"""providers.youtube — live YouTube retrieval as a magpie federated source.

Given a query it returns trust-LEAD snippet Hits, two ways:
  - a YouTube URL / video id in the query  -> that video's transcript (captions),
    chunked into snippets with their start time;
  - otherwise                              -> a live YouTube search (titles +
    descriptions of the top matching videos).

Per the Provider contract (providers/base.py): it SEARCHES live at call time and
NEVER ingests/copies into magpie's own store — magpie stays a search tool, not a
video memory layer. Captions only: no audio download, no whisper, no frames, so a
call stays inside the federation timeout. Deep download + frame/vision
understanding is a separate AGENT capability, deliberately not magpie's job.

Optional dependency: `yt-dlp`. Missing dep or any error -> [] (fail-open), never
raises — exactly like the web provider with `ddgs`.
"""
from __future__ import annotations

import re
import urllib.request
from typing import Any
from urllib.parse import urlparse

from ..redactor import redact
from .base import Hit, Provider, TrustTier

# SSRF guard: caption URLs from yt_dlp's info dict are uploader-controlled. Only
# fetch them from YouTube/Google CDNs, never an arbitrary (possibly internal) host.
_CAPTION_HOST_SUFFIXES = (".youtube.com", ".googlevideo.com", ".google.com",
                          ".ytimg.com", ".gstatic.com")
_CAPTION_HOSTS_EXACT = ("youtube.com", "google.com")
_MAX_VTT_BYTES = 5 * 1024 * 1024


def _allowed_caption_host(url: str) -> bool:
    try:
        h = (urlparse(url).hostname or "").lower()
        return h.endswith(_CAPTION_HOST_SUFFIXES) or h in _CAPTION_HOSTS_EXACT
    except Exception:
        return False

# youtube.com/watch?v=ID, youtu.be/ID, /shorts/ID, /embed/ID, or a bare 11-char id.
_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|v/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
    r"|^([A-Za-z0-9_-]{11})$"
)
_VTT_TS = re.compile(r"^(\d\d):(\d\d):(\d\d)\.\d{3}\s*-->")


def _extract_video_id(text: str) -> str | None:
    for tok in (text or "").split():
        m = _ID_RE.search(tok.strip())
        if m:
            return m.group(1) or m.group(2)
    return None


def _parse_vtt(text: str) -> list[dict]:
    """Minimal WebVTT -> [{start, text}]; strips cue tags and dedups repeats
    (YouTube auto-captions roll the same line across cues)."""
    out: list[dict] = []
    start = None
    buf: list[str] = []
    last = ""

    def flush():
        nonlocal last
        if start is not None and buf:
            line = re.sub(r"<[^>]+>", "", " ".join(buf)).strip()
            if line and line != last:
                out.append({"start": start, "text": line})
                last = line

    for raw in (text or "").splitlines():
        ln = raw.strip()
        m = _VTT_TS.match(ln)
        if m:
            flush()
            buf = []
            start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        elif ln in ("", "WEBVTT") or ln.startswith(("Kind:", "Language:", "NOTE")):
            continue
        else:
            buf.append(ln)
    flush()
    return out


def _chunk(segments: list[dict], *, max_chars: int = 400) -> list[dict]:
    chunks: list[dict] = []
    cur: list[str] = []
    start = None
    n = 0
    for s in segments:
        if start is None:
            start = s.get("start", 0)
        cur.append(s.get("text", ""))
        n += len(s.get("text", ""))
        if n >= max_chars:
            chunks.append({"start": start, "text": " ".join(cur).strip()})
            cur, start, n = [], None, 0
    if cur:
        chunks.append({"start": start or 0, "text": " ".join(cur).strip()})
    return [c for c in chunks if c["text"]]


class YoutubeProvider(Provider):
    category = "youtube"
    default_trust = TrustTier.LEAD            # "what a video said" — a lead to verify

    # --- network seams (thin, monkeypatchable in tests) ---------------------
    def _info(self, target: str) -> dict | None:
        try:
            import yt_dlp
        except Exception:
            return None
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True,
                                   "no_warnings": True}) as y:
                return y.extract_info(target, download=False)
        except Exception:
            return None

    def _http_get(self, url: str, *, timeout: float = 4.0) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(_MAX_VTT_BYTES).decode("utf-8", "replace")  # size-capped

    def _caption_segments(self, video_id: str) -> dict:
        """{'title','channel','segments'} for a video; manual EN captions first,
        then auto-captions. Empty segments on any miss."""
        info = self._info(f"https://www.youtube.com/watch?v={video_id}")
        if not info:
            return {"title": "", "channel": "", "segments": []}
        title = info.get("title", "") or ""
        channel = info.get("uploader") or info.get("channel", "") or ""

        def pick(d):
            d = d or {}
            return d.get("en") or next((v for k, v in d.items() if k.startswith("en")), None)

        track = pick(info.get("subtitles")) or pick(info.get("automatic_captions"))
        if not track:
            return {"title": title, "channel": channel, "segments": []}
        vtt = next((t for t in track if t.get("ext") == "vtt"), track[0])
        cap_url = vtt.get("url", "")
        if not _allowed_caption_host(cap_url):     # SSRF: YouTube/Google CDN only
            return {"title": title, "channel": channel, "segments": []}
        try:
            segs = _parse_vtt(self._http_get(cap_url))
        except Exception:
            segs = []
        return {"title": title, "channel": channel, "segments": segs}

    def _search_entries(self, query: str, k: int) -> list[dict]:
        info = self._info(f"ytsearch{max(1, k)}:{query}")
        return list((info or {}).get("entries") or [])

    # --- provider contract --------------------------------------------------
    def search(self, query: str, *, budget_tokens: int | None = None,
               scope: Any = None, k: int = 10) -> list[Hit]:
        query = (query or "").strip()
        if not query:
            return []
        vid = _extract_video_id(query) or _extract_video_id(str(scope or ""))
        try:
            return (self._transcript_hits(vid, k) if vid
                    else self._search_hits(query, k))
        except Exception:
            return []  # fail-open per the Provider contract

    def _transcript_hits(self, video_id: str, k: int) -> list[Hit]:
        meta = self._caption_segments(video_id)
        chunks = _chunk(meta["segments"])[:max(1, k)]
        url = f"https://www.youtube.com/watch?v={video_id}"
        hits: list[Hit] = []
        for rank, c in enumerate(chunks):
            hits.append(Hit(
                text=redact(c["text"]),
                source=self.name,
                trust=self.trust,
                category=self.category,
                score=float(len(chunks) - rank),
                provenance={"url": url, "video_id": video_id, "title": meta["title"],
                            "channel": meta["channel"], "start": c["start"]},
            ))
        return hits

    def _search_hits(self, query: str, k: int) -> list[Hit]:
        entries = self._search_entries(query, k)
        hits: list[Hit] = []
        for rank, e in enumerate(entries):
            title = (e.get("title") or "").strip()
            desc = (e.get("description") or "").strip().replace("\n", " ")
            text = redact(f"{title} — {desc}".strip(" —"))
            if not text:
                continue
            vid = e.get("id", "")
            hits.append(Hit(
                text=text[:600],
                source=self.name,
                trust=self.trust,
                category=self.category,
                score=float(len(entries) - rank),
                provenance={"url": f"https://www.youtube.com/watch?v={vid}",
                            "video_id": vid, "title": title,
                            "channel": e.get("uploader") or e.get("channel", "")},
            ))
        return hits

    def health(self) -> dict[str, Any]:
        ok = True
        try:
            import yt_dlp  # noqa: F401
        except Exception:
            ok = False
        return {"name": self.name, "category": self.category, "ok": ok,
                "backend": "yt-dlp (captions, live)"}
