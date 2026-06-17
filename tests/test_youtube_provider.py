"""Tests for the live YouTube federated provider: URL->transcript,
query->search, fail-open, redaction, trust tier, and the VTT/id helpers.
Network seams (_caption_segments / _search_entries) are monkeypatched — offline."""
from __future__ import annotations

from magpie_search.providers import make_provider, provider_class
from magpie_search.providers.base import TrustTier
from magpie_search.providers.youtube import (
    YoutubeProvider, _chunk, _extract_video_id, _parse_vtt,
)

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VID = "dQw4w9WgXcQ"


def _provider(**seams):
    p = YoutubeProvider("youtube")
    for name, fn in seams.items():
        setattr(p, name, fn)
    return p


def test_registered_as_builtin():
    assert provider_class("youtube") is YoutubeProvider
    assert isinstance(make_provider("youtube"), YoutubeProvider)
    assert make_provider("youtube").trust is TrustTier.LEAD


def test_extract_video_id_variants():
    assert _extract_video_id(URL) == VID
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == VID
    assert _extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == VID
    assert _extract_video_id("watch this dQw4w9WgXcQ") == VID
    assert _extract_video_id("no id here") is None


def test_parse_vtt_dedups_rolling_autocaptions():
    vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\nhello world\n\n"
        "00:00:02.000 --> 00:00:04.000\nhello world\n\n"      # rolled repeat -> dropped
        "00:00:04.000 --> 00:00:06.000\n<c>second</c> line\n"
    )
    segs = _parse_vtt(vtt)
    assert [s["text"] for s in segs] == ["hello world", "second line"]
    assert segs[1]["start"] == 4


def test_chunk_groups_to_max_chars():
    segs = [{"start": i, "text": "x" * 200} for i in range(4)]
    chunks = _chunk(segs, max_chars=400)
    assert len(chunks) == 2 and chunks[0]["start"] == 0


def test_url_query_returns_transcript_hits():
    p = _provider(_caption_segments=lambda vid: {
        "title": "Never Gonna", "channel": "Rick",
        "segments": [{"start": 0, "text": "we are no strangers to love " * 10},
                     {"start": 30, "text": "you know the rules " * 10}]})
    hits = p.search(URL, k=10)
    assert hits and all(h.category == "youtube" for h in hits)
    assert all(h.trust is TrustTier.LEAD for h in hits)
    assert hits[0].provenance["video_id"] == VID
    assert hits[0].provenance["url"] == URL
    assert "start" in hits[0].provenance


def test_plain_query_returns_search_hits():
    p = _provider(_search_entries=lambda q, k: [
        {"id": "aaaaaaaaaaa", "title": "Proof of Work explained",
         "description": "how PoW secures bitcoin", "uploader": "Crypto101"}])
    hits = p.search("proof of work", k=5)
    assert len(hits) == 1
    assert hits[0].provenance["url"].endswith("aaaaaaaaaaa")
    assert "Proof of Work" in hits[0].text


def test_redaction_applied_to_transcript():
    p = _provider(_caption_segments=lambda vid: {
        "title": "t", "channel": "c",
        "segments": [{"start": 0, "text": "my aws key is AKIAIOSFODNN7EXAMPLE here " * 5}]})
    hits = p.search(URL, k=3)
    assert hits and "AKIAIOSFODNN7EXAMPLE" not in hits[0].text and "REDACTED" in hits[0].text


def test_fail_open_on_seam_error():
    def boom(*a, **k):
        raise RuntimeError("network down")
    assert _provider(_caption_segments=boom).search(URL, k=3) == []
    assert _provider(_search_entries=boom).search("anything", k=3) == []


def test_empty_query_returns_empty():
    assert YoutubeProvider("youtube").search("   ", k=3) == []


def test_caption_ssrf_blocks_non_youtube_host():
    """An uploader-controlled caption URL on an internal host is never fetched."""
    p = YoutubeProvider("youtube")
    p._info = lambda target: {
        "title": "t", "uploader": "u",
        "subtitles": {"en": [{"ext": "vtt", "url": "http://169.254.169.254/x"}]}}
    called = {"v": False}

    def httpget(*a, **k):
        called["v"] = True
        return ""
    p._http_get = httpget
    hits = p.search(URL, k=3)
    assert hits == [] and called["v"] is False        # guarded BEFORE any fetch
