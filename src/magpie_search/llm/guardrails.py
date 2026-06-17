"""guardrails — schema validators + worker-specific hallucination probes.

Each probe returns a tuple (passed: bool, reason: str | None). Probes are
DEFENSE IN DEPTH: schema-clean output can still fail a probe and be marked
'degraded'. Workers decide whether to use degraded output or fall back.

Design principle: degrade gracefully, never hard-error on imperfect LLM output.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import threading as _threading

# Lazy embedder for semantic_grounding — reuses the model the transcript
# indexer downloaded (sentence-transformers/all-MiniLM-L6-v2, 80 MB).
_EMBEDDER = None
_EMBEDDER_LOCK = _threading.Lock()


def _get_embedder():
    """Lazy-init embedding model.

    Critical: uses the same cache dir as the transcript indexer (~/.magpie-search/models
    by default, env-override MAGPIE_SEARCH_MODELS_DIR; legacy AVIARY_MODELS_DIR honored).
    The fastembed default temp cache sometimes ends up with partial snapshots
    ("Could not find config.json") — reusing the indexer's cache means we hit
    a model that's already verified end-to-end via the corpus embed.

    Double-checked locking: cheap fast path when the model is already loaded,
    full lock acquisition only on the cold-start race window. Without the
    lock, two probes racing the first call could both construct a fresh
    `TextEmbedding` (~80 MB each) before either set the global."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        import os
        from pathlib import Path
        from fastembed import TextEmbedding
        cache_dir = (
            os.environ.get("MAGPIE_SEARCH_MODELS_DIR")
            or os.environ.get("AVIARY_MODELS_DIR")
            or str(Path.home() / ".magpie-search" / "models")
        )
        _EMBEDDER = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir,
        )
    return _EMBEDDER


def _cosine(a, b) -> float:
    """fastembed returns L2-normalized vectors, so cosine = dot product."""
    import numpy as np
    return float(np.dot(a, b))


def _chunk(text: str, *, size: int = 500, overlap: int = 100) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    step = max(1, size - overlap)
    i = 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        i += step
    return out or [text[:size]]

# ---------------------------------------------------------------------------
# Reranker probe — output must be a permutation/subset of input.
# Cross-encoder doesn't hallucinate per se (it outputs scores), but defensive.
# ---------------------------------------------------------------------------

def reranker_output_is_subset(
    output_rowids: Iterable[int],
    input_rowids: Iterable[int],
) -> tuple[bool, str | None]:
    in_set = set(input_rowids)
    out_list = list(output_rowids)
    if not out_list:
        return False, "empty output"
    fabricated = [r for r in out_list if r not in in_set]
    if fabricated:
        return False, f"fabricated rowids: {fabricated[:5]}"
    return True, None


# ---------------------------------------------------------------------------
# Summarizer probes — content must be grounded in source.
# ---------------------------------------------------------------------------

_LEN_MIN = 20
_LEN_MAX = 900  # raised after bakeoff — verbose dev sessions often run 700-850
# Foundational technical proper nouns. So pervasive in dev context that
# their presence in a summary without explicit source mention is not a
# fabrication signal — they're the substrate, not the content.
_TECH_PN_ALLOWLIST = frozenset({
    "python", "bash", "linux", "windows", "macos", "github", "git",
    "claude", "ai", "llm", "json", "yaml", "html", "css", "javascript",
    "docker", "node", "npm", "pip", "sql", "sqlite", "vscode",
    "ollama", "ssh", "http", "https", "url", "api", "cli", "gpu", "cpu",
    "ram", "vram", "powershell", "cmd",
})
# Looks like a code identifier / hash / id — letters + digits, >= 5 chars,
# and not pure-word. Catches the worst hallucination class for technical
# summaries (invented task IDs, commit hashes, file UUIDs).
_IDENTIFIER_RE = re.compile(r"\b(?=[a-zA-Z0-9_-]*[0-9])(?=[a-zA-Z0-9_-]*[a-zA-Z])[a-zA-Z0-9_-]{5,}\b")
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_\-']{2,}\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Function words / short connectors — uninformative for overlap and noisy for
# sentence-starter detection.
_STOPWORDS = frozenset({
    "the", "and", "but", "for", "with", "from", "into", "onto", "upon",
    "this", "that", "these", "those", "their", "there", "then", "than",
    "have", "has", "had", "was", "were", "are", "been", "being",
    "is", "it", "its", "of", "to", "in", "on", "at", "by", "as", "an", "a",
    "i", "you", "he", "she", "we", "they", "them", "us", "him", "her",
    "what", "when", "where", "who", "why", "how", "which", "while",
    "also", "still", "just", "only", "any", "all", "no", "not",
    "will", "would", "should", "could", "can", "may", "might",
    "session", "focused", "specifically", "used", "via", "such",
    "based", "using", "after", "before", "between", "within",
})


def summarizer_length_ok(text: str) -> tuple[bool, str | None]:
    n = len(text.strip())
    if n < _LEN_MIN:
        return False, f"too short ({n} chars)"
    if n > _LEN_MAX:
        return False, f"too long ({n} chars)"
    return True, None


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(s)}


_SUFFIXES = ("ies", "es", "ing", "ed", "ly", "er", "est", "s")


def _stem(w: str) -> str:
    """Crude suffix-stripping — catches plural/tense without nltk."""
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content_tokens(s: str) -> set[str]:
    """Lowercase, stopword-removed, stem-folded tokens — what carries
    meaning. "outputs" and "output" collapse to the same stem so the
    overlap check doesn't false-positive on inflection."""
    return {_stem(w) for w in _tokens(s) if w not in _STOPWORDS}


def summarizer_semantic_grounding(
    summary: str, source: str, *, threshold: float = 0.5,
) -> tuple[bool, str | None]:
    """Embedding-cosine grounding — primary defense against off-topic hallucination.

    Embeds the summary once; chunks the source into ~500-char pieces with
    100-char overlap, embeds each. Returns max cosine similarity over chunks.

    Why max not mean: summaries focus on key events from a session, not
    the average of all messages. A summary about 'fixing cuckoo restart'
    against a 458-msg session that ALSO covers diagrams + reranker should
    pass because it strongly matches the cuckoo chunk, even if it doesn't
    match the diagram chunks. Averaging would penalize correct focus.

    Threshold 0.5: well above noise (random pairs hover at 0.1-0.3), well
    below "exact match" (which would be 0.85+). Tunable via env.
    """
    if not summary.strip() or not source.strip():
        return False, "empty summary or source"
    try:
        embedder = _get_embedder()
        summary_vec = next(iter(embedder.embed([summary])))
        chunks = _chunk(source, size=500, overlap=100)
        if not chunks:
            return False, "no source chunks"
        chunk_vecs = list(embedder.embed(chunks))
        sims = [_cosine(summary_vec, cv) for cv in chunk_vecs]
        max_sim = max(sims)
    except Exception as e:
        # If embedding fails for any reason, fail closed — degraded.
        return False, f"semantic check failed: {type(e).__name__}: {e}"
    if max_sim < threshold:
        return False, (
            f"semantic max-chunk similarity {max_sim:.3f} < {threshold} "
            f"(over {len(sims)} chunks; mean={sum(sims)/len(sims):.3f})"
        )
    return True, None


def summarizer_identifier_safety(
    summary: str, source: str,
) -> tuple[bool, str | None]:
    """Code-shaped identifiers in summary must appear in source.

    Catches invented task IDs / hashes / UUIDs — phi3.5's most common
    technical-summary hallucination. Example: model writes 'task b9drq19jn'
    when the source has 'task b3d5bvfs7'."""
    # Token-exact: compare against identifiers actually present in the source,
    # not a substring of it — else a fabricated id `b9drq` would pass merely
    # because `xb9drqy` appears somewhere in the source.
    src_ids = {m.lower() for m in _IDENTIFIER_RE.findall(source)}
    summary_ids = _IDENTIFIER_RE.findall(summary)
    fabricated = [i for i in summary_ids if i.lower() not in src_ids]
    if fabricated:
        return False, f"fabricated identifiers: {fabricated[:5]}"
    return True, None


def summarizer_noun_overlap(
    summary: str, source: str, *, threshold: float = 0.55,
) -> tuple[bool, str | None]:
    """Content words in the summary should appear in the source text.

    Stopwords removed from both sides — uninformative for grounding.
    Default threshold 0.55 = 55% of summary's content words must appear
    in source. Loose enough to allow paraphrase ("modify" → "change"),
    tight enough that "Bitcoin mining and cake recipes" gets flagged
    when neither word is in the source."""
    s_tokens = _content_tokens(summary)
    src_tokens = _content_tokens(source)
    if not s_tokens:
        return False, "no content tokens in summary"
    overlap = s_tokens & src_tokens
    ratio = len(overlap) / len(s_tokens)
    if ratio < threshold:
        missing = list(s_tokens - src_tokens)[:8]
        return False, f"noun overlap {ratio:.2f} < {threshold}; missing: {missing}"
    return True, None


def summarizer_proper_noun_safety(
    summary: str, source: str,
) -> tuple[bool, str | None]:
    """No proper noun in the summary should be absent from the source.

    Catches the worst hallucination class — inventing names of people,
    products, libraries. A summary that says 'Alice picked Redis as the
    storage layer' when the source only mentions SQLite is the signal.

    Heuristic for "is this a real proper noun":
      - Capitalized word with 3+ letters
      - NOT the first word of a sentence (sentence-initial capitalization
        catches articles like "The", "And", "But" that aren't proper nouns)
      - NOT in the meta-word allowlist (days, months, common pronouns)
    """
    src_lower = source.lower()

    # Identify sentence-initial words to exclude from PN consideration.
    sentence_starters: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(summary):
        sentence = sentence.lstrip()
        if not sentence:
            continue
        m = _WORD_RE.match(sentence)
        if m:
            sentence_starters.add(m.group(0))

    summary_pns = _PROPER_NOUN_RE.findall(summary)
    META_OK = {"i", "ai", "monday", "tuesday", "wednesday", "thursday",
               "friday", "saturday", "sunday",
               "january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november", "december"}
    # Filter: real PNs are not sentence-starters of common english words,
    # not meta-words, and not present in the source (case-insensitive).
    #
    # Sentence-starter exemption is limited to words whose lowercase form
    # is a stopword/article ("The", "And", "But" at sentence start). Capi-
    # talized words at sentence start that ARE proper-nouny ("Redis was
    # used") still get checked against the source — otherwise the worst
    # hallucination class (inventing a PN as the first word of a sentence)
    # silently slips through.
    fabricated = []
    for pn in summary_pns:
        pn_lower = pn.lower()
        if pn in sentence_starters and pn_lower in _STOPWORDS:
            continue
        if pn_lower in META_OK:
            continue
        if pn_lower in _STOPWORDS:
            continue
        if pn_lower in _TECH_PN_ALLOWLIST:
            continue  # Python/Bash/etc — substrate vocab, not entity claims
        # Word-boundary match, not substring: a fabricated PN "Redis" must
        # NOT be excused because the source contains "redistribute".
        if re.search(r"\b" + re.escape(pn_lower) + r"\b", src_lower):
            continue
        fabricated.append(pn)
    if fabricated:
        return False, f"fabricated proper nouns: {fabricated[:5]}"
    return True, None


# ---------------------------------------------------------------------------
# Classifier probe — strict label set.
# ---------------------------------------------------------------------------

def classifier_strict_label(
    text: str, allowed: set[str],
) -> tuple[bool, str | None]:
    """Output must be EXACTLY one label, no commentary."""
    t = text.strip().strip(".").strip('"').strip("'").lower()
    if t in {l.lower() for l in allowed}:
        return True, None
    # If the model added context like "urgent — this is from <name>", strip it.
    first_token = re.split(r"[\s\-,:]", t)[0]
    if first_token in {l.lower() for l in allowed}:
        return True, None
    return False, f"output {text[:50]!r} not in {sorted(allowed)}"


# ---------------------------------------------------------------------------
# Generic — find any "I'm an AI..." style refusal that got past the client.
# ---------------------------------------------------------------------------

_NEW_REFUSAL_HINTS = (
    "language model", "as a model", "i don't have",
    "happy to help", "i'm just an", "my knowledge",
    "i was trained", "training data", "as of my last",
)


def detect_refusal_drift(text: str) -> tuple[bool, str | None]:
    """Looks for emergent refusal/meta-text the client's pattern list missed.

    Returns (True, None) if clean; (False, hint) if a new pattern surfaced.
    The trust monitor uses this to suggest additions to client's refusal list.
    """
    low = text.lower()
    hits = [m for m in _NEW_REFUSAL_HINTS if m in low]
    if hits:
        return False, f"possible new refusal pattern: {hits[0]}"
    return True, None
