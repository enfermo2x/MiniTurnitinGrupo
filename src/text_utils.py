"""Utilities for robust sentence splitting/tokenization.

Provides `safe_sent_tokenize` which tries NLTK and falls back to heuristics
when the tokenizer is not available or returns too few sentences.
"""
import re
from typing import List

try:
    import nltk
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    nltk.download("punkt", quiet=True)
except Exception:
    _nltk_sent_tokenize = None


_SENT_SPLIT_RE = re.compile(r'[.!?\n]+')


def safe_sent_tokenize(text: str, language: str = "spanish") -> List[str]:
    """Return a list of sentence-like segments for `text`.

    Attempts NLTK first; if it fails or returns fewer than 2 segments,
    falls back to splitting on punctuation/newlines and then to comma-based
    clause splitting. Ensures at least one returned element (empty text -> []).
    """
    if not text or not isinstance(text, str):
        return []

    # Try NLTK if available
    if _nltk_sent_tokenize is not None:
        try:
            sents = _nltk_sent_tokenize(text, language=language)
            # Filter out very short tokens
            sents = [s.strip() for s in sents if s.strip()]
            if len(sents) >= 2:
                return sents
        except Exception:
            pass

    # Fallback splitting: punctuation and newlines
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sents) >= 2:
        return sents

    # If still only one or zero segments, try splitting on commas/semicolons
    parts = [p.strip() for p in re.split(r'[;,:\u2014\-]\s*', text) if p.strip()]
    if len(parts) >= 2:
        return parts

    # As a last resort, return the full text as one segment
    return [text.strip()]
