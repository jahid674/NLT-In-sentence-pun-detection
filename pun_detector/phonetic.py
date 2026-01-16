"""Phonetic similarity + implicit partner search.

- Primary: CMU pronunciation edit distance (stress-stripped)
- Fallback: metaphone / soundex (via `phonetics` package)

The implicit partner search tries to find a word in the CMU dict that
"sounds like" the target, even if it is not present in the input text.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

from nltk.corpus import cmudict, wordnet
import phonetics

from .text_utils import lemma_set


@lru_cache(maxsize=1)
def _cmu_dict():
    return cmudict.dict()


def _strip_stress(pron: List[str]) -> List[str]:
    return [p[:-1] if p and p[-1].isdigit() else p for p in pron]


def _cmu_prons(word: str) -> List[List[str]]:
    w = word.lower()
    cmu = _cmu_dict()
    if w in cmu:
        return [_strip_stress(p) for p in cmu[w]]
    return []


def levenshtein(a, b) -> int:
    if isinstance(a, str):
        a = list(a)
    if isinstance(b, str):
        b = list(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def _edge_match_bonus(seq1, seq2, first_bonus=0.05, last_bonus=0.07) -> float:
    if not seq1 or not seq2:
        return 0.0
    bonus = 0.0
    if seq1[0] == seq2[0]:
        bonus += first_bonus
    if seq1[-1] == seq2[-1]:
        bonus += last_bonus
    return bonus


def _weighted_sim_from_distance(dist: float, len_a: int, len_b: int, beta: float = 2.0) -> float:
    denom = max(len_a, len_b, 1)
    d = dist / denom
    s = 1.0 - (d ** beta)
    return float(max(0.0, min(1.0, s)))


def _letter_penalty(w1: str, w2: str) -> float:
    d = levenshtein(w1.lower(), w2.lower())
    return d / max(len(w1), len(w2), 1)


def _phonetic_base_sim(w1: str, w2: str, beta: float = 2.0) -> float:
    pr1 = _cmu_prons(w1)
    pr2 = _cmu_prons(w2)

    best = 0.0
    if pr1 and pr2:
        for p1 in pr1:
            for p2 in pr2:
                d = levenshtein(p1, p2)
                s = _weighted_sim_from_distance(d, len(p1), len(p2), beta=beta)
                s += _edge_match_bonus(p1, p2)
                best = max(best, min(1.0, s))
        return best

    # Fallback: metaphone / soundex
    try:
        m1 = phonetics.metaphone(w1)
        m2 = phonetics.metaphone(w2)
        if m1 and m2:
            d = levenshtein(m1, m2)
            s = _weighted_sim_from_distance(d, len(m1), len(m2), beta=beta)
            s += _edge_match_bonus(list(m1), list(m2))
            return float(min(1.0, s))
    except Exception:
        pass

    s1, s2 = phonetics.soundex(w1), phonetics.soundex(w2)
    d = levenshtein(s1, s2)
    s = _weighted_sim_from_distance(d, len(s1), len(s2), beta=beta)
    s += _edge_match_bonus(list(s1), list(s2))
    return float(min(1.0, s))


def phonetic_similarity(word1: str, word2: str, *, penalty_weight: float = 0.0, beta: float = 2.0) -> float:
    """Phonetic similarity in [0, 1]."""
    if not word1 or not word2:
        return 0.0
    base = _phonetic_base_sim(word1, word2, beta=beta)
    if penalty_weight > 0:
        base = base - penalty_weight * _letter_penalty(word1, word2)
    return float(round(max(0.0, min(1.0, base)), 4))


@lru_cache(maxsize=1)
def _build_phonetic_indices():
    """Pre-index CMU vocabulary by metaphone/soundex for fast implicit matching."""
    cmu = _cmu_dict()
    metaphone_map: Dict[str, Set[str]] = {}
    soundex_map: Dict[str, Set[str]] = {}

    for w in cmu.keys():
        w_l = w.lower()
        try:
            m = phonetics.metaphone(w_l)
        except Exception:
            m = ""
        s = phonetics.soundex(w_l)

        if m:
            metaphone_map.setdefault(m, set()).add(w_l)
        if s:
            soundex_map.setdefault(s, set()).add(w_l)

    return metaphone_map, soundex_map


def best_implicit_phonetic_partner(
    word: str,
    *,
    max_candidates: int = 4000,
    penalty_weight: float = 0.0,
    beta: float = 2.0,
    avoid_same_lemma: bool = True,
) -> Tuple[Optional[str], float]:
    """Find a phonetic partner in CMU vocabulary for `word`.

    Returns (best_word, similarity). If no candidate, returns (None, 0.0).
    """
    if not word:
        return None, 0.0

    w = word.lower()
    meta_map, snd_map = _build_phonetic_indices()

    cands: Set[str] = set()
    try:
        m = phonetics.metaphone(w)
        if m and m in meta_map:
            cands |= meta_map[m]
    except Exception:
        pass

    try:
        s = phonetics.soundex(w)
        if s and s in snd_map:
            cands |= snd_map[s]
    except Exception:
        pass

    cands.discard(w)

    # Filter out lemma-equivalent variants (water/watery, run/running, ...)
    if avoid_same_lemma:
        src = lemma_set(w)
        cands = {c for c in cands if not (lemma_set(c) & src)}

    # Limit candidate set for speed
    if len(cands) > max_candidates:
        # deterministic-ish subset by sorting
        cands = set(sorted(cands)[:max_candidates])

    best_w, best_sim = None, 0.0
    for cand in cands:
        base = _phonetic_base_sim(w, cand, beta=beta)
        score = base - (penalty_weight * _letter_penalty(w, cand) if penalty_weight > 0 else 0.0)
        if score > best_sim:
            best_sim, best_w = score, cand

    if best_w is None:
        return None, 0.0
    return best_w, float(round(max(0.0, min(1.0, best_sim)), 4))
