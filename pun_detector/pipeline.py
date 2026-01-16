"""End-to-end feature extraction + pun/joke analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .aoa import AoAIndex
from .config import DEFAULT_THRESHOLDS, Thresholds
from .llm import llm_reasoning
from .phonetic import implicit_partner, phonetic_similarity
from .semantics import semantic_similarity
from .text_utils import clean_and_tag, lemma_token


def extract_features(
    text: str,
    age: float,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    aoa: Optional[AoAIndex] = None,
    penalty_weight: float = 0.0,
    beta: float = 2.0,
) -> Dict[str, Any]:
    """Extract candidate phonetic/semantic pairs and similarity scores."""

    tagged = clean_and_tag(text)

    # Focus on nouns/verbs (common carriers of ambiguity)
    tagged = [(w, p) for (w, p) in tagged if p.startswith(("N", "V"))]

    seen_lemmas = set()
    unique_words = []
    lemma_map = {}
    for w, p in tagged:
        lem = lemma_token(w, p)
        if lem not in seen_lemmas:
            seen_lemmas.add(lem)
            unique_words.append(w)
            lemma_map[w] = lem

    # ----- explicit phonetic pairs within the sentence -----
    best_pair_phon: Tuple[str, str] = ("", "")
    best_phon = 0.0
    for i, w1 in enumerate(unique_words):
        for w2 in unique_words[i + 1 :]:
            if lemma_map.get(w1, w1) == lemma_map.get(w2, w2):
                continue
            ps = phonetic_similarity(w1, w2, penalty_weight=penalty_weight, beta=beta)
            if ps > best_phon:
                best_pair_phon, best_phon = (w1, w2), ps

    # ----- implicit phonetic partner (not in sentence) -----
    implicit_used = False
    if best_phon < thresholds.phonetic_strong and unique_words:
        best_i_pair: Tuple[str, str] = ("", "")
        best_i = 0.0
        for w in unique_words:
            iw, isim = implicit_partner(w, penalty_weight=penalty_weight, beta=beta, avoid_same_lemma=True)
            if iw and isim > best_i:
                best_i_pair, best_i = (w, iw), isim
        if best_i > best_phon:
            best_pair_phon, best_phon = best_i_pair, best_i
            implicit_used = True

    # ----- semantic similarity within sentence -----
    best_pair_sem: Tuple[str, str] = ("", "")
    best_sem = -1.0
    for i, w1 in enumerate(unique_words):
        for w2 in unique_words[i + 1 :]:
            if lemma_map.get(w1, w1) == lemma_map.get(w2, w2):
                continue
            ss = semantic_similarity(text, w1, w2)
            if ss > best_sem:
                best_pair_sem, best_sem = (w1, w2), ss

    # ----- AoA -----
    aoa_score = None
    aoa_ok = None
    head_word = best_pair_phon[0] if best_pair_phon[0] else (unique_words[0] if unique_words else "")
    if aoa is not None and head_word:
        aoa_score = aoa.get(head_word)
        aoa_ok = aoa.is_age_appropriate(head_word, age)

    # human-friendly labels
    phon_label = "none"
    if best_pair_phon[0]:
        phon_label = f"({best_pair_phon[0]}, {best_pair_phon[1]})"
        if implicit_used:
            phon_label += " [implicit]"

    sem_label = "none"
    if best_pair_sem[0]:
        sem_label = f"({best_pair_sem[0]}, {best_pair_sem[1]})"

    return {
        "candidate_phonetic_pair": phon_label,
        "candidate_semantic_pair": sem_label,
        "phonetic_similarity": round(float(best_phon), 2),
        "semantic_similarity": round(float(best_sem), 2),
        "aoa_value": None if aoa_score is None else float(aoa_score),
        "aoa_age_ok": aoa_ok,
    }


def analyze_text(
    text: str,
    age: float,
    *,
    use_llm: bool = True,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    aoa: Optional[AoAIndex] = None,
) -> Dict[str, Any]:
    """Run feature extraction + (optional) LLM reasoning."""

    feats = extract_features(text, age, thresholds=thresholds, aoa=aoa)
    verdict = llm_reasoning(text, feats, thresholds=thresholds) if use_llm else {}
    return {**feats, **verdict}
