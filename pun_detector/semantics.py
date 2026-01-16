"""Semantic similarity using contextual BERT embeddings.

This is intentionally simple (and reproducible):
- Uses a HuggingFace transformer (default: bert-base-uncased)
- Extracts contextual embeddings for a target word inside a sentence by
  finding its token span and mean-pooling hidden states over that span.

If the target span is not found, falls back to sentence embedding of the word.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer

from .config import DEFAULT_MODEL_CONFIG


def _find_subseq(seq: List[int], sub: List[int]) -> List[Tuple[int, int]]:
    if not sub or not seq or len(sub) > len(seq):
        return []
    spans = []
    L = len(sub)
    for i in range(0, len(seq) - L + 1):
        if seq[i : i + L] == sub:
            spans.append((i, i + L))
    return spans


def _mean_pool(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return t.mean(dim=dim)


@lru_cache(maxsize=1)
def _hf():
    """Lazy-load HF tokenizer/model once."""
    name = DEFAULT_MODEL_CONFIG.hf_model_name
    tok = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    return tok, model, device


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def get_word_embedding(text: str) -> np.ndarray:
    tok, model, device = _hf()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True).to(device)
        out = model(**enc)
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return _normalize(emb)


def get_contextual_embedding(text: str, target: str) -> np.ndarray:
    tok, model, device = _hf()

    with torch.no_grad():
        enc_text = tok(text, return_tensors="pt", truncation=True, padding=True).to(device)
        out_text = model(**enc_text)

        H = out_text.last_hidden_state.squeeze(0)  # [T, D]
        ids_text = enc_text["input_ids"].squeeze(0).tolist()

        tgt_tokens = tok.tokenize(target)
        if not tgt_tokens:
            return get_word_embedding(target)
        tgt_ids = tok.convert_tokens_to_ids(tgt_tokens)

        spans = _find_subseq(ids_text, tgt_ids)
        if spans:
            reps = [_mean_pool(H[s:e, :]) for (s, e) in spans]
            rep = torch.stack(reps, dim=0).mean(dim=0).cpu().numpy()
            return _normalize(rep)

    return get_word_embedding(target)


def semantic_similarity(text: str, w1: str, w2: str) -> float:
    e1 = get_contextual_embedding(text, w1)
    e2 = get_contextual_embedding(text, w2)

    if not np.any(e1) or not np.any(e2):
        return 0.0

    c = 1.0 - cosine(e1, e2)
    if np.isnan(c):
        return 0.0

    return float(np.clip(c, -1.0, 1.0))
