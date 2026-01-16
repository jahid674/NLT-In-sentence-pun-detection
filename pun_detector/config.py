"""Project configuration.

IMPORTANT:
- Do NOT hardcode API keys in code.
- If you want to use Gemini-based reasoning, set GOOGLE_API_KEY as an env var.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # Phonetic similarity threshold to confidently claim a sound-based link
    phonetic_strong: float = 0.75

    # Semantic similarity in [-1, 1] using cosine similarity of contextual embeddings
    semantic_related: float = 0.35


@dataclass(frozen=True)
class ModelConfig:
    # Transformer used for contextual embeddings.
    # Keep it simple + widely available.
    hf_model_name: str = "bert-base-uncased"


DEFAULT_THRESHOLDS = Thresholds()
DEFAULT_MODEL_CONFIG = ModelConfig()
