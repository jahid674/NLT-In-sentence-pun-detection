"""A lightweight conversational wrapper with in-memory state.

- Keeps track of last analysis
- Hides similarity scores unless user asks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .pipeline import analyze_text


def wrap_by_words(text: str, max_words: int = 10) -> str:
    words = text.split()
    if not words:
        return ""
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


@dataclass
class PunAgent:
    aoa: Optional[Any] = None
    last_result: Optional[Dict[str, Any]] = None
    history: list = field(default_factory=list)

    def analyze(self, text: str, age: float, reveal_scores: bool = False) -> str:
        res = analyze_text(text=text, age=age, aoa=self.aoa)
        self.last_result = res
        self.history.append({"text": text, "age": age, "result": res})

        # Build a human-friendly response
        lines = []
        lines.append(f"Pun type: {res.get('pun_type', 'unknown')}")
        lines.append(f"Valid joke: {res.get('valid_joke', False)}")
        lines.append(f"Age appropriate: {res.get('age_appropriate', False)}")

        reason = res.get("humor_reason", "")
        if reason:
            lines.append("Explanation:")
            lines.append(wrap_by_words(reason, 10))

        lines.append("\nCandidate pairs:")
        lines.append(f"- Phonetic: {res.get('candidate_phonetic_pair', 'none')}")
        lines.append(f"- Semantic: {res.get('candidate_semantic_pair', 'none')}")

        if reveal_scores:
            lines.append("\nScores:")
            lines.append(f"- phonetic_similarity: {res.get('phonetic_similarity', 0.0)}")
            lines.append(f"- semantic_similarity: {res.get('semantic_similarity', 0.0)}")

        return "\n".join(lines)
