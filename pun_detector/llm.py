"""Optional LLM reasoning (Gemini).

This project can run without an LLM. If GOOGLE_API_KEY is set and
`google-generativeai` is installed, we ask Gemini to:
- label pun type (phonetic / semantic / non-joke)
- validate coherence
- explain humor briefly
- suggest age appropriateness

If unavailable, we return a deterministic rule-based fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional


def _safe_take_text(resp: Any) -> str:
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # Best-effort extraction for different response shapes
    try:
        cands = getattr(resp, "candidates", [])
        for c in cands:
            content = getattr(c, "content", None)
            parts = []
            if isinstance(content, dict):
                parts = content.get("parts", [])
            else:
                parts = getattr(content, "parts", []) if content is not None else []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    return t
    except Exception:
        pass
    return ""


def _wrap_by_words(text: str, max_words: int = 10) -> str:
    words = text.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _fallback_verdict(feats: Dict[str, Any]) -> Dict[str, Any]:
    # Simple rules: phonetic strong => phonetic pun; else semantic related => semantic; else non-joke
    phon = float(feats.get("phonetic_similarity", 0.0))
    sem = float(feats.get("semantic_similarity", 0.0))

    if phon >= 0.75:
        pun_type = "phonetic"
        valid = True
    elif sem >= 0.35:
        pun_type = "semantic"
        valid = True
    else:
        pun_type = "non-joke"
        valid = False

    return {
        "pun_type": pun_type,
        "valid_joke": bool(valid),
        "humor_reason": _wrap_by_words(
            "Rule-based fallback (LLM unavailable). "
            "Enable Gemini by setting GOOGLE_API_KEY." ,
            max_words=10,
        ),
        "age_appropriate": True,
    }


def gemini_reasoning(text: str, feats: Dict[str, Any], model_name: str = "gemini-1.5-pro", retries: int = 2) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _fallback_verdict(feats)

    try:
        import google.generativeai as genai
    except Exception:
        return _fallback_verdict(feats)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Analyze this text for humor and puns.

Text: "{text}"

Features:
- Candidate phonetic pair: {feats.get('candidate_phonetic_pairs')}
- Candidate semantic pair: {feats.get('candidate_semantic_pairs')}
- Phonetic similarity: {feats.get('phonetic_similarity')}
- Semantic similarity: {feats.get('semantic_similarity')}
- Age info: {feats.get('aoa_info')}

Tasks:
1. Classify the pun type: semantic, phonetic, or non-joke
2. Explain why it might be funny (short, clear)
3. If it doesn't form a coherent joke, set valid_joke=false
4. Determine whether it's age appropriate

Respond with ONLY a valid JSON object:
{{
  "pun_type": "semantic",
  "valid_joke": true,
  "humor_reason": "Brief explanation",
  "age_appropriate": true
}}
""".strip()

    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            r = model.generate_content(prompt)
            response_text = _safe_take_text(r).strip()

            # strip fences if any
            response_text = re.sub(r"^\s*```json\s*", "", response_text)
            response_text = re.sub(r"^\s*```\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text).strip()

            m = re.search(r"\{.*\}", response_text, re.DOTALL)
            if m:
                response_text = m.group(0)

            out = json.loads(response_text)
            # wrap humor reason for snipping
            if isinstance(out.get("humor_reason"), str):
                out["humor_reason"] = _wrap_by_words(out["humor_reason"], max_words=10)
            return out
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(code in msg for code in ("429", "503")) and i < retries:
                time.sleep(2 * (i + 1))
                continue
            break

    out = _fallback_verdict(feats)
    out["humor_reason"] = _wrap_by_words(f"LLM error: {str(last_err)[:160]}", max_words=10)
    out["valid_joke"] = False
    out["pun_type"] = "none"
    return out
