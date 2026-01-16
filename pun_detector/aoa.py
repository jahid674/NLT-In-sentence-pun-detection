"""Age-of-Acquisition (AoA) utilities.

AoA data is optional. If you provide an AoA file (e.g., Kuperman et al.),
this module can load it and score whether a word is suitable for a target age.

For GitHub:
- Do NOT commit large proprietary datasets.
- Instead, add instructions in README and keep the loader optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class AoALookup:
    table: Dict[str, float]

    def get(self, word: str) -> Optional[float]:
        return self.table.get((word or "").lower().strip())


def load_aoa_xlsx(path: str, word_col_hint: str = "word", rating_col_hint: str = "aoa") -> AoALookup:
    """Load AoA ratings from an Excel file.

    The loader tries to infer the relevant columns.
    """
    df = pd.read_excel(path)
    w = next((c for c in df.columns if word_col_hint in c.lower()), None)
    r = next((c for c in df.columns if rating_col_hint in c.lower() or "mean" in c.lower()), None)
    if w is None or r is None:
        raise ValueError(f"Could not find AoA columns in: {list(df.columns)}")

    df = df[[w, r]].dropna()
    df[w] = df[w].astype(str).str.lower().str.strip()
    df[r] = pd.to_numeric(df[r], errors="coerce")
    df = df.dropna()

    return AoALookup(dict(zip(df[w], df[r])))


def aoa_suitable(word: str, age: float, aoa: Optional[AoALookup] = None) -> Optional[bool]:
    """Return True/False if AoA rating is known; otherwise None."""
    if aoa is None:
        return None
    val = aoa.get(word)
    if val is None:
        return None
    try:
        return float(age) >= float(val)
    except Exception:
        return None
