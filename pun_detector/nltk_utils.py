"""NLTK setup helpers.

We keep NLTK downloads in one place so importing modules doesn't
unexpectedly download resources.

Call `ensure_nltk()` once at the top of your notebook/script.
"""

from __future__ import annotations

import nltk


_NLTK_PACKAGES = [
    "punkt",
    "cmudict",
    "averaged_perceptron_tagger",
    "stopwords",
    "wordnet",
    "omw-1.4",
]


def ensure_nltk(quiet: bool = True) -> None:
    """Download required NLTK resources if missing."""
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=quiet)
