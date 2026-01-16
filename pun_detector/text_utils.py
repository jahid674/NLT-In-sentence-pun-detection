"""Text utilities: contractions, tokenization, POS mapping, lemmatization."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


WNL = WordNetLemmatizer()

AUXILIARIES = {
    "am","is","are","was","were","be","been","being",
    "have","has","had","having",
    "do","does","did","doing",
    "can","could","may","might","must","shall","should","will","would",
    "ain","ain't","’m","'m","’re","'re","’ve","'ve","’ll","'ll","’d","'d",
}

STOP = set(stopwords.words("english"))
STOP.update({"n't", "not"})

CONTRACTION_PATTERNS = [
    (re.compile(r"\b(\w+)[’']ll\b", flags=re.IGNORECASE), r"\1 will"),
    (re.compile(r"\b(\w+)[’']re\b", flags=re.IGNORECASE), r"\1 are"),
    (re.compile(r"\b(\w+)[’']ve\b", flags=re.IGNORECASE), r"\1 have"),
    (re.compile(r"\b(\w+)[’']d\b", flags=re.IGNORECASE), r"\1 would"),
    (re.compile(r"\b(\w+)[’']m\b", flags=re.IGNORECASE), r"\1 am"),
    (re.compile(r"\b(\w+)n[’']t\b", flags=re.IGNORECASE), r"\1 not"),
    (re.compile(r"[’']", flags=re.IGNORECASE), r"'"),
]


def expand_contractions(text: str) -> str:
    t = text
    for pat, repl in CONTRACTION_PATTERNS:
        t = pat.sub(repl, t)
    return t


def penn_to_wn(penn_pos: str):
    """Map Penn Treebank POS to WordNet POS."""
    if penn_pos.startswith("J"):
        return wordnet.ADJ
    if penn_pos.startswith("V"):
        return wordnet.VERB
    if penn_pos.startswith("N"):
        return wordnet.NOUN
    if penn_pos.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemma_token(token: str, penn_pos: str = "N") -> str:
    return WNL.lemmatize(token, pos=penn_to_wn(penn_pos))


def lemma_set(word: str):
    """Lemmas across POS to detect same-base relations (water/watery)."""
    w = (word or "").lower().strip()
    if not w:
        return set()
    poss = [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]
    return {WNL.lemmatize(w, pos=p) for p in poss} | {WNL.lemmatize(w)}


def clean_and_tag(text: str) -> List[Tuple[str, str]]:
    """Tokenize, lowercase, remove non-alpha/short tokens, POS-tag, remove stopwords + auxiliaries."""
    text = expand_contractions(text)
    raw_tokens = word_tokenize(text)
    tokens = [t.lower() for t in raw_tokens if t.isalpha() and len(t) >= 3]

    try:
        tagged = pos_tag(tokens, lang="eng")
    except TypeError:
        tagged = pos_tag(tokens)

    filtered = [(w, p) for (w, p) in tagged if w not in STOP and w not in AUXILIARIES]
    return filtered
