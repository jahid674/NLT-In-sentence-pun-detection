# 🃏 Pun & Joke Detection using Linguistic Ambiguity

This project implements a Natural Language Processing (NLP) system to detect puns and jokes by leveraging lexical, semantic, and contextual ambiguity. The approach combines linguistic rules with modern embedding techniques to identify ambiguous words, explain why a sentence is humorous, and assess age appropriateness.

The implementation is demonstrated through a Jupyter Notebook.

---

## Project Overview

Puns and jokes often rely on:
- Words with **multiple meanings** (homographs, homonyms)
- **Contextual shifts** in interpretation
- **Semantic divergence** within a sentence

This project models humor detection by explicitly identifying and reasoning about such ambiguities rather than relying on black-box classifiers.

---

## Key Features

- 🔍 Detection of ambiguous words using **WordNet**
- 🧠 Semantic context analysis using **SBERT embeddings**
- 🧾 Explanation of multiple word meanings and humor mechanism
- 👶 Age-of-acquisition based **age suitability check**
- ❌ Identification of non-jokes and broken jokes
- 🧪 Tested on jokes, non-jokes, and logically invalid jokes

---

## 📂 Repository Structure

```text
.
├── Pun Detection.ipynb     # Main Jupyter Notebook
└── README.md              # Project documentation
