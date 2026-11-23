"""Passage retrieval dengan TF-IDF dan cosine similarity."""

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .contracts import RetrievedPassage
from .prepare_data import load_corpus
from .preprocessing import preprocess
from .question_processing import QUESTION_WORDS, STOPWORDS, tokenize


def _index_tokens(text: str) -> list[str]:
    """Token untuk TF-IDF. Kata yang berulang tetap disimpan agar TF terhitung."""
    return [t for t in tokenize(preprocess(text)) if t not in STOPWORDS and t not in QUESTION_WORDS]


@lru_cache(maxsize=1)
def _build_index():
    """Bangun matriks TF-IDF korpus sekali, lalu dipakai ulang."""
    passages = load_corpus()
    vectorizer = TfidfVectorizer(tokenizer=_index_tokens, lowercase=False, token_pattern=None)
    matrix = vectorizer.fit_transform([p["text"] for p in passages])
    return passages, vectorizer, matrix


def retrieve_tfidf(question: str, k: int = 3) -> list[RetrievedPassage]:
    """Kembalikan sampai k passage asli dengan skor cosine tertinggi."""
    if k < 1:
        raise ValueError("k minimal 1.")
    passages, vectorizer, matrix = _build_index()
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    ranked = scores.argsort()[::-1][:k]
    return [
        {
            "passage_id": passages[i]["passage_id"],
            "text": passages[i]["text"],
            "score": float(scores[i]),
            "source": passages[i]["source"],
        }
        for i in ranked
        if scores[i] > 0
    ]
