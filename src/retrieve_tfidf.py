"""Antarmuka Orang 2 untuk retrieval klasik."""

from .contracts import RetrievedPassage, UnavailableError


def retrieve_tfidf(question: str, k: int = 3) -> list[RetrievedPassage]:
    """Gunakan load_corpus dan salinan preprocess. Kembalikan sampai k passage asli."""
    raise UnavailableError("retrieve_tfidf (Orang 2) belum tersedia.")
