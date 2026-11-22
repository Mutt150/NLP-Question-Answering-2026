"""Antarmuka Orang 4 untuk retrieval berbasis embedding."""

from .contracts import RetrievedPassage, UnavailableError


def retrieve_dense(question: str, k: int = 3) -> list[RetrievedPassage]:
    """Gunakan model embedding dan load_corpus. Kembalikan sampai k passage asli."""
    raise UnavailableError("retrieve_dense (Orang 4) belum tersedia.")
