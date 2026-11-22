"""Antarmuka Orang 3 untuk ekstraksi jawaban klasik."""

from .contracts import Answer, QuestionInfo, RetrievedPassage, UnavailableError


def extract_answer(question: str, q_info: QuestionInfo, passages: list[RetrievedPassage]) -> Answer:
    """Gunakan tipe, petunjuk relasi dan kandidat dari passage untuk memilih jawaban."""
    raise UnavailableError("extract_answer (Orang 3) belum tersedia.")
