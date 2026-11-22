"""Antarmuka Orang 2 untuk analisis pertanyaan Bahasa Indonesia."""

from .contracts import QuestionInfo, UnavailableError


def analyze_question(question: str) -> QuestionInfo:
    """Terima pertanyaan pengguna. Kembalikan answer_type, keywords, relation_cues."""
    raise UnavailableError("analyze_question (Orang 2) belum tersedia.")
