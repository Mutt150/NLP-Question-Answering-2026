"""Antarmuka Orang 5 untuk generasi jawaban berbasis passage."""

from .contracts import Answer, QuestionInfo, RetrievedPassage, UnavailableError


def rag_answer(question: str, passages: list[RetrievedPassage], q_info: QuestionInfo) -> Answer:
    """Terima passage retrieval dan analisis pertanyaan, lalu kembalikan jawaban dan bukti."""
    raise UnavailableError("rag_answer (Orang 5) belum tersedia.")
