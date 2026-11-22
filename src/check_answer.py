"""Antarmuka Orang 3 untuk memeriksa tipe keluaran bersama Orang 5."""

from .contracts import AnswerType, UnavailableError


def check_answer_type(answer: str, answer_type: AnswerType) -> bool | None:
    """Kembalikan True jika sesuai, False jika berlainan, None jika perlu tinjauan."""
    raise UnavailableError("check_answer_type (Orang 3) belum tersedia.")
