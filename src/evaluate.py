"""Antarmuka Orang 4 untuk evaluasi dua sistem pada pertanyaan buatan kelompok."""

from pathlib import Path

from .contracts import EvaluationRow, UnavailableError


def evaluate_all(qa_test: Path) -> list[EvaluationRow]:
    """Baca CSV uji. Kirim hanya question ke run_classic/run_rag sebelum penilaian."""
    raise UnavailableError("evaluate_all (Orang 4) belum tersedia.")
