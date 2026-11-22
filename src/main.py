"""Kerangka integrasi CLI milik Orang 1. Jalankan sebagai python -m src.main."""

import argparse
import sys
from math import isfinite
from numbers import Real

from .contracts import Answer, QAResult, RetrievedPassage, UnavailableError


def _check_retrieval(passages: list[RetrievedPassage]) -> None:
    """Periksa kontrak sebelum passage masuk ke ekstraktor atau prompt."""
    if not isinstance(passages, list):
        raise ValueError("Hasil retrieval harus berupa daftar passage.")
    for passage in passages:
        if not isinstance(passage, dict) or not RetrievedPassage.__required_keys__.issubset(passage):
            raise ValueError("Passage retrieval harus memuat passage_id, text, score, dan source.")
        for key in ("passage_id", "text", "source"):
            if not isinstance(passage[key], str) or not passage[key].strip():
                raise ValueError(f"Kolom {key} pada passage retrieval harus berisi teks.")
        score = passage["score"]
        if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score):
            raise ValueError("Skor retrieval harus berupa angka berhingga.")


def _attach_passage(question: str, answer: Answer, passages: list[RetrievedPassage]) -> QAResult:
    """Cocokkan ID dan kutipan dengan keluaran retrieval agar sumber dapat ditampilkan."""
    if not isinstance(answer, dict) or not Answer.__required_keys__.issubset(answer):
        raise ValueError("Hasil modul jawaban harus memuat status, answer, passage_id, evidence, dan note.")
    if answer["status"] == "not_found":
        if any(answer[key] is not None for key in ("answer", "passage_id", "evidence")):
            raise ValueError("Status not_found harus memiliki jawaban, ID, dan bukti kosong.")
        return {**answer, "question": question, "source": None, "passage": None}
    text = answer["answer"]
    if answer["status"] not in ("answered", "needs_review") or not isinstance(text, str) or not text.strip():
        raise ValueError("Status atau teks jawaban tidak sesuai kontrak.")
    passage = next((p for p in passages if p["passage_id"] == answer["passage_id"]), None)
    if passage is None:
        raise ValueError("ID bukti tidak terdapat pada hasil retrieval.")
    evidence = answer["evidence"]
    if not isinstance(evidence, str) or not evidence.strip() or evidence not in passage["text"]:
        raise ValueError("Kutipan bukti tidak terdapat pada passage yang dirujuk.")
    return {**answer, "question": question, "source": passage["source"], "passage": passage}


def _read_question(question: str) -> str:
    if not question.strip():
        raise ValueError("Pertanyaan tidak boleh kosong.")
    return question.strip()


def run_classic(question: str) -> QAResult:
    """Hubungkan analisis Orang 2, retrieval klasik Orang 2, dan ekstraksi Orang 3."""
    question = _read_question(question)
    from .extract_answer import extract_answer
    from .question_processing import analyze_question
    from .retrieve_tfidf import retrieve_tfidf

    info = analyze_question(question)
    passages = retrieve_tfidf(question, k=3)
    _check_retrieval(passages)
    answer = extract_answer(question, info, passages)
    return _attach_passage(question, answer, passages)


def run_rag(question: str) -> QAResult:
    """Hubungkan analisis Orang 2, retrieval semantik Orang 4, dan generasi Orang 5."""
    question = _read_question(question)
    from .question_processing import analyze_question
    from .rag_generate import rag_answer
    from .retrieve_dense import retrieve_dense

    info = analyze_question(question)
    passages = retrieve_dense(question, k=3)
    _check_retrieval(passages)
    answer = rag_answer(question, passages, info)
    return _attach_passage(question, answer, passages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tanya jawab berdasarkan korpus Bahasa Indonesia.")
    parser.add_argument("--mode", required=True, choices=("classic", "rag"))
    parser.add_argument("--question", help="Tanpa opsi ini, CLI meminta pertanyaan melalui terminal.")
    args = parser.parse_args(argv)
    try:
        question = input("Pertanyaan Bahasa Indonesia: ") if args.question is None else args.question
        run = run_classic if args.mode == "classic" else run_rag
        result = run(question)
        passage = result["passage"]
        if passage is None:
            print("Jawaban tidak ditemukan dalam korpus.")
        else:
            print(f"Jawaban: {result['answer']}")
            print(f"Bukti: {result['evidence']}")
            print(f"Passage {passage['passage_id']}: {passage['text']}")
            print(f"Sumber: {result['source']}")
            if result["status"] == "needs_review":
                print("Jawaban perlu ditinjau anggota kelompok.")
        if result["note"]:
            print(f"Catatan: {result['note']}")
    except (UnavailableError, ImportError, OSError) as error:
        print(f"Mode {args.mode} belum dapat dijalankan. {error}", file=sys.stderr)
        return 3
    except EOFError:
        print("Input berakhir sebelum pertanyaan diterima.", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as error:
        print(f"Periksa input atau kontrak data. {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
