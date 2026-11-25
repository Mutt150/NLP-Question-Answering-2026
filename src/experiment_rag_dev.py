"""Skrip uji mandiri modul RAG (Orang 5) pada 25 soal DEV.

Catatan: memakai retrieve_tfidf sebagai retriever sementara sebelum
retrieve_dense (Orang 4) tersedia di repository.
Jalankan: python -m src.experiment_rag_dev [--limit 5]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .contracts import UnavailableError
from .prepare_data import DATA_DIR
from .question_processing import analyze_question
from .rag_generate import rag_answer
from .retrieve_tfidf import retrieve_tfidf

DEFAULT_DEV_PATH = DATA_DIR / "processed" / "qa_dev.csv"
DEFAULT_OUT_PATH = DATA_DIR.parent / "results" / "rag_dev.csv"
OUT_FIELDS = (
    "question_id",
    "question",
    "answer_type",
    "ground_truth",
    "expected_passage_id",
    "retrieved_ids",
    "status",
    "rag_answer",
    "rag_passage_id",
    "evidence",
    "note",
)


def run_dev_experiment(limit: int | None = None, out_path: Path = DEFAULT_OUT_PATH) -> list[dict[str, str]]:
    with DEFAULT_DEV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]

    results: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    for row in rows:
        qid = row["question_id"]
        question = row["question"]
        q_info = analyze_question(question)
        passages = retrieve_tfidf(question, k=3)
        retrieved_ids = "|".join(p["passage_id"] for p in passages)

        try:
            ans = rag_answer(question, passages, q_info)
            status = ans["status"]
            rag_ans = ans["answer"] or ""
            rag_pid = ans["passage_id"] or ""
            evidence = ans["evidence"] or ""
            note = ans["note"] or ""
        except UnavailableError as error:
            status = "error"
            rag_ans = ""
            rag_pid = ""
            evidence = ""
            note = str(error)

        counts[status] += 1
        print(f"[{qid}] ({status}) {rag_ans or '-'} | pid={rag_pid or '-'}")
        results.append({
            "question_id": qid,
            "question": question,
            "answer_type": row["answer_type"],
            "ground_truth": row["ground_truth"],
            "expected_passage_id": row["evidence_passage_id"],
            "retrieved_ids": retrieved_ids,
            "status": status,
            "rag_answer": rag_ans,
            "rag_passage_id": rag_pid,
            "evidence": evidence,
            "note": note,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRingkasan ({len(results)} soal): {dict(counts)}")
    print(f"Hasil disimpan ke {out_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uji rag_answer pada soal DEV (retriever sementara TF-IDF).")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah soal DEV yang diuji.")
    args = parser.parse_args(argv)
    run_dev_experiment(limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
