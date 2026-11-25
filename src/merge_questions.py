"""Gabungkan soal uji buatan tiap anggota menjadi data/qa_test.csv (milik Orang 4).

Tiap anggota mengisi SATU berkas CSV di data/qa_test_parts/ (misalnya jepri.csv) berisi 4 soal.
Kolom wajib, urutan persis: author,question,answer_type,ground_truth,evidence_doc_id,evidence_paragraph,evidence_text
Pemakaian: python -m src.merge_questions
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import get_args

from .contracts import AnswerType, CorpusPassage
from .evaluate import TEST_FIELDS
from .prepare_data import DATA_DIR, load_corpus, make_passage_id, read_rows

PART_FIELDS = TEST_FIELDS[1:]  # sama dengan soal final, tanpa question_id (ID diberikan saat digabung)
PARTS_DIR = DATA_DIR / "qa_test_parts"


def _key(text: str) -> str:
    return " ".join(text.split()).casefold()


def merge_questions(parts_dir: Path, passages: list[CorpusPassage], dev_questions: set[str] | None = None,
                    per_author: int = 4, total: int = 20) -> list[dict[str, str]]:
    """Validasi semua berkas anggota, kumpulkan SEMUA masalah sekaligus, lalu beri ID TEST001 dst."""
    files = sorted(Path(parts_dir).glob("*.csv"))
    if not files:
        raise ValueError(f"Tidak ada berkas .csv di {parts_dir}.")
    texts = {p["passage_id"]: p["text"] for p in passages}
    dev_keys = {_key(q) for q in dev_questions or ()}
    problems: list[str] = []
    rows: list[dict[str, str]] = []
    for path in files:
        try:
            rows.extend(read_rows(path, PART_FIELDS))
        except ValueError as error:
            problems.append(str(error))
    seen: dict[str, str] = {}
    for row in rows:
        who = f"{row['author']}: {row['question'][:40]}"
        if row["answer_type"] not in get_args(AnswerType):
            problems.append(f"{who} -> answer_type {row['answer_type']!r} tidak dikenal.")
        if _key(row["question"]) in seen:
            problems.append(f"{who} -> pertanyaan sama dengan milik {seen[_key(row['question'])]}.")
        seen.setdefault(_key(row["question"]), row["author"])
        if _key(row["question"]) in dev_keys:
            problems.append(f"{who} -> sama dengan soal DEV; buat soal baru.")
        try:
            passage_id = make_passage_id(row["evidence_doc_id"], int(row["evidence_paragraph"]))
        except ValueError:
            problems.append(f"{who} -> evidence_paragraph harus bilangan bulat.")
            continue
        if passage_id not in texts:
            problems.append(f"{who} -> passage bukti {passage_id} tidak ada.")
        elif row["evidence_text"] not in texts[passage_id]:
            problems.append(f"{who} -> evidence_text tidak ditemukan persis di {passage_id}.")
        elif row["ground_truth"] not in row["evidence_text"]:
            problems.append(f"{who} -> ground_truth tidak termuat di evidence_text.")
    for author, count in Counter(row["author"] for row in rows).items():
        if count != per_author:
            problems.append(f"{author} punya {count} soal, seharusnya {per_author}.")
    if len(rows) != total:
        problems.append(f"Jumlah soal {len(rows)}, seharusnya {total}.")
    if problems:
        raise ValueError("Soal belum lolos pemeriksaan:\n- " + "\n- ".join(problems))
    return [{"question_id": f"TEST{i:03d}", **row} for i, row in enumerate(rows, start=1)]


def _read_dev_questions(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["question"] for row in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gabungkan soal uji anggota menjadi qa_test.csv.")
    parser.add_argument("--parts", type=Path, default=PARTS_DIR)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "qa_test.csv")
    parser.add_argument("--per-author", type=int, default=4)
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Timpa qa_test.csv yang sudah ada.")
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and not args.force:
            print(f"{args.out} sudah ada. Pakai --force untuk menimpa.", file=sys.stderr)
            return 2
        rows = merge_questions(args.parts, load_corpus(), _read_dev_questions(DATA_DIR / "qa_dev.csv"),
                               args.per_author, args.total)
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 2
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} soal digabung ke {args.out}")
    for author, count in Counter(r["author"] for r in rows).items():
        print(f"  {author}: {count} soal")
    print("  Tipe jawaban:", dict(Counter(r["answer_type"] for r in rows)))
    print("Catatan: pemeriksaan silang oleh teman tetap harus dilakukan manual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
