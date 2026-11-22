"""Siapkan passage biografi dan tautkan bukti soal pengembangan ke passage asli."""

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import cast, get_args

from .contracts import AnswerType, CorpusPassage
from .preprocessing import split_paragraphs


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CORPUS_FIELDS = ("passage_id", "doc_id", "title", "text", "source")
SOURCE_FIELDS = (
    "doc_id", "title", "filename", "source_title", "publisher", "source_url", "access_date", "license",
)
DEV_FIELDS = (
    "question_id", "question", "answer_type", "ground_truth",
    "evidence_doc_id", "evidence_paragraph", "evidence_text",
)


def read_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    """Baca CSV dengan atau tanpa BOM dan periksa struktur sebelum memakai isinya."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"{path.name} harus memiliki kolom {', '.join(fields)}.")
        rows = []
        for row in reader:
            if None in row or not all(value and value.strip() for value in row.values()):
                raise ValueError(f"{path.name} baris {reader.line_num}: kolom kosong atau berlebih.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} belum memiliki data.")
    return rows


def make_passage_id(doc_id: str, paragraph_number: int) -> str:
    return f"{doc_id}_P{paragraph_number:02d}"


def build_passages(data_dir: Path) -> list[CorpusPassage]:
    sources = read_rows(data_dir / "sources.csv", SOURCE_FIELDS)
    metadata = {row["doc_id"]: row for row in sources}
    if len(metadata) != len(sources):
        raise ValueError("doc_id berulang di sources.csv.")

    files = sorted((data_dir / "corpus").glob("*.txt"))
    if {path.stem for path in files} != set(metadata):
        raise ValueError("Daftar TXT di corpus/ berbeda dari daftar doc_id di sources.csv.")

    passages = []
    for path in files:
        doc_id = path.stem
        source = metadata[doc_id]
        if not re.fullmatch(r"DOC[0-9]{3}", doc_id) or source["filename"] != path.name:
            raise ValueError(f"{doc_id}: periksa format ID dan filename di sources.csv.")
        # Baca byte agar akhir baris di dalam paragraf juga tetap persis.
        paragraphs = split_paragraphs(path.read_bytes().decode("utf-8"))
        if not paragraphs:
            raise ValueError(f"{path.name} tidak memiliki paragraf.")
        for number, text in enumerate(paragraphs, start=1):
            passages.append(CorpusPassage(
                passage_id=make_passage_id(doc_id, number), doc_id=doc_id,
                title=source["title"], text=text, source=source["source_url"],
            ))
    return passages


def link_dev_evidence(path: Path, passages: list[CorpusPassage]) -> list[dict[str, str]]:
    questions = read_rows(path, DEV_FIELDS)
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("question_id berulang di qa_dev.csv.")
    texts = {passage["passage_id"]: passage["text"] for passage in passages}
    for question in questions:
        qid = question["question_id"]
        if question["answer_type"] not in get_args(AnswerType):
            raise ValueError(f"{qid}: answer_type tidak dikenal.")
        try:
            number = int(question["evidence_paragraph"])
        except ValueError as error:
            raise ValueError(f"{qid}: evidence_paragraph harus bilangan bulat.") from error
        passage_id = make_passage_id(question["evidence_doc_id"], number)
        if passage_id not in texts:
            raise ValueError(f"{qid}: passage bukti {passage_id} tidak tersedia.")
        evidence = question["evidence_text"]
        if evidence not in texts[passage_id] or question["ground_truth"] not in evidence:
            raise ValueError(f"{qid}: periksa kutipan bukti dan jawaban acuan pada {passage_id}.")
        question["evidence_passage_id"] = passage_id
    return questions


def load_corpus(path: Path = DATA_DIR / "processed" / "passages.csv") -> list[CorpusPassage]:
    """Baca passage asli untuk kedua retriever. Hanya korpus yang masuk ke inferensi."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} belum tersedia. Jalankan python -m src.prepare_data.")
    rows = read_rows(path, CORPUS_FIELDS)
    if len({row["passage_id"] for row in rows}) != len(rows):
        raise ValueError(f"passage_id berulang di {path.name}.")
    return cast(list[CorpusPassage], rows)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(data_dir: Path = DATA_DIR) -> tuple[int, int]:
    """Validasi semua masukan sebelum mengganti kedua berkas hasil di processed/."""
    passages = build_passages(data_dir)
    questions = link_dev_evidence(data_dir / "qa_dev.csv", passages)
    output = data_dir / "processed"
    output.mkdir(exist_ok=True)

    # Satu penulis pada satu waktu. Simpan kedua CSV lengkap sebelum menerbitkannya.
    staging = output / ".penyiapan"
    try:
        staging.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"{staging} masih ada. Periksa proses penyiapan sebelumnya.") from error
    try:
        write_csv(staging / "passages.csv", CORPUS_FIELDS, passages)
        write_csv(staging / "qa_dev.csv", (*DEV_FIELDS, "evidence_passage_id"), questions)
        previous = staging / "passages.bak"
        target = output / "passages.csv"
        if target.exists():
            shutil.copyfile(target, previous)
        (staging / "passages.csv").replace(target)
        try:
            (staging / "qa_dev.csv").replace(output / "qa_dev.csv")
        except OSError:
            # Misalnya CSV kedua terkunci di Excel. Pulihkan pasangan hasil sebelumnya.
            if previous.exists():
                previous.replace(target)
            else:
                target.unlink()
            raise
    finally:
        for name in ("passages.csv", "qa_dev.csv", "passages.bak"):
            (staging / name).unlink(missing_ok=True)
        staging.rmdir()
    return len(passages), len(questions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Folder data berisi corpus/, sources.csv dan qa_dev.csv.",
    )
    args = parser.parse_args()
    try:
        passages, questions = prepare_data(args.data_dir)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Persiapan data gagal: {error}\n")
    print(f"{passages} passage dan {questions} soal DEV tersimpan di {args.data_dir / 'processed'}.")


if __name__ == "__main__":
    main()
