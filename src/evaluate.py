"""Evaluasi dua sistem pada pertanyaan buatan kelompok (milik Orang 4).

Alur: baca qa_test.csv -> kirim HANYA kolom question ke run_classic/run_rag -> simpan prediksi ->
isi usulan benar/salah otomatis -> tim memeriksa dan memperbaiki kolom *_correct secara manual ->
hitung akurasi dari berkas yang sudah diperiksa.

Pemakaian:
    python -m src.evaluate --test data/qa_test.csv --out results/evaluation_results.csv
    python -m src.evaluate --dev                       # uji coba pada soal DEV
    python -m src.evaluate --summarize results/evaluation_results.csv
    python -m src.evaluate --retrieval data/qa_dev.csv data/qa_dev_tambahan.csv --retriever both
"""

import argparse
import csv
import re
import sys
import textwrap
import unicodedata
from pathlib import Path

from .contracts import EvaluationRow, UnavailableError
from .prepare_data import DATA_DIR, DEV_FIELDS, make_passage_id, read_rows

ROOT = DATA_DIR.parent
TEST_FIELDS = (
    "question_id", "author", "question", "answer_type", "ground_truth",
    "evidence_doc_id", "evidence_paragraph", "evidence_text",
)
RESULT_FIELDS = (
    "question_id", "question", "ground_truth", "classic_answer", "classic_passage_id", "classic_correct",
    "rag_answer", "rag_passage_id", "rag_correct", "error_reason",
)
NOT_FOUND_TRUTH = "tidak ditemukan"
# Galat yang dicatat per soal agar satu kegagalan tidak menghentikan 20 soal lainnya.
RUN_ERRORS = (UnavailableError, ImportError, OSError, ValueError, KeyError)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def auto_grade(ground_truth: str, answer: str | None) -> bool:
    """SEMENTARA (usulan otomatis, diganti penilaian manual tim). Usulan penilaian: jawaban benar jika seluruh jawaban acuan termuat utuh dalam jawaban sistem.

    Ini hanya titik awal. Jawaban parsial ("1922" untuk "3 Juli 1922") dinilai salah, sedangkan jawaban
    yang memuat acuan di tengah kalimat panjang dinilai benar. Tim wajib memeriksa maknanya satu per satu.
    """
    truth = _normalize(ground_truth)
    if truth.startswith(NOT_FOUND_TRUTH):
        return answer is None
    if answer is None:
        return False
    return f" {truth} " in f" {_normalize(answer)} "


def _run_system(run, question: str) -> tuple[str | None, str | None, bool | None, str | None]:
    """Jalankan satu sistem. Kembalikan (jawaban, id passage, usulan benar/salah, catatan)."""
    try:
        result = run(question)
    except RUN_ERRORS as error:
        return None, None, None, f"{type(error).__name__}: {error}"
    status = result["status"]
    note = None if status == "answered" else status
    return result["answer"], result["passage_id"], None, note


def evaluate_all(qa_test: Path, fields: tuple[str, ...] = TEST_FIELDS) -> list[EvaluationRow]:
    """Baca CSV uji. Kirim hanya question ke run_classic/run_rag sebelum penilaian."""
    # BERGANTUNG (belum teruji dengan modul asli): run_classic memakai Orang 2 (analyze_question,
    # retrieve_tfidf) dan Orang 3 (extract_answer); run_rag memakai Orang 2 (analyze_question) dan
    # Orang 5 (rag_answer). Kode di sini tidak perlu diganti saat modul mereka masuk.
    from .main import run_classic, run_rag

    questions = read_rows(Path(qa_test), fields)
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("question_id berulang di berkas uji.")

    rows: list[EvaluationRow] = []
    for item in questions:
        notes = []
        outputs = {}
        for name, run in (("classic", run_classic), ("rag", run_rag)):
            answer, passage_id, _, note = _run_system(run, item["question"])
            failed = note is not None and note not in ("needs_review", "not_found")
            # Jawaban acuan baru dipakai di sini, sesudah prediksi selesai.
            correct = None if failed else auto_grade(item["ground_truth"], answer)
            outputs[name] = (answer, passage_id, correct)
            if note:
                notes.append(f"{name}: {note}")
        rows.append(EvaluationRow(
            question_id=item["question_id"], question=item["question"], ground_truth=item["ground_truth"],
            classic_answer=outputs["classic"][0], classic_passage_id=outputs["classic"][1],
            classic_correct=outputs["classic"][2],
            rag_answer=outputs["rag"][0], rag_passage_id=outputs["rag"][1], rag_correct=outputs["rag"][2],
            error_reason="; ".join(notes) or None,
        ))
    return rows


def write_results(rows: list[EvaluationRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:  # BOM agar terbaca benar di Excel.
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in RESULT_FIELDS})


def _parse_flag(value: str, where: str) -> bool | None:
    value = value.strip().casefold()
    if value == "":
        return None
    if value in ("true", "benar", "1"):
        return True
    if value in ("false", "salah", "0"):
        return False
    raise ValueError(f"{where}: nilai benar/salah tidak dikenal: {value!r}. Isi TRUE, FALSE, atau kosongkan.")


def read_results(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError(f"Kolom {Path(path).name} harus {', '.join(RESULT_FIELDS)}.")
        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"Baris {reader.line_num}: kolom berlebih.")
            for key in ("classic_correct", "rag_correct"):
                row[key] = _parse_flag(row[key], f"Baris {reader.line_num} kolom {key}")
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Akurasi = jawaban benar / jumlah soal x 100. Soal yang belum dinilai dihitung terpisah."""
    total = len(rows)
    if total == 0:
        raise ValueError("Tidak ada baris untuk dirangkum.")
    summary = {"total": total}
    for name in ("classic", "rag"):
        flags = [row[f"{name}_correct"] for row in rows]
        correct = sum(flag is True for flag in flags)
        summary[name] = {
            "correct": correct, "wrong": sum(flag is False for flag in flags),
            "ungraded": sum(flag is None for flag in flags), "accuracy": round(correct / total * 100, 2),
        }
    return summary


def evaluate_retrieval(path: Path, retriever, k: int = 3) -> list[dict]:
    """Hit@k: apakah passage bukti ada di top-k. Dipakai pada qa_dev.csv untuk menyetel, atau pada
    qa_test.csv untuk menganalisis kegagalan SESUDAH evaluasi (jangan menyetel metode dari soal final)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        items = list(csv.DictReader(handle))
    needed = {"question_id", "question", "evidence_doc_id", "evidence_paragraph"}
    if not items or not needed <= set(items[0]):
        raise ValueError(f"Berkas harus memuat kolom {', '.join(sorted(needed))} dan minimal satu baris.")
    results = []
    for item in items:
        target = make_passage_id(item["evidence_doc_id"], int(item["evidence_paragraph"]))
        hits = list(retriever(item["question"], k=k))
        ids = [hit["passage_id"] for hit in hits]
        rank = ids.index(target) + 1 if target in ids else None
        results.append({"question_id": item["question_id"], "question": item["question"],
                        "ground_truth": item.get("ground_truth", ""), "evidence_text": item.get("evidence_text", ""),
                        "evidence_passage_id": target, "rank": rank, "hit": rank is not None,
                        "retrieved": ids, "hits": hits})
    return results


def _load_retriever(name: str):
    if name == "dense":
        from .retrieve_dense import retrieve_dense
        return retrieve_dense
    # BERGANTUNG: retrieve_tfidf milik Orang 2, belum teruji dengan modul asli.
    from .retrieve_tfidf import retrieve_tfidf
    return retrieve_tfidf


def _sentence_viewer(name: str):
    """Pemilih kalimat untuk tampilan rinci (hanya dense yang punya)."""
    if name == "dense":
        from .retrieve_dense import best_sentence
        return best_sentence
    return None


def _wrap(text: str, indent: int) -> str:
    return textwrap.fill(text, width=100, initial_indent=" " * indent, subsequent_indent=" " * indent)


def _print_retrieval_detail(name: str, results: list[dict], sentence_of=None, only_misses: bool = False) -> None:
    """Per soal: pertanyaan, passage bukti yang seharusnya, lalu tiga hasil retrieval dengan tanda BENAR/-."""
    print(f"\n===== {name} =====")
    shown = 0
    for r in results:
        if only_misses and r["hit"]:
            continue
        shown += 1
        status = f"ditemukan di peringkat {r['rank']}" if r["hit"] else "TIDAK masuk top-3"
        print(f"\n{r['question_id']}  {r['question']}")
        if r["ground_truth"]:
            print(f"  jawaban acuan : {r['ground_truth']}")
        print(f"  passage bukti : {r['evidence_passage_id']}  ({status})")
        if r["evidence_text"]:
            print(_wrap("kutipan bukti : " + r["evidence_text"], 2))
        for position, hit in enumerate(r["hits"], start=1):
            mark = "BENAR" if hit["passage_id"] == r["evidence_passage_id"] else "-"
            print(f"  {position}. {hit['passage_id']}  skor={hit['score']:.4f}  {mark}")
            if sentence_of is not None:
                print(_wrap(sentence_of(r["question"], hit["text"]), 7))
    if shown == 0:
        print("(tidak ada soal yang gagal)")


def _print_retrieval(results_by_name: dict[str, list[dict]]) -> None:
    """Satu baris per soal berisi peringkat bukti di tiap retriever ('-' = tidak masuk top-3)."""
    names = list(results_by_name)
    ids = [r["question_id"] for r in results_by_name[names[0]]]
    print("soal      " + "  ".join(f"{name:>6}" for name in names) + "   passage bukti")
    for index, question_id in enumerate(ids):
        ranks = [results_by_name[name][index]["rank"] for name in names]
        evidence = results_by_name[names[0]][index]["evidence_passage_id"]
        print(f"{question_id:<9} " + "  ".join(f"{(r if r else '-'):>6}" for r in ranks) + f"   {evidence}")
    for name in names:
        results = results_by_name[name]
        hits = sum(r["hit"] for r in results)
        print(f"Hit@3 {name}: {hits}/{len(results)} = {hits / len(results) * 100:.1f}%")


def _print_summary(summary: dict) -> None:
    labels = {"classic": "QA tanpa LLM (TF-IDF + ekstraksi)", "rag": "RAG (embedding + LLM)"}
    for name in ("classic", "rag"):
        s = summary[name]
        print(f"{labels[name]}: benar {s['correct']}/{summary['total']}, salah {s['wrong']}, "
              f"belum dinilai {s['ungraded']}, akurasi {s['accuracy']}%")
    if summary["classic"]["ungraded"] or summary["rag"]["ungraded"]:
        print("Peringatan: masih ada baris yang belum dinilai. Akurasi di atas menganggapnya belum benar.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluasi QA klasik dan RAG pada soal uji kelompok.")
    parser.add_argument("--test", type=Path, help="Default data/qa_test.csv.")
    parser.add_argument("--out", type=Path, help="Default results/evaluation_results.csv (atau evaluation_dev.csv jika --dev).")
    parser.add_argument("--dev", action="store_true",
                        help="Uji coba pada data/qa_dev.csv, bukan soal final. Hasil ke results/evaluation_dev.csv.")
    parser.add_argument("--force", action="store_true", help="Timpa berkas hasil yang sudah ada.")
    parser.add_argument("--summarize", type=Path, help="Hitung akurasi dari CSV hasil yang sudah diperiksa.")
    parser.add_argument("--retrieval", type=Path, nargs="+",
                        help="Hitung Hit@3 dari satu atau beberapa CSV soal (qa_dev.csv, qa_dev_tambahan.csv, qa_test.csv).")
    parser.add_argument("--retriever", choices=("dense", "tfidf", "both"), default="dense")
    parser.add_argument("--detail", action="store_true", help="Dengan --retrieval: tampilkan hasil per soal beserta penanda BENAR.")
    parser.add_argument("--hanya-gagal", action="store_true", help="Dengan --retrieval: tampilkan rinci hanya soal yang buktinya tidak masuk top-3.")
    args = parser.parse_args(argv)
    try:
        if args.summarize:
            _print_summary(summarize(read_results(args.summarize)))
        elif args.retrieval:
            names = ("dense", "tfidf") if args.retriever == "both" else (args.retriever,)
            results_by_name = {}
            for name in names:
                try:
                    retriever = _load_retriever(name)
                    results_by_name[name] = [r for path in args.retrieval for r in evaluate_retrieval(path, retriever)]
                except (UnavailableError, ImportError, OSError) as error:
                    if len(names) == 1:
                        raise
                    print(f"[{name}] dilewati: {error}", file=sys.stderr)
            if not results_by_name:
                raise UnavailableError("Tidak ada retriever yang dapat dijalankan.")
            if args.detail or args.hanya_gagal:
                for name, results in results_by_name.items():
                    _print_retrieval_detail(name, results, _sentence_viewer(name), only_misses=args.hanya_gagal)
                print()
            _print_retrieval(results_by_name)
        else:
            test = args.test or DATA_DIR / ("qa_dev.csv" if args.dev else "qa_test.csv")
            out = args.out or ROOT / "results" / ("evaluation_dev.csv" if args.dev else "evaluation_results.csv")
            if out.exists() and not args.force:
                print(f"{out} sudah ada dan mungkin berisi penilaian manual. Pakai --force untuk menimpa.",
                      file=sys.stderr)
                return 2
            rows = evaluate_all(test, DEV_FIELDS if args.dev else TEST_FIELDS)
            write_results(rows, out)
            print(f"{len(rows)} soal dievaluasi. Hasil: {out}")
            print("Kolom *_correct baru usulan otomatis. Periksa maknanya, perbaiki di CSV, lalu jalankan --summarize.")
            if args.dev:
                print("Catatan: ini uji coba pada soal DEV (dipakai untuk pengembangan), bukan evaluasi akhir.")
    except (UnavailableError, ImportError, OSError) as error:
        print(f"Tidak dapat dijalankan. {error}", file=sys.stderr)
        return 3
    except (ValueError, KeyError) as error:
        print(f"Periksa input atau kontrak data. {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
