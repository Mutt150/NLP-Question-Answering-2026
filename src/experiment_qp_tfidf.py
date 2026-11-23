"""Eksperimen question processing dan retrieval TF-IDF pada soal DEV."""

import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .prepare_data import DATA_DIR, load_corpus
from .question_processing import detect_answer_type
from .retrieve_tfidf import _index_tokens, _stem_tokens, retrieve_tfidf

RESULTS_PATH = DATA_DIR.parent / "results" / "tfidf_dev.csv"
SUMMARY_PATH = DATA_DIR.parent / "results" / "tfidf_dev_summary.csv"
SYSTEM_CONFIG = "unigram+stem"


def _clean(text: str) -> str:
    return " ".join(_index_tokens(text))


def _stem_clean(text: str) -> str:
    return " ".join(_stem_tokens(text))


# preprocessing disamakan, yang dibandingkan hanya jenis term
CONFIGS = {
    "unigram": dict(tokenizer=_index_tokens, lowercase=False, token_pattern=None, ngram_range=(1, 1)),
    "unigram+bigram": dict(tokenizer=_index_tokens, lowercase=False, token_pattern=None, ngram_range=(1, 2)),
    "unigram+stem": dict(tokenizer=_stem_tokens, lowercase=False, token_pattern=None, ngram_range=(1, 1)),
    "stem+bigram": dict(tokenizer=_stem_tokens, lowercase=False, token_pattern=None, ngram_range=(1, 2)),
    "char 3-5": dict(preprocessor=_clean, lowercase=False, analyzer="char_wb", ngram_range=(3, 5)),
    "stem+char 3-5": dict(preprocessor=_stem_clean, lowercase=False, analyzer="char_wb", ngram_range=(3, 5)),
}

TYPE_LIMITATIONS = [
    ("Sebutkan nama ayah Kartini", "PERSON"),
    ("Bung Tomo wafat tahun?", "DATE"),
    ("Apakah nama partai yang didirikan Sjahrir?", "ORGANIZATION"),
    ("Organisasi yang dipimpin Hatta bernama apa?", "ORGANIZATION"),
]

RETRIEVAL_LIMITATIONS = [
    ("Tempat kelahiran Bung Tomo di Surabaya?", "DOC007_P02"),
]


def load_dev() -> list[dict[str, str]]:
    with (DATA_DIR / "processed" / "qa_dev.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rank_passages(config: dict, questions: list[str]) -> list[list[tuple[str, float]]]:
    """Urutkan seluruh passage untuk tiap pertanyaan, dari skor tertinggi."""
    passages = load_corpus()
    vectorizer = TfidfVectorizer(**config)
    matrix = vectorizer.fit_transform([p["text"] for p in passages])
    scores = cosine_similarity(vectorizer.transform(questions), matrix)
    return [[(passages[i]["passage_id"], float(row[i])) for i in row.argsort()[::-1]] for row in scores]


def rank_of(ranking: list[tuple[str, float]], passage_id: str) -> int | None:
    """Peringkat passage mulai dari 1. None jika skornya 0."""
    for position, (pid, score) in enumerate(ranking, start=1):
        if pid == passage_id:
            return position if score > 0 else None
    return None


def summarize(ranks: list[int | None]) -> tuple[int, int, float]:
    """Hit@1, Hit@3, dan MRR dari daftar peringkat passage kunci."""
    hit1 = sum(rank == 1 for rank in ranks)
    hit3 = sum(rank is not None and rank <= 3 for rank in ranks)
    mrr = sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
    return hit1, hit3, mrr


def answer_type_report(dev: list[dict[str, str]]) -> None:
    print("== Question processing (DEV) ==")
    wrong = [(r, detect_answer_type(r["question"])) for r in dev if detect_answer_type(r["question"]) != r["answer_type"]]
    print(f"answer_type benar : {len(dev) - len(wrong)}/{len(dev)}")
    for row, predicted in wrong:
        print(f"  {row['question_id']} {predicted} (kunci {row['answer_type']})")
    print()


def retrieval_report(dev: list[dict[str, str]]) -> tuple[list, list]:
    """Cetak hasil semua konfigurasi, lalu kembalikan ranking dan peringkat konfigurasi sistem."""
    print(f"== Retrieval TF-IDF (DEV, {len(dev)} soal) ==")
    print(f"{'konfigurasi':<16}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>7}   peringkat per soal")
    questions = [r["question"] for r in dev]
    rows = []
    summary = []
    for name, config in CONFIGS.items():
        rankings = rank_passages(config, questions)
        ranks = [rank_of(ranking, r["evidence_passage_id"]) for ranking, r in zip(rankings, dev)]
        hit1, hit3, mrr = summarize(ranks)
        summary.append({"config": name, "questions": len(dev), "hit1": hit1, "hit3": hit3, "mrr": round(mrr, 3)})
        shown = " ".join("-" if rank is None else str(rank) for rank in ranks)
        print(f"{name:<16}{hit1:>5}/{len(dev)}{hit3:>5}/{len(dev)}{mrr:>7.3f}   {shown}")
        for ranking, row, rank in zip(rankings, dev, ranks):
            rows.append({
                "question_id": row["question_id"],
                "config": name,
                "rank": "" if rank is None else rank,
                "hit1": int(rank == 1),
                "hit3": int(rank is not None and rank <= 3),
                "rr": 0 if rank is None else round(1 / rank, 3),
                "top3": "|".join(pid for pid, score in ranking[:3] if score > 0),
            })
        if name == SYSTEM_CONFIG:
            _check_same_as_system(rankings, questions)
            system_rankings, system_ranks = rankings, ranks
    print()
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(SUMMARY_PATH, summary)
    print(f"hasil per soal disimpan di {RESULTS_PATH.relative_to(DATA_DIR.parent)}")
    print(f"ringkasan disimpan di {SUMMARY_PATH.relative_to(DATA_DIR.parent)}")
    print()
    return system_rankings, system_ranks


def _check_same_as_system(rankings, questions) -> None:
    for ranking, question in zip(rankings, questions):
        expected = [p["passage_id"] for p in retrieve_tfidf(question, k=3)]
        if [pid for pid, score in ranking[:3] if score > 0] != expected:
            raise RuntimeError(f"Konfigurasi {SYSTEM_CONFIG} berbeda dengan retrieve_tfidf pada: {question}")


def write_summary(path, summary: list[dict]) -> None:
    """Simpan Hit@1, Hit@3, dan MRR per konfigurasi."""
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "questions", "hit1", "hit3", "mrr"])
        writer.writeheader()
        writer.writerows(summary)


def failure_report(dev: list[dict[str, str]], rankings: list, ranks: list) -> None:
    print(f"== Soal DEV di luar top 3 ({SYSTEM_CONFIG}) ==")
    for row, ranking, rank in zip(dev, rankings, ranks):
        if rank is not None and rank <= 3:
            continue
        top3 = " | ".join(pid for pid, score in ranking[:3] if score > 0)
        print(f"  {row['question_id']} {row['question']}")
        print(f"    kunci {row['evidence_passage_id']} peringkat {rank or '-'}, top 3: {top3}")
    print()


def limitation_report() -> None:
    print("== Contoh kasus di luar DEV: answer_type ==")
    for question, expected in TYPE_LIMITATIONS:
        print(f"  {question:<46} -> {detect_answer_type(question):<13} (seharusnya {expected})")
    print()
    print("== Contoh kasus di luar DEV: nama panggilan ==")
    for question, passage_id in RETRIEVAL_LIMITATIONS:
        ranks = []
        for name, config in CONFIGS.items():
            rank = rank_of(rank_passages(config, [question])[0], passage_id)
            ranks.append(f"{name}: {'-' if rank is None else rank}")
        print(f"  {question}")
        print(f"    {passage_id} -> " + ", ".join(ranks))


def main() -> None:
    dev = load_dev()
    answer_type_report(dev)
    rankings, ranks = retrieval_report(dev)
    failure_report(dev, rankings, ranks)
    limitation_report()


if __name__ == "__main__":
    main()
