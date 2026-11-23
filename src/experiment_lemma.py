"""Eksperimen pembanding: lemmatization Stanza vs stemming Sastrawi pada retrieval TF-IDF."""

import json

from .experiment_qp_tfidf import CONFIGS, RETRIEVAL_LIMITATIONS, load_dev, rank_of, rank_passages, summarize, write_summary
from .prepare_data import DATA_DIR, load_corpus
from .question_processing import QUESTION_WORDS, STOPWORDS, tokenize
from .retrieve_tfidf import _stem

LEMMA_CACHE_PATH = DATA_DIR.parent / "cache" / "lemma_cache.json"
SUMMARY_PATH = DATA_DIR.parent / "results" / "tfidf_lemma_summary.csv"


def _load_lemma_cache() -> dict[str, list[list[str]]]:
    if LEMMA_CACHE_PATH.exists():
        return json.loads(LEMMA_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


_lemma_cache = _load_lemma_cache()


def fill_lemma_cache(texts: list[str]) -> None:
    """Lemmatize teks yang belum ada di cache. Stanza dijalankan per teks utuh agar POS punya konteks."""
    missing = [t for t in dict.fromkeys(texts) if t not in _lemma_cache]
    if not missing:
        return
    import stanza

    nlp = stanza.Pipeline("id", processors="tokenize,pos,lemma", verbose=False)
    for text in missing:
        doc = nlp(text)
        _lemma_cache[text] = [[w.text.lower(), (w.lemma or w.text).lower()] for s in doc.sentences for w in s.words]
    LEMMA_CACHE_PATH.parent.mkdir(exist_ok=True)
    LEMMA_CACHE_PATH.write_text(json.dumps(_lemma_cache, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")


def _lemma_tokens(text: str) -> list[str]:
    """Lemma tanpa stopword dan kata tanya, aturan buangnya sama dengan konfigurasi lain."""
    tokens = [t for _, lemma in _lemma_cache[text] for t in tokenize(lemma)]
    return [t for t in tokens if t not in STOPWORDS and t not in QUESTION_WORDS]


LEMMA_CONFIGS = {
    "unigram": CONFIGS["unigram"],
    "unigram+stem": CONFIGS["unigram+stem"],
    "unigram+lemma": dict(tokenizer=_lemma_tokens, lowercase=False, token_pattern=None, ngram_range=(1, 1)),
}


def retrieval_report(dev: list[dict[str, str]]) -> None:
    print(f"== Retrieval TF-IDF (DEV, {len(dev)} soal) ==")
    print(f"{'konfigurasi':<16}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>7}   peringkat per soal")
    questions = [r["question"] for r in dev]
    summary = []
    for name, config in LEMMA_CONFIGS.items():
        ranks = [rank_of(ranking, r["evidence_passage_id"]) for ranking, r in zip(rank_passages(config, questions), dev)]
        hit1, hit3, mrr = summarize(ranks)
        summary.append({"config": name, "questions": len(dev), "hit1": hit1, "hit3": hit3, "mrr": round(mrr, 3)})
        shown = " ".join("-" if rank is None else str(rank) for rank in ranks)
        print(f"{name:<16}{hit1:>5}/{len(dev)}{hit3:>5}/{len(dev)}{mrr:>7.3f}   {shown}")
    print()
    write_summary(SUMMARY_PATH, summary)
    print(f"ringkasan disimpan di {SUMMARY_PATH.relative_to(DATA_DIR.parent)}")
    print()


def difference_report(texts: list[str]) -> None:
    """Bandingkan stem dan lemma untuk kata yang sama di korpus."""
    pairs = {}
    for text in texts:
        for word, lemma in _lemma_cache[text]:
            if word.isalpha() and word not in STOPWORDS and word not in QUESTION_WORDS:
                pairs[word] = (_stem(word), lemma)
    different = sorted((w, s, l) for w, (s, l) in pairs.items() if s != l)
    print("== Perbedaan stem dan lema ==")
    print(f"kata unik dibandingkan: {len(pairs)}, berbeda: {len(different)}")
    for word, stem, lemma in different[:30]:
        print(f"  {word:<18} stem {stem:<14} lema {lemma}")
    print()


def main() -> None:
    dev = load_dev()
    passages = [p["text"] for p in load_corpus()]
    extra = [q for q, _ in RETRIEVAL_LIMITATIONS]
    fill_lemma_cache(passages + [r["question"] for r in dev] + extra)
    retrieval_report(dev)
    print("== Keterbatasan retrieval (peringkat passage kunci) ==")
    for question, passage_id in RETRIEVAL_LIMITATIONS:
        ranks = [f"{name}: {rank_of(rank_passages(c, [question])[0], passage_id) or '-'}" for name, c in LEMMA_CONFIGS.items()]
        print(f"  {question}\n    {passage_id} -> " + ", ".join(ranks))
    print()
    difference_report(passages)


if __name__ == "__main__":
    main()
