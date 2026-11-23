"""Passage retrieval dengan TF-IDF dan cosine similarity."""

import json
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .contracts import RetrievedPassage
from .prepare_data import DATA_DIR, load_corpus
from .preprocessing import preprocess
from .question_processing import QUESTION_WORDS, STOPWORDS, tokenize

# hasil stem disimpan di file karena Sastrawi lambat untuk kata di luar kamus
STEM_CACHE_PATH = DATA_DIR.parent / "cache" / "stem_cache.json"


def _load_stem_cache() -> dict[str, str]:
    if STEM_CACHE_PATH.exists():
        return json.loads(STEM_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


_stem_cache = _load_stem_cache()
_cache_changed = False


@lru_cache(maxsize=1)
def _stemmer():
    """Kamus Sastrawi baru dimuat saat ada kata yang belum ada di cache."""
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    return StemmerFactory().create_stemmer()


def _save_stem_cache() -> None:
    global _cache_changed
    if not _cache_changed:
        return
    STEM_CACHE_PATH.parent.mkdir(exist_ok=True)
    text = json.dumps(dict(sorted(_stem_cache.items())), ensure_ascii=False, indent=0)
    STEM_CACHE_PATH.write_text(text + "\n", encoding="utf-8")
    _cache_changed = False


def _index_tokens(text: str) -> list[str]:
    """Token tanpa stopword dan kata tanya. Kata yang berulang tetap disimpan agar TF terhitung."""
    return [t for t in tokenize(preprocess(text)) if t not in STOPWORDS and t not in QUESTION_WORDS]


def _stem(word: str) -> str:
    global _cache_changed
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer().stem(word)
        _cache_changed = True
    return _stem_cache[word]


def _stem_tokens(text: str) -> list[str]:
    """Token yang sudah di-stem, dipakai TF-IDF."""
    return [_stem(t) for t in _index_tokens(text)]


@lru_cache(maxsize=1)
def _build_index():
    """Bangun matriks TF-IDF korpus sekali, lalu dipakai ulang."""
    passages = load_corpus()
    vectorizer = TfidfVectorizer(tokenizer=_stem_tokens, lowercase=False, token_pattern=None)
    matrix = vectorizer.fit_transform([p["text"] for p in passages])
    _save_stem_cache()
    return passages, vectorizer, matrix


def retrieve_tfidf(question: str, k: int = 3) -> list[RetrievedPassage]:
    """Kembalikan sampai k passage asli dengan skor cosine tertinggi."""
    if k < 1:
        raise ValueError("k minimal 1.")
    passages, vectorizer, matrix = _build_index()
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    ranked = scores.argsort()[::-1][:k]
    return [
        {
            "passage_id": passages[i]["passage_id"],
            "text": passages[i]["text"],
            "score": float(scores[i]),
            "source": passages[i]["source"],
        }
        for i in ranked
        if scores[i] > 0
    ]
