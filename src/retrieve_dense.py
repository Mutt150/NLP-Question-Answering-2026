"""Retrieval semantik milik Orang 4: embedding kalimat + cosine similarity (PBA11, PBA12).

Alur: teks passage -> vektor (sekali, lalu di-cache) ; pertanyaan -> vektor ;
skor = cosine similarity ; kembalikan top-k passage ASLI beserta ID dan skor.
Model embedding adalah pretrained dari pustaka sentence-transformers (bukan buatan sendiri).
Perhitungan normalisasi vektor, cosine, pengurutan, cache, dan format keluaran ditulis sendiri.
"""

import argparse
import hashlib
import re
import sys
import textwrap
from pathlib import Path
from typing import Protocol

import numpy as np

from .contracts import CorpusPassage, RetrievedPassage, UnavailableError
from .prepare_data import DATA_DIR, load_corpus
from .preprocessing import preprocess

MODEL_NAME = "intfloat/multilingual-e5-small"
# Keluarga model E5 dilatih dengan awalan ini. Ganti keduanya jadi "" jika memakai model lain
# (misalnya paraphrase-multilingual-MiniLM-L12-v2).
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
CACHE_DIR = DATA_DIR / "cache"


# Singkatan yang diikuti titik tetapi bukan akhir kalimat.
_ABBREVIATIONS = {"dr", "drs", "ir", "prof", "hj", "mr", "no", "dll", "dsb", "dst", "st", "kh", "tn", "ny", "nn"}


def _ends_with_abbreviation(sentence: str) -> bool:
    last = sentence.rsplit(None, 1)[-1]
    return bool(re.fullmatch(r"(?:[A-Za-z]\.)+", last)) or last.rstrip(".").casefold() in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Pecah paragraf menjadi kalimat. Titik pada inisial (R. A.) dan singkatan (No., Dr.) tidak memecah."""
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"\u201c(]?[A-Z0-9])", " ".join(text.split()))
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return [sentence for sentence in sentences if sentence]


class Encoder(Protocol):
    cache_key: str

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        """Kembalikan matriks (jumlah teks, dimensi)."""


class SentenceTransformerEncoder:
    """Pembungkus tipis model sentence-transformers. Model baru dimuat saat pertama dipakai."""

    def __init__(self, model_name: str = MODEL_NAME, query_prefix: str = QUERY_PREFIX,
                 passage_prefix: str = PASSAGE_PREFIX):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.cache_key = f"{model_name}|{passage_prefix}"
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise UnavailableError(
                    "Paket sentence-transformers belum terpasang. "
                    "Jalankan python -m pip install -r requirements.txt."
                ) from error
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as error:  # Unduhan/pemuatan model dapat gagal dengan banyak jenis galat.
                raise UnavailableError(f"Model embedding {self.model_name} tidak dapat dimuat: {error}") from error
        return self._model

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.passage_prefix
        model = self._load()
        vectors = model.encode(
            [prefix + text for text in texts], batch_size=16, convert_to_numpy=True,
            normalize_embeddings=False, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float64)


def _unit_rows(matrix, rows: int) -> np.ndarray:
    """Periksa bentuk lalu skalakan tiap baris ke panjang 1, sehingga dot product = cosine."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != rows:
        raise ValueError(f"Encoder harus mengembalikan matriks dengan {rows} baris.")
    if not np.isfinite(matrix).all():
        raise ValueError("Vektor embedding memuat nilai tak berhingga atau NaN.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Vektor embedding bernilai nol sehingga cosine tidak terdefinisi.")
    return matrix / norms


class DenseRetriever:
    def __init__(self, passages: list[CorpusPassage], encoder: Encoder, cache_dir: Path | None = CACHE_DIR):
        if not passages:
            raise ValueError("Korpus kosong.")
        self.passages = list(passages)
        self.encoder = encoder
        self.vectors = self._load_or_build(cache_dir)

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(self.encoder.cache_key.encode("utf-8"))
        for passage in self.passages:
            digest.update(b"\0" + passage["passage_id"].encode("utf-8"))
            digest.update(b"\0" + preprocess(passage["text"]).encode("utf-8"))
        return digest.hexdigest()

    def _load_or_build(self, cache_dir: Path | None) -> np.ndarray:
        fingerprint = self._fingerprint()
        path = None if cache_dir is None else Path(cache_dir) / f"dense_{fingerprint[:16]}.npz"
        if path is not None and path.is_file():
            try:
                with np.load(path) as data:
                    vectors = data["vectors"]
                    if str(data["fingerprint"]) == fingerprint and vectors.shape[0] == len(self.passages):
                        return _unit_rows(vectors, len(self.passages))
            except (OSError, ValueError, KeyError):
                pass  # Cache rusak atau usang: hitung ulang.
        texts = [preprocess(passage["text"]) for passage in self.passages]
        vectors = _unit_rows(self.encoder.encode(texts, is_query=False), len(texts))
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + ".tmp")
                with temp.open("wb") as handle:
                    np.savez(handle, vectors=vectors, fingerprint=np.array(fingerprint))
                temp.replace(path)
            except OSError:
                pass  # Cache hanya mempercepat; kegagalan menulisnya tidak boleh menggagalkan retrieval.
        return vectors

    def search(self, question: str, k: int = 3) -> list[RetrievedPassage]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Pertanyaan tidak boleh kosong.")
        if k < 1:
            raise ValueError("k harus minimal 1.")
        query = _unit_rows(self.encoder.encode([preprocess(question)], is_query=True), 1)[0]
        scores = self.vectors @ query
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedPassage(
                passage_id=self.passages[i]["passage_id"], text=self.passages[i]["text"],
                score=float(scores[i]), source=self.passages[i]["source"],
            )
            for i in order
        ]

    def best_sentence(self, question: str, text: str) -> str:
        """Kalimat dalam teks yang paling mirip dengan pertanyaan (cosine, model yang sama).
        Hanya untuk tampilan: kalimat itu belum tentu memuat jawaban."""
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return text.strip()
        query = _unit_rows(self.encoder.encode([preprocess(question)], is_query=True), 1)[0]
        vectors = _unit_rows(self.encoder.encode([preprocess(x) for x in sentences], is_query=False), len(sentences))
        return sentences[int(np.argmax(vectors @ query))]


_retriever: DenseRetriever | None = None


def get_retriever() -> DenseRetriever:
    """Bangun retriever sekali per proses. Gagal dengan UnavailableError jika model tidak tersedia."""
    global _retriever
    if _retriever is None:
        _retriever = DenseRetriever(load_corpus(), SentenceTransformerEncoder())
    return _retriever


def retrieve_dense(question: str, k: int = 3) -> list[RetrievedPassage]:
    """Gunakan model embedding dan load_corpus. Kembalikan sampai k passage asli."""
    return get_retriever().search(question, k)


def best_sentence(question: str, text: str) -> str:
    return get_retriever().best_sentence(question, text)


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.retrieve_dense",
                                     description="Tampilkan top-k passage untuk satu pertanyaan.")
    parser.add_argument("question", nargs="+", help="Pertanyaan (boleh tanpa tanda kutip).")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--penuh", action="store_true", help="Tampilkan paragraf utuh, bukan satu kalimat.")
    args = parser.parse_args(argv)
    question = " ".join(args.question)
    try:
        for rank, hit in enumerate(retrieve_dense(question, k=args.k), start=1):
            shown = hit["text"] if args.penuh else best_sentence(question, hit["text"])
            print(f"{rank}. {hit['passage_id']}  skor={hit['score']:.4f}")
            print(textwrap.fill(shown, width=100, initial_indent="   ", subsequent_indent="   "))
    except (UnavailableError, OSError, ValueError) as error:
        print(f"Gagal: {error}", file=sys.stderr)
        return 1
    if not args.penuh:
        print("\n(Kalimat yang paling mirip dengan pertanyaan, belum tentu memuat jawaban. Tambah --penuh untuk paragraf utuh.)")
    return 0


if __name__ == "__main__":
    # Contoh: python -m src.retrieve_dense Siapa ayah Martha Christina Tiahahu?
    raise SystemExit(_cli(sys.argv[1:]))
