import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.contracts import RetrievedPassage, UnavailableError
from src.main import _check_retrieval
from src.prepare_data import load_corpus
from src.retrieve_dense import DenseRetriever, SentenceTransformerEncoder, _cli, split_sentences

DIM = 64


class BagOfWordsEncoder:
    """Encoder palsu untuk tes mekanik. Ini BUKAN model semantik dan tidak mengukur kualitas retrieval."""

    cache_key = "fake|"

    def __init__(self):
        self.calls = []

    def encode(self, texts, *, is_query):
        self.calls.append((len(texts), is_query))
        matrix = np.zeros((len(texts), DIM))
        for row, text in enumerate(texts):
            for word in text.casefold().replace("?", " ").replace(".", " ").split():
                matrix[row, zlib.crc32(word.encode()) % DIM] += 1
        return matrix


def passage(pid, text):
    return {"passage_id": pid, "doc_id": pid[:6], "title": "t", "text": text, "source": f"https://contoh/{pid}"}


PASSAGES = [
    passage("A_P01", "Kartini lahir di Jepara pada 21 April 1879."),
    passage("B_P01", "Cut Nyak Dhien dilahirkan di Lampadang\nwilayah VI Mukim."),
    passage("C_P01", "Diponegoro memimpin Perang Jawa melawan Belanda."),
]


class DenseRetrieverTests(unittest.TestCase):
    def make(self, cache_dir=None, passages=PASSAGES, encoder=None):
        return DenseRetriever(passages, encoder or BagOfWordsEncoder(), cache_dir=cache_dir)

    def test_urutan_skor_teks_asli_dan_format_kontrak(self):
        hits = self.make().search("Di mana Cut Nyak Dhien dilahirkan", k=3)
        self.assertEqual(hits[0]["passage_id"], "B_P01")
        self.assertEqual(hits[0]["text"], PASSAGES[1]["text"])  # teks asli, baris baru tetap ada
        self.assertEqual(hits[0]["source"], "https://contoh/B_P01")
        self.assertIsInstance(hits[0]["score"], float)
        self.assertEqual([h["score"] for h in hits], sorted((h["score"] for h in hits), reverse=True))
        self.assertTrue(RetrievedPassage.__required_keys__ <= hits[0].keys())
        _check_retrieval(hits)

    def test_jumlah_hasil_mengikuti_k(self):
        retriever = self.make()
        self.assertEqual(len(retriever.search("Kartini", k=1)), 1)
        self.assertEqual(len(retriever.search("Kartini", k=10)), 3)

    def test_input_tidak_valid(self):
        retriever = self.make()
        for question, k in (("", 3), ("   ", 3), ("Kartini", 0)):
            with self.subTest(question=question, k=k), self.assertRaises(ValueError):
                retriever.search(question, k=k)
        with self.assertRaises(ValueError):
            DenseRetriever([], BagOfWordsEncoder(), cache_dir=None)

    def test_vektor_nol_ditolak(self):
        with self.assertRaisesRegex(ValueError, "nol"):
            self.make().search("???")  # tidak ada kata, vektor kueri nol

    def test_cache_dipakai_ulang_dan_dihitung_ulang_jika_teks_berubah(self):
        with tempfile.TemporaryDirectory() as folder:
            first = BagOfWordsEncoder()
            self.make(folder, encoder=first)
            self.assertEqual(first.calls, [(3, False)])
            second = BagOfWordsEncoder()
            self.make(folder, encoder=second)
            self.assertEqual(second.calls, [])  # dibaca dari cache
            changed = [PASSAGES[0], passage("B_P01", "Teks lain sama sekali."), PASSAGES[2]]
            third = BagOfWordsEncoder()
            self.make(folder, passages=changed, encoder=third)
            self.assertEqual(third.calls, [(3, False)])

    def test_cache_rusak_tidak_menggagalkan_retrieval(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make(folder)
            for cache in Path(folder).glob("dense_*.npz"):
                cache.write_bytes(b"bukan npz")
            encoder = BagOfWordsEncoder()
            self.make(folder, encoder=encoder)
            self.assertEqual(encoder.calls, [(3, False)])

    def test_paket_embedding_belum_terpasang_menjadi_unavailable(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaisesRegex(UnavailableError, "sentence-transformers"):
                SentenceTransformerEncoder().encode(["uji"], is_query=True)

    def test_seluruh_korpus_nyata_lolos_kontrak_untuk_soal_dev(self):
        retriever = DenseRetriever(load_corpus(), BagOfWordsEncoder(), cache_dir=None)
        self.assertEqual(len(retriever.passages), 75)
        hits = retriever.search("Siapa ayah Martha Christina Tiahahu?", k=3)
        _check_retrieval(hits)
        self.assertEqual(len({h["passage_id"] for h in hits}), 3)


class SentenceDisplayTests(unittest.TestCase):
    def test_singkatan_dan_inisial_tidak_memecah_kalimat(self):
        text = "R. A. Kartini lahir di Jepara. Ia bersekolah di ELS. Putra G.P.H. Soerjaningrat lahir pada 1889. Nomor No. 29 dipakai."
        self.assertEqual(split_sentences(text), [
            "R. A. Kartini lahir di Jepara.", "Ia bersekolah di ELS.",
            "Putra G.P.H. Soerjaningrat lahir pada 1889.", "Nomor No. 29 dipakai."])

    def test_spasi_dan_baris_baru_dirapikan_serta_kosong_ditangani(self):
        self.assertEqual(split_sentences("Satu kalimat\nyang  terputus."), ["Satu kalimat yang terputus."])
        self.assertEqual(split_sentences("   "), [])

    def test_kalimat_terpilih_utuh_tanpa_titik_titik(self):
        retriever = DenseRetriever([passage("X_P01", "Kalimat satu.")], BagOfWordsEncoder(), cache_dir=None)
        text = "Pada 1660 VOC menyerang Makassar. Benteng Sombaopu dikuasai Kompeni pada 24 Juni 1669. Sultan wafat pada 1670."
        chosen = retriever.best_sentence("Di mana Benteng Sombaopu dikuasai Kompeni", text)
        self.assertEqual(chosen, "Benteng Sombaopu dikuasai Kompeni pada 24 Juni 1669.")
        self.assertNotIn("...", chosen)
        self.assertEqual(retriever.best_sentence("apa saja", "Hanya satu kalimat."), "Hanya satu kalimat.")

    def run_cli(self, argv):
        import contextlib
        import io
        retriever = DenseRetriever(PASSAGES, BagOfWordsEncoder(), cache_dir=None)
        out = io.StringIO()
        with patch("src.retrieve_dense.get_retriever", return_value=retriever), contextlib.redirect_stdout(out):
            code = _cli(argv)
        return code, out.getvalue()

    def test_cli_menampilkan_top_k_dengan_kalimat_tanpa_jawaban_dan_tanpa_titik_titik(self):
        code, out = self.run_cli(["Cut", "Nyak", "Dhien", "dilahirkan", "di", "Lampadang"])
        self.assertEqual(code, 0)
        self.assertIn("1. B_P01  skor=", out)
        self.assertIn("Lampadang", out)
        self.assertIn("belum tentu memuat jawaban", out)
        self.assertNotIn("Jawaban:", out)
        self.assertNotIn("...", out)

    def test_cli_penuh_menampilkan_paragraf_utuh(self):
        code, out = self.run_cli(["Cut", "Nyak", "Dhien", "dilahirkan", "di", "Lampadang", "--penuh"])
        self.assertEqual(code, 0)
        self.assertIn("wilayah VI Mukim", out)
        self.assertNotIn("belum tentu", out)
        self.assertNotIn("Jawaban:", out)


if __name__ == "__main__":
    unittest.main()
