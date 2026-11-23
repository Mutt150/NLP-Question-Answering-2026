import csv
import unittest

from src.prepare_data import DATA_DIR, load_corpus
from src.retrieve_tfidf import _build_index, retrieve_tfidf


def load_dev():
    with (DATA_DIR / "processed" / "qa_dev.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def passage_ids(question, k=3):
    return [p["passage_id"] for p in retrieve_tfidf(question, k=k)]


class RetrievalFormatTests(unittest.TestCase):
    def test_format_sesuai_kontrak(self):
        for passage in retrieve_tfidf("Siapa ayah Martha Christina Tiahahu?"):
            self.assertEqual(set(passage), {"passage_id", "text", "score", "source"})
            self.assertIsInstance(passage["score"], float)
            self.assertTrue(passage["source"].startswith("https://"))

    def test_jumlah_hasil_mengikuti_k(self):
        self.assertEqual(len(retrieve_tfidf("Kartini", k=1)), 1)
        self.assertEqual(len(retrieve_tfidf("Kartini", k=5)), 5)
        with self.assertRaises(ValueError):
            retrieve_tfidf("Kartini", k=0)

    def test_skor_urut_menurun_dan_positif(self):
        scores = [p["score"] for p in retrieve_tfidf("Kapan Dewi Sartika mendirikan Sakola Istri?", k=10)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(score > 0 for score in scores))

    def test_teks_asli_tidak_diubah(self):
        corpus = {p["passage_id"]: p["text"] for p in load_corpus()}
        for passage in retrieve_tfidf("Di mana Cut Nyak Dhien dilahirkan?"):
            self.assertEqual(passage["text"], corpus[passage["passage_id"]])

    def test_tanpa_kata_cocok_hasil_kosong(self):
        self.assertEqual(retrieve_tfidf("xyz qwerty"), [])

    def test_indeks_dibangun_sekali(self):
        self.assertIs(_build_index(), _build_index())


class RetrievalQualityTests(unittest.TestCase):
    def test_hit_at_3_dev(self):
        rows = load_dev()
        hits = sum(row["evidence_passage_id"] in passage_ids(row["question"]) for row in rows)
        self.assertGreaterEqual(hits, 19, f"Hit@3 DEV hanya {hits}/{len(rows)}")

    def test_kata_sama_dengan_passage(self):
        self.assertEqual(passage_ids("Di kampung mana Bung Tomo dilahirkan?")[0], "DOC007_P02")
        self.assertEqual(passage_ids("Siapa ayah Martha Christina Tiahahu?")[0], "DOC015_P02")


class KnownLimitationTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_parafrase_nama_dan_imbuhan(self):
        # passage menulis "Sutomo dilahirkan", pertanyaan memakai "Bung Tomo" dan "kelahiran"
        self.assertIn("DOC007_P02", passage_ids("Tempat kelahiran Bung Tomo di Surabaya?"))


if __name__ == "__main__":
    unittest.main()
