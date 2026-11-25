import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.contracts import UnavailableError
from src.evaluate import (
    RESULT_FIELDS, TEST_FIELDS, auto_grade, evaluate_all, evaluate_retrieval, main, read_results,
    summarize, write_results,
)

ROWS = [
    ("T1", "Ani", "Kapan X lahir?", "DATE", "3 Juli 1922", "DOC001", "4", "Pada 3 Juli 1922, X lahir."),
    ("T2", "Budi", "Siapa ayah Y?", "PERSON", "Kapitan Paulus", "DOC015", "2", "Ayah Y adalah Kapitan Paulus."),
    ("T3", "Cici", "Di mana Z?", "LOCATION", "Aceh", "DOC003", "4", "Z di Aceh."),
]


def write_test_csv(folder, rows=ROWS):
    path = Path(folder) / "qa_test.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TEST_FIELDS)
        writer.writerows(rows)
    return path


def result(answer, status="answered"):
    if status == "not_found":
        return {"status": status, "answer": None, "passage_id": None, "evidence": None, "note": None,
                "question": "q", "source": None, "passage": None}
    return {"status": status, "answer": answer, "passage_id": "DOC001_P04", "evidence": "e", "note": None,
            "question": "q", "source": "s", "passage": {}}


class AutoGradeTests(unittest.TestCase):
    def test_penilaian_usulan(self):
        cases = [
            ("3 Juli 1922", "3 Juli 1922", True),
            ("3 Juli 1922", "pada 3 juli 1922.", True),
            ("3 Juli 1922", "1922", False),  # jawaban parsial
            ("Mohammad Hatta", "Hatta", False),
            ("12 tahun", None, False),
            ("tidak ditemukan", None, True),
            ("tidak ditemukan", "1945", False),
        ]
        for truth, answer, expected in cases:
            with self.subTest(truth=truth, answer=answer):
                self.assertIs(auto_grade(truth, answer), expected)


class EvaluateAllTests(unittest.TestCase):
    def test_hanya_pertanyaan_dikirim_ke_sistem_dan_hasil_terisi(self):
        seen = []

        def classic(question):
            seen.append(("classic", question))
            return result("3 Juli 1922" if "lahir" in question else "salah")

        def rag(question):
            seen.append(("rag", question))
            return result("Kapitan Paulus", "needs_review") if "ayah" in question else result(None, "not_found")

        with tempfile.TemporaryDirectory() as folder, \
                patch("src.main.run_classic", classic), patch("src.main.run_rag", rag):
            rows = evaluate_all(write_test_csv(folder))
        self.assertEqual({q for _, q in seen}, {"Kapan X lahir?", "Siapa ayah Y?", "Di mana Z?"})
        self.assertEqual(len(seen), 6)
        self.assertEqual([r["question_id"] for r in rows], ["T1", "T2", "T3"])
        self.assertIs(rows[0]["classic_correct"], True)
        self.assertIs(rows[1]["classic_correct"], False)
        self.assertIs(rows[1]["rag_correct"], True)
        self.assertIsNone(rows[2]["rag_answer"])
        self.assertIs(rows[2]["rag_correct"], False)
        self.assertEqual(rows[1]["error_reason"], "rag: needs_review")
        self.assertEqual(rows[2]["error_reason"], "rag: not_found")

    def test_galat_satu_sistem_dicatat_dan_tidak_menghentikan_soal_lain(self):
        def broken(question):
            raise UnavailableError("belum ada")

        with tempfile.TemporaryDirectory() as folder, \
                patch("src.main.run_classic", broken), patch("src.main.run_rag", lambda q: result("Aceh")):
            rows = evaluate_all(write_test_csv(folder))
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[0]["classic_answer"])
        self.assertIsNone(rows[0]["classic_correct"])  # tidak dinilai, bukan otomatis salah
        self.assertIn("UnavailableError", rows[0]["error_reason"])
        self.assertIs(rows[2]["rag_correct"], True)

    def test_template_kosong_dan_id_berulang_ditolak(self):
        with tempfile.TemporaryDirectory() as folder:
            blank = write_test_csv(folder, [("TEST001", "", "", "", "", "", "", "")])
            with self.assertRaisesRegex(ValueError, "kosong"):
                evaluate_all(blank)
            dup = write_test_csv(folder, [ROWS[0], ROWS[0]])
            with self.assertRaisesRegex(ValueError, "berulang"):
                evaluate_all(dup)


class ResultsFileTests(unittest.TestCase):
    def sample(self):
        base = dict.fromkeys(RESULT_FIELDS)
        return [
            {**base, "question_id": "T1", "question": "q1", "ground_truth": "a", "classic_answer": "a",
             "classic_correct": True, "rag_answer": "b", "rag_correct": False},
            {**base, "question_id": "T2", "question": "q2", "ground_truth": "a", "classic_correct": None,
             "rag_correct": True, "error_reason": "classic: UnavailableError: x"},
        ]

    def test_tulis_baca_dan_rangkum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sub" / "hasil.csv"
            write_results(self.sample(), path)
            rows = read_results(path)
        self.assertEqual([r["classic_correct"] for r in rows], [True, None])
        self.assertEqual([r["rag_correct"] for r in rows], [False, True])
        s = summarize(rows)
        self.assertEqual(s["classic"], {"correct": 1, "wrong": 0, "ungraded": 1, "accuracy": 50.0})
        self.assertEqual(s["rag"], {"correct": 1, "wrong": 1, "ungraded": 0, "accuracy": 50.0})

    def test_perbaikan_manual_di_csv_dipakai_saat_merangkum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hasil.csv"
            write_results(self.sample(), path)
            path.write_text(path.read_text(encoding="utf-8-sig").replace("False", "TRUE"), encoding="utf-8-sig")
            self.assertEqual(summarize(read_results(path))["rag"]["correct"], 2)

    def test_nilai_tidak_dikenal_ditolak(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hasil.csv"
            write_results(self.sample(), path)
            path.write_text(path.read_text(encoding="utf-8-sig").replace("False", "mungkin"), encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "tidak dikenal"):
                read_results(path)

    def test_cli_tidak_menimpa_hasil_yang_sudah_ada(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "hasil.csv"
            out.write_text("penilaian manual", encoding="utf-8")
            code = main(["--test", str(write_test_csv(folder)), "--out", str(out)])
            self.assertEqual(code, 2)
            self.assertEqual(out.read_text(encoding="utf-8"), "penilaian manual")


class RetrievalHitTests(unittest.TestCase):
    def test_hit_dan_miss(self):
        def fake(question, k=3):
            return [{"passage_id": pid} for pid in (["DOC001_P04", "X"] if "lahir" in question else ["Y", "Z"])][:k]

        with tempfile.TemporaryDirectory() as folder:
            results = evaluate_retrieval(write_test_csv(folder, ROWS[:2]), fake)
        self.assertEqual([(r["hit"], r["rank"]) for r in results], [(True, 1), (False, None)])



class RetrievalCliTests(unittest.TestCase):
    def run_cli(self, loader):
        import contextlib
        import io
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as folder, patch("src.evaluate._load_retriever", loader), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--retrieval", str(write_test_csv(folder, ROWS[:2])), "--retriever", "both"])
        return code, out.getvalue(), err.getvalue()

    def test_both_menampilkan_dua_retriever(self):
        def loader(name):
            return lambda q, k=3: [{"passage_id": "DOC001_P04"}]
        code, out, _ = self.run_cli(loader)
        self.assertEqual(code, 0)
        self.assertIn("Hit@3 dense: 1/2", out)
        self.assertIn("Hit@3 tfidf: 1/2", out)

    def test_both_tetap_jalan_jika_satu_retriever_belum_ada(self):
        def loader(name):
            if name == "tfidf":
                raise UnavailableError("belum ada")
            return lambda q, k=3: [{"passage_id": "DOC001_P04"}]
        code, out, err = self.run_cli(loader)
        self.assertEqual(code, 0)
        self.assertIn("Hit@3 dense", out)
        self.assertNotIn("Hit@3 tfidf", out)
        self.assertIn("[tfidf] dilewati", err)

    def test_both_gagal_jika_tidak_ada_retriever(self):
        def loader(name):
            raise UnavailableError("belum ada")
        code, _, _ = self.run_cli(loader)
        self.assertEqual(code, 3)

    def test_beberapa_berkas_digabung(self):
        import contextlib
        import io
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.evaluate._load_retriever", lambda name: (lambda q, k=3: [{"passage_id": "DOC001_P04"}])), \
                contextlib.redirect_stdout(out):
            first = write_test_csv(folder, ROWS[:1])
            second = Path(folder) / "kedua.csv"
            second.write_text(first.read_text(encoding="utf-8-sig").replace("T1", "T9"), encoding="utf-8-sig")
            code = main(["--retrieval", str(first), str(second), "--retriever", "dense"])
        self.assertEqual(code, 0)
        self.assertIn("Hit@3 dense: 2/2", out.getvalue())


class DevModeTests(unittest.TestCase):
    def test_dev_memakai_kolom_dev_tanpa_author_dan_menulis_ke_berkas_dev(self):
        import contextlib
        import io
        header = ("question_id", "question", "answer_type", "ground_truth", "evidence_doc_id",
                  "evidence_paragraph", "evidence_text")
        with tempfile.TemporaryDirectory() as folder:
            dev = Path(folder) / "qa_dev.csv"
            with dev.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerow(("DEV001", "Di mana Z?", "LOCATION", "Aceh", "DOC003", "4", "Z di Aceh."))
            out = Path(folder) / "hasil_dev.csv"
            with patch("src.main.run_classic", lambda q: result("Aceh")), \
                    patch("src.main.run_rag", lambda q: result("Aceh")), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                code = main(["--dev", "--test", str(dev), "--out", str(out)])
            self.assertEqual(code, 0)
            rows = read_results(out)
        self.assertEqual([r["question_id"] for r in rows], ["DEV001"])
        self.assertIs(rows[0]["classic_correct"], True)
        self.assertIn("bukan evaluasi akhir", printed.getvalue())


class RetrievalDetailTests(unittest.TestCase):
    def run_detail(self, extra):
        import contextlib
        import io

        def retriever(question, k=3):
            return [{"passage_id": "X_P01", "text": "Teks pertama. Teks kedua.", "score": 0.5},
                    {"passage_id": "DOC001_P04", "text": "Bukti ada di sini.", "score": 0.4}][:k]

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, patch("src.evaluate._load_retriever", lambda name: retriever), \
                patch("src.evaluate._sentence_viewer", lambda name: (lambda q, t: t.split(". ")[0])), \
                contextlib.redirect_stdout(out):
            rows = [ROWS[0], ("T2", "Budi", "Siapa ayah Y?", "PERSON", "Kapitan", "DOC015", "2", "Ayah Y adalah Kapitan.")]
            code = main(["--retrieval", str(write_test_csv(folder, rows)), "--retriever", "dense", *extra])
        return code, out.getvalue()

    def test_detail_menandai_passage_bukti_dan_menampilkan_kalimat(self):
        code, out = self.run_detail(["--detail"])
        self.assertEqual(code, 0)
        self.assertIn("T1  Kapan X lahir?", out)
        self.assertIn("jawaban acuan : 3 Juli 1922", out)
        self.assertIn("2. DOC001_P04  skor=0.4000  BENAR", out)
        self.assertIn("ditemukan di peringkat 2", out)
        self.assertIn("TIDAK masuk top-3", out)
        self.assertIn("Teks pertama", out)
        self.assertIn("Hit@3 dense: 1/2", out)

    def test_hanya_gagal_menyembunyikan_soal_yang_berhasil(self):
        code, out = self.run_detail(["--hanya-gagal"])
        self.assertEqual(code, 0)
        self.assertNotIn("T1  Kapan X lahir?", out)
        self.assertIn("T2  Siapa ayah Y?", out)


if __name__ == "__main__":
    unittest.main()
