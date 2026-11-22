import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, call, patch

from src.contracts import QAResult, RetrievedPassage, UnavailableError
from src.main import _attach_passage, main, run_classic, run_rag


# Semua keluaran di bawah adalah fixture unit test untuk alur pemanggilan.
QUESTION = "Pertanyaan uji kontrak?"
INFO = {"answer_type": "FACT", "keywords": ["uji"], "relation_cues": []}
PASSAGE = {"passage_id": "fixture", "text": "Teks uji.", "source": "sumber uji", "score": 0.0}
ANSWER = {
    "status": "answered", "answer": "uji", "passage_id": "fixture",
    "evidence": "Teks uji.", "note": None,
}


class IntegrationContractTests(unittest.TestCase):
    def test_urutan_input_dan_hasil_kedua_pipeline(self):
        cases = [
            (run_classic, "src.retrieve_tfidf.retrieve_tfidf", "src.extract_answer.extract_answer",
             call.finish(QUESTION, INFO, [PASSAGE])),
            (run_rag, "src.retrieve_dense.retrieve_dense", "src.rag_generate.rag_answer",
             call.finish(QUESTION, [PASSAGE], INFO)),
        ]
        for run, retrieve_name, finish_name, last_call in cases:
            with self.subTest(mode=run.__name__):
                steps = Mock()
                steps.analyze.return_value = INFO
                steps.retrieve.return_value = [PASSAGE]
                steps.finish.return_value = ANSWER
                with (
                    patch("src.question_processing.analyze_question", steps.analyze),
                    patch(retrieve_name, steps.retrieve),
                    patch(finish_name, steps.finish),
                ):
                    result = run(QUESTION)
                self.assertEqual(steps.mock_calls, [
                    call.analyze(QUESTION), call.retrieve(QUESTION, k=3), last_call,
                ])
                self.assertEqual(result, {
                    **ANSWER, "question": QUESTION, "source": PASSAGE["source"], "passage": PASSAGE,
                })
                self.assertTrue(RetrievedPassage.__required_keys__ <= PASSAGE.keys())
                self.assertTrue(QAResult.__required_keys__ <= result.keys())

    def test_id_dan_kutipan_bukti_harus_sesuai(self):
        with self.assertRaisesRegex(ValueError, "ID bukti"):
            _attach_passage(QUESTION, ANSWER, [])
        with self.assertRaisesRegex(ValueError, "Kutipan"):
            _attach_passage(QUESTION, {**ANSWER, "evidence": "di luar teks"}, [PASSAGE])

    def test_hasil_modul_jawaban_kosong_atau_salah_tipe_ditolak(self):
        cases = [
            (None, "Hasil modul jawaban"),
            ({"status": "answered"}, "Hasil modul jawaban"),
            ({**ANSWER, "answer": 1945}, "teks jawaban"),
            ({**ANSWER, "evidence": 1945}, "Kutipan"),
        ]
        for result, message in cases:
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, message):
                _attach_passage(QUESTION, result, [PASSAGE])

    def test_passage_retrieval_tidak_lengkap_berhenti_sebelum_pemilihan_jawaban(self):
        cases = [
            (run_classic, "src.retrieve_tfidf.retrieve_tfidf", "src.extract_answer.extract_answer"),
            (run_rag, "src.retrieve_dense.retrieve_dense", "src.rag_generate.rag_answer"),
        ]
        passage = PASSAGE.copy()
        del passage["score"]
        for run, retrieve_name, finish_name in cases:
            with (
                self.subTest(mode=run.__name__),
                patch("src.question_processing.analyze_question", return_value=INFO),
                patch(retrieve_name, return_value=[passage]),
                patch(finish_name) as finish,
            ):
                with self.assertRaisesRegex(ValueError, "score"):
                    run(QUESTION)
                finish.assert_not_called()

    def test_not_found_dan_needs_review(self):
        empty = {"status": "not_found", "answer": None, "passage_id": None, "evidence": None, "note": None}
        self.assertEqual(_attach_passage(QUESTION, empty, []), {
            **empty, "question": QUESTION, "source": None, "passage": None,
        })
        with self.assertRaisesRegex(ValueError, "not_found"):
            _attach_passage(QUESTION, {**empty, "passage_id": "fixture"}, [PASSAGE])
        review = {**ANSWER, "status": "needs_review", "note": "Perlu tinjauan tipe."}
        result = _attach_passage(QUESTION, review, [PASSAGE])
        self.assertEqual(result, {
            **review, "question": QUESTION, "source": PASSAGE["source"], "passage": PASSAGE,
        })

    def test_cli_melaporkan_modul_atau_model_yang_belum_tersedia(self):
        for mode, retrieve_name in (("classic", "src.retrieve_tfidf.retrieve_tfidf"),
                                    ("rag", "src.retrieve_dense.retrieve_dense")):
            errors = (UnavailableError("modul uji belum tersedia"), FileNotFoundError("model uji hilang"))
            for error in errors:
                with self.subTest(mode=mode, error=type(error).__name__):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        patch("src.question_processing.analyze_question", return_value=INFO),
                        patch(retrieve_name, side_effect=error),
                        redirect_stdout(stdout), redirect_stderr(stderr),
                    ):
                        code = main(["--mode", mode, "--question", QUESTION])
                    self.assertEqual(code, 3)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertIn(str(error), stderr.getvalue())

    def test_pertanyaan_kosong_ditolak_sebelum_memanggil_anggota(self):
        for run in (run_classic, run_rag):
            with self.subTest(mode=run.__name__):
                with self.assertRaisesRegex(ValueError, "kosong"):
                    run("  ")
