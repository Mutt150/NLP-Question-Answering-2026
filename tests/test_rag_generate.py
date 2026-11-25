import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.contracts import Answer, QuestionInfo, RetrievedPassage, UnavailableError
from src.main import _attach_passage, run_rag
from src.rag_generate import _load_dotenv, build_prompt, call_llm, parse_llm_response, rag_answer

PASSAGES: list[RetrievedPassage] = [
    {
        "passage_id": "DOC001_P04",
        "text": "Pada 3 Juli 1922, ia akhirnya mendirikan Perguruan Nasional Taman Siswa di Yogyakarta.",
        "score": 0.85,
        "source": "https://id.wikipedia.org/wiki/Ki_Hadjar_Dewantara",
    },
    {
        "passage_id": "DOC015_P02",
        "text": "Ia adalah putri dari Kapitan Paulus Tiahahu, seorang pemimpin dari Soa Uluputi.",
        "score": 0.42,
        "source": "https://id.wikipedia.org/wiki/Martha_Christina_Tiahahu",
    },
]

Q_DATE: QuestionInfo = {
    "answer_type": "DATE",
    "keywords": ["ki", "hadjar", "dewantara", "taman", "siswa"],
    "relation_cues": ["diri"],
}


class RagPromptAndConfigTests(unittest.TestCase):
    def test_build_prompt_memuat_pertanyaan_tipe_relasi_dan_semua_passage(self):
        prompt = build_prompt("Kapan Taman Siswa didirikan?", PASSAGES, Q_DATE)
        self.assertIn("Kapan Taman Siswa didirikan?", prompt)
        self.assertIn("DATE", prompt)
        self.assertIn("diri", prompt)
        for p in PASSAGES:
            self.assertIn(p["passage_id"], prompt)
            self.assertIn(p["text"], prompt)

    def test_call_llm_menolak_jika_konfigurasi_kosong(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_env = Path(tmpdir) / ".env"
            clean_env = {
                k: v for k, v in os.environ.items()
                if k not in ("RAG_PROVIDER", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "OLLAMA_MODEL")
            }
            with (
                patch.dict(os.environ, clean_env, clear=True),
                patch("src.rag_generate._ENV_PATH", missing_env),
            ):
                with self.assertRaises(UnavailableError):
                    call_llm("tes")

    def test_load_dotenv_membaca_file_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text('# komentar\nGEMINI_API_KEY="rahasia-123"\nRAG_MODEL=gemini-2.5-flash\n', encoding="utf-8")
            loaded = _load_dotenv(env_file)
            self.assertEqual(loaded["GEMINI_API_KEY"], "rahasia-123")
            self.assertEqual(loaded["RAG_MODEL"], "gemini-2.5-flash")

    def test_parse_llm_response_code_fence_dan_error(self):
        fenced = '```json\n{"found": true, "answer": "3 Juli 1922"}\n```'
        self.assertEqual(parse_llm_response(fenced)["answer"], "3 Juli 1922")
        with self.assertRaises(UnavailableError):
            parse_llm_response("bukan json sama sekali")


class RagAnswerTests(unittest.TestCase):
    def test_happy_path_answered_dan_lolos_kontrak(self):
        payload = json.dumps({
            "found": True,
            "answer": "3 Juli 1922",
            "passage_id": "DOC001_P04",
            "evidence": "Pada 3 Juli 1922, ia akhirnya mendirikan Perguruan Nasional Taman Siswa di Yogyakarta.",
            "reason": None,
        })
        result = rag_answer(
            "Kapan Ki Hadjar Dewantara mendirikan Taman Siswa?",
            PASSAGES,
            Q_DATE,
            llm_caller=lambda _prompt: payload,
        )
        self.assertTrue(Answer.__required_keys__ <= result.keys())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "3 Juli 1922")
        self.assertEqual(result["passage_id"], "DOC001_P04")
        self.assertIsNone(result["note"])
        attached = _attach_passage("Kapan Ki Hadjar Dewantara mendirikan Taman Siswa?", result, PASSAGES)
        self.assertEqual(attached["source"], PASSAGES[0]["source"])

    def test_passage_kosong_mengembalikan_not_found_tanpa_memanggil_llm(self):
        called = False

        def fail_if_called(_prompt: str) -> str:
            nonlocal called
            called = True
            return "{}"

        result = rag_answer("Siapa ayah Kartini?", [], Q_DATE, llm_caller=fail_if_called)
        self.assertFalse(called)
        self.assertEqual(result["status"], "not_found")
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["passage_id"])
        self.assertIsNone(result["evidence"])
        attached = _attach_passage("Siapa ayah Kartini?", result, [])
        self.assertIsNone(attached["passage"])

    def test_llm_menyatakan_tidak_ditemukan(self):
        for body in (
            json.dumps({"found": False, "answer": None, "passage_id": None, "evidence": None, "reason": "Tidak disebut."}),
            json.dumps({"found": True, "answer": "Jawaban tidak ditemukan", "passage_id": "DOC001_P04", "evidence": "Pada 3 Juli 1922"}),
        ):
            with self.subTest(body=body):
                result = rag_answer("Di mana ia wafat?", PASSAGES, Q_DATE, llm_caller=lambda _p, b=body: b)
                self.assertEqual(result["status"], "not_found")
                self.assertIsNone(result["answer"])
                self.assertIsNone(result["passage_id"])
                self.assertIsNone(result["evidence"])
                _attach_passage("Di mana ia wafat?", result, PASSAGES)

    def test_halusinasi_passage_atau_kutipan_ditolak_aman(self):
        bad_id = json.dumps({
            "found": True,
            "answer": "3 Juli 1922",
            "passage_id": "DOC999_P99",
            "evidence": "Kalimat karangan bebas.",
        })
        result = rag_answer("Kapan?", PASSAGES, Q_DATE, llm_caller=lambda _p: bad_id)
        self.assertEqual(result["status"], "not_found")
        _attach_passage("Kapan?", result, PASSAGES)

    def test_kutipan_tidak_persis_tetapi_jawaban_ada_di_kalimat_jadi_needs_review(self):
        paraphrased_quote = json.dumps({
            "found": True,
            "answer": "3 Juli 1922",
            "passage_id": "DOC001_P04",
            "evidence": "Didirikan pada 3 Juli 1922 oleh beliau.",
        })
        result = rag_answer("Kapan?", PASSAGES, Q_DATE, llm_caller=lambda _p: paraphrased_quote)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["evidence"], PASSAGES[0]["text"])
        self.assertIn("ditinjau", result["note"] or "")
        _attach_passage("Kapan?", result, PASSAGES)

    def test_ketidaksesuaian_tipe_menjadi_needs_review(self):
        wrong_type_payload = json.dumps({
            "found": True,
            "answer": "Yogyakarta",
            "passage_id": "DOC001_P04",
            "evidence": PASSAGES[0]["text"],
        })
        result = rag_answer("Kapan?", PASSAGES, Q_DATE, llm_caller=lambda _p: wrong_type_payload)
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("DATE", result["note"] or "")

        # Simulasikan modul Orang 3 sudah diimplementasikan dan mengembalikan False / None
        for verdict in (False, None):
            with (
                self.subTest(verdict=verdict),
                patch("src.check_answer.check_answer_type", return_value=verdict),
            ):
                checked = rag_answer("Kapan?", PASSAGES, Q_DATE, llm_caller=lambda _p: wrong_type_payload)
                self.assertEqual(checked["status"], "needs_review")

    def test_kegagalan_llm_meneruskan_unavailable_error(self):
        def broken_llm(_prompt: str) -> str:
            raise UnavailableError("API timeout")

        with self.assertRaises(UnavailableError):
            rag_answer("Kapan?", PASSAGES, Q_DATE, llm_caller=broken_llm)

    def test_integrasi_end_to_end_run_rag(self):
        payload = json.dumps({
            "found": True,
            "answer": "Kapitan Paulus Tiahahu",
            "passage_id": "DOC015_P02",
            "evidence": "Ia adalah putri dari Kapitan Paulus Tiahahu, seorang pemimpin dari Soa Uluputi.",
        })
        with (
            patch("src.retrieve_dense.retrieve_dense", return_value=PASSAGES),
            patch("src.rag_generate.call_llm", return_value=payload),
        ):
            result = run_rag("Siapa ayah Martha Christina Tiahahu?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Kapitan Paulus Tiahahu")
        self.assertEqual(result["passage_id"], "DOC015_P02")
        self.assertEqual(result["source"], PASSAGES[1]["source"])
