"""Unit test untuk extract_answer.py milik Orang 3."""

import unittest

from src.contracts import QuestionInfo, RetrievedPassage
from src.extract_answer import extract_answer
from src.main import _attach_passage


class ExtractAnswerTests(unittest.TestCase):
    def setUp(self):
        self.dummy_passages: list[RetrievedPassage] = [
            {
                "passage_id": "P001",
                "text": "Ki Hadjar Dewantara dilahirkan pada 2 Mei 1889 di Yogyakarta.",
                "score": 0.85,
                "source": "DOC001",
            },
            {
                "passage_id": "P002",
                "text": "Ia wafat pada 26 April 1959 di Kota Yogyakarta dan dimakamkan di sana.",
                "score": 0.80,
                "source": "DOC001",
            },
            {
                "passage_id": "P003",
                "text": "Ki Hadjar mendirikan Perguruan Nasional Taman Siswa pada 3 Juli 1922.",
                "score": 0.75,
                "source": "DOC001",
            },
        ]

    def test_kapan_wafat_memilih_tanggal_wafat_bukan_lahir(self):
        question = "Kapan Ki Hadjar Dewantara wafat?"
        q_info: QuestionInfo = {
            "answer_type": "DATE",
            "keywords": ["ki", "hadjar", "dewantara", "wafat"],
            "relation_cues": ["wafat"],
        }
        res = extract_answer(question, q_info, self.dummy_passages)

        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["answer"], "26 April 1959")
        self.assertEqual(res["passage_id"], "P002")
        self.assertIn("26 April 1959", res["evidence"])
        self.assertIn(res["evidence"], self.dummy_passages[1]["text"])

    def test_bukti_di_passage_ketiga_tetap_ditemukan(self):
        question = "Kapan Ki Hadjar Dewantara mendirikan Taman Siswa?"
        q_info: QuestionInfo = {
            "answer_type": "DATE",
            "keywords": ["ki", "hadjar", "mendirikan", "taman", "siswa"],
            "relation_cues": ["diri"],
        }
        res = extract_answer(question, q_info, self.dummy_passages)

        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["answer"], "3 Juli 1922")
        self.assertEqual(res["passage_id"], "P003")
        self.assertIn(res["evidence"], self.dummy_passages[2]["text"])

    def test_nama_tokoh_di_pertanyaan_tidak_dipilih_sebagai_jawaban_person(self):
        question = "Siapa ayah Martha Christina Tiahahu?"
        q_info: QuestionInfo = {
            "answer_type": "PERSON",
            "keywords": ["ayah", "martha", "christina", "tiahahu"],
            "relation_cues": ["ayah"],
        }
        passages: list[RetrievedPassage] = [
            {
                "passage_id": "P010",
                "text": "Martha Christina Tiahahu adalah pahlawan nasional dari Maluku.",
                "score": 0.88,
                "source": "DOC015",
            },
            {
                "passage_id": "P011",
                "text": "Ia adalah putri dari Kapitan Paulus Tiahahu, seorang pemimpin Soa Uluputi.",
                "score": 0.82,
                "source": "DOC015",
            },
            {
                "passage_id": "P012",
                "text": "Martha berjuang melawan Belanda bersama rakyat.",
                "score": 0.70,
                "source": "DOC015",
            },
        ]
        res = extract_answer(question, q_info, passages)

        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["answer"], "Kapitan Paulus Tiahahu")
        self.assertEqual(res["passage_id"], "P011")
        self.assertIn(res["evidence"], passages[1]["text"])

    def test_tidak_ada_kandidat_bertipe_cocok_menghasilkan_not_found(self):
        question = "Kapan peristiwa itu terjadi?"
        q_info: QuestionInfo = {
            "answer_type": "DATE",
            "keywords": ["peristiwa", "terjadi"],
            "relation_cues": [],
        }
        passages: list[RetrievedPassage] = [
            {
                "passage_id": "P020",
                "text": "Peristiwa perlawanan berlangsung sengit di perbukitan.",
                "score": 0.75,
                "source": "DOC001",
            },
            {
                "passage_id": "P021",
                "text": "Pasukan bergerak tanpa mengenal rasa gentar.",
                "score": 0.70,
                "source": "DOC001",
            },
            {
                "passage_id": "P022",
                "text": "Masyarakat memberikan dukungan penuh pada pejuang.",
                "score": 0.65,
                "source": "DOC001",
            },
        ]
        res = extract_answer(question, q_info, passages)

        self.assertEqual(res["status"], "not_found")
        self.assertIsNone(res["answer"])
        self.assertIsNone(res["passage_id"])
        self.assertIsNone(res["evidence"])
        self.assertIsNotNone(res["note"])

    def test_dua_kandidat_sama_kuat_menghasilkan_needs_review(self):
        question = "Kapan kongres diadakan?"
        q_info: QuestionInfo = {
            "answer_type": "DATE",
            "keywords": ["kapan", "kongres"],
            "relation_cues": [],
        }
        passages: list[RetrievedPassage] = [
            {
                "passage_id": "P030",
                "text": "Kongres pertama diadakan pada 1928 dan kongres kedua pada 1930.",
                "score": 0.80,
                "source": "DOC001",
            },
            {
                "passage_id": "P031",
                "text": "Pertemuan kongres dihadiri oleh para pemuda.",
                "score": 0.70,
                "source": "DOC001",
            },
            {
                "passage_id": "P032",
                "text": "Semua anggota menyepakati keputusan kongres.",
                "score": 0.60,
                "source": "DOC001",
            },
        ]
        res = extract_answer(question, q_info, passages)

        self.assertIn(res["status"], ("needs_review", "answered"))
        self.assertIsNotNone(res["answer"])
        self.assertIsNotNone(res["evidence"])

    def test_passages_kosong_dan_kurang_dari_tiga_aman(self):
        q_info: QuestionInfo = {
            "answer_type": "DATE",
            "keywords": ["wafat"],
            "relation_cues": ["wafat"],
        }
        res_empty = extract_answer("Kapan wafat?", q_info, [])
        self.assertEqual(res_empty["status"], "not_found")
        self.assertIsNone(res_empty["answer"])
        self.assertIsNone(res_empty["passage_id"])
        self.assertIsNone(res_empty["evidence"])

        res_short = extract_answer("Kapan wafat?", q_info, self.dummy_passages[:2])
        self.assertEqual(res_short["status"], "not_found")
        self.assertIsNone(res_short["answer"])

    def test_evidence_selalu_substring_persis_dari_teks_passage(self):
        question = "Di mana Bung Tomo dilahirkan?"
        q_info: QuestionInfo = {
            "answer_type": "LOCATION",
            "keywords": ["bung", "tomo", "dilahirkan"],
            "relation_cues": ["lahir"],
        }
        passages: list[RetrievedPassage] = [
            {
                "passage_id": "DOC007_P02",
                "text": "Sutomo dilahirkan di Kampung Blauran, Surabaya pada 3 Oktober 1920.",
                "score": 0.90,
                "source": "DOC007",
            },
            {
                "passage_id": "DOC007_P03",
                "text": "Ia aktif di pergerakan kepanduan.",
                "score": 0.60,
                "source": "DOC007",
            },
            {
                "passage_id": "DOC007_P04",
                "text": "Perjuangannya membangkitkan semangat pemuda Surabaya.",
                "score": 0.50,
                "source": "DOC007",
            },
        ]
        res = extract_answer(question, q_info, passages)

        self.assertEqual(res["status"], "answered")
        target_p = next(p for p in passages if p["passage_id"] == res["passage_id"])
        self.assertIn(res["evidence"], target_p["text"])
        self.assertEqual(res["evidence"], target_p["text"])

    def test_hasil_memiliki_lima_kunci_wajib(self):
        q_info: QuestionInfo = {
            "answer_type": "DATE",
            "keywords": ["wafat"],
            "relation_cues": ["wafat"],
        }
        res = extract_answer("Kapan wafat?", q_info, self.dummy_passages)
        expected_keys = {"status", "answer", "passage_id", "evidence", "note"}
        self.assertEqual(set(res.keys()), expected_keys)

    def test_lolos_validasi_attach_passage_main(self):
        question = "Kapan Ki Hadjar Dewantara wafat?"
        q_info: QuestionInfo = {
            "answer_type": "DATE",
            "keywords": ["ki", "hadjar", "dewantara", "wafat"],
            "relation_cues": ["wafat"],
        }
        res = extract_answer(question, q_info, self.dummy_passages)
        attached = _attach_passage(question, res, self.dummy_passages)

        self.assertEqual(attached["question"], question)
        self.assertEqual(attached["status"], "answered")
        self.assertIsNotNone(attached["passage"])
        self.assertEqual(attached["passage"]["passage_id"], "P002")


if __name__ == "__main__":
    unittest.main()
