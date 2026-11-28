import unittest

from src.extract_answer import extract_answer


class TestExtractAnswer(unittest.TestCase):
    def test_disambiguate_birth_vs_death_date(self) -> None:
        passage = [
            {
                "passage_id": "DOC001_P01",
                "title": "Pangeran Diponegoro",
                "text": "Pangeran Diponegoro (Yogyakarta, 11 November 1785 – Makassar, 8 Januari 1855) adalah pahlawan nasional.",
                "score": 0.9,
                "source": "test",
            }
        ]
        q_info = {
            "answer_type": "DATE",
            "keywords": ["pangeran", "diponegoro", "wafat"],
            "relation_cues": ["wafat"],
        }
        res = extract_answer("Kapan Pangeran Diponegoro wafat?", q_info, passage)
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["answer"], "8 Januari 1855")

    def test_avoid_self_echo_location(self) -> None:
        passage = [
            {
                "passage_id": "DOC014_P06",
                "title": "Hasanuddin dari Gowa",
                "text": "Kompeni berhasil menguasai benteng terkuat Gowa yaitu Benteng Sombaopu pada tanggal 24 Juni 1669.",
                "score": 0.85,
                "source": "test",
            }
        ]
        q_info = {
            "answer_type": "LOCATION",
            "keywords": ["benteng", "sombaopu"],
            "relation_cues": [],
        }
        res = extract_answer("Di mana Benteng Sombaopu?", q_info, passage)
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["answer"], "Gowa")

    def test_reject_wrong_subject_passage(self) -> None:
        passage = [
            {
                "passage_id": "DOC009_P03",
                "title": "Dewi Sartika",
                "text": "Sekolah tersebut kemudian direlokasi ke Jalan Ciguriang dan berubah nama menjadi Sakola Kaoetamaan Istri pada tahun 1910.",
                "score": 0.7,
                "source": "test",
            }
        ]
        q_info = {
            "answer_type": "PERSON",
            "keywords": ["istri", "mohammad", "hatta"],
            "relation_cues": ["istri"],
        }
        res = extract_answer("Siapa istri Mohammad Hatta?", q_info, passage)
        self.assertEqual(res["status"], "not_found")
        self.assertIsNone(res["answer"])

    def test_prefer_main_entity_over_parenthetical(self) -> None:
        passage = [
            {
                "passage_id": "DOC012_P03",
                "title": "Tan Malaka",
                "text": "Pada tahun 1908, Tan Malaka bersekolah di Kweekschool (kini SMA Negeri 2 Bukittinggi), sekolah guru negeri, di Fort de Kock.",
                "score": 0.9,
                "source": "test",
            }
        ]
        q_info = {
            "answer_type": "ORGANIZATION",
            "keywords": ["sekolah", "tan", "malaka", "1908"],
            "relation_cues": [],
        }
        res = extract_answer(
            "Sekolah apa yang dimasuki Tan Malaka pada tahun 1908?", q_info, passage
        )
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["answer"], "Kweekschool")

    def test_fact_boundary_stops_before_preposition(self) -> None:
        passage = [
            {
                "passage_id": "DOC014_P06",
                "title": "Hasanuddin dari Gowa",
                "text": "Sultan Hasanuddin menandatangani Perjanjian Bungaya pada tanggal 18 November 1667 di Bungaya.",
                "score": 0.9,
                "source": "test",
            }
        ]
        q_info = {
            "answer_type": "FACT",
            "keywords": ["perjanjian", "sultan", "hasanuddin"],
            "relation_cues": [],
        }
        res = extract_answer(
            "Perjanjian apa yang ditandatangani Sultan Hasanuddin pada 18 November 1667?",
            q_info,
            passage,
        )
        self.assertEqual(res["answer"], "Perjanjian Bungaya")


if __name__ == "__main__":
    unittest.main()
