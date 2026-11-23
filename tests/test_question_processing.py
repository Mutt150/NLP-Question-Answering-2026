import csv
import unittest
from typing import get_args

from src.contracts import AnswerType
from src.prepare_data import DATA_DIR
from src.question_processing import analyze_question, detect_answer_type, extract_keywords, find_relation_cues


def load_dev():
    with (DATA_DIR / "processed" / "qa_dev.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class AnswerTypeTests(unittest.TestCase):
    def assert_types(self, cases):
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(detect_answer_type(question), expected)

    def test_soal_dev_sesuai_kunci(self):
        self.assert_types((row["question"], row["answer_type"]) for row in load_dev())

    def test_partikel_kah(self):
        self.assert_types([
            ("Kapankah Pangeran Diponegoro ditangkap?", "DATE"),
            ("Siapakah ayah Kartini?", "PERSON"),
            ("Di manakah Soekarno dilahirkan?", "LOCATION"),
            ("Berapakah usia Kartini saat menikah?", "NUMBER"),
            ("Partai apakah yang didirikan Sjahrir?", "ORGANIZATION"),
        ])

    def test_urutan_aturan(self):
        self.assert_types([
            ("Tahun berapa Hatta lahir?", "DATE"),
            ("Pada tanggal berapa Tan Malaka wafat?", "DATE"),
            ("Berapa tahun Cut Nyak Dhien diasingkan?", "NUMBER"),
        ])

    def test_pola_apa_nama(self):
        self.assert_types([
            ("Apa nama organisasi yang dipimpin Mohammad Hatta di Belanda?", "ORGANIZATION"),
            ("Apa nama kampung tempat Bung Tomo dilahirkan?", "LOCATION"),
            ("Apa peran Partai Sosialis Indonesia bagi Sjahrir?", "FACT"),
            ("Apa nama gelar yang diberikan kepada Diponegoro?", "FACT"),
        ])

    def test_lokasi_dengan_dan_tanpa_spasi(self):
        self.assert_types([
            ("Di mana Hatta lahir?", "LOCATION"),
            ("Dimana Hatta lahir?", "LOCATION"),
            ("Mohammad Hatta lahir di mana?", "LOCATION"),
        ])

    def test_pertanyaan_non_factoid(self):
        self.assert_types([
            ("Mengapa Diponegoro melawan Belanda?", "OTHER"),
            ("Bagaimana Kartini memperjuangkan pendidikan?", "OTHER"),
            ("Apakah Sjahrir menjadi perdana menteri?", "OTHER"),
        ])


class KnownLimitationTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_kalimat_perintah(self):
        self.assertEqual(detect_answer_type("Sebutkan nama ayah Kartini"), "PERSON")

    @unittest.expectedFailure
    def test_tanpa_kata_tanya(self):
        self.assertEqual(detect_answer_type("Bung Tomo wafat tahun?"), "DATE")

    @unittest.expectedFailure
    def test_apakah_nama_di_awal(self):
        self.assertEqual(detect_answer_type("Apakah nama partai yang didirikan Sjahrir?"), "ORGANIZATION")

    @unittest.expectedFailure
    def test_apa_nama_dengan_kata_bernama(self):
        self.assertEqual(detect_answer_type("Organisasi yang dipimpin Hatta bernama apa?"), "ORGANIZATION")


class KeywordTests(unittest.TestCase):
    def test_kata_tanya_dan_stopword_dibuang(self):
        keywords = extract_keywords("Di mana Cut Nyak Dhien dilahirkan?")
        self.assertEqual(keywords, ["cut", "nyak", "dhien", "dilahirkan"])

    def test_angka_tetap_ada(self):
        self.assertIn("1926", extract_keywords("Organisasi apa yang dipimpin Hatta pada 1926?"))

    def test_tanpa_duplikat_dan_urutan_terjaga(self):
        keywords = extract_keywords("Hatta dan Soekarno, siapa yang lebih tua, Hatta atau Soekarno?")
        self.assertEqual(keywords, ["hatta", "soekarno", "lebih", "tua"])


class RelationCueTests(unittest.TestCase):
    def assert_cues(self, cases):
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(find_relation_cues(question), expected)

    def test_bentuk_berimbuhan(self):
        self.assert_cues([
            ("Di mana Cut Nyak Dhien dilahirkan?", ["lahir"]),
            ("Kapan Ki Hadjar Dewantara mendirikan Taman Siswa?", ["diri"]),
            ("Di mana Diponegoro diasingkan?", ["asing"]),
            ("Di mana Kartini dimakamkan?", ["makam"]),
        ])

    def test_bentuk_luluh(self):
        self.assert_cues([
            ("Siapa yang memimpin Perang Padri?", ["pimpin"]),
            ("Siapa yang menangkap Pattimura?", ["tangkap"]),
        ])

    def test_sinonim_digabung(self):
        self.assert_cues([
            ("Kapan Ki Hadjar Dewantara wafat?", ["wafat"]),
            ("Kapan Tan Malaka meninggal?", ["wafat"]),
        ])

    def test_kata_mirip_tidak_ikut(self):
        self.assert_cues([
            ("Di mana ibu kota Kerajaan Gowa?", []),
            ("Bahasa asing apa yang dikuasai Agus Salim?", []),
            ("Sampai usia berapa Kartini diperbolehkan bersekolah di ELS?", []),
        ])


class AnalyzeQuestionTests(unittest.TestCase):
    def test_format_sesuai_kontrak(self):
        for row in load_dev():
            with self.subTest(question_id=row["question_id"]):
                info = analyze_question(row["question"])
                self.assertEqual(set(info), {"answer_type", "keywords", "relation_cues"})
                self.assertIn(info["answer_type"], get_args(AnswerType))
                self.assertTrue(all(isinstance(word, str) for word in info["keywords"]))
                self.assertTrue(all(isinstance(cue, str) for cue in info["relation_cues"]))

    def test_contoh_lengkap(self):
        self.assertEqual(analyze_question("Siapa ayah Martha Christina Tiahahu?"), {
            "answer_type": "PERSON",
            "keywords": ["ayah", "martha", "christina", "tiahahu"],
            "relation_cues": ["ayah"],
        })


if __name__ == "__main__":
    unittest.main()
