import csv
import tempfile
import unittest
from pathlib import Path

from src.merge_questions import PART_FIELDS, merge_questions
from src.prepare_data import load_corpus

PASSAGES = load_corpus()
BASE = {  # bukti nyata dari korpus; hanya pertanyaannya yang dibuat berbeda
    "a1": ("Q A1 kapan Taman Siswa didirikan?", "DATE", "3 Juli 1922", "DOC001", "4",
           "Pada 3 Juli 1922, ia akhirnya mendirikan Perguruan Nasional Taman Siswa di Yogyakarta."),
    "a2": ("Q A2 sampai usia berapa Kartini sekolah?", "NUMBER", "12 tahun", "DOC002", "2",
           "Sampai usia 12 tahun, Kartini diperbolehkan bersekolah di Europeesche Lagere School (ELS)."),
    "b1": ("Q B1 di mana Cut Nyak Dhien lahir?", "LOCATION", "Lampadang", "DOC003", "4",
           "Cut Nyak Dhien dilahirkan di Lampadang, wilayah VI Mukim, Aceh Besar."),
    "b2": ("Q B2 siapa ayah Martha?", "PERSON", "Kapitan Paulus Tiahahu", "DOC015", "2",
           "Ia adalah putri dari Kapitan Paulus Tiahahu, seorang pemimpin dari Soa Uluputi."),
}


def write_part(folder, name, keys, edit=None):
    with (Path(folder) / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PART_FIELDS)
        for key in keys:
            row = list(BASE[key])
            if edit and key in edit:
                row[edit[key][0]] = edit[key][1]
            writer.writerow([name, *row])


def merge(folder, **kwargs):
    return merge_questions(folder, PASSAGES, per_author=2, total=4, **kwargs)


class MergeQuestionsTests(unittest.TestCase):
    def test_gabungan_valid_diberi_id_berurutan(self):
        with tempfile.TemporaryDirectory() as folder:
            write_part(folder, "ani", ["a1", "a2"])
            write_part(folder, "budi", ["b1", "b2"])
            rows = merge(folder)
        self.assertEqual([r["question_id"] for r in rows], ["TEST001", "TEST002", "TEST003", "TEST004"])
        self.assertEqual([r["author"] for r in rows], ["ani", "ani", "budi", "budi"])

    def test_semua_masalah_dilaporkan_sekaligus(self):
        with tempfile.TemporaryDirectory() as folder:
            write_part(folder, "ani", ["a1", "a2"], edit={"a1": (5, "kutipan yang tidak ada di paragraf")})
            write_part(folder, "budi", ["b1"], edit={"b1": (1, "KOTAK")})
            with self.assertRaises(ValueError) as ctx:
                merge(folder)
        message = str(ctx.exception)
        for part in ("evidence_text tidak ditemukan", "answer_type", "budi punya 1 soal", "seharusnya 4"):
            self.assertIn(part, message)

    def test_soal_ganda_dan_salinan_dev_ditolak(self):
        with tempfile.TemporaryDirectory() as folder:
            write_part(folder, "ani", ["a1", "a2"])
            write_part(folder, "budi", ["b1", "a1"], edit={"a1": (0, "Q A1 kapan Taman Siswa didirikan?")})
            with self.assertRaisesRegex(ValueError, "pertanyaan sama"):
                merge(folder)
        with tempfile.TemporaryDirectory() as folder:
            write_part(folder, "ani", ["a1", "a2"])
            write_part(folder, "budi", ["b1", "b2"])
            with self.assertRaisesRegex(ValueError, "soal DEV"):
                merge(folder, dev_questions={"q b1 di mana cut nyak dhien lahir?"})

    def test_jawaban_di_luar_kutipan_dan_kolom_kosong_ditolak(self):
        with tempfile.TemporaryDirectory() as folder:
            write_part(folder, "ani", ["a1", "a2"], edit={"a2": (2, "13 tahun")})
            write_part(folder, "budi", ["b1", "b2"], edit={"b2": (0, "")})
            with self.assertRaises(ValueError) as ctx:
                merge(folder)
        self.assertIn("ground_truth tidak termuat", str(ctx.exception))
        self.assertIn("kolom kosong", str(ctx.exception))

    def test_folder_kosong(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "Tidak ada berkas"):
            merge(folder)


if __name__ == "__main__":
    unittest.main()
