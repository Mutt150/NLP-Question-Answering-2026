import csv
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import get_args
from urllib.parse import parse_qs, urlsplit

from src.contracts import AnswerType
from src.prepare_data import CORPUS_FIELDS, DATA_DIR, DEV_FIELDS, load_corpus, prepare_data


SOURCE_CSVS = ("sources.csv", "qa_dev.csv", "qa_test_template.csv")


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames), list(reader)


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class DatasetTests(unittest.TestCase):
    def test_checksum_15_txt_dan_tiga_csv_serta_metadata(self):
        names = [f"DOC{number:03d}.txt" for number in range(1, 16)]
        self.assertEqual(sorted(p.name for p in (DATA_DIR / "corpus").glob("*.txt")), names)
        # SHA256SUMS mencatat sidik jari berkas untuk versi dataset yang disepakati.
        expected_files = [f"corpus/{name}" for name in names] + list(SOURCE_CSVS)
        lines = (DATA_DIR / "SHA256SUMS").read_text(encoding="ascii").splitlines()
        checksums = [line.split("  ", 1) for line in lines]
        self.assertCountEqual([name for checksum, name in checksums], expected_files)
        for checksum, relative in checksums:
            with self.subTest(file=relative):
                actual = hashlib.sha256((DATA_DIR / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, checksum)

        _, sources = read_table(DATA_DIR / "sources.csv")
        self.assertEqual(sorted(row["filename"] for row in sources), names)
        self.assertEqual(len({row["doc_id"] for row in sources}), 15)
        for row in sources:
            self.assertEqual(row["filename"], row["doc_id"] + ".txt")
            self.assertTrue(row["title"] and row["license"])
            query = parse_qs(urlsplit(row["source_url"]).query)
            self.assertTrue(query.get("oldid", [""])[0].isdigit())

    def test_75_passage_utuh_dengan_id_dan_sumber_yang_tepat(self):
        fields, exported = read_table(DATA_DIR / "processed" / "passages.csv")
        self.assertEqual(fields, CORPUS_FIELDS)
        self.assertEqual(load_corpus(), exported)
        self.assertEqual(len(exported), 75)
        by_id = {p["passage_id"]: p for p in exported}
        self.assertEqual(len(by_id), 75)
        # Paket sumber memakai satu baris kosong LF dan satu LF penutup per file.
        expected_ids = []
        for source in read_table(DATA_DIR / "sources.csv")[1]:
            raw = (DATA_DIR / "corpus" / source["filename"]).read_bytes().decode("utf-8")
            paragraphs = raw.removesuffix("\n").split("\n\n")
            for number, original in enumerate(paragraphs, start=1):
                passage_id = f"{source['doc_id']}_P{number:02d}"
                expected_ids.append(passage_id)
                self.assertTrue(original.strip())
                self.assertEqual(by_id[passage_id], {
                    "passage_id": passage_id, "doc_id": source["doc_id"], "title": source["title"],
                    "text": original, "source": source["source_url"],
                })
        self.assertCountEqual(by_id.keys(), expected_ids)

    def test_25_soal_dev_dan_kutipan_pada_passage_asli(self):
        _, originals = read_table(DATA_DIR / "qa_dev.csv")
        fields, linked = read_table(DATA_DIR / "processed" / "qa_dev.csv")
        self.assertEqual(fields, (*DEV_FIELDS, "evidence_passage_id"))
        expected_ids = [f"DEV{number:03d}" for number in range(1, 26)]
        self.assertEqual([row["question_id"] for row in linked], expected_ids)
        passages = {p["passage_id"]: p["text"] for p in load_corpus()}
        for row, original in zip(linked, originals, strict=True):
            passage_id = f"{original['evidence_doc_id']}_P{int(original['evidence_paragraph']):02d}"
            self.assertEqual(row, {**original, "evidence_passage_id": passage_id})
            self.assertIn(row["answer_type"], get_args(AnswerType))
            self.assertIn(row["evidence_text"], passages[passage_id])
            self.assertIn(row["ground_truth"], row["evidence_text"])

    def test_template_20_id_dengan_semua_kolom_lain_kosong(self):
        fields, rows = read_table(DATA_DIR / "qa_test_template.csv")
        self.assertEqual(fields, ("question_id", "author", *DEV_FIELDS[1:]))
        self.assertEqual([r["question_id"] for r in rows], [f"TEST{i:03d}" for i in range(1, 21)])
        for row in rows:
            self.assertTrue(all(row[field] == "" for field in fields if field != "question_id"))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data = Path(folder.name) / "data"
        shutil.copytree(DATA_DIR / "corpus", self.data / "corpus")
        for name in SOURCE_CSVS:
            shutil.copyfile(DATA_DIR / name, self.data / name)
        self.output = self.data / "processed"

    def test_regenerasi_dua_kali_dan_template_tetap_utuh(self):
        # Penyiapan berjalan di folder sementara supaya data proyek tetap utuh.
        for _ in range(2):
            self.assertEqual(prepare_data(self.data), (75, 25))
            for name in ("passages.csv", "qa_dev.csv"):
                self.assertEqual((self.output / name).read_bytes(), (DATA_DIR / "processed" / name).read_bytes())
        for name in SOURCE_CSVS:
            self.assertEqual((self.data / name).read_bytes(), (DATA_DIR / name).read_bytes())

    def test_metadata_ganda_file_hilang_dan_filename_keliru_ditolak(self):
        path = self.data / "sources.csv"
        _, sources = read_table(path)
        cases = [
            (sources + [sources[0]], "berulang"),
            (sources[1:], "Daftar TXT"),
            ([{**sources[0], "filename": "keliru.txt"}, *sources[1:]], "filename"),
        ]
        for rows, message in cases:
            with self.subTest(kasus=message):
                write_table(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    prepare_data(self.data)
                self.assertFalse(self.output.exists())

    def test_soal_dev_salah_ditolak_dan_hasil_sebelumnya_tetap_utuh(self):
        prepare_data(self.data)
        path = self.data / "qa_dev.csv"
        _, questions = read_table(path)
        cases = [
            ("evidence_paragraph", "0"), ("evidence_paragraph", "99"), ("evidence_paragraph", "empat"),
            ("evidence_doc_id", "DOC999"), ("evidence_text", "Kutipan di luar paragraf."),
            ("ground_truth", "Di luar kutipan."),
        ]
        for field, value in cases:
            with self.subTest(kolom=field, nilai=value):
                write_table(path, [{**questions[0], field: value}, *questions[1:]])
                with self.assertRaisesRegex(ValueError, "DEV001"):
                    prepare_data(self.data)
                for name in ("passages.csv", "qa_dev.csv"):
                    self.assertEqual((self.output / name).read_bytes(), (DATA_DIR / "processed" / name).read_bytes())

    def test_dokumen_kosong_ditolak(self):
        (self.data / "corpus" / "DOC001.txt").write_bytes(b" \r\n\t\n")
        with self.assertRaisesRegex(ValueError, "DOC001.txt"):
            prepare_data(self.data)

    def test_loader_menolak_id_passage_ganda(self):
        row = load_corpus()[0]
        path = self.data / "ganda.csv"
        write_table(path, [row, row])
        with self.assertRaisesRegex(ValueError, "passage_id berulang"):
            load_corpus(path)

    def test_gagal_menerbitkan_csv_kedua_memulihkan_hasil_pertama(self):
        self.output.mkdir()
        previous = self.output / "passages.csv"
        previous.write_bytes(b"hasil sebelumnya")
        (self.output / "qa_dev.csv").mkdir()  # Path ini tidak dapat diganti dengan file.
        with self.assertRaises(OSError):
            prepare_data(self.data)
        self.assertEqual(previous.read_bytes(), b"hasil sebelumnya")
        self.assertTrue((self.output / "qa_dev.csv").is_dir())
