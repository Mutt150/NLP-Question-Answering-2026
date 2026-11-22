import unittest

from src.preprocessing import preprocess, split_paragraphs


class PreprocessingTests(unittest.TestCase):
    def test_spasi_dirapikan_dan_fakta_tetap_utuh(self):
        raw = "\tNama (né\u00a0Athar),  angka 1926\n\ndan kata tidak. [1] "
        expected = "Nama (né Athar), angka 1926 dan kata tidak. [1]"
        self.assertEqual(preprocess(raw), expected)
        self.assertEqual(preprocess(expected), expected)

    def test_teks_kosong(self):
        self.assertEqual(preprocess(""), "")
        self.assertEqual(preprocess(" \r\n\t"), "")
        self.assertEqual(split_paragraphs(" \r\n\t"), [])

    def test_batas_paragraf_dan_teks_multibaris(self):
        first = "  R. A. Kartini  \r\nBaris berikutnya.  "
        second = "Mohammad Hatta (né Athar)\t"
        document = "\r\n" + first + "\r\n \t\r\n\r\n" + second
        self.assertEqual(split_paragraphs(document), [first, second])
        self.assertEqual(split_paragraphs(document + "\r\n"), [first, second])
