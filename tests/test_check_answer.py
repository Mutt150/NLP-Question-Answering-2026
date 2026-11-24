"""Unit test untuk check_answer.py milik Orang 3."""

import unittest

from src.check_answer import check_answer_type


class CheckAnswerTypeTests(unittest.TestCase):
    def test_date_positive_and_negative(self):
        self.assertTrue(check_answer_type("3 Juli 1922", "DATE"))
        self.assertTrue(check_answer_type("16 Januari 1904", "DATE"))
        self.assertTrue(check_answer_type("1911", "DATE"))
        self.assertTrue(check_answer_type("24 – 25 Desember 1921", "DATE"))
        self.assertFalse(check_answer_type("Sutomo", "DATE"))
        self.assertFalse(check_answer_type("Yogyakarta", "DATE"))
        self.assertFalse(check_answer_type("", "DATE"))

    def test_number_positive_and_negative(self):
        self.assertTrue(check_answer_type("12 tahun", "NUMBER"))
        self.assertTrue(check_answer_type("3 anak perempuan", "NUMBER"))
        self.assertTrue(check_answer_type("3", "NUMBER"))
        self.assertFalse(check_answer_type("1945", "NUMBER"))
        self.assertFalse(check_answer_type("Hatta", "NUMBER"))
        self.assertFalse(check_answer_type("", "NUMBER"))

    def test_person_positive_and_negative(self):
        self.assertTrue(check_answer_type("Mohammad Hatta", "PERSON"))
        self.assertTrue(check_answer_type("Semaun", "PERSON"))
        self.assertTrue(check_answer_type("Kapitan Paulus Tiahahu", "PERSON"))
        self.assertFalse(check_answer_type("1922", "PERSON"))
        self.assertFalse(check_answer_type("tahun 1920", "PERSON"))
        self.assertFalse(check_answer_type("", "PERSON"))

    def test_location_positive_and_negative(self):
        self.assertTrue(check_answer_type("Yogyakarta", "LOCATION"))
        self.assertTrue(check_answer_type("Kampung Blauran", "LOCATION"))
        self.assertTrue(check_answer_type("di Sumedang", "LOCATION"))
        self.assertFalse(check_answer_type("1945", "LOCATION"))
        self.assertFalse(check_answer_type("", "LOCATION"))

    def test_organization_positive_and_negative(self):
        self.assertTrue(check_answer_type("Perhimpunan Indonesia", "ORGANIZATION"))
        self.assertTrue(check_answer_type("Partai Sosialis Indonesia", "ORGANIZATION"))
        self.assertTrue(check_answer_type("PKI", "ORGANIZATION"))
        self.assertFalse(check_answer_type("12345", "ORGANIZATION"))
        self.assertFalse(check_answer_type("", "ORGANIZATION"))

    def test_fact_and_other_returns_none(self):
        self.assertIsNone(check_answer_type("Wakil Redaktur", "FACT"))
        self.assertIsNone(check_answer_type("penyakit ari-ari", "FACT"))
        self.assertIsNone(check_answer_type("Sersan Mayor", "OTHER"))
        self.assertIsNone(check_answer_type("Perjanjian Bungaya", "OTHER"))

    def test_empty_string_returns_false(self):
        self.assertFalse(check_answer_type("   ", "PERSON"))
        self.assertFalse(check_answer_type("", "DATE"))


if __name__ == "__main__":
    unittest.main()
