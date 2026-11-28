import unittest

from src.check_answer import check_answer_type


class TestCheckAnswer(unittest.TestCase):
    def test_date_valid_and_invalid(self) -> None:
        self.assertTrue(check_answer_type("17 Agustus 1945", "DATE"))
        self.assertTrue(check_answer_type("November 1825", "DATE"))
        self.assertFalse(check_answer_type("Yogyakarta", "DATE"))

    def test_number_distinguishes_date(self) -> None:
        self.assertTrue(check_answer_type("9 sekolah", "NUMBER"))
        self.assertTrue(check_answer_type("empat anak", "NUMBER"))
        # Tanggal kalender tanpa satuan kuantitas tidak boleh lolos sebagai NUMBER
        self.assertFalse(check_answer_type("16 Januari 1904", "NUMBER"))

    def test_person_rejects_org_and_place(self) -> None:
        self.assertTrue(check_answer_type("Pangeran Diponegoro", "PERSON"))
        self.assertTrue(check_answer_type("Raden Ajeng Kartini", "PERSON"))
        self.assertFalse(check_answer_type("Sakola Kaoetamaan Istri", "PERSON"))
        self.assertFalse(check_answer_type("Yogyakarta", "PERSON"))

    def test_location_and_organization(self) -> None:
        self.assertTrue(check_answer_type("Gowa", "LOCATION"))
        self.assertTrue(check_answer_type("Fort de Kock", "LOCATION"))
        self.assertFalse(check_answer_type("Sultan Hasanuddin", "LOCATION"))

        self.assertTrue(check_answer_type("SMA Negeri 2 Bukittinggi", "ORGANIZATION"))
        self.assertTrue(check_answer_type("BPUPKI", "ORGANIZATION"))
        self.assertFalse(check_answer_type("1908", "ORGANIZATION"))


if __name__ == "__main__":
    unittest.main()
