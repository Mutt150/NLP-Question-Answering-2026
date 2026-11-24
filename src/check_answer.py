"""Pemeriksaan tipe untuk Sistem 1 dan Sistem 2 (RAG)."""

import re

from .contracts import AnswerType

# Pola regex untuk validasi tipe
_MONTH_PATTERN = (
    r"(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember|"
    r"januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)"
)

# Tanggal: memuat angka tahun 4 digit, tanggal lengkap, atau bulan + tahun
_DATE_RE = re.compile(
    rf"^(?:(?:\d{{1,2}}\s+)?{_MONTH_PATTERN}(?:\s+\d{{4}})?|\d{{1,2}}\s+[–\-]\s+\d{{1,2}}\s+{_MONTH_PATTERN}\s+\d{{4}}|\b(?:1[6-9]\d{{2}}|20\d{{2}})\b)$",
    re.IGNORECASE,
)

# Number: digit murni atau digit + kata satuan/nomina
_NUMBER_RE = re.compile(
    r"^\d+(?:\s+(?:tahun|anak(?: perempuan| laki-laki)?|orang|kali|hari|bulan|minggu|jiwa|ribu|juta|persen|%))?$",
    re.IGNORECASE,
)

# Location: nama berawalan huruf kapital, dapat didahului preposisi/penanda lokasi
_LOCATION_INDICATORS = (
    r"(?:Kampung|Desa|Kota|Wilayah|Karesidenan|Provinsi|Pulau|Kabupaten|Gua|Kecamatan|Taman Makam)"
)

_ORG_INDICATORS = (
    r"(?:Partai|Perhimpunan|Sakola|Sekolah|Kweekschool|Sarekat|Lembaga|Organisasi|Harian|Komite|Yayasan|Perguruan)"
)


def check_answer_type(answer: str, answer_type: AnswerType) -> bool | None:
    """Periksa apakah format jawaban sesuai dengan answer_type."""
    if not isinstance(answer, str) or not answer.strip():
        return False

    text = answer.strip()

    if answer_type in ("FACT", "OTHER"):
        return None

    if answer_type == "DATE":
        # Cocok jika berupa tanggal lengkap, bulan+tahun, atau tahun kalender
        if _DATE_RE.search(text):
            return True
        # Jika sama sekali tidak ada digit angka, bukan tanggal
        if not re.search(r"\d", text):
            return False
        return bool(re.search(rf"\b(?:1[6-9]\d{{2}}|20\d{{2}})\b|{_MONTH_PATTERN}", text, re.IGNORECASE))

    if answer_type == "NUMBER":
        # Tahun 4 digit berdiri sendiri bukan kuantitas angka biasa
        if re.match(r"^(?:1[6-9]\d{2}|20\d{2})$", text):
            return False
        if _NUMBER_RE.match(text):
            return True
        # Jika tidak ada digit sama sekali, bukan angka
        if not re.search(r"\d", text):
            return False
        return bool(re.search(r"^\d+", text))

    if answer_type == "PERSON":
        # Nama orang tidak boleh memuat angka
        if re.search(r"\d", text):
            return False
        # Minimal harus berawalan huruf kapital
        words = text.split()
        if not words:
            return False
        # Paling tidak kata pertama dan mayoritas kata berhuruf kapital
        capital_count = sum(1 for w in words if w[0].isupper())
        return capital_count >= 1 and (capital_count / len(words)) >= 0.5

    if answer_type == "LOCATION":
        # Lokasi tidak boleh murni angka
        if text.isdigit():
            return False
        # Bersihkan preposisi awal jika ada
        cleaned = re.sub(r"^(?:di|ke|dari)\s+", "", text, flags=re.IGNORECASE)
        words = cleaned.split()
        if not words:
            return False
        if any(re.match(rf"^{_LOCATION_INDICATORS}$", w, re.IGNORECASE) for w in words):
            return True
        return words[0][0].isupper()

    if answer_type == "ORGANIZATION":
        if text.isdigit():
            return False
        # Akronim atau frasa kapital
        if text.isupper() and len(text) >= 2:
            return True
        words = text.split()
        if not words:
            return False
        if any(re.match(rf"^{_ORG_INDICATORS}$", w, re.IGNORECASE) for w in words):
            return True
        if words[0][0].isupper():
            return True
        return None

    return None
