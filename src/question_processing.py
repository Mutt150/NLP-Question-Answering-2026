"""Question processing untuk pipeline klasik dan RAG."""

import re

from .contracts import AnswerType, QuestionInfo

# urutan berpengaruh, pola spesifik ditaruh di atas
# (kah)? supaya bentuk seperti "kapankah" dan "siapakah" tetap cocok
# "apakah" hanya di awal kalimat (ya/tidak), agar "partai apakah" tidak ikut
ANSWER_RULES: list[tuple[str, AnswerType]] = [
    (r"\b(mengapa|kenapa|bagaimana)(kah)?\b|^apakah\b", "OTHER"),
    (r"\b(tahun|tanggal|bulan|hari) (berapa|apa)(kah)?\b", "DATE"),
    (r"\bkapan(kah)?\b", "DATE"),
    (r"\bsiapa(kah)?\b", "PERSON"),
    (r"\b(organisasi|partai|perkumpulan|lembaga|sekolah|perguruan|kerajaan|harian) (apa|mana)(kah)?\b", "ORGANIZATION"),
    (r"\b(di|ke|dari) ?mana(kah)?\b", "LOCATION"),
    (r"\b(kota|kampung|desa|daerah|wilayah|negara|negeri|pulau|provinsi|tempat) (apa|mana)(kah)?\b", "LOCATION"),
    (r"\bberapa(kah)?\b", "NUMBER"),
    (r"\bapa\b", "FACT"),
]

_COMPILED_RULES = [(re.compile(pattern), answer_type) for pattern, answer_type in ANSWER_RULES]


def detect_answer_type(question: str) -> AnswerType:
    """Tentukan tipe jawaban dari kata tanya. Tanpa pola yang cocok hasilnya OTHER."""
    text = question.lower()
    for pattern, answer_type in _COMPILED_RULES:
        if pattern.search(text):
            return answer_type
    return "OTHER"


def analyze_question(question: str) -> QuestionInfo:
    """Ubah pertanyaan menjadi answer_type, keywords, dan relation_cues."""
    return {
        "answer_type": detect_answer_type(question),
        "keywords": [],
        "relation_cues": [],
    }
