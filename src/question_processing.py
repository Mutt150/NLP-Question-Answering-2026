"""Question processing untuk pipeline klasik dan RAG."""

import re

from .contracts import AnswerType, QuestionInfo

# urutan berpengaruh, pola spesifik ditaruh di atas
ANSWER_RULES: list[tuple[str, AnswerType]] = [
    (r"\b(mengapa|kenapa|bagaimana)(kah)?\b|^apakah\b", "OTHER"),
    (r"\b(tahun|tanggal|bulan|hari) (berapa|apa)(kah)?\b", "DATE"),
    (r"\bkapan(kah)?\b", "DATE"),
    (r"\bsiapa(kah)?\b", "PERSON"),
    (r"\b(organisasi|partai|perkumpulan|lembaga|sekolah|perguruan|kerajaan|harian) (apa|mana)(kah)?\b", "ORGANIZATION"),
    (r"\bapa(kah)? nama (organisasi|partai|perkumpulan|lembaga|sekolah|perguruan|kerajaan|harian)\b", "ORGANIZATION"),
    (r"\b(di|ke|dari) ?mana(kah)?\b", "LOCATION"),
    (r"\b(kota|kampung|desa|daerah|wilayah|negara|negeri|pulau|provinsi|tempat) (apa|mana)(kah)?\b", "LOCATION"),
    (r"\bapa(kah)? nama (kota|kampung|desa|daerah|wilayah|negara|negeri|pulau|provinsi|tempat)\b", "LOCATION"),
    (r"\bberapa(kah)?\b", "NUMBER"),
    (r"\bapa\b", "FACT"),
]

_COMPILED_RULES = [(re.compile(pattern), answer_type) for pattern, answer_type in ANSWER_RULES]

QUESTION_WORDS = {
    "siapa", "siapakah", "kapan", "kapankah", "mana", "manakah", "berapa", "berapakah",
    "apa", "apakah", "mengapa", "kenapa", "bagaimana", "bagaimanakah",
}

STOPWORDS = {
    "yang", "di", "ke", "dari", "pada", "dan", "atau", "ia", "dia", "beliau",
    "adalah", "ialah", "merupakan", "ini", "itu", "tersebut", "dengan", "untuk",
    "bagi", "kepada", "oleh", "dalam", "atas", "tentang", "sebagai", "para",
    "saat", "ketika", "sejak", "selama", "setelah", "sebelum", "sampai", "hingga",
    "telah", "sudah", "akan", "juga", "pernah", "nama",
}

RELATION_CUES = {
    "lahir": r"(di|ke)?lahir(kan|an)?",
    "wafat": r"wafat|meninggal|gugur|kematian",
    "diri": r"(men|di|pen|ber)diri(kan)?",
    "pimpin": r"(di)?pimpin|memimpin|pemimpin|kepemimpinan",
    "asing": r"(di|meng|peng)asing(kan|an)",
    "tangkap": r"(di)?tangkap|menangkap|penangkapan",
    "nikah": r"(me|di|per)?nikah(i|kan|an)?",
    "jabat": r"(men|di)?jabat(an)?",
    "makam": r"(di)?makam(kan)?|pemakaman",
    "proklamasi": r"(mem|di)?proklamasi(kan)?",
    "ayah": r"ayah(nya|anda)?",
    "ibu": r"ibu(nya|nda)?(?! kota)",
    "istri": r"istri(nya)?",
    "suami": r"suami(nya)?",
    "anak": r"anak(nya)?|putra|putri",
}

_COMPILED_CUES = {cue: re.compile(rf"\b(?:{pattern})\b") for cue, pattern in RELATION_CUES.items()}


def detect_answer_type(question: str) -> AnswerType:
    """Tentukan tipe jawaban dari kata tanya. Tanpa pola yang cocok hasilnya OTHER."""
    text = question.lower()
    for pattern, answer_type in _COMPILED_RULES:
        if pattern.search(text):
            return answer_type
    return "OTHER"


def tokenize(text: str) -> list[str]:
    """Pecah teks menjadi token huruf kecil tanpa tanda baca."""
    return re.findall(r"\w+", text.lower())


def extract_keywords(question: str) -> list[str]:
    """Ambil token pertanyaan selain kata tanya dan stopword."""
    tokens = [t for t in tokenize(question) if t not in QUESTION_WORDS and t not in STOPWORDS]
    return list(dict.fromkeys(tokens))


def find_relation_cues(question: str) -> list[str]:
    """Cari kata relasi pada pertanyaan, dikembalikan dalam bentuk dasar."""
    text = question.lower()
    return [cue for cue, pattern in _COMPILED_CUES.items() if pattern.search(text)]


def analyze_question(question: str) -> QuestionInfo:
    """Ubah pertanyaan menjadi answer_type, keywords, dan relation_cues."""
    return {
        "answer_type": detect_answer_type(question),
        "keywords": extract_keywords(question),
        "relation_cues": find_relation_cues(question),
    }
