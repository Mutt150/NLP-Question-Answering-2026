"""Antarmuka Orang 3 untuk NER dan pemeriksaan tipe keluaran bersama Orang 5."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import os
import re
from typing import Any

from .contracts import AnswerType

logger = logging.getLogger(__name__)

# Model NER lokal Bahasa Indonesia yang ringan untuk dijalankan di CPU laptop.
DEFAULT_NER_MODEL = os.environ.get("QA_NER_MODEL", "cahya/bert-base-indonesian-NER")
USE_LOCAL_NER = os.environ.get("QA_USE_NER", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

NER_LABEL_TO_ANSWER_TYPE: dict[str, str] = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "FAC": "LOCATION",
    "LOCATION": "LOCATION",
    "ORG": "ORGANIZATION",
    "NOR": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "DAT": "DATE",
    "DATE": "DATE",
    "TIM": "DATE",
    "TIME": "DATE",
    "QTY": "NUMBER",
    "CRD": "NUMBER",
    "ORD": "NUMBER",
    "PRC": "NUMBER",
    "MON": "NUMBER",
    "NUMBER": "NUMBER",
    "QUANTITY": "NUMBER",
    "EVT": "FACT",
    "EVENT": "FACT",
    "LAW": "FACT",
    "WOA": "FACT",
    "PRD": "FACT",
}


@dataclass(frozen=True)
class NEREntity:
    """Representasi entitas terdeteksi dari model NER lokal."""

    text: str
    raw_label: str
    answer_type: str
    start: int
    end: int
    score: float


_NER_PIPELINE: Any = None
_NER_LOAD_ATTEMPTED: bool = False
_NER_LOAD_ERROR: str | None = None

MONTHS_PATTERN = (
    r"(?:Januari|Februari|Pebruari|Maret|April|Mei|Juni|Juli|Agustus|"
    r"September|Oktober|Nopember|November|Desember|"
    r"Muharam|Safar|Rabiul\s+Awal|Rabiul\s+Akhir|Jumadil\s+Awal|Jumadil\s+Akhir|"
    r"Rajab|Syakban|Ramadan|Syawal|Zulkaidah|Zulhijah|"
    r"Sura|Sapar|Mulud|Bakda\s+Mulud|Jumadilawal|Jumadilakir|Rejeb|Ruwah|Pasa|Sawal|Dulkangidah|Besar)"
)
DAYS_PATTERN = r"(?:Senin|Selasa|Rabu|Kamis|Jumat|Jum'at|Sabtu|Minggu|Ahad|Wage|Kliwon|Legi|Pahing|Pon)"

FULL_DATE_RE = re.compile(
    rf"(?:{DAYS_PATTERN},?\s+)?\b\d{{1,2}}\s+{MONTHS_PATTERN}\s+\d{{3,4}}(?:\s*(?:M|SM|H))?\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    rf"\b(?:awal|pertengahan|akhir)?\s*{MONTHS_PATTERN}\s+\d{{3,4}}(?:\s*(?:M|SM|H))?\b",
    re.IGNORECASE,
)
DAY_MONTH_RE = re.compile(rf"\b\d{{1,2}}\s+{MONTHS_PATTERN}\b", re.IGNORECASE)
NUMERIC_DATE_RE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
CENTURY_RE = re.compile(r"\babad\s+(?:ke-\d{1,2}|[IVX]+)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|[1-9][0-9]{2}\s*(?:M|SM|H))\b")
DAY_ONLY_RE = re.compile(rf"^{DAYS_PATTERN}$", re.IGNORECASE)

NUMBER_WORDS = {
    "nol",
    "satu",
    "dua",
    "tiga",
    "empat",
    "lima",
    "enam",
    "tujuh",
    "delapan",
    "sembilan",
    "sepuluh",
    "sebelas",
    "belas",
    "puluh",
    "seratus",
    "ratus",
    "seribu",
    "ribu",
    "sejuta",
    "juta",
    "miliar",
    "triliun",
    "pertama",
    "kedua",
    "ketiga",
    "keempat",
    "kelima",
    "keenam",
    "ketujuh",
    "kedelapan",
    "kesembilan",
    "kesepuluh",
    "tunggal",
    "sepasang",
    "lusin",
    "kodi",
    "setengah",
    "separuh",
}

QUANTITY_UNITS = {
    "orang",
    "anak",
    "putra",
    "putri",
    "istri",
    "suami",
    "saudara",
    "kali",
    "tahun",
    "bulan",
    "hari",
    "minggu",
    "abad",
    "buku",
    "karya",
    "bahasa",
    "pasukan",
    "anggota",
    "cabang",
    "sekolah",
    "persen",
    "juta",
    "ribu",
    "miliar",
    "buah",
    "lembar",
    "helai",
    "ekor",
    "unit",
    "negara",
    "kota",
    "pulau",
    "periode",
    "masa",
    "jabatan",
    "gelar",
    "wilayah",
    "provinsi",
}

PERSON_TITLES_RE = re.compile(
    r"(?:^|\b)(?:"
    r"(?:Prof|Dr|Drs|Dra|Ir|Mr|Ny|Nn|H|Hj|K\.H|KH|R\.A|RA|R\.M|RM|R\.Ng|R|St|B\.R\.M|BRM)\."
    r"|(?:Sutan|Tengku|Teuku|Cut|Tuanku|Pangeran|Sultan|Raden|Ajeng|Mas|Ki|Kyai|Kiai|"
    r"Syekh|Habib|Buya|Bung|Jenderal|Letjen|Mayjen|Brigjen|Kolonel|Mayor|Kapten|"
    r"Laksamana|Marsekal|Bendara|Baginda|Datuk|Datuak|Opu|Daeng|Andi|La|I)\b"
    r")",
)
STANDALONE_TITLES = {
    "prof",
    "dr",
    "drs",
    "dra",
    "ir",
    "mr",
    "ny",
    "nn",
    "h",
    "hj",
    "kh",
    "ra",
    "rm",
    "rng",
    "r",
    "st",
    "brm",
    "sutan",
    "tengku",
    "teuku",
    "cut",
    "tuanku",
    "pangeran",
    "sultan",
    "raden",
    "ajeng",
    "mas",
    "ki",
    "kyai",
    "kiai",
    "syekh",
    "habib",
    "buya",
    "bung",
    "jenderal",
    "letjen",
    "mayjen",
    "brigjen",
    "kolonel",
    "mayor",
    "kapten",
    "laksamana",
    "marsekal",
    "haji",
    "hajjah",
    "bendara",
    "baginda",
    "datuk",
    "datuak",
}
PERSON_PARTICLES = {"bin", "binti", "van", "der", "de", "den", "da", "al", "el", "ibn"}

ORG_KEYWORDS_RE = re.compile(
    r"\b(?:partai|organisasi|perkumpulan|perhimpunan|persatuan|serikat|sarekat|ikatan|"
    r"himpunan|lembaga|badan|komite|panitia|dewan|majelis|kongres|gerakan|front|"
    r"barisan|laskar|tentara|angkatan|pasukan|divisi|resimen|batalyon|sekolah|sakola|"
    r"school|kweekschool|perguruan|universitas|institut|instituut|akademi|fakultas|"
    r"pesantren|madrasah|kementerian|departemen|kabinet|pemerintah|kerajaan|kesultanan|"
    r"harian|surat\s+kabar|koran|majalah|penerbit|kantor\s+berita|yayasan|koperasi|"
    r"bank|perusahaan|dinas|perpustakaan|kearsipan|kompeni|balai|sma|smk|smp|sd|els|his)\b",
    re.IGNORECASE,
)

KNOWN_ORGS_RE = re.compile(
    r"\b(?:BPUPKI|PPKI|PNI|PKI|NU|Masyumi|PSI|Parindra|Gerindo|GAPI|Putera|PETA|Heiho|"
    r"TKR|TRI|TNI|KNIP|BKR|Budi\s+Utomo|Boedi\s+Oetomo|Sarekat\s+Islam|Indische\s+Partij|"
    r"Muhammadiyah|Nahdlatul\s+Ulama|Taman\s+Siswa|Jong\s+Java|Jong\s+Sumatranen\s+Bond|"
    r"Perhimpunan\s+Indonesia|Tri\s+Koro\s+Dharmo|Putri\s+Mardika|Poetri\s+Mardika|"
    r"Aisyiyah|Al-Irsyad|Persis|Volksraad|STOVIA|OSVIA|MULO|AMS|HBS|RHS|HIS|ELS|THHK|VOC|"
    r"Kweekschool|NIAS|GHS|Sakola\s+Istri|Sakola\s+Kaoetamaan\s+Istri|Sakola\s+Rad[eé]n\s+D[eé]wi|"
    r"Kompeni)\b",
)

LOCATION_KEYWORDS_RE = re.compile(
    r"\b(?:kota|kabupaten|provinsi|desa|kampung|nagari|luhak|kecamatan|kelurahan|distrik|"
    r"keresidenan|wilayah|daerah|pulau|kepulauan|gunung|sungai|danau|selat|teluk|"
    r"tanjung|pantai|lembah|negara|negeri|republik|hindia\s+belanda|benteng|fort|gua|goa|"
    r"taman\s+makam\s+pahlawan|tmp|pemakaman|pekuburan|jalan|pendopo)\b",
    re.IGNORECASE,
)

KNOWN_PLACES_RE = re.compile(
    r"\b(?:Indonesia|Hindia\s+Belanda|Belanda|Jepang|Inggris|Prancis|Jerman|Amerika|"
    r"Arab|Mekah|Mekkah|Madinah|Mesir|Kairo|Singapura|Malaysia|Malaka|Aceh|Sumatra|Sumatera|"
    r"Jawa|Kalimantan|Borneo|Sulawesi|Celebes|Papua|Irian|Maluku|Bali|Lombok|Timor|"
    r"Madura|Banten|Batavia|Jakarta|Surabaya|Bandung|Yogyakarta|Jogjakarta|Surakarta|"
    r"Solo|Semarang|Medan|Padang|Bukittinggi|Palembang|Makassar|Manado|Ambon|Denpasar|"
    r"Bengkulu|Ende|Banda\s+Neira|Digul|Boven\s+Digoel|Blitar|Jombang|Tuban|Gresik|"
    r"Demak|Kudus|Jepara|Rembang|Purworejo|Magelang|Banyumas|Garut|Tasikmalaya|"
    r"Cirebon|Bogor|Depok|Tangerang|Bekasi|Den\s+Haag|Leiden|Amsterdam|Rotterdam|"
    r"Bonjol|Pagaruyung|Minahasa|Nusalaut|Saparua|Fort\s+de\s+Kock|Tegalrejo|"
    r"Gua\s+Selarong|Bungaya|Gowa|Tallo|Sombaopu|Kalibata|TMP\s+Kalibata|"
    r"Taman\s+Makam\s+Pahlawan\s+Kalibata|Koto\s+Gadang|Agam|Pasaman|Payakumbuh|"
    r"Pariaman|Batusangkar|Cicalengka|Cianjur|Sumedang|Tapanuli|Banjarmasin|Martapura|"
    r"Ternate|Tidore|Bone|Mataram|Kutaraja|Banda\s+Aceh|Meulaboh|Malang|Kediri|Madiun|"
    r"Pekalongan|Tegal|Salatiga|Klaten|Pacitan|Mojokerto|Sidoarjo|Jayakarta)\b",
)

NON_ENTITY_PHRASES_RE = re.compile(
    r"^(?:Bahasa\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+|Salah\s+satu|Perang\s+\w+|Perjanjian\s+\w+|Konvensi\s+\w+)$",
    re.IGNORECASE,
)

ROMAN_NUMERAL_RE = re.compile(
    r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)$"
)


def get_ner_pipeline() -> Any:
    """Muat model NER lokal (CPU laptop) secara lazy dan simpan sebagai singleton."""
    global _NER_PIPELINE, _NER_LOAD_ATTEMPTED, _NER_LOAD_ERROR

    if not USE_LOCAL_NER:
        return None
    if _NER_LOAD_ATTEMPTED:
        return _NER_PIPELINE

    _NER_LOAD_ATTEMPTED = True
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(DEFAULT_NER_MODEL)
        _NER_PIPELINE = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,  # CPU lokal agar stabil di laptop tanpa GPU
        )
    except (
        Exception
    ) as exc:  # pragma: no cover - fallback saat offline/dependensi tidak ada
        _NER_LOAD_ERROR = str(exc)
        _NER_PIPELINE = None
        logger.warning(
            "Model NER lokal '%s' tidak dapat dimuat (%s); beralih ke ekstraksi berbasis aturan.",
            DEFAULT_NER_MODEL,
            exc,
        )
    return _NER_PIPELINE


def ner_status() -> dict[str, Any]:
    """Informasi status model NER lokal untuk debugging/laporan."""
    pipe = get_ner_pipeline()
    return {
        "enabled": USE_LOCAL_NER,
        "model_name": DEFAULT_NER_MODEL,
        "loaded": pipe is not None,
        "error": _NER_LOAD_ERROR,
    }


def _normalize_ner_label(raw_label: str) -> str | None:
    cleaned = raw_label.strip().upper()
    if cleaned.startswith(("B-", "I-", "E-", "S-")):
        cleaned = cleaned[2:]
    return NER_LABEL_TO_ANSWER_TYPE.get(cleaned)


def _merge_adjacent_entities(text: str, entities: list[NEREntity]) -> list[NEREntity]:
    """Gabungkan entitas bertipe sama yang terpecah oleh spasi/titik/partikel."""
    if not entities:
        return []

    entities_sorted = sorted(entities, key=lambda e: (e.start, e.end))
    merged: list[NEREntity] = [entities_sorted[0]]

    for cur in entities_sorted[1:]:
        prev = merged[-1]
        if cur.answer_type == prev.answer_type and cur.start >= prev.end:
            gap = text[prev.end : cur.start]
            gap_stripped = gap.strip()
            can_merge = False
            if gap == "" or re.fullmatch(r"[\s.-]+", gap):
                can_merge = True
            elif (
                prev.answer_type == "PERSON"
                and gap_stripped.lower() in PERSON_PARTICLES
                and len(gap) <= 10
            ):
                can_merge = True
            elif (
                prev.answer_type == "ORGANIZATION"
                and gap_stripped.lower()
                in {"dan", "of", "der", "van", "en", "voor", "di", "de"}
                and len(gap) <= 8
            ):
                can_merge = True

            if can_merge:
                new_start = prev.start
                new_end = cur.end
                span_text = text[new_start:new_end].strip(" ,.;:()\"'")
                if span_text:
                    real_s = text.find(span_text, new_start, new_end)
                    real_e = real_s + len(span_text)
                    avg_score = (prev.score + cur.score) / 2.0
                    merged[-1] = NEREntity(
                        text=span_text,
                        raw_label=prev.raw_label,
                        answer_type=prev.answer_type,
                        start=real_s,
                        end=real_e,
                        score=avg_score,
                    )
                    continue
        merged.append(cur)

    return merged


@lru_cache(maxsize=512)
def extract_ner_entities(text: str, min_score: float = 0.50) -> tuple[NEREntity, ...]:
    """Jalankan model NER lokal pada teks dan kembalikan daftar entitas terstandarisasi."""
    if not text or not text.strip():
        return ()

    pipe = get_ner_pipeline()
    if pipe is None:
        return ()

    try:
        raw_preds = pipe(text)
    except Exception as exc:  # pragma: no cover
        logger.warning("Inferensi NER lokal gagal: %s", exc)
        return ()

    collected: list[NEREntity] = []
    text_len = len(text)

    for pred in raw_preds:
        score = float(pred.get("score", 0.0))
        if score < min_score:
            continue
        raw_label = str(pred.get("entity_group") or pred.get("entity") or "")
        mapped = _normalize_ner_label(raw_label)
        if not mapped:
            continue

        start = pred.get("start")
        end = pred.get("end")
        if start is None or end is None:
            continue
        start_i = max(0, int(start))
        end_i = min(text_len, int(end))
        if start_i >= end_i:
            continue

        # Pastikan batas kata tidak terpotong di tengah token alfanumerik (subword BERT)
        while start_i > 0 and text[start_i - 1].isalnum():
            start_i -= 1
        while end_i < text_len and text[end_i].isalnum():
            end_i += 1

        raw_span = text[start_i:end_i]
        cleaned_span = raw_span.strip(" ,.;:()\"'[]")
        if not cleaned_span:
            continue

        real_start = text.find(cleaned_span, start_i, end_i)
        if real_start < 0:
            continue
        real_end = real_start + len(cleaned_span)

        collected.append(
            NEREntity(
                text=cleaned_span,
                raw_label=raw_label,
                answer_type=mapped,
                start=real_start,
                end=real_end,
                score=score,
            )
        )

    return tuple(_merge_adjacent_entities(text, collected))


def _has_pure_date_or_number(text: str) -> bool:
    if (
        FULL_DATE_RE.fullmatch(text)
        or MONTH_YEAR_RE.fullmatch(text)
        or DAY_MONTH_RE.fullmatch(text)
    ):
        return True
    if (
        NUMERIC_DATE_RE.fullmatch(text)
        or YEAR_RE.fullmatch(text)
        or CENTURY_RE.fullmatch(text)
    ):
        return True
    if re.fullmatch(r"[\d.,\s%-]+", text):
        return True
    return False


def _capitalized_words(text: str) -> list[str]:
    return re.findall(r"\b[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]*\b", text)


def _ner_supports_answer_type(text: str, expected_type: AnswerType) -> bool | None:
    """Gunakan NER lokal untuk memverifikasi entitas pendek yang belum pasti secara aturan."""
    if expected_type not in ("PERSON", "LOCATION", "ORGANIZATION"):
        return None
    if len(text.split()) > 10:
        return None

    # Bungkus dalam konteks netral agar BERT NER memiliki sinyal sintaksis yang lebih stabil
    if expected_type == "PERSON":
        probe = f"Tokoh tersebut bernama {text}."
    elif expected_type == "LOCATION":
        probe = f"Peristiwa itu terjadi di {text}."
    else:
        probe = f"Organisasi tersebut bernama {text}."

    offset = probe.find(text)
    entities = extract_ner_entities(probe, min_score=0.60)
    overlapping = [
        ent for ent in entities if ent.end > offset and ent.start < offset + len(text)
    ]
    if not overlapping:
        return None

    best = max(
        overlapping,
        key=lambda e: (min(e.end, offset + len(text)) - max(e.start, offset), e.score),
    )
    if best.answer_type == expected_type and best.score >= 0.65:
        return True
    if (
        best.answer_type in ("PERSON", "LOCATION", "ORGANIZATION")
        and best.answer_type != expected_type
        and best.score >= 0.85
    ):
        return False
    return None


def check_answer_type(answer: str, answer_type: AnswerType) -> bool | None:
    """Kembalikan True jika sesuai, False jika berlainan, None jika perlu tinjauan."""
    if not isinstance(answer, str) or not answer.strip():
        return False

    text = " ".join(answer.strip().split())
    words = text.split()
    word_count = len(words)

    if answer_type in ("FACT", "OTHER"):
        if not re.search(r"[A-Za-zÀ-ÿ0-9]", text):
            return False
        return None

    if answer_type == "DATE":
        has_date = bool(
            FULL_DATE_RE.search(text)
            or MONTH_YEAR_RE.search(text)
            or DAY_MONTH_RE.search(text)
            or NUMERIC_DATE_RE.search(text)
            or CENTURY_RE.search(text)
            or YEAR_RE.search(text)
            or DAY_ONLY_RE.match(text)
        )
        if not has_date:
            return False
        return True if word_count <= 12 else None

    if answer_type == "NUMBER":
        lower_tokens = re.findall(r"\w+", text.lower())
        has_digits = bool(re.search(r"\b\d+(?:[.,]\d+)?\b", text))
        has_num_word = any(tok in NUMBER_WORDS for tok in lower_tokens)
        if not (has_digits or has_num_word):
            return False
        is_calendar_date = bool(
            FULL_DATE_RE.search(text)
            or MONTH_YEAR_RE.search(text)
            or DAY_MONTH_RE.search(text)
            or NUMERIC_DATE_RE.search(text)
        )
        has_unit = any(tok in QUANTITY_UNITS for tok in lower_tokens) or "%" in text
        if is_calendar_date and not has_unit:
            return False
        return True if word_count <= 12 else None

    if answer_type == "PERSON":
        if _has_pure_date_or_number(text):
            return False
        if NON_ENTITY_PHRASES_RE.match(text):
            return False
        cap_words = _capitalized_words(text)
        if not cap_words:
            return False
        if word_count == 1 and words[0].rstrip(".").lower() in STANDALONE_TITLES:
            return False
        if ORG_KEYWORDS_RE.search(text) or KNOWN_ORGS_RE.search(text):
            return False
        if LOCATION_KEYWORDS_RE.search(text) and not PERSON_TITLES_RE.search(text):
            return False
        if re.match(r"^(?:di|ke|dari)\s+", text, re.IGNORECASE):
            return False
        if KNOWN_PLACES_RE.fullmatch(text):
            return False
        if PERSON_TITLES_RE.search(text) or any(
            w.lower() in PERSON_PARTICLES for w in words
        ):
            return True if word_count <= 10 else None
        if word_count <= 6:
            non_cap = [
                w
                for w in words
                if not re.match(r"^[A-ZÀ-Ý]", w) and w.lower() not in PERSON_PARTICLES
            ]
            if not non_cap:
                ner_verdict = _ner_supports_answer_type(text, "PERSON")
                if ner_verdict is False:
                    return False
                return True
        return _ner_supports_answer_type(text, "PERSON")

    if answer_type == "LOCATION":
        if _has_pure_date_or_number(text):
            return False
        if NON_ENTITY_PHRASES_RE.match(text):
            return False
        cap_words = _capitalized_words(text)
        if not cap_words:
            return False
        is_known_loc = bool(
            LOCATION_KEYWORDS_RE.search(text) or KNOWN_PLACES_RE.search(text)
        )
        if PERSON_TITLES_RE.search(text):
            return False
        if any(w.lower() in PERSON_PARTICLES for w in words) and not is_known_loc:
            return False
        if ORG_KEYWORDS_RE.search(text) or KNOWN_ORGS_RE.search(text):
            return False
        if is_known_loc:
            return True if word_count <= 10 else None
        if (
            re.match(r"^(?:di|ke|dari)\s+[A-ZÀ-Ý]", text, re.IGNORECASE)
            and word_count <= 8
        ):
            return True
        ner_verdict = _ner_supports_answer_type(text, "LOCATION")
        if ner_verdict is not None:
            return ner_verdict
        if word_count <= 5:
            non_cap = [
                w
                for w in words
                if not re.match(r"^[A-ZÀ-Ý]", w)
                and w.lower()
                not in {"dan", "di", "ke", "dari", "de", "den", "van", "der"}
            ]
            if not non_cap:
                return True
        return None

    if answer_type == "ORGANIZATION":
        if _has_pure_date_or_number(text):
            return False
        cap_words = _capitalized_words(text)
        acronyms = [
            tok
            for tok in re.findall(r"\b[A-Z]{2,10}\b", text)
            if not ROMAN_NUMERAL_RE.match(tok)
        ]
        if not cap_words and not acronyms:
            return False
        if PERSON_TITLES_RE.search(text) and not (
            ORG_KEYWORDS_RE.search(text) or KNOWN_ORGS_RE.search(text)
        ):
            return False
        if ORG_KEYWORDS_RE.search(text) or KNOWN_ORGS_RE.search(text) or acronyms:
            return True if word_count <= 12 else None
        return _ner_supports_answer_type(text, "ORGANIZATION")

    return None
