"""Antarmuka Orang 3 untuk ekstraksi jawaban klasik berbasis NER + Regex."""

from __future__ import annotations

import re

from .check_answer import (
    CENTURY_RE,
    DAY_MONTH_RE,
    FULL_DATE_RE,
    KNOWN_ORGS_RE,
    KNOWN_PLACES_RE,
    LOCATION_KEYWORDS_RE,
    MONTH_YEAR_RE,
    NEREntity,
    NUMERIC_DATE_RE,
    ORG_KEYWORDS_RE,
    PERSON_PARTICLES,
    ROMAN_NUMERAL_RE,
    STANDALONE_TITLES,
    YEAR_RE,
    check_answer_type,
    extract_ner_entities,
)
from .contracts import Answer, AnswerType, QuestionInfo, RetrievedPassage
from .question_processing import QUESTION_WORDS, STOPWORDS

ABBREVIATIONS = {
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
    "no",
    "hlm",
    "dkk",
    "dll",
    "dsb",
    "dst",
    "tgl",
    "thn",
    "sh",
    "mh",
    "ba",
    "ma",
    "phd",
    "letjen",
    "mayjen",
    "brigjen",
    "jend",
    "kol",
    "may",
    "kapt",
    "laks",
    "mars",
    "brm",
    "tmp",
}

EXPANDED_CUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "lahir": re.compile(
        r"\b(?:(?:di|ke|ter)?lahir(?:kan|an|nya)?|berasal\s+dari)\b",
        re.IGNORECASE,
    ),
    "wafat": re.compile(
        r"\b(?:wafat|meninggal(?:\s+dunia)?|gugur|kematian|tutup\s+usia|berpulang|mangkal|mangkat)\b",
        re.IGNORECASE,
    ),
    "diri": re.compile(
        r"\b(?:(?:men|di|pen|ber)diri(?:kan)?|memprakarsai|menggagas|membentuk|dibentuk)\b",
        re.IGNORECASE,
    ),
    "pimpin": re.compile(
        r"\b(?:(?:di)?pimpin|memimpin|pemimpin|kepemimpinan|ketua|mengetuai|panglima|komandan)\b",
        re.IGNORECASE,
    ),
    "asing": re.compile(
        r"\b(?:(?:di|meng|peng)asing(?:kan|an)|dibuang|pembuangan|diinternir)\b",
        re.IGNORECASE,
    ),
    "tangkap": re.compile(
        r"\b(?:(?:di)?tangkap|menangkap|penangkapan|ditahan|penahanan|dipenjara)\b",
        re.IGNORECASE,
    ),
    "nikah": re.compile(
        r"\b(?:(?:me|di|per)?nikah(?:i|kan|an)?|memperistri|dipersunting|kawin|menikahi)\b",
        re.IGNORECASE,
    ),
    "jabat": re.compile(
        r"\b(?:(?:men|di)?jabat(?:an)?|diangkat\s+menjadi|dilantik\s+sebagai|terpilih\s+sebagai)\b",
        re.IGNORECASE,
    ),
    "makam": re.compile(
        r"\b(?:(?:di|me|pe)?makam(?:kan|an)?|dikebumikan|dikubur(?:kan)?|"
        r"kuburan|pekuburan|pusara|peristirahatan\s+terakhir|"
        r"taman\s+makam\s+pahlawan|tmp)\b",
        re.IGNORECASE,
    ),
    "proklamasi": re.compile(r"\b(?:(?:mem|di)?proklamasi(?:kan)?)\b", re.IGNORECASE),
    "ayah": re.compile(
        r"\b(?:ayah(?:nya|anda)?|bapak(?:nya)?|berayah|"
        r"(?:putra|putri|anak)\s+(?:dari|pasangan|sulung|bungsu)|"
        r"lahir\s+dari\s+pasangan|orang\s+tua(?:nya)?)\b",
        re.IGNORECASE,
    ),
    "ibu": re.compile(
        r"\b(?:ibu(?:nya|nda)?(?!\s+kota)|ibunda|"
        r"(?:putra|putri|anak)\s+(?:dari|pasangan)|lahir\s+dari\s+pasangan)\b",
        re.IGNORECASE,
    ),
    "istri": re.compile(
        r"\b(?:istri(?:nya)?|memperistri|beristrikan|dipersunting|menikahi)\b",
        re.IGNORECASE,
    ),
    "suami": re.compile(r"\b(?:suami(?:nya)?|bersuamikan)\b", re.IGNORECASE),
    "anak": re.compile(
        r"\b(?:anak(?:nya)?|putra(?:nya)?|putri(?:nya)?|keturunan|dikaruniai)\b",
        re.IGNORECASE,
    ),
    "nama_asli": re.compile(
        r"\b(?:nama\s+(?:asli|kecil|lahir|lengkap|sebenarnya)(?:nya)?|"
        r"bernama\s+(?:asli|kecil|lengkap)|"
        r"(?:di|ter)?lahir(?:kan)?\s+(?:dengan\s+nama|sebagai)|"
        r"diberi\s+nama)\b",
        re.IGNORECASE,
    ),
}

STRICT_BIO_CUES = {
    "lahir",
    "wafat",
    "makam",
    "ayah",
    "ibu",
    "istri",
    "suami",
    "anak",
    "nikah",
    "asing",
    "tangkap",
    "proklamasi",
    "nama_asli",
}

QUESTION_EXTRA_CUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "nama_asli": re.compile(
        r"\bnama\s+(?:asli|kecil|lahir|lengkap|sebenarnya)\b", re.IGNORECASE
    ),
    "lahir": re.compile(r"\b(?:dilahirkan|lahir|kelahiran)\b", re.IGNORECASE),
    "wafat": re.compile(r"\b(?:wafat|meninggal|gugur)\b", re.IGNORECASE),
    "makam": re.compile(
        r"\b(?:dimakamkan|makam|dikebumikan|dikuburkan)\b", re.IGNORECASE
    ),
    "ayah": re.compile(r"\b(?:ayah|bapak)\b", re.IGNORECASE),
    "ibu": re.compile(r"\bibu(?!\s+kota)\b", re.IGNORECASE),
    "istri": re.compile(r"\bistri\b", re.IGNORECASE),
    "suami": re.compile(r"\bsuami\b", re.IGNORECASE),
}

GENERIC_QUERY_WORDS = {
    "benteng",
    "kota",
    "kabupaten",
    "provinsi",
    "desa",
    "kampung",
    "pulau",
    "gunung",
    "sungai",
    "kerajaan",
    "kesultanan",
    "negara",
    "negeri",
    "tempat",
    "daerah",
    "wilayah",
    "organisasi",
    "partai",
    "perkumpulan",
    "lembaga",
    "sekolah",
    "perguruan",
    "harian",
    "perjanjian",
    "konvensi",
    "pertempuran",
    "perang",
    "pangkat",
    "jabatan",
    "penyakit",
    "gelar",
    "karya",
    "buku",
    "judul",
    "asli",
    "kecil",
    "lengkap",
    "jumlah",
    "berapa",
    "tahun",
    "tanggal",
    "bulan",
    "hari",
    "pertama",
    "kedua",
    "ketiga",
    "terakhir",
    "bertugas",
    "kemiliteran",
    "bekerja",
    "mulai",
    "memasuki",
    "bersekolah",
    "menyingkir",
    "membuat",
    "markas",
    "ditandatangani",
    "dicapai",
    "meninggal",
    "wafat",
    "lahir",
    "dilahirkan",
    "dimakamkan",
    "diasingkan",
    "ditangkap",
    "menikah",
    "ayah",
    "ibu",
    "istri",
    "suami",
    "anak",
    "putra",
    "putri",
    "pimpin",
    "dipimpin",
    "memimpin",
    "didirikan",
    "mendirikan",
    "nama",
    "sebenarnya",
    "dimasuki",
}

COMMON_GIVEN_NAMES = {
    "mohammad",
    "muhammad",
    "mohamad",
    "muhamad",
    "ahmad",
    "achmad",
    "abdul",
    "siti",
    "raden",
    "imam",
    "agus",
    "sri",
    "dewi",
    "cut",
    "teuku",
    "tengku",
}

NON_LOCATION_QUESTION_PREFIXES = {
    "perjanjian",
    "perang",
    "pertempuran",
    "konvensi",
    "sumpah",
    "kongres",
    "deklarasi",
    "sultan",
    "pangeran",
    "tuanku",
    "imam",
    "raden",
    "cut",
    "teuku",
    "tengku",
    "jenderal",
    "bung",
    "ki",
    "kyai",
    "kiai",
}

NON_PERSON_LEADERS = {
    "pada",
    "selain",
    "setelah",
    "sebelum",
    "ketika",
    "saat",
    "dalam",
    "untuk",
    "sebagai",
    "karena",
    "namun",
    "tetapi",
    "kemudian",
    "lalu",
    "oleh",
    "dari",
    "di",
    "ke",
    "ia",
    "dia",
    "beliau",
    "mereka",
    "kami",
    "kita",
    "tahun",
    "tanggal",
    "bulan",
    "hari",
    "masa",
    "perang",
    "kongres",
    "sumpah",
    "proklamasi",
    "republik",
    "pemerintah",
    "indonesia",
    "belanda",
    "jepang",
    "kota",
    "pulau",
    "provinsi",
    "kerajaan",
    "kesultanan",
    "partai",
    "organisasi",
    "sekolah",
    "sakola",
    "perguruan",
    "universitas",
    "harian",
    "surat",
    "majalah",
    "januari",
    "februari",
    "pebruari",
    "maret",
    "april",
    "mei",
    "juni",
    "juli",
    "agustus",
    "september",
    "oktober",
    "nopember",
    "november",
    "desember",
    "senin",
    "selasa",
    "rabu",
    "kamis",
    "jumat",
    "sabtu",
    "minggu",
    "sejak",
    "hingga",
    "sampai",
    "meski",
    "meskipun",
    "walaupun",
    "sementara",
    "sedangkan",
    "bahwa",
    "dengan",
    "tanpa",
    "antara",
    "atas",
    "bawah",
    "salah",
    "bahasa",
    "dinas",
    "perpustakaan",
    "kearsipan",
    "luhak",
    "nagari",
    "benteng",
    "gua",
    "goa",
    "gunung",
    "sungai",
    "selat",
    "teluk",
    "tanjung",
    "perjanjian",
    "konvensi",
    "pertempuran",
    "dekrit",
    "maklumat",
    "undang",
    "kini",
    "sekarang",
    "dahulu",
    "taman",
    "kompeni",
    "naskah",
    "buku",
}

CLAUSE_SPLIT_RE = re.compile(
    r"(?:,\s*(?:dan|serta|sedangkan|sementara|lalu|kemudian)\b|;|[–—]|\s+-\s+|\(|\)|"
    r"\s+(?:dan|serta|lalu|kemudian)\s+(?=(?:di|ke|men|mem|meng|meny|ber|ter|"
    r"wafat|meninggal|gugur|lahir|menikah|menjabat|diasingkan|ditangkap|dimakamkan)\w*))",
    re.IGNORECASE,
)

# Pola rentang hidup di dalam tanda kurung, mis. (Yogyakarta, 11 November 1785 – Makassar, 8 Januari 1855)
LIFESPAN_PAREN_RE = re.compile(r"\(([^()]{3,140}?)\s*[–—-]\s*([^()]{3,140}?)\)")

# Penanda penjelasan/modernisasi dalam tanda kurung yang bukan jawaban utama historis
PAREN_EXPLANATORY_RE = re.compile(
    r"^\s*(?:kini|sekarang|dahulu|sebelumnya|bahasa\s+\w+|belanda|inggris|indonesia|atau|singkatan|yakni)\b",
    re.IGNORECASE,
)

# Penanda nama asli dalam tanda kurung
PAREN_BIRTHNAME_RE = re.compile(
    r"\b(?:lahir\s+dengan\s+nama|nama\s+(?:asli|kecil|lahir|lengkap)|bernama\s+asli|terlahir)\b",
    re.IGNORECASE,
)

# Penanda bahwa kata "Istri" adalah bagian nama sekolah/organisasi/buku, bukan relasi pernikahan
NON_SPOUSE_ISTRI_CTX_RE = re.compile(
    r"\b(?:sakola|sekolah|kaoetamaan|kautamaan|keutamaan|buku|yayasan|perkumpulan|organisasi|majalah)\s+(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)

LOC_PREP_RE = re.compile(
    r"\b(?:di|ke|dari)\s+("
    r"(?:(?:Kota|Kabupaten|Provinsi|Desa|Kampung|Nagari|Luhak|Kecamatan|Keresidenan|"
    r"Wilayah|Daerah|Pulau|Kepulauan|Gunung|Sungai|Benteng|Gua|Goa|Kerajaan|Kesultanan|"
    r"Negara|Negeri|Republik|Taman\s+Makam\s+Pahlawan|TMP|Jalan)\s+)?"
    r"[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+(?:\s+(?:(?:Kota|Kabupaten|Provinsi|Desa|Kampung|Pulau|Hindia|Belanda|"
    r"Barat|Timur|Utara|Selatan|Tengah|Tenggara|Daya|Besar|Kecil|Raya|Baru|Lama|Makam|Pahlawan)|[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+))*"
    r"(?:,\s*(?:(?:Provinsi|Kabupaten|Pulau|Hindia|Belanda|Jawa|Sumatra|Sumatera|Kalimantan|Sulawesi|"
    r"Papua|Maluku|Bali|Nusa|Negeri)\s*)?[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+(?:\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+)*)?"
    r")",
)

LOC_PREFIX_RE = re.compile(
    r"\b((?:Kota|Kabupaten|Provinsi|Desa|Kampung|Nagari|Luhak|Pulau|Kepulauan|Benteng|"
    r"Gua|Goa|Kerajaan|Kesultanan|Taman\s+Makam\s+Pahlawan|TMP)\s+"
    r"[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+(?:\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+){0,3})",
)

TITLED_PERSON_RE = re.compile(
    r"(?:"
    r"(?:(?:Prof|Dr|Drs|Dra|Ir|Mr|Ny|Nn|H|Hj|K\.H|KH|R\.A|RA|R\.M|RM|R\.Ng|R|St|B\.R\.M|BRM)\.\s*)+"
    r"|(?:(?:Sutan|Tengku|Teuku|Cut|Tuanku|Pangeran|Sultan|Raden|Ajeng|Mas|Ki|Kyai|Kiai|"
    r"Syekh|Habib|Buya|Bung|Jenderal|Letjen|Mayjen|Brigjen|Kolonel|Mayor|Kapten|"
    r"Laksamana|Marsekal|Bendara|Baginda|Datuk|Datuak)\s+)+"
    r")"
    r"(?:[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+(?:\s+(?:bin|binti|van|der|de|den|da|al-|el-|[A-Z]\.|[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+)){0,6})",
)

LEADING_TITLE_RE = re.compile(
    r"(?:(?:(?:Prof|Dr|Drs|Dra|Ir|Mr|Ny|Nn|H|Hj|K\.H|KH|R\.A|RA|R\.M|RM|R\.Ng|R|St|B\.R\.M|BRM)\.\s*)+"
    r"|(?:(?:Sutan|Tengku|Teuku|Cut|Tuanku|Pangeran|Sultan|Raden|Ajeng|Mas|Ki|Kyai|Kiai|"
    r"Syekh|Habib|Buya|Bung|Jenderal|Letjen|Mayjen|Brigjen|Kolonel|Mayor|Kapten|"
    r"Laksamana|Marsekal|Bendara|Baginda|Datuk|Datuak)\s+)+)$"
)

MULTI_PERSON_RE = re.compile(
    r"\b[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+(?:\s+(?:bin|binti|van|der|de|den|da|al-|el-|[A-Z]\.|[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]+)){1,5}\b"
)

SINGLE_PERSON_RE = re.compile(r"\b[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]{2,}\b")

ORG_PHRASE_RE = re.compile(
    r"\b(?:Partai|Organisasi|Perkumpulan|Perhimpunan|Persatuan|Serikat|Sarekat|Ikatan|"
    r"Himpunan|Lembaga|Badan|Komite|Panitia|Dewan|Majelis|Kongres|Gerakan|Front|"
    r"Barisan|Laskar|Tentara|Angkatan|Sekolah|Sakola|Perguruan|Universitas|Institut|Akademi|"
    r"Pesantren|Madrasah|Kementerian|Departemen|Kabinet|Kerajaan|Kesultanan|"
    r"Harian|Surat\s+Kabar|Majalah|Yayasan|Dinas|Perpustakaan|SMA|SMK|SMP|SD)\s+"
    r"[A-ZÀ-Ý0-9][A-Za-zÀ-ÿ0-9'-]*(?:\s+(?:dan|of|der|van|en|voor|in|de|het|[A-ZÀ-Ý0-9][A-Za-zÀ-ÿ0-9'-]*)){0,6}",
)

ACRONYM_ORG_RE = re.compile(r"\b[A-Z]{2,10}\b")

NUMBER_UNIT_RE = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|"
    r"sebelas|dua\s+belas|tiga\s+belas|empat\s+belas|lima\s+belas|dua\s+puluh|"
    r"tiga\s+puluh|empat\s+puluh|lima\s+puluh|seratus|seribu|sejuta|pertama|kedua|"
    r"ketiga|keempat|kelima|tunggal|sepasang)\s+"
    r"(?:orang|anak|putra|putri|istri|suami|saudara|kali|tahun|bulan|hari|minggu|abad|"
    r"buku|karya|bahasa|pasukan|anggota|cabang|sekolah|persen|juta|ribu|miliar|"
    r"buah|periode|masa|jabatan|gelar|wilayah|provinsi)\b",
    re.IGNORECASE,
)

STANDALONE_NUM_RE = re.compile(
    r"\b(?:ke-\d+|\d+(?:[.,]\d+)?\s*%|\d+(?:[.,]\d+)?|satu|dua|tiga|empat|lima|enam|"
    r"tujuh|delapan|sembilan|sepuluh|sebelas|dua\s+belas|seratus|seribu|pertama|"
    r"kedua|ketiga|keempat|kelima|tunggal)\b",
    re.IGNORECASE,
)

FACT_COPULA_RE = re.compile(
    r"\b(?:adalah|ialah|merupakan|berjudul|bernama|dijuluki|bergelar|yaitu|yakni|sebagai)\s+([^.;!?]+)",
    re.IGNORECASE,
)

FACT_HEAD_NOUN_RE = re.compile(
    r"^(?:karena\s+|dengan\s+)?([a-zà-ÿ]+)\s+apa(?:kah)?\b",
    re.IGNORECASE,
)

FACT_STOP_BOUNDARY_RE = re.compile(
    r"\s+(?:pada|di|ke|dari|yang|oleh|dengan|untuk|karena|sejak|hingga|sampai|"
    r"ketika|saat|setelah|sebelum|tahun|tanggal|bulan|hari|dalam|antara|serta|"
    r"tetapi|namun|sehingga)\b.*$",
    re.IGNORECASE,
)


def _light_stem(word: str) -> str:
    w = word.lower()
    if len(w) <= 4:
        return w
    for suf in ("nya", "lah", "kah", "kan", "an", "i"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    for pref in (
        "meng",
        "meny",
        "men",
        "mem",
        "me",
        "peng",
        "peny",
        "pen",
        "pem",
        "pe",
        "ber",
        "bel",
        "ter",
        "di",
        "ke",
        "se",
    ):
        if w.startswith(pref) and len(w) - len(pref) >= 4:
            w = w[len(pref) :]
            break
    return w


def _enrich_relation_cues(question: str, relation_cues: list[str]) -> list[str]:
    """Tambahkan petunjuk relasi implisit dari redaksi pertanyaan tanpa mengubah input asli."""
    cues = list(relation_cues)
    seen = set(cues)
    for cue_name, pat in QUESTION_EXTRA_CUE_PATTERNS.items():
        if cue_name not in seen and pat.search(question):
            cues.append(cue_name)
            seen.add(cue_name)
    return cues


def _extract_focus_terms(question: str, keywords: list[str]) -> list[str]:
    """Ambil token entitas/subjek spesifik dari pertanyaan (mengabaikan gelar & kata generik)."""
    focus: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"(?<!^)\b[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]{2,}\b", question):
        tok = m.group(0).lower()
        if (
            tok not in QUESTION_WORDS
            and tok not in STOPWORDS
            and tok not in GENERIC_QUERY_WORDS
            and tok not in STANDALONE_TITLES
            and tok not in seen
        ):
            focus.append(tok)
            seen.add(tok)
    if focus:
        return focus
    for kw in keywords:
        kl = kw.lower()
        if (
            kl not in GENERIC_QUERY_WORDS
            and kl not in STANDALONE_TITLES
            and not kl.isdigit()
            and len(kl) > 2
            and kl not in seen
        ):
            focus.append(kl)
            seen.add(kl)
    return focus


def _passage_matches_focus(passage: RetrievedPassage, focus_terms: list[str]) -> bool:
    """Pastikan subjek utama pertanyaan benar-benar hadir pada passage ini."""
    if not focus_terms:
        return True

    combined = f"{passage.get('title', '')} {passage.get('text', '')}".lower()
    p_tokens = set(re.findall(r"\w+", combined))
    p_stems = {_light_stem(t) for t in p_tokens}

    def _has_term(term: str) -> bool:
        return term in p_tokens or _light_stem(term) in p_stems

    if len(focus_terms) == 1:
        return _has_term(focus_terms[0])

    # Untuk subjek multi-kata (mis. "Mohammad Hatta", "Tuanku Imam Bonjol"),
    # hindari kecocokan semu yang hanya mencocokkan nama depan umum ("mohammad", "imam").
    distinctive = [t for t in focus_terms if t not in COMMON_GIVEN_NAMES]
    hits_all = sum(1 for t in focus_terms if _has_term(t))
    if hits_all >= 2:
        return True
    if distinctive:
        return any(_has_term(t) for t in distinctive)
    return hits_all >= 1


def _iter_lifespan_parens(text: str) -> list[tuple[int, int, int, int]]:
    """Kembalikan rentang (left_s, left_e, right_s, right_e) untuk kurung biografi bertahun."""
    results: list[tuple[int, int, int, int]] = []
    for m in LIFESPAN_PAREN_RE.finditer(text):
        left_txt = m.group(1)
        right_txt = m.group(2)
        if re.search(r"\b\d{3,4}\b", left_txt) and re.search(r"\b\d{3,4}\b", right_txt):
            results.append((m.start(1), m.end(1), m.start(2), m.end(2)))
    return results


def _find_valid_cue_matches(text: str, cue_name: str) -> list[re.Match[str]]:
    """Cari kecocokan cue yang valid secara konteks (menghindari false positive seperti 'Sakola Istri')."""
    pat = EXPANDED_CUE_PATTERNS.get(cue_name)
    if pat is None:
        return []

    valid: list[re.Match[str]] = []
    for m in pat.finditer(text):
        if cue_name == "istri":
            matched_word = m.group(0)
            prefix = text[max(0, m.start() - 35) : m.start()]
            # Tolak jika "Istri" bagian dari nama sekolah/organisasi/buku (mis. "Sakola Kaoetamaan Istri")
            if NON_SPOUSE_ISTRI_CTX_RE.search(prefix):
                continue
            if matched_word[0].isupper() and m.start() > 0:
                prev_char = text[m.start() - 1]
                if prev_char.isspace():
                    # Di tengah kalimat dan berkapital → kemungkinan nama lembaga/buku
                    prev_word_m = re.search(r"([A-Za-zÀ-ÿ'-]+)\s+$", prefix)
                    if prev_word_m and prev_word_m.group(1)[0].isupper():
                        continue
        valid.append(m)
    return valid


def _passage_has_strict_cue(passage_text: str, strict_cues: list[str]) -> bool:
    """Periksa apakah passage memiliki minimal satu petunjuk relasi ketat yang valid."""
    if not strict_cues:
        return True
    for cue in strict_cues:
        if _find_valid_cue_matches(passage_text, cue):
            return True
    if any(c in ("lahir", "wafat") for c in strict_cues) and _iter_lifespan_parens(
        passage_text
    ):
        return True
    return False


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    n = len(text)
    start = 0

    def _emit(raw_start: int, raw_end: int) -> None:
        s, e = raw_start, raw_end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            spans.append((text[s:e], s, e))

    i = 0
    while i < n:
        ch = text[i]
        if ch == "\n":
            _emit(start, i)
            start = i + 1
        elif ch in ".!?":
            next_ch = text[i + 1] if i + 1 < n else ""
            if not next_ch or next_ch.isspace():
                is_abbrev = False
                if ch == ".":
                    j = i - 1
                    while j >= start and text[j].isalpha():
                        j -= 1
                    token_before = text[j + 1 : i]
                    if token_before:
                        if (
                            len(token_before) == 1
                            or token_before.lower() in ABBREVIATIONS
                        ):
                            is_abbrev = True
                        elif j >= start and text[j] == ".":
                            is_abbrev = True
                if not is_abbrev:
                    _emit(start, i + 1)
                    start = i + 1
        i += 1

    _emit(start, n)
    return spans


def _clause_bounds(sentence: str) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    last = 0
    for m in CLAUSE_SPLIT_RE.finditer(sentence):
        bounds.append((last, m.start()))
        last = m.end()
    bounds.append((last, len(sentence)))
    return bounds


def _clause_index(bounds: list[tuple[int, int]], pos: int) -> int:
    for idx, (s, e) in enumerate(bounds):
        if s <= pos <= e:
            return idx
    return 0


def _cue_positions(sentence: str, cues: list[str]) -> tuple[list[int], list[int]]:
    target_positions: list[int] = []
    other_positions: list[int] = []
    target_set = set(cues)
    for cue_name in EXPANDED_CUE_PATTERNS:
        for m in _find_valid_cue_matches(sentence, cue_name):
            if cue_name in target_set:
                target_positions.append(m.start())
            else:
                other_positions.append(m.start())
    return target_positions, other_positions


def _keyword_spans(sentence: str, keywords: list[str]) -> list[tuple[int, int]]:
    if not keywords:
        return []
    kw_exact = set(keywords)
    kw_stems = {_light_stem(k) for k in keywords}
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\w+", sentence.lower()):
        tok = m.group(0)
        if tok in kw_exact or _light_stem(tok) in kw_stems:
            spans.append((m.start(), m.end()))
    return spans


def _lifespan_bonus(
    span_start: int, span_end: int, sentence: str, cues: list[str]
) -> float:
    """Deteksi pola biografi (tempat/tgl_lahir – tempat/tgl_wafat) dalam tanda kurung."""
    if not ("lahir" in cues or "wafat" in cues):
        return 0.0
    for left_s, left_e, right_s, right_e in _iter_lifespan_parens(sentence):
        in_left = left_s <= span_start and span_end <= left_e
        in_right = right_s <= span_start and span_end <= right_e
        if in_left:
            return 4.5 if "lahir" in cues else -4.5
        if in_right:
            return 4.5 if "wafat" in cues else -4.5
    return 0.0


def _parenthetical_adjustment(
    span_start: int,
    span_end: int,
    sentence: str,
    cues: list[str],
    answer_type: AnswerType,
) -> float:
    """Beri penalti pada teks tambahan di dalam tanda kurung seperti '(kini SMA ...)'."""
    open_idx = sentence.rfind("(", 0, span_start)
    if open_idx == -1:
        return 0.0
    close_before = sentence.rfind(")", open_idx, span_start)
    if close_before != -1:
        return 0.0
    close_idx = sentence.find(")", span_end)
    paren_content = sentence[
        open_idx + 1 : close_idx if close_idx != -1 else len(sentence)
    ]

    # Jangan penalti kurung biografi lifespan saat pertanyaan tentang lahir/wafat
    if ("lahir" in cues or "wafat" in cues) and answer_type in ("DATE", "LOCATION"):
        for ls, le, rs, re_ in _iter_lifespan_parens(sentence):
            if (ls <= span_start and span_end <= le) or (
                rs <= span_start and span_end <= re_
            ):
                return 0.0

    # Jangan penalti kurung nama lahir saat pertanyaan tentang nama asli
    if (
        "nama_asli" in cues
        and answer_type == "PERSON"
        and PAREN_BIRTHNAME_RE.search(paren_content)
    ):
        return 1.5

    if PAREN_EXPLANATORY_RE.search(paren_content):
        return -6.0
    return -2.5


def _is_self_echo(
    candidate: str,
    span_start: int,
    sentence: str,
    question: str,
    answer_type: AnswerType,
) -> bool:
    """Tolak kandidat yang sekadar mengulang entitas subjek di dalam pertanyaan."""
    if answer_type not in ("PERSON", "LOCATION", "ORGANIZATION"):
        return False

    q_lower = question.lower()
    q_tokens = set(re.findall(r"\w+", q_lower))
    cand_raw_tokens = [t.lower() for t in re.findall(r"\w+", candidate)]
    cand_core = [
        t
        for t in cand_raw_tokens
        if len(t) > 1
        and t not in STOPWORDS
        and t not in STANDALONE_TITLES
        and t not in PERSON_PARTICLES
    ]
    if not cand_core:
        return True

    all_in_question = all(t in q_tokens for t in cand_core)
    if not all_in_question:
        return False

    # Pengecualian presisi untuk LOCATION (mis. "Perjanjian Bungaya" -> "di Bungaya",
    # "Tuanku Imam Bonjol" -> "lahir di Bonjol")
    if answer_type == "LOCATION" and len(cand_core) == 1:
        prefix_in_sent = sentence[max(0, span_start - 8) : span_start].lower()
        has_loc_prep = bool(re.search(r"\b(?:di|ke|dari)\s+$", prefix_in_sent))
        if has_loc_prep:
            tok = cand_core[0]
            m_q = re.search(rf"\b(\w+)\s+{re.escape(tok)}\b", q_lower)
            if m_q and m_q.group(1) in NON_LOCATION_QUESTION_PREFIXES:
                return False

    return True


def _proximity_bonus(
    span_start: int,
    span_end: int,
    sentence: str,
    target_cue_pos: list[int],
    other_cue_pos: list[int],
    kw_spans: list[tuple[int, int]],
) -> float:
    mid = (span_start + span_end) // 2
    bounds = _clause_bounds(sentence)
    cand_clause = _clause_index(bounds, mid)
    score = 0.0

    if target_cue_pos:
        min_dist = min(abs(mid - p) for p in target_cue_pos)
        score += max(0.0, 3.0 - (min_dist / 35.0))
        target_clauses = {_clause_index(bounds, p) for p in target_cue_pos}
        if cand_clause in target_clauses:
            score += 3.0
            if any(
                p <= span_start
                for p in target_cue_pos
                if _clause_index(bounds, p) == cand_clause
            ):
                score += 1.0
        else:
            score -= 2.0
            if other_cue_pos:
                other_clauses = {_clause_index(bounds, p) for p in other_cue_pos}
                if cand_clause in other_clauses:
                    score -= 2.5

    # Hanya hitung kedekatan terhadap keyword yang TIDAK tumpang tindih dengan kandidat sendiri
    external_kw_spans = [
        (ks, ke) for ks, ke in kw_spans if not (max(span_start, ks) < min(span_end, ke))
    ]
    if external_kw_spans:
        min_kw_dist = min(
            min(abs(span_start - ke), abs(ks - span_end))
            for ks, ke in external_kw_spans
        )
        score += max(0.0, 2.2 - (min_kw_dist / 35.0))
        # Jendela <= 6 mencakup preposisi pendek (" di ", " ke ", " dari ")
        if min_kw_dist <= 6:
            score += 2.8
        kw_clauses = {
            _clause_index(bounds, (ks + ke) // 2) for ks, ke in external_kw_spans
        }
        if cand_clause in kw_clauses:
            score += 1.2

    return score


def _ner_conflict_type(
    span_start: int,
    span_end: int,
    ner_entities: tuple[NEREntity, ...],
    expected_type: AnswerType,
    min_score: float = 0.80,
) -> bool:
    """Kembalikan True jika model NER lokal sangat yakin span ini bertipe entitas lain."""
    for ent in ner_entities:
        if ent.score < min_score:
            continue
        if (
            ent.answer_type in ("PERSON", "LOCATION", "ORGANIZATION")
            and ent.answer_type != expected_type
        ):
            overlap = min(span_end, ent.end) - max(span_start, ent.start)
            span_len = max(span_end - span_start, 1)
            if overlap / span_len >= 0.75:
                return True
    return False


def _merge_candidate_sources(
    primary: list[tuple[str, int, int, float]],
    secondary: list[tuple[str, int, int, float]],
    consensus_bonus: float = 1.4,
) -> list[tuple[str, int, int, float]]:
    """Gabungkan kandidat NER lokal dan Regex; beri bonus konsensus jika keduanya sepakat."""
    merged: dict[tuple[int, int], tuple[str, int, int, float]] = {}

    for text, s, e, score in primary:
        prev = merged.get((s, e))
        if prev is None or score > prev[3]:
            merged[(s, e)] = (text, s, e, score)

    for text, s, e, score in secondary:
        if (s, e) in merged:
            old_text, _, _, old_score = merged[(s, e)]
            merged[(s, e)] = (old_text, s, e, max(old_score, score) + consensus_bonus)
        else:
            # Cek apakah overlap substansial dengan kandidat primary
            matched_key = None
            for ps, pe in merged:
                overlap = min(e, pe) - max(s, ps)
                if overlap > 0 and overlap / max(min(e - s, pe - ps), 1) >= 0.8:
                    matched_key = (ps, pe)
                    break
            if matched_key is not None:
                old_text, ps, pe, old_score = merged[matched_key]
                # Pertahankan span yang lebih lengkap (mis. dengan gelar atau prefix)
                if len(text) > len(old_text) and s <= ps and e >= pe:
                    del merged[matched_key]
                    merged[(s, e)] = (
                        text,
                        s,
                        e,
                        max(old_score, score) + consensus_bonus,
                    )
                else:
                    merged[matched_key] = (
                        old_text,
                        ps,
                        pe,
                        old_score + (consensus_bonus * 0.6),
                    )
            else:
                merged[(s, e)] = (text, s, e, score)

    return list(merged.values())


def _extract_date_candidates(
    sentence: str,
    ner_entities: tuple[NEREntity, ...] = (),
) -> list[tuple[str, int, int, float]]:
    cands: list[tuple[str, int, int, float]] = []
    used_ranges: list[tuple[int, int]] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(max(s, us) < min(e, ue) for us, ue in used_ranges)

    for pat, base_score in (
        (FULL_DATE_RE, 3.6),
        (MONTH_YEAR_RE, 3.1),
        (DAY_MONTH_RE, 2.7),
        (NUMERIC_DATE_RE, 2.7),
        (CENTURY_RE, 2.5),
        (YEAR_RE, 2.5),
    ):
        for m in pat.finditer(sentence):
            s, e = m.span()
            if _overlaps(s, e):
                continue
            val = sentence[s:e].strip(" ,.;:()")
            if val:
                real_s = sentence.find(val, s, e)
                real_e = real_s + len(val)
                used_ranges.append((real_s, real_e))
                # Beri sedikit bonus jika NER lokal juga melabelinya DATE
                ner_hit = any(
                    ent.answer_type == "DATE"
                    and min(real_e, ent.end) > max(real_s, ent.start)
                    for ent in ner_entities
                )
                cands.append(
                    (val, real_s, real_e, base_score + (0.6 if ner_hit else 0.0))
                )
    return cands


def _extract_location_candidates(
    sentence: str,
    question: str,
    q_keywords: list[str],
    ner_entities: tuple[NEREntity, ...] = (),
) -> list[tuple[str, int, int, float]]:
    kw_set = set(q_keywords)
    ner_cands: list[tuple[str, int, int, float]] = []
    regex_cands: list[tuple[str, int, int, float]] = []
    seen_regex: set[tuple[int, int]] = set()

    # 1. Kandidat dari model NER lokal (LOC / GPE / FAC)
    for ent in ner_entities:
        if ent.answer_type != "LOCATION":
            continue
        val = ent.text.strip(" ,.;:()")
        if not val:
            continue
        real_s = sentence.find(val, ent.start, ent.end)
        if real_s < 0:
            continue
        real_e = real_s + len(val)
        if _is_self_echo(val, real_s, sentence, question, "LOCATION"):
            continue
        if check_answer_type(val, "LOCATION") is False:
            continue
        base = 4.0 * ent.score + 0.4
        prefix_ctx = sentence[max(0, real_s - 6) : real_s].lower()
        if re.search(r"\b(?:di|ke|dari)\s+$", prefix_ctx):
            base += 0.9
        ner_cands.append((val, real_s, real_e, base))

    # 2. Kandidat pelengkap berbasis Regex
    for pat, base_score in (
        (LOC_PREP_RE, 3.6),
        (LOC_PREFIX_RE, 3.1),
        (KNOWN_PLACES_RE, 2.8),
    ):
        for m in pat.finditer(sentence):
            grp = 1 if m.lastindex else 0
            s, e = m.span(grp)
            val = sentence[s:e].strip(" ,.;:()")
            if not val:
                continue
            real_s = sentence.find(val, s, e)
            if real_s < 0:
                continue
            real_e = real_s + len(val)
            if (real_s, real_e) in seen_regex:
                continue
            if _is_self_echo(val, real_s, sentence, question, "LOCATION"):
                continue
            if check_answer_type(val, "LOCATION") is False:
                continue
            if (
                not LOCATION_KEYWORDS_RE.search(val)
                and not KNOWN_PLACES_RE.search(val)
                and _ner_conflict_type(
                    real_s, real_e, ner_entities, "LOCATION", min_score=0.82
                )
            ):
                continue
            cand_tokens = [t for t in re.findall(r"\w+", val.lower()) if len(t) > 2]
            overlap = sum(1 for t in cand_tokens if t in kw_set)
            penalty = 2.0 * (overlap / max(len(cand_tokens), 1)) if cand_tokens else 0.0
            seen_regex.add((real_s, real_e))
            regex_cands.append((val, real_s, real_e, base_score - penalty))

    return _merge_candidate_sources(ner_cands, regex_cands)


def _expand_person_title_left(
    sentence: str, start: int, end: int
) -> tuple[str, int, int]:
    """Perluas entitas PERSON dari NER ke kiri jika didahului gelar (mis. K.H., Pangeran, Sultan)."""
    prefix = sentence[max(0, start - 32) : start]
    m = LEADING_TITLE_RE.search(prefix)
    if not m:
        return sentence[start:end], start, end
    new_start = start - len(m.group(0))
    val = sentence[new_start:end].strip(" ,.;:()")
    real_s = sentence.find(val, new_start, end)
    if real_s < 0:
        return sentence[start:end], start, end
    return val, real_s, real_s + len(val)


def _extract_person_candidates(
    sentence: str,
    question: str,
    q_keywords: list[str],
    ner_entities: tuple[NEREntity, ...] = (),
) -> list[tuple[str, int, int, float]]:
    kw_set = set(q_keywords)
    ner_cands: list[tuple[str, int, int, float]] = []
    regex_cands: list[tuple[str, int, int, float]] = []
    seen_regex: set[tuple[int, int]] = set()

    # 1. Kandidat dari model NER lokal (PER)
    for ent in ner_entities:
        if ent.answer_type != "PERSON":
            continue
        val, real_s, real_e = _expand_person_title_left(sentence, ent.start, ent.end)
        val = val.strip(" ,.;:()")
        if not val:
            continue
        real_s = sentence.find(val, real_s, real_e)
        if real_s < 0:
            continue
        real_e = real_s + len(val)
        words_val = val.split()
        if (
            len(words_val) == 1
            and words_val[0].rstrip(".").lower() in STANDALONE_TITLES
        ):
            continue
        if words_val[0].rstrip(".").lower() in NON_PERSON_LEADERS:
            continue
        if _is_self_echo(val, real_s, sentence, question, "PERSON"):
            continue
        if check_answer_type(val, "PERSON") is False:
            continue
        ner_cands.append((val, real_s, real_e, 4.3 * ent.score + 0.5))

    # 2. Kandidat pelengkap berbasis Regex
    for pat, base_score in (
        (TITLED_PERSON_RE, 4.1),
        (MULTI_PERSON_RE, 2.7),
        (SINGLE_PERSON_RE, 1.5),
    ):
        for m in pat.finditer(sentence):
            s, e = m.span()
            val = sentence[s:e].strip(" ,.;:()")
            if not val:
                continue
            real_s = sentence.find(val, s, e)
            if real_s < 0:
                continue
            real_e = real_s + len(val)
            if (real_s, real_e) in seen_regex:
                continue
            if any(
                real_s >= os and real_e <= oe and (real_s, real_e) != (os, oe)
                for os, oe in seen_regex
            ):
                continue
            words_val = val.split()
            if (
                len(words_val) == 1
                and words_val[0].rstrip(".").lower() in STANDALONE_TITLES
            ):
                continue
            if words_val[0].rstrip(".").lower() in NON_PERSON_LEADERS:
                continue
            prefix_ctx = sentence[max(0, real_s - 6) : real_s].lower()
            if re.search(r"\b(?:di|ke|dari)\s+$", prefix_ctx):
                continue
            if _is_self_echo(val, real_s, sentence, question, "PERSON"):
                continue
            if check_answer_type(val, "PERSON") is False:
                continue
            if _ner_conflict_type(
                real_s, real_e, ner_entities, "PERSON", min_score=0.75
            ):
                continue
            cand_tokens = [
                t
                for t in re.findall(r"\w+", val.lower())
                if len(t) > 2 and t not in STANDALONE_TITLES
            ]
            overlap = sum(1 for t in cand_tokens if t in kw_set)
            penalty = 2.5 * (overlap / max(len(cand_tokens), 1)) if cand_tokens else 0.0
            seen_regex.add((real_s, real_e))
            regex_cands.append((val, real_s, real_e, base_score - penalty))

    return _merge_candidate_sources(ner_cands, regex_cands)


def _extract_org_candidates(
    sentence: str,
    question: str,
    q_keywords: list[str],
    ner_entities: tuple[NEREntity, ...] = (),
) -> list[tuple[str, int, int, float]]:
    kw_set = set(q_keywords)
    ner_cands: list[tuple[str, int, int, float]] = []
    regex_cands: list[tuple[str, int, int, float]] = []
    seen_regex: set[tuple[int, int]] = set()

    # 1. Kandidat dari model NER lokal (ORG / NOR)
    for ent in ner_entities:
        if ent.answer_type != "ORGANIZATION":
            continue
        val = ent.text.strip(" ,.;:()")
        if not val or ROMAN_NUMERAL_RE.match(val):
            continue
        real_s = sentence.find(val, ent.start, ent.end)
        if real_s < 0:
            continue
        real_e = real_s + len(val)
        if _is_self_echo(val, real_s, sentence, question, "ORGANIZATION"):
            continue
        if check_answer_type(val, "ORGANIZATION") is False:
            continue
        ner_cands.append((val, real_s, real_e, 4.1 * ent.score + 0.4))

    # 2. Kandidat pelengkap berbasis Regex
    for pat, base_score in (
        (ORG_PHRASE_RE, 3.8),
        (KNOWN_ORGS_RE, 3.9),
        (ACRONYM_ORG_RE, 2.5),
    ):
        for m in pat.finditer(sentence):
            s, e = m.span()
            val = sentence[s:e].strip(" ,.;:()")
            if not val or ROMAN_NUMERAL_RE.match(val):
                continue
            real_s = sentence.find(val, s, e)
            if real_s < 0:
                continue
            real_e = real_s + len(val)
            if (real_s, real_e) in seen_regex:
                continue
            if _is_self_echo(val, real_s, sentence, question, "ORGANIZATION"):
                continue
            if check_answer_type(val, "ORGANIZATION") is False:
                continue
            if (
                not ORG_KEYWORDS_RE.search(val)
                and not KNOWN_ORGS_RE.search(val)
                and _ner_conflict_type(
                    real_s, real_e, ner_entities, "ORGANIZATION", min_score=0.80
                )
            ):
                continue
            cand_tokens = [t for t in re.findall(r"\w+", val.lower()) if len(t) > 1]
            overlap = sum(1 for t in cand_tokens if t in kw_set)
            penalty = 2.0 * (overlap / max(len(cand_tokens), 1)) if cand_tokens else 0.0
            seen_regex.add((real_s, real_e))
            regex_cands.append((val, real_s, real_e, base_score - penalty))

    return _merge_candidate_sources(ner_cands, regex_cands)


def _extract_number_candidates(
    sentence: str,
    ner_entities: tuple[NEREntity, ...] = (),
) -> list[tuple[str, int, int, float]]:
    cands: list[tuple[str, int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    date_spans = [
        (m.start(), m.end())
        for pat in (FULL_DATE_RE, MONTH_YEAR_RE, DAY_MONTH_RE, NUMERIC_DATE_RE)
        for m in pat.finditer(sentence)
    ]
    year_context_spans = [
        (m.start(), m.end())
        for m in re.finditer(r"\btahun\s+\d{3,4}\b", sentence, re.IGNORECASE)
    ]

    def _in_date(s: int, e: int) -> bool:
        return any(ds <= s and e <= de for ds, de in date_spans + year_context_spans)

    for pat, base_score in ((NUMBER_UNIT_RE, 3.6), (STANDALONE_NUM_RE, 2.0)):
        for m in pat.finditer(sentence):
            s, e = m.span()
            if _in_date(s, e):
                continue
            val = sentence[s:e].strip(" ,.;:()")
            if not val:
                continue
            real_s = sentence.find(val, s, e)
            if real_s < 0:
                continue
            real_e = real_s + len(val)
            if (real_s, real_e) in seen:
                continue
            if check_answer_type(val, "NUMBER") is False:
                continue
            ner_hit = any(
                ent.answer_type == "NUMBER"
                and min(real_e, ent.end) > max(real_s, ent.start)
                for ent in ner_entities
            )
            seen.add((real_s, real_e))
            cands.append((val, real_s, real_e, base_score + (0.5 if ner_hit else 0.0)))
    return cands


def _extract_fact_candidate(
    question: str,
    sentence: str,
    ner_entities: tuple[NEREntity, ...] = (),
) -> str:
    """Ekstrak frasa inti untuk pertanyaan FACT (perjanjian, pangkat, penyakit, karya, dll.)."""
    m_head = FACT_HEAD_NOUN_RE.search(question.strip())
    if m_head:
        head = m_head.group(1).lower()

        # 1. Cek entitas NER kategori FACT (EVT/LAW/WOA/PRD) yang memuat head noun
        for ent in ner_entities:
            if (
                ent.answer_type == "FACT"
                and head in ent.text.lower()
                and ent.text in sentence
            ):
                return ent.text

        # 2. Pola Proper Noun (case-sensitive setelah head agar tidak menyedot "pada tanggal ...")
        m_proper = re.search(
            rf"\b(?i:{re.escape(head)})\s+"
            r"([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9'-]*(?:\s+(?:(?:van|de|der|den|of|al-)\s+)?[A-ZÀ-Ý][A-Za-zÀ-ÿ0-9'-]*){0,4})",
            sentence,
        )
        if m_proper:
            if head in ("pangkat", "gelar", "judul"):
                phrase = m_proper.group(1).strip(" ,.;:()")
            else:
                phrase = m_proper.group(0).strip(" ,.;:()")
            if 2 <= len(phrase) <= 80 and phrase in sentence:
                return phrase

        # 3. Pola frasa benda non-kapital (mis. "penyakit ari-ari")
        m_common = re.search(
            rf"\b(?i:{re.escape(head)})\s+([a-zà-ÿ0-9-]+(?:\s+[a-zà-ÿ0-9-]+){{0,4}})",
            sentence,
        )
        if m_common:
            raw_phrase = FACT_STOP_BOUNDARY_RE.sub("", m_common.group(0)).strip(
                " ,.;:()"
            )
            if (
                len(raw_phrase.split()) >= 2
                and 2 <= len(raw_phrase) <= 80
                and raw_phrase in sentence
            ):
                return raw_phrase

    m_cop = FACT_COPULA_RE.search(sentence)
    if m_cop:
        phrase = m_cop.group(1).strip(" ,.;:()")
        if 2 <= len(phrase) <= 120 and phrase in sentence:
            return phrase
    return sentence


def extract_answer(
    question: str, q_info: QuestionInfo, passages: list[RetrievedPassage]
) -> Answer:
    """Gunakan NER lokal, Regex, tipe jawaban, dan petunjuk relasi untuk memilih jawaban."""
    if not passages:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": "Tidak ada passage yang ditemukan oleh retriever.",
        }

    answer_type: AnswerType = q_info.get("answer_type", "OTHER")
    keywords: list[str] = list(q_info.get("keywords", []))
    relation_cues: list[str] = _enrich_relation_cues(
        question, list(q_info.get("relation_cues", []))
    )

    if not keywords and question and question.strip():
        keywords = [
            t
            for t in re.findall(r"\w+", question.lower())
            if t not in QUESTION_WORDS and t not in STOPWORDS
        ]

    # 1. Validasi subjek utama (focus_terms) PER PASSAGE agar tidak mencampur tokoh berbeda
    focus_terms = _extract_focus_terms(question, keywords)
    eligible_passages: list[tuple[int, RetrievedPassage]] = []
    for p_idx, p in enumerate(passages):
        p_text = p.get("text", "")
        if not p_text or not p_text.strip():
            continue
        if _passage_matches_focus(p, focus_terms):
            eligible_passages.append((p_idx, p))

    if not eligible_passages:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": "Subjek atau entitas utama pertanyaan tidak ditemukan dalam passage terambil.",
        }

    # 2. Validasi petunjuk relasi ketat pada passage yang lolos filter subjek
    strict_cues_in_q = [c for c in relation_cues if c in STRICT_BIO_CUES]
    if strict_cues_in_q:
        cue_matched_passages = [
            (p_idx, p)
            for p_idx, p in eligible_passages
            if _passage_has_strict_cue(p.get("text", ""), strict_cues_in_q)
        ]
        if not cue_matched_passages:
            return {
                "status": "not_found",
                "answer": None,
                "passage_id": None,
                "evidence": None,
                "note": f"Petunjuk relasi ({', '.join(strict_cues_in_q)}) tidak ditemukan dalam passage terambil.",
            }
        eligible_passages = cue_matched_passages

    fact_head: str | None = None
    if answer_type == "FACT":
        m_head = FACT_HEAD_NOUN_RE.search(question.strip())
        if m_head:
            fact_head = m_head.group(1).lower()

    # 3. Peringkat kalimat kandidat di dalam passage yang lolos filter
    max_p_score = max(
        (float(p.get("score", 0.0)) for _, p in eligible_passages), default=0.0
    )
    scored_sentences: list[tuple[float, int, str]] = []

    for p_idx, passage in eligible_passages:
        text = passage.get("text", "")
        p_score = float(passage.get("score", 0.0))
        norm_p_score = (p_score / max_p_score) if max_p_score > 0 else 0.0

        for sentence, _, _ in _split_sentences(text):
            sent_tokens = re.findall(r"\w+", sentence.lower())
            if not sent_tokens:
                continue
            sent_exact = set(sent_tokens)
            sent_stems = {_light_stem(t) for t in sent_tokens}

            matched_kws = [
                k for k in keywords if k in sent_exact or _light_stem(k) in sent_stems
            ]
            kw_overlap = len(matched_kws)
            kw_ratio = kw_overlap / max(len(keywords), 1)

            matched_cues = [
                c for c in relation_cues if _find_valid_cue_matches(sentence, c)
            ]
            has_lifespan_cue = bool(
                any(c in ("lahir", "wafat") for c in relation_cues)
                and _iter_lifespan_parens(sentence)
            )
            cue_bonus = 2.8 * len(matched_cues) + (
                2.8 if has_lifespan_cue and not matched_cues else 0.0
            )

            if kw_overlap == 0 and not matched_cues and not has_lifespan_cue:
                continue

            # Jika ada strict cue pada pertanyaan, utamakan kalimat yang memang memuat cue tersebut
            strict_sent_penalty = 0.0
            if strict_cues_in_q and not (
                any(c in matched_cues for c in strict_cues_in_q) or has_lifespan_cue
            ):
                strict_sent_penalty = -3.0

            pronoun_bonus = (
                1.4 if (focus_terms and sent_exact & {"ia", "dia", "beliau"}) else 0.0
            )

            type_bonus = 0.0
            if answer_type == "DATE" and _extract_date_candidates(sentence):
                type_bonus = 2.0
            elif answer_type == "NUMBER" and _extract_number_candidates(sentence):
                type_bonus = 2.0
            elif answer_type == "FACT" and fact_head and fact_head in sent_exact:
                type_bonus = 3.0

            total_score = (
                (2.0 * kw_overlap)
                + (1.5 * kw_ratio)
                + cue_bonus
                + strict_sent_penalty
                + pronoun_bonus
                + type_bonus
                + (0.5 * norm_p_score)
                - (0.05 * p_idx)
            )
            scored_sentences.append((total_score, p_idx, sentence))

    if not scored_sentences:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": "Tidak ditemukan kalimat bukti yang relevan dengan pertanyaan.",
        }

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    # 4. Ekstraksi kandidat entitas dengan model NER lokal + Regex pada top-6 kalimat terbaik
    if answer_type in ("DATE", "LOCATION", "PERSON", "ORGANIZATION", "NUMBER"):
        best_score = float("-inf")
        best_overall: tuple[int, str, str] | None = None

        for sent_score, p_idx, sentence in scored_sentences[:6]:
            ner_entities = extract_ner_entities(sentence)
            target_cue_pos, other_cue_pos = _cue_positions(sentence, relation_cues)
            kw_spans = _keyword_spans(sentence, keywords)

            if answer_type == "DATE":
                raw_cands = _extract_date_candidates(sentence, ner_entities)
            elif answer_type == "LOCATION":
                raw_cands = _extract_location_candidates(
                    sentence, question, keywords, ner_entities
                )
            elif answer_type == "PERSON":
                raw_cands = _extract_person_candidates(
                    sentence, question, keywords, ner_entities
                )
            elif answer_type == "ORGANIZATION":
                raw_cands = _extract_org_candidates(
                    sentence, question, keywords, ner_entities
                )
            else:
                raw_cands = _extract_number_candidates(sentence, ner_entities)

            for cand_text, c_start, c_end, base_score in raw_cands:
                if cand_text not in sentence:
                    continue
                prox = _proximity_bonus(
                    c_start, c_end, sentence, target_cue_pos, other_cue_pos, kw_spans
                )
                life_bonus = (
                    _lifespan_bonus(c_start, c_end, sentence, relation_cues)
                    if answer_type in ("DATE", "LOCATION")
                    else 0.0
                )
                paren_adj = _parenthetical_adjustment(
                    c_start, c_end, sentence, relation_cues, answer_type
                )

                # Bonus jika kandidat langsung diawali preposisi sesudah kata kunci kerja (mis. "bersekolah di Kweekschool")
                prep_verb_bonus = 0.0
                if answer_type in ("LOCATION", "ORGANIZATION"):
                    before_ctx = sentence[max(0, c_start - 6) : c_start].lower()
                    if re.search(r"\b(?:di|ke)\s+$", before_ctx):
                        prep_verb_bonus = 1.0

                cand_score = (
                    sent_score
                    + base_score
                    + prox
                    + life_bonus
                    + paren_adj
                    + prep_verb_bonus
                )
                if cand_score > best_score:
                    best_score = cand_score
                    best_overall = (p_idx, cand_text, sentence)

        if best_overall is not None:
            p_idx, cand_text, sentence = best_overall
            passage_id = passages[p_idx]["passage_id"]
            type_check = check_answer_type(cand_text, answer_type)
            if type_check is True:
                return {
                    "status": "answered",
                    "answer": cand_text,
                    "passage_id": passage_id,
                    "evidence": sentence,
                    "note": None,
                }
            return {
                "status": "needs_review",
                "answer": cand_text,
                "passage_id": passage_id,
                "evidence": sentence,
                "note": f"Kandidat bertipe {answer_type} memerlukan tinjauan.",
            }

        # Jika strict cue aktif atau tidak ada kandidat entitas sama sekali, jangan paksa kalimat penuh
        if strict_cues_in_q:
            return {
                "status": "not_found",
                "answer": None,
                "passage_id": None,
                "evidence": None,
                "note": f"Kandidat valid bertipe {answer_type} tidak ditemukan pada passage terambil.",
            }

        _, best_p_idx, best_sentence = scored_sentences[0]
        return {
            "status": "needs_review",
            "answer": best_sentence,
            "passage_id": passages[best_p_idx]["passage_id"],
            "evidence": best_sentence,
            "note": f"Entitas spesifik bertipe {answer_type} tidak terisolasi; menampilkan kalimat bukti.",
        }

    # 5. Ekstraksi untuk tipe FACT dan OTHER
    _, best_p_idx, best_sentence = scored_sentences[0]
    passage_id = passages[best_p_idx]["passage_id"]
    if answer_type == "FACT":
        ner_entities = extract_ner_entities(best_sentence)
        extracted = _extract_fact_candidate(question, best_sentence, ner_entities)
        return {
            "status": "needs_review",
            "answer": extracted,
            "passage_id": passage_id,
            "evidence": best_sentence,
            "note": "Jawaban tipe FACT memerlukan tinjauan.",
        }

    return {
        "status": "needs_review",
        "answer": best_sentence,
        "passage_id": passage_id,
        "evidence": best_sentence,
        "note": "Jawaban tipe OTHER memerlukan tinjauan.",
    }
