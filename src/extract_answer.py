"""Ekstraksi jawaban untuk Sistem 1 (tanpa LLM)."""

import re
from typing import TypedDict

from .contracts import Answer, AnswerType, QuestionInfo, RetrievedPassage

# Konstanta ambang batas skor keputusan
SCORE_THRESHOLD: float = 3.5
MARGIN_THRESHOLD: float = 0.5

# Pola nama bulan Bahasa Indonesia
_MONTH_RE = (
    r"(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)"
)

# Pola relasi diperluas untuk mencakup bentuk dasar, berimbuhan, dan variasi frasa
EXPANDED_CUES: dict[str, str] = {
    "lahir": r"(?:di|ke)?lahir(?:kan|an)?|kelahiran|lahirnya",
    "wafat": r"wafat|meninggal|gugur|kematian",
    "diri": r"(?:men|di|pen|ber)diri(?:kan)?|berdirinya|pendiri",
    "pimpin": r"(?:di)?pimpin|memimpin|pemimpin|kepemimpinan|pimpinan",
    "asing": r"(?:di|meng|peng)asing(?:kan|an)",
    "tangkap": r"(?:di)?tangkap|menangkap|penangkapan",
    "nikah": r"(?:me|di|per)?nikah(?:i|kan|an)?|menikah\s+dengan",
    "jabat": r"(?:men|di)?jabat(?:an)?|sebagai|menjadi",
    "makam": r"(?:di)?makam(?:kan)?|pemakaman|makam",
    "proklamasi": r"(?:mem|di)?proklamasi(?:kan)?",
    "ayah": r"ayah(?:nya|anda)?|putr[ai]\s+dari|anak\s+dari",
    "ibu": r"ibu(?:nya|nda)?(?! kota)|putr[ai]\s+dari|anak\s+dari",
    "istri": r"istri(?:nya)?|menikah\s+dengan",
    "suami": r"suami(?:nya)?|menikah\s+dengan",
    "anak": r"anak(?:nya)?|putra|putri|dikaruniai",
}

_COMPILED_EXPANDED_CUES = {
    cue: re.compile(rf"\b(?:{pat})\b", re.IGNORECASE) for cue, pat in EXPANDED_CUES.items()
}

_PERSON_TITLES = (
    r"(?:Raden\s+Mas|Raden\s+Adipati|Raden|Pangeran|Sultan|Kapitan|Khatib|Presiden|"
    r"Wakil\s+Presiden|Bupati|Haji|Kyai|Cut|Teuku|Den|Gubernur|Jenderal|Inspektur)"
)

_ORG_PREFIXES = (
    r"(?:Partai|Perhimpunan|Sakola|Sekolah|Kweekschool|Sarekat|Lembaga|Organisasi|Harian|"
    r"Komite|Yayasan|Perguruan(?:\s+Nasional)?)"
)

_LOC_INDICATORS = (
    r"(?:Kampung|Desa|Kota|Wilayah|Karesidenan|Provinsi|Pulau|Kabupaten|Gua|Kecamatan|"
    r"Taman\s+Makam\s+Pahlawan|Jalan)"
)


_NON_PERSON_WORDS = {
    "bahasa", "indonesia", "dinas", "perpustakaan", "kearsipan", "museum",
    "departemen", "kementerian", "pemerintah", "pemerintahan", "kautamaan",
    "istri", "hari", "nasional", "sekolah", "rakyat", "perang", "padri",
    "hindia", "belanda",
}


class Candidate(TypedDict):
    text: str
    passage_id: str
    evidence: str
    score: float
    cues_matched: int
    keywords_matched: int
    char_offset: int


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Pecah teks passage menjadi kalimat dengan menyimpan rentang karakter asli."""
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|\n+|$)", text):
        s = match.start()
        e = match.end()
        raw_sent = text[s:e]
        if raw_sent.strip():
            l_strip = len(raw_sent) - len(raw_sent.lstrip())
            r_strip = len(raw_sent) - len(raw_sent.rstrip())
            act_start = s + l_strip
            act_end = e - r_strip
            if act_end > act_start:
                spans.append((text[act_start:act_end], act_start, act_end))
    return spans


def _is_question_overlap(candidate_text: str, question_keywords: list[str]) -> bool:
    """Periksa apakah seluruh kata pada kandidat berasal dari kata kunci pertanyaan."""
    cand_tokens = set(re.findall(r"\w+", candidate_text.lower()))
    if not cand_tokens:
        return True
    return cand_tokens.issubset(set(question_keywords))


def _extract_candidates_for_type(
    sent_text: str,
    answer_type: AnswerType,
    q_info: QuestionInfo,
) -> list[tuple[str, int]]:
    """Ekstraksi kandidat string beserta offset karakternya di dalam kalimat."""
    results: list[tuple[str, int]] = []
    keywords = q_info.get("keywords", [])

    if answer_type == "DATE":
        # 1. Rentang tanggal atau tanggal lengkap (misal: "3 Juli 1922", "24 – 25 Desember 1921")
        full_date_pat = re.compile(
            rf"\b\d{{1,2}}(?:\s*[–\-]\s*\d{{1,2}})?\s+{_MONTH_RE}\s+\d{{4}}\b",
            re.IGNORECASE,
        )
        for m in full_date_pat.finditer(sent_text):
            results.append((m.group(0), m.start()))

        # 2. Bulan + Tahun (misal: "Oktober 1921")
        month_year_pat = re.compile(rf"\b{_MONTH_RE}\s+\d{{4}}\b", re.IGNORECASE)
        for m in month_year_pat.finditer(sent_text):
            if not any(m.group(0) in r[0] for r in results):
                results.append((m.group(0), m.start()))

        # 3. Tanggal + Bulan (misal: "18 November")
        date_month_pat = re.compile(rf"\b\d{{1,2}}\s+{_MONTH_RE}\b", re.IGNORECASE)
        for m in date_month_pat.finditer(sent_text):
            if not any(m.group(0) in r[0] for r in results):
                results.append((m.group(0), m.start()))

        # 4. Tahun 4 digit tunggal (1600 - 2099)
        year_pat = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2})\b")
        for m in year_pat.finditer(sent_text):
            if not any(m.group(0) in r[0] for r in results):
                results.append((m.group(0), m.start()))

    elif answer_type == "NUMBER":
        # Digit dengan kata satuan
        num_unit_pat = re.compile(
            r"\b\d+\s+(?:tahun|anak(?: perempuan| laki-laki)?|orang|kali|hari|bulan|minggu)\b",
            re.IGNORECASE,
        )
        for m in num_unit_pat.finditer(sent_text):
            results.append((m.group(0), m.start()))

        # Digit angka tunggal (kecualikan tahun kalender 1600-2099)
        num_pat = re.compile(r"\b\d+\b")
        for m in num_pat.finditer(sent_text):
            val = int(m.group(0))
            if not (1600 <= val <= 2099):
                if not any(m.group(0) in r[0] for r in results):
                    results.append((m.group(0), m.start()))

    elif answer_type == "PERSON":
        # Pola nama dengan gelar/sapaan
        titled_pat = re.compile(
            rf"\b(?:{_PERSON_TITLES})\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        )
        for m in titled_pat.finditer(sent_text):
            results.append((m.group(0), m.start()))

        # Pola nama orang umum (2 sampai 4 kata berawalan kapital)
        name_pat = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
        for m in name_pat.finditer(sent_text):
            name = m.group(0)
            words_lower = set(w.lower() for w in name.split())
            if not (words_lower & _NON_PERSON_WORDS):
                if not re.search(rf"\b(?:{_LOC_INDICATORS}|{_ORG_PREFIXES})\b", name):
                    if not any(name in r[0] for r in results):
                        results.append((name, m.start()))

        # Nama tunggal setelah kata relasional khusus (misal: "menggantikan Semaun", "dengan Siti Rahmiati")
        single_pat = re.compile(
            r"\b(?:menggantikan|dengan|bersama|putri\s+dari|putra\s+dari|ayahnya|ibunya)\s+([A-Z][a-z]+)\b"
        )
        for m in single_pat.finditer(sent_text):
            single_name = m.group(1)
            words_lower = set(w.lower() for w in single_name.split())
            if not (words_lower & _NON_PERSON_WORDS):
                if not any(single_name in r[0] for r in results):
                    results.append((single_name, m.start(1)))

    elif answer_type == "LOCATION":
        # Lokasi yang didahului preposisi atau kata penanda tempat
        prep_loc_pat = re.compile(
            rf"\b(?:[dD]i|[kK]e|[dD]ari|[wW]ilayah|[kK]ampung|[dD]esa|markas\s+di|pengasingan\s+ke)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        )
        for m in prep_loc_pat.finditer(sent_text):
            loc_name = m.group(1)
            # Hilangkan kata penutup umum jika terikut
            loc_name = re.sub(r"\s+(?:pada|dan|yang|untuk)$", "", loc_name)
            results.append((loc_name, m.start(1)))

        # Lokasi dengan indikator eksplisit (misal "Kampung Blauran", "Gua Selarong", "Taman Makam Pahlawan Kalibata")
        ind_loc_pat = re.compile(
            rf"\b(?:{_LOC_INDICATORS})\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        )
        for m in ind_loc_pat.finditer(sent_text):
            if not any(m.group(0) in r[0] for r in results):
                results.append((m.group(0), m.start()))

    elif answer_type == "ORGANIZATION":
        # Organisasi dengan awalan resmi (Partai, Perhimpunan, Sakola, Kweekschool, dll.)
        org_pat = re.compile(
            rf"\b(?:{_ORG_PREFIXES})(?:\s+[A-Z][a-z]+)*\b"
        )
        for m in org_pat.finditer(sent_text):
            results.append((m.group(0), m.start()))

        # Akronim kapital (misal: PKI, ELS, PI, PNI)
        acronym_pat = re.compile(r"\b[A-Z]{2,6}\b")
        for m in acronym_pat.finditer(sent_text):
            if not any(m.group(0) in r[0] for r in results):
                results.append((m.group(0), m.start()))

    elif answer_type in ("FACT", "OTHER"):
        # 1. Jabatan setelah kata "sebagai" atau "pangkat"
        job_pat = re.compile(
            r"\b(?:sebagai|pangkat|menjabat)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        )
        for m in job_pat.finditer(sent_text):
            results.append((m.group(1), m.start(1)))

        # 2. Nama peristiwa / perjanjian ("Perjanjian Bungaya")
        event_pat = re.compile(r"\bPerjanjian\s+[A-Z][a-z]+\b")
        for m in event_pat.finditer(sent_text):
            results.append((m.group(0), m.start()))

        # 3. Penyebab ("karena penyakit ari-ari")
        cause_pat = re.compile(r"\bkarena\s+([a-z\s-]+?)(?=[,.]|$)", re.IGNORECASE)
        for m in cause_pat.finditer(sent_text):
            results.append((m.group(1).strip(), m.start(1)))

    # Saring kandidat yang merupakan pengulangan kata kunci pertanyaan
    filtered: list[tuple[str, int]] = []
    for cand_text, offset in results:
        cand_clean = cand_text.strip(" ,.-")
        if cand_clean and not _is_question_overlap(cand_clean, keywords):
            filtered.append((cand_clean, offset))

    return filtered


def _score_candidate(
    cand_text: str,
    cand_offset: int,
    sent_text: str,
    q_info: QuestionInfo,
    passage_score: float,
) -> tuple[float, int, int]:
    """Hitung skor kandidat berdasarkan kecocokan relasi, kata kunci, jarak, dan retrieval."""
    score = 2.0  # Bobot dasar kecocokan tipe
    keywords = q_info.get("keywords", [])
    relation_cues = q_info.get("relation_cues", [])

    sent_lower = sent_text.lower()

    # 1. Kecocokan petunjuk relasi (relation cues)
    cues_matched = 0
    cue_positions: list[int] = []
    for cue in relation_cues:
        pat = _COMPILED_EXPANDED_CUES.get(cue)
        if pat:
            m = pat.search(sent_lower)
            if m:
                cues_matched += 1
                cue_positions.append(m.start())
        elif cue in sent_lower:
            cues_matched += 1
            cue_positions.append(sent_lower.find(cue))

    if cues_matched > 0:
        score += 3.5 * cues_matched
    elif relation_cues:
        # Penalti jika pertanyaan memiliki petunjuk relasi spesifik tetapi kalimat tidak memuatnya
        score -= 2.0

    # 2. Kemunculan kata kunci pertanyaan di kalimat
    keywords_matched = 0
    kw_positions: list[int] = []
    for kw in keywords:
        pos = sent_lower.find(kw.lower())
        if pos != -1:
            keywords_matched += 1
            kw_positions.append(pos)

    score += 1.0 * keywords_matched

    # 3. Kedekatan jarak karakter ke kata relasi atau kata kunci terdekat
    all_target_positions = cue_positions or kw_positions
    if all_target_positions:
        min_dist = min(abs(cand_offset - pos) for pos in all_target_positions)
        score += 2.0 / (1.0 + (min_dist / 25.0))

    # 4. Bonus nama organisasi resmi
    if q_info.get("answer_type") == "ORGANIZATION":
        if re.search(rf"\b(?:{_ORG_PREFIXES})\b", cand_text):
            score += 2.5

    # 5. Tie-breaker kecil dari skor retrieval passage
    score += passage_score * 0.1

    return score, cues_matched, keywords_matched


def extract_answer(
    question: str,
    q_info: QuestionInfo,
    passages: list[RetrievedPassage],
) -> Answer:
    """Ekstraksi jawaban terbaik dari passage hasil retrieval."""
    if not passages or len(passages) < 3:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": "Passage hasil retrieval kosong atau kurang dari 3.",
        }

    answer_type = q_info.get("answer_type", "OTHER")
    candidates: list[Candidate] = []

    # Iterasi tiap passage dan kalimatnya
    for passage in passages:
        p_id = passage["passage_id"]
        p_text = passage["text"]
        p_score = passage.get("score", 0.0)

        sentences = _split_sentences_with_spans(p_text)
        for sent_text, start_idx, end_idx in sentences:
            # Dapatkan bukti (evidence) sebagai substring persis
            exact_evidence = p_text[start_idx:end_idx]

            raw_cands = _extract_candidates_for_type(sent_text, answer_type, q_info)
            for cand_text, offset in raw_cands:
                c_score, c_cues, c_kw = _score_candidate(
                    cand_text, offset, sent_text, q_info, p_score
                )
                candidates.append(
                    {
                        "text": cand_text,
                        "passage_id": p_id,
                        "evidence": exact_evidence,
                        "score": c_score,
                        "cues_matched": c_cues,
                        "keywords_matched": c_kw,
                        "char_offset": offset,
                    }
                )

    if not candidates:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": f"Tidak ditemukan kandidat jawaban untuk tipe {answer_type}.",
        }

    # Deduplikasi kandidat yang sama dengan mengambil skor tertinggi
    dedup: dict[str, Candidate] = {}
    for c in candidates:
        key = c["text"].strip().lower()
        if key not in dedup or c["score"] > dedup[key]["score"]:
            dedup[key] = c

    sorted_candidates = sorted(dedup.values(), key=lambda x: x["score"], reverse=True)
    best = sorted_candidates[0]

    # Evaluasi ambang batas
    if best["score"] < SCORE_THRESHOLD:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": f"Skor kandidat tertinggi ({best['score']:.2f}) di bawah ambang batas.",
        }

    # Jika ada relation_cues tetapi kandidat terbaik tidak cocok dengan relation cue apa pun
    if q_info.get("relation_cues") and best["cues_matched"] == 0:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": "Kandidat tidak didukung oleh petunjuk relasi pertanyaan.",
        }

    # Evaluasi persaingan antar dua kandidat teratas
    if len(sorted_candidates) > 1:
        second = sorted_candidates[1]
        score_diff = best["score"] - second["score"]
        if score_diff < MARGIN_THRESHOLD:
            return {
                "status": "needs_review",
                "answer": best["text"],
                "passage_id": best["passage_id"],
                "evidence": best["evidence"],
                "note": (
                    f"Dua kandidat bersaing ketat: '{best['text']}' ({best['score']:.2f}) "
                    f"vs '{second['text']}' ({second['score']:.2f})."
                ),
            }

    return {
        "status": "answered",
        "answer": best["text"],
        "passage_id": best["passage_id"],
        "evidence": best["evidence"],
        "note": None,
    }
