"""Generator RAG berbasis LLM milik Orang 5 (Ammar)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .contracts import Answer, AnswerType, QuestionInfo, RetrievedPassage, UnavailableError

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _REPO_ROOT / ".env"

_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}

_MONTH_NAMES = (
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
)

_NUMBER_WORDS = {
    "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan",
    "sembilan", "sepuluh", "sebelas", "belas", "puluh", "ratus", "ribu", "juta",
    "miliar", "pertama", "kedua", "ketiga", "keempat", "kelima",
}

_SUPPORT_STOPWORDS = {"di", "ke", "dari", "yang", "dan", "atau", "pada", "tahun", "tanggal", "sebagai"}


def _load_dotenv(env_path: Path | None = None) -> dict[str, str]:
    """Baca pasangan KEY=VALUE dari file .env tanpa mengubah os.environ."""
    path = _ENV_PATH if env_path is None else Path(env_path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def _get_config(key: str, dotenv: dict[str, str], default: str | None = None) -> str | None:
    """Ambil konfigurasi dari environment variable dengan fallback file .env."""
    value = os.environ.get(key)
    if value is not None and value.strip():
        return value.strip()
    env_val = dotenv.get(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    return default


def build_prompt(question: str, passages: list[RetrievedPassage], q_info: QuestionInfo) -> str:
    """Susun prompt terstruktur dari pertanyaan, analisis tipe, dan passage hasil retrieval."""
    answer_type = q_info.get("answer_type", "OTHER")
    cues = q_info.get("relation_cues") or []
    cues_text = ", ".join(cues) if cues else "-"
    passages_block = "\n\n".join(
        f"[{p['passage_id']}] {p['text']}" for p in passages
    )
    return (
        "Anda adalah asisten tanya-jawab faktual Bahasa Indonesia untuk korpus biografi pahlawan nasional.\n"
        "Jawab pertanyaan HANYA berdasarkan passage yang diberikan di bawah ini.\n"
        "Jangan gunakan pengetahuan luar dan jangan mengarang fakta.\n\n"
        "Aturan jawaban:\n"
        f"- Tipe jawaban yang diharapkan: {answer_type}\n"
        f"- Petunjuk relasi pertanyaan: {cues_text}\n"
        "- Berikan jawaban yang singkat, padat, dan sesuai tipe jawaban.\n"
        "- Salin kutipan kalimat atau frasa yang PERSIS SAMA (verbatim) dari passage terpilih ke field `evidence`.\n"
        "- Jika passage tidak memuat bukti yang cukup, kembalikan `found`: false serta `answer`, `passage_id`, dan `evidence` bernilai null.\n"
        "- Kembalikan HANYA objek JSON valid tanpa teks tambahan dengan format:\n"
        '{"found": true/false, "answer": "..." atau null, "passage_id": "..." atau null, "evidence": "..." atau null, "reason": "..." atau null}\n\n'
        f"Pertanyaan: {question}\n\n"
        f"Passage Konteks:\n{passages_block}\n"
    )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """Kirim permintaan HTTP POST JSON dan kembalikan respons JSON."""
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise UnavailableError(f"Layanan LLM gagal merespons (HTTP {error.code}).") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise UnavailableError("Gagal menghubungi layanan LLM atau koneksi timeout.") from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise UnavailableError("Respons API LLM bukan JSON yang valid.") from error
    if not isinstance(parsed, dict):
        raise UnavailableError("Format respons API LLM tidak sesuai.")
    return parsed


def _resolve_provider(dotenv: dict[str, str]) -> str:
    """Tentukan penyedia LLM dari RAG_PROVIDER atau deteksi otomatis variabel konfigurasi."""
    explicit = _get_config("RAG_PROVIDER", dotenv)
    if explicit:
        provider = explicit.lower()
        if provider not in ("gemini", "groq", "openai", "ollama"):
            raise UnavailableError(f"RAG_PROVIDER '{explicit}' tidak dikenali.")
        return provider
    if _get_config("GEMINI_API_KEY", dotenv):
        return "gemini"
    if _get_config("GROQ_API_KEY", dotenv):
        return "groq"
    if _get_config("OPENAI_API_KEY", dotenv):
        return "openai"
    if _get_config("OLLAMA_MODEL", dotenv):
        return "ollama"
    raise UnavailableError(
        "Konfigurasi LLM belum diatur. Set GEMINI_API_KEY, GROQ_API_KEY, "
        "OPENAI_API_KEY, atau OLLAMA_MODEL (bisa melalui file .env)."
    )


def call_llm(prompt: str, timeout: float = 30.0) -> str:
    """Panggil LLM tanpa dependensi eksternal melalui REST API penyedia yang dikonfigurasi."""
    dotenv = _load_dotenv()
    provider = _resolve_provider(dotenv)
    timeout_raw = _get_config("RAG_TIMEOUT", dotenv)
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except ValueError as error:
            raise UnavailableError("Nilai RAG_TIMEOUT harus berupa angka.") from error

    model_override = _get_config("RAG_MODEL", dotenv) or _get_config("LLM_MODEL", dotenv)

    if provider == "gemini":
        api_key = _get_config("GEMINI_API_KEY", dotenv)
        if not api_key:
            raise UnavailableError("GEMINI_API_KEY belum diatur.")
        model = model_override or _DEFAULT_MODELS["gemini"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        data = _post_json(url, payload, {"x-goog-api-key": api_key}, timeout)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as error:
            raise UnavailableError("Struktur respons Gemini tidak berisi teks jawaban.") from error

    if provider in ("groq", "openai"):
        if provider == "groq":
            api_key = _get_config("GROQ_API_KEY", dotenv)
            default_url = "https://api.groq.com/openai/v1/chat/completions"
            url = _get_config("GROQ_BASE_URL", dotenv) or _get_config("LLM_BASE_URL", dotenv) or default_url
        else:
            api_key = _get_config("OPENAI_API_KEY", dotenv)
            default_url = "https://api.openai.com/v1/chat/completions"
            url = _get_config("OPENAI_BASE_URL", dotenv) or _get_config("LLM_BASE_URL", dotenv) or default_url
        if not api_key:
            raise UnavailableError(f"{provider.upper()}_API_KEY belum diatur.")
        model = model_override or _DEFAULT_MODELS[provider]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        data = _post_json(url, payload, {"Authorization": f"Bearer {api_key}"}, timeout)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise UnavailableError(f"Struktur respons {provider} tidak berisi teks jawaban.") from error

    # provider == "ollama"
    model = model_override or _get_config("OLLAMA_MODEL", dotenv)
    if not model:
        raise UnavailableError("OLLAMA_MODEL belum diatur.")
    base_url = (_get_config("OLLAMA_BASE_URL", dotenv) or "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    data = _post_json(url, payload, {}, timeout)
    text = data.get("response")
    if not isinstance(text, str) or not text.strip():
        raise UnavailableError("Struktur respons Ollama tidak berisi teks jawaban.")
    return text


def parse_llm_response(raw_text: str) -> dict[str, Any]:
    """Ekstrak dan parse objek JSON dari keluaran mentah LLM."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise UnavailableError("Respons LLM kosong atau bukan JSON yang valid.")
    cleaned = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not obj_match:
            raise UnavailableError("Respons LLM kosong atau bukan JSON yang valid.")
        try:
            parsed = json.loads(obj_match.group(0))
        except json.JSONDecodeError as error:
            raise UnavailableError("Respons LLM kosong atau bukan JSON yang valid.") from error

    if not isinstance(parsed, dict):
        raise UnavailableError("Respons LLM kosong atau bukan JSON yang valid.")
    return parsed


def _match_whitespace_collapsed(evidence: str, text: str) -> str | None:
    """Cari substring persis di passage meski LLM mengubah spasi atau baris baru."""
    tokens = evidence.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, text)
    return match.group(0) if match else None


def _split_sentences_preserve(text: str) -> list[str]:
    """Pecah passage menjadi potongan kalimat yang tetap merupakan substring persis."""
    sentences: list[str] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        chunk = match.group(0).strip()
        if chunk:
            sentences.append(chunk)
    return sentences


def _resolve_passage_and_evidence(
    passages: list[RetrievedPassage],
    passage_id: Any,
    evidence: Any,
    answer: str,
) -> tuple[RetrievedPassage | None, str | None, bool, str | None]:
    """Cocokkan ID passage dan pastikan kutipan bukti ada persis di dalam passage."""
    pid = passage_id.strip() if isinstance(passage_id, str) else ""
    ev = evidence.strip() if isinstance(evidence, str) else ""

    passage = next((p for p in passages if p["passage_id"] == pid), None) if pid else None
    if passage is None and ev:
        passage = next((p for p in passages if ev in p["text"]), None)
    if passage is None:
        return None, None, False, "ID passage dari LLM tidak ada pada hasil retrieval."

    text = passage["text"]
    if ev and ev in text:
        return passage, ev, True, None

    if ev:
        collapsed = _match_whitespace_collapsed(ev, text)
        if collapsed is not None:
            return passage, collapsed, True, None

    answer_lower = answer.lower()
    for sentence in _split_sentences_preserve(text):
        if answer_lower in sentence.lower():
            return (
                passage,
                sentence,
                False,
                "Kutipan LLM tidak persis sama dengan teks passage; kalimat bukti dicocokkan ulang dan perlu ditinjau.",
            )

    return None, None, False, "Kutipan bukti LLM tidak ditemukan pada passage sumber."


def _fallback_check_answer_type(answer: str, answer_type: AnswerType) -> bool | None:
    """Pemeriksaan tipe sederhana saat check_answer.py (Orang 3) belum tersedia."""
    text = answer.strip()
    if not text:
        return False
    lower = text.lower()
    if answer_type == "DATE":
        if re.search(r"\b\d{4}\b", text) or re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", text):
            return True
        if any(month in lower for month in _MONTH_NAMES):
            return True
        return False
    if answer_type == "NUMBER":
        if re.search(r"\d", text):
            return True
        tokens = set(re.findall(r"\w+", lower))
        if tokens & _NUMBER_WORDS:
            return True
        return False
    return True


def _verify_answer_type(answer: str, answer_type: AnswerType) -> tuple[bool | None, str | None]:
    """Gunakan check_answer_type milik Orang 3 dengan fallback bila modul belum siap."""
    try:
        from .check_answer import check_answer_type

        verdict = check_answer_type(answer, answer_type)
    except UnavailableError:
        verdict = _fallback_check_answer_type(answer, answer_type)

    if verdict is False:
        return False, f"Tipe jawaban '{answer}' belum sesuai dengan ekspektasi {answer_type}."
    if verdict is None:
        return None, f"Kesesuaian tipe jawaban dengan {answer_type} perlu ditinjau."
    return True, None


def _check_answer_support(answer: str, evidence: str, passage_text: str) -> bool:
    """Pastikan token inti jawaban muncul di teks passage sumber."""
    tokens = [
        tok for tok in re.findall(r"\w+", answer.lower())
        if tok not in _SUPPORT_STOPWORDS
    ]
    if not tokens:
        return answer.strip().lower() in passage_text.lower()
    haystack = (evidence + " " + passage_text).lower()
    matched = sum(1 for tok in tokens if tok in haystack)
    return (matched / len(tokens)) >= 0.5


def _is_not_found_answer(found_flag: Any, answer_text: str | None) -> bool:
    if found_flag is False:
        return True
    if not answer_text:
        return True
    lowered = answer_text.strip().lower()
    return lowered.startswith("tidak ditemukan") or lowered.startswith("jawaban tidak ditemukan")


def rag_answer(
    question: str,
    passages: list[RetrievedPassage],
    q_info: QuestionInfo,
    *,
    llm_caller: Callable[[str], str] | None = None,
) -> Answer:
    """Terima passage retrieval dan analisis pertanyaan, lalu kembalikan jawaban dan bukti."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Pertanyaan tidak boleh kosong.")
    if not passages:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": "Tidak ada passage hasil retrieval.",
        }

    prompt = build_prompt(question.strip(), passages, q_info)
    caller = call_llm if llm_caller is None else llm_caller
    raw_response = caller(prompt)
    parsed = parse_llm_response(raw_response)

    raw_answer = parsed.get("answer")
    answer_text = raw_answer.strip() if isinstance(raw_answer, str) and raw_answer.strip() else None
    raw_reason = parsed.get("reason")
    reason_note = raw_reason.strip() if isinstance(raw_reason, str) and raw_reason.strip() else None

    if _is_not_found_answer(parsed.get("found"), answer_text):
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": reason_note or "Bukti tidak mencukupi pada passage retrieval.",
        }

    assert answer_text is not None
    passage, evidence, exact_quote, ground_note = _resolve_passage_and_evidence(
        passages, parsed.get("passage_id"), parsed.get("evidence"), answer_text
    )
    if passage is None or evidence is None:
        return {
            "status": "not_found",
            "answer": None,
            "passage_id": None,
            "evidence": None,
            "note": ground_note,
        }

    notes: list[str] = []
    if ground_note:
        notes.append(ground_note)

    answer_type: AnswerType = q_info.get("answer_type", "OTHER")
    type_ok, type_note = _verify_answer_type(answer_text, answer_type)
    if type_note:
        notes.append(type_note)

    supported = _check_answer_support(answer_text, evidence, passage["text"])
    if not supported:
        notes.append("Sebagian kata kunci jawaban tidak ditemukan secara langsung di passage.")

    needs_review = (not exact_quote) or (type_ok is not True) or (not supported)
    return {
        "status": "needs_review" if needs_review else "answered",
        "answer": answer_text,
        "passage_id": passage["passage_id"],
        "evidence": evidence,
        "note": " ".join(notes) if notes else None,
    }
