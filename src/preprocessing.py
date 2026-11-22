"""Segmentasi korpus dan normalisasi ringan untuk salinan teks retrieval (PBA02)."""


def split_paragraphs(text: str) -> list[str]:
    """Pisahkan paragraf pada baris kosong. Simpan spasi dan baris di dalam paragraf."""
    paragraphs = []
    paragraph_lines = []
    lines = text.splitlines(keepends=True)
    lines.append("")  # Tutup juga paragraf terakhir pada file tanpa baris kosong di akhir.
    for line in lines:
        if line.strip():
            paragraph_lines.append(line)
        elif paragraph_lines:
            paragraphs.append("".join(paragraph_lines).rstrip("\r\n"))
            paragraph_lines.clear()
    return paragraphs


def preprocess(text: str) -> str:
    """Rapatkan whitespace. Pertahankan kapitalisasi, angka, tanda baca dan kata."""
    return " ".join(text.split())
