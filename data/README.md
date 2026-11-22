# Biografi dan Perjuangan Pahlawan Nasional Indonesia

Korpus ekstraktif bahasa Indonesia untuk proyek Pemrosesan Bahasa Alami Universitas Brawijaya tahun 2026. Terdapat 15 biografi, 75 paragraf dan 5.200 kata. Gunakan satu paragraf sebagai satu passage untuk membandingkan QA klasik dan RAG pada bahan bacaan yang sama.

## Isi paket

| Lokasi                                           | Isi                                                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `data/corpus/DOC001.txt` sampai `DOC015.txt` | Satu tokoh per TXT. Paragraf dipisahkan oleh satu baris kosong.                                            |
| `data/sources.csv`                             | Satu artikel Wikipedia bahasa Indonesia per dokumen, dengan tautan permanen dan lisensi.                   |
| `data/qa_dev.csv`                              | Sepuluh pertanyaan pengembangan dengan jawaban pendek dan bukti.                                           |
| `data/qa_test_template.csv`                    | Dua puluh ID untuk evaluasi akhir. Lima anggota mengisi dan memeriksa masing-masing empat pertanyaan baru. |

### Kolom CSV

- **Sumber.** `doc_id` adalah ID tokoh. `title` adalah nama tokoh dalam dataset dan `filename` adalah nama TXT. `source_title` memuat judul artikel, `publisher` memuat nama situs, dan `source_url` membuka versi artikel yang dikutip. `access_date` menggunakan format bulan/hari/tahun. `license` memuat nama lisensi dan tautan ketentuannya.
- **Pengembangan.** `question_id`, `question`, dan `answer_type` memuat ID, pertanyaan, dan tipe jawaban. `ground_truth` berisi jawaban acuan. `evidence_doc_id` dan `evidence_paragraph` menunjuk dokumen dan paragraf, dihitung mulai dari 1. `evidence_text` adalah kalimat pendukung yang terdapat dalam paragraf itu.
- **Evaluasi akhir.** Template menggunakan kolom pengembangan serta `author` untuk nama penyusun. Hanya kolom `question_id` yang terisi.

## Asal teks

Paragraf dipilih dari badan artikel Wikipedia bahasa Indonesia.

## Tokoh dan ukuran dokumen

| ID     | Tokoh                    | Paragraf | Kata |
| ------ | ------------------------ | -------: | ---: |
| DOC001 | Ki Hadjar Dewantara      |        5 |  353 |
| DOC002 | R. A. Kartini            |        5 |  464 |
| DOC003 | Cut Nyak Dhien           |        4 |  250 |
| DOC004 | Mohammad Hatta           |        4 |  335 |
| DOC005 | Soekarno                 |        4 |  407 |
| DOC006 | Pangeran Diponegoro      |        6 |  373 |
| DOC007 | Bung Tomo                |        5 |  269 |
| DOC008 | Pattimura                |        4 |  253 |
| DOC009 | Dewi Sartika             |        4 |  305 |
| DOC010 | H. Agus Salim            |        6 |  349 |
| DOC011 | Sutan Sjahrir            |        6 |  363 |
| DOC012 | Tan Malaka               |        5 |  448 |
| DOC013 | Tuanku Imam Bonjol       |        6 |  298 |
| DOC014 | Sultan Hasanuddin        |        6 |  419 |
| DOC015 | Martha Christina Tiahahu |        5 |  314 |

## Membaca dengan Python

Jalankan contoh dari folder hasil ekstraksi yang berisi `README.md` dan `data/`. TXT menggunakan UTF-8. CSV menggunakan UTF-8 dengan BOM agar karakter terbaca di Excel.

```python
from pathlib import Path
import pandas as pd

data = Path("data")
sumber = pd.read_csv(data / "sources.csv", encoding="utf-8-sig", keep_default_na=False)
dev = pd.read_csv(data / "qa_dev.csv", encoding="utf-8-sig", keep_default_na=False)

soal = dev.iloc[0]
teks = (data / "corpus" / f"{soal.evidence_doc_id}.txt").read_text(encoding="utf-8")
paragraf = teks.strip().split("\n\n")
bukti = paragraf[int(soal.evidence_paragraph) - 1]
assert soal.evidence_text in bukti
```
