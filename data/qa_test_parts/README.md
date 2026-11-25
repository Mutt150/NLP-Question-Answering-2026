# Soal uji per anggota

Tiap anggota membuat SATU berkas CSV di folder ini (misalnya `jepri.csv`) berisi 4 soal baru.
Judul kolom (baris pertama), urutan persis:

```
author,question,answer_type,ground_truth,evidence_doc_id,evidence_paragraph,evidence_text
```

- `answer_type`: PERSON, DATE, LOCATION, NUMBER, ORGANIZATION, FACT, atau OTHER.
- `evidence_doc_id` contoh `DOC004`; `evidence_paragraph` urutan paragraf mulai dari 1.
- `evidence_text` harus kutipan persis dari paragraf itu, dan `ground_truth` harus ada di dalamnya.
- Jangan menyalin soal di `data/qa_dev.csv`. Jangan mengisi `question_id`.

Setelah 5 berkas terkumpul: `python -m src.merge_questions`
