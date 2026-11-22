# TP PBA 2026 – Question Answering Bahasa Indonesia

Kelompok kami membandingkan dua cara menjawab pertanyaan bahasa Indonesia dari biografi pahlawan nasional. QA klasik mencari passage dengan TF-IDF lalu mengambil jawaban dari teksnya. RAG mencari passage dengan embedding lalu meminta LLM menjawab berdasarkan bukti itu. Data dan kerangka terminal sudah bisa dicoba. Modul pencarian dan penjawab masih menunggu implementasi Anggota 2–5.

## Mulai dari sini

Jalankan perintah berikut dari folder utama repo, tempat `README.md` dan `src/` berada. Proyek sudah diuji dengan Python 3.12.4. Untuk tahap ini, Python saja cukup. `requirements.txt` berisi komentar karena semua impor aktif memakai pustaka bawaan. Saat modul teman-teman mulai memakai paket tambahan, catat versi yang sudah dicoba di sana dan instal dengan `python -m pip install -r requirements.txt`.

```shell
python -m src.prepare_data
python -B -m unittest discover -s tests -v
```

Perintah pertama membaca TXT dan metadata lalu membuat ulang dua CSV di `data/processed/`. Perintah kedua menjalankan tiga kelompok tes. `test_data.py` mengecek isi berkas, ID, bukti DEV dan template. `test_preprocessing.py` menjaga agar pemisahan paragraf dan perapian spasi sesuai harapan. `test_integration.py` memakai modul pengganti selama tes untuk mengecek urutan panggilan, format hasil dan pesan kesalahan.

Untuk mencoba pertanyaan yang sama di kedua mode, jalankan perintah di bawah. Hapus opsi `--question` kalau ingin mengetik langsung di terminal.

```shell
python -m src.main --mode classic --question "Siapa ayah Martha Christina Tiahahu?"
python -m src.main --mode rag --question "Siapa ayah Martha Christina Tiahahu?"
```

Saat ini keduanya memberi pesan bahwa `analyze_question` belum tersedia (kode keluar 3). Setelah modul tim siap, perintah yang sama akan menampilkan jawaban, kutipan, passage, dan URL sumber.

## Isi repo dan alur data

```text
data/
  corpus/DOC001.txt … DOC015.txt   Teks 15 tokoh, satu tokoh per file
  sources.csv                      Judul, nama file, tautan revisi, dan lisensi
  qa_dev.csv                       10 pertanyaan untuk pengembangan
  qa_test_template.csv             20 ID final, kolom lainnya masih kosong
  processed/passages.csv           75 passage yang dibaca kedua sistem
  processed/qa_dev.csv             Salinan DEV dengan kolom evidence_passage_id
  README.md                        Asal teks dan cara membaca dataset
  SHA256SUMS                       Acuan tes untuk mendeteksi perubahan pada 15 TXT dan 3 CSV sumber
src/                               Penyiapan data dan modul QA
tests/                             Tes otomatis data dan kode
results/                           Tabel evaluasi setelah pengujian tim
laporan/                           Laporan dan pembagian kontribusi
```

Mutiara menyusun sendiri paket dataset biografi. Teks biografi dikutip dari kontributor Wikipedia bahasa Indonesia dengan lisensi CC BY-SA 4.0. Tautan revisi tiap artikel ada di metadata. [Asal teks](data/README.md) · [Sumber tiap tokoh](data/sources.csv)

**PBA02.** Di `src/preprocessing.py`, `split_paragraphs()` memisahkan teks pada baris kosong. Satu paragraf menjadi satu passage. Spasi di dalam paragraf, huruf kapital, angka dan tanda baca tetap tersedia untuk NER, RAG dan kutipan. Kalau butuh spasi yang seragam untuk pencarian, pakai `preprocess()` pada salinan teks.

**Contoh.** DEV001 merujuk dokumen `DOC001`, paragraf 4. Penyiapan data membentuk ID `DOC001_P04`, mengecek kutipannya, lalu menyimpan ID itu di DEV olahan. Korpus berisi `passage_id, doc_id, title, text, source`. File yang disebut `corpus.csv` di rencana awal sekarang tersedia sebagai `data/processed/passages.csv`. Kedua retriever bisa membacanya dengan `load_corpus()`.

## Siapa mengerjakan file apa?

File Python di tabel sudah ada. Tiap orang tinggal mengisi modul bagiannya. Rincian lengkap ada di bagian 5 rencana kelompok, lalu bagian 8–10 untuk jadwal, laporan, dan evaluasi. [Rencana final lima anggota](TP_PBA_2026_Rencana_Final_5_Anggota.md)

| Anggota               | File yang dipegang                                                                                                                                                   | Yang perlu dikerjakan                                                                                                                                                                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 - Mutiara** | `data/`, `src/prepare_data.py`, `src/preprocessing.py`, `src/contracts.py`, `src/main.py`, `tests/`, `README.md`, `requirements.txt`, `.gitignore` | Siapkan data, jaga ID dan kutipan, sepakati format dengan tim, lalu hubungkan kedua sistem. Setelah modul masuk, coba 3–5 pertanyaan DEV yang sama di kedua mode dan simpan contoh jawaban beserta sumber untuk laporan.                                                                                 |
| **2 - Farhan**  | `src/question_processing.py`, `src/retrieve_tfidf.py`                                                                                                            | Isi `analyze_question(question)` untuk tipe jawaban, kata kunci dan petunjuk relasi seperti *lahir*. Isi `retrieve_tfidf(question, k=3)` untuk mengurutkan passage dengan TF-IDF dan cosine similarity. Coba pertanyaan yang pilihan katanya mirip dan berbeda dari teks.                           |
| **3 - Khansa**  | `src/extract_answer.py`, `src/check_answer.py`                                                                                                                   | Isi `extract_answer(question, q_info, passages)`. Gunakan NER untuk nama, lokasi dan organisasi, serta Regex untuk tanggal dan angka. Pilih jawaban dan kutipannya. Coba passage dengan dua tanggal atau nama. Isi `check_answer_type()` untuk dipakai bersama Ammar.                                 |
| **4 - Jepri**   | `src/retrieve_dense.py`, `src/evaluate.py`                                                                                                                       | Isi `retrieve_dense(question, k=3)` dengan embedding multibahasa, simpan vektor korpus, lalu ambil tiga passage teratas. Isi `evaluate_all(qa_test)` dan buat `results/evaluation_results.csv` saat kedua sistem siap. Tim memeriksa benar/salah dan menghitung akurasi masing-masing dari 20 soal. |
| **5 - Ammar**   | `src/rag_generate.py`                                                                                                                                              | Isi `rag_answer(question, passages, q_info)`. Susun prompt dari pertanyaan dan tiga passage Jepri beserta ID. Coba modelnya, cocokkan kutipan, dan gunakan pemeriksaan tipe dari Khansa. Bedakan kasus bukti kurang, jawaban yang perlu ditinjau, dan model gagal dipanggil.                            |

**Bagian bareng.** Tiap orang membuat empat pertanyaan final dan memeriksa bukti milik teman. Saat tim mulai menyusun soal, salin template ke `data/qa_test.csv` dan isi salinannya. Untuk kolaborasi GitHub, kerjakan modul di branch masing-masing, jalankan tes, lalu buka pull request ke `main` untuk ditinjau teman.

**Laporan.** Farhan memegang Bab I–II. Mutiara dan Jepri menyusun Bab III beserta diagram kedua sistem. Semua mengisi Bab IV untuk kodenya sendiri. Jepri, Khansa dan Ammar menyusun Bab V dari hasil nyata. Mutiara mengoordinasikan abstrak, Bab VI, penggabungan `laporan/jawaban.docx` atau PDF, dan catatan kontribusi di `laporan/tanggungjawab.docx`. Migrasi data, kerangka integrasi, dokumentasi proyek dan tes Anggota 1 mendapat bantuan AI. Catat kontribusi serta pustaka yang benar-benar dipakai sesuai arahan dosen.

## Kontrak yang dipakai bersama

Alur PBA14 di `main.py` bisa diikuti dari kiri ke kanan.

- `run_classic(question)` → `analyze_question` → `retrieve_tfidf` → `extract_answer`.
- `run_rag(question)` → `analyze_question` → `retrieve_dense` → `rag_answer`.

Kedua retriever mengembalikan daftar terurut dengan `passage_id, text, score, source`. Kedua mode menghasilkan `question, answer, passage_id, evidence, source, status`, ditambah `passage` dan `note`. Untuk `answered` atau `needs_review`, kutipan harus ada di passage yang dipilih. Untuk `not_found`, jawaban dan rujukannya kosong. Saat evaluasi, Jepri hanya mengirim pertanyaan ke sistem. Jawaban acuan dan bukti berlabel dipakai sesudah prediksi. [Kontrak dan daftar tipe jawaban](src/contracts.py)
