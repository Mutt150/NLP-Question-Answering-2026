# TP PBA 2026 – Question Answering Bahasa Indonesia

Kelompok kami membandingkan dua cara menjawab pertanyaan bahasa Indonesia berdasarkan biografi 15 pahlawan nasional. Sistem pertama, QA klasik, mencari passage dengan TF-IDF, kemudian mengambil jawaban langsung dari teks. Sistem kedua, RAG, mencari passage dengan embedding, kemudian meminta LLM menjawab berdasarkan passage tersebut. Kedua sistem menerima pertanyaan yang sama sehingga hasilnya dapat dibandingkan.

Per 25 November 2026, analisis pertanyaan, pencarian TF-IDF, pencarian dense, dan program evaluasi telah tersedia. Pengambil jawaban klasik dan generator RAG masih dalam pengerjaan.

## Mulai dari sini

Seluruh perintah dijalankan dari folder utama repo, yaitu folder yang berisi `README.md` dan `src/`. Proyek ini diuji dengan Python 3.12.4. Modul TF-IDF memakai scikit-learn untuk TF-IDF dan cosine similarity serta Sastrawi untuk stemming bahasa Indonesia. Pencarian dense memakai sentence-transformers. Pada penggunaan pertama, pencarian dense mengunduh model E5 berukuran sekitar 470 MB sehingga memerlukan koneksi internet.

```shell
python -m pip install -r requirements.txt
python -m src.prepare_data
python -B -m unittest discover -s tests -v
```

`src.prepare_data` membaca berkas TXT dan metadata, kemudian membuat ulang dua CSV di `data/processed/`. Tes memeriksa data, preprocessing, analisis pertanyaan, retrieval, evaluasi, dan sambungan antarmodul. Hasil `expected failure` menandai contoh yang masih menjadi keterbatasan modul, bukan kesalahan.

Untuk mencoba satu pertanyaan pada kedua mode, jalankan perintah berikut. Hapus opsi `--question` apabila pertanyaan ingin diketik langsung di terminal.

```shell
python -m src.main --mode classic --question "Siapa ayah Martha Christina Tiahahu?"
python -m src.main --mode rag --question "Siapa ayah Martha Christina Tiahahu?"
```

Apabila ada modul yang belum tersedia, terminal menampilkan nama modul tersebut dan keluar dengan kode 3. Setelah seluruh modul tersedia, perintah yang sama menampilkan jawaban, kutipan, passage, dan URL sumber.

Perintah lain yang dapat digunakan:

```shell
# tiga passage teratas dari pencarian dense
python -m src.retrieve_dense Di mana Benteng Sombaopu?

# perbandingan pencarian dense dan TF-IDF pada 25 soal DEV
# (tambahkan --hanya-gagal untuk melihat rincian soal yang buktinya tidak masuk tiga besar)
python -m src.evaluate --retrieval data/qa_dev.csv --retriever both

# passage tiga besar TF-IDF pada tiap soal DEV, hasilnya di results/tfidf_dev.csv
python -m src.experiment_qp_tfidf

# uji coba alur evaluasi pada soal DEV (bukan evaluasi akhir)
python -m src.evaluate --dev
```

Eksperimen pembanding `python -m src.experiment_lemma` memakai cache lema yang sudah tersedia. Untuk menghasilkan lema pada teks baru, eksperimen tersebut memerlukan Stanza dan model bahasa Indonesia.

## Isi repo dan alur data

```text
data/
  corpus/DOC001.txt … DOC015.txt   Teks 15 tokoh, satu tokoh per berkas
  sources.csv                      Judul, nama berkas, tautan revisi, dan lisensi
  qa_dev.csv                       25 pertanyaan untuk pengembangan (DEV001–DEV025)
  qa_test_template.csv             20 ID final, kolom lainnya masih kosong
  qa_test_parts/                   Soal uji tiap anggota sebelum digabung
  processed/passages.csv           75 passage yang dibaca kedua sistem
  processed/qa_dev.csv             Salinan DEV dengan kolom evidence_passage_id
  README.md                        Asal teks dan cara membaca dataset
  SHA256SUMS                       Acuan tes untuk mendeteksi perubahan pada 15 TXT dan 3 CSV sumber
src/                               Penyiapan data dan modul QA
tests/                             Tes otomatis untuk data dan kode
cache/                             Hasil stemming dan lemmatisasi untuk dipakai ulang
results/                           Hasil eksperimen DEV dan tempat evaluasi final
laporan/                           Laporan dan pembagian kontribusi
```

Paket dataset biografi disusun oleh Mutiara. Teks biografi dikutip dari kontributor Wikipedia bahasa Indonesia dengan lisensi CC BY-SA 4.0, dan tautan revisi tiap artikel tercantum pada metadata. [Asal teks](data/README.md) · [Sumber tiap tokoh](data/sources.csv)

**PBA02.** Pada `src/preprocessing.py`, fungsi `split_paragraphs()` memisahkan teks pada baris kosong, dan satu paragraf menjadi satu passage. Spasi di dalam paragraf, huruf kapital, angka, dan tanda baca tetap dipertahankan untuk kebutuhan NER, RAG, dan kutipan. Apabila pencarian memerlukan spasi yang seragam, gunakan `preprocess()` pada salinan teks.

**Contoh.** DEV001 merujuk dokumen `DOC001`, paragraf 4. Penyiapan data membentuk ID `DOC001_P04`, memeriksa kutipannya, lalu menyimpan ID tersebut pada DEV olahan. Korpus berisi `passage_id, doc_id, title, text, source`. Berkas yang disebut `corpus.csv` pada rencana awal kini tersedia sebagai `data/processed/passages.csv`, dan kedua retriever dapat membacanya dengan `load_corpus()`.

## Pembagian tugas

Tabel berikut menunjukkan tugas dan berkas yang dipegang tiap anggota. Rincian lengkap tersedia pada bagian 5 rencana kelompok, sedangkan jadwal, laporan, dan evaluasi pada bagian 8–10. [Rencana final lima anggota](TP_PBA_2026_Rencana_Final_5_Anggota.md)

| Anggota         | Berkas yang dipegang                                                                                                                               | Tugas                                                                                                                                                                                                                                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1 – Mutiara** | `data/`, `src/prepare_data.py`, `src/preprocessing.py`, `src/contracts.py`, `src/main.py`, `tests/`, `README.md`, `requirements.txt`, `.gitignore` | Menyiapkan data, menjaga ID dan kutipan, menyepakati format dengan tim, serta menghubungkan kedua sistem. Setelah seluruh modul tersedia, mencoba 3–5 pertanyaan DEV yang sama pada kedua mode dan menyimpan contoh jawaban beserta sumbernya untuk laporan.                                                                   |
| **2 – Farhan**  | `src/question_processing.py`, `src/retrieve_tfidf.py`                                                                                              | `analyze_question(question)` mengenali tipe jawaban, kata kunci, dan petunjuk relasi seperti _lahir_. `retrieve_tfidf(question, k=3)` mengurutkan passage dengan TF-IDF dan cosine similarity. Kedua fungsi diuji pada soal DEV.                                                                                               |
| **3 – Khansa**  | `src/extract_answer.py`, `src/check_answer.py`                                                                                                     | `extract_answer(question, q_info, passages)` memilih jawaban dan kutipannya, dengan NER untuk nama, lokasi, dan organisasi serta Regex untuk tanggal dan angka. `check_answer_type()` memeriksa kesesuaian jawaban dengan tipenya dan dipakai bersama Ammar.                                                                   |
| **4 – Jepri**   | `src/retrieve_dense.py`, `src/evaluate.py`                                                                                                         | `retrieve_dense(question, k=3)` mengembalikan tiga passage teratas berdasarkan embedding multibahasa dengan vektor korpus yang disimpan. `evaluate_all(qa_test)` menjalankan 20 soal pada kedua sistem dan menulis `results/evaluation_results.csv`. Tim memeriksa benar-salahnya dan menghitung akurasi masing-masing sistem. |
| **5 – Ammar**   | `src/rag_generate.py`                                                                                                                              | `rag_answer(question, passages, q_info)` menyusun prompt dari pertanyaan dan tiga passage beserta ID-nya, memanggil model, mencocokkan kutipan, dan memakai pemeriksaan tipe dari Khansa. Kasus bukti kurang, jawaban yang perlu ditinjau, dan kegagalan pemanggilan model dibedakan.                                          |

### Catatan masing-masing anggota

**Mutiara (Anggota 1).** Saya menyusun dataset dan kontrak data, lalu menyambungkan kedua sistem melalui `main.py`. Migrasi data, kerangka integrasi, dokumentasi, dan tes bagian saya dibantu oleh AI.

**Farhan (Anggota 2).** Saya mengerjakan analisis pertanyaan dan pencarian TF-IDF. Perbandingan beberapa varian TF-IDF pada soal DEV tersimpan di `results/`. Bantuan AI: [diisi oleh yang bersangkutan].

**Khansa (Anggota 3).** Saya mengerjakan pengambilan jawaban dari passage. Bagian yang masih saya perbaiki adalah pemilihan jawaban ketika satu passage memuat lebih dari satu tanggal atau nama. Bantuan AI: [diisi oleh yang bersangkutan].

**Jepri (Anggota 4).** Saya mengerjakan pencarian passage untuk sistem RAG dan program evaluasi kedua sistem. Pada pencarian dense, passage dan pertanyaan diubah menjadi vektor dengan `intfloat/multilingual-e5-small`, dibandingkan dengan cosine similarity, dan tiga teratas dikembalikan beserta skornya. Vektor passage disimpan di `data/cache/` agar tidak dihitung ulang. Pada 25 soal DEV, passage bukti masuk tiga besar pada 22 soal untuk pencarian dense dan 20 soal untuk TF-IDF. Selisih dua soal tersebut belum cukup untuk menyimpulkan bahwa salah satu metode lebih baik. Catatan lengkap, cara menjalankan, dan daftar pekerjaan yang tersisa tersedia pada [catatan bagian Anggota 4](docs/README_anggota4.md). Draf kode, tes, dan draf soal uji saya dibuat dengan bantuan AI, kemudian saya jalankan dan uji sendiri.

**Ammar (Anggota 5).** Saya mengerjakan pembuatan jawaban dengan LLM berdasarkan tiga passage dari pencarian dense. Bantuan AI: [diisi oleh yang bersangkutan].

## Soal uji dan laporan

Setiap anggota membuat empat soal final dan memeriksa bukti milik anggota lain. Soal disimpan di `data/qa_test_parts/` dalam satu berkas CSV per anggota. Setelah seluruh berkas terkumpul, `python -m src.merge_questions` menggabungkannya menjadi `data/qa_test.csv` sambil memeriksa jumlah soal, soal ganda, dan kecocokan bukti dengan teks.

Pada laporan, Farhan memegang Bab I–II. Mutiara dan Jepri menyusun Bab III beserta diagram kedua sistem. Seluruh anggota mengisi Bab IV untuk kodenya masing-masing. Jepri, Khansa, dan Ammar menyusun Bab V dari hasil yang benar-benar diperoleh. Mutiara mengoordinasikan abstrak, Bab VI, penggabungan `laporan/jawaban.docx` atau PDF, dan catatan kontribusi pada `laporan/tanggungjawab.docx`. Kontribusi dan pustaka yang dipakai dicatat sesuai arahan dosen.

## Kontrak yang dipakai bersama

Alur PBA14 pada `main.py` dapat diikuti dari kiri ke kanan.

- `run_classic(question)` → `analyze_question` → `retrieve_tfidf` → `extract_answer`.
- `run_rag(question)` → `analyze_question` → `retrieve_dense` → `rag_answer`.

Kedua retriever mengembalikan daftar terurut dengan `passage_id, text, score, source`. Kedua mode menghasilkan `question, answer, passage_id, evidence, source, status`, ditambah `passage` dan `note`. Untuk status `answered` atau `needs_review`, kutipan harus berada pada passage yang dipilih. Untuk `not_found`, jawaban dan rujukannya kosong. Saat evaluasi, hanya pertanyaan yang dikirim ke sistem. Jawaban acuan dan bukti berlabel dipakai setelah prediksi selesai. [Kontrak dan daftar tipe jawaban](src/contracts.py)

## Cara bekerja sama

Setiap anggota mengerjakan modulnya pada branch masing-masing, menjalankan tes, kemudian membuka pull request ke `main` untuk ditinjau anggota lain. Berkas milik anggota lain tidak diubah tanpa pemberitahuan terlebih dahulu. Folder lingkungan virtual, `data/cache/`, dan hasil uji coba lokal seperti `results/evaluation_dev.csv` tidak disertakan dalam commit.

## Status terakhir (25 November 2026)

- [x] Dataset, passage, kontrak, dan `main.py` (Mutiara)
- [x] Analisis pertanyaan dan TF-IDF (Farhan), sudah digabung ke `main`
- [ ] Pengambil jawaban klasik (Khansa), masih ditinjau dan diperbaiki
- [x] Pencarian dense dan program evaluasi (Jepri), siap ditinjau
- [ ] Generator RAG (Ammar), belum tersedia di repo
- [ ] Soal uji final sebanyak 20 soal
- [ ] Evaluasi akhir, penilaian bersama, dan laporan
