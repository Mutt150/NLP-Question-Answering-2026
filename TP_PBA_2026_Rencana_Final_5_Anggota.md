# RENCANA FINAL TP PBA 2026
## Dua Sistem Question Answering — Pembagian 5 Anggota dan 2 Opsi Dataset

**Status:** Rencana kerja kelompok, siap dibagikan dan disesuaikan dengan nama anggota.  
**Acuan utama:** `TP - PBA 2026.docx` (Pak Putra Pandu Adikara), materi PBA di folder project, dan kesepakatan kelompok.  
**Keputusan kelompok:** Mengerjakan **DUA sistem**: **(1) QA tanpa LLM** dan **(2) QA dengan RAG + LLM**.  
**Periode rencana:** 14 hari. Deadline/jam sebenarnya tetap mengikuti BRONE.  
**Prinsip:** **Satu korpus yang sama untuk dua sistem.** Hanya **pilih SATU opsi dataset**, bukan mengerjakan keduanya.

> **Gambaran singkat proyek:** Pengguna memasukkan pertanyaan Bahasa Indonesia. Sistem 1 mencari teks relevan lalu mengekstraksi jawaban tanpa LLM. Sistem 2 mencari teks relevan menggunakan embedding lalu meminta LLM menjawab berdasarkan teks itu. Keduanya dibandingkan pada 20 pertanyaan yang sama.

---

## 1. Yang benar-benar diminta Pak Pandu

- Kelompok maksimal **5 orang**, menggunakan **Python**; kelompok kita sepakat membuat **kedua pendekatan QA**.
- **Sistem 1 (tanpa LLM):** jelaskan *question processing* (klasifikasi pertanyaan/jenis jawaban), *passage retrieval*, dan *answer processing* (ekstraksi jawaban).
- **Sistem 2 (RAG dengan LLM):** jelaskan retrieval, pemberian konteks ke LLM, generasi jawaban, dan kaitannya dengan materi PBA.
- **Bukan retrieval saja:** perlu menerapkan teknik PBA seperti POS Tagging, NER/Information Extraction, Regex, Text Encoding/Embedding, dan yang memang relevan.
- Gunakan **20 pertanyaan evaluasi** sesuai korpus, **dibuat oleh anggota kelompok** (disepakati **4 pertanyaan/orang**), dibandingkan dengan jawaban benar (*ground truth*), hitung **benar/salah**, lalu jelaskan penyebabnya.
- Buat laporan **Bab I–VI**, abstrak, referensi; jelaskan proses dan **kode yang dibuat sendiri vs library yang digunakan**.
- Kumpulkan kode `.py` atau `.ipynb`, korpus/data, hasil pengujian, **`jawaban.docx`/PDF** (dengan screenshot), dan **`tanggungjawab.docx`** dalam ZIP/RAR.
- **Kontribusi teknis tiap orang harus nyata**, bukan hanya menulis laporan/presentasi. Penilaian bersifat individual.
- Format laporan: **minimal 15 halaman** (di luar judul), margin **kiri 3 cm; lainnya 2 cm**, spasi **1,15**, **Aptos/Calibri 12** (judul 14). Daftar referensi **Harvard Anglia-Ruskin**.
- **Bobot terbesar:** Bab III Metodologi **20%** dan Bab IV Implementasi **20%**, total **40%**.

**Catatan:** Ketentuan resmi mengenai plagiarisme dan batasan penggunaan AI wajib diikuti. Setiap anggota tetap bertanggung jawab atas implementasi, sumber, eksperimen, dan penjelasannya.

---

## 2. Dataset: dua opsi, pilih satu

Dataset **belum perlu dikunci sekarang**. Orang 1 menguji opsi A lebih dulu; jika sampel tidak cocok untuk ekstraksi jawaban klasik, pilih opsi B. **Keputusan dataset harus selesai di awal**, jangan terus berganti saat coding.

### OPSI A — Dataset publik Hugging Face (prioritas diuji dulu)

- **Nama:** IDK-MRC
- **Sumber:** https://huggingface.co/datasets/rifkiaputri/idk-mrc
- **Akses Python:** `load_dataset("rifkiaputri/idk-mrc")` dari package `datasets`.
- Dataset memiliki **teks konteks** dan anotasi QA yang berguna untuk eksplorasi. Struktur pertanyaan/jawaban dapat berupa objek bersarang, jadi cek kolom aktual setelah dataset di-*load*.

**Pekerjaan Orang 1:**
1. Muat dataset; tampilkan **split, nama kolom, jumlah baris, dan 5 contoh**.
2. Periksa contoh *context*, pertanyaan, jawaban, serta status *answerable/unanswerable* jika ada.
3. Pilih sekitar **20–30 konteks unik** yang jelas, cukup beragam, dan tidak terlalu panjang (jumlah usulan, bukan syarat resmi).
4. Ubah menjadi **`data/corpus.csv`** dengan format yang disepakati. Pertahankan ID konteks dan informasi asal dataset/split.
5. Jadikan pertanyaan bawaan sebagai **referensi/pengujian pengembangan**, **bukan langsung menyalin** 20 soal final.
6. Periksa jika ada teks berulang, jawaban ambigu, dan apakah pertanyaan *unanswerable* ternyata terjawab oleh konteks lain di korpus gabungan.
7. Catat lisensi dan atribusi dataset berdasarkan sumber resminya.

**Kapan dipilih?** Jika konteksnya relatif mudah untuk retrieval dan NER/Regex bisa mengambil jawaban pendek. **Keuntungan:** tidak perlu mengumpulkan teks dari awal.

### OPSI B — Korpus buatan kelompok (cadangan yang lebih terkontrol)

- **Topik yang disarankan:** Biografi/perjuangan pahlawan nasional Indonesia (Bahasa Indonesia).
- **Sumber teks:** misalnya artikel ensiklopedia/Wikipedia atau sumber tepercaya lain; catat URL dan tanggal akses.
- Ambil sekitar **12–15 teks faktual** dengan informasi nama, tempat, tanggal, organisasi, dan peristiwa (jumlah usulan, bukan syarat resmi).

**Pekerjaan Orang 1:**
1. Pilih **satu topik saja** dan 12–15 sumber teks.
2. Simpan teks faktual yang relevan; jangan hanya tabel *infobox* supaya QA klasik benar-benar memproses kalimat.
3. Bersihkan bagian yang tidak relevan (menu, rujukan halaman, simbol, duplikat) tanpa merusak fakta.
4. Bagi menjadi passage/paragraf yang tetap dapat dilacak ke dokumen sumber.
5. Simpan formatnya sebagai **`data/corpus.csv`** dan metadata sumber.
6. Pastikan ada cukup variasi fakta untuk membuat 20 pertanyaan uji.

**Kapan dipilih?** Jika IDK-MRC ternyata terlalu heterogen atau sulit dipakai oleh algoritma ekstraksi jawaban sederhana. **Keuntungan:** jenis fakta dan tingkat kesulitan lebih terkendali. **Konsekuensi:** sumber dan ground truth perlu lebih banyak diverifikasi manual.

### Keputusan praktis (maksimal 1–2 hari)

| Yang diuji | Opsi A: IDK-MRC | Opsi B: Korpus sendiri |
|---|---|---|
| Bisa mendapatkan 5 passage faktual yang layak? | Cek sampel | Cek sumber |
| Bisa dibuat pertanyaan siapa/kapan/di mana/berapa? | Cek konteks | Cek artikel |
| Jawaban memang ada sebagai bukti teks? | Cek anotasi dan teks | Cek teks |
| Dapat dipakai *kedua* sistem? | Ya jika diubah ke format bersama | Ya jika diubah ke format bersama |

**Aturan final:** Jika Opsi A lolos uji awal, **pilih A**. Jika tidak, **pilih B**. Setelah dipilih, gunakan format file **yang sama** di bawah ini.

---

## 3. Format data yang wajib disepakati sejak awal

Tidak perlu membuat dataset training apabila kalian **tidak melakukan training/fine-tuning**. Yang utama adalah **korpus**, **pertanyaan pengembangan** untuk mencoba modul, dan **20 pertanyaan uji final**.

**`data/corpus.csv` — semua passage yang boleh dicari oleh sistem**

| Kolom | Isi |
|---|---|
| `passage_id` | ID unik, misalnya `P001` |
| `doc_id` | ID dokumen/konteks induk, misalnya `D001` |
| `title` | Judul/topik singkat atau ID konteks |
| `text` | Teks asli passage yang akan dibaca sistem |
| `source` | URL sumber atau IDK-MRC + split/ID data |

**`data/qa_dev.csv` — beberapa pertanyaan untuk mencoba sistem**

| Kolom | Isi |
|---|---|
| `question_id` | Contoh `DEV01` |
| `question` | Pertanyaan pengembangan |
| `ground_truth` | Jawaban benar untuk pengecekan saat development |
| `evidence_passage_id` | Passage sumber jawaban |

**`data/qa_test.csv` — 20 pertanyaan FINAL yang disusun tim**

| Kolom | Isi |
|---|---|
| `question_id` | `Q01` s.d. `Q20` |
| `author` | Anggota pembuat pertanyaan |
| `question` | Pertanyaan buatan anggota |
| `answer_type` | `PERSON`, `DATE`, `LOCATION`, `NUMBER`, atau `OTHER` |
| `ground_truth` | Jawaban benar yang sudah diverifikasi |
| `evidence_passage_id` | ID passage tempat bukti jawaban berada |
| `evidence_text` | Kutipan singkat teks bukti |

**Aturan penting:**
- Tiap orang membuat **4 pertanyaan final**; **anggota lain memeriksa** jawaban dan lokasi buktinya.
- Jangan sampai jawaban/ground truth menjadi input ke sistem saat inferensi.
- Jangan mengutak-atik metode berdasarkan kesalahan pada 20 soal final sampai hasil terlihat bagus. Gunakan `qa_dev.csv` untuk menyetel aturan; simpan 20 soal final untuk pengujian akhir.
- Simpan **teks asli** untuk NER dan LLM; hasil normalisasi teks untuk TF-IDF dibuat terpisah.
- Untuk pertanyaan "tidak ditemukan", pastikan **benar-benar tidak terjawab di seluruh korpus**, bukan hanya pada satu passage.

---

## 4. Dua sistem yang dibangun

### SISTEM 1 — QA TANPA LLM (*classical/extractive QA*)

```text
Pertanyaan pengguna
   ↓
[Orang 2] Question processing: deteksi kata tanya, answer type, POS/kata kunci
   ↓
[Orang 2] TF-IDF + cosine similarity: ambil top-3 passage
   ↓
[Orang 3] NER + Regex: daftar kandidat jawaban
   ↓
[Orang 3] Pilih kandidat sesuai tipe + relasi kata dalam kalimat
   ↓
[Orang 1] Jawaban pendek + passage sumber
```

Contoh: "Kapan X wafat?" Jangan otomatis memilih tanggal pertama, karena passage dapat juga memuat tanggal lahir. Jika tidak ada kandidat/bukti yang memadai, tampilkan **"Jawaban tidak ditemukan dalam korpus"**.

### SISTEM 2 — QA RAG DENGAN LLM

```text
[Orang 1] Korpus/passage yang sama
   ↓
[Orang 4] Sentence embedding untuk passage + pertanyaan
   ↓
[Orang 4] Cosine similarity: ambil top-3 passage
   ↓
[Orang 5] Susun prompt: instruksi + passage sumber + pertanyaan
   ↓
[Orang 5] LLM menghasilkan jawaban berdasarkan konteks
   ↓
[Orang 3/5] Pemeriksaan tipe jawaban dan dukungan bukti secara sederhana
   ↓
[Orang 1] Jawaban + passage sumber
```

**Minimum:** gunakan satu model embedding multilingual yang bisa dijalankan dan satu LLM yang sudah diuji. Model/API dapat dipilih berdasarkan ketersediaan perangkat/kuota. Jangan mengklaim validasi NER otomatis menjamin tidak ada halusinasi; hasil yang meragukan tetap dianalisis manual.

**Yang tidak perlu dulu:** training POS/NER/LLM dari nol, reranker, vector database, hybrid search, dependency parsing wajib, fine-tuning, UI Streamlit. Boleh ditambahkan **hanya jika dua sistem minimum sudah berjalan**.

---

## 5. PEMBAGIAN TUGAS FINAL 5 ORANG

> **Orang 1 adalah kamu.** Orang 2–5 masing-masing punya pekerjaan kode yang dapat diuji. Tidak ada orang yang hanya kebagian laporan.

### ORANG 1 — Repository, Dataset, Preprocessing, dan Integrasi (**KAMU**)

**File milikmu:** `src/prepare_data.py`, `src/preprocessing.py`, `src/main.py`, `README.md`, `requirements.txt`, `.gitignore`, folder `data/`.

**Apa yang kamu kerjakan:**
1. Buat **repository GitHub** dan struktur folder proyek, atur cara menjalankan program.
2. **Eksplorasi dua opsi dataset** (prioritas IDK-MRC), kemudian tentukan satu bersama kelompok.
3. Siapkan `corpus.csv`, `qa_dev.csv`, dan template `qa_test.csv` dengan kolom yang sudah disepakati.
4. Buat preprocessing **sederhana**: bersihkan spasi/simbol berlebih, segmentasi passage, dan siapkan teks untuk retrieval **tanpa mengubah teks asli**.
5. Tetapkan **kontrak format fungsi** agar kode teman-teman bisa digabung.
6. Buat `main.py`: ambil pertanyaan, jalankan pipeline klasik/RAG, tampilkan **jawaban + passage sumber**.
7. Integrasikan modul Orang 2–5; uji *end-to-end*, tangani error antarfile, dan dokumentasikan cara menjalankan.

**Hasil yang harus kamu serahkan:**
- Repo dapat di-*clone* dan diinstal dari `requirements.txt`.
- `data/corpus.csv` valid, memiliki ID unik, dan asal data jelas.
- Program utama bisa menjalankan **dua mode** (`classic` dan `rag`) dengan input pertanyaan yang sama.
- `README.md` menjelaskan instalasi, dataset, cara run, dan pembagian tim.

**Bukan tugasmu:** menulis seluruh TF-IDF, NER, embedding, atau prompt LLM milik anggota lain.

### ORANG 2 — Question Processing + TF-IDF Retrieval (QA klasik)

**File milik:** `src/question_processing.py`, `src/retrieve_tfidf.py`.

**Apa yang dikerjakan:**
1. Deteksi kata tanya **siapa/kapan/di mana/berapa/apa** dengan aturan/Regex.
2. Tetapkan **expected answer type** dan ambil keyword penting; gunakan POS Tagging Bahasa Indonesia bila model tersedia dan bekerja dengan benar.
3. Buat **TF-IDF** atas passage dari `corpus.csv`, kemudian **cosine similarity** untuk memberi peringkat passage.
4. Ambil **top-3** passage; keluarkan ID, teks, dan skor.
5. Buat tes beberapa pertanyaan yang memiliki kemiripan kata tinggi/rendah.

**Hasil wajib:** Pertanyaan menghasilkan `{answer_type, keywords, relation_cues}`, lalu retrieval menghasilkan daftar **top-3 passage**. Keluaran question processing juga dapat dipakai Orang 5 untuk mengarahkan format jawaban RAG.

### ORANG 3 — NER, Ekstraksi Jawaban, dan Pengecekan Tipe

**File milik:** `src/extract_answer.py`, `src/check_answer.py` (fungsi tipe jawaban).

**Apa yang dikerjakan:**
1. Gunakan NER/model Bahasa Indonesia untuk mencari **orang/lokasi/organisasi** jika label tersedia; gunakan Regex untuk **tanggal/angka**.
2. Kumpulkan kandidat jawaban dari passage top-3 milik Orang 2.
3. Buat **aturan memilih jawaban terbaik**, misalnya tipe cocok, kata relasional (*lahir/wafat/mendirikan*) cocok, dan kalimat bukti relevan.
4. Jika kandidat kurang kuat, beri status **tidak ditemukan**, bukan memaksa jawaban.
5. Sediakan pengecekan **tipe** keluaran RAG (mis. pertanyaan "kapan" cenderung memerlukan tanggal/tahun). Ini **bukan verifikasi fakta penuh**.

**Hasil wajib:** Untuk satu pertanyaan, terlihat **kandidat → alasan pilihan → jawaban singkat + passage_id**. Minimal tes pada contoh pertanyaan dengan lebih dari satu tanggal/nama agar terlihat proses seleksinya.

### ORANG 4 — Semantic Retrieval RAG + Evaluasi Dua Sistem

**File milik:** `src/retrieve_dense.py`, `src/evaluate.py`.

**Apa yang dikerjakan:**
1. Pilih satu model **sentence embedding multilingual pretrained** yang bisa dijalankan.
2. Ubah passage dari `corpus.csv` menjadi vektor, lalu simpan/cache hasilnya agar tidak dihitung ulang tiap query.
3. Ubah pertanyaan menjadi vektor dan hitung **cosine similarity**; kembalikan top-3 passage.
4. Susun program untuk menjalankan **20 pertanyaan final pada kedua sistem** dan menyimpan prediksinya.
5. Siapkan tabel penilaian benar/salah serta perhitungan **akurasi**; penilaian makna dan analisis kegagalan diverifikasi bersama.

**Hasil wajib:** Top-3 hasil semantic retrieval beserta skor, plus `results/evaluation_results.csv` dengan jawaban kedua sistem, benar/salah, dan rekap akurasi. Tidak boleh menggunakan jawaban benar sebagai konteks saat inferensi.

### ORANG 5 — RAG Generator, Prompt, dan Pemeriksaan Bukti

**File milik:** `src/rag_generate.py` (dan fungsi pemeriksaan dukungan sumber di modul yang disepakati).

**Apa yang dikerjakan:**
1. Pilih satu LLM yang realistis dijalankan (API/free tier atau model kecil lokal/Colab), setelah dites akses dan batasannya.
2. Rancang prompt yang berisi **pertanyaan + top-3 passage** dari Orang 4 + ID sumber, tanpa memberikan ground truth.
3. Implementasikan pemanggilan model; minta jawaban ringkas hanya dari konteks dan instruksi **"tidak ditemukan"** bila tidak ada bukti.
4. Kembalikan jawaban beserta passage yang digunakan. Tangani kegagalan model/API, timeout, atau jawaban kosong.
5. Bekerja sama dengan Orang 3 untuk pemeriksaan tipe; lakukan **pemeriksaan dukungan sumber sederhana** dan tandai keluaran yang patut ditinjau manual.

**Hasil wajib:** `rag_answer(...)` menghasilkan jawaban dan sumber dari passage retrieval yang diberikan, berjalan pada beberapa contoh pertanyaan, dan tidak mengarang output saat model gagal.

### Ringkasan pembagian paling singkat

| Orang | Fokus utama | Bagian kode utama | Bukti pekerjaan |
|---|---|---|---|
| **1 (kamu)** | Repo + dataset + preprocessing + integrasi | `prepare_data.py`, `preprocessing.py`, `main.py` | Dataset siap; **2 sistem bisa dijalankan** |
| **2** | Question Processing + TF-IDF | `question_processing.py`, `retrieve_tfidf.py` | Tipe pertanyaan + **top-3 passage klasik** |
| **3** | NER + ekstraksi jawaban | `extract_answer.py`, `check_answer.py` | **Jawaban pendek dan bukti** dari passage |
| **4** | Embedding + evaluasi | `retrieve_dense.py`, `evaluate.py` | **Top-3 passage RAG + hasil evaluasi** |
| **5** | LLM + prompt RAG | `rag_generate.py` | **Jawaban RAG + sumber** |

---

## 6. Kontrak antarmodul: sepakati sebelum mulai coding

Setiap anggota boleh memakai library pilihan yang relevan, tapi **output fungsi wajib seragam**.

| Fungsi | PIC | Input | Output utama |
|---|---|---|---|
| `load_corpus()` | 1 | `data/corpus.csv` | Daftar passage `{passage_id, text, source}` |
| `preprocess(text)` | 1 | teks | teks/token untuk retrieval (tanpa merusak teks asli) |
| `analyze_question(question)` | 2 | string pertanyaan | `{answer_type, keywords, relation_cues}` |
| `retrieve_tfidf(question, k=3)` | 2 | pertanyaan | daftar `{passage_id, text, score}` |
| `extract_answer(question, q_info, passages)` | 3 | pertanyaan + analisis + passage | `{answer, passage_id, evidence}` |
| `retrieve_dense(question, k=3)` | 4 | pertanyaan | daftar `{passage_id, text, score}` |
| `rag_answer(question, passages, q_info)` | 5 | pertanyaan + passage + tipe | `{answer, passage_id, evidence_or_note}` |
| `run_classic(question)` / `run_rag(question)` | 1 | pertanyaan | hasil kedua pipeline dalam format seragam |
| `evaluate_all(qa_test)` | 4 | 20 pertanyaan final | tabel prediksi + status penilaian |

**Penting:** `retrieval` mengembalikan **passage beserta ID dan skor**, bukan hanya sebuah string. Yang memutuskan jawaban klasik adalah Orang 3; yang menghasilkan jawaban RAG adalah Orang 5. Orang 1 menghubungkan, **bukan menulis ulang** modul lain.

Contoh format *passage* yang diberikan dari retrieval ke modul berikutnya:

```json
{
  "passage_id": "P001",
  "text": "Teks passage asli yang ditemukan dari korpus.",
  "score": 0.82,
  "source": "D001"
}
```

Nilai skor itu hanya **contoh format**, bukan hasil eksperimen.

---

## 7. Struktur repository yang dipakai

```text
TP-PBA-2026/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── corpus.csv
│   ├── qa_dev.csv
│   ├── qa_test.csv
│   └── sources.csv
├── src/
│   ├── main.py                  # Orang 1: integrasi
│   ├── prepare_data.py          # Orang 1: muat/kurasi dataset
│   ├── preprocessing.py         # Orang 1: bersihkan teks
│   ├── question_processing.py   # Orang 2
│   ├── retrieve_tfidf.py       # Orang 2
│   ├── extract_answer.py        # Orang 3
│   ├── check_answer.py          # Orang 3 (cek tipe, dibantu Orang 5)
│   ├── retrieve_dense.py        # Orang 4
│   ├── evaluate.py              # Orang 4
│   └── rag_generate.py          # Orang 5
├── tests/                       # tes unit sederhana per modul
├── notebooks/                   # opsional, eksperimen dataset/LLM
├── results/                     # file prediksi dan evaluasi nyata
└── laporan/
    ├── jawaban.docx             # laporan final Bab I–VI
    └── tanggungjawab.docx       # kontribusi teknis per orang
```

**Lingkungan kerja:** **Python `.py` + VS Code + GitHub** untuk kerja utama; **Colab opsional** untuk eksplorasi dataset/model yang membutuhkan GPU. Tidak perlu membuat aplikasi web dulu. `main.py` berbasis terminal sudah memadai sebagai target minimum demo.

**Cara kolaborasi:** satu repository, branch terpisah per orang (`anggota-1-data`, `anggota-2-classic`, dan seterusnya), lalu Pull Request/review sebelum merge agar tidak menimpa file teman.

---

## 8. Timeline 14 hari yang realistis

| Hari | Apa yang dikerjakan | Yang HARUS ada di akhir |
|---|---|---|
| **1** | Orang 1 membuat repo, coba 5 data IDK-MRC. Semua sepakat kontrak fungsi. | Repo bersama + sampel dan keputusan awal dataset |
| **2** | Putuskan dataset A/B, buat `corpus.csv` awal. Anggota lain mulai dengan data contoh. | Format data dikunci + lima anggota mulai coding |
| **3–4** | Orang 1 finalisasi data; Orang 2 question processing/TF-IDF; Orang 3 NER/ekstraksi; Orang 4 embedding; Orang 5 model/prompt. | Masing-masing modul bisa dicoba sendiri |
| **5–6** | Integrasi versi awal: QA klasik dan RAG; perbaiki error input/output. | **Dua sistem menjawab minimal 3–5 contoh pertanyaan** |
| **7** | Demo progres, dokumentasi flowchart, catat masalah. | Screenshot/demo + arsitektur awal |
| **8–9** | Masing-masing tulis 4 pertanyaan final; validasi silang; perbaiki algoritma menggunakan **soal dev**, bukan soal final. | 20 QA final terverifikasi dan sistem stabil |
| **10** | Jalankan evaluasi 20 soal pada kedua sistem; simpan prediksi. | Tabel benar/salah + metrik dan error cases |
| **11–12** | Analisis kesalahan, tulis Bab I–V secara paralel, ambil screenshot. | Draft laporan + pembahasan berbasis hasil nyata |
| **13** | Integrasi laporan, Bab VI, cek sumber, kontribusi, file dan instalasi ulang. | Paket tugas hampir final, bisa dijalankan ulang |
| **14** | Periksa ZIP dan unggah sebelum batas BRONE. | **File terkumpul** |

**Ketentuan kerja:** jika jadwal resmi lebih pendek/berbeda, sesuaikan. Jangan menunggu hari ke-11 untuk mulai mencatat hasil uji dan dokumentasi.

---

## 9. Pembagian laporan (di luar tanggung jawab coding)

| Bagian | Koordinator | Isi yang harus disediakan |
|---|---|---|
| **Abstrak + Bab VI + penggabungan** | **1** | Ringkasan data, metode dua sistem, angka evaluasi nyata, kesimpulan sesuai rumusan masalah |
| **Bab I + Bab II** | **2** | Latar belakang, rumusan masalah/tujuan/manfaat, penelitian terdahulu, teori metode dan evaluasi, referensi ilmiah |
| **Bab III Metodologi** | **1 + 4** | Diagram utuh input → preprocessing → kedua pipeline → evaluasi; penjelasan dataset, skenario uji, metrik |
| **Bab IV Implementasi** | **Semua, sesuai modulnya** | Langkah implementasi, fungsi kode, input/output, library, screenshot, alasan teknik |
| **Bab V Hasil & Pembahasan** | **4 + 3 + 5** | Rekap 20 pertanyaan, hasil dua sistem, alasan jawaban benar/salah, contoh kegagalan |
| **`tanggungjawab.docx`** | **1, diverifikasi semua** | Kontribusi teknis yang benar-benar dilakukan per orang, bukan daftar bab saja |

**Poin penting Bab III–IV:** tampilkan **flowchart khusus sistem**, bukan hanya “studi literatur → implementasi → kesimpulan”; jelaskan **apa masukannya, apa prosesnya, apa keluarannya** dan berikan contoh keluaran setiap modul.

---

## 10. Evaluasi yang wajib dilakukan semua anggota

**Setiap orang membuat empat pertanyaan final** dari korpus yang sudah dipilih. Usahakan gabungan soal mencakup orang, lokasi, tanggal, angka, dan beberapa pertanyaan dengan hubungan fakta, selama jawabannya benar-benar ada dan dapat diverifikasi.

**Langkah evaluasi:**
1. Kunci 20 pertanyaan dan ground truth; lakukan verifikasi silang oleh teman.
2. Jalankan 20 pertanyaan **yang sama** pada **QA tanpa LLM** dan **RAG**.
3. Simpan prediksi, ID passage yang diambil, dan status error/tidak ditemukan jika ada.
4. Nilai **benar/salah berdasarkan makna jawaban dan bukti** (bukan cuma kecocokan huruf), menggunakan aturan yang sama bagi kedua sistem.
5. Hitung: `Accuracy = jumlah jawaban benar / 20 × 100%` untuk **masing-masing** sistem.
6. Analisis beberapa kesalahan: retrieval tidak menemukan sumber, klasifikasi tipe salah, ekstraksi salah, atau jawaban RAG tanpa dukungan sumber.
7. Catat **semua** hasil, termasuk yang gagal/error. **Jangan mengarang angka** atau menghapus hasil gagal diam-diam.

**Format `results/evaluation_results.csv`:**

`question_id, question, ground_truth, classic_answer, classic_passage_id, classic_correct, rag_answer, rag_passage_id, rag_correct, error_reason`

Tambahan **opsional**: Hit@3 untuk retrieval jika ada waktu. **Tidak wajib** menambahkan seluruh EM/F1/MRR jika tidak dipakai atau tidak dijelaskan dengan benar.

---

## 11. Hubungan dengan materi kuliah PBA

| Materi | Penerapan nyata dalam kode | Pemilik |
|---|---|---|
| **PBA02 Pemrosesan Teks** | Pembersihan teks, tokenisasi, segmentasi passage, Regex | 1, 2 |
| **PBA04 POS Tagging** | Mengambil kata kunci/kelas kata pada pertanyaan | 2 |
| **PBA05 Information Extraction** | NER dan kandidat jawaban | 3 |
| **PBA06 Semantik** | Dasar konsep kemiripan semantik | 4 (landasan) |
| **PBA10 CFG/Dependency Grammar** | Relevan untuk relasi sintaksis; **opsional**, tidak diklaim digunakan jika tidak diterapkan | Opsional |
| **PBA11 Text Encoding** | TF-IDF dan embedding vektor untuk retrieval | 2, 4 |
| **PBA12 Deep Learning untuk NLP** | Dasar penggunaan model pretrained | 3, 4, 5 (landasan) |
| **PBA13 Transformer & LLM** | Generator RAG berbasis LLM, prompt, keterbatasan halusinasi | 5 |
| **PBA14 Question Answering** | Question Processing → Passage Retrieval → Answer Processing; RAG dan evaluasi | Semua |

**Peringatan istilah:** *NP chunking* (chunk frasa pada PBA05) **berbeda** dengan *document chunking* (pembagian dokumen menjadi passage untuk RAG). Jangan mengklaim keduanya sama.

---

## 12. Checklist final sebelum submit

### Source code dan data
- [ ] Repo GitHub mempunyai seluruh kode Python dan `requirements.txt`.
- [ ] Kedua sistem dapat menerima pertanyaan dan menghasilkan **jawaban + sumber**.
- [ ] Korpus, sumber/metadata, data pengembangan dan 20 pertanyaan uji final tersedia.
- [ ] Hasil preprocessing dan eksperimen yang relevan disertakan.
- [ ] `results/evaluation_results.csv` berisi **hasil nyata** kedua sistem.
- [ ] Setiap anggota punya kontribusi teknis, tes, dan riwayat pekerjaan yang dapat dijelaskan.

### Laporan
- [ ] Identitas kelompok dan pembagian tanggung jawab teknis.
- [ ] Abstrak dengan hasil angka kuantitatif setelah evaluasi.
- [ ] Bab I–VI sesuai instruksi dan keterkaitan rumusan masalah–tujuan–kesimpulan.
- [ ] Flowchart metodologi yang spesifik, diagram dua sistem, dan pembahasan tiap metode.
- [ ] Penjelasan kode **buatan kelompok vs library** dan screenshot.
- [ ] Tabel 20 soal, benar/salah, akurasi, dan **analisis penyebab**.
- [ ] Referensi menggunakan Harvard Anglia-Ruskin dan sumber data dicantumkan.
- [ ] Format minimal 15 halaman, margin dan font sesuai ketentuan.

### Pengumpulan
- [ ] `jawaban.docx` atau PDF dan `tanggungjawab.docx` disertakan.
- [ ] Semua source code, data, hasil, dan kebutuhan menjalankan program masuk ZIP/RAR.
- [ ] Nama berkas mengikuti format resmi, contoh: `[TP][PBA-B] NamaKetua.ZIP` (sesuaikan kelas dan arahan BRONE; naskah resmi memuat dua redaksi format yang sedikit berbeda).
- [ ] Satu orang mengunggah **sebelum deadline/jam server BRONE**.

---

## 13. Tiga keputusan saat rapat pertama

1. **Setujui pembagian lima orang** dan isi nama asli di sebelah nomor anggotanya.
2. **Orang 1 menguji IDK-MRC hari pertama**, lalu tim putuskan lanjut IDK-MRC atau beralih ke korpus sendiri maksimal hari kedua.
3. **Sepakati format `corpus.csv`, hasil retrieval, dan `qa_test.csv`**, lalu semua langsung bekerja paralel. Tidak perlu menunggu aplikasi lengkap.

**Prioritas:** dataset terkunci → dua QA berjalan → 20 soal dievaluasi → laporan dan ZIP rapi. Kerjakan yang memengaruhi nilai terlebih dahulu; UI dan eksperimen tambahan belakangan.
