# Workout Movement Detector & Repetition Counter

Proyek Ujian Akhir Semester (UAS) mata kuliah **Kecerdasan Buatan (Artificial Intelligence)**.  
Aplikasi berbasis visi komputer (*computer vision*) dan logika fuzzy (*fuzzy logic*) untuk mendeteksi gerakan latihan fisik (workout), menghitung repetisi secara otomatis, serta mengklasifikasikan intensitas latihan secara real-time.

---

## 1. Tentang Proyek

Sering kali saat melakukan latihan kebugaran (seperti push-up, pull-up, squat, dan latihan gym lainnya), seseorang kesulitan menjaga konsistensi hitungan repetisi serta mengukur apakah intensitas latihan yang dilakukan sudah optimal.

Proyek ini dibangun sebagai solusi otomatis yang menggabungkan:
1. **Object Detection (YOLOv11)**: Mendeteksi keberadaan orang dan postur latihan workout pada setiap frame video.
2. **State-Machine Tracking**: Melacak pergerakan koordinat vertikal (posisi Y) bounding box untuk mendeteksi fase gerakan (`ATAS` dan `BAWAH`) guna menghitung repetisi secara akurat tanpa sensor fisik tambahan.
3. **Fuzzy Logic (Mamdani Inference)**: Menganalisis beban latihan dengan variabel input berupa *total repetisi* dan *rata-rata durasi per repetisi*, menghasilkan kategori intensitas: **Ringan**, **Sedang**, atau **Berat**.
4. **All-in-One Implementation**: Seluruh pipeline (inferensi, counter repetisi, fuzzy logic, serta fitur pelatihan model) disatukan ke dalam **1 file program utama (`workout_detector.py`)** secara prosedural dan efisien.

---

## 2. Struktur Berkas Proyek Utama

Repositori ini difokuskan secara khusus hanya pada berkas-berkas utama proyek:

```
GERAKAN WORKOUT/
│
├── workout_detector.py         # Program utama (Deteksi, Rep Counter, Fuzzy Logic, & Training)
├── requirements.txt            # Daftar pustaka dependensi Python
├── README.md                   # Dokumentasi teknis proyek
├── PRD_UAS_WorkoutDetection.md # Dokumen Product Requirements Document (PRD)
└── .gitignore                  # Konfigurasi filter berkas Git
```

> **Catatan:** Bobot model (`.pt`), dataset gambar berukuran besar, rekaman video keluaran (`.mp4`), serta berkas modul praktikum lama sengaja diabaikan melalui `.gitignore` agar repositori tetap ringan dan bersih.

---

## 3. Fitur Utama

- **Deteksi Real-Time**: Mendukung input dari berkas video rekaman (MP4, AVI, MOV), webcam internal laptop, maupun kamera USB eksternal.
- **Penghitung Repetisi Otomatis**: Algoritma pelacakan titik tengah bounding box dengan filter threshold untuk meredam noise deteksi.
- **Sistem Inferensi Fuzzy**: Menggunakan pustaka `scikit-fuzzy` dengan fungsi keanggotaan segitiga (`trimf`) dan aturan inferensi berbasis aturan praktikum.
- **Dukungan Hardware Adaptif**: Otomatis mendeteksi akselerasi GPU NVIDIA (CUDA) atau secara mandiri beralih ke mode CPU yang dioptimalkan untuk prosesor AMD Ryzen 5 6600H atau Intel.
- **Menu Fleksibel Pemilihan Model**: Memungkinkan pengguna memilih varian model bawaan YOLOv11 (Nano `yolo11n.pt`, Small `yolo11s.pt`, Medium `yolo11m.pt`) atau bobot kustom hasil pelatihan sendiri (`best.pt`).

---

## 4. Persyaratan Sistem & Instalasi

### Persyaratan Minimum
- Python versi 3.9 atau yang lebih baru
- Sistem Operasi: Windows 10/11, Linux, atau macOS
- Perangkat Keras:
  - Processor: AMD Ryzen 5 / Intel Core i5 atau setara
  - RAM: Minimal 8 GB
  - GPU: Opsional (NVIDIA dengan dukungan CUDA direkomendasikan jika ingin melatih model)

### Langkah Instalasi

1. Clone repositori ini ke komputer lokal:
   ```bash
   git clone https://github.com/khazennn234-gif/WORKOUT-DETECTOR-AI.git
   cd WORKOUT-DETECTOR-AI
   ```

2. Buat virtual environment (opsional namun disarankan):
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Linux/macOS:
   source venv/bin/activate
   ```

3. Pasang semua dependensi yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Cara Menjalankan Program

Jalankan berkas utama:
```bash
python workout_detector.py
```

Setelah dijalankan, menu interaktif akan tampil di terminal:

```
=======================================================
SISTEM DETEKSI GERAKAN WORKOUT & REPETISI (UAS AI)
=======================================================
Device: Mode CPU (AMD Ryzen / Intel / Tanpa CUDA)

MENU UTAMA:
1. Jalankan Deteksi Workout & Hitung Repetisi
2. Training / Fine-Tune Model YOLOv11 (Opsional)
3. Uji Coba Simulasi Fuzzy Logic Intensitas
4. Keluar
-------------------------------------------------------
Pilih menu (1/2/3/4):
```

### Panduan Operasional Menu:

- **Menu 1 (Jalankan Deteksi Workout)**:
  1. Pilih varian model YOLO. Rekomendasi untuk laptop CPU AMD Ryzen: pilih opsi `2` (`yolo11n.pt`) agar proses berjalan cepat dan responsif.
  2. Pilih sumber video:
     - Masukkan `1` untuk membaca berkas video (misal: `workout.mp4` di direktori yang sama).
     - Masukkan `2` untuk menggunakan webcam bawaan laptop.
     - Masukkan `3` untuk menggunakan kamera USB eksternal.
  3. Jendela OpenCV akan menampilkan bounding box, label confidence, counter repetisi, dan status intensitas secara real-time.
  4. Tekan tombol **`q`** pada keyboard untuk menghentikan deteksi.
  5. Rekaman video hasil inferensi akan tersimpan otomatis sebagai `output_workout.mp4`.

- **Menu 2 (Pelatihan Model Kustom)**:
  - Digunakan jika ingin melatih model YOLOv11 pada dataset kustom `workout.v12i.yolov11` (format YOLO).

- **Menu 3 (Simulasi Fuzzy Logic)**:
  - Menguji fungsi keanggotaan dan aturan inferensi fuzzy secara terisolasi tanpa perlu memutar video.

---

## 6. Teori & Metodologi AI

### Logika Penghitung Repetisi
1. Titik koordinat vertikal subjek dihitung melalui pusat bounding box:
   $$center\_y = \frac{y_1 + y_2}{2}$$
2. Perubahan posisi dibandingkan terhadap frame sebelumnya:
   - Jika pergeseran melewati ambang batas ke atas ($center\_y < prev\_y - threshold$), state diatur menjadi `ATAS`.
   - Jika pergeseran melewati ambang batas ke bawah ($center\_y > prev\_y + threshold$), state diatur menjadi `BAWAH`.
3. Satu repetisi penuh dicatat ketika terjadi transisi lengkap dari fase `BAWAH` menuju fase `ATAS`.

### Logika Fuzzy (Penentu Intensitas)
- **Input 1**: Total repetisi ($0$ hingga $30$ rep, himpunan: *sedikit*, *sedang*, *banyak*).
- **Input 2**: Kecepatan rata-rata ($0$ hingga $10$ detik/rep, himpunan: *cepat*, *sedang*, *lambat*).
- **Output**: Skor intensitas ($0$ hingga $100$, himpunan: *Ringan*, *Sedang*, *Berat*).

---

## 7. Dependensi Utama

- `ultralytics` (YOLOv11 Engine & PyTorch)
- `opencv-python` (Pengolahan Citra & Video Capture)
- `scikit-fuzzy` (Sistem Kontrol Logika Fuzzy)
- `numpy` (Manipulasi Array Matriks Koordinat)

---

## 8. Lisensi & Kredit

- Proyek ini dikembangkan untuk keperluan akademik Ujian Akhir Semester mata kuliah Kecerdasan Buatan.
- Dataset latihan bersumber dari Roboflow Universe (*workout-alzp1*).
