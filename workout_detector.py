# ============================================================
# workout_detector.py
# File Tunggal Proyek UAS Kecerdasan Buatan (AI)
#
# Mengintegrasikan seluruh modul praktikum dalam satu file:
#   - Modul 10: YOLO (Object Detection)
#   - Modul 9 : Pengolahan Citra & Video (OpenCV)
#   - Modul 6 : Fuzzy Logic (Klasifikasi Intensitas Latihan)
#   - Modul 3 : Manajemen Dataset & Labeling
#   - Modul 4 : Preprocessing Data Koordinat
#
# Kompatibel dengan:
#   - GPU NVIDIA (CUDA)
#   - CPU Mode (AMD Ryzen 5 6600H / Intel / Tanpa GPU NVIDIA)
# ============================================================

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time
import os


# ============================================================
# BAGIAN 1: SISTEM INFERENSI FUZZY (MODUL 6 - FUZZY LOGIC)
# Diadaptasi langsung dari kode praktikum FUZZY.py
# ============================================================

# 1. Definisi Variabel Input (Antecedent) dan Output (Consequent)
jumlah_rep = ctrl.Antecedent(np.arange(0, 31, 1), 'jumlah_rep')
kecepatan = ctrl.Antecedent(np.arange(0, 11, 1), 'kecepatan')
intensitas = ctrl.Consequent(np.arange(0, 101, 1), 'intensitas')

# 2. Fungsi Keanggotaan (Membership Functions) menggunakan trimf
jumlah_rep['sedikit'] = fuzz.trimf(jumlah_rep.universe, [0, 0, 10])
jumlah_rep['sedang']  = fuzz.trimf(jumlah_rep.universe, [5, 15, 25])
jumlah_rep['banyak']  = fuzz.trimf(jumlah_rep.universe, [20, 30, 30])

kecepatan['cepat']  = fuzz.trimf(kecepatan.universe, [0, 0, 4])
kecepatan['sedang'] = fuzz.trimf(kecepatan.universe, [2, 5, 8])
kecepatan['lambat'] = fuzz.trimf(kecepatan.universe, [6, 10, 10])

intensitas['ringan'] = fuzz.trimf(intensitas.universe, [0, 0, 50])
intensitas['sedang'] = fuzz.trimf(intensitas.universe, [0, 50, 100])
intensitas['berat']  = fuzz.trimf(intensitas.universe, [50, 100, 100])

# 3. Pembentukan Aturan Logika Fuzzy (Fuzzy Rules)
aturan1 = ctrl.Rule(jumlah_rep['banyak'] & kecepatan['cepat'],  intensitas['berat'])
aturan2 = ctrl.Rule(jumlah_rep['banyak'] & kecepatan['sedang'], intensitas['berat'])
aturan3 = ctrl.Rule(jumlah_rep['sedang'] & kecepatan['cepat'],  intensitas['sedang'])
aturan4 = ctrl.Rule(jumlah_rep['sedang'] & kecepatan['sedang'], intensitas['sedang'])
aturan5 = ctrl.Rule(jumlah_rep['sedang'] & kecepatan['lambat'], intensitas['ringan'])
aturan6 = ctrl.Rule(jumlah_rep['sedikit'],                       intensitas['ringan'])
aturan7 = ctrl.Rule(kecepatan['lambat'],                         intensitas['ringan'])

# 4. Bangun Sistem Kontrol dan Simulasi Fuzzy
sistem_fuzzy = ctrl.ControlSystem([
    aturan1, aturan2, aturan3, aturan4, aturan5, aturan6, aturan7
])
pengendali_fuzzy = ctrl.ControlSystemSimulation(sistem_fuzzy)


def hitung_intensitas_fuzzy(total_rep, detik_per_rep):
    """
    Menghitung intensitas latihan berdasarkan total repetisi dan durasi per rep.
    Mengembalikan label: 'Ringan', 'Sedang', atau 'Berat'.
    """
    if total_rep <= 0:
        return "Ringan"

    # Batasi nilai agar tetap di dalam semesta pembicaraan (universe)
    val_rep = min(max(total_rep, 0), 30)
    val_kec = min(max(detik_per_rep, 0), 10)

    pengendali_fuzzy.input['jumlah_rep'] = val_rep
    pengendali_fuzzy.input['kecepatan']  = val_kec
    pengendali_fuzzy.compute()

    nilai_output = pengendali_fuzzy.output['intensitas']

    if nilai_output < 35:
        return "Ringan"
    elif nilai_output < 65:
        return "Sedang"
    else:
        return "Berat"


def uji_coba_fuzzy():
    """
    Fungsi pengujian simulasi sistem fuzzy secara mandiri.
    """
    print("\n" + "=" * 50)
    print("PENGUJIAN SISTEM FUZZY INTENSITAS")
    print("=" * 50)
    rep_test = 15
    kec_test = 3.0

    pengendali_fuzzy.input['jumlah_rep'] = rep_test
    pengendali_fuzzy.input['kecepatan']  = kec_test
    pengendali_fuzzy.compute()

    hasil = pengendali_fuzzy.output['intensitas']
    label = hitung_intensitas_fuzzy(rep_test, kec_test)

    print(f"Jumlah Repetisi : {rep_test} rep")
    print(f"Waktu per Rep   : {kec_test} detik")
    print(f"Nilai Defuzzifikasi : {hasil:.2f}")
    print(f"Hasil Klasifikasi   : {label}")
    print("=" * 50 + "\n")


# ============================================================
# BAGIAN 2: DETEKSI PERANGKAT (HARDWARE DETECTION)
# ============================================================

def deteksi_perangkat():
    """
    Mendeteksi apakah sistem memiliki GPU NVIDIA (CUDA)
    atau fallback menggunakan CPU (cocok untuk AMD Ryzen / Intel).
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: GPU NVIDIA Terdeteksi ({gpu_name}) - Mode CUDA")
        return "cuda"
    else:
        print("Device: Mode CPU (AMD Ryzen / Intel / Tanpa CUDA)")
        return "cpu"


# ============================================================
# BAGIAN 3: PELATIHAN MODEL YOLO (MODUL 10 & MODUL 3)
# ============================================================

def latih_model_yolo(device):
    """
    Melakukan fine-tuning model YOLOv11 pada dataset workout.
    """
    print("\n" + "=" * 50)
    print("PELATIHAN MODEL YOLOV11 (FINE-TUNING)")
    print("=" * 50)

    yaml_path = "workout.v12i.yolov11/data.yaml"
    if not os.path.exists(yaml_path):
        print(f"Error: File konfigurasi dataset '{yaml_path}' tidak ditemukan.")
        print("Pastikan folder dataset 'workout.v12i.yolov11' tersedia di folder ini.")
        return

    model_base = "yolo11n.pt" if device == "cpu" else "yolo11m.pt"
    print(f"Memuat base model: {model_base}")
    model = YOLO(model_base)
    model.to(device)

    epochs_pilihan = input("Masukkan jumlah epoch (default 10 untuk CPU, 30 untuk GPU): ").strip()
    if not epochs_pilihan.isdigit():
        epochs = 10 if device == "cpu" else 30
    else:
        epochs = int(epochs_pilihan)

    print(f"Memulai training sebanyak {epochs} epoch pada device: {device}...")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8 if device == "cpu" else 16,
        name="workout_detector",
        device=device
    )

    print("\nTraining selesai.")
    print("Bobot terbaik tersimpan di: runs/detect/workout_detector/weights/best.pt\n")


# ============================================================
# BAGIAN 4: MENU PILIHAN SUMBER VIDEO & MODEL
# ============================================================

def pilih_model_yolo(device):
    """
    Menampilkan menu pemilihan bobot model YOLO.
    """
    path_custom = "runs/detect/workout_detector/weights/best.pt"

    print("\n" + "=" * 45)
    print("PILIH MODEL YOLO")
    print("=" * 45)
    print("1. Model Custom (runs/.../best.pt)")
    print("2. yolo11n.pt  (Nano - Paling Ringan, Rekomendasi CPU)")
    print("3. yolo11s.pt  (Small - Ringan)")
    print("4. yolo11m.pt  (Medium - Standar Praktikum)")
    print("5. yolo11l.pt  (Large)")
    print("6. yolo11x.pt  (Extra Large)")
    print("=" * 45)

    pilihan = input("Pilihan model (1-6, default 2 untuk CPU): ").strip()

    daftar_model = {
        "1": path_custom,
        "2": "yolo11n.pt",
        "3": "yolo11s.pt",
        "4": "yolo11m.pt",
        "5": "yolo11l.pt",
        "6": "yolo11x.pt"
    }

    if pilihan not in daftar_model:
        pilihan = "2" if device == "cpu" else "4"

    target_model = daftar_model[pilihan]

    # Cek ketersediaan file custom model
    if pilihan == "1" and not os.path.exists(path_custom):
        print(f"\nModel custom '{path_custom}' belum ada.")
        print("Beralih ke model pretrained yolo11n.pt (atau yolo11m.pt)...\n")
        target_model = "yolo11n.pt" if device == "cpu" else "yolo11m.pt"

    print(f"Model terpilih: {target_model}")
    model = YOLO(target_model)
    model.to(device)
    return model


def pilih_sumber_video():
    """
    Menampilkan menu pemilihan sumber aliran video (OpenCV VideoCapture).
    """
    print("\n" + "=" * 45)
    print("PILIH SUMBER VIDEO INPUT")
    print("=" * 45)
    print("1. File Video (MP4 / AVI / MOV)")
    print("2. Webcam Internal Laptop")
    print("3. Kamera Eksternal (USB)")
    print("=" * 45)

    pilihan = input("Masukkan pilihan (1 / 2 / 3): ").strip()

    if pilihan == "1":
        path_video = input("Masukkan nama berkas video (contoh: workout.mp4): ").strip()
        if not os.path.exists(path_video):
            print(f"Error: Berkas video '{path_video}' tidak ditemukan.")
            return None, None
        cap = cv2.VideoCapture(path_video)
        sumber_info = f"File: {path_video}"

    elif pilihan == "2":
        cap = cv2.VideoCapture(0)
        sumber_info = "Webcam Internal (Index 0)"

    elif pilihan == "3":
        indeks = input("Masukkan indeks kamera USB (default 1): ").strip()
        cam_idx = int(indeks) if indeks.isdigit() else 1
        cap = cv2.VideoCapture(cam_idx)
        sumber_info = f"Kamera Eksternal (Index {cam_idx})"

    else:
        print("Pilihan tidak valid.")
        return None, None

    if not cap.isOpened():
        print(f"Error: Gagal membuka sumber video ({sumber_info}).")
        return None, None

    return cap, sumber_info


# ============================================================
# BAGIAN 5: INFERENSI & PENGHITUNG REPETISI (MODUL 10, 9, 6)
# ============================================================

def jalankan_deteksi_workout(device):
    """
    Pipeline utama: membaca video frame-by-frame, inferensi YOLO,
    tracking bounding box untuk hitung repetisi, klasifikasi fuzzy,
    serta menampilkan overlay visual.
    """
    model = pilih_model_yolo(device)
    cap, sumber_info = pilih_sumber_video()

    if cap is None:
        return

    # Inisialisasi VideoWriter untuk menyimpan video keluaran
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_in if fps_in and fps_in > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_filename = "output_workout.mp4"
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Parameter Threshold & State Tracking
    CONFIDENCE_MIN = 0.35
    THRESHOLD_GERAK = 18

    rep_count = 0
    state = None
    prev_y = None
    waktu_mulai = time.time()
    label_intensitas = "-"

    print("\n" + "=" * 50)
    print(f"Deteksi aktif pada: {sumber_info}")
    print("Tekan tombol 'q' pada jendela video untuk berhenti.")
    print("=" * 50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Inferensi Objek dengan YOLO (Modul 10)
        results = model(frame)

        for result in results:
            boxes = result.boxes

            for box in boxes:
                confidence = float(box.conf[0])

                if confidence >= CONFIDENCE_MIN:
                    # Ambil koordinat piksel bounding box
                    x1, y1, x2, y2 = np.asarray(box.xyxy[0]).astype(int)

                    # 2. Perhitungan Titik Tengah Y (Modul 4 - Preprocessing)
                    center_y = (y1 + y2) // 2

                    # 3. Logika State Machine Rep Counter
                    if prev_y is not None:
                        if center_y < prev_y - THRESHOLD_GERAK:
                            state_baru = "ATAS"
                        elif center_y > prev_y + THRESHOLD_GERAK:
                            state_baru = "BAWAH"
                        else:
                            state_baru = state

                        # Transisi BAWAH -> ATAS dihitung sebagai 1 repetisi penuh
                        if state == "BAWAH" and state_baru == "ATAS":
                            rep_count += 1
                            print(f"[Counter] Repetisi ke-{rep_count} tercatat!")

                            # Update klasifikasi intensitas fuzzy secara berkala
                            waktu_berjalan = time.time() - waktu_mulai
                            durasi_per_rep = waktu_berjalan / rep_count
                            label_intensitas = hitung_intensitas_fuzzy(rep_count, durasi_per_rep)

                        state = state_baru
                    else:
                        state = "ATAS"

                    prev_y = center_y

                    # 4. Gambar Bounding Box & Label (Modul 9 - OpenCV)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_box = f"Workout {confidence * 100:.0f}%"
                    cv2.putText(frame, label_box, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)

                    # Label status posisi gerakan
                    cv2.putText(frame, f"Posisi: {state}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)

        # 5. Overlay Panel Informasi pada Frame (Pojok Kiri Atas)
        cv2.rectangle(frame, (8, 8), (320, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"REPETISI   : {rep_count}", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"INTENSITAS : {label_intensitas}", (18, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"DEVICE     : {device.upper()}", (18, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Simpan frame ke berkas keluaran
        out.write(frame)

        # Tampilkan frame pada jendela OpenCV
        cv2.imshow("Workout Repetition Counter - Tekan 'q' untuk keluar", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_waktu = time.time() - waktu_mulai

    # Ringkasan Akhir Sesi
    print("\n" + "=" * 50)
    print("RINGKASAN SESI LATIHAN")
    print("=" * 50)
    print(f"Total Repetisi    : {rep_count} rep")
    print(f"Durasi Total      : {total_waktu:.1f} detik")

    if rep_count > 0:
        rata_per_rep = total_waktu / rep_count
        intensitas_akhir = hitung_intensitas_fuzzy(rep_count, rata_per_rep)
        print(f"Rata-rata/Rep     : {rata_per_rep:.2f} detik")
        print(f"Intensitas Akhir  : {intensitas_akhir}")

    print(f"Video Hasil Rekaman: {output_filename}")
    print("=" * 50 + "\n")

    # Bersihkan resource OpenCV
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ============================================================
# PROGRAM UTAMA: MENU PILIHAN AKSI
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("SISTEM DETEKSI GERAKAN WORKOUT & REPETISI (UAS AI)")
    print("=" * 55)

    # Deteksi hardware
    device_aktif = deteksi_perangkat()

    while True:
        print("\nMENU UTAMA:")
        print("1. Jalankan Deteksi Workout & Hitung Repetisi")
        print("2. Training / Fine-Tune Model YOLOv11 (Opsional)")
        print("3. Uji Coba Simulasi Fuzzy Logic Intensitas")
        print("4. Keluar")
        print("-" * 55)

        pilihan_menu = input("Pilih menu (1/2/3/4): ").strip()

        if pilihan_menu == "1":
            jalankan_deteksi_workout(device_aktif)
        elif pilihan_menu == "2":
            latih_model_yolo(device_aktif)
        elif pilihan_menu == "3":
            uji_coba_fuzzy()
        elif pilihan_menu == "4":
            print("\nProgram selesai. Terima kasih.\n")
            break
        else:
            print("Pilihan tidak valid. Silakan masukkan angka 1 sampai 4.")
