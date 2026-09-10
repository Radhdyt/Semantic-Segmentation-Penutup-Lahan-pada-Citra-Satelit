# Semantic Segmentation Penutup Lahan pada Citra Satelit

Implementasi model Deep Learning (arsitektur **U-Net**) resolusi tinggi untuk segmentasi semantik tutupan lahan (Land Cover) dari citra satelit berbasis dataset **LandCover.ai**.

Dibuat untuk keperluan akademis (Skripsi/Tugas Akhir): pipeline Python murni, bersih, terstruktur, dan reproducible tanpa GUI.

## Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Model | U-Net (PyTorch) |
| Input | Citra satelit (JPG/TIF), mask PNG/TIF |
| Dataset | Format LandCover.ai (train/val/test split) |
| Inference | Sliding window patch-based (tanpa resize, jaga resolusi HD) |
| Visualisasi | Automatic overlay side-by-side (alpha-blended) |

## Fitur Utama

- **U-Net Architecture**: arsitektur standar yang terbukti andal untuk fitur spasial geografis.
- **Sliding Window Inference**: tidak me-resize gambar asli, melainkan memindai patch-by-patch untuk menjaga ketajaman bangunan/jalan.
- **Auto-Save Checkpoints**: training bisa dijeda/dilanjutkan (resume) dari epoch terakhir.
- **Automatic Overlay Visualization**: hasil prediksi disandingkan dengan gambar asli (subplot + alpha-blend).

## Struktur Kelas Area (Warna)

| Label | Kelas | Warna |
|-------|-------|-------|
| 0 | Pasir/Background/Jalan | Kuning |
| 1 | Bangunan | Coklat |
| 2 | Vegetasi/Hutan | Hijau |
| 3 | Perairan/Lautan | Biru |

---

## 🚀 Cara Menjalankan (Getting Started)

### 1. Persiapan Lingkungan (Prerequisites)

Python 3.8+. Disarankan GPU (NVIDIA CUDA) agar training cepat.

```bash
pip install -r requirements.txt
```

### 2. Persiapan Dataset

Struktur folder `dataset/`:

```text
dataset/
├── images/             # (Jika ada gambar mentah)
├── masks/              # (Jika ada mask mentah)
├── output/             # Folder hasil crop/split ukuran 256x256 (PENTING)
├── train.txt           # Berisi list nama file untuk dilatih (tanpa ekstensi)
├── val.txt             # Berisi list nama file untuk divalidasi
└── test.txt            # Berisi list nama file untuk diuji
```

_Note: `unet_segmentation_pipeline.py` otomatis membaca file `.jpg` dan `_m.png` dari `dataset/output/`._

### 3. Melatih Model (Training)

Train dari awal atau lanjutkan checkpoint yang terhenti:

```bash
python unet_segmentation_pipeline.py
```

Progress tersimpan di `dataset/checkpoints/unet_checkpoint.pth`.

### 4. Uji Coba Model (Inference)

1. Masukkan gambar satelit baru ke folder `input_ujicoba/`.
2. Jalankan skrip inference (`python.py`):

```bash
python python.py
```

3. Visualisasi side-by-side muncul dengan detail zoom-in akurasi model. Tutup jendela (`X`) untuk lanjut ke gambar berikutnya.

---

## 📝 Catatan untuk Repositori Clone

- File besar (dataset gambar ribuan file, checkpoint `.pth`) diabaikan via `.gitignore`.
- Setelah clone, sediakan dataset sendiri dan latih ulang model dengan `unet_segmentation_pipeline.py`.