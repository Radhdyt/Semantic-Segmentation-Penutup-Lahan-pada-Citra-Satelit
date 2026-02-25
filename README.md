# Semantic Segmentation Penutup Lahan pada Citra Satelit

Repositori ini berisi implementasi lengkap model Deep Learning (Arsitektur U-Net) ber-resolusi tinggi untuk melakukan segmentasi semantik tutupan lahan (Land Cover) menggunakan citra satelit (seperti dataset LandCover.ai).

Sistem ini dirancang khusus untuk keperluan akademis (Skripsi/Tugas Akhir) sehingga murni berbasis pipeline Python yang bersih, terstruktur, dan mudah diproduksi ulang (reproducible) tanpa antarmuka (GUI) yang memakan memori.

## Fitur Utama

- **U-Net Architecture**: Model standar yang sudah terbukti andal dalam membedakan fitur spasial geografis.
- **Sliding Window Inference**: Skrip uji coba tidak melakukan _resize_ pada gambar asli yang besar, melainkan memindai berulang kali lapis demi lapis (_patch-based_) untuk menjaga resolusi bangunan/jalan tetap HD dan tajam.
- **Auto-Save Checkpoints**: Pelatihan model dapat dijeda/dihentikan di tengah jalan dan akan otomatis dilanjutkan (_resume_) dari epoch terakhir yang berhasil diselesaikan.
- **Automatic Overlay Visualization**: Hasil tebakan langsung disandingkan (_subplot_) secara berdampingan dengan gambar aslinya dengan _alpha-blended overlay_.

## Struktur Kelas Area (Warna)

Model ini secara bawaan dilatih untuk mendeteksi 4 kelas area utama:

1. **Pasir/Background/Jalan** (Kuning) - Label `0`
2. **Bangunan** (Coklat) - Label `1`
3. **Vegetasi/Hutan** (Hijau) - Label `2`
4. **Perairan/Lautan** (Biru) - Label `3`

---

## 🚀 Cara Menjalankan (Getting Started)

### 1. Persiapan Lingkungan (Prerequisites)

Pastikan Python 3.8+ sudah terinstall. Sangat disarankan untuk menggunakan GPU (NVIDIA CUDA) agar proses training berjalan cepat.

Install semua _library_ yang dibutuhkan dengan perintah berikut:

```bash
pip install -r requirements.txt
```

### 2. Persiapan Dataset

Siapkan dataset Anda (format `.jpg`/`.tif` untuk gambar dan `.png`/`.tif` untuk mask) lalu susun ke dalam folder `dataset/` dengan struktur seperti ini:

```text
dataset/
├── images/             # (Jika ada gambar mentah)
├── masks/              # (Jika ada mask mentah)
├── output/             # Folder hasil crop/split ukuran 256x256 (PENTING)
├── train.txt           # Berisi list nama file untuk dilatih (tanpa ekstensi)
├── val.txt             # Berisi list nama file untuk divalidasi
└── test.txt            # Berisi list nama file untuk diuji
```

_Note: Script `unet_segmentation_pipeline.py` akan otomatis membaca file `.jpg` dan `_m.png` dari dalam folder `dataset/output/`._

### 3. Melatih Model (Training)

Untuk mulai melatih model dari awal (atau melanjutkan _checkpoint_ yang terhenti), jalankan skrip utama:

```bash
python unet_segmentation_pipeline.py
```

Model akan menyimpan _progress_-nya di dalam folder `dataset/checkpoints/unet_checkpoint.pth`.

### 4. Uji Coba Model (Inference)

Jika Anda ingin menguji kepintaran model pada gambar pulau yang baru (di luar dataset awal):

1. Masukkan gambar-gambar satelit Anda ke dalam folder `input_ujicoba/`.
2. Jalankan skrip inference:

```bash
python python.py
```

3. Jendela visualisasi _side-by-side_ akan muncul menampilkan detail _Zoom-in_ akurasi model. Tutup jendela (`X`) untuk beralih ke gambar pengujian selanjutnya secara otomatis.

---

## 📝 Catatan Tambahan untuk Repositori Clone

- File berukuran raksasa seperti _dataset_ gambar ribuan file dan _checkpoint_ bobot AI (`.pth`) telah diabaikan (`.gitignore`) agar tidak membebani proses _upload/clone_ GitHub.
- Jika Anda melakukan _clone_, Anda harus menyediakan dataset dan melatih ulang modelnya dari awal di PC Anda sendiri menggunakan `unet_segmentation_pipeline.py`.
