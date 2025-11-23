# 🚗 AutoCluster AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-Web%20App-green) ![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

**AutoCluster AI** adalah aplikasi web berbasis Machine Learning untuk mengelompokkan (clustering) data mobil dan memprediksi kategori mobil baru berdasarkan fitur sederhana seperti *price* dan *seats*. Proyek ini menggabungkan pendekatan *unsupervised* (K‑Means) untuk pelabelan awal dan *supervised* (Random Forest) untuk membuat prediksi yang dapat dipakai pada aplikasi web.

---

## 🔎 Ringkasan singkat

* **Input utama:** `price` (float) dan `seats` (int)
* **Output utama:** label kategori mobil (mis. `Super Luxury Sport`, `Family Compact`, dll.)
* **Pipeline:** preprocessing → K‑Means (auto‑label) → sampling hybrid → Random Forest → model `.pkl`
* **Validasi:** Stratified 10‑Fold CV, laporan metrik (accuracy, precision, recall, f1), Silhouette & Davies‑Bouldin untuk clustering.

---

## 📚 Table of Contents

1. [Fitur Utama](#-fitur-utama)
2. [Arsitektur & Metode](#-arsitektur--metode)
3. [Instalasi](#-instalasi)
4. [Jalankan Aplikasi](#-jalankan-aplikasi)
5. [Contoh Penggunaan](#-contoh-penggunaan)
6. [Format Dataset](#-format-dataset)
7. [Training ulang model](#-training-ulang-model)
8. [Evaluasi & Analisis](#-evaluasi--analisis)
9. [Deploy & Produksi](#-deploy--produksi)

---

## ✨ Fitur Utama

* Dashboard web untuk eksplorasi dataset hasil clustering (tabel interaktif, pagination 50/halaman).
* Sistem *auto‑labeling* yang menerjemahkan cluster numerik ke label deskriptif berdasar centroid (rata‑rata harga & kursi).
* Prediktor real‑time: masukkan `price` dan `seats`, keluarkan kategori menggunakan model Random Forest terlatih.
* Penanganan ketidakseimbangan: kombinasi RandomOverSampler + SMOTE (aturan threshold untuk cluster kecil).
* Model final disimpan sebagai `.pkl` untuk inference di Flask.

---

## 🧠 Arsitektur & Metodologi (detail)

### 1. Preprocessing

* Bersihkan kolom `price` dan `seats` (hapus nilai null / konversi format string ke numeric).
* Standarisasi dengan `StandardScaler` (Z‑score) sebelum K‑Means.
* Analisis outlier (IQR) — opsi untuk trimming atau capping.

### 2. Clustering (K‑Means)

* Gunakan Elbow Method & Silhouette untuk menentukan K (di proyek ini K=6).
* Simpan centroid dan statistik cluster (mean price, mean seats, size).
* Hasil cluster digunakan sebagai "label" sementara untuk model supervised.

### 3. Sampling (Hybrid)

* Rule: jika cluster size < 7 → RandomOverSampler; else gunakan SMOTE untuk memperhalus distribusi.
* Tujuan: mencegah bias classifier terhadap cluster mayoritas sekaligus menjaga varian fitur.

### 4. Classification (Random Forest)

* Fitur: `price`, `seats` (bisa ditambah fitur turunan seperti price_per_seat, log(price)).
* Validasi: Stratified 10‑Fold CV.
* Pilihan hyperparam (grid/random search): `n_estimators`, `max_depth`, `min_samples_leaf`.
* Simpan model terbaik sebagai `model.pkl`.

---

## 🚀 Instalasi

1. Clone repo

```bash
git clone <repo-url>
cd autocluster-ai
```

2. Buat virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\\Scripts\\activate   # Windows
```

3. Install dependency

```bash
pip install -r requirements.txt
# atau manual:
# pip install flask pandas numpy scikit-learn matplotlib imbalanced-learn

---

## ▶️ Menjalankan Aplikasi (development)

```bash
# Pastikan model.pkl berada di folder model/
python app.py
# Akses: http://127.0.0.1:5000
```

---

## ⚙️ Contoh Penggunaan

### 1. Web UI

* Buka `http://localhost:5000` → Dashboard → Prediction form.

### 2. API (contoh endpoint)

* **POST** `/predict`

  * Body JSON: `{ "price": 75000, "seats": 2 }`
  * Response: `{ "label": "Super Luxury Sport", "probability": 0.92 }`

### 3. CLI sederhana (opsional)

```bash
python predict_cli.py --price 45000 --seats 5
# Output: Predicted cluster: Family Compact (0.86)
```

---

## 📥 Format Dataset

Contoh CSV minimal (header):

```
price,seats,other_feature_1,other_feature_2
63090,4,...
45000,5,...
```

Pastikan kolom `price` numeric (tanpa simbol), `seats` integer.

---

## 🔁 Training ulang model atau Explore Lanjutan

Langkah singkat:

1. Buat `train.py` dengan dataset baru.
2. Script akan melakukan: cleaning → scaling → K‑Means → hybrid sampling → CV → training → simpan `model.pkl`.

Contoh:

```bash
python train.py --data data/cars.csv --k 6 --out models/model.pkl
```

Argument opsional: `--k`, `--cv-folds`, `--seed`, `--save-figs`.

---
