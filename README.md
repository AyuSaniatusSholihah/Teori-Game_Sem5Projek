# Analisis Pengaruh Media Sosial terhadap Kesehatan Mental Menggunakan Accumulated Local Effects (ALE)

> Repositori kode penelitian untuk makalah format **IEEE**  
> Program Studi · Semester 5 · Mata Kuliah Teori Game

---

## Daftar Isi
1. [Deskripsi Penelitian](#1-deskripsi-penelitian)
2. [Rumusan dan Tujuan Penelitian](#2-rumusan-dan-tujuan-penelitian)
3. [Struktur Proyek](#3-struktur-proyek)
4. [Persyaratan Sistem](#4-persyaratan-sistem)
5. [Cara Menjalankan](#5-cara-menjalankan)
6. [Pipeline Penelitian](#6-pipeline-penelitian)
7. [Status Pekerjaan](#7-status-pekerjaan)
8. [Hasil yang Sudah Diperoleh](#8-hasil-yang-sudah-diperoleh)
9. [Ketidaksesuaian dengan Draf Makalah](#9-ketidaksesuaian-dengan-draf-makalah)
10. [Catatan Metodologis](#10-catatan-metodologis)

---

## 1. Deskripsi Penelitian

Penelitian ini menganalisis pengaruh perilaku penggunaan media sosial terhadap skor gangguan suasana hati (*mood disorder*) menggunakan data survei swalaporan dari **513 responden** (`cleaned_data-2.csv`).

Model prediksi dilatih untuk memprediksi variabel target `mood_disorder` (skala Likert 1–5) dari **31 variabel independen** swalaporan. Model terbaik kemudian diinterpretasikan menggunakan:
- **Gini Importance** — untuk mengidentifikasi fitur paling berperan secara global.
- **Accumulated Local Effects (ALE) 1D** — untuk mengestimasi bentuk dan besar efek lokal tiap indikator penggunaan media sosial secara akurat, bebas dari bias korelasi antar fitur yang menjadi kelemahan metode PDP (*Partial Dependence Plot*).

**Kontribusi makalah:**
1. Perbandingan tiga model regresi pada data swalaporan.
2. Estimasi kurva ALE 1D dengan **implementasi mandiri** (*dari nol tanpa library ALE eksternal*) yang dirancang khusus untuk fitur berskala Likert diskret.
3. Pembahasan batasan inferensi dari data swalaporan.

---

## 2. Rumusan dan Tujuan Penelitian

### Rumusan Masalah
1. Model regresi mana — di antara **Ridge Regression**, **Gradient Boosting**, dan **Random Forest** — yang memberikan kinerja prediksi terbaik untuk skor `mood_disorder`?
2. Fitur apa yang paling berperan pada model terbaik menurut **Gini Importance**?
3. Bagaimana **bentuk dan besar efek** empat indikator penggunaan media sosial terhadap prediksi skor menurut kurva **ALE**?

### Tujuan Penelitian
- Membandingkan kinerja ketiga model pada data uji menggunakan metrik RMSE dan $R^2$.
- Mengidentifikasi fitur terpenting pada model terbaik melalui Gini Importance.
- Mengestimasi kurva ALE 1 dimensi untuk empat indikator kunci media sosial.

---

## 3. Struktur Proyek

```
sc/sc/
├── ALE.py                        ← Script utama (runner): orkestrasi seluruh pipeline
├── ale_core.py                   ← Modul inti komputasi: data, model ML, Gini, dan ALE
├── ale_visualize.py              ← Modul visualisasi: fungsi plot & simpan gambar grafik
├── pdp_vs_ale_illustration.py    ← Script ilustrasi konseptual PDP vs ALE
├── cleaned_data-2.csv            ← Dataset (513 responden, 32 kolom)
│
├── ale_curves.png                ← [OUTPUT] Grafik kurva ALE 4 fitur kunci
├── feature_importance.png        ← [OUTPUT] Grafik Gini Importance Top 10
├── pred_vs_actual.png            ← [OUTPUT] Grafik prediksi vs aktual (RF)
├── residual_distribution.png     ← [OUTPUT] Distribusi residual (RF)
├── pdp_vs_ale_illustration.png   ← [OUTPUT] Ilustrasi konseptual PDP vs ALE
└── BRIEFING_PENELITIAN_ALE.md    ← Dokumentasi teknis algoritma & verifikasi
```

---

## 4. Persyaratan Sistem

| Komponen | Versi yang Digunakan |
|:---|:---|
| Python | 3.14.5 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| matplotlib | (terinstal) |

**Instalasi dependensi:**
```powershell
pip install pandas numpy scikit-learn matplotlib
```

> **Catatan:** Tidak diperlukan library ALE eksternal (PyALE, alibi, DALEX, dsb.). Algoritma ALE diimplementasikan secara mandiri di dalam `ALE.py`.

---

## 5. Cara Menjalankan

### Menjalankan Pipeline Utama (ALE.py)
```powershell
cd "D:\Nia\Kuliah\SEM 5\teori game\sc\sc"
python ALE.py
```
Atau tekan tombol **▶ Run** di VS Code saat file `ALE.py` terbuka.

**Output yang dihasilkan:**
- Statistik deskriptif 5 variabel kunci di terminal.
- Tabel kinerja model (RMSE dan $R^2$) di terminal.
- Tabel Top 5 Gini Importance di terminal.
- Tabel nilai ALE skala 1–5 + narasi efek untuk 4 fitur di terminal.
- `ale_curves.png` — grafik kurva ALE 2×2.
- `feature_importance.png` — bar chart Gini Importance Top 10.
- `pred_vs_actual.png` — scatter plot prediksi vs nilai aktual (RF).
- `residual_distribution.png` — histogram distribusi residual (RF).

### Menjalankan Ilustrasi Konseptual PDP vs ALE
```powershell
python pdp_vs_ale_illustration.py
```
**Output:** File `pdp_vs_ale_illustration.png` — gambar penjelasan perbedaan metode PDP (yang tidak realistis untuk data berkorelasi) dan ALE (yang hanya menggeser observasi lokal dalam bin).

---

## 6. Pipeline Penelitian

```
[cleaned_data-2.csv]  513 baris × 32 kolom
         │
         ▼
[Pemisahan Variabel]
  ├── Target y   : mood_disorder (Likert 1–5)
  └── Fitur  X   : 31 kolom lainnya (semua int64)
         │
         ▼
[train_test_split]  80% latih (410) | 20% uji (103)  random_state=42
         │
         ▼
[Pelatihan 3 Model Regresi pada Data Latih]
  ├── Ridge Regression         alpha=1.0
  ├── Gradient Boosting        n_estimators=100, learning_rate=0.05
  └── Random Forest            n_estimators=300
         │
         ▼
[Evaluasi pada Data Uji]  → RMSE & R²
  └── Model terbaik: Random Forest (RMSE terkecil)
         │
         ├────────────────────────────────────┐
         ▼                                    ▼
[Gini Feature Importance]           [ALE 1D — 4 Fitur Kunci]
  Seluruh 31 fitur diurutkan          social_media_sleep
  berdasarkan kontribusi              focus_quality
  penurunan impuritas (MDI)           compare_with_others
                                      sad_content_feed
         │                                    │
         ▼                                    ▼
[feature_importance.png]           [ale_curves.png]
```

**Pseudocode Algoritma ALE (implementasi di `ALE.py` Baris 81–131):**
```
Masukan : model f̂ (RF terlatih), data latih X, fitur j, grid z = [1, 2, 3, 4, 5]
Keluaran: ALE(v) untuk v = 1, …, 5

 1: ALE_raw[1] ← 0
 2: untuk m = 1 sampai 4 lakukan
 3:   S_m ← {i : z_m ≤ x_ij < z_(m+1)}; untuk m=4 gunakan x_ij ≤ z_5
 4:   jika |S_m| = 0 maka Δ_m ← 0
 5:   selain itu
 6:     X_high ← X[S_m] dengan x_j = z_(m+1)
        X_low  ← X[S_m] dengan x_j = z_m
 7:     Δ_m ← rata-rata( f̂(X_high) − f̂(X_low) )
 8:   ALE_raw[m+1] ← ALE_raw[m] + Δ_m
 9: P(v) ← frekuensi relatif x_j = v pada X, untuk v = 1, …, 5
10: ḡ   ← Σ_v ALE_raw[v] · P(v)
11: ALE(v) ← ALE_raw[v] − ḡ, untuk v = 1, …, 5
12: kembalikan ALE
```

---

## 7. Status Pekerjaan

### ✅ Sudah Selesai Dikerjakan

| No | Pekerjaan | File / Bukti |
|:---:|:---|:---|
| 1 | Pemuatan dataset `cleaned_data-2.csv` (513 baris, 0 missing value) | `ALE.py` Baris 17–18 |
| 2 | Pemisahan fitur ($X$, 31 kolom) dan target ($y$: `mood_disorder`) | `ALE.py` Baris 26–27 |
| 3 | Pembagian data 80:20 dengan `random_state=42` | `ALE.py` Baris 32–34 |
| 4 | Pelatihan **Ridge Regression** (`alpha=1.0`) | `ALE.py` Baris 43 |
| 5 | Pelatihan **Gradient Boosting** (`n_estimators=100, lr=0.05`) | `ALE.py` Baris 44 |
| 6 | Pelatihan **Random Forest** (`n_estimators=300`) | `ALE.py` Baris 45 |
| 7 | Evaluasi 3 model: RMSE & $R^2$ pada data uji | `ALE.py` Baris 50–57 |
| 8 | Pemilihan model terbaik otomatis berdasarkan RMSE terkecil | `ALE.py` Baris 57 |
| 9 | Kalkulasi **Gini Importance** untuk seluruh 31 fitur | `ALE.py` Baris 65–73 |
| 10 | Implementasi mandiri algoritma **ALE 1D** (tanpa library eksternal) | `ALE.py` Baris 81–131 |
| 11 | Kalkulasi & cetak nilai ALE skala 1–5 untuk 4 fitur kunci | `ALE.py` Baris 170–188 |
| 12 | **Narasi otomatis** interpretasi efek ALE tiap skala | `ALE.py` Baris 183–187 |
| 13 | **Statistik deskriptif** (Min, Max, Mean, SD) 5 variabel kunci di kode | `ALE.py` Baris 32–50 |
| 14 | **Visualisasi kurva ALE** (grid 2×2, 300 DPI, siap IEEE) | `ale_curves.png` |
| 15 | **Visualisasi Gini Importance** (bar chart horizontal Top 10) | `feature_importance.png` |
| 16 | **Visualisasi Prediksi vs Aktual** scatter plot Random Forest | `pred_vs_actual.png` |
| 17 | **Visualisasi Distribusi Residual** histogram RF | `residual_distribution.png` |
| 18 | **Ilustrasi konseptual PDP vs ALE** (gambar berkorelasi) | `pdp_vs_ale_illustration.py` / `.png` |
| 19 | Dokumentasi teknis & verifikasi kode vs draf makalah | `BRIEFING_PENELITIAN_ALE.md` |
| 20 | Verifikasi pseudocode algoritma ALE (100% sesuai kode) | — |

### ⚠️ Belum Dikerjakan / Perlu Perhatian

| No | Pekerjaan | Prioritas | Keterangan |
|:---:|:---|:---:|:---|
| 1 | Normalisasi fitur untuk Ridge Regression | 🟡 Sedang | Tanpa `StandardScaler`, Ridge menghasilkan $R^2 = -0.088$. Sudah ada disclaimer di kode (Baris 72–79). Perlu disebutkan sebagai keterbatasan di subbab Metode makalah. |
| 2 | Interval kepercayaan / bootstrap ALE | 🟢 Opsional | Implementasi ALE saat ini tidak menyertakan bootstrap CI — perlu disebutkan sebagai batasan penelitian. |
| 3 | Cross-validation / hyperparameter tuning | 🟢 Opsional | Tidak ada CV saat ini. Sudah ada disclaimer di kode (Baris 61–68). Perlu disebutkan secara eksplisit di subbab Metode. |

---

## 8. Hasil yang Sudah Diperoleh

### Kinerja Model pada Data Uji
| Peringkat | Model | RMSE | $R^2$ |
|:---:|:---|:---:|:---:|
| 1 | **Random Forest** | **1.3012** | **0.0487** |
| 2 | Gradient Boosting | 1.3261 | 0.0118 |
| 3 | Ridge Regression | 1.3913 | -0.0878 |

### Top 5 Gini Importance (Random Forest, seluruh 31 fitur)
| Peringkat | Fitur | Gini Importance (%) |
|:---:|:---|:---:|
| 1 | `anxious_stressed` | 16.44% |
| 2 | `social_media_sleep` | 14.52% |
| 3 | `focus_quality` | 9.01% |
| 4 | `nervous_anxious` | 5.27% |
| 5 | `content_type_social_media` | 4.81% |

### Nilai Kurva ALE (Random Forest, data latih X_train)
| Fitur | Skala 1 | Skala 2 | Skala 3 | Skala 4 | Skala 5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `social_media_sleep` | -0.3807 | -0.3591 | -0.0971 | +0.1717 | +0.4679 |
| `focus_quality` | -0.1483 | -0.1118 | -0.0591 | +0.0291 | +0.2398 |
| `compare_with_others` | -0.0500 | -0.0075 | -0.0029 | +0.0250 | +0.0933 |
| `sad_content_feed` | -0.1840 | -0.0155 | +0.0122 | +0.0470 | +0.0601 |

> Nilai ALE positif → prediksi `mood_disorder` meningkat (memburuk).  
> Nilai ALE negatif → prediksi `mood_disorder` menurun (membaik).  
> ALE dikompensasikan sehingga $\mathbb{E}[ALE(X_j)] = 0$.

---

## 9. Angka Resmi Penelitian (Acuan Penulisan Makalah)

> [!IMPORTANT]
> **Kode adalah sumber kebenaran (*ground truth*).** Angka di bawah ini adalah hasil aktual dari `ALE.py` dengan 31 fitur lengkap dan merupakan angka yang harus digunakan di seluruh bagian makalah IEEE.

**Konfigurasi fitur yang dipakai:** Seluruh 31 variabel independen diikutsertakan — termasuk `anxious_stressed` dan `nervous_anxious` — karena keduanya merupakan bagian sah dari data swalaporan dan berkontribusi terhadap prediksi `mood_disorder`.

### Angka Resmi: Kinerja Model
| Peringkat | Model | RMSE | $R^2$ |
|:---:|:---|:---:|:---:|
| **1** | **Random Forest** | **1.3012** | **0.0487** |
| 2 | Gradient Boosting | 1.3261 | 0.0118 |
| 3 | Ridge Regression | 1.3913 | -0.0878 |

### Angka Resmi: Gini Importance (Top 5)
| Peringkat | Fitur | Gini (%) |
|:---:|:---|:---:|
| 1 | `anxious_stressed` | 16.44% |
| 2 | `social_media_sleep` | 14.52% |
| 3 | `focus_quality` | 9.01% |
| 4 | `nervous_anxious` | 5.27% |
| 5 | `content_type_social_media` | 4.81% |

### Angka Resmi: Nilai ALE (skala 1–5)
| Fitur | Skala 1 | Skala 2 | Skala 3 | Skala 4 | Skala 5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `social_media_sleep` | -0.3807 | -0.3591 | -0.0971 | +0.1717 | +0.4679 |
| `focus_quality` | -0.1483 | -0.1118 | -0.0591 | +0.0291 | +0.2398 |
| `compare_with_others` | -0.0500 | -0.0075 | -0.0029 | +0.0250 | +0.0933 |
| `sad_content_feed` | -0.1840 | -0.0155 | +0.0122 | +0.0470 | +0.0601 |

### Angka Resmi: Statistik Deskriptif (dari dataset)
| Variabel | Mean | SD |
|:---|:---:|:---:|
| `mood_disorder` | 3.06 | 1.35 |
| `social_media_sleep` | 3.16 | 1.29 |
| `focus_quality` | 3.15 | 1.38 |
| `compare_with_others` | 2.76 | 1.30 |
| `sad_content_feed` | 3.41 | 1.45 |

---

## 10. Catatan Metodologis

1. **ALE vs PDP:** Implementasi ALE (bukan PDP) dipilih karena data swalaporan kemungkinan besar memiliki korelasi antar fitur. ALE menghasilkan efek marjinal yang realistis dengan hanya menggeser observasi di dalam bin lokal $[z_{k-1}, z_k]$, bukan memaksa seluruh data ke satu nilai seperti PDP. Lihat ilustrasi: `pdp_vs_ale_illustration.png`.

2. **Skala Likert Diskret:** Implementasi ALE dirancang khusus untuk fitur ordinal Likert 1–5, menggunakan bin setengah terbuka $[z_{k-1}, z_k)$ pada bin 1–3 dan inklusif $[z_{k-1}, z_k]$ pada bin terakhir untuk mencegah *double-counting*.

3. **Centering ALE:** Nilai ALE dipusatkan menggunakan rata-rata berbobot distribusi frekuensi empiris $P(X_j = v)$, sesuai spesifikasi teori Apley & Zhu (2020), sehingga $\mathbb{E}[ALE(X_j)] = 0$.

4. **Data untuk ALE:** ALE dihitung pada **data latih** (`X_train`, 410 sampel), konsisten dengan praktik standar XAI (*eXplainable Artificial Intelligence*).

5. **Tanpa Scaling Ridge:** Tidak ada normalisasi fitur (`StandardScaler`) di kode saat ini. Ini berdampak signifikan pada kinerja Ridge yang menghasilkan $R^2$ negatif.

---

*README ini dibuat berdasarkan analisis kode pada `ALE.py` dan verifikasi output program. Untuk dokumentasi teknis lengkap, lihat `BRIEFING_PENELITIAN_ALE.md`.*
