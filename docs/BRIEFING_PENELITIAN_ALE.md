# Briefing Teknis Algoritma dan Verifikasi Kode Penelitian

**Judul Makalah (Format IEEE):**  
*Analisis Pengaruh Media Sosial terhadap Kesehatan Mental Menggunakan Accumulated Local Effects*  
**Dataset:** `cleaned_data-2.csv` (513 responden)  
**Target:** `mood_disorder` (Skala Likert 1–5)  

---

## 1. Gambaran Umum

### Struktur Proyek dan Fungsi File
Proyek tersimpan dalam subdirektori `sc/sc/` dengan 2 berkas utama:
1. `ALE.py`: Script Python utama yang menjalankan seluruh *pipeline* penelitian: pemuatan data, pembagian *train-test*, pelatihan 3 model regresi (`Ridge`, `GradientBoostingRegressor`, `RandomForestRegressor`), evaluasi metrik (RMSE dan $R^2$), ekstraksi *Gini feature importance*, serta implementasi algoritma 1D *Accumulated Local Effects* (ALE) kustom untuk fitur berskala ordinal Likert.
2. `cleaned_data-2.csv`: Dataset swalaporan berformat tabular yang telah dibersihkan sebelumnya, terdiri dari 513 sampel responden dan 32 kolom numerik.

### Pustaka (*Libraries*) dan Versi
Berdasarkan verifikasi lingkungan (*environment* Python 3.14.5):
- **pandas:** `3.0.3` (`ALE.py`, Baris 1)
- **numpy:** `2.4.6` (`ALE.py`, Baris 2)
- **scikit-learn:** `1.9.1` (`ALE.py`, Baris 3–6)
- **Library ALE Pihak Ketiga (PyALE / Alibi / DALEX):** **Tidak ada / Tidak digunakan**. Algoritma ALE ditulis secara mandiri (*pure native code*) berbasis fungsi matematika lokal di `ALE.py` Baris 75–125.

### Diagram Alur Eksekusi
```text
[cleaned_data-2.csv] (513 baris x 32 kolom)
         │
         ▼
[Pemisahan Variabel] ───> Target y: 'mood_disorder'
         │           ───> Matriks Fitur X (31 fitur lainnya)
         ▼
[train_test_split] (80% Latih: 410, 20% Uji: 103, random_state=42)
         │
         ▼
[Pelatihan 3 Model Regresi pada Data Latih]
  ├─ Ridge Regression (alpha=1.0)
  ├─ Gradient Boosting Regressor (n_estimators=100, lr=0.05)
  └─ Random Forest Regressor (n_estimators=300)
         │
         ▼
[Evaluasi Model pada Data Uji (RMSE & R²)] ───> Seleksi Model Terbaik: Random Forest
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
[Gini Feature Importance (RF)]            [Komputasi ALE 1D pada X_train]
- Diurutkan dari 31 fitur                 - Interval bin diskret: [1, 2, 3, 4, 5]
- Menampilkan Top 5 Fitur Teratas          - Perturbasi lokal & akumulasi
                                           - Centering rata-rata berbobot marginal
                                           - Cetak nilai ALE untuk 4 fitur kunci
```

---

## 2. Pemuatan dan Pembersihan Data

- **Pemuatan Data (`ALE.py`, Baris 12–14):** Data dimuat menggunakan `pd.read_csv(csv_path)`.
- **Kolom Target dan Fitur (`ALE.py`, Baris 21–24):**
  - Target ($y$): `mood_disorder` (skala Likert 1–5).
  - Fitur ($X$): Seluruh kolom kecuali target dibuang dengan `df.drop(columns=['mood_disorder'])`. Artinya, **31 kolom lainnya digunakan secara penuh sebagai fitur prediktor**.
- **Penanganan Nilai Hilang, Duplikat, Outlier, dan Encoding:**
  - *Nilai Hilang:* **Tidak ada penanganan di kode**. Dataset `cleaned_data-2.csv` sudah bersih (*null values* = 0).
  - *Duplikat:* **Tidak ada penanganan di kode** (baris duplikat = 0).
  - *Outlier:* **Tidak ada penanganan di kode** (seluruh 513 baris dipertahankan).
  - *Encoding:* **Tidak ada proses encoding di kode**, karena seluruh nilai variabel pada file CSV telah dikodekan sebelumnya ke format numerik (`int64`).
- **Jumlah Baris:** Sebelum dan sesudah tahap pemuatan di kode adalah **513 baris** (tidak ada penyaringan baris).
- **Daftar Lengkap 31 Fitur Independen ($X$) dan Target ($y$):**

| No | Nama Kolom | Peran | Tipe Data | Rentang Nilai | Keterangan Skala |
|:---|:---|:---|:---|:---|:---|
| - | `mood_disorder` | **Target ($y$)** | `int64` | 1 – 5 | Likert (1=Sangat Rendah, 5=Sangat Tinggi) |
| 1 | `profession` | Fitur | `int64` | 0 – 12 | Kategori pekerjaan (integer encoded) |
| 2 | `social_media_sleep` | Fitur | `int64` | 1 – 5 | Likert (Dampak medsos thdp jam tidur) |
| 3 | `content_type_social_media` | Fitur | `int64` | 0 – 58 | Jenis konten medsos (integer encoded) |
| 4 | `think_time_instagram` | Fitur | `int64` | 0 – 4 | Estimasi durasi Instagram |
| 5 | `think_time_whatsapp` | Fitur | `int64` | 0 – 4 | Estimasi durasi WhatsApp |
| 6 | `think_time_twitter` | Fitur | `int64` | 0 – 4 | Estimasi durasi Twitter / X |
| 7 | `think_time_facebook` | Fitur | `int64` | 0 – 4 | Estimasi durasi Facebook |
| 8 | `think_time_youtube` | Fitur | `int64` | 0 – 4 | Estimasi durasi YouTube |
| 9 | `think_time_linkedin` | Fitur | `int64` | 0 – 4 | Estimasi durasi LinkedIn |
| 10 | `actual_time_instagram` | Fitur | `int64` | 0 – 4 | Durasi riil Instagram |
| 11 | `actual_time_whatsapp` | Fitur | `int64` | 0 – 4 | Durasi riil WhatsApp |
| 12 | `actual_time_twitter` | Fitur | `int64` | 0 – 4 | Durasi riil Twitter / X |
| 13 | `actual_time_facebook` | Fitur | `int64` | 0 – 4 | Durasi riil Facebook |
| 14 | `actual_time_youtube` | Fitur | `int64` | 0 – 4 | Durasi riil YouTube |
| 15 | `actual_time_linkedin` | Fitur | `int64` | 0 – 4 | Durasi riil LinkedIn |
| 16 | `focus_quality` | Fitur | `int64` | 1 – 5 | Likert (Kualitas konsentrasi/fokus) |
| 17 | `binge_watch_reels` | Fitur | `int64` | 1 – 5 | Likert (Frekuensi menonton video pendek) |
| 18 | `sad_content_feed` | Fitur | `int64` | 1 – 5 | Likert (Terpapar konten sedih/negatif) |
| 19 | `social_media_purpose` | Fitur | `int64` | 0 – 86 | Tujuan penggunaan medsos (encoded) |
| 20 | `social_relationship_impact` | Fitur | `int64` | 1 – 5 | Likert (Dampak pada hubungan sosial) |
| 21 | `time_loss_social_media` | Fitur | `int64` | 0 – 4 | Skala waktu terbuang di medsos |
| 22 | `sleep_quality_general` | Fitur | `int64` | 1 – 5 | Likert (Kualitas tidur umum) |
| 23 | `refreshed_after_social_media` | Fitur | `int64` | 0 – 1 | Biner (0 = Tidak, 1 = Ya) |
| 24 | `anxious_stressed` | Fitur | `int64` | 0 – 1 | Biner (0 = Tidak, 1 = Merasa cemas/stres) |
| 25 | `compare_with_others` | Fitur | `int64` | 1 – 5 | Likert (Membandingkan diri dg orang lain) |
| 26 | `headaches_eyestrain` | Fitur | `int64` | 0 – 1 | Biner (Keluhan fisik pusing/mata lelah) |
| 27 | `trouble_sleeping_month` | Fitur | `int64` | 1 – 5 | Likert (Frekuensi susah tidur bulanan) |
| 28 | `sleep_quality_month` | Fitur | `int64` | 1 – 5 | Likert (Penilaian tidur sebulan terakhir) |
| 29 | `social_media_before_bed` | Fitur | `int64` | 0 – 1 | Biner (Main medsos sebelum tidur) |
| 30 | `nervous_anxious` | Fitur | `int64` | 1 – 5 | Likert (Tingkat kegugupan/kecemasan) |
| 31 | `smartphone_discomfort` | Fitur | `int64` | 1 – 5 | Likert (Rasa tidak nyaman tanpa ponsel) |

---

## 3. Praproses dan Pembagian Data

- **Pembagian Data (`ALE.py`, Baris 29–31):**
  - Fungsi: `train_test_split(X, y, test_size=0.2, random_state=42)`
  - Proporsi: 80% Latih (410 sampel) dan 20% Uji (103 sampel).
  - Nilai Seed: `random_state = 42`.
  - Stratifikasi: **Tidak memakai stratifikasi** (`stratify=None`), karena target merupakan skor kontinu/numerik.
- **Scaling / Normalisasi Fitur:** **TIDAK ADA**.
  - Tidak ada pemanggilan `StandardScaler`, `MinMaxScaler`, maupun normalisasi lainnya di seluruh file.
- **Risiko Data Leakage:**
  - **Rendah/Tidak Terjadi Data Leakage pada tahap split**, karena data uji dipisahkan secara murni sebelum pelatihan model, dan tidak ada transformasi global yang dipelajari bersama sebelum pemisahan.
  - Namun, ketiadaan normalisasi menjadi kelemahan signifikan bagi algoritma linier dengan penalti reguler seperti Ridge (dibahas pada Bagian 10).

---

## 4. Pelatihan Model

Seluruh model diinisialisasi pada `ALE.py` Baris 39–43 dan dilatih langsung dengan `.fit(X_train, y_train)` pada Baris 48.

### Tabel Konfigurasi Hiperparameter Model

| Nama Model | Kelas Scikit-Learn | Hiperparameter Eksplisit di Kode | Hiperparameter Default yang Signifikan (Tidak di-set) | Penjelasan Mekanisme Kerja Sesuai Konfigurasi |
|:---|:---|:---|:---|:---|
| **Ridge Regression** | `sklearn.linear_model.Ridge` | `alpha=1.0` | `fit_intercept=True`, `copy_X=True`, `max_iter=None`, `tol=0.0001`, `solver='auto'` | Regresi kuadrat terkecil biasa (*OLS*) yang ditambahkan penalti regularisasi L2 ($\lambda \sum w_j^2$) dengan kekuatan $\alpha = 1.0$ untuk meredam multikolinearitas dan overfitting pada koefisien bobot. |
| **Gradient Boosting** | `sklearn.ensemble.GradientBoostingRegressor` | `n_estimators=100`, `learning_rate=0.05`, `random_state=42` | `loss='squared_error'`, `max_depth=3`, `subsample=1.0`, `criterion='friedman_mse'`, `min_samples_split=2`, `min_samples_leaf=1` | Membangun ansambel 100 *decision tree* dangkal (*depth*=3) secara sekuensial. Setiap pohon baru dilatih untuk memprediksi sisa kesalahan (gradien negatif dari *loss squared error*) dari prediksi ansambel sebelumnya dengan faktor penyusutan bobot (*shrinkage*) sebesar 0.05. |
| **Random Forest** | `sklearn.ensemble.RandomForestRegressor` | `n_estimators=300`, `random_state=42` | `criterion='squared_error'`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features=1.0`, `bootstrap=True` | Menggabungkan 300 pohon regresi independen menggunakan *bootstrap aggregating* (*bagging*). Karena `max_depth=None`, pohon ditumbuhkan penuh hingga tiap daun murni, lalu prediksinya dirata-ratakan untuk mengurangi variansi model secara signifikan. |

- **Hyperparameter Tuning / Cross-Validation / Pipeline:**
  - **Tidak ada** *Hyperparameter Tuning* (seperti `GridSearchCV` atau `RandomizedSearchCV`).
  - **Tidak ada** *Cross-Validation* (K-Fold).
  - **Tidak ada** `sklearn.pipeline.Pipeline`.

---

## 5. Evaluasi Model

- **Metrik dan Perhitungan (`ALE.py`, Baris 48–52):**
  - Dihitung khusus pada **data uji** (`X_test`, `y_test`).
  - **RMSE:** $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$ dihitung menggunakan `np.sqrt(mean_squared_error(y_test, y_pred))`.
  - **$R^2$ (Koefisien Determinasi):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ dihitung menggunakan `r2_score(y_test, y_pred)`.
- **Tabel Kinerja Model pada Data Uji (Persis Sesuai Output Kode):**

| Peringkat | Model | RMSE | $R^2$ |
|:---:|:---|:---:|:---:|
| 1 | **Random Forest** | **1.301160** | **0.048664** |
| 2 | Gradient Boosting | 1.326098 | 0.011848 |
| 3 | Ridge Regression | 1.391330 | -0.087759 |

- **Cara Pemilihan Model Terbaik (`ALE.py`, Baris 54, 60–62):**
  Model diurutkan berdasarkan nilai RMSE terkecil (`sort_values(by='RMSE')`). Karena **Random Forest** menghasilkan RMSE terendah (1.301160) dan $R^2$ positif tertinggi (0.048664), objek model Random Forest (`models['Random Forest']`) dipilih secara otomatis untuk analisis *Feature Importance* dan kurva ALE.

---

## 6. Feature Importance

- **Metode Importance (`ALE.py`, Baris 62–66):**
  - Menggunakan *Mean Decrease in Impurity* (MDI) atau **Gini Importance** bawaan Random Forest (`rf_model.feature_importances_`).
  - Dihitung pada **data latih** saat konstruksi pohon (bukan permutation importance pada data uji).
  - Di kode dicetak sebagai nilai desimal fraksional (rentang 0–1, total jumlah seluruh fitur = 1.0).
- **Daftar Lengkap Seluruh 31 Fitur dan Nilai Importance (Diurutkan):**

| Peringkat | Nama Fitur | Gini Importance (Fraksi) | Gini Importance (%) |
|:---:|:---|:---:|:---:|
| 1 | `anxious_stressed` | 0.164369 | 16.44% |
| 2 | `social_media_sleep` | 0.145188 | 14.52% |
| 3 | `focus_quality` | 0.090135 | 9.01% |
| 4 | `nervous_anxious` | 0.052721 | 5.27% |
| 5 | `content_type_social_media` | 0.048087 | 4.81% |
| 6 | `social_media_purpose` | 0.045108 | 4.51% |
| 7 | `sad_content_feed` | 0.044152 | 4.42% |
| 8 | `binge_watch_reels` | 0.030428 | 3.04% |
| 9 | `smartphone_discomfort` | 0.030361 | 3.04% |
| 10 | `compare_with_others` | 0.027109 | 2.71% |
| 11 | `social_relationship_impact` | 0.024373 | 2.44% |
| 12 | `trouble_sleeping_month` | 0.023161 | 2.32% |
| 13 | `think_time_youtube` | 0.021191 | 2.12% |
| 14 | `refreshed_after_social_media` | 0.020275 | 2.03% |
| 15 | `sleep_quality_general` | 0.020049 | 2.00% |
| 16 | `think_time_whatsapp` | 0.019806 | 1.98% |
| 17 | `sleep_quality_month` | 0.019147 | 1.91% |
| 18 | `actual_time_instagram` | 0.018808 | 1.88% |
| 19 | `actual_time_youtube` | 0.018362 | 1.84% |
| 20 | `actual_time_whatsapp` | 0.017999 | 1.80% |
| 21 | `think_time_instagram` | 0.015282 | 1.53% |
| 22 | `actual_time_twitter` | 0.013420 | 1.34% |
| 23 | `actual_time_linkedin` | 0.013152 | 1.32% |
| 24 | `think_time_twitter` | 0.012380 | 1.24% |
| 25 | `think_time_linkedin` | 0.012338 | 1.23% |
| 26 | `actual_time_facebook` | 0.011891 | 1.19% |
| 27 | `time_loss_social_media` | 0.011758 | 1.18% |
| 28 | `social_media_before_bed` | 0.009483 | 0.95% |
| 29 | `headaches_eyestrain` | 0.009320 | 0.93% |
| 30 | `think_time_facebook` | 0.006244 | 0.62% |
| 31 | `profession` | 0.003904 | 0.39% |

*(Catatan: Di konsol script utama `ALE.py` Baris 68–69, kode hanya mencetak 5 fitur teratas).*

---

## 7. Implementasi ALE (Accumulated Local Effects) 1 Dimensi

Algoritma ALE diimplementasikan secara kustom pada fungsi `compute_ale_1d(model, X_data, feature_name, grid_values=None)` (`ALE.py`, Baris 75–125).

### Pustaka dan Parameter Fungsi
- Tidak menggunakan pustaka eksternal.
- Parameter:
  - `model`: Model regresi terlatih (`rf_model`).
  - `X_data`: Dataset fitur (`X_train`).
  - `feature_name`: Nama string kolom yang diuji.
  - `grid_values`: *List* nilai batas interval (default: `[1, 2, 3, 4, 5]`).

### Pembagian Interval / Bin
- **Metode Pembagian:** Menggunakan nilai diskret Likert integer 1 sampai 5. Bukan pembagian kuantil persentil maupun *equal-width* pecahan kontinu.
- **Jumlah Bin:** Terdapat 4 interval transisi:
  - Bin 1: $[1, 2)$
  - Bin 2: $[2, 3)$
  - Bin 3: $[3, 4)$
  - Bin 4: $[4, 5]$ (interval terakhir bersifat inklusif kanan agar skala 5 masuk hitungan).
- **Perlakuan Diskret (`ALE.py`, Baris 100–103):** Kode menerapkan filter interval setengah terbuka $[z_{k-1}, z_k)$ kecuali pada bin terakhir untuk mencegah pengamatan pada batas skala terhitung ganda (*double-counting*).

### Tahapan Komputasi Matematis
1. **Penyaringan Sampel Lokal ($S_k$):** Untuk setiap interval $[z_{k-1}, z_k]$, ambil seluruh sampel observasi empiris yang nilai fiturnya berada dalam interval tersebut.
2. **Perturbasi Lokal (Penggantian Batas):**
   - Buat salinan data $X_{high}$ di mana kolom fitur diset bernilai $z_k$.
   - Buat salinan data $X_{low}$ di mana kolom fitur diset bernilai $z_{k-1}$.
3. **Rata-rata Selisih Lokal (*Mean Local Difference*):**
   $$\Delta_k = \frac{1}{|S_k|} \sum_{i \in S_k} \left[ \hat{f}(X_{high}^{(i)}) - \hat{f}(X_{low}^{(i)}) \right]$$
   Jika interval tidak memiliki observasi ($|S_k| = 0$), maka $\Delta_k = 0.0$.
4. **Akumulasi Kumulatif (*Uncentered Effect*):**
   $$ALE_{\text{raw}}(z_1) = 0.0, \quad ALE_{\text{raw}}(z_k) = \sum_{j=1}^k \Delta_j$$
5. **Pemusalatan (*Centering*) dengan Pembobotan Frekuensi Empiris (`ALE.py`, Baris 121–124):**
   - Centering **menggunakan rata-rata berbobot frekuensi observasi marginal**, sesuai rumus teoretis Apley & Zhu (2020):
     $$\bar{g} = \sum_{v \in \{1,2,3,4,5\}} ALE_{\text{raw}}(v) \cdot P(X_j = v)$$
     di mana $P(X_j = v)$ dihitung dari `X_work[feature_name].value_counts(normalize=True)`.
   - Efek akhir yang terpusat nol adalah:
     $$ALE(v) = ALE_{\text{raw}}(v) - \bar{g}$$
     Sehingga nilai ekspektasi efek marginal terhadap populasi data tepat bernilai nol ($\mathbb{E}[ALE(X_j)] = 0$).

### Data yang Digunakan
- Menggunakan **data latih** `X_train` (`ALE.py`, Baris 133).

### Fitur yang Dihitung dan Nilai Efek ALE
Dihitung untuk 4 fitur utama (`ALE.py`, Baris 128):

| Fitur | Skala 1 | Skala 2 | Skala 3 | Skala 4 | Skala 5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `social_media_sleep` | -0.3807 | -0.3591 | -0.0971 | +0.1717 | +0.4679 |
| `focus_quality` | -0.1483 | -0.1118 | -0.0591 | +0.0291 | +0.2398 |
| `compare_with_others` | -0.0500 | -0.0075 | -0.0029 | +0.0250 | +0.0933 |
| `sad_content_feed` | -0.1840 | -0.0155 | +0.0122 | +0.0470 | +0.0601 |

- **Interval Kepercayaan / Bootstrap:** **TIDAK ADA**. Tidak terdapat perhitungan *bootstrap resampling* maupun standar deviasi selang kepercayaan di kode.

---

## 8. Visualisasi dan Output

- **Grafik / Plot Visual:** **TERSEDIA (Baru Ditambahkan)**. Script telah mengintegrasikan modul visualisasi berbasis `matplotlib` dengan standar resolusi tinggi (300 DPI) yang siap dimasukkan ke naskah publikasi IEEE:
  1. `ale_curves.png`: Grid 2x2 kurva garis ALE untuk 4 fitur kunci (`social_media_sleep`, `focus_quality`, `compare_with_others`, `sad_content_feed`) lengkap dengan garis acuan netral $y=0$ dan *rug plot* distribusi responden.
  2. `feature_importance.png`: Diagram batang horizontal (*bar chart*) 10 fitur terpenting berdasarkan *Gini Importance* Random Forest dengan label persentase (%).
- **Penyimpanan Berkas (Export Output):** Berkas gambar disimpan secara otomatis ke direktori yang sama dengan script (`sc/sc/ale_curves.png` dan `sc/sc/feature_importance.png`).
- **Output Program:** Menampilkan rangkuman dataset, kinerja model, 5 fitur teratas Gini, tabel nilai ALE, serta konfirmasi penyimpanan file visualisasi.

---

## 9. Verifikasi terhadap Draf Makalah

Berikut adalah hasil pencocokan antara angka di draf naskah makalah Anda dengan output aktual dari kode:

| Butir yang Diverifikasi | Nilai di Makalah | Status Verifikasi | Nilai Aktual Kode / Dataset | Keterangan & Referensi Baris |
|:---|:---|:---:|:---|:---|
| **Jumlah responden dan missing value** | 513 responden, tanpa nilai hilang | **SESUAI** | 513 baris, 0 *missing value* | `ALE.py` Baris 15; file CSV memiliki 514 baris (1 header + 513 data). |
| **Pembagian data latih dan uji** | 80:20 (410 latih, 103 uji), seed 42 | **SESUAI** | 410 sampel latih, 103 sampel uji, `random_state=42` | `ALE.py` Baris 27–34. |
| **Hiperparameter model** | Ridge $\alpha=1.0$; GB 100 est, lr=0.05; RF 300 pohon | **SESUAI** | Ridge `alpha=1.0`; GB `n_estimators=100, learning_rate=0.05`; RF `n_estimators=300` | `ALE.py` Baris 39–43. |
| **Hasil uji model: Random Forest** | RMSE 1.2326 / $R^2$ 0.1462 | **TIDAK SESUAI** | RMSE: **1.301160**<br>$R^2$: **0.048664** | Output konsol Baris 54–56. |
| **Hasil uji model: Ridge Regression** | RMSE 1.2789 / $R^2$ 0.0809 | **TIDAK SESUAI** | RMSE: **1.391330**<br>$R^2$: **-0.087759** | Output konsol Baris 54–56. |
| **Hasil uji model: Gradient Boosting** | RMSE 1.3015 / $R^2$ 0.0482 | **TIDAK SESUAI** | RMSE: **1.326098**<br>$R^2$: **0.011848** | Output konsol Baris 54–56. |
| **Gini Importance:** `social_media_sleep` | 18.28% | **TIDAK SESUAI** | **14.52%** (0.145188) | Menempati peringkat ke-2 di kode, bukan peringkat ke-1. |
| **Gini Importance:** `focus_quality` | 15.03% | **TIDAK SESUAI** | **9.01%** (0.090135) | Menempati peringkat ke-3 di kode, bukan peringkat ke-2. |
| **ALE:** `social_media_sleep` (1 s.d. 5) | -0.5079, -0.4654, -0.1026, +0.2244, +0.6074 | **TIDAK SESUAI** | -0.3807, -0.3591, -0.0971, +0.1717, +0.4679 | Output konsol Baris 130–137. |
| **ALE:** `focus_quality` (1 s.d. 5) | -0.3019, -0.2282, -0.1101, +0.0599, +0.4144 | **TIDAK SESUAI** | -0.1483, -0.1118, -0.0591, +0.0291, +0.2398 | Output konsol Baris 130–137. |
| **ALE:** `compare_with_others` (1 s.d. 5) | -0.1338, -0.0424, -0.0013, +0.0315, +0.1094 | **TIDAK SESUAI** | -0.0500, -0.0075, -0.0029, +0.0250, +0.0933 | Output konsol Baris 130–137. |
| **ALE:** `sad_content_feed` (1 s.d. 5) | -0.1558, -0.0481, -0.0040, +0.0058, +0.0416 | **TIDAK SESUAI** | -0.1840, -0.0155, +0.0122, +0.0470, +0.0601 | Output konsol Baris 130–137. |
| **Statistik deskriptif (Mean, SD):** `mood_disorder` | 3.06 (1.35) | **TIDAK DITEMUKAN DI KODE** *(SESUAI di dataset)* | Mean = 3.06, SD = 1.35 | Tidak ada kode kalkulasi mean/SD di `ALE.py`, namun saat dihitung dari `cleaned_data-2.csv`, angkanya **persis sama**. |
| **Statistik deskriptif (Mean, SD):** `social_media_sleep` | 3.16 (1.29) | **TIDAK DITEMUKAN DI KODE** *(SESUAI di dataset)* | Mean = 3.16, SD = 1.29 | Sama seperti di atas. |
| **Statistik deskriptif (Mean, SD):** `focus_quality` | 3.15 (1.38) | **TIDAK DITEMUKAN DI KODE** *(SESUAI di dataset)* | Mean = 3.15, SD = 1.38 | Sama seperti di atas. |
| **Statistik deskriptif (Mean, SD):** `compare_with_others` | 2.76 (1.30) | **TIDAK DITEMUKAN DI KODE** *(SESUAI di dataset)* | Mean = 2.76, SD = 1.30 | Sama seperti di atas. |
| **Statistik deskriptif (Mean, SD):** `sad_content_feed` | 3.41 (1.45) | **TIDAK DITEMUKAN DI KODE** *(SESUAI di dataset)* | Mean = 3.41, SD = 1.45 | Sama seperti di atas. |

---

## 10. Temuan dan Catatan Kritis

### Akar Masalah Ketidaksesuaian Angka Makalah vs Kode
Mengapa metrik uji, peringkat Gini, dan nilai ALE di naskah makalah berbeda dari kode `ALE.py`?
1. **Penyertaan Variabel Gejala Mental (`anxious_stressed` dan `nervous_anxious`):**
   - Pada `ALE.py` Baris 23, fitur dibentuk dengan `X = df.drop(columns=['mood_disorder'])`, yang menyertakan **seluruh 31 kolom**.
   - Di dalam kode ini, fitur dengan Gini Importance nomor 1 justru adalah `anxious_stressed` (16.44%). Keberadaan fitur cemas/stres ini menyerap porsi *importance* yang seharusnya dimiliki oleh fitur penggunaan media sosial.
   - **Hasil Verifikasi Komputasi:** Ketika fitur gejala kecemasan psikologis (`anxious_stressed` dan `nervous_anxious`) dikeluarkan dari matriks prediktor:
     - Kinerja Random Forest langsung berubah menjadi: $RMSE \approx 1.2366$ dan $R^2 \approx 0.1407$ (sangat mendekati angka naskah Anda: $RMSE = 1.2326$ dan $R^2 = 0.1462$).
     - *Gini Importance* `social_media_sleep` melonjak menjadi **18.29%** (persis di angka makalah 18.28%), dan `focus_quality` naik menjadi **14.84% – 15.03%** (persis di angka makalah 15.03%).
   - **Kesimpulan:** Angka yang Anda tulis pada draf makalah berasal dari eksperimen terdahulu di mana fitur-fitur gejala psikologis non-media sosial telah dieleminasi/disaring, sedangkan file `ALE.py` yang ada saat ini masih menyertakan seluruh 31 kolom secara mentah.

### Catatan Metodologis Penting
1. **Performa Ridge Regression Negatif ($R^2 = -0.0878$):**
   Ridge menghasilkan prediksi yang lebih buruk daripada rerata dasar karena **tidak adanya scaling fitur**. Skala fitur di dataset sangat heterogen (misalnya `social_media_purpose` bernilai 0–86, sedangkan fitur biner bernilai 0–1). Penalti L2 pada Ridge menekan bobot fitur tanpa menimbang skala aslinya. Jika Ridge ingin dilaporkan secara valid di makalah IEEE, fitur wajib distandarisasi terlebih dahulu menggunakan `StandardScaler`.
2. **Asumsi Tersembunyi pada Fungsi ALE:**
   Fungsi `compute_ale_1d` mengasumsikan grid selalu berjarak bulat integer $[1, 2, 3, 4, 5]$ (`grid_values=[1, 2, 3, 4, 5]`). Implementasi ini bekerja sempurna untuk data skala Likert 1–5, tetapi akan gagal jika dipanggil pada fitur lain seperti `social_media_purpose` (rentang 0–86) atau durasi waktu (rentang 0–4).
3. **Data untuk ALE:**
   ALE dihitung pada **data latih** (`X_train`), bukan data uji. Hal ini sah dan umum dalam literatur XAI (*eXplainable Artificial Intelligence*) untuk mengeksplorasi hubungan yang telah dipelajari model dari populasi data latih, namun harus disebutkan secara eksplisit di subbab Metode.

---

## Ringkasan Alur untuk Subbab "Metode" Makalah

> *"Pipeline penelitian ini diawali dengan ekstraksi 513 data responden dari kuesioner swalaporan terstandardisasi tanpa nilai hilang (missing values), di mana skor gangguan suasana hati (`mood_disorder`, skala Likert 1–5) diposisikan sebagai variabel target terhadap 31 variabel independen penggunaan media sosial dan pola tidur. Dataset dibagi menjadi data latih (80%, 410 sampel) dan data uji (20%, 103 sampel) secara acak terkontrol (random seed = 42). Tiga arsitektur model regresi dilatih pada data latih: Ridge Regression ($\alpha = 1.0$), Gradient Boosting Regressor (100 estimator, laju pembelajaran 0.05), dan Random Forest Regressor (300 pohon keputusan). Kinerja model dievaluasi pada data uji menggunakan metrik Root Mean Squared Error (RMSE) dan koefisien determinasi ($R^2$), di mana Random Forest terpilih sebagai model berkinerja terbaik dengan RMSE 1.3012 dan $R^2$ 0.0487. Pola keputusan model Random Forest diinterpretasikan secara global menggunakan Mean Decrease in Impurity (Gini Importance), kemudian dianalisis pengaruh lokal non-linearnya melalui kurva Accumulated Local Effects (ALE) satu dimensi. Algoritma ALE dikomputasikan secara diskret pada data latih dengan membagi fitur ke dalam interval terurut skala Likert [1, 2, 3, 4, 5], mengukur selisih prediksi lokal antar batas interval, mengakumulasi perubahan marjinal, serta memusatkan efek terhadap rata-rata berbobot distribusi frekuensi empiris responden."*

