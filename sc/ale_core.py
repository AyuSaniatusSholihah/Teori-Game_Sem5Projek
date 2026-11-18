"""
ale_core.py
-----------
Modul inti penelitian ALE: pemuatan data, pelatihan model,
evaluasi, Gini importance, dan algoritma ALE 1 dimensi.

Semua fungsi di sini murni komputasi — tidak ada visualisasi.
Dapat diimpor dari script lain atau diuji secara mandiri.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =========================================================
# A. PEMUATAN DAN EKSPLORASI DATA
# =========================================================

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Memuat dataset CSV dan menampilkan informasi dasar.

    Parameter
    ---------
    csv_path : str — path lengkap ke file CSV

    Kembalikan
    ----------
    df : DataFrame berisi seluruh data
    """
    df = pd.read_csv(csv_path)
    print("=" * 55)
    print("  ANALISIS PENGARUH MEDIA SOSIAL TERHADAP KESEHATAN")
    print("  MENTAL MENGGUNAKAN ACCUMULATED LOCAL EFFECTS (ALE)")
    print("=" * 55)
    print(f"\nInformasi Dataset:")
    print(f"  - Total Responden (Baris) : {df.shape[0]}")
    print(f"  - Total Kolom             : {df.shape[1]}")
    print(f"  - Nilai Hilang            : {df.isnull().sum().sum()}")
    print(f"  - Baris Duplikat          : {df.duplicated().sum()}")
    return df


def descriptive_stats(df: pd.DataFrame,
                      key_vars: list = None) -> pd.DataFrame:
    """
    Menghitung statistik deskriptif (Min, Max, Mean, SD)
    untuk variabel-variabel kunci.

    Parameter
    ---------
    df       : DataFrame sumber data
    key_vars : list nama kolom; default = target + 4 indikator medsos

    Kembalikan
    ----------
    desc_df : DataFrame berisi statistik deskriptif
    """
    if key_vars is None:
        key_vars = ['mood_disorder', 'social_media_sleep',
                    'focus_quality', 'compare_with_others',
                    'sad_content_feed']

    rows = []
    for col in key_vars:
        rows.append({
            'Variabel' : col,
            'Min'      : int(df[col].min()),
            'Max'      : int(df[col].max()),
            'Mean'     : round(df[col].mean(), 2),
            'SD'       : round(df[col].std(), 2)
        })
    desc_df = pd.DataFrame(rows)
    print("\n--- Statistik Deskriptif Variabel Kunci ---")
    print(desc_df.to_string(index=False))
    print()
    return desc_df


def prepare_features(df: pd.DataFrame,
                     target_col: str = 'mood_disorder'):
    """
    Memisahkan fitur (X) dan target (y) dari DataFrame.

    Catatan: Seluruh 31 kolom independen diikutsertakan —
    termasuk anxious_stressed & nervous_anxious — karena
    merupakan bagian sah dari data swalaporan responden.

    Parameter
    ---------
    df         : DataFrame sumber
    target_col : nama kolom target (default 'mood_disorder')

    Kembalikan
    ----------
    X : DataFrame fitur (31 kolom)
    y : Series target
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Membagi data menjadi set latih dan uji.

    Catatan metodologis:
    - Tidak menggunakan stratifikasi (target numerik kontinu).
    - Tidak ada cross-validation; pembagian tunggal dipilih
      karena ukuran dataset terbatas (513 sampel) dan
      kebutuhan konsistensi pelaporan antar model.

    Parameter
    ---------
    test_size    : proporsi data uji (default 0.2 = 20%)
    random_state : seed acak (default 42)

    Kembalikan
    ----------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Jumlah Data Latih : {X_train.shape[0]} sampel "
          f"({int((1 - test_size) * 100)}%)")
    print(f"Jumlah Data Uji   : {X_test.shape[0]} sampel "
          f"({int(test_size * 100)}%)")
    print(f"Random State      : {random_state}\n")
    return X_train, X_test, y_train, y_test


# =========================================================
# B. PELATIHAN DAN EVALUASI MODEL
# =========================================================

def build_models() -> dict:
    """
    Mendefinisikan tiga model regresi beserta hiperparameternya.

    Hiperparameter eksplisit:
    - Ridge            : alpha=1.0
    - Gradient Boosting: n_estimators=100, learning_rate=0.05
    - Random Forest    : n_estimators=300

    Catatan Ridge: Tidak ada StandardScaler di pipeline ini.
    Karena skala fitur sangat heterogen (mis. social_media_purpose
    0–86 vs fitur biner 0–1), Ridge dapat menghasilkan R² negatif.
    Ini merupakan keterbatasan yang perlu disebutkan di makalah.

    Kembalikan
    ----------
    dict berisi {nama_model: objek_model}
    """
    return {
        'Ridge Regression' : Ridge(alpha=1.0),
        'Gradient Boosting': GradientBoostingRegressor(
                                n_estimators=100,
                                learning_rate=0.05,
                                random_state=42),
        'Random Forest'    : RandomForestRegressor(
                                n_estimators=300,
                                random_state=42)
    }


def train_and_evaluate(models: dict,
                       X_train, X_test,
                       y_train, y_test) -> pd.DataFrame:
    """
    Melatih seluruh model pada data latih dan mengevaluasi
    kinerjanya (RMSE & R²) pada data uji.

    Parameter
    ---------
    models  : dict dari build_models()
    X_train, X_test, y_train, y_test : hasil split_data()

    Kembalikan
    ----------
    eval_df : DataFrame hasil evaluasi, diurutkan RMSE terkecil
    """
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        results.append({'Model': name, 'RMSE': rmse, 'R²': r2})

    eval_df = pd.DataFrame(results).sort_values(by='RMSE')
    print("--- Kinerja Model pada Data Uji ---")
    print(eval_df.to_string(index=False))
    print()
    return eval_df


def get_best_model(models: dict, eval_df: pd.DataFrame):
    """
    Mengambil objek model terbaik berdasarkan RMSE terkecil.

    Kembalikan
    ----------
    best_name  : str nama model terbaik
    best_model : objek sklearn model terlatih
    """
    best_name  = eval_df.iloc[0]['Model']
    best_model = models[best_name]
    print(f"Model terbaik: {best_name} "
          f"(RMSE={eval_df.iloc[0]['RMSE']:.4f}, "
          f"R²={eval_df.iloc[0]['R²']:.4f})\n")
    return best_name, best_model


# =========================================================
# C. GINI FEATURE IMPORTANCE
# =========================================================

def compute_gini_importance(model, X: pd.DataFrame,
                             top_n: int = 5) -> pd.DataFrame:
    """
    Menghitung Gini Importance (Mean Decrease in Impurity)
    dari model Random Forest untuk seluruh fitur.

    Parameter
    ---------
    model : RandomForestRegressor terlatih
    X     : DataFrame fitur (untuk nama kolom)
    top_n : jumlah fitur teratas yang dicetak (default 5)

    Kembalikan
    ----------
    fi_df : DataFrame berisi seluruh fitur dan nilai importance,
            diurutkan dari yang terbesar
    """
    fi_df = pd.DataFrame({
        'Fitur'          : X.columns,
        'Gini Importance': model.feature_importances_
    }).sort_values(by='Gini Importance', ascending=False).reset_index(drop=True)

    print(f"--- Top {top_n} Fitur Berdasarkan Gini Importance ---")
    print(fi_df.head(top_n).to_string(index=False))
    print()
    return fi_df


# =========================================================
# D. ALGORITMA ALE 1 DIMENSI
# =========================================================

def compute_ale_1d(model, X_data: pd.DataFrame,
                   feature_name: str,
                   grid_values: list = None) -> dict:
    """
    Menghitung nilai kurva ALE (Accumulated Local Effects)
    satu dimensi untuk fitur berskala Likert (1–5).

    Implementasi mandiri tanpa library ALE eksternal
    (PyALE / alibi / DALEX), dirancang khusus untuk
    fitur ordinal diskret.

    Pseudocode (Apley & Zhu 2020):
    --------------------------------
     1: ALE_raw[1] <- 0
     2: untuk m = 1 sampai 4:
     3:   S_m <- {i: z_m <= x_ij < z_(m+1)}
              (bin terakhir: inklusif z_5)
     4:   jika |S_m| = 0: delta_m <- 0
     5:   selain itu:
     6:     X_high <- X[S_m] dg x_j = z_(m+1)
             X_low  <- X[S_m] dg x_j = z_m
     7:     delta_m <- mean(f(X_high) - f(X_low))
     8:   ALE_raw[m+1] <- ALE_raw[m] + delta_m
     9: P(v) <- frekuensi relatif x_j = v di X_data
    10: g_bar <- sum_v ALE_raw[v] * P(v)
    11: ALE(v) <- ALE_raw[v] - g_bar
    12: kembalikan ALE
    --------------------------------

    Parameter
    ---------
    model        : model sklearn terlatih (memiliki .predict)
    X_data       : DataFrame fitur — gunakan data LATIH
    feature_name : str nama kolom fitur yang dianalisis
    grid_values  : list int batas bin (default [1, 2, 3, 4, 5])

    Kembalikan
    ----------
    dict {nilai_skala (int): efek_ALE_terpusat (float)}
    """
    if grid_values is None:
        grid_values = [1, 2, 3, 4, 5]
    grid_values = list(grid_values)

    # -- Validasi input --
    if len(grid_values) < 2 or grid_values != sorted(set(grid_values)):
        raise ValueError(
            "grid_values harus berisi minimal dua nilai unik yang terurut"
        )
    feature_values = X_data[feature_name].dropna()
    if not feature_values.between(grid_values[0], grid_values[-1]).all():
        raise ValueError(
            f"Nilai {feature_name} berada di luar rentang grid_values"
        )

    X_work       = X_data.copy()
    grid_effects = {grid_values[0]: 0.0}
    acc          = 0.0

    # -- Langkah 2–8: Hitung efek lokal per bin dan akumulasi --
    for k in range(1, len(grid_values)):
        z_prev = grid_values[k - 1]
        z_curr = grid_values[k]

        # Bin setengah terbuka [z_prev, z_curr)
        # Bin terakhir inklusif kanan [z_prev, z_curr]
        # → mencegah observasi di batas terhitung ganda
        if k == len(grid_values) - 1:
            mask = ((X_work[feature_name] >= z_prev) &
                    (X_work[feature_name] <= z_curr))
        else:
            mask = ((X_work[feature_name] >= z_prev) &
                    (X_work[feature_name] <  z_curr))
        sub_df = X_work[mask]

        if len(sub_df) > 0:
            X_high = sub_df.copy(); X_high[feature_name] = z_curr
            X_low  = sub_df.copy(); X_low[feature_name]  = z_prev
            mean_diff = np.mean(
                model.predict(X_high) - model.predict(X_low)
            )
        else:
            mean_diff = 0.0

        acc += mean_diff
        grid_effects[z_curr] = acc

    # -- Langkah 9–11: Centering berbobot frekuensi empiris --
    # E[ALE(X_j)] = 0 setelah centering
    value_counts = (X_work[feature_name]
                    .value_counts(normalize=True)
                    .to_dict())
    mean_g = sum(grid_effects.get(val, 0) * prob
                 for val, prob in value_counts.items())

    return {val: grid_effects[val] - mean_g for val in grid_values}


def compute_ale_all(model, X_train: pd.DataFrame,
                    key_features: list) -> dict:
    """
    Menghitung dan mencetak ALE untuk daftar fitur kunci,
    lengkap dengan narasi efek tiap skala.

    Parameter
    ---------
    model        : model terlatih
    X_train      : data latih
    key_features : list nama kolom yang dianalisis

    Kembalikan
    ----------
    ale_all : dict {nama_fitur: dict_ale}
    """
    ale_all = {}
    print("--- Nilai Kurva ALE (Skala Likert 1–5) ---")
    for feat in key_features:
        if feat not in X_train.columns:
            print(f"[PERINGATAN] Kolom '{feat}' tidak ditemukan, dilewati.")
            continue
        ale_res  = compute_ale_1d(model, X_train, feat)
        ale_all[feat] = ale_res
        print(f"\nFitur: {feat}")
        for skala, efek in ale_res.items():
            arah = "meningkatkan" if efek > 0 else "menurunkan"
            print(f"  Skala {skala}: {efek:+.4f}  "
                  f"→ {arah} prediksi sebesar {abs(efek):.4f} unit")
        print("-" * 55)
    print()
    return ale_all
