"""
ale_visualize.py
----------------
Modul visualisasi untuk penelitian ALE.

Semua fungsi di sini hanya membuat dan menyimpan grafik —
tidak ada komputasi model atau algoritma di sini.
Dapat diimpor dari script lain atau dikembangkan secara mandiri.
"""

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',
                      os.path.join(tempfile.gettempdir(), 'mpl_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# A. KURVA ALE 1 DIMENSI (Grid 2×2)
# =========================================================

def plot_ale_curves(ale_all: dict,
                    X_train: pd.DataFrame,
                    out_dir: str,
                    feature_titles: dict = None,
                    filename: str = 'ale_curves.png') -> str:
    """
    Membuat dan menyimpan grid 2×2 kurva ALE untuk
    empat fitur kunci.

    Setiap subplot berisi:
    - Garis kurva ALE dengan titik per skala
    - Garis acuan netral y = 0
    - Rug plot distribusi responden

    Parameter
    ---------
    ale_all       : dict {nama_fitur: dict_ale} dari compute_ale_all()
    X_train       : DataFrame data latih (untuk rug plot)
    out_dir       : str direktori penyimpanan file gambar
    feature_titles: dict {nama_fitur: judul_subplot}
    filename      : str nama file output (default 'ale_curves.png')

    Kembalikan
    ----------
    out_path : str path lengkap file yang disimpan
    """
    if feature_titles is None:
        feature_titles = {
            'social_media_sleep'  : 'Dampak Medsos pd Jam Tidur',
            'focus_quality'       : 'Kualitas Konsentrasi / Fokus',
            'compare_with_others' : 'Membandingkan Diri dg Orang Lain',
            'sad_content_feed'    : 'Paparan Feed Konten Sedih'
        }

    key_features = list(ale_all.keys())
    fig, axes    = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes         = axes.flatten()

    for idx, feat in enumerate(key_features):
        if idx >= len(axes):
            break
        ax       = axes[idx]
        ale_dict = ale_all[feat]
        x_vals   = list(ale_dict.keys())
        y_vals   = list(ale_dict.values())

        # Kurva ALE
        ax.plot(x_vals, y_vals, marker='o', color='#1f77b4',
                linewidth=2, markersize=6)
        # Garis acuan netral
        ax.axhline(0, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7)
        # Rug plot (distribusi sampel)
        rug = X_train[feat]
        ax.plot(rug, np.full_like(rug, min(y_vals) - 0.05),
                '|', color='black', alpha=0.3, markersize=8)

        title = feature_titles.get(feat, feat)
        ax.set_title(f"{feat}\n({title})",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Skala Likert (1 – 5)", fontsize=9)
        ax.set_ylabel("Efek ALE thdp Mood Disorder", fontsize=9)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Grafik kurva ALE disimpan         : {out_path}")
    return out_path


# =========================================================
# B. GINI FEATURE IMPORTANCE (Horizontal Bar Chart)
# =========================================================

def plot_feature_importance(fi_df: pd.DataFrame,
                             out_dir: str,
                             top_n: int = 10,
                             filename: str = 'feature_importance.png') -> str:
    """
    Membuat dan menyimpan bar chart horizontal Gini Importance.

    Parameter
    ---------
    fi_df    : DataFrame dari compute_gini_importance()
               kolom: 'Fitur', 'Gini Importance'
    out_dir  : str direktori penyimpanan
    top_n    : jumlah fitur teratas yang ditampilkan (default 10)
    filename : str nama file output

    Kembalikan
    ----------
    out_path : str path file yang disimpan
    """
    top = (fi_df.head(top_n)
                .sort_values(by='Gini Importance', ascending=True))

    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.barh(top['Fitur'],
                    top['Gini Importance'] * 100,
                    color='#2ca02c', edgecolor='black', alpha=0.85)
    plt.xlabel("Gini Importance (%)", fontsize=10, fontweight='bold')
    plt.title(f"Top {top_n} Fitur Terpenting (Random Forest)",
              fontsize=11, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.2,
                 bar.get_y() + bar.get_height() / 2,
                 f"{w:.2f}%",
                 va='center', ha='left', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Grafik Feature Importance disimpan: {out_path}")
    return out_path


# =========================================================
# C. PREDIKSI vs AKTUAL (Scatter Plot)
# =========================================================

def plot_pred_vs_actual(model, X_test: pd.DataFrame,
                        y_test: pd.Series,
                        out_dir: str,
                        model_name: str = 'Random Forest',
                        filename: str = 'pred_vs_actual.png') -> str:
    """
    Membuat scatter plot prediksi (ŷ) vs nilai aktual (y).

    Garis diagonal merah menunjukkan prediksi sempurna.
    Simpangan titik dari garis = error prediksi model.

    Parameter
    ---------
    model      : model sklearn terlatih
    X_test     : DataFrame fitur data uji
    y_test     : Series nilai aktual data uji
    out_dir    : str direktori penyimpanan
    model_name : str label nama model di judul grafik
    filename   : str nama file output

    Kembalikan
    ----------
    out_path : str path file yang disimpan
    """
    y_pred = model.predict(X_test)

    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y_test, y_pred,
                alpha=0.6, color='#1f77b4',
                edgecolors='white', linewidth=0.4,
                s=40, label='Observasi')
    plt.plot([1, 5], [1, 5], 'r--',
             linewidth=1.2, label='Prediksi Sempurna')
    plt.xlabel("Nilai Aktual $y$", fontsize=10)
    plt.ylabel(r"Nilai Prediksi $\hat{y}$", fontsize=10)
    plt.title(f"Prediksi vs Aktual ({model_name})",
              fontsize=11, fontweight='bold')
    plt.xlim(0.5, 5.5); plt.ylim(0.5, 5.5)
    plt.xticks([1, 2, 3, 4, 5])
    plt.yticks([1, 2, 3, 4, 5])
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Grafik Prediksi vs Aktual disimpan: {out_path}")
    return out_path


# =========================================================
# D. DISTRIBUSI RESIDUAL (Histogram)
# =========================================================

def plot_residuals(model, X_test: pd.DataFrame,
                   y_test: pd.Series,
                   out_dir: str,
                   model_name: str = 'Random Forest',
                   filename: str = 'residual_distribution.png') -> str:
    """
    Membuat histogram distribusi residual (y - ŷ).

    Distribusi mendekati normal dan terpusat di 0
    menunjukkan model tidak sistematis over/under-predict.

    Parameter
    ---------
    model      : model sklearn terlatih
    X_test     : DataFrame fitur data uji
    y_test     : Series nilai aktual data uji
    out_dir    : str direktori penyimpanan
    model_name : str label nama model di judul grafik
    filename   : str nama file output

    Kembalikan
    ----------
    out_path : str path file yang disimpan
    """
    y_pred    = model.predict(X_test)
    residuals = y_test.values - y_pred
    mean_res  = residuals.mean()

    plt.figure(figsize=(6, 4), dpi=300)
    plt.hist(residuals, bins=20,
             color='#ff7f0e', edgecolor='black', alpha=0.8)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.2,
                label='y = 0')
    plt.axvline(mean_res, color='blue', linestyle=':', linewidth=1.2,
                label=f'Mean residual = {mean_res:.3f}')
    plt.xlabel(r"Residual ($y - \hat{y}$)", fontsize=10)
    plt.ylabel("Frekuensi", fontsize=10)
    plt.title(f"Distribusi Residual ({model_name})",
              fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Grafik Distribusi Residual disimpan: {out_path}")
    return out_path
