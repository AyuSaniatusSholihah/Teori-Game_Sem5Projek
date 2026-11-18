"""
main.py
-------
Titik masuk utama (entry point) penelitian:
Analisis Pengaruh Media Sosial terhadap Kesehatan Mental
Menggunakan Accumulated Local Effects (ALE).

Struktur Modular:
  data/    → dataset (cleaned_data-2.csv)
  src/     → modul algoritma (ale_core.py) & visualisasi (ale_visualize.py)
  output/  → file gambar & grafik publikasi IEEE
  docs/    → dokumen briefing & metodologi penelitian
  scripts/ → script ilustrasi konseptual PDP vs ALE
"""

import os
import sys

# Tambahkan folder root ke sys.path agar impor modul src/ selalu aman
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ale_core import (
    load_dataset,
    descriptive_stats,
    prepare_features,
    split_data,
    build_models,
    train_and_evaluate,
    get_best_model,
    compute_gini_importance,
    compute_ale_all,
)
from src.ale_visualize import (
    plot_ale_curves,
    plot_feature_importance,
    plot_pred_vs_actual,
    plot_residuals,
)

# ── Konfigurasi Path dan Parameter ────────────────────────
DATA_PATH    = os.path.join(PROJECT_ROOT, 'data', 'cleaned_data-2.csv')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'output')
TARGET_COL   = 'mood_disorder'
KEY_FEATURES = ['social_media_sleep', 'focus_quality',
                'compare_with_others', 'sad_content_feed']

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 1. Pemuatan Data
    df = load_dataset(DATA_PATH)

    # 2. Statistik Deskriptif Variabel Kunci
    desc_df = descriptive_stats(df)

    # 3. Penyiapan Fitur X dan Target y
    X, y = prepare_features(df, target_col=TARGET_COL)

    # 4. Pembagian Data Latih dan Uji (80:20, seed 42)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Pelatihan dan Evaluasi Tiga Model (Ridge, GB, RF)
    models = build_models()
    eval_df = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # 6. Pemilihan Model Terbaik Otomatis (RMSE Terkecil)
    best_name, best_model = get_best_model(models, eval_df)

    # 7. Analisis Kepentingan Fitur Global (Gini Importance)
    fi_df = compute_gini_importance(best_model, X)

    # 8. Analisis Efek Lokal Non-Linear (Kurva ALE 1D)
    ale_all = compute_ale_all(best_model, X_train, KEY_FEATURES)

    # 9. Penyimpanan Grafik Format Publikasi IEEE (300 DPI)
    print("--- Menyimpan Grafik ke Folder output/ ---")
    plot_ale_curves(ale_all, X_train, OUTPUT_DIR)
    plot_feature_importance(fi_df, OUTPUT_DIR)
    plot_pred_vs_actual(best_model, X_test, y_test, OUTPUT_DIR, best_name)
    plot_residuals(best_model, X_test, y_test, OUTPUT_DIR, best_name)

    print(f"\nSelesai! Seluruh grafik visualisasi tersimpan di: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
