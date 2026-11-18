"""
ALE.py  —  Script Utama (Runner)
---------------------------------
Mengorkestrasi seluruh pipeline penelitian dengan mengimpor
fungsi dari dua modul terpisah:

  ale_core.py      → komputasi: data, model, algoritma ALE
  ale_visualize.py → visualisasi: semua grafik & plot

Untuk mengubah ALGORITMA → edit ale_core.py
Untuk mengubah TAMPILAN   → edit ale_visualize.py
Untuk mengubah ALUR       → edit file ini (ALE.py)
"""

import os

# ── Import modul inti (komputasi) ──────────────────────────
from ale_core import (
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

# ── Import modul visualisasi ───────────────────────────────
from ale_visualize import (
    plot_ale_curves,
    plot_feature_importance,
    plot_pred_vs_actual,
    plot_residuals,
)

# ──────────────────────────────────────────────────────────
# KONFIGURASI (ubah di sini jika perlu)
# ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(SCRIPT_DIR, 'cleaned_data-2.csv')
TARGET_COL   = 'mood_disorder'
KEY_FEATURES = ['social_media_sleep', 'focus_quality',
                'compare_with_others', 'sad_content_feed']

# ──────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ──────────────────────────────────────────────────────────

# 1. Muat data
df = load_dataset(CSV_PATH)

# 2. Statistik deskriptif variabel kunci
desc_df = descriptive_stats(df)

# 3. Siapkan fitur dan target
X, y = prepare_features(df, target_col=TARGET_COL)

# 4. Bagi data latih / uji
X_train, X_test, y_train, y_test = split_data(X, y)

# 5. Latih dan evaluasi tiga model
models  = build_models()
eval_df = train_and_evaluate(models, X_train, X_test, y_train, y_test)

# 6. Pilih model terbaik
best_name, best_model = get_best_model(models, eval_df)

# 7. Hitung Gini Feature Importance
fi_df = compute_gini_importance(best_model, X)

# 8. Hitung ALE untuk fitur kunci
ale_all = compute_ale_all(best_model, X_train, KEY_FEATURES)

# ──────────────────────────────────────────────────────────
# VISUALISASI
# ──────────────────────────────────────────────────────────
print("--- Menyimpan Grafik ---")

plot_ale_curves(ale_all, X_train, SCRIPT_DIR)
plot_feature_importance(fi_df, SCRIPT_DIR)
plot_pred_vs_actual(best_model, X_test, y_test, SCRIPT_DIR, best_name)
plot_residuals(best_model, X_test, y_test, SCRIPT_DIR, best_name)

print("\nSelesai. Semua output berhasil dibuat.")
