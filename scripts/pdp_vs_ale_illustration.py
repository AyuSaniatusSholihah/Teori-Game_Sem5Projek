import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'mpl_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# ILUSTRASI KONSEPTUAL: PDP vs ALE
# ------------------------------------------------------------
# Menjelaskan perbedaan mendasar antara dua metode
# interpretasi model machine learning:
#   (a) PDP  - Partial Dependence Plot
#   (b) ALE  - Accumulated Local Effects
# ============================================================

# ── 1. BANGKITKAN DATA SIMULASI ──────────────────────────────
# Dibuat data berkorelasi untuk memperlihatkan masalah PDP
# ketika X1 dan X2 saling berkorelasi.
np.random.seed(42)
n  = 80                           # jumlah titik simulasi
X1 = np.random.uniform(1, 5, n)  # fitur yang dijelaskan
X2 = 0.7 * X1 + np.random.normal(0, 0.6, n)  # fitur berkorelasi dengan X1

# ── 2. NILAI PARAMETER UNTUK ILUSTRASI ──────────────────────
v    = 2.4   # nilai pengganti pada PDP (sumbu vertikal merah)
z_lo = 2.8   # batas bawah bin ALE  [z_{k-1}]
z_hi = 3.4   # batas atas bin ALE   [z_k]

# ── 3. BUAT KANVAS: 2 PANEL VERTIKAL ────────────────────────
fig, (ax_pdp, ax_ale) = plt.subplots(2, 1, figsize=(6.5, 9), dpi=300)
fig.subplots_adjust(hspace=0.42)

# ============================================================
# PANEL (a): PDP
# - Semua observasi dipaksa geser ke X1 = v
# - Kombinasi (v, X2) bisa tidak realistis karena X1 dan X2
#   berkorelasi namun diperlakukan seolah independen
# ============================================================
ax_pdp.set_title(r"(a) PDP: $X_1$ diganti $v$ untuk SEMUA observasi",
                 fontsize=11, fontweight='bold', pad=8, loc='left')

# Titik data asli (abu-abu)
ax_pdp.scatter(X1, X2, color='#b0b0b0', s=20, zorder=2, alpha=0.8)

# Garis vertikal merah putus-putus di X1 = v
ax_pdp.axvline(v, color='#cc0000', linewidth=1.4, linestyle='--', zorder=3)

# Semua titik dipindah ke kolom X1 = v (lingkaran kosong merah)
ax_pdp.scatter(np.full(n, v), X2,
               facecolors='white', edgecolors='#cc0000',
               s=22, linewidths=0.8, zorder=4)

# Kotak dashed menandai kolom titik tidak realistis
x_lo_box, x_hi_box = v - 0.12, v + 0.12
y_lo_box, y_hi_box = X2.min() - 0.1, X2.max() + 0.1
rect = mpatches.FancyBboxPatch(
    (x_lo_box, y_lo_box),
    x_hi_box - x_lo_box,
    y_hi_box - y_lo_box,
    boxstyle="square,pad=0.05",
    linewidth=1.2, edgecolor='#cc0000',
    linestyle='--', facecolor='none', zorder=5
)
ax_pdp.add_patch(rect)

# Anotasi "tidak realistis"
ax_pdp.annotate(
    "Titik ini tidak pernah\nada di data nyata\n(tidak realistis)",
    xy=(v + 0.05, X2.max() - 0.2),
    xytext=(v + 0.9, X2.max()),
    fontsize=8, color='#cc0000',
    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=0.9),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cc0000', alpha=0.9)
)

# Label X1 = v di bawah sumbu
ax_pdp.text(v, -0.05, r'$X_1 = v$',
            ha='center', va='top',
            transform=ax_pdp.get_xaxis_transform(),
            color='#cc0000', fontsize=8.5)

ax_pdp.set_xlabel(r"Fitur $X_1$ (yang dijelaskan)", fontsize=9, labelpad=4)
ax_pdp.set_ylabel(r"Fitur $X_2$ (berkorelasi)", fontsize=9, labelpad=4)
ax_pdp.tick_params(labelsize=8)
ax_pdp.set_xlim(0.5, 5.8)
ax_pdp.spines['top'].set_visible(False)
ax_pdp.spines['right'].set_visible(False)

# ============================================================
# PANEL (b): ALE
# - Hanya observasi DALAM interval [z_lo, z_hi] yang digeser
# - X2 dibiarkan sesuai nilai aslinya → tetap realistis
# ============================================================
ax_ale.set_title(
    r"(b) ALE: selisih prediksi lokal di dalam interval $[z_{k-1}, z_k]$",
    fontsize=11, fontweight='bold', pad=8, loc='left'
)

# Pisahkan titik dalam dan luar bin
in_bin  = (X1 >= z_lo) & (X1 <= z_hi)
X1_in, X2_in   = X1[in_bin],  X2[in_bin]
X1_out, X2_out = X1[~in_bin], X2[~in_bin]

# Titik luar bin (abu-abu)
ax_ale.scatter(X1_out, X2_out, color='#b0b0b0', s=20, zorder=2, alpha=0.8)

# Shading biru muda untuk area bin
ax_ale.axvspan(z_lo, z_hi, alpha=0.12, color='#2166ac', zorder=1)

# Garis batas bin
ax_ale.axvline(z_lo, color='#2166ac', linewidth=1.3, zorder=3)
ax_ale.axvline(z_hi, color='#2166ac', linewidth=1.3, zorder=3)

# Label z_{k-1} dan z_k di atas grafik
y_top = X2.max() + 0.6
ax_ale.text(z_lo, y_top, r'$z_{k-1}$', ha='center', va='bottom',
            color='#2166ac', fontsize=9, fontweight='bold')
ax_ale.text(z_hi, y_top, r'$z_k$', ha='center', va='bottom',
            color='#2166ac', fontsize=9, fontweight='bold')

# Titik dalam bin (biru solid)
ax_ale.scatter(X1_in, X2_in, color='#1565c0', s=30, zorder=5)

# Panah: tiap titik → batas bawah z_lo  (merah)
for x, y in zip(X1_in, X2_in):
    ax_ale.annotate("", xy=(z_lo, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->',
                                    color='#cc0000', lw=0.7, mutation_scale=6))

# Titik yang digeser ke z_lo (lingkaran kosong merah)
ax_ale.scatter(np.full(len(X1_in), z_lo), X2_in,
               facecolors='white', edgecolors='#cc0000',
               s=30, linewidths=0.9, zorder=6)

# Panah: tiap titik → batas atas z_hi  (hijau)
for x, y in zip(X1_in, X2_in):
    ax_ale.annotate("", xy=(z_hi, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->',
                                    color='#0a7340', lw=0.7, mutation_scale=6))

# Titik yang digeser ke z_hi (lingkaran kosong hijau)
ax_ale.scatter(np.full(len(X1_in), z_hi), X2_in,
               facecolors='white', edgecolors='#0a7340',
               s=30, linewidths=0.9, zorder=6)

# Anotasi penjelasan ALE
ax_ale.annotate(
    "Hanya observasi di dalam\ninterval digeser ke batas\nbawah & atas;\n"
    r"$X_2$ tetap → tetap dekat data nyata",
    xy=(z_hi + 0.05, X2_in.mean()),
    xytext=(z_hi + 0.7, X2_in.mean() - 0.4),
    fontsize=7.5, color='#1a1a1a',
    arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
    bbox=dict(boxstyle='round,pad=0.3', fc='#e8f4fd', ec='#2166ac', alpha=0.95)
)

ax_ale.set_xlabel(r"Fitur $X_1$ (yang dijelaskan)", fontsize=9, labelpad=4)
ax_ale.set_ylabel(r"Fitur $X_2$ (berkorelasi)", fontsize=9, labelpad=4)
ax_ale.tick_params(labelsize=8)
ax_ale.set_xlim(0.5, 5.8)
ax_ale.spines['top'].set_visible(False)
ax_ale.spines['right'].set_visible(False)

# ── 4. CAPTION & SIMPAN ──────────────────────────────────────
fig.text(0.98, 0.01,
         "Ilustrasi konseptual dengan data simulasi",
         ha='right', va='bottom', fontsize=7,
         color='#555555', style='italic')

# Simpan ke folder output/
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
out_dir      = os.path.join(project_root, 'output')
os.makedirs(out_dir, exist_ok=True)
out_path     = os.path.join(out_dir, 'pdp_vs_ale_illustration.png')
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Gambar berhasil disimpan: {out_path}")

