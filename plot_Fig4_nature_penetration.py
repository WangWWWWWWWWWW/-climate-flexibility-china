# -*- coding: utf-8 -*-
"""
Nature Figure 4: Redesigned penetration by day type

Key changes from original:
  - Panel a: Proper violin + box + swarm-style, showing full distribution shape
  - Panel b: Paired bars with difference annotation arrows
  - Nature fonts, 183mm width, clean spines
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from path_config import (
    BASE_DIR, DATA_DIR, FIG_DIR, OUT_DIR, EXCEL_DIR, SOURCE_DATA_DIR,
    INPUT_TABLES_DIR, FIGURES_DIR, CODE_DIR, ERA5_HOURLY, ERA5_DAILY_RSDS,
    CMIP6_DIR, STEP1_DATA, ZHUO_DIR, HYDRO_PATH, TX_PATH, MOESM4_PATH,
    LOAD_FILE, GRID_COORDS, GRID_MAP, ensure_output_dirs,
)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 7,
    'axes.linewidth': 0.5,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

DATA_DIR = DATA_DIR
OUT_DIR = FIGURES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS_SSP = {'ssp245': '#4393c3', 'ssp370': '#d73027'}
SCENARIOS = ['NDC', 'GM2.0', 'CN2050']
SCENARIO_FILE = {'NDC': 'NDC', 'GM2.0': 'GM2.0', 'CN2050': 'CN2050'}

# ============================================================
# Load precomputed Fig4 data
# ============================================================
print('Loading Fig4 intermediate data...')
fig4_path = DATA_DIR / 'fig4_penetration_by_day_type.npz'
if not fig4_path.exists():
    print(f'ERROR: {fig4_path} not found. Run plot_Fig4_penetration_by_day_type.py first.')
    raise SystemExit(1)

fig4 = np.load(fig4_path)

# ============================================================
# Figure
# ============================================================
print('Plotting Nature Figure 4...')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183/25.4, 75/25.4),
                                gridspec_kw={'width_ratios': [1.3, 1]})

# --- Panel a: Violin + box for CN2050 ---
ax1.text(-0.14, 1.05, 'a', transform=ax1.transAxes, fontsize=9, fontweight='bold', va='top')

positions_groups = {'Normal': [0.8, 1.2], 'LSLW': [2.0, 2.4]}

for ssp_idx, ssp in enumerate(['ssp245', 'ssp370']):
    key_prefix = f'CN2050_{ssp}'
    pen_nat = fig4[f'pen_nat_{key_prefix}']
    pen_lslw = fig4[f'pen_lslw_{key_prefix}']
    dtype = fig4[f'dtype_{key_prefix}']

    data_normal = pen_nat[dtype == 0]
    data_sync = pen_lslw[dtype == 2]

    color = COLORS_SSP[ssp]

    for group_idx, (group_name, data_arr) in enumerate([('Normal', data_normal), ('LSLW', data_sync)]):
        pos = positions_groups[group_name][ssp_idx]

        # Violin
        parts = ax1.violinplot([data_arr], positions=[pos], widths=0.32,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor('none')

        # Box (IQR)
        q1, med, q3 = np.percentile(data_arr, [25, 50, 75])
        mean_val = data_arr.mean()
        box_w = 0.12
        ax1.bar(pos, q3 - q1, bottom=q1, width=box_w,
               color=color, alpha=0.7, edgecolor='none', zorder=3)
        ax1.plot([pos - box_w/2, pos + box_w/2], [med, med],
                color='white', lw=1.0, zorder=4)
        ax1.scatter([pos], [mean_val], color='white', s=10, zorder=5,
                   edgecolors='none', marker='o')

ax1.set_xticks([1.0, 2.2])
ax1.set_xticklabels(['Normal days', 'Synch. LSLW\n($\\geq$3 provinces)'])
ax1.set_ylabel('Renewable penetration (%)')
ax1.set_xlim(0.3, 2.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS_SSP['ssp245'], alpha=0.7, label='SSP2-4.5'),
                   Patch(facecolor=COLORS_SSP['ssp370'], alpha=0.7, label='SSP3-7.0')]
ax1.legend(handles=legend_elements, loc='upper right', frameon=False)

# --- Panel b: Paired bars with difference annotation ---
ax2.text(-0.14, 1.05, 'b', transform=ax2.transAxes, fontsize=9, fontweight='bold', va='top')

GCM_LIST = ['ACCESS-CM2', 'EC-Earth3', 'GFDL-ESM4', 'MRI-ESM2-0', 'NorESM2-MM']
FUTURE_YEARS = 25
LOW_PEN_THRESHOLD = 50

bar_data = {}
for sc in SCENARIOS:
    for ssp in ['ssp245', 'ssp370']:
        key_prefix = f'{SCENARIO_FILE[sc]}_{ssp}'
        pen = fig4[f'pen_nat_{key_prefix}']
        n_low = (pen < LOW_PEN_THRESHOLD).sum()
        freq_yr = n_low / (len(GCM_LIST) * FUTURE_YEARS)
        bar_data[(sc, ssp)] = freq_yr

x = np.arange(len(SCENARIOS))
bar_w = 0.3

bars_245 = [bar_data[(sc, 'ssp245')] for sc in SCENARIOS]
bars_370 = [bar_data[(sc, 'ssp370')] for sc in SCENARIOS]

b1 = ax2.bar(x - bar_w/2, bars_245, bar_w, color=COLORS_SSP['ssp245'],
             alpha=0.8, edgecolor='none', label='SSP2-4.5')
b2 = ax2.bar(x + bar_w/2, bars_370, bar_w, color=COLORS_SSP['ssp370'],
             alpha=0.8, edgecolor='none', label='SSP3-7.0')

# Difference annotations
for i in range(len(SCENARIOS)):
    diff = bars_370[i] - bars_245[i]
    mid_y = max(bars_245[i], bars_370[i]) + 8
    ax2.annotate(f'+{diff:.0f}',
                xy=(x[i], mid_y), ha='center', va='bottom',
                fontsize=5.5, color='#d73027', fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(SCENARIOS)
ax2.set_ylabel('Days/yr with penetration < 50%')
ax2.legend(loc='upper right', frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=0.8)
out_path = OUT_DIR / 'Fig4_penetration_by_day_type.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f'Saved: {out_path.name}')
print('Done.')
