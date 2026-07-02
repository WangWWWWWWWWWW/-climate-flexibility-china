# -*- coding: utf-8 -*-
"""Nature Figure 3b: Spatial synchronization curve — Nature standard."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.fontsize': 6,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

DATA_DIR = DATA_DIR
OUT_DIR = FIGURES_DIR

COLORS = {'historical': '#555555', 'ssp245': '#4393c3', 'ssp370': '#d73027'}
LABELS = {'historical': 'Historical', 'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0'}
MARKERS = {'historical': 's', 'ssp245': 'o', 'ssp370': 'D'}

d = np.load(DATA_DIR / 'lslw_results.npz', allow_pickle=True)
sync_dist = {p: d[f'sync_dist_{p}'] for p in ['historical', 'ssp245', 'ssp370']}

k_range = np.arange(1, 16)

# Nature single column: 89mm
fig, ax = plt.subplots(figsize=(89/25.4, 70/25.4))

for period in ['historical', 'ssp245', 'ssp370']:
    vals = sync_dist[period][1:16].copy()
    vals_plot = np.where(vals > 0, vals, np.nan)
    ax.plot(k_range, vals_plot, marker=MARKERS[period], color=COLORS[period],
            lw=1.0, ms=3.5, label=LABELS[period],
            markeredgecolor='white', markeredgewidth=0.3,
            zorder=3 if period == 'ssp370' else 2)

ax.set_yscale('log')
ax.set_ylim(0.005, 200)
ax.set_xlim(0.5, 15.5)
ax.set_xticks(range(1, 16))

ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{x:g}' if x >= 0.01 else f'{x:.3f}'))
ax.set_yticks([0.01, 0.1, 1, 10, 100])
ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=20))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.set_xlabel('Number of synchronized provinces ($k$)')
ax.set_ylabel('Days/yr with $N \\geq k$')

ax.grid(True, which='major', alpha=0.12, linewidth=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=False, loc='upper right')

plt.tight_layout(pad=0.5)
out = OUT_DIR / 'Fig3b_spatial_synchronization.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f'Saved: {out.name}')
