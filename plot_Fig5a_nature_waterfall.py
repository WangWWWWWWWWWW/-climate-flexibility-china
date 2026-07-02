# -*- coding: utf-8 -*-
"""Nature Figure 5a: Dispatch waterfall decomposition — Nature standard."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    'legend.fontsize': 6,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

DATA_DIR = DATA_DIR
OUT_DIR = FIGURES_DIR

CAP_SCENARIOS = ['NDC', 'GM2.0', 'CN2050']
SSP_LIST = ['ssp245', 'ssp370']
SCENARIO_LABELS = {'NDC': 'NDC', 'GM2.0': 'GM2.0', 'CN2050': 'CN2050'}
SSP_LABELS = {'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0'}

LAYER_COLORS = {
    'Gas': '#f4a582',
    'Coal': '#666666',
    'TX': '#66c2a5',
    'Unmet': '#b2182b'
}

# Load
d = np.load(DATA_DIR / 'provincial_postTX_results.npz', allow_pickle=True)
nat_mean = {}
for cap in CAP_SCENARIOS:
    for ssp in SSP_LIST:
        pf = f'{cap}_{ssp}'
        nat_mean[(cap, ssp)] = {
            'deficit': float(d[f'{pf}_deficit_autarky_TWh']),
            'gas': float(d[f'{pf}_gas_TWh']),
            'coal': float(d[f'{pf}_coal_TWh']),
            'tx': float(d[f'{pf}_ctx_value_TWh']),
            'unmet': float(d[f'{pf}_national_unmet_TWh'])
        }

def draw_waterfall(ax, nm, title_str):
    deficit = nm['deficit']
    layers = [('Gas', nm['gas']), ('Coal', nm['coal']),
              ('TX', nm['tx']), ('Unmet', nm['unmet'])]

    # Deficit bar
    ax.bar(0, deficit, 0.6, color='#c6a4d8', alpha=0.5,
           edgecolor='#8e6aaf', linewidth=0.8, linestyle='--')
    ax.text(0, deficit/2, f'{deficit:,.0f}', ha='center', va='center',
            fontsize=5.5, fontweight='bold', color='#333')

    top = deficit
    for li, (name, val) in enumerate(layers):
        xi = li + 1
        bottom = top - val
        if val < 1:
            top = bottom
            continue
        ax.bar(xi, val, 0.6, bottom=bottom, color=LAYER_COLORS[name],
               alpha=0.9, edgecolor='white', linewidth=0.2)
        mid = bottom + val/2
        fc = 'white' if val > 500 else '#333'
        ax.text(xi, mid, f'{val:,.0f}', ha='center', va='center',
                fontsize=5, fontweight='bold', color=fc)
        # Connector
        ax.plot([xi-0.6, xi-0.3], [top, top], color='#aaa', lw=0.4, ls='--')
        top = bottom

    ax.set_xticks(range(5))
    ax.set_xticklabels(['Deficit', 'Gas', 'Coal', 'TX', 'Unmet'],
                       fontsize=5.5, rotation=30, ha='right')
    ax.set_title(title_str, fontsize=6.5, fontweight='bold', pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)

# Nature double-col: 183mm
fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 100/25.4), sharey=True)
fig.subplots_adjust(wspace=0.08, hspace=0.55)

for ri, ssp in enumerate(SSP_LIST):
    for ci, cap in enumerate(CAP_SCENARIOS):
        ax = axes[ri, ci]
        nm = nat_mean[(cap, ssp)]
        title = f'{SCENARIO_LABELS[cap]} ({SSP_LABELS[ssp]})'
        draw_waterfall(ax, nm, title)
        if ci == 0:
            ax.set_ylabel('TWh/yr')

handles = [Patch(facecolor=LAYER_COLORS[n], alpha=0.9, label=n)
           for n in ['Gas', 'Coal', 'TX', 'Unmet']]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.0))

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = OUT_DIR / 'Fig5a_dispatch_waterfall.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out.name}')
