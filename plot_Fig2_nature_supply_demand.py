# -*- coding: utf-8 -*-
"""
Nature Figure 2: 2x3 stacked area (top) + 2x3 penetration line (bottom)
  Top row (a): Monthly RE generation vs demand — 3 pathways × 2 SSPs
  Bottom row (b): Monthly penetration rate (%) — same 6 panels
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 5.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 0.8,
})

DATA_DIR = DATA_DIR
OUT_DIR = FIGURES_DIR

MONTH_LABELS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

COL_HYDRO = '#2166ac'
COL_WIND = '#4393c3'
COL_SOLAR = '#fdae61'

SCENARIOS = [
    ('NDC', 'ssp245', 'NDC\nSSP2-4.5'),
    ('NDC', 'ssp370', 'NDC\nSSP3-7.0'),
    ('GM2.0', 'ssp245', 'GM2.0\nSSP2-4.5'),
    ('GM2.0', 'ssp370', 'GM2.0\nSSP3-7.0'),
    ('CN2050', 'ssp245', 'CN2050\nSSP2-4.5'),
    ('CN2050', 'ssp370', 'CN2050\nSSP3-7.0'),
]

# Load data
print('Loading data...')
data_all = {}
for sc_file, ssp, label in SCENARIOS:
    fpath = DATA_DIR / f'supply_demand_2050_{sc_file}_{ssp}.npz'
    if not fpath.exists():
        print(f'  Warning: {fpath.name} not found')
        continue
    d = np.load(fpath, allow_pickle=True)
    wind = d['gen_wind'].sum(axis=1)    # (12,) national
    solar = d['gen_solar'].sum(axis=1)
    hydro = d['gen_hydro'].sum(axis=1)
    demand = d['monthly_demand'].sum(axis=1)
    re_total = wind + solar + hydro
    penetration = re_total / demand * 100
    data_all[(sc_file, ssp)] = {
        'wind': wind, 'solar': solar, 'hydro': hydro,
        'demand': demand, 'penetration': penetration,
    }
    print(f'  Loaded: {sc_file}_{ssp}, annual pen={re_total.sum()/demand.sum()*100:.1f}%')

# Plot: 4 rows x 3 cols (top 2 rows = stacked area, bottom 2 rows = penetration)
# Actually: 2 rows x 3 cols for area, then 1 row x 3 cols for penetration below
# Better: use gridspec with 2 major rows

fig = plt.figure(figsize=(183/25.4, 150/25.4))
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

months = np.arange(1, 13)

# --- Top row (a): Stacked area, SSP2-4.5 ---
# --- Second row: Stacked area, SSP3-7.0 ---
# Actually let's do 4 rows: row0-1 = area (SSP245, SSP370), row2-3 = pen (SSP245, SSP370)
# Simpler: 2 rows x 3 cols. Row 0 = stacked area for each pathway (both SSPs overlaid as demand lines)
# Best approach for 6 panels: 2 rows x 3 cols

# Let me do: top 2x3 = stacked area, bottom separate 1x3 = penetration comparison
gs_top = GridSpec(2, 3, figure=fig, top=0.91, bottom=0.42, hspace=0.35, wspace=0.28)
gs_bot = GridSpec(1, 3, figure=fig, top=0.35, bottom=0.06, wspace=0.28)

pathways = ['NDC', 'GM2.0', 'CN2050']
pathway_labels = ['NDC', 'GM2.0', 'CN2050']
ssps = ['ssp245', 'ssp370']
ssp_labels = ['SSP2-4.5', 'SSP3-7.0']

# Panel label
fig.text(0.02, 0.96, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.02, 0.37, 'b', fontsize=10, fontweight='bold', va='top')

# --- Top shared legend ---
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles_top = [
    Patch(color=COL_HYDRO, alpha=0.85, label='Hydro'),
    Patch(color=COL_WIND, alpha=0.85, label='Wind'),
    Patch(color=COL_SOLAR, alpha=0.85, label='Solar'),
    Line2D([0], [0], color='k', lw=1.2, label='Demand'),
]
fig.legend(handles=handles_top, loc='upper center', ncol=4, frameon=False,
           fontsize=6, bbox_to_anchor=(0.5, 0.995), columnspacing=1.5, handletextpad=0.5)

# --- (a) Stacked area: 2 rows (SSP) x 3 cols (pathway) ---
for ri, ssp in enumerate(ssps):
    for ci, pathway in enumerate(pathways):
        ax = fig.add_subplot(gs_top[ri, ci])
        key = (pathway, ssp)
        if key not in data_all:
            continue
        d = data_all[key]

        ax.fill_between(months, 0, d['hydro'],
                        color=COL_HYDRO, alpha=0.85, linewidth=0)
        ax.fill_between(months, d['hydro'], d['hydro'] + d['wind'],
                        color=COL_WIND, alpha=0.85, linewidth=0)
        ax.fill_between(months, d['hydro'] + d['wind'],
                        d['hydro'] + d['wind'] + d['solar'],
                        color=COL_SOLAR, alpha=0.85, linewidth=0)
        ax.plot(months, d['demand'], color='k', lw=1.2, zorder=5)

        ax.set_xlim(1, 12)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_ylim(0)

        if ci == 0:
            ax.set_ylabel('TWh/month')
        ax.set_title(f'{pathway_labels[ci]} ({ssp_labels[ri]})', fontsize=6, pad=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# --- (b) Penetration rate: 1 row x 3 cols, both SSPs on same panel ---
for ci, pathway in enumerate(pathways):
    ax = fig.add_subplot(gs_bot[0, ci])

    for ssp, color, ls, marker, label in [
        ('ssp245', '#4393c3', '-', 'o', 'SSP2-4.5'),
        ('ssp370', '#d73027', '--', 's', 'SSP3-7.0'),
    ]:
        key = (pathway, ssp)
        if key not in data_all:
            continue
        d = data_all[key]
        ax.plot(months, d['penetration'], ls, color=color, lw=1.3,
                marker=marker, ms=4, markeredgecolor='white', markeredgewidth=0.4,
                label=label)

    # 100% reference line
    ax.axhline(100, color='grey', ls=':', lw=0.5, alpha=0.5)

    ax.set_xlim(1, 12)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)

    # Y-axis: don't start from 0, zoom in on the range to show difference
    all_pen = []
    for ssp in ssps:
        key = (pathway, ssp)
        if key in data_all:
            all_pen.extend(data_all[key]['penetration'].tolist())
    ymin = max(0, min(all_pen) - 10)
    ymax = max(all_pen) + 10
    ax.set_ylim(ymin, ymax)

    if ci == 0:
        ax.set_ylabel('RE penetration (%)')
    ax.set_title(pathway_labels[ci], fontsize=6.5, fontweight='bold', pad=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ci == 0:
        ax.legend(frameon=False, fontsize=5.5, loc='lower left')

out = OUT_DIR / 'Fig2_supply_demand_balance.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved: {out}')
