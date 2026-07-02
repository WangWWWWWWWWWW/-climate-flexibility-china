# -*- coding: utf-8 -*-
"""
Nature Fig 6 combined: (a) Energy balance stacked bar + (b) Four-quadrant scatter
Output: Fig6_typology_combined.png (does NOT overwrite old Fig6_provincial_typology.png)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
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
    'legend.fontsize': 5.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

OUT_DIR = FIGURES_DIR

df = pd.read_excel(SOURCE_DATA_DIR / 'table_Fig6_provincial_typology.xlsx')

prov_short = df['Province_short'].values
N_PROV = len(df)

wind    = df['Wind_TWh'].values.astype(float)
solar   = df['Solar_TWh'].values.astype(float)
hydro   = df['Hydro_TWh'].values.astype(float)
nuclear = df['Nuclear_TWh'].values.astype(float)
demand  = df['Demand_TWh'].values.astype(float)
gas     = df['Gas_TWh'].values.astype(float)
coal    = df['Coal_TWh'].values.astype(float)
tx_recv = df['TX_Received_TWh'].values.astype(float)
tx_sent = df['TX_Sent_TWh'].values.astype(float)
unmet   = df['Unmet_TWh'].values.astype(float)

u_cn370 = unmet
delta_u = df['High_Emissions_Sensitivity_TWh'].values.astype(float)

net_tx = df['Net_TX_TWh'].values.astype(float)
total_gen = wind + solar + hydro + nuclear + gas + coal
is_exporter = df['Is_Net_Exporter'].values.astype(bool)

# Energy balance decomposition
wind_pct = np.zeros(N_PROV)
solar_pct = np.zeros(N_PROV)
hydro_nuc_pct = np.zeros(N_PROV)
gas_pct = np.zeros(N_PROV)
import_pct = np.zeros(N_PROV)
unmet_pct = np.zeros(N_PROV)

for i in range(N_PROV):
    if demand[i] <= 0:
        continue
    unmet_pct[i] = unmet[i] / demand[i] * 100
    if is_exporter[i]:
        local_use = demand[i] - unmet[i]
        gen_sum = total_gen[i]
        scale = local_use / gen_sum if gen_sum > 0 else 0
        wind_pct[i] = wind[i] * scale / demand[i] * 100
        solar_pct[i] = solar[i] * scale / demand[i] * 100
        hydro_nuc_pct[i] = (hydro[i] + nuclear[i]) * scale / demand[i] * 100
        gas_pct[i] = gas[i] * scale / demand[i] * 100
    else:
        wind_pct[i] = wind[i] / demand[i] * 100
        solar_pct[i] = solar[i] / demand[i] * 100
        hydro_nuc_pct[i] = (hydro[i] + nuclear[i]) / demand[i] * 100
        gas_pct[i] = gas[i] / demand[i] * 100
        import_pct[i] = max(net_tx[i], 0) / demand[i] * 100

sort_idx = np.argsort(unmet)[::-1]

# Four-quadrant classification
u_p75 = np.percentile(u_cn370, 75)
delta_p75 = np.percentile(delta_u, 75)
median_u = np.median(u_cn370)
is_support = (u_cn370 < median_u) & (net_tx < -0.1)

cat = np.zeros(N_PROV, dtype=int)
for i in range(N_PROV):
    if u_cn370[i] >= u_p75 and delta_u[i] >= delta_p75:
        cat[i] = 1
    elif u_cn370[i] >= u_p75 and delta_u[i] < delta_p75:
        cat[i] = 2
    elif u_cn370[i] < u_p75 and delta_u[i] >= delta_p75:
        cat[i] = 3
    else:
        cat[i] = 4

CAT_COLORS = {1: '#d73027', 2: '#fc8d59', 3: '#4393c3', 4: '#999999'}

CLR_WIND = '#4E79A7'
CLR_SOLAR = '#E6A23C'
CLR_HYDRO = '#76B7B2'
CLR_GAS = '#B07AA1'
CLR_IMPORT = '#59A14F'
CLR_UNMET = '#E15759'

# === Figure ===
fig = plt.figure(figsize=(183/25.4, 155/25.4))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1], wspace=0.35)

# --- Panel a ---
ax = fig.add_subplot(gs[0, 0])
ax.text(-0.15, 1.02, 'a', fontsize=10, fontweight='bold', transform=ax.transAxes, va='top')

y_pos = np.arange(N_PROV)
bar_h = 0.6
components = [
    (wind_pct, CLR_WIND, 'Wind'),
    (solar_pct, CLR_SOLAR, 'Solar'),
    (hydro_nuc_pct, CLR_HYDRO, 'Hydro+Nuclear'),
    (gas_pct, CLR_GAS, 'Gas'),
    (import_pct, CLR_IMPORT, 'Net import'),
    (unmet_pct, CLR_UNMET, 'Residual'),
]
left = np.zeros(N_PROV)
for vals, color, label in components:
    sorted_vals = vals[sort_idx]
    if sorted_vals.max() < 0.05:
        continue
    ax.barh(y_pos, sorted_vals, left=left, height=bar_h,
            color=color, alpha=0.85, label=label, edgecolor='white', linewidth=0.2)
    left += sorted_vals

ytick_labels = []
for i in sort_idx:
    lbl = prov_short[i]
    if is_exporter[i]:
        lbl += ' \u2020'
    ytick_labels.append(lbl)

ax.set_yticks(y_pos)
ax.set_yticklabels(ytick_labels, fontsize=5.5)
ax.invert_yaxis()
ax.set_xlabel('Share of provincial demand (%)')
ax.set_xlim(0, 108)
ax.axvline(100, color='black', ls='-', lw=0.4, alpha=0.3)
ax.legend(fontsize=5, ncol=3, loc='lower left',
          bbox_to_anchor=(0.0, 1.005, 1.0, 0.08), mode='expand', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel b ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(-0.15, 1.02, 'b', fontsize=10, fontweight='bold', transform=ax2.transAxes, va='top')

ax2.axvline(u_p75, color='grey', ls='--', lw=0.5, alpha=0.5)
ax2.axhline(delta_p75, color='grey', ls='--', lw=0.5, alpha=0.5)

sizes = demand / demand.max() * 200
sizes = np.clip(sizes, 15, 200)

for i in range(N_PROV):
    if is_support[i]:
        ax2.scatter(u_cn370[i], delta_u[i], s=sizes[i], facecolors='none',
                   edgecolors='#2ca02c', linewidths=1.3, zorder=5, alpha=0.9)
    else:
        ax2.scatter(u_cn370[i], delta_u[i], s=sizes[i], c=CAT_COLORS[cat[i]],
                   edgecolors='white', linewidths=0.3, zorder=4, alpha=0.8)

OFFSETS = {
    'GD': (-7, -9, 'right'), 'HA': (-7, 5, 'right'), 'SD': (5, -8, 'left'),
    'HN': (5, 5, 'left'), 'JX': (5, -8, 'left'), 'GX': (5, 5, 'left'),
    'SN': (-7, -7, 'right'), 'GZ': (5, 5, 'left'), 'HB': (-7, 5, 'right'),
    'AH': (5, 5, 'left'), 'ZJ': (5, -7, 'left'), 'JS': (-7, 5, 'right'),
    'IM': (8, 6, 'left'), 'XJ': (8, -6, 'left'), 'GS': (8, 0, 'left'),
    'SC': (-8, 6, 'right'), 'QH': (8, 6, 'left'), 'YN': (-8, -6, 'right'),
    'BJ': (8, 6, 'left'), 'NX': (8, -6, 'left'),
    'SX': (-8, -6, 'right'), 'JL': (8, 6, 'left'), 'HL': (-8, 6, 'right'),
    'HI': (8, -8, 'left'), 'XZ': (-8, 8, 'right'),
    'TJ': (8, -6, 'left'), 'SH': (8, 6, 'left'),
    'FJ': (-8, -6, 'right'), 'LN': (8, -6, 'left'),
    'CQ': (-8, -8, 'right'), 'HE': (-8, 6, 'right'),
}

# Label provinces — text placed to the RIGHT of each point (or left if crowded on right)
# Skip lower-left cluster except key support provinces
KEY_SUPPORT = {'IM', 'SC', 'YN', 'XJ'}

# Manual side assignment: 'r' = right of point, 'l' = left of point
LABEL_SIDE = {
    'GD': 'l', 'HA': 'l', 'SD': 'r', 'HN': 'r', 'JX': 'r',
    'GX': 'r', 'GZ': 'l', 'SN': 'r', 'JS': 'l', 'ZJ': 'r', 'AH': 'r',
    'HB': 'l', 'LN': 'r',
    'IM': 'r', 'SC': 'l', 'YN': 'l', 'XJ': 'r',
}

for i in range(N_PROV):
    ps = prov_short[i]
    # Skip non-key provinces in lower-left cluster
    if u_cn370[i] < 25 and delta_u[i] < 6 and ps not in KEY_SUPPORT:
        continue
    side = LABEL_SIDE.get(ps, 'r')
    if side == 'r':
        ox, ha = 6, 'left'
    else:
        ox, ha = -6, 'right'
    color = '#2ca02c' if ps in KEY_SUPPORT else '#222222'
    ax2.annotate(ps, (u_cn370[i], delta_u[i]), xytext=(ox, 0),
                textcoords='offset points', fontsize=4.5, ha=ha, va='center',
                color=color, fontweight='bold')

ax2.set_xlabel('Residual flexibility demand (TWh/yr)')
ax2.set_ylabel('Climate sensitivity\n($\\Delta U$ = SSP3-7.0 $-$ SSP2-4.5, TWh/yr)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', markersize=6, label='Compound stress'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#fc8d59', markersize=6, label='Structural burden'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4393c3', markersize=6, label='Climate sensitive'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=6, label='Lower priority'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#2ca02c', markeredgewidth=1.3, markersize=6, label='Cross-prov. support'),
]
ax2.legend(handles=legend_elements, fontsize=4.5, loc='upper left',
           frameon=True, framealpha=0.95, edgecolor='#ddd', borderpad=0.3)

plt.tight_layout()
out = OUT_DIR / 'Fig6_typology_combined.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
