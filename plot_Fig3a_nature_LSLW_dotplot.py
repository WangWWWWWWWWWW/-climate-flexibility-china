# -*- coding: utf-8 -*-
"""Nature Figure 3a: Provincial LSLW frequency dot plot — Nature standard."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

DATA_DIR = DATA_DIR
OUT_DIR = FIGURES_DIR

PROVINCE_ORDER = [
    'Beijing','Tianjin','Hebei','Shanxi','Inner Mongolia',
    'Liaoning','Jilin','Heilongjiang','Shanghai','Jiangsu',
    'Zhejiang','Anhui','Fujian','Jiangxi','Shandong',
    'Henan','Hubei','Hunan','Guangdong','Guangxi',
    'Hainan','Chongqing','Sichuan','Guizhou','Yunnan',
    'Tibet','Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang'
]
PROV_ABBR = {
    'Beijing':'BJ','Tianjin':'TJ','Hebei':'HE','Shanxi':'SX',
    'Inner Mongolia':'NM','Liaoning':'LN','Jilin':'JL',
    'Heilongjiang':'HL','Shanghai':'SH','Jiangsu':'JS',
    'Zhejiang':'ZJ','Anhui':'AH','Fujian':'FJ','Jiangxi':'JX',
    'Shandong':'SD','Henan':'HA','Hubei':'HB','Hunan':'HN',
    'Guangdong':'GD','Guangxi':'GX','Hainan':'HI',
    'Chongqing':'CQ','Sichuan':'SC','Guizhou':'GZ',
    'Yunnan':'YN','Tibet':'XZ','Shaanxi':'SN','Gansu':'GS',
    'Qinghai':'QH','Ningxia':'NX','Xinjiang':'XJ'
}
REGIONS = {
    'North':['Beijing','Tianjin','Hebei','Shanxi','Inner Mongolia'],
    'Northeast':['Liaoning','Jilin','Heilongjiang'],
    'East':['Shanghai','Jiangsu','Zhejiang','Anhui','Fujian','Jiangxi','Shandong'],
    'Central':['Henan','Hubei','Hunan'],
    'South':['Guangdong','Guangxi','Hainan'],
    'Southwest':['Chongqing','Sichuan','Guizhou','Yunnan','Tibet'],
    'Northwest':['Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang'],
}
REGION_COLORS = {
    'North':'#c0392b','Northeast':'#2980b9','East':'#27ae60',
    'Central':'#8e44ad','South':'#e67e22','Southwest':'#795548',
    'Northwest':'#e91e90',
}
PROV_TO_REGION = {p: r for r, ps in REGIONS.items() for p in ps}
N_PROV = 31
PERIODS = ['historical','ssp245','ssp370']
COLORS = {'historical':'#555555', 'ssp245':'#4393c3', 'ssp370':'#d73027'}
LABELS = {'historical':'Historical', 'ssp245':'SSP2-4.5', 'ssp370':'SSP3-7.0'}
MARKERS = {'historical':'o', 'ssp245':'s', 'ssp370':'D'}

# Load
d = np.load(DATA_DIR / 'lslw_results.npz', allow_pickle=True)
freq_mean = {p: d[f'freq_{p}'] for p in PERIODS}
sort_idx = np.argsort(freq_mean['ssp370'])[::-1]

# Plot — Nature single column (89mm)
fig, ax = plt.subplots(figsize=(89/25.4, 170/25.4))
y_pos = np.arange(N_PROV)
dy = {'historical': 0.22, 'ssp245': 0.0, 'ssp370': -0.22}

for period in PERIODS:
    vals = freq_mean[period][sort_idx]
    ys = y_pos + dy[period]
    ax.scatter(vals, ys, c=COLORS[period], marker=MARKERS[period],
              s=18, edgecolors='white', linewidths=0.3, zorder=3, alpha=0.9,
              label=LABELS[period])

# Connecting lines
for i in range(N_PROV):
    pi = sort_idx[i]
    x_h = freq_mean['historical'][pi]
    x_3 = freq_mean['ssp370'][pi]
    ax.plot([x_h, x_3], [y_pos[i]+dy['historical'], y_pos[i]+dy['ssp370']],
            color='#e0e0e0', linewidth=0.4, zorder=1)

# Province labels
sorted_abbrs = [PROV_ABBR[PROVINCE_ORDER[i]] for i in sort_idx]
sorted_colors = [REGION_COLORS[PROV_TO_REGION[PROVINCE_ORDER[i]]] for i in sort_idx]
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_abbrs, fontsize=6)
for tl, col in zip(ax.get_yticklabels(), sorted_colors):
    tl.set_color(col)
    tl.set_fontweight('bold')

ax.set_xlabel('LSLW frequency (days/yr)', fontsize=7)
ax.set_ylim(N_PROV - 0.5, -0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Alternating shading
for i in range(0, N_PROV, 2):
    ax.axhspan(i - 0.45, i + 0.45, color='#f5f5f5', zorder=0)

# Legends
ssp_handles = [plt.Line2D([], [], color=COLORS[p], marker=MARKERS[p], linestyle='None',
               markersize=3.5, markeredgecolor='white', markeredgewidth=0.3,
               label=LABELS[p]) for p in PERIODS]
region_handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]

ax.legend(handles=ssp_handles, fontsize=5.5, loc='lower right', frameon=True,
          framealpha=0.95, edgecolor='#ddd', ncol=1,
          handletextpad=0.3, borderpad=0.3)

# Region legend as second legend — place at bottom-left (no data overlap there)
leg2 = fig.legend(handles=region_handles, fontsize=4.5, loc='lower left',
                  ncol=4, frameon=True, framealpha=0.95, edgecolor='#ddd',
                  bbox_to_anchor=(0.05, -0.01), handletextpad=0.3,
                  columnspacing=0.5, borderpad=0.3)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
out = OUT_DIR / 'Fig3a_LSLW_frequency_dotplot.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out.name}')
