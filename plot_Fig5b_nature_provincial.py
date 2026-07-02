# -*- coding: utf-8 -*-
"""Nature Figure 5b: Provincial residual flexibility decomposition — Nature standard."""

import numpy as np
import pandas as pd
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
    'legend.fontsize': 5.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

OUT_DIR = FIGURES_DIR

PROVINCE_ORDER = [
    'Beijing','Tianjin','Hebei','Shanxi','Inner Mongolia',
    'Liaoning','Jilin','Heilongjiang','Shanghai','Jiangsu',
    'Zhejiang','Anhui','Fujian','Jiangxi','Shandong',
    'Henan','Hubei','Hunan','Guangdong','Guangxi',
    'Hainan','Chongqing','Sichuan','Guizhou','Yunnan',
    'Tibet','Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang'
]
PROV_SHORT = [
    'BJ','TJ','HE','SX','IM','LN','JL','HL','SH','JS',
    'ZJ','AH','FJ','JX','SD','HA','HB','HN','GD','GX',
    'HI','CQ','SC','GZ','YN','XZ','SN','GS','QH','NX','XJ'
]
REGION_PROVINCES = {
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
PROV_TO_REGION = {p: r for r, ps in REGION_PROVINCES.items() for p in ps}

CLR_BASE = '#bdbdbd'
CLR_MOD  = '#92c5de'
CLR_DEEP = '#2166ac'
CLR_CLIM = '#d73027'

# Load package-local source data generated for Fig. 5b.
df = pd.read_excel(SOURCE_DATA_DIR / 'table_Fig5b_provincial_residual.xlsx')

comp_baseline = df['Baseline_NDC_SSP245_TWh'].values.astype(float)
comp_mod_decarb = df['Moderate_Decarb_GM_minus_NDC_TWh'].values.astype(float)
comp_deep_decarb = df['Deep_Decarb_CN_minus_GM_TWh'].values.astype(float)
comp_climate = df['Climate_SSP370_minus_SSP245_TWh'].values.astype(float)
u_cn370 = df['Total_CN2050_SSP370_TWh'].values.astype(float)

sort_idx = np.argsort(u_cn370)[::-1]
keep = u_cn370[sort_idx] > 0.5
sort_f = sort_idx[keep]
n_show = len(sort_f)

# Nature single column: 89mm wide
fig, ax = plt.subplots(figsize=(89/25.4, 160/25.4))

y_pos = np.arange(n_show)
bar_h = 0.6

v1 = comp_baseline[sort_f]
v2 = comp_mod_decarb[sort_f]
v3 = comp_deep_decarb[sort_f]
v4 = comp_climate[sort_f]

left = np.zeros(n_show)
ax.barh(y_pos, v1, height=bar_h, left=left, color=CLR_BASE,
        edgecolor='white', linewidth=0.2, label='Baseline (NDC, SSP2-4.5)')
left += v1
ax.barh(y_pos, v2, height=bar_h, left=left, color=CLR_MOD,
        edgecolor='white', linewidth=0.2, label='Moderate decarb')
left += v2
ax.barh(y_pos, v3, height=bar_h, left=left, color=CLR_DEEP,
        edgecolor='white', linewidth=0.2, label='Deep decarb')
left += v3
ax.barh(y_pos, v4, height=bar_h, left=left, color=CLR_CLIM,
        edgecolor='white', linewidth=0.2, label='Climate ($\\Delta$SSP)')

# Y labels
ax.set_yticks(y_pos)
labels = [PROV_SHORT[i] for i in sort_f]
ax.set_yticklabels(labels, fontsize=5.5)
ax.invert_yaxis()
for tick, idx in zip(ax.get_yticklabels(), sort_f):
    tick.set_color(REGION_COLORS[PROV_TO_REGION[PROVINCE_ORDER[idx]]])
    tick.set_fontweight('bold')

ax.set_xlabel('Residual flexibility demand (TWh/yr)')
ax.set_xlim(0, u_cn370[sort_f].max() * 1.08)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(fontsize=5, loc='lower right', frameon=True, framealpha=0.95,
          edgecolor='#ddd', borderpad=0.3)

plt.tight_layout(pad=0.5)
out = OUT_DIR / 'Fig5b_provincial_residual.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f'Saved: {out.name}')
