# -*- coding: utf-8 -*-
"""
Extended Data Figure: Attribution of high-emissions climate sensitivity.
Signed stacked bar chart showing wind, solar, demand, and nonlinear coupling
contributions to the SSP3-7.0 minus SSP2-4.5 residual flexibility increment.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from path_config import FIGURES_DIR, PROCESSED_RESULTS_DIR, ensure_output_dirs

# Paths
DATA_PATH = PROCESSED_RESULTS_DIR / 'step3F_attribution_results.csv'
OUT_DIR = FIGURES_DIR
ensure_output_dirs()

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 7,
    'axes.linewidth': 0.5,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 6.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Colors
COLORS = {
    'wind': '#4393c3',
    'solar': '#fdae61',
    'demand': '#d73027',
    'interaction': '#878787',
}

# Load data
df = pd.read_csv(DATA_PATH)
df = df.set_index('pathway').loc[['NDC', 'GM2.0', 'CN2050']]

# Plot
fig, ax = plt.subplots(figsize=(89/25.4, 70/25.4), constrained_layout=True)

x = np.arange(len(df))
width = 0.5
pos_bottom = np.zeros(len(df))
neg_bottom = np.zeros(len(df))

components = ['wind', 'solar', 'demand']
labels = ['Wind', 'Solar', 'Demand']

for comp, label in zip(components, labels):
    vals = df[f'{comp}_TWh_yr'].values.astype(float)
    bottoms = np.where(vals >= 0, pos_bottom, neg_bottom)
    ax.bar(x, vals, width, bottom=bottoms, color=COLORS[comp],
           edgecolor='white', linewidth=0.4, label=label, zorder=3)
    pos_bottom += np.where(vals > 0, vals, 0)
    neg_bottom += np.where(vals < 0, vals, 0)

# Total with error bars
totals = df['total_TWh_yr'].values.astype(float)
total_std = df['total_std_TWh_yr'].values.astype(float)
ax.errorbar(x, totals, yerr=total_std, fmt='D', color='black',
            markersize=3.5, markeredgewidth=0, elinewidth=0.7,
            capsize=2, capthick=0.7, zorder=6, label='Net total')

# Annotate totals
for i, (v, s) in enumerate(zip(totals, total_std)):
    ax.text(x[i], v + s + 12, f'{v:.0f}', ha='center', va='bottom', fontsize=6.5)

ax.axhline(0, color='#333333', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(['NDC', 'GM2.0', 'CN2050'])
ax.set_ylabel('Residual flexibility increment\n(TWh yr$^{-1}$)')
ax.set_ylim(-80, 450)
ax.set_xlim(-0.5, 2.5)
ax.grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.7, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=False, ncol=3, fontsize=5.5, loc='upper left',
          bbox_to_anchor=(0, 1.02), columnspacing=0.8, handletextpad=0.4)

out = OUT_DIR / 'ED_Fig8_attribution_stacked.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
print(f'Saved: {out.with_suffix(".pdf")}')
