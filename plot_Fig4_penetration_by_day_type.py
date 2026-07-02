# -*- coding: utf-8 -*-
"""
Fig 4: Renewable penetration rate by day type
  4A: Normal / LSLW / Synchronized LSLW (≥3 provinces) penetration distribution
  4B: Frequency of low-penetration days under SSP245 vs SSP370

Output: figures/Main_Fig04_penetration_by_day_type.png
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
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.linewidth': 0.7,
    'figure.dpi': 300,
})

# ============================================================
# Paths & Constants
# ============================================================
DATA_DIR = DATA_DIR
OUT_DIR = FIGURES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVINCE_ORDER = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia',
    'Liaoning', 'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu',
    'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong',
    'Henan', 'Hubei', 'Hunan', 'Guangdong', 'Guangxi',
    'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 'Yunnan',
    'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]
N_PROV = 31

GCM_LIST = ['ACCESS-CM2', 'EC-Earth3', 'GFDL-ESM4', 'MRI-ESM2-0', 'NorESM2-MM']
SSP_LIST = ['ssp245', 'ssp370']
SCENARIOS = ['NDC', 'GM2.0', 'CN2050']
SCENARIO_FILE = {'NDC': 'NDC', 'GM2.0': 'GM2.0', 'CN2050': 'CN2050'}

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
FUTURE_YEARS = 25

SYNC_THRESHOLD = 3  # ≥3 provinces simultaneous LSLW = synchronized

COLORS_SSP = {'ssp245': '#4393c3', 'ssp370': '#d73027'}
COLORS_SCENARIO = {'NDC': '#666666', 'GM2.0': '#74add1', 'CN2050': '#d73027'}

# ============================================================
# Helper: months array → day-of-year
# ============================================================
def months_to_doy(months_arr):
    doy = np.zeros(len(months_arr), dtype=int)
    month_start = np.cumsum([0] + DAYS_PER_MONTH[:-1])
    current_month = -1
    day_in_month = 0
    for i, m in enumerate(months_arr):
        if m != current_month:
            current_month = m
            day_in_month = 0
        doy[i] = month_start[m] + day_in_month
        day_in_month += 1
    return doy % 365

# ============================================================
# Load LSLW thresholds
# ============================================================
print('[1] Loading LSLW P10 thresholds...')
lslw_data = np.load(DATA_DIR / 'lslw_results.npz')
wind_p10_daily  = lslw_data['wind_p10_daily']   # (365, 31)
solar_p10_daily = lslw_data['solar_p10_daily']   # (365, 31)

# ============================================================
# Compute daily penetration + classify days
# ============================================================
print('[2] Computing daily penetration rates for all scenarios...')

results = {}  # (scenario, ssp) → dict with arrays

for sc in SCENARIOS:
    for ssp in SSP_LIST:
        sc_file = SCENARIO_FILE[sc]
        sd = np.load(DATA_DIR / f'supply_demand_2050_{sc_file}_{ssp}.npz')
        cap_wind  = sd['cap_wind']       # (31,) MW
        cap_solar = sd['cap_solar']      # (31,) MW
        monthly_demand = sd['monthly_demand']  # (12, 31) TWh

        # Daily demand: monthly_demand / days_in_month  → TWh/day
        daily_demand_by_month = np.zeros((12, N_PROV))
        for m in range(12):
            daily_demand_by_month[m] = monthly_demand[m] / DAYS_PER_MONTH[m]

        # pen_national: for normal days (whole country)
        # pen_lslw: for LSLW days (only LSLW-affected provinces)
        all_pen_national = []   # national penetration on normal days
        all_pen_lslw = []       # LSLW-province penetration on LSLW/sync days
        all_day_type = []       # 0=normal, 1=LSLW(isolated), 2=synchronized LSLW
        all_n_sync = []         # number of provinces in LSLW that day

        for gcm in GCM_LIST:
            fpath = DATA_DIR / f'CMIP6_daily_CF_corrected_{ssp}_{gcm}.npz'
            d = np.load(fpath)
            cf_wind  = d['cf_wind']   # (n_days, 31)
            cf_solar = d['cf_solar']  # (n_days, 31)
            months   = d['months']    # (n_days,)
            doy_arr  = months_to_doy(months)
            n_days   = cf_wind.shape[0]

            for i in range(n_days):
                m = int(months[i])
                doy = doy_arr[i]

                # Daily generation (TWh) per province
                gen_w = cap_wind * cf_wind[i] * 24 / 1e6
                gen_s = cap_solar * cf_solar[i] * 24 / 1e6
                gen_re = gen_w + gen_s  # (31,)

                demand_day = daily_demand_by_month[m]  # (31,)

                # LSLW classification per province
                is_lslw = (cf_wind[i] <= wind_p10_daily[doy]) & \
                           (cf_solar[i] <= solar_p10_daily[doy])
                n_lslw_prov = is_lslw.sum()

                if n_lslw_prov >= SYNC_THRESHOLD:
                    day_type = 2  # synchronized
                elif n_lslw_prov >= 1:
                    day_type = 1  # isolated LSLW
                else:
                    day_type = 0  # normal

                if day_type == 0:
                    # Normal day: national penetration
                    pen = gen_re.sum() / demand_day.sum() * 100
                else:
                    # LSLW day: penetration of LSLW-affected provinces only
                    lslw_re = gen_re[is_lslw].sum()
                    lslw_dem = demand_day[is_lslw].sum()
                    pen = lslw_re / lslw_dem * 100 if lslw_dem > 0 else 0

                all_pen_national.append(gen_re.sum() / demand_day.sum() * 100)
                all_pen_lslw.append(pen)
                all_day_type.append(day_type)
                all_n_sync.append(n_lslw_prov)

        all_pen_national = np.array(all_pen_national)
        all_pen_lslw = np.array(all_pen_lslw)
        all_day_type = np.array(all_day_type)
        all_n_sync = np.array(all_n_sync)

        results[(sc, ssp)] = {
            'pen_national': all_pen_national,
            'pen_lslw': all_pen_lslw,
            'day_type': all_day_type,
            'n_sync': all_n_sync,
        }

        # Quick stats
        mask_normal = all_day_type == 0
        mask_lslw   = all_day_type == 1
        mask_sync   = all_day_type == 2
        print(f'  {sc:8s} {ssp}: normal={mask_normal.sum()}, '
              f'LSLW={mask_lslw.sum()}, sync={mask_sync.sum()}, '
              f'pen: normal={all_pen_national[mask_normal].mean():.1f}%, '
              f'LSLW_prov={all_pen_lslw[mask_lslw].mean():.1f}%, '
              f'sync_prov={all_pen_lslw[mask_sync].mean():.1f}%')

# ============================================================
# Save intermediate data
# ============================================================
print('[3] Saving intermediate data...')
save_dict = {}
for (sc, ssp), v in results.items():
    key_prefix = f'{SCENARIO_FILE[sc]}_{ssp}'
    save_dict[f'pen_nat_{key_prefix}'] = v['pen_national']
    save_dict[f'pen_lslw_{key_prefix}'] = v['pen_lslw']
    save_dict[f'dtype_{key_prefix}'] = v['day_type']
    save_dict[f'nsync_{key_prefix}'] = v['n_sync']
np.savez_compressed(DATA_DIR / 'fig4_penetration_by_day_type.npz', **save_dict)
print(f"Saved intermediate data: {DATA_DIR / 'fig4_penetration_by_day_type.npz'}")
print('\nDone!')
raise SystemExit(0)

# ============================================================
# Plot
# ============================================================
print('[4] Plotting...')

fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={'width_ratios': [1.2, 1]})
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.93, wspace=0.35)

# --- 4A: Box/violin plot of penetration by day type, for CN2050 ---
ax = axes[0]
ax.set_title('a', fontsize=11, fontweight='bold', loc='left', pad=6)

# Use CN2050 — Normal vs Synchronized LSLW
day_labels = ['Normal', 'LSLW']
positions_245 = [0.8, 1.8]
positions_370 = [1.2, 2.2]
box_width = 0.35

from matplotlib.patches import Patch

for ssp, positions, color in [
    ('ssp245', positions_245, COLORS_SSP['ssp245']),
    ('ssp370', positions_370, COLORS_SSP['ssp370']),
]:
    r = results[('CN2050', ssp)]
    pen_nat = r['pen_national']
    pen_lslw = r['pen_lslw']
    dtype = r['day_type']

    data_groups = [
        pen_nat[dtype == 0],
        pen_lslw[dtype == 2],
    ]

    # Box = IQR, black line = median, dot = mean
    box_stats = []
    for data in data_groups:
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        box_stats.append({
            'med': med, 'q1': q1, 'q3': q3,
            'whislo': q1, 'whishi': q3,
            'fliers': [],
            'mean': data.mean()
        })

    bp = ax.bxp(box_stats, positions=positions, widths=box_width,
                patch_artist=True, showfliers=False, showmeans=True,
                meanprops=dict(marker='o', markerfacecolor='white',
                               markeredgecolor='white', markersize=4),
                medianprops=dict(color='white', linewidth=1.2),
                whiskerprops=dict(linewidth=0),
                capprops=dict(linewidth=0))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('none')
        patch.set_linewidth(0)

ax.set_xticks([1.0, 2.0])
ax.set_xticklabels(day_labels, fontsize=8.5)
ax.set_ylabel('Renewable penetration (%)', fontsize=8.5)
ax.set_xlim(0.3, 2.7)

legend_elements = [Patch(facecolor=COLORS_SSP['ssp245'], alpha=0.6, label='SSP2-4.5'),
                   Patch(facecolor=COLORS_SSP['ssp370'], alpha=0.6, label='SSP3-7.0')]
ax.legend(handles=legend_elements, fontsize=7.5, loc='upper right', framealpha=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- 4B: Frequency of low-penetration days (< 50%) across scenarios ---
ax2 = axes[1]
ax2.set_title('b', fontsize=11, fontweight='bold', loc='left', pad=6)

LOW_PEN_THRESHOLD = 50  # %

bar_data = {}
for sc in SCENARIOS:
    for ssp in SSP_LIST:
        r = results[(sc, ssp)]
        pen = r['pen_national']  # national penetration for all days
        n_low = (pen < LOW_PEN_THRESHOLD).sum()
        freq_yr = n_low / (len(GCM_LIST) * FUTURE_YEARS)
        bar_data[(sc, ssp)] = freq_yr

x = np.arange(len(SCENARIOS))
bar_w = 0.32

bars_245 = [bar_data[(sc, 'ssp245')] for sc in SCENARIOS]
bars_370 = [bar_data[(sc, 'ssp370')] for sc in SCENARIOS]

b1 = ax2.bar(x - bar_w/2, bars_245, bar_w, color=COLORS_SSP['ssp245'],
             alpha=0.75, label='SSP2-4.5', edgecolor='white', linewidth=0.5)
b2 = ax2.bar(x + bar_w/2, bars_370, bar_w, color=COLORS_SSP['ssp370'],
             alpha=0.75, label='SSP3-7.0', edgecolor='white', linewidth=0.5)

# Value labels on top of bars
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2,
                 f'{h:.0f}', ha='center', va='bottom', fontsize=7)

ax2.set_xticks(x)
ax2.set_xticklabels(SCENARIOS, fontsize=8.5)
ax2.set_ylabel('Low-penetration days per year', fontsize=8.5)
ax2.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Save
outpath = OUT_DIR / 'Main_Fig04_penetration_by_day_type.png'
fig.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'\nSaved: {outpath}')

# Also save PDF
fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
print(f'Saved: {outpath.with_suffix(".pdf")}')

plt.close()
print('\nDone!')
