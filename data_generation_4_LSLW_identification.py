# -*- coding: utf-8 -*-
"""
step3D_compound_drought.py  (v3 - 25yr future, 6 scenarios, aligned with Wang NSR 2025)

Step3D: Compound Energy Drought (LSLW) + Spatial Synchronization + Transmission Resilience

Key methodology changes from v1:
  - QM-corrected daily CF (wind CF ~0.22, not raw 0.04)
  - Historical period: 1990-2014 (25 years, not 5)
  - Rolling daily P10 threshold (+/-7 day window, per calendar day)
  - months encoded as 0-11 (properly handled)

Run this AFTER step3A2_daily_QM_correction.py completes.

Input:
  - Corrected daily CF: CMIP6_daily_CF_corrected_{period}_{gcm}.npz
  - Hydro CF: hydro_monthly_climatology.npz
  - Supply-demand: supply_demand_2050_{scenario}_{ssp}.npz
  - Transmission: transmission_network_data.xlsx

Output:
  - data/lslw_results.npz
  - figure/fig_lslw_*.png
  - excel/lslw_results.xlsx
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    'figure.dpi': 150,
})

# ============================================================
# Paths
# ============================================================
DATA_DIR = DATA_DIR
FIG_DIR = FIGURES_DIR
EXCEL_DIR = SOURCE_DATA_DIR
HYDRO_CF = HYDRO_PATH
TRANS_FILE = TX_PATH

for d in [FIG_DIR, EXCEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Constants
# ============================================================
PROVINCE_ORDER = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia',
    'Liaoning', 'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu',
    'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong',
    'Henan', 'Hubei', 'Hunan', 'Guangdong', 'Guangxi',
    'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 'Yunnan',
    'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]
N_PROV = 31

ABBR2EN = {
    'BJ': 'Beijing', 'TJ': 'Tianjin', 'HE': 'Hebei', 'SX': 'Shanxi',
    'NM': 'Inner Mongolia', 'LN': 'Liaoning', 'JL': 'Jilin', 'HL': 'Heilongjiang',
    'SH': 'Shanghai', 'JS': 'Jiangsu', 'ZJ': 'Zhejiang', 'AH': 'Anhui',
    'FJ': 'Fujian', 'JX': 'Jiangxi', 'SD': 'Shandong', 'HA': 'Henan',
    'HB': 'Hubei', 'HN': 'Hunan', 'GD': 'Guangdong', 'GX': 'Guangxi',
    'HI': 'Hainan', 'CQ': 'Chongqing', 'SC': 'Sichuan', 'GZ': 'Guizhou',
    'YN': 'Yunnan', 'XZ': 'Tibet', 'SN': 'Shaanxi', 'GS': 'Gansu',
    'QH': 'Qinghai', 'NX': 'Ningxia', 'XJ': 'Xinjiang'
}
EN2ABBR = {v: k for k, v in ABBR2EN.items()}

GCM_LIST = [
    'ACCESS-CM2', 'EC-Earth3',
    'GFDL-ESM4', 'MRI-ESM2-0', 'NorESM2-MM'
]
N_GCM = len(GCM_LIST)
SSP_LIST = ['ssp245', 'ssp370']

HIST_YEARS = 25   # 1990-2014
FUTURE_YEARS = 25  # 2036-2060

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Threshold percentile (primary P10, sensitivity P15/P20)
PRIMARY_PERCENTILE = 10
ROLLING_WINDOW = 7  # +/- 7 days for daily P10

LINE_LOSS = 0.0482

REGIONS = {
    'North': ['Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia'],
    'Northeast': ['Liaoning', 'Jilin', 'Heilongjiang'],
    'East': ['Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong'],
    'Central': ['Henan', 'Hubei', 'Hunan'],
    'South': ['Guangdong', 'Guangxi', 'Hainan'],
    'Southwest': ['Chongqing', 'Sichuan', 'Guizhou', 'Yunnan', 'Tibet'],
    'Northwest': ['Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang']
}

COLORS_SSP = {'historical': '#333333', 'ssp245': '#4393c3', 'ssp370': '#d73027'}
COLORS_SSP_LIGHT = {'historical': '#999999', 'ssp245': '#92c5de', 'ssp370': '#f4a582'}

print('=' * 70)
print('Step3D v2: Compound Energy Drought + Spatial Synchronization')
print('          + Transmission Resilience')
print('          (QM-corrected, 25yr historical, rolling daily P10)')
print('=' * 70)

# ============================================================
# Helper: load corrected daily CF
# ============================================================
def load_daily_cf(period, gcm):
    """Load QM-corrected daily CF."""
    fpath = DATA_DIR / f'CMIP6_daily_CF_corrected_{period}_{gcm}.npz'
    d = np.load(fpath)
    return d['cf_wind'], d['cf_solar'], d['months']

# ============================================================
# Phase 1: Rolling daily P10 thresholds from historical
# ============================================================
print('\n' + '=' * 70)
print('Phase 1: Compute rolling daily P10 thresholds (1990-2014)')
print('=' * 70)

# Collect historical data: day-of-year for rolling window
print('\n[1a] Loading all historical corrected daily CF...')

# We need day-of-year (DOY) for each day in the historical data
# months is 0-11 in the npz files
# Reconstruct DOY from months array

def months_to_doy(months_arr):
    """Convert months (0-11) array to day-of-year (0-364) array.
    Approximate: assumes non-leap, assigns sequential DOY within each month."""
    doy = np.zeros(len(months_arr), dtype=int)
    month_start = np.cumsum([0] + DAYS_PER_MONTH[:-1])  # [0, 31, 59, ...]
    # Track position within current month
    current_month = -1
    day_in_month = 0
    for i, m in enumerate(months_arr):
        if m != current_month:
            current_month = m
            day_in_month = 0
        doy[i] = month_start[m] + day_in_month
        day_in_month += 1
    return doy % 365  # wrap leap day

# Collect all historical data with DOY labels
hist_data_by_doy = {doy: {'wind': [], 'solar': []} for doy in range(365)}

for gi, gcm in enumerate(GCM_LIST):
    cf_w, cf_s, months = load_daily_cf('historical', gcm)
    doy_arr = months_to_doy(months)

    for i in range(len(doy_arr)):
        d = doy_arr[i]
        hist_data_by_doy[d]['wind'].append(cf_w[i])    # (31,)
        hist_data_by_doy[d]['solar'].append(cf_s[i])

    if gi == 0:
        print(f'  {gcm}: {cf_w.shape[0]} days, wind={cf_w.mean():.4f}, solar={cf_s.mean():.4f}')

print(f'  Samples per DOY: ~{len(hist_data_by_doy[0]["wind"])} '
      f'(expected ~{HIST_YEARS * N_GCM})')

# Compute rolling P10: for each DOY, pool data from DOY +/- ROLLING_WINDOW
print(f'\n[1b] Computing rolling P10 (window=+/-{ROLLING_WINDOW} days)...')

wind_p10_daily = np.zeros((365, N_PROV))   # (365, 31)
solar_p10_daily = np.zeros((365, N_PROV))

for doy in range(365):
    # Collect data from window
    wind_pool = []
    solar_pool = []
    for offset in range(-ROLLING_WINDOW, ROLLING_WINDOW + 1):
        d = (doy + offset) % 365
        wind_pool.extend(hist_data_by_doy[d]['wind'])
        solar_pool.extend(hist_data_by_doy[d]['solar'])

    wind_arr = np.array(wind_pool)   # (n_samples, 31)
    solar_arr = np.array(solar_pool)

    for pi in range(N_PROV):
        wind_p10_daily[doy, pi] = np.percentile(wind_arr[:, pi], PRIMARY_PERCENTILE)
        solar_p10_daily[doy, pi] = np.percentile(solar_arr[:, pi], PRIMARY_PERCENTILE)

print(f'  Wind P10 range:  {wind_p10_daily.min():.4f} - {wind_p10_daily.max():.4f}')
print(f'  Solar P10 range: {solar_p10_daily.min():.4f} - {solar_p10_daily.max():.4f}')
print(f'  Wind P10 annual mean: {wind_p10_daily.mean():.4f}')
print(f'  Solar P10 annual mean: {solar_p10_daily.mean():.4f}')

# Show examples
for prov in ['Inner Mongolia', 'Guangdong', 'Sichuan', 'Xinjiang']:
    pi = PROVINCE_ORDER.index(prov)
    print(f'  {prov:18s}: wind_P10 mean={wind_p10_daily[:, pi].mean():.4f}, '
          f'solar_P10 mean={solar_p10_daily[:, pi].mean():.4f}')

# ============================================================
# Phase 1c: Identify LSLW days
# ============================================================
print('\n[1c] Identifying LSLW days...')

lslw_days = {}    # period -> list of (n_days, 31) bool per GCM
lslw_months = {}  # period -> list of (n_days,) int per GCM
lslw_doys = {}    # period -> list of (n_days,) int per GCM

for period in ['historical'] + SSP_LIST:
    lslw_days[period] = []
    lslw_months[period] = []
    lslw_doys[period] = []

    for gi, gcm in enumerate(GCM_LIST):
        cf_w, cf_s, months = load_daily_cf(period, gcm)
        doy_arr = months_to_doy(months)
        n_days = cf_w.shape[0]

        # LSLW: wind <= P10(doy) AND solar <= P10(doy)
        is_lslw = np.zeros((n_days, N_PROV), dtype=bool)
        for i in range(n_days):
            d = doy_arr[i]
            is_lslw[i] = (cf_w[i] <= wind_p10_daily[d]) & (cf_s[i] <= solar_p10_daily[d])

        lslw_days[period].append(is_lslw)
        lslw_months[period].append(months)
        lslw_doys[period].append(doy_arr)

    n_years = HIST_YEARS if period == 'historical' else FUTURE_YEARS
    total_lslw = sum(d.sum() for d in lslw_days[period])
    total_prov_days = sum(d.shape[0] * N_PROV for d in lslw_days[period])
    freq_mean = total_lslw / (N_GCM * n_years * N_PROV)
    print(f'  {period:12s}: {total_lslw:,} LSLW prov-days, '
          f'mean={freq_mean:.1f} days/prov/yr ({total_lslw/total_prov_days*100:.2f}%)')

# ============================================================
# Phase 2: Frequency, duration, seasonality
# ============================================================
print('\n' + '=' * 70)
print('Phase 2: LSLW frequency, duration, seasonality')
print('=' * 70)

def compute_freq(lslw_list, n_years):
    """Annual LSLW days per province, GCM-averaged."""
    freq_gcm = np.zeros((N_GCM, N_PROV))
    for gi in range(N_GCM):
        freq_gcm[gi] = lslw_list[gi].sum(axis=0) / n_years
    return freq_gcm.mean(axis=0), freq_gcm.std(axis=0)

def compute_max_consecutive(lslw_arr):
    """Max consecutive LSLW days per province."""
    n_days, n_prov = lslw_arr.shape
    max_cons = np.zeros(n_prov)
    for pi in range(n_prov):
        seq = lslw_arr[:, pi].astype(int)
        current = 0
        for d in range(n_days):
            if seq[d]:
                current += 1
                max_cons[pi] = max(max_cons[pi], current)
            else:
                current = 0
    return max_cons

def compute_monthly_freq(lslw_list, months_list, n_years):
    """Monthly LSLW frequency (days/month), GCM-averaged."""
    monthly = np.zeros((N_GCM, 12, N_PROV))
    for gi in range(N_GCM):
        for m in range(12):
            mask = months_list[gi] == m  # months are 0-11
            if mask.sum() > 0:
                monthly[gi, m] = lslw_list[gi][mask].sum(axis=0) / n_years
    return monthly.mean(axis=0), monthly.std(axis=0)

# 2a. Annual frequency
print('\n[2a] Annual LSLW frequency...')
freq, freq_std = {}, {}
for period in ['historical'] + SSP_LIST:
    n_yr = HIST_YEARS if period == 'historical' else FUTURE_YEARS
    freq[period], freq_std[period] = compute_freq(lslw_days[period], n_yr)

print(f'  {"Province":18s} {"Hist":>6s} {"SSP245":>7s} {"SSP370":>7s} {"d370":>8s}')
for pi in range(N_PROV):
    h, s2, s3 = freq['historical'][pi], freq['ssp245'][pi], freq['ssp370'][pi]
    delta = (s3 - h) / h * 100 if h > 0.1 else 0
    print(f'  {PROVINCE_ORDER[pi]:18s} {h:6.1f} {s2:7.1f} {s3:7.1f} {delta:+7.0f}%')

for period in ['historical'] + SSP_LIST:
    print(f'  National mean ({period}): {freq[period].mean():.1f} +/- {freq_std[period].mean():.1f}')

d245 = (freq['ssp245'].mean() - freq['historical'].mean()) / freq['historical'].mean() * 100
d370 = (freq['ssp370'].mean() - freq['historical'].mean()) / freq['historical'].mean() * 100
print(f'  Change: SSP245 {d245:+.1f}%, SSP370 {d370:+.1f}%')

# 2b. Max consecutive
print('\n[2b] Max consecutive LSLW days...')
max_cons = {}
for period in ['historical'] + SSP_LIST:
    cons_gcm = np.zeros((N_GCM, N_PROV))
    for gi in range(N_GCM):
        cons_gcm[gi] = compute_max_consecutive(lslw_days[period][gi])
    max_cons[period] = cons_gcm.mean(axis=0)
    print(f'  {period}: national mean = {max_cons[period].mean():.1f} days')

# 2c. Monthly seasonality
print('\n[2c] Monthly seasonality...')
monthly_freq, monthly_freq_std = {}, {}
for period in ['historical'] + SSP_LIST:
    n_yr = HIST_YEARS if period == 'historical' else FUTURE_YEARS
    monthly_freq[period], monthly_freq_std[period] = compute_monthly_freq(
        lslw_days[period], lslw_months[period], n_yr)

print(f'  {"Month":>5s} {"Hist":>6s} {"SSP245":>7s} {"SSP370":>7s}')
for m in range(12):
    h = monthly_freq['historical'][m].mean()
    s2 = monthly_freq['ssp245'][m].mean()
    s3 = monthly_freq['ssp370'][m].mean()
    print(f'  {MONTH_LABELS[m]:>5s} {h:6.2f} {s2:7.2f} {s3:7.2f}')

# ============================================================
# Phase 3: Spatial synchronization
# ============================================================
print('\n' + '=' * 70)
print('Phase 3: Spatial synchronization')
print('=' * 70)

def compute_sync_distribution(lslw_list, n_years):
    """P(N_sync >= k) in days/year."""
    sync_days = np.zeros((N_GCM, N_PROV + 1))
    for gi in range(N_GCM):
        daily_count = lslw_list[gi].sum(axis=1)
        for k in range(N_PROV + 1):
            sync_days[gi, k] = (daily_count >= k).sum() / n_years
    return sync_days.mean(axis=0), sync_days.std(axis=0)

print('\n[3a] Synchronization distribution...')
sync_dist, sync_dist_std = {}, {}
for period in ['historical'] + SSP_LIST:
    n_yr = HIST_YEARS if period == 'historical' else FUTURE_YEARS
    sync_dist[period], sync_dist_std[period] = compute_sync_distribution(
        lslw_days[period], n_yr)

print(f'  {"k":>3s} {"Hist":>8s} {"SSP245":>8s} {"SSP370":>8s} {"d370":>8s}')
for k in [1, 3, 5, 8, 10, 15]:
    h, s2, s3 = sync_dist['historical'][k], sync_dist['ssp245'][k], sync_dist['ssp370'][k]
    d3 = (s3 - h) / h * 100 if h > 0.01 else float('nan')
    print(f'  {k:3d} {h:8.1f} {s2:8.1f} {s3:8.1f} {d3:+7.0f}%')

# 3b. Pairwise Jaccard
print('\n[3b] Pairwise Jaccard similarity...')

def compute_pairwise_jaccard(lslw_list):
    all_lslw = np.concatenate(lslw_list, axis=0)
    jaccard = np.zeros((N_PROV, N_PROV))
    for i in range(N_PROV):
        for j in range(i + 1, N_PROV):
            inter = (all_lslw[:, i] & all_lslw[:, j]).sum()
            union = (all_lslw[:, i] | all_lslw[:, j]).sum()
            if union > 0:
                jaccard[i, j] = inter / union
                jaccard[j, i] = jaccard[i, j]
    return jaccard

jaccard = {p: compute_pairwise_jaccard(lslw_days[p])
           for p in ['historical'] + SSP_LIST}

# Top pairs
pairs = []
for i in range(N_PROV):
    for j in range(i+1, N_PROV):
        pairs.append((PROVINCE_ORDER[i], PROVINCE_ORDER[j],
                      jaccard['historical'][i,j], jaccard['ssp370'][i,j]))
pairs.sort(key=lambda x: -x[3])
print(f'  Top 10 co-occurring pairs (SSP370):')
for p1, p2, jh, j3 in pairs[:10]:
    ch = (j3 - jh) / jh * 100 if jh > 0.001 else float('nan')
    print(f'    {p1:18s} - {p2:18s}: hist={jh:.3f}, SSP370={j3:.3f} ({ch:+.0f}%)')

# 3c. Regional sync
print('\n[3c] Regional synchronization...')
def compute_regional_sync(lslw_list, indices, n_years):
    sync = np.zeros(N_GCM)
    for gi in range(N_GCM):
        count = lslw_list[gi][:, indices].sum(axis=1)
        sync[gi] = (count >= len(indices) / 2).sum() / n_years
    return sync.mean()

for rname, provs in REGIONS.items():
    idx = [PROVINCE_ORDER.index(p) for p in provs]
    h = compute_regional_sync(lslw_days['historical'], idx, HIST_YEARS)
    s3 = compute_regional_sync(lslw_days['ssp370'], idx, FUTURE_YEARS)
    ch = (s3 - h) / h * 100 if h > 0.01 else float('nan')
    print(f'  {rname:12s}: hist={h:.1f} -> SSP370={s3:.1f} days/yr ({ch:+.0f}%)')

# ============================================================
# Phase 4: Transmission resilience
# ============================================================
print('\n' + '=' * 70)
print('Phase 4: Transmission resilience')
print('=' * 70)

# 4a. Load transmission network
print('\n[4a] Loading transmission network...')
abbr_list = [EN2ABBR[p] for p in PROVINCE_ORDER]
ch_df = pd.read_excel(TRANS_FILE, sheet_name='channel', index_col=0)
cap_df = pd.read_excel(TRANS_FILE, sheet_name='channel_capacity', index_col=0)
cap_df = cap_df.loc[cap_df.index.notna()]
transport_matrix = ch_df.loc[abbr_list, abbr_list].values.astype(float)
channel_capacity = cap_df.loc[abbr_list, abbr_list].values.astype(float)
print(f'  Active channels: {int((transport_matrix > 0).sum())}')

# 4b. Hydro CF
print('\n[4b] Loading hydro monthly CF...')
cf_hydro = np.load(HYDRO_CF)['cf_mean']  # (12, 31)
print(f'  Hydro CF mean: {cf_hydro.mean():.4f}')

# 4c. Capacity data (6 scenarios: 3 capacity x 2 SSP)
print('\n[4c] Loading capacity data (6 scenarios)...')
CAP_SCENARIOS = ['NDC', 'GM2.0', 'CN2050']
CAP_LABELS = {'NDC': 'NDC', 'GM2.0': 'GM2.0', 'CN2050': 'CN2050'}

cap_data = {}  # key: (cap_scen, ssp) -> dict with cap_wind, cap_solar, cap_hydro, monthly_demand
for cap_scen in CAP_SCENARIOS:
    for ssp in SSP_LIST:
        sd = np.load(DATA_DIR / f'supply_demand_2050_{cap_scen}_{ssp}.npz')
        cap_data[(cap_scen, ssp)] = {
            'cap_wind': sd['cap_wind'],
            'cap_solar': sd['cap_solar'],
            'cap_hydro': sd['cap_hydro'],
            'monthly_demand': sd['monthly_demand'],
        }
        w = sd['cap_wind'].sum()/1000
        s = sd['cap_solar'].sum()/1000
        h = sd['cap_hydro'].sum()/1000
        print(f'  {cap_scen:10s} {ssp}: wind={w:.0f}GW, solar={s:.0f}GW, hydro={h:.0f}GW')

# 4d. Compute TR for all 6 scenarios
print('\n[4d] Computing Transmission Reducibility (6 scenarios)...')

def compute_tr(period, cap_scen, ssp_for_cap):
    """Compute TR for a given climate period using a given capacity scenario."""
    cd = cap_data[(cap_scen, ssp_for_cap)]
    cw, cs, ch_cap, dem = cd['cap_wind'], cd['cap_solar'], cd['cap_hydro'], cd['monthly_demand']

    results_gcm = []
    for gi, gcm in enumerate(GCM_LIST):
        cf_w, cf_s, months = load_daily_cf(period, gcm)
        is_lslw = lslw_days[period][gi]
        n_days = cf_w.shape[0]

        helped = 0
        total = 0

        for d in range(n_days):
            m = months[d]  # 0-11
            lslw_today = is_lslw[d]
            if not lslw_today.any():
                continue

            demand_d = dem[m] / DAYS_PER_MONTH[m]  # (31,) TWh/day
            gen_w = cw * cf_w[d] * 24 / 1e6
            gen_s = cs * cf_s[d] * 24 / 1e6
            gen_h = ch_cap * cf_hydro[m] * 24 / 1e6
            surplus = gen_w + gen_s + gen_h - demand_d

            for pi in range(N_PROV):
                if not lslw_today[pi]:
                    continue
                total += 1
                if surplus[pi] >= 0:
                    helped += 1
                    continue
                # Check transmission
                avail = 0.0
                for pj in range(N_PROV):
                    if pj != pi and transport_matrix[pj, pi] > 0 and surplus[pj] > 0:
                        max_t = channel_capacity[pj, pi] * 24 / 1000 * (1 - LINE_LOSS)
                        avail += min(surplus[pj], max_t)
                if avail > 0:
                    helped += 1

        tr = helped / total if total > 0 else 0
        results_gcm.append({'tr': tr, 'total': total, 'helped': helped})
    return results_gcm

# Compute TR for all combinations
# For historical period: use each capacity scenario with ssp245 (demand doesn't change much)
# For future periods: match SSP
tr_results_6 = {}  # key: (cap_scen, period) -> list of GCM results
for cap_scen in CAP_SCENARIOS:
    # Historical (use ssp245 demand as reference)
    key = (cap_scen, 'historical')
    print(f'  {cap_scen:10s} x historical...')
    tr_results_6[key] = compute_tr('historical', cap_scen, 'ssp245')
    trs = [r['tr'] for r in tr_results_6[key]]
    print(f'    TR: {np.mean(trs):.3f} +/- {np.std(trs):.3f}')

    # Future periods
    for ssp in SSP_LIST:
        key = (cap_scen, ssp)
        print(f'  {cap_scen:10s} x {ssp}...')
        tr_results_6[key] = compute_tr(ssp, cap_scen, ssp)
        trs = [r['tr'] for r in tr_results_6[key]]
        print(f'    TR: {np.mean(trs):.3f} +/- {np.std(trs):.3f}')

# Also keep backward-compatible tr_results for Phase 6 summary (CN2050)
tr_results = {
    'historical': tr_results_6[('CN2050', 'historical')],
    'ssp245': tr_results_6[('CN2050', 'ssp245')],
    'ssp370': tr_results_6[('CN2050', 'ssp370')],
}

# 4e. Seasonal TR (6 scenarios)
print('\n[4e] Seasonal TR (6 scenarios)...')

def compute_seasonal_tr(period, cap_scen, ssp_for_cap):
    cd = cap_data[(cap_scen, ssp_for_cap)]
    cw, cs, ch_cap, dem = cd['cap_wind'], cd['cap_solar'], cd['cap_hydro'], cd['monthly_demand']

    monthly_tr = np.zeros((N_GCM, 12))
    monthly_n = np.zeros((N_GCM, 12))

    for gi, gcm in enumerate(GCM_LIST):
        cf_w, cf_s, months = load_daily_cf(period, gcm)
        is_lslw = lslw_days[period][gi]
        n_days = cf_w.shape[0]

        helped_m = np.zeros(12)
        total_m = np.zeros(12)

        for d in range(n_days):
            m = months[d]
            lslw_today = is_lslw[d]
            if not lslw_today.any():
                continue

            demand_d = dem[m] / DAYS_PER_MONTH[m]
            gen_w = cw * cf_w[d] * 24 / 1e6
            gen_s = cs * cf_s[d] * 24 / 1e6
            gen_h = ch_cap * cf_hydro[m] * 24 / 1e6
            surplus = gen_w + gen_s + gen_h - demand_d

            for pi in range(N_PROV):
                if not lslw_today[pi]:
                    continue
                total_m[m] += 1
                if surplus[pi] >= 0:
                    helped_m[m] += 1
                    continue
                avail = 0.0
                for pj in range(N_PROV):
                    if pj != pi and transport_matrix[pj, pi] > 0 and surplus[pj] > 0:
                        max_t = channel_capacity[pj, pi] * 24 / 1000 * (1 - LINE_LOSS)
                        avail += min(surplus[pj], max_t)
                if avail > 0:
                    helped_m[m] += 1

        for m in range(12):
            monthly_tr[gi, m] = helped_m[m] / total_m[m] if total_m[m] > 0 else 0
            monthly_n[gi, m] = total_m[m]

    return monthly_tr.mean(axis=0), monthly_n.mean(axis=0)

seasonal_tr_6, seasonal_n_6 = {}, {}
for cap_scen in CAP_SCENARIOS:
    key = (cap_scen, 'historical')
    seasonal_tr_6[key], seasonal_n_6[key] = compute_seasonal_tr('historical', cap_scen, 'ssp245')
    for ssp in SSP_LIST:
        key = (cap_scen, ssp)
        seasonal_tr_6[key], seasonal_n_6[key] = compute_seasonal_tr(ssp, cap_scen, ssp)

# Print seasonal TR table for all scenarios
print(f'\n  {"Month":>5s}', end='')
for cap_scen in CAP_SCENARIOS:
    for period in ['historical'] + SSP_LIST:
        print(f'  {CAP_LABELS[cap_scen]}_{period[:4]:>5s}', end='')
print()
for m in range(12):
    print(f'  {MONTH_LABELS[m]:>5s}', end='')
    for cap_scen in CAP_SCENARIOS:
        for period in ['historical'] + SSP_LIST:
            key = (cap_scen, period if period != 'historical' else 'historical')
            print(f'  {seasonal_tr_6[key][m]:10.3f}', end='')
    print()

# Backward-compatible seasonal_tr for existing plots
seasonal_tr = {p: seasonal_tr_6[('CN2050', p)] for p in ['historical'] + SSP_LIST}

# ============================================================
# Phase 5: Visualization (6 figures)
# ============================================================
print('\n' + '=' * 70)
print('Phase 5: Visualization')
print('=' * 70)

# Region sorting for heatmaps
region_order = []
region_labels = []
for rname, provs in REGIONS.items():
    start = len(region_order)
    for p in provs:
        region_order.append(PROVINCE_ORDER.index(p))
    region_labels.append((rname, start, len(region_order) - 1))

# ---- Fig A: Provincial frequency change ----
print('\n[Fig A] Provincial LSLW frequency change...')
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
for si, ssp in enumerate(SSP_LIST):
    ax = axes[si]
    delta_pct = np.zeros(N_PROV)
    for pi in range(N_PROV):
        h = freq['historical'][pi]
        delta_pct[pi] = (freq[ssp][pi] - h) / h * 100 if h > 0.1 else 0
    sort_idx = np.argsort(delta_pct)[::-1]
    colors = [plt.cm.Reds(min(1, abs(delta_pct[i]) / 200)) if delta_pct[i] > 0
              else plt.cm.Blues(min(1, abs(delta_pct[i]) / 200)) for i in sort_idx]
    y = np.arange(N_PROV)
    ax.barh(y, delta_pct[sort_idx], color=colors, height=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([PROVINCE_ORDER[i] for i in sort_idx], fontsize=8)
    ax.set_xlabel('Change in LSLW frequency (%)', fontsize=11)
    ax.set_title(f'LSLW Frequency Change ({ssp.upper()})', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    nat = (freq[ssp].mean() - freq['historical'].mean()) / freq['historical'].mean() * 100
    ax.text(0.97, 0.03, f'National: {nat:+.1f}%', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lslw_A_frequency_change.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved')

# ---- Fig B: Seasonality heatmap ----
print('\n[Fig B] Seasonality heatmap...')
fig, axes = plt.subplots(1, 3, figsize=(24, 10))
for si, period in enumerate(['historical', 'ssp245', 'ssp370']):
    ax = axes[si]
    data = monthly_freq[period][:, region_order].T
    im = ax.imshow(data, aspect='auto', cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_yticks(range(N_PROV))
    ax.set_yticklabels([PROVINCE_ORDER[i] for i in region_order], fontsize=7)
    title = 'Historical (1990-2014)' if period == 'historical' else f'{period.upper()} (2036-2060)'
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6, label='LSLW days/month')
    for rname, start, end in region_labels:
        if start > 0:
            ax.axhline(start - 0.5, color='black', linewidth=1.2)
        if si == 0:
            ax.text(-1.5, (start + end) / 2, rname, ha='right', va='center',
                    fontsize=8, fontweight='bold', style='italic')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lslw_B_seasonality_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved')

# ---- Fig C: Spatial sync curves ----
print('\n[Fig C] Spatial synchronization...')
fig, ax = plt.subplots(figsize=(10, 7))
k_range = np.arange(1, 21)
for period in ['historical'] + SSP_LIST:
    label = 'Historical' if period == 'historical' else period.upper()
    vals = sync_dist[period][1:21]
    stds = sync_dist_std[period][1:21]
    ax.semilogy(k_range, vals, 'o-', color=COLORS_SSP[period], lw=2.5, ms=6, label=label)
    ax.fill_between(k_range, np.maximum(vals - stds, 0.01), vals + stds,
                    color=COLORS_SSP_LIGHT[period], alpha=0.3)
ax.set_xlabel('Synchronized provinces (k)', fontsize=12)
ax.set_ylabel('Days/year with N_sync >= k', fontsize=12)
ax.set_title('Spatial Synchronization of LSLW Events', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1, 20)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for k_ann in [5, 10]:
    h = sync_dist['historical'][k_ann]
    s3 = sync_dist['ssp370'][k_ann]
    if h > 0.01:
        ch = (s3 - h) / h * 100
        ax.annotate(f'k={k_ann}: +{ch:.0f}%', xy=(k_ann, s3), xytext=(k_ann+1.5, s3*2),
                    fontsize=9, fontweight='bold', color=COLORS_SSP['ssp370'],
                    arrowprops=dict(arrowstyle='->', color=COLORS_SSP['ssp370'], lw=1.5))
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lslw_C_spatial_sync.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved')

# ---- Fig D: Pairwise Jaccard ----
print('\n[Fig D] Pairwise Jaccard...')
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for si, (pname, jdata) in enumerate([('Historical', jaccard['historical']),
                                      ('SSP245', jaccard['ssp245']),
                                      ('SSP370', jaccard['ssp370'])]):
    ax = axes[si]
    data_r = jdata[np.ix_(region_order, region_order)]
    im = ax.imshow(data_r, cmap='Blues', vmin=0, vmax=0.3, interpolation='nearest')
    ax.set_xticks(range(N_PROV))
    ax.set_xticklabels([PROVINCE_ORDER[i] for i in region_order], fontsize=5, rotation=90)
    ax.set_yticks(range(N_PROV))
    ax.set_yticklabels([PROVINCE_ORDER[i] for i in region_order], fontsize=5)
    ax.set_title(pname, fontsize=12, fontweight='bold')
    for rname, start, end in region_labels:
        if start > 0:
            ax.axhline(start-0.5, color='black', linewidth=0.8)
            ax.axvline(start-0.5, color='black', linewidth=0.8)
    plt.colorbar(im, ax=ax, shrink=0.6, label='Jaccard similarity')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lslw_D_pairwise_cooccurrence.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved')

# ---- Fig E: Transmission Reducibility (6 scenarios) ----
print('\n[Fig E] Transmission Reducibility (6 scenarios)...')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# E1: Grouped bar chart - 3 capacity scenarios x 3 periods
ax = axes[0]
x = np.arange(len(CAP_SCENARIOS))
width = 0.25
periods_plot = ['historical', 'ssp245', 'ssp370']
period_labels = ['Historical', 'SSP245', 'SSP370']
period_colors = [COLORS_SSP[p] for p in periods_plot]

for i, (period, plabel, pcol) in enumerate(zip(periods_plot, period_labels, period_colors)):
    tr_vals = []
    tr_errs = []
    for cap_scen in CAP_SCENARIOS:
        key = (cap_scen, period)
        trs = [r['tr'] for r in tr_results_6[key]]
        tr_vals.append(np.mean(trs))
        tr_errs.append(np.std(trs))
    bars = ax.bar(x + i * width, tr_vals, width, yerr=tr_errs, capsize=3,
                  color=pcol, alpha=0.85, edgecolor='black', linewidth=0.4, label=plabel)
    for j, (v, e) in enumerate(zip(tr_vals, tr_errs)):
        ax.text(x[j] + i * width, v + e + 0.01, f'{v:.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels([CAP_LABELS[s] for s in CAP_SCENARIOS], fontsize=11)
ax.set_ylabel('Transmission Reducibility (TR)', fontsize=11)
ax.set_title('TR by Capacity Scenario', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# E2: Seasonal TR for all 3 capacity scenarios, SSP245 + SSP370 (no historical lines)
ax = axes[1]
cap_styles = {'NDC': ('--', 's', 'NDC'), 'GM2.0': ('-.', 'D', 'GM2.0'), 'CN2050': ('-', 'o', 'CN2050')}
ssp_colors = {'ssp245': '#4393c3', 'ssp370': '#d73027'}
for cap_scen, (ls, mk, lbl) in cap_styles.items():
    for ssp in SSP_LIST:
        key = (cap_scen, ssp)
        plabel = f'{lbl} {ssp.upper()}'
        ax.plot(range(12), seasonal_tr_6[key], marker=mk, linestyle=ls,
                color=ssp_colors[ssp], lw=2, ms=5, label=plabel, alpha=0.9)
# Historical baseline (average across 3 cap scenarios)
hist_baseline = np.mean([seasonal_tr_6[(cs, 'historical')] for cs in CAP_SCENARIOS], axis=0)
ax.plot(range(12), hist_baseline, 'k:', lw=1.5, alpha=0.5, label='Hist baseline')
ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_LABELS, fontsize=9)
ax.set_ylabel('Transmission Reducibility', fontsize=11)
ax.set_title('Seasonal TR: 3 Scenarios', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.axvspan(10.5, 11.5, color='#fee0d2', alpha=0.3)
ax.axvspan(-0.5, 1.5, color='#fee0d2', alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lslw_E_transmission_reducibility.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved')

# ---- Fig F: Composite summary ----
print('\n[Fig F] Composite summary...')
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# F1: Top 15 frequency change
ax = fig.add_subplot(gs[0, 0])
delta_370 = np.array([(freq['ssp370'][pi] - freq['historical'][pi]) / freq['historical'][pi] * 100
                       if freq['historical'][pi] > 0.1 else 0 for pi in range(N_PROV)])
si = np.argsort(delta_370)[::-1][:15]
y = np.arange(len(si))
ax.barh(y, delta_370[si], color=[plt.cm.Reds(min(1, delta_370[i]/300)) for i in si], height=0.7)
ax.set_yticks(y); ax.set_yticklabels([PROVINCE_ORDER[i] for i in si], fontsize=8)
ax.set_xlabel('LSLW freq. change (%)'); ax.set_title('(a) Frequency change (SSP370)', fontsize=10, fontweight='bold', loc='left')
ax.grid(axis='x', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# F2: Sync curves
ax = fig.add_subplot(gs[0, 1])
for period in ['historical'] + SSP_LIST:
    label = 'Historical' if period == 'historical' else period.upper()
    ax.semilogy(np.arange(1,16), sync_dist[period][1:16], 'o-', color=COLORS_SSP[period], lw=2, ms=5, label=label)
ax.set_xlabel('Synchronized provinces'); ax.set_ylabel('Days/year (N>=k)')
ax.set_title('(b) Spatial synchronization', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# F3: Seasonality
ax = fig.add_subplot(gs[0, 2])
for period in ['historical'] + SSP_LIST:
    label = 'Historical' if period == 'historical' else period.upper()
    ax.plot(range(12), monthly_freq[period].mean(axis=1), 'o-', color=COLORS_SSP[period], lw=2, ms=5, label=label)
ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_LABELS, fontsize=8)
ax.set_ylabel('LSLW days/month'); ax.set_title('(c) Seasonality', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# F4: Max consecutive
ax = fig.add_subplot(gs[1, 0])
si_c = np.argsort(max_cons['ssp370'] - max_cons['historical'])[::-1][:15]
y = np.arange(len(si_c))
ax.barh(y, max_cons['historical'][si_c], height=0.35, color='#4393c3', alpha=0.7, label='Historical')
ax.barh(y+0.35, max_cons['ssp370'][si_c], height=0.35, color='#d73027', alpha=0.7, label='SSP370')
ax.set_yticks(y+0.175); ax.set_yticklabels([PROVINCE_ORDER[i] for i in si_c], fontsize=8)
ax.set_xlabel('Max consecutive LSLW days')
ax.set_title('(d) Max consecutive LSLW', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# F5: TR bars (6 scenarios)
ax = fig.add_subplot(gs[1, 1])
x = np.arange(len(CAP_SCENARIOS))
width = 0.25
periods_plot = ['historical', 'ssp245', 'ssp370']
for i, period in enumerate(periods_plot):
    tr_vals = [np.mean([r['tr'] for r in tr_results_6[(cs, period)]]) for cs in CAP_SCENARIOS]
    ax.bar(x + i * width, tr_vals, width,
           color=COLORS_SSP[period], alpha=0.85, edgecolor='black', linewidth=0.3,
           label='Hist' if period == 'historical' else period.upper())
ax.set_xticks(x + width); ax.set_xticklabels([CAP_LABELS[s] for s in CAP_SCENARIOS], fontsize=9)
ax.set_ylabel('TR'); ax.set_title('(e) TR by scenario', fontsize=10, fontweight='bold', loc='left')
ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=7)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# F6: Seasonal TR (3 scenarios × 2 SSP, no historical)
ax = fig.add_subplot(gs[1, 2])
ssp_colors_f = {'ssp245': '#4393c3', 'ssp370': '#d73027'}
for cap_scen, (ls, mk, lbl) in [('NDC', ('--', 's', 'NDC')), ('GM2.0', ('-.', 'D', 'GM2.0')), ('CN2050', ('-', 'o', 'CN2050'))]:
    for ssp in SSP_LIST:
        key = (cap_scen, ssp)
        ax.plot(range(12), seasonal_tr_6[key], marker=mk, linestyle=ls,
                color=ssp_colors_f[ssp], lw=1.5, ms=4, label=f'{lbl} {ssp.upper()}', alpha=0.9)
hist_base_f = np.mean([seasonal_tr_6[(cs, 'historical')] for cs in CAP_SCENARIOS], axis=0)
ax.plot(range(12), hist_base_f, 'k:', lw=1.2, alpha=0.5, label='Hist')
ax.axvspan(10.5, 11.5, color='#fee0d2', alpha=0.3)
ax.axvspan(-0.5, 1.5, color='#fee0d2', alpha=0.3)
ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_LABELS, fontsize=7)
ax.set_ylabel('TR'); ax.set_title('(f) Seasonal TR: 3 scenarios', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=5, ncol=2); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.savefig(FIG_DIR / 'fig_lslw_F_composite.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved')

# ============================================================
# Phase 6: Save data + Excel
# ============================================================
print('\n' + '=' * 70)
print('Phase 6: Save results')
print('=' * 70)

# Build TR arrays for npz save
tr_6scen_mean = {}
tr_6scen_seasonal = {}
for cap_scen in CAP_SCENARIOS:
    for period in ['historical'] + SSP_LIST:
        key = (cap_scen, period)
        tr_6scen_mean[f'tr_{CAP_LABELS[cap_scen]}_{period}'] = np.mean([r['tr'] for r in tr_results_6[key]])
        tr_6scen_seasonal[f'seasonal_tr_{CAP_LABELS[cap_scen]}_{period}'] = seasonal_tr_6[key]

np.savez_compressed(DATA_DIR / 'lslw_results.npz',
    wind_p10_daily=wind_p10_daily, solar_p10_daily=solar_p10_daily,
    freq_historical=freq['historical'], freq_ssp245=freq['ssp245'], freq_ssp370=freq['ssp370'],
    freq_std_historical=freq_std['historical'], freq_std_ssp245=freq_std['ssp245'], freq_std_ssp370=freq_std['ssp370'],
    max_cons_historical=max_cons['historical'], max_cons_ssp245=max_cons['ssp245'], max_cons_ssp370=max_cons['ssp370'],
    monthly_freq_historical=monthly_freq['historical'], monthly_freq_ssp245=monthly_freq['ssp245'], monthly_freq_ssp370=monthly_freq['ssp370'],
    sync_dist_historical=sync_dist['historical'], sync_dist_ssp245=sync_dist['ssp245'], sync_dist_ssp370=sync_dist['ssp370'],
    jaccard_historical=jaccard['historical'], jaccard_ssp245=jaccard['ssp245'], jaccard_ssp370=jaccard['ssp370'],
    provinces=PROVINCE_ORDER,
    **tr_6scen_mean, **tr_6scen_seasonal)
print(f'  Saved: lslw_results.npz')

# Excel
with pd.ExcelWriter(EXCEL_DIR / 'lslw_results.xlsx', engine='openpyxl') as writer:
    # Provincial frequency
    rows = []
    for pi, prov in enumerate(PROVINCE_ORDER):
        h, s2, s3 = freq['historical'][pi], freq['ssp245'][pi], freq['ssp370'][pi]
        rows.append({'Province': prov, 'Hist_days_yr': h, 'SSP245_days_yr': s2, 'SSP370_days_yr': s3,
                     'Change_SSP245_pct': (s2-h)/h*100 if h>0.1 else 0,
                     'Change_SSP370_pct': (s3-h)/h*100 if h>0.1 else 0,
                     'MaxCons_Hist': max_cons['historical'][pi],
                     'MaxCons_SSP370': max_cons['ssp370'][pi]})
    pd.DataFrame(rows).to_excel(writer, sheet_name='Provincial_Frequency', index=False)

    # Monthly seasonality
    rows = []
    for m in range(12):
        row = {'Month': MONTH_LABELS[m]}
        for p in ['historical', 'ssp245', 'ssp370']:
            row[f'{p}_national'] = monthly_freq[p][m].mean()
        rows.append(row)
    pd.DataFrame(rows).to_excel(writer, sheet_name='Monthly_Seasonality', index=False)

    # Spatial sync
    rows = []
    for k in range(1, N_PROV+1):
        h, s2, s3 = sync_dist['historical'][k], sync_dist['ssp245'][k], sync_dist['ssp370'][k]
        rows.append({'k': k, 'Hist': h, 'SSP245': s2, 'SSP370': s3,
                     'Change_SSP370_pct': (s3-h)/h*100 if h>0.01 else None})
    pd.DataFrame(rows).to_excel(writer, sheet_name='Spatial_Sync', index=False)

    # TR (6 scenarios)
    rows = []
    for cap_scen in CAP_SCENARIOS:
        for period in ['historical'] + SSP_LIST:
            key = (cap_scen, period)
            trs = [r['tr'] for r in tr_results_6[key]]
            rows.append({'Capacity': CAP_LABELS[cap_scen], 'Climate': period,
                         'TR_mean': np.mean(trs), 'TR_std': np.std(trs),
                         'Total_LSLW': np.mean([r['total'] for r in tr_results_6[key]])})
    pd.DataFrame(rows).to_excel(writer, sheet_name='TR_Summary', index=False)

    # Seasonal TR (6 scenarios)
    rows = []
    for m in range(12):
        row = {'Month': MONTH_LABELS[m]}
        for cap_scen in CAP_SCENARIOS:
            for period in ['historical'] + SSP_LIST:
                key = (cap_scen, period)
                col = f'TR_{CAP_LABELS[cap_scen]}_{period}'
                row[col] = seasonal_tr_6[key][m]
        rows.append(row)
    pd.DataFrame(rows).to_excel(writer, sheet_name='Seasonal_TR', index=False)

    # Summary
    rows = [
        {'Metric': 'LSLW freq (days/yr)', 'Hist': freq['historical'].mean(),
         'SSP245': freq['ssp245'].mean(), 'SSP370': freq['ssp370'].mean(),
         'Change_SSP370': f'{d370:+.1f}%'},
        {'Metric': 'Sync >=5 prov (days/yr)', 'Hist': sync_dist['historical'][5],
         'SSP245': sync_dist['ssp245'][5], 'SSP370': sync_dist['ssp370'][5]},
    ]
    for cap_scen in CAP_SCENARIOS:
        for ssp in SSP_LIST:
            key = (cap_scen, ssp)
            tr_val = np.mean([r['tr'] for r in tr_results_6[key]])
            rows.append({'Metric': f'TR_{CAP_LABELS[cap_scen]}_{ssp}',
                         'Hist': np.mean([r['tr'] for r in tr_results_6[(cap_scen, 'historical')]]),
                         ssp.upper(): tr_val})
    pd.DataFrame(rows).to_excel(writer, sheet_name='Summary', index=False)

print(f'  Saved: lslw_results.xlsx')

# ============================================================
# Final
# ============================================================
print('\n' + '=' * 70)
print('Step3D v3 COMPLETE!')
print('=' * 70)
print(f'  LSLW freq change: SSP245 {d245:+.1f}%, SSP370 {d370:+.1f}%')
h5, s3_5 = sync_dist['historical'][5], sync_dist['ssp370'][5]
sync_ch = (s3_5-h5)/h5*100 if h5>0.01 else float('nan')
print(f'  Sync >=5 prov: {h5:.1f} -> {s3_5:.1f} days/yr ({sync_ch:+.0f}%)')
print(f'\n  TR by scenario (SSP370):')
for cap_scen in CAP_SCENARIOS:
    tr_h = np.mean([r['tr'] for r in tr_results_6[(cap_scen, 'historical')]])
    tr_s3 = np.mean([r['tr'] for r in tr_results_6[(cap_scen, 'ssp370')]])
    print(f'    {CAP_LABELS[cap_scen]:8s}: {tr_h:.3f} -> {tr_s3:.3f}')
print(f'\nOutputs: {FIG_DIR}/fig_lslw_*.png, {EXCEL_DIR}/lslw_results.xlsx')
print('=' * 70)
