# -*- coding: utf-8 -*-
"""
step3A3_gridlevel_QM.py - Grid-level QDM bias correction for CMIP6 (v5)

Method: Quantile Delta Mapping (Cannon et al. 2015) on METEOROLOGICAL VARIABLES
  - QDM applied to wind speed (ws100) and solar radiation (rsds) — NOT capacity factors
  - This matches the standard approach in Nature papers (Wang NSR 2025 via ISIMIP,
    Zheng Nature 2025): bias-correct met variables first, then compute CF
  - ERA5 hourly -> daily mean met vars -> quantiles as QDM reference
  - CMIP6 daily met vars -> regridded to ERA5 grid -> QDM corrected
  - Corrected met vars -> power curve / solar formula -> raw CF
  - Raw CF scaled by constant per-province factor (Zhuo et al. 2022) -> actual CF level
  - Multiplicative QDM preserves climate change signal

Key insight: Applying nonlinear power curve to daily-mean wind speed gives
  CF(E[ws]) << E[CF(ws)] due to Jensen's inequality (~1.4x underestimate).
  This is handled by the per-province scaling factor, not by the QDM step.
  The scaling factor only adjusts magnitude; seasonal shape comes from real climate data.

Changes from v4:
  - QDM on met variables (ws100, rsds) instead of CF
  - 8 GCMs (was 5)
  - Glob-based file discovery for CMIP6 (handles variant/grid differences)
  - Per-province constant scaling factor from Zhuo et al. 2022

Output: province-averaged daily CF (same format as step3D expects)
  - CMIP6_daily_CF_corrected_{period}_{gcm}.npz
  - Keys: cf_wind, cf_solar, months, provinces
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import xarray as xr
import scipy.io as sio
from pathlib import Path
from path_config import (
    BASE_DIR, DATA_DIR, FIG_DIR, OUT_DIR, EXCEL_DIR, SOURCE_DATA_DIR,
    INPUT_TABLES_DIR, FIGURES_DIR, CODE_DIR, ERA5_HOURLY, ERA5_DAILY_RSDS,
    CMIP6_DIR, STEP1_DATA, ZHUO_DIR, HYDRO_PATH, TX_PATH, MOESM4_PATH,
    LOAD_FILE, GRID_COORDS, GRID_MAP, ensure_output_dirs,
)
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
# Raw ERA5 data are external; configure with LSLW_ERA5_HOURLY if rerunning this step.
# Raw ERA5 daily rsds are external; configure with LSLW_ERA5_DAILY_RSDS if rerunning this step.
# Raw CMIP6 data are external; configure with LSLW_CMIP6_DIR if rerunning this step.
# Step-1 grid files are external; configure with LSLW_STEP1_DATA if rerunning this step.
DATA_DIR = DATA_DIR
CACHE_DIR    = DATA_DIR / 'gridlevel_qdm_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

GCM_LIST = [
    'ACCESS-CM2', 'BCC-CSM2-MR', 'CNRM-CM6-1', 'EC-Earth3',
    'GFDL-ESM4', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM'
]
HIST_YEARS = list(range(1990, 2015))
FUTURE_YEARS = list(range(2036, 2061))

N_QUANTILES = 101
Q_LEVELS = np.linspace(0, 100, N_QUANTILES)   # for np.percentile
Q_FRAC = np.linspace(0, 1, N_QUANTILES)       # for np.interp

HEIGHT_FACTOR = (100.0 / 10.0) ** (1.0 / 7.0)

# Scaling factor clip bounds
SCALE_WIND_MAX = 50.0
SCALE_SOLAR_MAX = 20.0

print('=' * 70)
print('Step3A3 v5: Grid-Level QDM on Meteorological Variables')
print('  QDM on ws100 + rsds (standard Nature approach)')
print('  Then CF from corrected met vars + Zhuo scaling')
print(f'  {len(GCM_LIST)} GCMs, hist={HIST_YEARS[0]}-{HIST_YEARS[-1]}, '
      f'future={FUTURE_YEARS[0]}-{FUTURE_YEARS[-1]}')
print('=' * 70)

# ============================================================
# CF formulas (applied AFTER QDM correction of met variables)
# ============================================================

def wind_power_curve(ws):
    """IEC Class II turbine power curve. ws in m/s."""
    cf = np.zeros_like(ws, dtype=np.float32)
    m1 = (ws >= 3) & (ws < 13)
    cf[m1] = ((ws[m1] - 3) / 10.0) ** 3
    cf[(ws >= 13) & (ws <= 25)] = 1.0
    return cf


def solar_cf_simple(irradiance_Wm2):
    """Simplified solar CF = G/1000 (STC). No temperature correction."""
    return np.clip(irradiance_Wm2 / 1000.0, 0, 1).astype(np.float32)


# ============================================================
# Phase 0: Load province mask + ERA5 grid + regrid weights
# ============================================================
print('\n' + '=' * 70)
print('Phase 0: Province mask & grids')
print('=' * 70)

era5_sample = xr.open_dataset(ERA5_HOURLY / 'u100' / 'ERA5_HOURLY_u100_1990.nc')
era5_lat = era5_sample['latitude'].values.astype(np.float64)
era5_lon = era5_sample['longitude'].values.astype(np.float64)
N_LAT, N_LON = len(era5_lat), len(era5_lon)
era5_sample.close()
print(f'  ERA5 hourly grid: {N_LAT} x {N_LON}')

province_mask = np.load(STEP1_DATA / 'grid_province_map.npy')
assert province_mask.shape == (N_LAT, N_LON)
print(f'  Province mask loaded')

prov_indices = {}
for pi in range(N_PROV):
    prov_indices[pi] = np.where(province_mask == pi)

valid_mask = province_mask >= 0
valid_li, valid_loi = np.where(valid_mask)
N_VALID = len(valid_li)
print(f'  Valid grid cells: {N_VALID}')

cmip_sample_path = list((CMIP6_DIR / 'ACCESS-CM2' / 'historical' / 'sfcWind').glob('*.nc'))[0]
cmip_sample = xr.open_dataset(cmip_sample_path)
cmip_lat = cmip_sample['lat'].values.astype(np.float64)
cmip_lon = cmip_sample['lon'].values.astype(np.float64)
cmip_sample.close()
print(f'  CMIP6 grid: {len(cmip_lat)} x {len(cmip_lon)}')

print('  Building regrid weights...')


def build_regrid_weights(src_lat, src_lon, tgt_lat, tgt_lon):
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        lat_flip = True
    else:
        lat_flip = False

    n_slat, n_slon = len(src_lat), len(src_lon)
    n_tgt = len(tgt_lat) * len(tgt_lon)
    indices = np.zeros((n_tgt, 4), dtype=np.int32)
    weights = np.zeros((n_tgt, 4), dtype=np.float32)

    for ti in range(len(tgt_lat)):
        for tj in range(len(tgt_lon)):
            fid = ti * len(tgt_lon) + tj
            la, lo = tgt_lat[ti], tgt_lon[tj]
            i0 = max(0, min(np.searchsorted(src_lat, la) - 1, n_slat - 2))
            j0 = max(0, min(np.searchsorted(src_lon, lo) - 1, n_slon - 2))
            i1, j1 = i0 + 1, j0 + 1
            dy = np.clip((la - src_lat[i0]) / (src_lat[i1] - src_lat[i0]), 0, 1)
            dx = np.clip((lo - src_lon[j0]) / (src_lon[j1] - src_lon[j0]), 0, 1)
            w = [(1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx]
            if lat_flip:
                i0f, i1f = n_slat-1-i0, n_slat-1-i1
                indices[fid] = [i0f*n_slon+j0, i0f*n_slon+j1, i1f*n_slon+j0, i1f*n_slon+j1]
            else:
                indices[fid] = [i0*n_slon+j0, i0*n_slon+j1, i1*n_slon+j0, i1*n_slon+j1]
            weights[fid] = w
    return indices, weights


regrid_idx, regrid_wt = build_regrid_weights(cmip_lat, cmip_lon, era5_lat, era5_lon)
print(f'  Regrid weights ready')


def regrid_fast(data_3d, n_tlat, n_tlon):
    """Regrid CMIP6 -> ERA5 grid via bilinear interpolation."""
    n_days = data_3d.shape[0]
    flat = data_3d.reshape(n_days, -1)
    corners = flat[:, regrid_idx]
    result = (corners * regrid_wt[np.newaxis, :, :]).sum(axis=2)
    return result.reshape(n_days, n_tlat, n_tlon).astype(np.float32)


# ============================================================
# Helper: load CMIP6 met variables (NOT CF)
# ============================================================

def load_cmip6_met(gcm, scenario, year):
    """Load CMIP6 daily sfcWind + rsds, regrid, return ws100 and rsds on ERA5 grid.

    Returns:
        ws100: (n_days, N_LAT, N_LON) 100m wind speed in m/s
        rsds:  (n_days, N_LAT, N_LON) surface downward shortwave radiation in W/m2
        months: (n_days,) month index 0-11
    """
    base = CMIP6_DIR / gcm / scenario

    # Glob for files (handles different variant/grid labels across GCMs)
    wind_files = sorted((base / 'sfcWind').glob(f'sfcWind_day_{gcm}_{scenario}_*_{year}.nc'))
    rsds_files = sorted((base / 'rsds').glob(f'rsds_day_{gcm}_{scenario}_*_{year}.nc'))

    if not wind_files or not rsds_files:
        return None, None, None

    ds_w = xr.open_dataset(wind_files[0])
    ds_r = xr.open_dataset(rsds_files[0])

    sfcWind = ds_w['sfcWind'].values.astype(np.float32)
    rsds_raw = ds_r['rsds'].values.astype(np.float32)
    times = ds_w['time'].values
    months = np.array([np.datetime64(t, 'M').astype(int) % 12 for t in times])

    ds_w.close()
    ds_r.close()

    # Regrid to ERA5 grid
    sfcWind_r = regrid_fast(sfcWind, N_LAT, N_LON)
    rsds_r = regrid_fast(rsds_raw, N_LAT, N_LON)

    # Convert 10m -> 100m wind speed (power law)
    ws100 = sfcWind_r * HEIGHT_FACTOR

    # Clip rsds to physical range
    rsds_r = np.clip(rsds_r, 0, 1500)

    return ws100, rsds_r, months


# ============================================================
# Phase 1: ERA5 met variable quantiles (1990-2014)
# ============================================================
print('\n' + '=' * 70)
print('Phase 1: ERA5 met variable quantiles (1990-2014)')
print('  ws100 from u100/v100 (hourly), rsds from daily files (ERA5-Land)')
print('=' * 70)

era5_ws_q_path = CACHE_DIR / 'era5_ws100_quantiles.npz'
era5_ssrd_q_path = CACHE_DIR / 'era5_rsds_quantiles_v2.npz'  # v2: from daily files, not broken hourly

# Also compute ERA5 raw CF for scaling factor
era5_raw_cf_path = CACHE_DIR / 'era5_raw_cf_for_scaling_v2.npz'

# --- Wind speed quantiles (from hourly u100/v100 — these are correct) ---
if era5_ws_q_path.exists():
    print('  Loading cached ERA5 ws100 quantiles...')
    era5_ws_q = np.load(era5_ws_q_path)['q']        # (12, 157, 261, 101)
    print(f'  ws100 quantiles shape: {era5_ws_q.shape}')
else:
    print('  Computing ERA5 ws100 quantiles from hourly data (25 years)...')
    t0 = time.time()
    ws_by_month = {m: [] for m in range(12)}

    for yi, year in enumerate(HIST_YEARS):
        yt0 = time.time()
        ds_u = xr.open_dataset(ERA5_HOURLY / 'u100' / f'ERA5_HOURLY_u100_{year}.nc')
        ds_v = xr.open_dataset(ERA5_HOURLY / 'v100' / f'ERA5_HOURLY_v100_{year}.nc')
        u100 = ds_u['u100'].values.astype(np.float32)
        v100 = ds_v['v100'].values.astype(np.float32)
        times = ds_u['valid_time'].values
        ds_u.close(); ds_v.close()

        ws100_h = np.sqrt(u100**2 + v100**2)
        del u100, v100
        n_hours = ws100_h.shape[0]
        n_days = n_hours // 24
        ws100_d = ws100_h[:n_days*24].reshape(n_days, 24, N_LAT, N_LON).mean(axis=1)
        del ws100_h

        day_times = times[:n_days*24:24]
        months_arr = np.array([np.datetime64(t, 'M').astype(int) % 12 for t in day_times])
        for m in range(12):
            mm = months_arr == m
            if mm.sum() > 0:
                ws_by_month[m].append(ws100_d[mm])

        if yi % 5 == 0:
            prov_ws = np.nanmean(ws100_d[:, valid_li, valid_loi], axis=1).mean()
            print(f'    {year}: ws100={prov_ws:.2f} m/s ({time.time()-yt0:.0f}s)')

    print(f'  Computing ws100 quantiles ({time.time()-t0:.0f}s)...')
    era5_ws_q = np.zeros((12, N_LAT, N_LON, N_QUANTILES), dtype=np.float32)
    for m in range(12):
        w_all = np.concatenate(ws_by_month[m], axis=0)
        era5_ws_q[m] = np.nanpercentile(w_all, Q_LEVELS, axis=0).transpose(1, 2, 0)
        mname = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]
        print(f'    {mname}: {w_all.shape[0]} days, '
              f'ws100 median={np.nanmedian(w_all[:, valid_li, valid_loi]):.2f} m/s')
    del ws_by_month
    np.savez_compressed(era5_ws_q_path, q=era5_ws_q)
    print(f'  Saved ERA5 ws100 quantiles ({time.time()-t0:.0f}s)')

# --- Radiation quantiles (from daily rsds files — ERA5-Land, properly processed) ---
# NOTE: ERA5 hourly ssrd files have a data processing issue (only morning hours have
# nonzero values, afternoon is zero). Use pre-processed daily rsds files instead.
# These files contain sum(24h W/m2) — divide by 24 to get daily mean W/m2.
if era5_ssrd_q_path.exists():
    print('  Loading cached ERA5 rsds quantiles (v2, from daily files)...')
    era5_ssrd_q = np.load(era5_ssrd_q_path)['q']     # (12, N_LAT, N_LON, 101)
    print(f'  rsds quantiles shape: {era5_ssrd_q.shape}')
else:
    print('  Computing ERA5 rsds quantiles from daily files (25 years)...')
    t0 = time.time()

    # Daily rsds files have a slightly different grid — build regrid weights
    ds_rsds_sample = xr.open_dataset(ERA5_DAILY_RSDS / 'rsds_daily_china_025deg_1990.nc')
    rsds_lat = ds_rsds_sample['lat'].values.astype(np.float64)
    rsds_lon = ds_rsds_sample['lon'].values.astype(np.float64)
    ds_rsds_sample.close()
    print(f'  Daily rsds grid: {len(rsds_lat)} x {len(rsds_lon)}')
    print(f'  Target ERA5 grid: {N_LAT} x {N_LON}')

    # Build lat/lon index mapping: for each ERA5 grid cell, find nearest daily rsds cell
    # Both are 0.25 degree, so nearest-neighbor is sufficient
    rsds_lat_idx = np.array([np.argmin(np.abs(rsds_lat - la)) for la in era5_lat])
    rsds_lon_idx = np.array([np.argmin(np.abs(rsds_lon - lo)) for lo in era5_lon])

    # Check coverage: ERA5 grid cells outside daily rsds grid get NaN
    lat_in_range = (era5_lat >= rsds_lat.min() - 0.15) & (era5_lat <= rsds_lat.max() + 0.15)
    lon_in_range = (era5_lon >= rsds_lon.min() - 0.15) & (era5_lon <= rsds_lon.max() + 0.15)
    coverage = lat_in_range[:, None] & lon_in_range[None, :]
    n_covered = coverage.sum()
    n_valid_covered = (coverage & valid_mask).sum()
    print(f'  ERA5 cells covered by daily rsds: {n_covered}/{N_LAT*N_LON} '
          f'(valid provinces: {n_valid_covered}/{N_VALID})')

    rsds_by_month = {m: [] for m in range(12)}
    for yi, year in enumerate(HIST_YEARS):
        yt0 = time.time()
        rsds_file = ERA5_DAILY_RSDS / f'rsds_daily_china_025deg_{year}.nc'
        ds_r = xr.open_dataset(rsds_file)
        rsds_raw = ds_r['rsds'].values.astype(np.float32) / 24.0  # sum(W/m2) -> daily mean W/m2
        times_r = ds_r['time'].values
        ds_r.close()

        # Map to ERA5 grid using nearest-neighbor
        rsds_on_era5 = np.full((rsds_raw.shape[0], N_LAT, N_LON), np.nan, dtype=np.float32)
        for i in range(N_LAT):
            if not lat_in_range[i]:
                continue
            for j in range(N_LON):
                if not lon_in_range[j]:
                    continue
                rsds_on_era5[:, i, j] = rsds_raw[:, rsds_lat_idx[i], rsds_lon_idx[j]]

        months_arr = np.array([np.datetime64(t, 'M').astype(int) % 12 for t in times_r])
        for m in range(12):
            mm = months_arr == m
            if mm.sum() > 0:
                rsds_by_month[m].append(rsds_on_era5[mm])

        if yi % 5 == 0:
            prov_rsds = np.nanmean(rsds_on_era5[:, valid_li, valid_loi], axis=1).mean()
            print(f'    {year}: rsds={prov_rsds:.1f} W/m2 ({time.time()-yt0:.0f}s)')

    print(f'  All years loaded ({time.time()-t0:.0f}s). Computing rsds quantiles...')
    era5_ssrd_q = np.zeros((12, N_LAT, N_LON, N_QUANTILES), dtype=np.float32)
    for m in range(12):
        s_all = np.concatenate(rsds_by_month[m], axis=0)
        era5_ssrd_q[m] = np.nanpercentile(s_all, Q_LEVELS, axis=0).transpose(1, 2, 0)
        mname = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]
        print(f'    {mname}: {s_all.shape[0]} days, '
              f'rsds median={np.nanmedian(s_all[:, valid_li, valid_loi]):.1f} W/m2')
    del rsds_by_month
    np.savez_compressed(era5_ssrd_q_path, q=era5_ssrd_q)
    print(f'  Saved ERA5 rsds quantiles v2 ({time.time()-t0:.0f}s)')

# ---- Compute ERA5 raw CF for scaling factor ----
if era5_raw_cf_path.exists():
    print('  Loading cached ERA5 raw CF for scaling (v2)...')
    _d = np.load(era5_raw_cf_path)
    era5_raw_wind_cf = _d['wind_cf']    # (31,) annual mean
    era5_raw_solar_cf = _d['solar_cf']  # (31,) annual mean
else:
    print('  Computing ERA5 raw CF from quantile distribution (for scaling factor)...')
    era5_raw_wind_cf_monthly = np.zeros((12, N_PROV), dtype=np.float32)
    era5_raw_solar_cf_monthly = np.zeros((12, N_PROV), dtype=np.float32)

    for m in range(12):
        for pi in range(N_PROV):
            idx = prov_indices[pi]
            if len(idx[0]) == 0:
                continue
            # Approximate E[CF(ws)] from quantile distribution
            ws_q_prov = era5_ws_q[m, idx[0], idx[1], :]  # (n_cells, 101)
            cf_q = wind_power_curve(ws_q_prov)
            era5_raw_wind_cf_monthly[m, pi] = np.nanmean(cf_q)

            ssrd_q_prov = era5_ssrd_q[m, idx[0], idx[1], :]
            cf_s_q = solar_cf_simple(ssrd_q_prov)
            era5_raw_solar_cf_monthly[m, pi] = np.nanmean(cf_s_q)

    era5_raw_wind_cf = era5_raw_wind_cf_monthly.mean(axis=0)    # (31,)
    era5_raw_solar_cf = era5_raw_solar_cf_monthly.mean(axis=0)  # (31,)

    np.savez_compressed(era5_raw_cf_path,
                        wind_cf=era5_raw_wind_cf, solar_cf=era5_raw_solar_cf,
                        wind_cf_monthly=era5_raw_wind_cf_monthly,
                        solar_cf_monthly=era5_raw_solar_cf_monthly)
    print('  Saved ERA5 raw CF for scaling')

print(f'\n  ERA5 raw CF (from daily-mean met vars):')
print(f'    Wind: national mean = {era5_raw_wind_cf.mean():.4f}')
print(f'    Solar: national mean = {era5_raw_solar_cf.mean():.4f}')

# ============================================================
# Phase 1B: Compute per-province scaling factor
# ============================================================
print('\n' + '=' * 70)
print('Phase 1B: Per-province scaling factor (Zhuo et al. 2022)')
print('  scale = Zhuo_annual_CF / ERA5_raw_annual_CF')
print('  Only adjusts magnitude; seasonal shape from real climate data')
print('=' * 70)

# Load Zhuo annual CF
mat_wind = sio.loadmat(ZHUO_DIR / 'Onshore_wind_data_8760.mat')
wind_key = [k for k in mat_wind.keys() if not k.startswith('_')][0]
zhuo_wind_annual = mat_wind[wind_key].transpose(1, 0, 2).reshape(8760, 31).astype(np.float32).mean(axis=0)

mat_solar = sio.loadmat(ZHUO_DIR / 'Utility_PV_data_8760.mat')
solar_key = [k for k in mat_solar.keys() if not k.startswith('_')][0]
zhuo_solar_annual = mat_solar[solar_key].transpose(1, 0, 2).reshape(8760, 31).astype(np.float32).mean(axis=0)

scale_wind = zhuo_wind_annual / np.maximum(era5_raw_wind_cf, 1e-6)
scale_solar = zhuo_solar_annual / np.maximum(era5_raw_solar_cf, 1e-6)

# Clip extremes (Chongqing, Sichuan have negligible wind — scale would be 100x+)
scale_wind = np.clip(scale_wind, 0.5, SCALE_WIND_MAX)
scale_solar = np.clip(scale_solar, 0.5, SCALE_SOLAR_MAX)

print(f'\n  Scaling factors:')
print(f'  {"Province":18s} {"Wind scale":>10s} {"Solar scale":>12s}')
for i, prov in enumerate(PROVINCE_ORDER):
    flag_w = ' *' if scale_wind[i] >= SCALE_WIND_MAX * 0.95 else ''
    print(f'  {prov:18s} {scale_wind[i]:10.1f}{flag_w} {scale_solar[i]:12.1f}')
print(f'\n  * = clipped to max {SCALE_WIND_MAX}')
print(f'  National mean: wind={scale_wind.mean():.1f}x, solar={scale_solar.mean():.1f}x')


# ============================================================
# Phase 2: CMIP6 historical met variable quantiles (1990-2014)
# ============================================================
print('\n' + '=' * 70)
print('Phase 2: CMIP6 historical met variable quantiles (1990-2014)')
print('=' * 70)

cmip_hist_q = {}

for gcm in GCM_LIST:
    cache_path = CACHE_DIR / f'cmip6_hist_metq_{gcm}.npz'
    if cache_path.exists():
        print(f'  {gcm}: cached')
        d = np.load(cache_path)
        cmip_hist_q[gcm] = {'ws100': d['ws100_q'], 'rsds': d['rsds_q']}
        continue

    print(f'  {gcm}: loading 25yr historical...')
    t0 = time.time()

    ws_by_month = {m: [] for m in range(12)}
    rsds_by_month = {m: [] for m in range(12)}

    for year in HIST_YEARS:
        ws100, rsds, months = load_cmip6_met(gcm, 'historical', year)
        if ws100 is None:
            print(f'    {year}: skipped (files missing)')
            continue
        for m in range(12):
            mm = months == m
            if mm.sum() > 0:
                ws_by_month[m].append(ws100[mm])
                rsds_by_month[m].append(rsds[mm])
        if (year - 1990) % 5 == 0:
            print(f'    {year}...')

    ws_q = np.zeros((12, N_LAT, N_LON, N_QUANTILES), dtype=np.float32)
    rsds_q = np.zeros((12, N_LAT, N_LON, N_QUANTILES), dtype=np.float32)
    for m in range(12):
        w = np.concatenate(ws_by_month[m], axis=0)
        s = np.concatenate(rsds_by_month[m], axis=0)
        ws_q[m] = np.nanpercentile(w, Q_LEVELS, axis=0).transpose(1, 2, 0)
        rsds_q[m] = np.nanpercentile(s, Q_LEVELS, axis=0).transpose(1, 2, 0)

    cmip_hist_q[gcm] = {'ws100': ws_q, 'rsds': rsds_q}
    np.savez_compressed(cache_path, ws100_q=ws_q, rsds_q=rsds_q)
    print(f'    Done ({time.time()-t0:.0f}s)')
    del ws_by_month, rsds_by_month

# ============================================================
# Phase 2B: CMIP6 future met variable quantiles (2036-2060)
# ============================================================
print('\n' + '=' * 70)
print('Phase 2B: CMIP6 future met variable quantiles (2036-2060)')
print('=' * 70)

cmip_fut_q = {}

for gcm in GCM_LIST:
    for ssp in ['ssp245', 'ssp370']:
        key = f'{gcm}_{ssp}'
        cache_path = CACHE_DIR / f'cmip6_fut_metq_{key}.npz'
        if cache_path.exists():
            print(f'  {key}: cached')
            d = np.load(cache_path)
            cmip_fut_q[key] = {'ws100': d['ws100_q'], 'rsds': d['rsds_q']}
            continue

        print(f'  {key}: loading 25yr future...')
        t0 = time.time()

        ws_by_month = {m: [] for m in range(12)}
        rsds_by_month = {m: [] for m in range(12)}

        for year in FUTURE_YEARS:
            ws100, rsds, months = load_cmip6_met(gcm, ssp, year)
            if ws100 is None:
                print(f'    {year}: skipped (files missing)')
                continue
            for m in range(12):
                mm = months == m
                if mm.sum() > 0:
                    ws_by_month[m].append(ws100[mm])
                    rsds_by_month[m].append(rsds[mm])
            if (year - 2036) % 5 == 0:
                print(f'    {year}...')

        ws_q = np.zeros((12, N_LAT, N_LON, N_QUANTILES), dtype=np.float32)
        rsds_q = np.zeros((12, N_LAT, N_LON, N_QUANTILES), dtype=np.float32)
        for m in range(12):
            w = np.concatenate(ws_by_month[m], axis=0)
            s = np.concatenate(rsds_by_month[m], axis=0)
            ws_q[m] = np.nanpercentile(w, Q_LEVELS, axis=0).transpose(1, 2, 0)
            rsds_q[m] = np.nanpercentile(s, Q_LEVELS, axis=0).transpose(1, 2, 0)

        cmip_fut_q[key] = {'ws100': ws_q, 'rsds': rsds_q}
        np.savez_compressed(cache_path, ws100_q=ws_q, rsds_q=rsds_q)
        print(f'    Done ({time.time()-t0:.0f}s)')
        del ws_by_month, rsds_by_month

# ============================================================
# Phase 3: QDM on met variables -> CF -> scale -> province avg
# ============================================================
print('\n' + '=' * 70)
print('Phase 3: QDM correction on met vars -> CF -> province average')
print('  QDM(ws100) -> power_curve -> raw_CF * scale_wind')
print('  QDM(rsds)  -> G/1000      -> raw_CF * scale_solar')
print('=' * 70)


def vectorized_qdm(vals_2d, hist_q_2d, fut_q_2d, obs_q_2d, clip_max=50.0):
    """Quantile Delta Mapping (Cannon et al. 2015), multiplicative.

    vals_2d:   (n_days, n_pts) raw CMIP6 values
    hist_q_2d: (n_pts, 101) CMIP6 historical quantiles
    fut_q_2d:  (n_pts, 101) CMIP6 future quantiles (=hist_q for historical)
    obs_q_2d:  (n_pts, 101) ERA5 observation quantiles
    clip_max:  upper bound for corrected values (50 for ws m/s, 1500 for rsds W/m2)
    Returns:   (n_days, n_pts) QDM-corrected values
    """
    n_days, n_pts = vals_2d.shape
    n_q = hist_q_2d.shape[1]  # 101

    chunk_size = 3000
    result = np.zeros_like(vals_2d)

    for c0 in range(0, n_pts, chunk_size):
        c1 = min(c0 + chunk_size, n_pts)
        v = vals_2d[:, c0:c1]
        hq = hist_q_2d[c0:c1, :]
        fq = fut_q_2d[c0:c1, :]
        oq = obs_q_2d[c0:c1, :]
        nc = c1 - c0

        # Step 1: val -> CDF position tau using hist_q
        ge = v[:, :, np.newaxis] >= hq[np.newaxis, :, :]  # (n_days, nc, 101)
        idx = ge.sum(axis=2) - 1
        idx = np.clip(idx, 0, n_q - 2)
        del ge

        pts = np.arange(nc)
        lower_h = hq[pts[np.newaxis, :], idx]
        upper_h = hq[pts[np.newaxis, :], idx + 1]
        denom = upper_h - lower_h
        denom = np.where(denom > 0, denom, 1.0)
        frac = np.clip((v - lower_h) / denom, 0, 1)

        tau = Q_FRAC[idx] + frac * (Q_FRAC[idx + 1] - Q_FRAC[idx])

        # Step 2: x_QM = F_obs^{-1}(tau)
        float_idx = tau * (n_q - 1)
        idx2 = np.clip(float_idx.astype(np.int32), 0, n_q - 2)
        frac2 = np.clip(float_idx - idx2, 0, 1)

        lower_o = oq[pts[np.newaxis, :], idx2]
        upper_o = oq[pts[np.newaxis, :], idx2 + 1]
        x_qm = lower_o + frac2 * (upper_o - lower_o)

        # Step 3: delta(tau) = F_fut^{-1}(tau) / F_hist^{-1}(tau)
        fut_val = fq[pts[np.newaxis, :], idx2] + frac2 * (fq[pts[np.newaxis, :], idx2 + 1] - fq[pts[np.newaxis, :], idx2])
        hist_val = hq[pts[np.newaxis, :], idx2] + frac2 * (hq[pts[np.newaxis, :], idx2 + 1] - hq[pts[np.newaxis, :], idx2])

        # Multiplicative delta (avoid division by zero)
        hist_val_safe = np.where(hist_val > 1e-6, hist_val, 1e-6)
        delta = fut_val / hist_val_safe
        delta = np.clip(delta, 0.1, 10.0)

        # Step 4: x_QDM = x_QM * delta
        result[:, c0:c1] = np.clip(x_qm * delta, 0, clip_max)

    return result


periods = {
    'historical': ('historical', HIST_YEARS),
    'ssp245': ('ssp245', FUTURE_YEARS),
    'ssp370': ('ssp370', FUTURE_YEARS),
}

for gcm in GCM_LIST:
    hist_ws_q = cmip_hist_q[gcm]['ws100']     # (12, 157, 261, 101)
    hist_rsds_q = cmip_hist_q[gcm]['rsds']

    for period_name, (scenario, years) in periods.items():
        out_path = DATA_DIR / f'CMIP6_daily_CF_corrected_{period_name}_{gcm}.npz'
        print(f'\n  {gcm} x {period_name}...')
        t0 = time.time()

        # Future quantiles for QDM delta (for historical: use hist_q -> delta=1)
        if period_name == 'historical':
            fut_ws_q = hist_ws_q
            fut_rsds_q = hist_rsds_q
        else:
            key = f'{gcm}_{scenario}'
            fut_ws_q = cmip_fut_q[key]['ws100']
            fut_rsds_q = cmip_fut_q[key]['rsds']

        all_wind, all_solar, all_months = [], [], []

        for yi, year in enumerate(years):
            ws100_raw, rsds_raw, months = load_cmip6_met(gcm, scenario, year)
            if ws100_raw is None:
                continue
            n_days = ws100_raw.shape[0]

            # Corrected met vars (full grid)
            ws100_corr = np.full((n_days, N_LAT, N_LON), np.nan, dtype=np.float32)
            rsds_corr = np.full((n_days, N_LAT, N_LON), np.nan, dtype=np.float32)

            for m in range(12):
                day_idx = np.where(months == m)[0]
                if len(day_idx) == 0:
                    continue

                # Extract valid grid points: (n_days_month, N_VALID)
                ws_vals = ws100_raw[day_idx][:, valid_li, valid_loi]
                rsds_vals = rsds_raw[day_idx][:, valid_li, valid_loi]

                # Quantiles for valid points: (N_VALID, 101)
                h_ws_q = hist_ws_q[m][valid_li, valid_loi]
                f_ws_q = fut_ws_q[m][valid_li, valid_loi]
                o_ws_q = era5_ws_q[m][valid_li, valid_loi]

                h_rsds_q = hist_rsds_q[m][valid_li, valid_loi]
                f_rsds_q = fut_rsds_q[m][valid_li, valid_loi]
                o_rsds_q = era5_ssrd_q[m][valid_li, valid_loi]

                # QDM on wind speed
                ws_corr = vectorized_qdm(ws_vals, h_ws_q, f_ws_q, o_ws_q, clip_max=50.0)
                # QDM on radiation
                rsds_c = vectorized_qdm(rsds_vals, h_rsds_q, f_rsds_q, o_rsds_q, clip_max=1500.0)

                # Write back
                for di_local in range(len(day_idx)):
                    ws100_corr[day_idx[di_local], valid_li, valid_loi] = ws_corr[di_local]
                    rsds_corr[day_idx[di_local], valid_li, valid_loi] = rsds_c[di_local]

            # Compute CF from corrected met variables
            wind_cf_grid = wind_power_curve(ws100_corr)
            solar_cf_grid = solar_cf_simple(rsds_corr)

            # Province average + scaling
            cf_wind_prov = np.zeros((n_days, N_PROV), dtype=np.float32)
            cf_solar_prov = np.zeros((n_days, N_PROV), dtype=np.float32)
            for pi in range(N_PROV):
                idx = prov_indices[pi]
                if len(idx[0]) == 0:
                    continue
                raw_w = np.nanmean(wind_cf_grid[:, idx[0], idx[1]], axis=1)
                raw_s = np.nanmean(solar_cf_grid[:, idx[0], idx[1]], axis=1)
                cf_wind_prov[:, pi] = np.clip(raw_w * scale_wind[pi], 0, 1)
                cf_solar_prov[:, pi] = np.clip(raw_s * scale_solar[pi], 0, 1)

            all_wind.append(cf_wind_prov)
            all_solar.append(cf_solar_prov)
            all_months.append(months)

            if (yi + 1) % 5 == 0:
                print(f'    {year} ({yi+1}/{len(years)}, {time.time()-t0:.0f}s)')

        cf_w_all = np.concatenate(all_wind, axis=0)
        cf_s_all = np.concatenate(all_solar, axis=0)
        m_all = np.concatenate(all_months, axis=0)

        np.savez_compressed(out_path,
            cf_wind=cf_w_all, cf_solar=cf_s_all,
            months=m_all, provinces=PROVINCE_ORDER)

        print(f'    -> wind CF={np.nanmean(cf_w_all):.4f}, solar CF={np.nanmean(cf_s_all):.4f} '
              f'({time.time()-t0:.0f}s)')

# ============================================================
# Phase 4: Verification
# ============================================================
print('\n' + '=' * 70)
print('Phase 4: Verification')
print('=' * 70)

# Compare with step1 ERA5 CF (province-level reference)
print('\n  Reference: Step1 ERA5 CF (expected wind~0.22, solar~0.16)')
try:
    ref = np.load(STEP1_DATA / 'CF_corrected' / 'CF_corrected_2000.npz')
    print(f'  Step1 2000: wind={ref["cf_wind"].mean():.4f}, solar={ref["cf_solar"].mean():.4f}')
except Exception as e:
    print(f'  (Could not load step1 reference: {e})')

check_provs = ['Beijing', 'Inner Mongolia', 'Sichuan', 'Tibet', 'Qinghai', 'Xinjiang']

print(f'\n  Wind CF:')
print(f'  {"Province":18s} {"GCM":15s} {"Hist":>8s} {"SSP245":>8s} {"SSP370":>8s} {"d245":>8s} {"d370":>8s}')
for gcm in GCM_LIST:
    for prov in check_provs:
        pi = PROVINCE_ORDER.index(prov)
        vals = {}
        for p in ['historical', 'ssp245', 'ssp370']:
            fpath = DATA_DIR / f'CMIP6_daily_CF_corrected_{p}_{gcm}.npz'
            if not fpath.exists():
                vals[p] = 0.0
                continue
            d = np.load(fpath)
            vals[p] = d['cf_wind'][:, pi].mean()
        h = vals['historical']
        d245 = (vals['ssp245'] - h) / h * 100 if h > 0.001 else 0
        d370 = (vals['ssp370'] - h) / h * 100 if h > 0.001 else 0
        print(f'  {prov:18s} {gcm:15s} {h:8.4f} {vals["ssp245"]:8.4f} {vals["ssp370"]:8.4f} {d245:+7.1f}% {d370:+7.1f}%')

print(f'\n  Solar CF:')
print(f'  {"Province":18s} {"GCM":15s} {"Hist":>8s} {"SSP245":>8s} {"SSP370":>8s} {"d245":>8s} {"d370":>8s}')
for gcm in GCM_LIST[:2]:
    for prov in check_provs:
        pi = PROVINCE_ORDER.index(prov)
        vals = {}
        for p in ['historical', 'ssp245', 'ssp370']:
            fpath = DATA_DIR / f'CMIP6_daily_CF_corrected_{p}_{gcm}.npz'
            if not fpath.exists():
                vals[p] = 0.0
                continue
            d = np.load(fpath)
            vals[p] = d['cf_solar'][:, pi].mean()
        h = vals['historical']
        d245 = (vals['ssp245'] - h) / h * 100 if h > 0.001 else 0
        d370 = (vals['ssp370'] - h) / h * 100 if h > 0.001 else 0
        print(f'  {prov:18s} {gcm:15s} {h:8.4f} {vals["ssp245"]:8.4f} {vals["ssp370"]:8.4f} {d245:+7.1f}% {d370:+7.1f}%')

# National summary
print(f'\n  National summary ({len(GCM_LIST)}-GCM mean):')
for period in ['historical', 'ssp245', 'ssp370']:
    w_vals, s_vals = [], []
    for gcm in GCM_LIST:
        fpath = DATA_DIR / f'CMIP6_daily_CF_corrected_{period}_{gcm}.npz'
        if not fpath.exists():
            continue
        d = np.load(fpath)
        w_vals.append(d['cf_wind'].mean())
        s_vals.append(d['cf_solar'].mean())
    if w_vals:
        print(f'  {period:12s}: wind={np.mean(w_vals):.4f} +/- {np.std(w_vals):.4f}, '
              f'solar={np.mean(s_vals):.4f} +/- {np.std(s_vals):.4f}')

print('\n' + '=' * 70)
print('Step3A3 v5 QDM-on-met-vars COMPLETE! Next: run step3D_compound_drought.py')
print('=' * 70)
