# -*- coding: utf-8 -*-
"""
step3E_provincial_storage.py
============================
Compute provincial post-TX storage needs (modified constrained_rebalance).
Panel A: LSLW amplification, Panel B: waterfall decomposition,
Panel D: provincial tile matrix.

Output:
  - data/provincial_postTX_results.npz
  - figure_new/fig_panelA_lslw_amplification.png
  - figure_new/fig_panelB_national_4layer.png
  - figure_new/fig_panelB_waterfall.png
  - figure_new/fig_panelD_v3b_tile.png
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
from path_config import (
    BASE_DIR, DATA_DIR, FIG_DIR, OUT_DIR, EXCEL_DIR, SOURCE_DATA_DIR,
    INPUT_TABLES_DIR, FIGURES_DIR, CODE_DIR, ERA5_HOURLY, ERA5_DAILY_RSDS,
    CMIP6_DIR, STEP1_DATA, ZHUO_DIR, HYDRO_PATH, TX_PATH, MOESM4_PATH,
    LOAD_FILE, GRID_COORDS, GRID_MAP, ensure_output_dirs,
)
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Constants (from step3E_flexibility_requirement.py)
# ================================================================

BASE_DIR = BASE_DIR
DATA_DIR = DATA_DIR
FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_DIR = SOURCE_DATA_DIR
HYDRO_PATH = HYDRO_PATH
TX_PATH = TX_PATH

PROVINCE_ORDER = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia',
    'Liaoning', 'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu',
    'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong',
    'Henan', 'Hubei', 'Hunan', 'Guangdong', 'Guangxi',
    'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 'Yunnan',
    'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]
N_PROV = 31

PROV_SHORT = [
    'BJ', 'TJ', 'HE', 'SX', 'IM', 'LN', 'JL', 'HL', 'SH', 'JS',
    'ZJ', 'AH', 'FJ', 'JX', 'SD', 'HA', 'HB', 'HN', 'GD', 'GX',
    'HI', 'CQ', 'SC', 'GZ', 'YN', 'XZ', 'SN', 'GS', 'QH', 'NX', 'XJ'
]

# 7 regions
REGION_NAMES = ['North', 'Northeast', 'East', 'Central', 'South',
                'Southwest', 'Northwest']
REGION_PROVINCES = {
    'North': ['Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia'],
    'Northeast': ['Liaoning', 'Jilin', 'Heilongjiang'],
    'East': ['Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian',
             'Jiangxi', 'Shandong'],
    'Central': ['Henan', 'Hubei', 'Hunan'],
    'South': ['Guangdong', 'Guangxi', 'Hainan'],
    'Southwest': ['Chongqing', 'Sichuan', 'Guizhou', 'Yunnan', 'Tibet'],
    'Northwest': ['Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'],
}
REGION_COLORS = {
    'North': '#e41a1c', 'Northeast': '#377eb8', 'East': '#4daf4a',
    'Central': '#984ea3', 'South': '#ff7f00', 'Southwest': '#a65628',
    'Northwest': '#f781bf',
}

# Province → region mapping
PROV_TO_REGION = {}
for _r, _ps in REGION_PROVINCES.items():
    for _p in _ps:
        PROV_TO_REGION[_p] = _r

GCM_LIST = ['ACCESS-CM2', 'EC-Earth3', 'GFDL-ESM4', 'MRI-ESM2-0', 'NorESM2-MM']
N_GCM = len(GCM_LIST)
CAP_SCENARIOS = ['NDC', 'GM2.0', 'CN2050']
SSP_LIST = ['ssp245', 'ssp370']

DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

NUCLEAR_CAP_MW = np.zeros(N_PROV)
NUCLEAR_CAP_MW[5] = 20000; NUCLEAR_CAP_MW[9] = 30000
NUCLEAR_CAP_MW[10] = 30000; NUCLEAR_CAP_MW[12] = 20000
NUCLEAR_CAP_MW[14] = 20000; NUCLEAR_CAP_MW[18] = 40000
NUCLEAR_CAP_MW[19] = 20000; NUCLEAR_CAP_MW[20] = 20000
NUCLEAR_CF = 0.85
NUCLEAR_GEN_GW = NUCLEAR_CAP_MW * NUCLEAR_CF / 1000.0

FOSSIL_NAT = {
    'NDC': {'coal': 910.0, 'gas': 137.0},
    'GM2.0':   {'coal': 156.0, 'gas': 449.0},
    'CN2050':  {'coal': 0.0,   'gas': 389.0},
}

COLOR_SSP245 = '#4393c3'
COLOR_SSP370 = '#d73027'
SCENARIO_LABELS = {'NDC': 'NDC', 'GM2.0': 'GM2.0', 'CN2050': 'CN2050'}
SSP_LABELS = {'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0'}

# ================================================================
# Data Loading
# ================================================================

def load_daily_cf(period, gcm):
    d = np.load(DATA_DIR / f'CMIP6_daily_CF_corrected_{period}_{gcm}.npz')
    return d['cf_wind'], d['cf_solar'], d['months']

def load_scenario_data(cap_scen, ssp):
    d = np.load(DATA_DIR / f'supply_demand_2050_{cap_scen}_{ssp}.npz',
                allow_pickle=True)
    return {k: d[k].astype(np.float64)
            for k in ['cap_wind', 'cap_solar', 'cap_hydro', 'monthly_demand']}

def load_hydro_cf():
    return np.load(HYDRO_PATH)['cf_mean'].astype(np.float64)

def load_lslw_thresholds():
    d = np.load(DATA_DIR / 'lslw_results.npz')
    return d['wind_p10_daily'], d['solar_p10_daily']

def load_transmission_capacity():
    df = pd.read_excel(TX_PATH, sheet_name='channel_capacity',
                       index_col=0, nrows=31)
    return df.values.astype(np.float64)

def compute_doy(months):
    n = len(months)
    doy = np.zeros(n, dtype=np.int32)
    yr_day = 0
    for i in range(n):
        if i > 0 and months[i] == 0 and months[i - 1] == 11:
            yr_day = 0
        doy[i] = min(yr_day, 364)
        yr_day += 1
    return doy

# ================================================================
# Provincial Fossil Capacity
# ================================================================

_cn2050_gas_cache = None

def load_cn2050_provincial_fossil():
    global _cn2050_gas_cache
    if _cn2050_gas_cache is not None:
        return _cn2050_gas_cache
    df = pd.read_excel(MOESM4_PATH, sheet_name='Fig.5', index_col=0)
    gas = np.zeros(N_PROV)
    coal = np.zeros(N_PROV)
    gas_cols = [c for c in df.columns if 'Gas' in str(c)]
    coal_cols = [c for c in df.columns if 'Coal' in str(c)]
    for i, prov in enumerate(PROVINCE_ORDER):
        if prov in df.index:
            gas[i] = sum(df.loc[prov, c] for c in gas_cols)
            coal[i] = sum(df.loc[prov, c] for c in coal_cols)
    _cn2050_gas_cache = (gas, coal)
    return gas, coal

def get_provincial_fossil(cap_scen, ssp):
    if cap_scen == 'CN2050':
        return load_cn2050_provincial_fossil()
    sd = load_scenario_data(cap_scen, ssp)
    annual_demand = sd['monthly_demand'].sum(axis=0)
    share = annual_demand / annual_demand.sum()
    gas_MW = share * FOSSIL_NAT[cap_scen]['gas'] * 1000
    coal_MW = share * FOSSIL_NAT[cap_scen]['coal'] * 1000
    return gas_MW, coal_MW

# ================================================================
# Transmission (modified for provincial output)
# ================================================================

def find_grid_regions(tx_cap, threshold=500):
    parent = list(range(N_PROV))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py
    for i in range(N_PROV):
        for j in range(N_PROV):
            if tx_cap[i, j] >= threshold or tx_cap[j, i] >= threshold:
                union(i, j)
    return np.array([find(i) for i in range(N_PROV)])

def build_inter_region_links(tx_cap, regions):
    unique_regions = np.unique(regions)
    region_map = {r: idx for idx, r in enumerate(unique_regions)}
    n_r = len(unique_regions)
    inter_cap = np.zeros((n_r, n_r))
    for i in range(N_PROV):
        for j in range(N_PROV):
            ri, rj = region_map[regions[i]], region_map[regions[j]]
            if ri != rj and 0 < tx_cap[i, j] < 500:
                inter_cap[ri, rj] += tx_cap[i, j]
    return inter_cap, unique_regions, region_map


def build_lp_structure(inter_cap_GW, n_r):
    """Build LP constraint matrix structure (invariant across days).

    Variables: [f[0], ..., f[n_e-1], u[0], ..., u[n_r-1]]
    f[e] = flow on directed edge e (GW), u[r] = unmet at region r (GW).
    """
    edges = []
    for ri in range(n_r):
        for rj in range(n_r):
            if ri != rj and inter_cap_GW[ri, rj] > 0:
                edges.append((ri, rj))
    n_e = len(edges)
    n_vars = n_e + n_r

    inflow_edges = {r: [] for r in range(n_r)}
    outflow_edges = {r: [] for r in range(n_r)}
    for e_idx, (ri, rj) in enumerate(edges):
        outflow_edges[ri].append(e_idx)
        inflow_edges[rj].append(e_idx)

    c = np.zeros(n_vars)
    c[n_e:] = 1.0

    rows_data, rows_row, rows_col = [], [], []
    row_idx = 0

    # Constraint: -u[r] - inflow[r] + outflow[r] <= -region_net[r]
    for r in range(n_r):
        rows_data.append(-1.0); rows_row.append(row_idx); rows_col.append(n_e + r)
        for e_idx in inflow_edges[r]:
            rows_data.append(-1.0); rows_row.append(row_idx); rows_col.append(e_idx)
        for e_idx in outflow_edges[r]:
            rows_data.append(1.0); rows_row.append(row_idx); rows_col.append(e_idx)
        row_idx += 1

    # Constraint: outflow[r] - inflow[r] <= surplus[r]
    for r in range(n_r):
        for e_idx in outflow_edges[r]:
            rows_data.append(1.0); rows_row.append(row_idx); rows_col.append(e_idx)
        for e_idx in inflow_edges[r]:
            rows_data.append(-1.0); rows_row.append(row_idx); rows_col.append(e_idx)
        row_idx += 1

    A_ub = csr_matrix((rows_data, (rows_row, rows_col)),
                      shape=(row_idx, n_vars))
    cap_bounds = np.array([inter_cap_GW[ri, rj] for (ri, rj) in edges])

    return edges, c, A_ub, cap_bounds, n_e


def constrained_rebalance_provincial(residual, regions, inter_cap_GW,
                                     region_map, unique_regions):
    """LP-optimal constrained TX returning provincial post-TX deficit.

    Uses scipy.optimize.linprog (HiGHS) to solve optimal inter-region
    power flow for each day independently.

    Returns:
        national_deficit: (n_days,)
        prov_post_tx: (n_days, 31)
    """
    n_days = residual.shape[0]
    n_r = len(unique_regions)
    prov_region_idx = np.array([region_map[regions[p]] for p in range(N_PROV)])

    # Step 1: aggregate to regions
    region_net = np.zeros((n_days, n_r))
    for p in range(N_PROV):
        region_net[:, prov_region_idx[p]] += residual[:, p]

    # Build LP structure once
    edges, c, A_ub, cap_bounds, n_e = build_lp_structure(inter_cap_GW, n_r)
    n_vars = n_e + n_r
    bounds = [(0, cap_bounds[e]) for e in range(n_e)] + \
             [(0, None) for _ in range(n_r)]

    # Step 2: solve LP per day
    r_deficit_lp = np.zeros((n_days, n_r))
    lp_fail_count = 0

    for d in range(n_days):
        deficit_d = np.maximum(region_net[d], 0)
        surplus_d = np.maximum(-region_net[d], 0)

        if deficit_d.sum() < 1e-6:
            continue

        b_ub = np.concatenate([-region_net[d], surplus_d])
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs',
                         options={'presolve': True, 'disp': False})

        if result.success:
            r_deficit_lp[d] = result.x[n_e:]
        else:
            r_deficit_lp[d] = deficit_d
            lp_fail_count += 1

    if lp_fail_count > 0:
        print(f'  WARNING: {lp_fail_count}/{n_days} LP solves failed')

    national_deficit = r_deficit_lp.sum(axis=1)

    # Step 3: redistribute regional deficit → provinces
    prov_post_tx = np.zeros((n_days, N_PROV), dtype=np.float32)
    for ri in range(n_r):
        prov_indices = np.where(prov_region_idx == ri)[0]
        if len(prov_indices) == 0:
            continue
        prov_deficits = np.maximum(residual[:, prov_indices], 0)
        region_total_def = prov_deficits.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            shares = np.where(region_total_def > 0,
                              prov_deficits / region_total_def, 0)
        remaining = r_deficit_lp[:, ri:ri+1]
        prov_post_tx[:, prov_indices] = (remaining * shares).astype(np.float32)

    return national_deficit, prov_post_tx


# ================================================================
# Dispatch Chain (Layers 1-4, no storage)
# ================================================================

def compute_re_residual(cap_scen, ssp, gcm, hydro_cf):
    cf_wind, cf_solar, months = load_daily_cf(ssp, gcm)
    sd = load_scenario_data(cap_scen, ssp)
    n = len(months)
    demand = np.zeros((n, N_PROV))
    hydro_d = np.zeros((n, N_PROV))
    for i in range(n):
        m = months[i]
        demand[i] = sd['monthly_demand'][m] / DAYS_PER_MONTH[m] * 1000 / 24
        hydro_d[i] = hydro_cf[m]
    gen_w = cf_wind * sd['cap_wind'][None, :] / 1000
    gen_s = cf_solar * sd['cap_solar'][None, :] / 1000
    gen_h = hydro_d * sd['cap_hydro'][None, :] / 1000
    residual = demand - gen_w - gen_s - gen_h - NUCLEAR_GEN_GW[None, :]
    return residual, cf_wind, cf_solar, months


def provincial_fossil_dispatch(residual, gas_cap_GW, coal_cap_GW):
    deficit = np.maximum(residual, 0)
    gas_used = np.minimum(deficit, gas_cap_GW[None, :])
    remaining = deficit - gas_used
    coal_used = np.minimum(remaining, coal_cap_GW[None, :])
    post_fossil = residual - gas_used - coal_used
    return post_fossil, gas_used, coal_used


def compute_lslw_mask(cf_w, cf_s, months, wind_p10, solar_p10):
    n = len(months)
    doy = compute_doy(months)
    mask = np.zeros((n, N_PROV), dtype=bool)
    for i in range(n):
        d = min(doy[i], 364)
        mask[i] = (cf_w[i] <= wind_p10[d]) & (cf_s[i] <= solar_p10[d])
    return mask


def run_dispatch_provincial(cap_scen, ssp, gcm, hydro_cf,
                            regions, inter_cap_GW, region_map, unique_regions):
    """Run Layers 1-4, return provincial post-TX + national summary."""
    residual, cf_w, cf_s, months = compute_re_residual(cap_scen, ssp, gcm, hydro_cf)
    n_days = len(months)
    n_years = n_days / 365.0
    to_TWh = 24 / 1000 / n_years

    gas_MW, coal_MW = get_provincial_fossil(cap_scen, ssp)
    gas_GW = gas_MW / 1000
    coal_GW = coal_MW / 1000

    # Layers 2-3
    post_fossil, gas_used, coal_used = provincial_fossil_dispatch(
        residual, gas_GW, coal_GW)

    # Layer 4: LP-optimal constrained TX
    nat_deficit, prov_post_tx = constrained_rebalance_provincial(
        post_fossil, regions, inter_cap_GW, region_map, unique_regions)

    # Provincial annual unmet TWh
    prov_unmet_TWh = prov_post_tx.sum(axis=0) * to_TWh  # (31,)

    # Provincial gas/coal usage TWh
    prov_gas_TWh = gas_used.sum(axis=0) * to_TWh
    prov_coal_TWh = coal_used.sum(axis=0) * to_TWh

    # National summary
    deficit_autarky = np.maximum(residual, 0).sum(axis=1)
    deficit_post_fossil = np.maximum(post_fossil, 0).sum(axis=1)
    gas_total = gas_used.sum(axis=1)
    coal_total = coal_used.sum(axis=1)
    ctx_value = deficit_post_fossil - nat_deficit

    return {
        'prov_unmet_TWh': prov_unmet_TWh,
        'prov_gas_TWh': prov_gas_TWh,
        'prov_coal_TWh': prov_coal_TWh,
        'deficit_autarky_TWh': deficit_autarky.sum() * to_TWh,
        'gas_TWh': gas_total.sum() * to_TWh,
        'coal_TWh': coal_total.sum() * to_TWh,
        'ctx_value_TWh': ctx_value.sum() * to_TWh,
        'national_unmet_TWh': nat_deficit.sum() * to_TWh,
        # For LSLW stratification
        'residual': residual,
        'cf_w': cf_w, 'cf_s': cf_s, 'months': months,
        'deficit_autarky_daily': deficit_autarky,
        'gas_total_daily': gas_total,
        'coal_total_daily': coal_total,
        'ctx_value_daily': ctx_value,
        'nat_deficit_daily': nat_deficit,
        'prov_post_tx_daily': prov_post_tx,
    }


# ================================================================
# Main Computation
# ================================================================

def compute_all():
    print('=' * 60)
    print('Computing provincial post-TX storage needs...')
    print('=' * 60)

    hydro_cf = load_hydro_cf()
    wind_p10, solar_p10 = load_lslw_thresholds()
    tx_cap = load_transmission_capacity()
    regions = find_grid_regions(tx_cap)
    inter_cap, u_reg, r_map = build_inter_region_links(tx_cap, regions)
    n_e = int((inter_cap > 0).sum())
    print(f'  {len(u_reg)} AC regions, {n_e} directed UHVDC edges')

    # Provincial unmet: (cap, ssp) → (N_GCM, 31) TWh/yr
    prov_results = {}
    # National summary: (cap, ssp) → dict of 5-GCM means
    national_results = {}
    # LSLW stratification for Panel a
    lslw_data = {}

    for ssp in SSP_LIST:
        for gi, gcm in enumerate(GCM_LIST):
            print(f'  {ssp} / {gcm}...')
            for cap_scen in CAP_SCENARIOS:
                res = run_dispatch_provincial(
                    cap_scen, ssp, gcm, hydro_cf,
                    regions, inter_cap, r_map, u_reg)

                key = (cap_scen, ssp)
                prov_results.setdefault(key, []).append(res['prov_unmet_TWh'])

                national_results.setdefault(key, {
                    k: [] for k in ['deficit_autarky_TWh', 'gas_TWh',
                                    'coal_TWh', 'ctx_value_TWh',
                                    'national_unmet_TWh']
                })
                for k in national_results[key]:
                    national_results[key][k].append(res[k])

                # LSLW for CN2050
                if cap_scen == 'CN2050':
                    is_lslw = compute_lslw_mask(
                        res['cf_w'], res['cf_s'], res['months'],
                        wind_p10, solar_p10)
                    sync = is_lslw.sum(axis=1) >= 3
                    normal = ~sync
                    n_s = int(sync.sum())
                    strat = {}
                    for name, arr in [
                        ('deficit_autarky', res['deficit_autarky_daily']),
                        ('gas_total', res['gas_total_daily']),
                        ('ctx_value', res['ctx_value_daily']),
                        ('national_unmet', res['nat_deficit_daily']),
                    ]:
                        strat[f'{name}_normal'] = float(arr[normal].mean())
                        strat[f'{name}_lslw'] = float(arr[sync].mean()) if n_s > 0 else 0
                    # Provincial post-TX on LSLW vs normal
                    prov_lslw = res['prov_post_tx_daily'][sync].mean(axis=0) if n_s > 0 else np.zeros(N_PROV)
                    prov_normal = res['prov_post_tx_daily'][normal].mean(axis=0)
                    strat['prov_unmet_lslw_GW'] = prov_lslw
                    strat['prov_unmet_normal_GW'] = prov_normal
                    lslw_data.setdefault(ssp, []).append(strat)

    # Average over GCMs
    prov_mean = {}  # (cap, ssp) → (31,) mean
    prov_std = {}
    prov_each = {}  # (cap, ssp) → (N_GCM, 31)
    for key in prov_results:
        arr = np.array(prov_results[key])  # (N_GCM, 31)
        prov_mean[key] = arr.mean(axis=0)
        prov_std[key] = arr.std(axis=0)
        prov_each[key] = arr

    nat_mean = {}
    for key in national_results:
        nat_mean[key] = {k: np.mean(v) for k, v in national_results[key].items()}

    lslw_mean = {}
    for ssp in SSP_LIST:
        lslw_mean[ssp] = {}
        for k in lslw_data[ssp][0]:
            vals = [d[k] for d in lslw_data[ssp]]
            if isinstance(vals[0], np.ndarray):
                lslw_mean[ssp][k] = np.mean(vals, axis=0)
            else:
                lslw_mean[ssp][k] = np.mean(vals)

    # Print summary
    print(f'\n{"="*60}')
    print('National 4-layer summary (no storage):')
    hdr = f'  {"Scen":8s} {"SSP":8s} {"Deficit":>7s} {"Gas":>6s} {"Coal":>6s} {"TX":>6s} {"Unmet":>7s}'
    print(hdr)
    for cap in CAP_SCENARIOS:
        for ssp in SSP_LIST:
            nm = nat_mean[(cap, ssp)]
            print(f'  {SCENARIO_LABELS[cap]:8s} {SSP_LABELS[ssp]:8s} '
                  f'{nm["deficit_autarky_TWh"]:7.0f} '
                  f'{nm["gas_TWh"]:6.0f} {nm["coal_TWh"]:6.0f} '
                  f'{nm["ctx_value_TWh"]:6.0f} '
                  f'{nm["national_unmet_TWh"]:7.0f}')

    print(f'\nProvincial unmet top-10 (CN2050, SSP3-7.0):')
    u = prov_mean[('CN2050', 'ssp370')]
    order = np.argsort(u)[::-1]
    for i in range(min(10, N_PROV)):
        p = order[i]
        print(f'  {PROVINCE_ORDER[p]:15s} {u[p]:6.1f} TWh/yr '
              f'(±{prov_std[("CN2050","ssp370")][p]:.1f})')

    # Save
    save_dict = {
        'provinces': np.array(PROVINCE_ORDER),
        'prov_short': np.array(PROV_SHORT),
    }
    for cap in CAP_SCENARIOS:
        for ssp in SSP_LIST:
            pf = f'{cap}_{ssp}'
            save_dict[f'{pf}_prov_unmet_TWh_mean'] = prov_mean[(cap, ssp)].astype(np.float32)
            save_dict[f'{pf}_prov_unmet_TWh_std'] = prov_std[(cap, ssp)].astype(np.float32)
            save_dict[f'{pf}_prov_unmet_TWh_each'] = prov_each[(cap, ssp)].astype(np.float32)
            for k, v in nat_mean[(cap, ssp)].items():
                save_dict[f'{pf}_{k}'] = np.float32(v)
    np.savez_compressed(DATA_DIR / 'provincial_postTX_results.npz', **save_dict)
    print(f'\nSaved: {DATA_DIR / "provincial_postTX_results.npz"}')

    return prov_mean, prov_std, prov_each, nat_mean, lslw_mean


# ================================================================
# Visualization Helpers
# ================================================================

def setup_rcparams():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 9,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'axes.linewidth': 0.7, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
    })


# ================================================================
# Panel A: LSLW Amplification (Cleveland dot plot)
# ================================================================

def plot_panel_a(lslw_mean):
    setup_rcparams()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    metrics = [
        ('deficit_autarky', 'RE deficit'),
        ('gas_total', 'Gas used'),
        ('ctx_value', 'TX value'),
        ('national_unmet', 'Unmet'),
    ]
    y_pos = np.arange(len(metrics))

    for ssp, color, y_off in [('ssp245', COLOR_SSP245, 0.12),
                               ('ssp370', COLOR_SSP370, -0.12)]:
        for dt, marker, fill, label_suf in [
            ('normal', 'o', 'white', 'Normal'),
            ('lslw', 'o', color, 'LSLW k≥3'),
        ]:
            vals = [lslw_mean[ssp][f'{m}_{dt}'] for m, _ in metrics]
            edgecolor = color
            facecolor = fill
            label = f'{SSP_LABELS[ssp]} {label_suf}'
            ax.scatter(vals, y_pos + y_off, c=facecolor, edgecolors=edgecolor,
                       s=55, marker=marker, linewidths=1.2, label=label, zorder=5)

        # Connecting lines (band) between normal and LSLW
        for i, (m, _) in enumerate(metrics):
            v_n = lslw_mean[ssp][f'{m}_normal']
            v_l = lslw_mean[ssp][f'{m}_lslw']
            ax.plot([v_n, v_l], [y_pos[i]+y_off, y_pos[i]+y_off],
                    color=color, lw=2, alpha=0.4, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([lab for _, lab in metrics])
    ax.set_xlabel('Mean daily value (GW)')
    ax.invert_yaxis()
    ax.legend(fontsize=6, ncol=2, loc='upper right',
              framealpha=0.9, edgecolor='gray',
              bbox_to_anchor=(1.0, 1.12))
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'fig_panelA_lslw_amplification.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved Panel A: fig_panelA_lslw_amplification.png')


# ================================================================
# Panel B: National 4-Layer Stacked (no storage)
# ================================================================

def plot_panel_b(nat_mean):
    setup_rcparams()
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(CAP_SCENARIOS))
    w = 0.35

    for si_ssp, ssp in enumerate(SSP_LIST):
        gas, coal, tx, unmet = [], [], [], []
        for cap in CAP_SCENARIOS:
            nm = nat_mean[(cap, ssp)]
            gas.append(nm['gas_TWh'])
            coal.append(nm['coal_TWh'])
            tx.append(nm['ctx_value_TWh'])
            unmet.append(nm['national_unmet_TWh'])
        gas, coal, tx, unmet = [np.array(v) for v in [gas, coal, tx, unmet]]

        off = -w/2 if ssp == 'ssp245' else w/2
        alp = 0.75 if ssp == 'ssp245' else 0.95

        # Stack bottom→top: Gas → Coal → TX → Unmet
        b = np.zeros(len(CAP_SCENARIOS))
        for vals, color, label in [
            (gas,   '#f4a582', 'Gas'),
            (coal,  '#878787', 'Coal'),
            (tx,    '#4dac26', 'Transmission'),
            (unmet, '#b2182b', 'Unmet (storage need)'),
        ]:
            ax.bar(x + off, vals, w, bottom=b, color=color, alpha=alp,
                   label=label if si_ssp == 0 else '',
                   edgecolor='white', linewidth=0.3)
            b += vals

        # Total label
        total = gas + coal + tx + unmet
        for i in range(len(CAP_SCENARIOS)):
            ax.text(x[i] + off, total[i] + 50, SSP_LABELS[ssp],
                    ha='center', va='bottom', fontsize=7,
                    color=COLOR_SSP245 if ssp == 'ssp245' else COLOR_SSP370)
            # Unmet value
            if unmet[i] > 50:
                ax.text(x[i] + off, b[i] - unmet[i]/2 - 30,
                        f'{unmet[i]:.0f}',
                        ha='center', va='center', fontsize=6.5,
                        color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in CAP_SCENARIOS])
    ax.set_ylabel('Annual energy (TWh/yr)')
    ax.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.9,
    bbox_to_anchor = (0.0, 1.15))
    ax.set_ylim(bottom=0)
    ax.set_title('National Flexibility Decomposition (no storage)',
                 fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'fig_panelB_national_4layer.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved Panel B: fig_panelB_national_4layer.png')


CAP_COLORS = {'NDC': '#666666', 'GM2.0': '#74add1', 'CN2050': '#d73027'}

def _get_sort_order(prov_mean):
    """Sort provinces by CN2050 SSP370 unmet, filter >0.5 TWh in any scenario."""
    u_max = np.zeros(N_PROV)
    for cap in CAP_SCENARIOS:
        for ssp in SSP_LIST:
            u_max = np.maximum(u_max, prov_mean[(cap, ssp)])
    order = np.argsort(prov_mean[('CN2050', 'ssp370')])[::-1]
    return [p for p in order if u_max[p] > 0.5]


def plot_panelD_v3b_tile(prov_mean):
    """B: Tile matrix — Y=province, X=6 columns, color=unmet TWh."""
    setup_rcparams()
    order = _get_sort_order(prov_mean)

    # Build matrix: rows=provinces (sorted), cols=6 scenario combos
    col_labels = []
    matrix = []
    for cap in CAP_SCENARIOS:
        for ssp in SSP_LIST:
            col_labels.append(f'{SCENARIO_LABELS[cap]}\n{SSP_LABELS[ssp]}')
            matrix.append([prov_mean[(cap, ssp)][p] for p in order])
    matrix = np.array(matrix).T  # (n_prov, 6)

    fig, ax = plt.subplots(figsize=(6, 9))

    # Use custom colormap: white → yellow → orange → dark red
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('storage',
        ['#ffffff', '#fff7bc', '#fec44f', '#ec7014', '#b2182b'], N=256)

    vmax = matrix.max() * 1.0
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=vmax,
                   interpolation='nearest')

    # Grid lines
    for i in range(len(order) + 1):
        ax.axhline(i - 0.5, color='white', lw=0.8)
    for j in range(7):
        ax.axvline(j - 0.5, color='white', lw=0.8)
    # Thick lines between scenarios
    for j in [1.5, 3.5]:
        ax.axvline(j, color='#333333', lw=1.5)

    # Number labels removed for clean Nature style (values in SI table)

    ax.set_xticks(range(6))
    ax.set_xticklabels(col_labels, fontsize=7, fontweight='bold')
    # Color x-tick labels by scenario
    xtick_colors = [CAP_COLORS[c] for c in CAP_SCENARIOS for _ in SSP_LIST]
    for tick, c in zip(ax.get_xticklabels(), xtick_colors):
        tick.set_color(c)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PROVINCE_ORDER[p] for p in order],
                       fontsize=6.5)
    for tick, p in zip(ax.get_yticklabels(), order):
        tick.set_color(REGION_COLORS[PROV_TO_REGION[PROVINCE_ORDER[p]]])

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=25)
    cbar.set_label('Unmet demand (TWh/yr)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title('Provincial Storage Need\n', fontsize=10,
                 fontweight='bold', pad=25)

    fig.savefig(FIG_DIR / 'fig_panelD_v3b_tile.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: fig_panelD_v3b_tile.png')


def _draw_waterfall_panel(ax, nm, title_str):
    """Draw a single waterfall panel on the given axes."""
    layer_colors = {
        'Gas': '#f4a582', 'Coal': '#878787',
        'TX': '#66c2a5', 'Unmet': '#b2182b',
    }
    layer_names = ['Gas', 'Coal', 'TX', 'Unmet']

    deficit = nm['deficit_autarky_TWh']
    gas = nm['gas_TWh']
    coal = nm['coal_TWh']
    tx = nm['ctx_value_TWh']
    unmet = nm['national_unmet_TWh']

    labels = ['Deficit', 'Gas', 'Coal', 'TX', 'Unmet']
    layer_vals = [gas, coal, tx, unmet]
    top = deficit
    x_pos = np.arange(len(labels))

    # Deficit bar: transparent with dashed outline
    ax.bar(0, deficit, 0.65, bottom=0, color='#c6a4d8', alpha=0.45,
           edgecolor='#8e6aaf', linewidth=1.2, linestyle='--')
    ax.text(0, deficit / 2, f'{deficit:,.0f}', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white')

    # Cascading bars
    for li, (name, val) in enumerate(zip(layer_names, layer_vals)):
        xi = li + 1
        bottom = top - val
        color = layer_colors[name]

        if val < 1:
            ax.plot([xi - 0.7, xi + 0.7], [top, top],
                    color='gray', lw=0.5, ls='--', alpha=0.5)
            top = bottom
            continue

        ax.bar(xi, val, 0.65, bottom=bottom, color=color, alpha=0.9,
               edgecolor='white', linewidth=0.3)

        mid = bottom + val / 2
        fontcolor = 'white' if val > 400 else '#333333'
        ax.text(xi, mid, f'{val:,.0f}', ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=fontcolor)
        ax.plot([xi - 0.7, xi - 0.325], [top, top],
                color='gray', lw=0.6, ls='--', alpha=0.6)
        top = bottom

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
    ax.set_title(title_str, fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)


def plot_panel_b_waterfall(nat_mean):
    """Panel B: Waterfall chart — 2×3 (top: SSP3-7.0, bottom: SSP2-4.5)."""
    setup_rcparams()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.45)

    for ri, ssp in enumerate(SSP_LIST):  # ssp245 top, ssp370 bottom
        for ci, cap in enumerate(CAP_SCENARIOS):
            ax = axes[ri, ci]
            nm = nat_mean[(cap, ssp)]
            title = f'{SCENARIO_LABELS[cap]}\n({SSP_LABELS[ssp]})'
            _draw_waterfall_panel(ax, nm, title)
            if ci == 0:
                ax.set_ylabel('Annual energy (TWh/yr)')

    # Shared legend at top
    from matplotlib.patches import Patch
    layer_colors = {'Gas': '#f4a582', 'Coal': '#878787',
                    'TX': '#66c2a5', 'Unmet': '#b2182b'}
    layer_names = ['Gas', 'Coal', 'TX', 'Unmet']
    handles = [Patch(facecolor=layer_colors[n], label=n) for n in layer_names]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8,
               framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, 1.01))

    fig.savefig(FIG_DIR / 'fig_panelB_waterfall.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved Panel B: fig_panelB_waterfall.png')


# ================================================================
# Main
# ================================================================

def load_from_npz():
    """Load precomputed provincial data (skip recomputation)."""
    d = np.load(DATA_DIR / 'provincial_postTX_results.npz', allow_pickle=True)
    prov_mean = {}
    prov_std = {}
    nat_mean = {}
    for cap in CAP_SCENARIOS:
        for ssp in SSP_LIST:
            pf = f'{cap}_{ssp}'
            prov_mean[(cap, ssp)] = d[f'{pf}_prov_unmet_TWh_mean']
            prov_std[(cap, ssp)] = d[f'{pf}_prov_unmet_TWh_std']
            nat_mean[(cap, ssp)] = {
                k: float(d[f'{pf}_{k}'])
                for k in ['deficit_autarky_TWh', 'gas_TWh', 'coal_TWh',
                           'ctx_value_TWh', 'national_unmet_TWh']
            }
    return prov_mean, prov_std, nat_mean


def main():
    import sys
    # If --plot-only, skip computation and read from saved npz
    if '--plot-only' in sys.argv:
        print('Loading precomputed data...')
        prov_mean, prov_std, nat_mean = load_from_npz()
        print('Drawing figures...')
        plot_panel_b_waterfall(nat_mean)
        plot_panelD_v3b_tile(prov_mean)
        print(f'\nDone! Check figure_new/ for:')
        print('  fig_panelB_waterfall.png  (waterfall decomposition)')
        print('  fig_panelD_v3b_tile.png   (tile matrix)')
        return

    prov_mean, prov_std, prov_each, nat_mean, lslw_mean = compute_all()

    print(f'\nDrawing figures...')
    plot_panel_a(lslw_mean)
    plot_panel_b(nat_mean)
    plot_panel_b_waterfall(nat_mean)
    plot_panelD_v3b_tile(prov_mean)

    print(f'\n{"="*60}')
    print('All done! Check figure_new/ for:')
    print('  fig_panelA_lslw_amplification.png')
    print('  fig_panelB_national_4layer.png')
    print('  fig_panelB_waterfall.png  (waterfall decomposition)')
    print('  fig_panelD_v3b_tile.png   (tile matrix)')
    print(f'Data: {DATA_DIR / "provincial_postTX_results.npz"}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
