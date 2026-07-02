# -*- coding: utf-8 -*-
"""
generate_all_excel_tables.py
============================
Generate all 13 Excel files for the submission package.

  Table files (8):  one per main figure
  Input files (5):  input parameters / source data

Output directory:
  data/source_data/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from path_config import (
    BASE_DIR, DATA_DIR, FIG_DIR, OUT_DIR, EXCEL_DIR, SOURCE_DATA_DIR,
    INPUT_TABLES_DIR, FIGURES_DIR, CODE_DIR, ERA5_HOURLY, ERA5_DAILY_RSDS,
    CMIP6_DIR, STEP1_DATA, ZHUO_DIR, HYDRO_PATH, TX_PATH, MOESM4_PATH,
    LOAD_FILE, GRID_COORDS, GRID_MAP, ensure_output_dirs,
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
BASE_DIR = BASE_DIR
DATA_DIR = DATA_DIR
EXCEL_DIR = SOURCE_DATA_DIR
OUT_DIR = SOURCE_DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

HYDRO_PATH = HYDRO_PATH
TX_PATH = TX_PATH

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
PROV_SHORT = [
    'BJ','TJ','HE','SX','IM','LN','JL','HL','SH','JS',
    'ZJ','AH','FJ','JX','SD','HA','HB','HN','GD','GX',
    'HI','CQ','SC','GZ','YN','XZ','SN','GS','QH','NX','XJ'
]
N_PROV = 31

SCENARIOS = ['NDC', 'GM2.0', 'CN2050']
SCENARIO_FILE = {'NDC': 'NDC', 'GM2.0': 'GM2.0', 'CN2050': 'CN2050'}
SSP_LIST = ['ssp245', 'ssp370']
SSP_LABELS = {'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0'}
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

GCM_LIST = ['ACCESS-CM2', 'EC-Earth3', 'GFDL-ESM4', 'MRI-ESM2-0', 'NorESM2-MM']

REGIONS = {
    'North': ['Beijing','Tianjin','Hebei','Shanxi','Inner Mongolia'],
    'Northeast': ['Liaoning','Jilin','Heilongjiang'],
    'East': ['Shanghai','Jiangsu','Zhejiang','Anhui','Fujian','Jiangxi','Shandong'],
    'Central': ['Henan','Hubei','Hunan'],
    'South': ['Guangdong','Guangxi','Hainan'],
    'Southwest': ['Chongqing','Sichuan','Guizhou','Yunnan','Tibet'],
    'Northwest': ['Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang'],
}
PROV_TO_REGION = {p: r for r, ps in REGIONS.items() for p in ps}

FOSSIL_NAT = {
    'NDC': {'coal': 910.0, 'gas': 137.0},
    'GM2.0':   {'coal': 156.0, 'gas': 449.0},
    'CN2050':  {'coal': 0.0,   'gas': 389.0},
}

NUCLEAR_CAP_MW = np.zeros(N_PROV)
NUCLEAR_CAP_MW[5] = 20000; NUCLEAR_CAP_MW[9] = 30000
NUCLEAR_CAP_MW[10] = 30000; NUCLEAR_CAP_MW[12] = 20000
NUCLEAR_CAP_MW[14] = 20000; NUCLEAR_CAP_MW[18] = 40000
NUCLEAR_CAP_MW[19] = 20000; NUCLEAR_CAP_MW[20] = 20000

# ============================================================
# Helper: load supply_demand npz
# ============================================================
def load_sd(cap_scen, ssp):
    f = SCENARIO_FILE[cap_scen]
    return np.load(DATA_DIR / f'supply_demand_2050_{f}_{ssp}.npz', allow_pickle=True)


# ############################################################
#  TABLE 1: table_Fig2a_monthly_generation_demand.xlsx
# ############################################################
def gen_table_fig2a():
    print('[1/13] table_Fig2a ...')
    rows = []
    for sc in SCENARIOS:
        for ssp in SSP_LIST:
            d = load_sd(sc, ssp)
            wind_nat = d['gen_wind'].sum(axis=1)     # (12,)
            solar_nat = d['gen_solar'].sum(axis=1)
            hydro_nat = d['gen_hydro'].sum(axis=1)
            demand_nat = d['monthly_demand'].sum(axis=1)
            for m in range(12):
                rows.append({
                    'Scenario': sc,
                    'SSP': SSP_LABELS[ssp],
                    'Month': MONTH_LABELS[m],
                    'Wind_TWh': round(float(wind_nat[m]), 2),
                    'Solar_TWh': round(float(solar_nat[m]), 2),
                    'Hydro_TWh': round(float(hydro_nat[m]), 2),
                    'Demand_TWh': round(float(demand_nat[m]), 2),
                    'RE_Total_TWh': round(float(wind_nat[m] + solar_nat[m] + hydro_nat[m]), 2),
                })
    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / 'table_Fig2a_monthly_generation_demand.xlsx', index=False)


# ############################################################
#  TABLE 2: table_Fig2b_supply_demand_gap.xlsx
# ############################################################
def gen_table_fig2b():
    print('[2/13] table_Fig2b ...')
    # Sheet 1: National monthly gap
    rows_nat = []
    for sc in SCENARIOS:
        for ssp in SSP_LIST:
            d = load_sd(sc, ssp)
            gen_re = d['gen_wind'] + d['gen_solar'] + d['gen_hydro']
            gap = gen_re - d['monthly_demand']
            gap_nat = gap.sum(axis=1)
            for m in range(12):
                rows_nat.append({
                    'Scenario': sc, 'SSP': SSP_LABELS[ssp],
                    'Month': MONTH_LABELS[m],
                    'National_Gap_TWh': round(float(gap_nat[m]), 2),
                })
    df_nat = pd.DataFrame(rows_nat)

    # Sheet 2: Provincial annual gap
    rows_prov = []
    for sc in SCENARIOS:
        for ssp in SSP_LIST:
            d = load_sd(sc, ssp)
            gen_re = d['gen_wind'] + d['gen_solar'] + d['gen_hydro']
            gap_annual = (gen_re - d['monthly_demand']).sum(axis=0)  # (31,)
            for p in range(N_PROV):
                rows_prov.append({
                    'Scenario': sc, 'SSP': SSP_LABELS[ssp],
                    'Province': PROVINCE_ORDER[p],
                    'Province_short': PROV_SHORT[p],
                    'Region': PROV_TO_REGION[PROVINCE_ORDER[p]],
                    'Annual_Gap_TWh': round(float(gap_annual[p]), 2),
                })
    df_prov = pd.DataFrame(rows_prov)

    with pd.ExcelWriter(OUT_DIR / 'table_Fig2b_supply_demand_gap.xlsx') as w:
        df_nat.to_excel(w, sheet_name='National_Monthly_Gap', index=False)
        df_prov.to_excel(w, sheet_name='Provincial_Annual_Gap', index=False)


# ############################################################
#  TABLE 3: table_Fig3a_LSLW_frequency.xlsx
# ############################################################
def gen_table_fig3a():
    print('[3/13] table_Fig3a ...')
    d = np.load(DATA_DIR / 'lslw_results.npz', allow_pickle=True)
    rows = []
    for p in range(N_PROV):
        rows.append({
            'Province': PROVINCE_ORDER[p],
            'Province_short': PROV_SHORT[p],
            'Region': PROV_TO_REGION[PROVINCE_ORDER[p]],
            'LSLW_Freq_Historical': round(float(d['freq_historical'][p]), 3),
            'LSLW_Freq_SSP245': round(float(d['freq_ssp245'][p]), 3),
            'LSLW_Freq_SSP370': round(float(d['freq_ssp370'][p]), 3),
            'LSLW_Freq_Std_Historical': round(float(d['freq_std_historical'][p]), 3),
            'LSLW_Freq_Std_SSP245': round(float(d['freq_std_ssp245'][p]), 3),
            'LSLW_Freq_Std_SSP370': round(float(d['freq_std_ssp370'][p]), 3),
        })
    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / 'table_Fig3a_LSLW_frequency.xlsx', index=False)


# ############################################################
#  TABLE 4: table_Fig3b_spatial_synchronization.xlsx
# ############################################################
def gen_table_fig3b():
    print('[4/13] table_Fig3b ...')
    d = np.load(DATA_DIR / 'lslw_results.npz', allow_pickle=True)
    rows = []
    for k in range(1, 16):
        h = float(d['sync_dist_historical'][k])
        s2 = float(d['sync_dist_ssp245'][k])
        s3 = float(d['sync_dist_ssp370'][k])
        rows.append({
            'k': k,
            'Sync_DaysPerYear_Historical': round(h, 3),
            'Sync_DaysPerYear_SSP245': round(s2, 3),
            'Sync_DaysPerYear_SSP370': round(s3, 3),
            'Risk_Ratio_SSP370': round(s3 / h, 2) if h > 0 else 'new',
        })
    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / 'table_Fig3b_spatial_synchronization.xlsx', index=False)


# ############################################################
#  TABLE 5: table_Fig4_penetration_by_day_type.xlsx
# ############################################################
def gen_table_fig4():
    print('[5/13] table_Fig4 ...')
    d = np.load(DATA_DIR / 'fig4_penetration_by_day_type.npz', allow_pickle=True)
    FUTURE_YEARS = 25
    LOW_PEN = 50

    # Sheet 1: Panel A — CN2050 penetration distribution stats
    rows_a = []
    for ssp in SSP_LIST:
        pf = f'CN2050_{ssp}'
        pen_nat = d[f'pen_nat_{pf}']
        pen_lslw = d[f'pen_lslw_{pf}']
        dtype = d[f'dtype_{pf}']

        for day_label, mask_val, arr in [
            ('Normal', 0, pen_nat),
            ('Sync_LSLW', 2, pen_lslw),
        ]:
            mask = dtype == mask_val
            vals = arr[mask]
            if len(vals) == 0:
                continue
            rows_a.append({
                'SSP': SSP_LABELS[ssp],
                'Day_Type': day_label,
                'N_days': int(mask.sum()),
                'Mean_Penetration_pct': round(float(vals.mean()), 2),
                'Median_Penetration_pct': round(float(np.median(vals)), 2),
                'Q1_Penetration_pct': round(float(np.percentile(vals, 25)), 2),
                'Q3_Penetration_pct': round(float(np.percentile(vals, 75)), 2),
            })
    df_a = pd.DataFrame(rows_a)

    # Sheet 2: Panel B — low-penetration day frequency (all scenarios)
    rows_b = []
    for sc in SCENARIOS:
        for ssp in SSP_LIST:
            pf = f'{SCENARIO_FILE[sc]}_{ssp}'
            pen = d[f'pen_nat_{pf}']
            n_low = int((pen < LOW_PEN).sum())
            freq_yr = round(n_low / (len(GCM_LIST) * FUTURE_YEARS), 1)
            rows_b.append({
                'Scenario': sc,
                'SSP': SSP_LABELS[ssp],
                'Low_Penetration_Days_Per_Year': freq_yr,
                'Total_Low_Pen_Days': n_low,
            })
    df_b = pd.DataFrame(rows_b)

    with pd.ExcelWriter(OUT_DIR / 'table_Fig4_penetration_by_day_type.xlsx') as w:
        df_a.to_excel(w, sheet_name='PanelA_Penetration_Stats', index=False)
        df_b.to_excel(w, sheet_name='PanelB_Low_Pen_Frequency', index=False)


# ############################################################
#  TABLE 6: table_Fig5a_dispatch_decomposition.xlsx
# ############################################################
def gen_table_fig5a():
    print('[6/13] table_Fig5a ...')
    d = np.load(DATA_DIR / 'provincial_postTX_results.npz', allow_pickle=True)
    rows = []
    for sc in SCENARIOS:
        for ssp in SSP_LIST:
            pf = f'{SCENARIO_FILE[sc]}_{ssp}'
            rows.append({
                'Scenario': sc,
                'SSP': SSP_LABELS[ssp],
                'Deficit_TWh': round(float(d[f'{pf}_deficit_autarky_TWh']), 1),
                'Gas_TWh': round(float(d[f'{pf}_gas_TWh']), 1),
                'Coal_TWh': round(float(d[f'{pf}_coal_TWh']), 1),
                'TX_TWh': round(float(d[f'{pf}_ctx_value_TWh']), 1),
                'Unmet_TWh': round(float(d[f'{pf}_national_unmet_TWh']), 1),
            })
    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / 'table_Fig5a_dispatch_decomposition.xlsx', index=False)


# ############################################################
#  TABLE 7: table_Fig5b_provincial_residual.xlsx
# ############################################################
def gen_table_fig5b():
    print('[7/13] table_Fig5b ...')
    d = np.load(DATA_DIR / 'provincial_postTX_results.npz', allow_pickle=True)
    u_ndc245 = d['NDC_ssp245_prov_unmet_TWh_mean'].astype(float)
    u_gm245  = d['GM2.0_ssp245_prov_unmet_TWh_mean'].astype(float)
    u_cn245  = d['CN2050_ssp245_prov_unmet_TWh_mean'].astype(float)
    u_cn370  = d['CN2050_ssp370_prov_unmet_TWh_mean'].astype(float)

    rows = []
    for p in range(N_PROV):
        rows.append({
            'Province': PROVINCE_ORDER[p],
            'Province_short': PROV_SHORT[p],
            'Region': PROV_TO_REGION[PROVINCE_ORDER[p]],
            'Baseline_NDC_SSP245_TWh': round(float(u_ndc245[p]), 2),
            'Moderate_Decarb_GM_minus_NDC_TWh': round(float(u_gm245[p] - u_ndc245[p]), 2),
            'Deep_Decarb_CN_minus_GM_TWh': round(float(u_cn245[p] - u_gm245[p]), 2),
            'Climate_SSP370_minus_SSP245_TWh': round(float(u_cn370[p] - u_cn245[p]), 2),
            'Total_CN2050_SSP370_TWh': round(float(u_cn370[p]), 2),
        })
    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / 'table_Fig5b_provincial_residual.xlsx', index=False)


# ############################################################
#  TABLE 8: table_Fig6_provincial_typology.xlsx
# ############################################################
def gen_table_fig6():
    print('[8/13] table_Fig6 ...')
    df = pd.read_excel(OUT_DIR / 'table_Fig6_provincial_typology.xlsx')
    df.to_excel(OUT_DIR / 'table_Fig6_provincial_typology.xlsx', index=False)


# ############################################################
#  INPUT 9: input_capacity_scenarios.xlsx
# ############################################################
def gen_input_capacity():
    print('[9/13] input_capacity_scenarios ...')
    rows = []
    for sc in SCENARIOS:
        sf = SCENARIO_FILE[sc]
        # Wind/Solar/Hydro from npz (same for both SSPs, take ssp245)
        d = load_sd(sc, 'ssp245')
        cap_w = d['cap_wind']    # (31,) MW
        cap_s = d['cap_solar']   # (31,) MW
        cap_h = d['cap_hydro']   # (31,) MW

        # Gas/Coal provincial
        if sc == 'CN2050':
            gas_mw = np.zeros(N_PROV)
            coal_mw = np.zeros(N_PROV)
            if MOESM4_PATH.exists():
                df_m = pd.read_excel(MOESM4_PATH, sheet_name='Fig.5', index_col=0)
                gas_cols = [c for c in df_m.columns if 'Gas' in str(c)]
                coal_cols = [c for c in df_m.columns if 'Coal' in str(c)]
                for i, prov in enumerate(PROVINCE_ORDER):
                    if prov in df_m.index:
                        gas_mw[i] = sum(df_m.loc[prov, c] for c in gas_cols)
                        coal_mw[i] = sum(df_m.loc[prov, c] for c in coal_cols)
                gas_source = 'Zhuo MOESM4 Fig.5'
                coal_source = 'Zhuo MOESM4 Fig.5'
            else:
                df_existing = pd.read_excel(INPUT_TABLES_DIR / 'input_capacity_scenarios.xlsx')
                df_existing = df_existing[df_existing['Scenario'] == 'CN2050'].set_index('Province')
                for i, prov in enumerate(PROVINCE_ORDER):
                    gas_mw[i] = float(df_existing.loc[prov, 'Gas_GW']) * 1000
                    coal_mw[i] = float(df_existing.loc[prov, 'Coal_GW']) * 1000
                gas_source = 'Package input_capacity_scenarios.xlsx'
                coal_source = 'Package input_capacity_scenarios.xlsx'
        else:
            annual_demand = load_sd(sc, 'ssp245')['monthly_demand'].sum(axis=0)
            share = annual_demand / annual_demand.sum()
            gas_mw = share * FOSSIL_NAT[sf]['gas'] * 1000
            coal_mw = share * FOSSIL_NAT[sf]['coal'] * 1000
            gas_source = f'National {FOSSIL_NAT[sf]["gas"]} GW allocated by demand share'
            coal_source = f'National {FOSSIL_NAT[sf]["coal"]} GW allocated by demand share'

        for p in range(N_PROV):
            rows.append({
                'Province': PROVINCE_ORDER[p],
                'Province_short': PROV_SHORT[p],
                'Scenario': sc,
                'Wind_GW': round(float(cap_w[p]) / 1000, 3),
                'Solar_GW': round(float(cap_s[p]) / 1000, 3),
                'Hydro_GW': round(float(cap_h[p]) / 1000, 3),
                'Nuclear_GW': round(float(NUCLEAR_CAP_MW[p]) / 1000, 3),
                'Gas_GW': round(float(gas_mw[p]) / 1000, 3),
                'Coal_GW': round(float(coal_mw[p]) / 1000, 3),
                'Gas_Source': gas_source,
                'Coal_Source': coal_source,
            })
    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / 'input_capacity_scenarios.xlsx', index=False)


# ############################################################
#  INPUT 10: input_demand_scenarios.xlsx
# ############################################################
def gen_input_demand():
    print('[10/13] input_demand_scenarios ...')
    # Sheet 1: Annual total
    rows_annual = []
    for sc in SCENARIOS:
        for ssp in SSP_LIST:
            sf = SCENARIO_FILE[sc]
            d = np.load(DATA_DIR / f'demand_2050_{sf}_{ssp}.npz', allow_pickle=True)
            annual = d['annual_total']  # (31,)
            for p in range(N_PROV):
                rows_annual.append({
                    'Province': PROVINCE_ORDER[p],
                    'Province_short': PROV_SHORT[p],
                    'Scenario': sc,
                    'SSP': SSP_LABELS[ssp],
                    'Annual_Demand_TWh': round(float(annual[p]), 2),
                })
    df_annual = pd.DataFrame(rows_annual)

    # Sheet 2: Monthly distribution
    rows_monthly = []
    for sc in SCENARIOS:
        for ssp in SSP_LIST:
            sf = SCENARIO_FILE[sc]
            d = np.load(DATA_DIR / f'demand_2050_{sf}_{ssp}.npz', allow_pickle=True)
            md = d['monthly_demand']  # (12, 31)
            for p in range(N_PROV):
                row = {
                    'Province': PROVINCE_ORDER[p],
                    'Province_short': PROV_SHORT[p],
                    'Scenario': sc,
                    'SSP': SSP_LABELS[ssp],
                }
                for m in range(12):
                    row[MONTH_LABELS[m]] = round(float(md[m, p]), 3)
                rows_monthly.append(row)
    df_monthly = pd.DataFrame(rows_monthly)

    with pd.ExcelWriter(OUT_DIR / 'input_demand_scenarios.xlsx') as w:
        df_annual.to_excel(w, sheet_name='Annual_Total', index=False)
        df_monthly.to_excel(w, sheet_name='Monthly_Distribution', index=False)


# ############################################################
#  INPUT 11: input_transmission_network.xlsx
# ############################################################
def gen_input_transmission():
    print('[11/13] input_transmission_network ...')
    # Sheet 1: UHVDC links. Prefer legacy flow output when available;
    # otherwise use the package-local deposited input table.
    old_flow = EXCEL_DIR / 'flexibility_panelC_flow.xlsx'
    package_tx = INPUT_TABLES_DIR / 'input_transmission_network.xlsx'
    if old_flow.exists():
        df_uhvdc = pd.read_excel(old_flow, sheet_name='UHVDC_Flow')
        uhvdc_cols = ['From', 'From_short', 'To', 'To_short',
                      'From_lon', 'From_lat', 'To_lon', 'To_lat', 'Capacity_GW']
        df_uhvdc_out = df_uhvdc[uhvdc_cols].copy()
    elif package_tx.exists():
        df_uhvdc_out = pd.read_excel(package_tx, sheet_name='UHVDC_Links')
        df_ac = pd.read_excel(package_tx, sheet_name='AC_Region_Grouping')
        with pd.ExcelWriter(OUT_DIR / 'input_transmission_network.xlsx') as w:
            df_uhvdc_out.to_excel(w, sheet_name='UHVDC_Links', index=False)
            df_ac.to_excel(w, sheet_name='AC_Region_Grouping', index=False)
        return
    else:
        raise FileNotFoundError('No package-local or legacy transmission input table found')

    # Sheet 2: AC region grouping (31x31 matrix → find_grid_regions with threshold=500)
    tx_df = pd.read_excel(TX_PATH, sheet_name='channel_capacity',
                          index_col=0, nrows=31)
    tx_cap = tx_df.values.astype(float)

    # Union-Find to determine AC regions
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
            if tx_cap[i, j] >= 500 or tx_cap[j, i] >= 500:
                union(i, j)
    region_ids = [find(i) for i in range(N_PROV)]

    # Map region_id to region number
    unique_ids = sorted(set(region_ids))
    id_to_num = {uid: idx + 1 for idx, uid in enumerate(unique_ids)}

    rows_ac = []
    for p in range(N_PROV):
        rows_ac.append({
            'Province': PROVINCE_ORDER[p],
            'Province_short': PROV_SHORT[p],
            'AC_Region_ID': id_to_num[region_ids[p]],
        })
    df_ac = pd.DataFrame(rows_ac)

    with pd.ExcelWriter(OUT_DIR / 'input_transmission_network.xlsx') as w:
        df_uhvdc_out.to_excel(w, sheet_name='UHVDC_Links', index=False)
        df_ac.to_excel(w, sheet_name='AC_Region_Grouping', index=False)


# ############################################################
#  INPUT 12: input_hydro_climatology.xlsx
# ############################################################
def gen_input_hydro():
    print('[12/13] input_hydro_climatology ...')
    if HYDRO_PATH.suffix.lower() == '.xlsx':
        df = pd.read_excel(HYDRO_PATH)
        df.to_excel(OUT_DIR / 'input_hydro_climatology.xlsx', index=False)
        return
    d = np.load(HYDRO_PATH, allow_pickle=True)
    cf_mean = d['cf_mean']  # (12, 31)
    cf_std = d['cf_std']    # (12, 31)
    provinces = d['provinces']

    rows = []
    for p in range(N_PROV):
        row = {
            'Province': str(provinces[p]),
            'Province_short': PROV_SHORT[p],
        }
        for m in range(12):
            row[f'CF_Mean_{MONTH_LABELS[m]}'] = round(float(cf_mean[m, p]), 4)
            row[f'CF_Std_{MONTH_LABELS[m]}'] = round(float(cf_std[m, p]), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / 'input_hydro_climatology.xlsx', index=False)


# ############################################################
#  INPUT 13: input_temperature_elasticity.xlsx
# ############################################################
def gen_input_temp_elasticity():
    print('[13/13] input_temperature_elasticity ...')
    df = pd.DataFrame([
        {'Parameter': 'T_heat', 'Value': 12.5, 'Unit': '°C',
         'Description': 'Heating threshold: below this, demand increases with cooling'},
        {'Parameter': 'T_cool', 'Value': 19.6, 'Unit': '°C',
         'Description': 'Cooling threshold: above this, demand increases with warming'},
        {'Parameter': 'beta_heat', 'Value': 0.026, 'Unit': '1/°C',
         'Description': 'Heating elasticity: demand increases 2.6% per °C below T_heat'},
        {'Parameter': 'beta_cool', 'Value': 0.035, 'Unit': '1/°C',
         'Description': 'Cooling elasticity: demand increases 3.5% per °C above T_cool'},
        {'Parameter': 'clip_min', 'Value': 0.80, 'Unit': '-',
         'Description': 'Minimum temperature adjustment factor'},
        {'Parameter': 'clip_max', 'Value': 1.30, 'Unit': '-',
         'Description': 'Maximum temperature adjustment factor'},
    ])
    df.to_excel(OUT_DIR / 'input_temperature_elasticity.xlsx', index=False)


# ############################################################
#  MAIN
# ############################################################
if __name__ == '__main__':
    gen_table_fig2a()
    gen_table_fig2b()
    gen_table_fig3a()
    gen_table_fig3b()
    gen_table_fig4()
    gen_table_fig5a()
    gen_table_fig5b()
    gen_table_fig6()
    gen_input_capacity()
    gen_input_demand()
    gen_input_transmission()
    gen_input_hydro()
    gen_input_temp_elasticity()

    print('\n========================================')
    print(f'All 13 Excel files saved to:\n  {OUT_DIR}')
    print('========================================')
