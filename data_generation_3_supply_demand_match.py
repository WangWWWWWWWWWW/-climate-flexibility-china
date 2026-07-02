"""
step3C_supply_demand_match.py

Step3C: 气候变化下的供需匹配分析

逻辑：
1. 读取装机容量（Zhuo分情景全国总量 × CN2050省级分布比例）
2. 读取未来CF（Step3A3日度QDM校正 → 聚合为月度，5 GCM均值）
3. 读取未来用电量（Step3B输出，6种情景）
4. 读取水电CF（历史气候态，假设不变）
5. 计算：可再生能源发电 = 装机 × CF × 小时数
6. 分析：渗透率、供需缺口、省级分布

情景矩阵：3装机情景(NDC, GM2.0, CN2050) × 2气候情景(SSP245, SSP370) = 6种
装机容量：三个情景不同！来自Zhuo Nature Communications 2022

输入：
- 装机全国总量: Zhuo SFigure_9_data.xlsx (NDC/GM2.0/CN2050)
- 装机省级分布: Zhuo Figure_data_5.xlsx (CN2050, 31省)
- 风光CF: Step3A3 → CMIP6_daily_CF_corrected_{period}_{gcm}.npz → 聚合月度
- 用电量: Step3B → demand_2050_[scenario]_[ssp].npz (12, 31)
- 水电CF: hydro_monthly_climatology.npz (12, 31)

输出：
- data/supply_demand_2050_[scenario]_[ssp].npz
- figure/fig_renewable_generation.png
- figure/fig_penetration_rate.png
- figure/fig_supply_demand_gap.png
- excel/supply_demand_results.xlsx
"""

import numpy as np
import pandas as pd
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

# ============================================================
# 路径设置
# ============================================================
ZHUO_CAP_NATIONAL = ZHUO_DIR / 'SFigure_9_data.xlsx'
ZHUO_CAP_PROVINCE = ZHUO_DIR / 'Figure_data_5.xlsx'
HYDRO_CF = HYDRO_PATH

DATA_DIR = DATA_DIR
FIG_DIR = FIGURES_DIR
EXCEL_DIR = SOURCE_DATA_DIR

# ============================================================
# 常量
# ============================================================
SSP_LIST = ['ssp245', 'ssp370']
SCENARIOS = ['NDC', 'GM2.0', 'CN2050']

PROVINCE_ORDER = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia',
    'Liaoning', 'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu',
    'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong',
    'Henan', 'Hubei', 'Hunan', 'Guangdong', 'Guangxi',
    'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 'Yunnan',
    'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
HOURS_PER_MONTH = [d * 24 for d in DAYS_PER_MONTH]

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

COLORS_SCENARIO = {'NDC': '#666666', 'GM2.0': '#74add1', 'CN2050': '#d73027'}

print('=' * 60)
print('Step3C: 气候变化下的供需匹配分析')
print('=' * 60)

# ============================================================
# 1. 读取分情景装机容量
# ============================================================
print('\n[1/5] 读取分情景装机容量...')

# --- 1a. 读取 Zhuo 全国总量 (SFigure_9_data.xlsx) ---
df_s9 = pd.read_excel(ZHUO_CAP_NATIONAL, sheet_name='SFig.9', header=0)

# 2050 data columns used here: NDC, GM2.0 and CN2050.
# 行0=Hydro, 1=PV utility, 2=PV distributed, 3=Wind onshore, 4=Wind offshore
national_cap = {}  # scenario → {'wind': GW, 'solar': GW, 'hydro': GW}

scenario_cols = {'NDC': 37, 'GM2.0': 38, 'CN2050': 39}

for name, col in scenario_cols.items():
    hydro     = float(df_s9.iloc[0, col])   # GW
    solar_u   = float(df_s9.iloc[1, col])
    solar_d   = float(df_s9.iloc[2, col])
    wind_on   = float(df_s9.iloc[3, col])
    wind_off  = float(df_s9.iloc[4, col])

    national_cap[name] = {
        'wind':  wind_on + wind_off,
        'solar': solar_u + solar_d,
        'hydro': hydro
    }

# The package baseline pathway is NDC.

print('  全国装机总量 (GW):')
print(f'  {"":12s} {"Wind":>10s} {"Solar":>10s} {"Hydro":>10s} {"Total":>10s}')
for sc in SCENARIOS:
    c = national_cap[sc]
    print(f'  {sc:12s} {c["wind"]:10.0f} {c["solar"]:10.0f} {c["hydro"]:10.0f} '
          f'{c["wind"]+c["solar"]+c["hydro"]:10.0f}')

# --- 1b. 读取 CN2050 省级装机 (Figure_data_5.xlsx) ---
df_f5 = pd.read_excel(ZHUO_CAP_PROVINCE, sheet_name='Fig.5', header=0)

# 列: Province Name, Hydro, Utility PV, Distributed PV, Onshore wind, Offshore wind, ...
# 单位: MW
cn50_provincial = {}  # province → {'wind': MW, 'solar': MW, 'hydro': MW}

for i in range(31):
    prov = str(df_f5.iloc[i, 0]).strip()
    hydro   = float(df_f5.iloc[i, 1])    # Hydro MW
    solar_u = float(df_f5.iloc[i, 2])    # Utility PV MW
    solar_d = float(df_f5.iloc[i, 3])    # Distributed PV MW
    wind_on = float(df_f5.iloc[i, 4])    # Onshore wind MW
    wind_off = float(df_f5.iloc[i, 5])   # Offshore wind MW

    cn50_provincial[prov] = {
        'wind':  wind_on + wind_off,
        'solar': solar_u + solar_d,
        'hydro': hydro
    }

# --- 1c. 计算省级分布比例，缩放各情景 ---
# CN2050 全国总量 (MW)
cn50_total_wind  = sum(v['wind']  for v in cn50_provincial.values())
cn50_total_solar = sum(v['solar'] for v in cn50_provincial.values())
cn50_total_hydro = sum(v['hydro'] for v in cn50_provincial.values())

# 各情景的省级装机 (MW): scenario → (31,) for wind/solar/hydro
cap_by_scenario = {}

for sc in SCENARIOS:
    cap_wind  = np.zeros(31)
    cap_solar = np.zeros(31)
    cap_hydro = np.zeros(31)

    target_wind  = national_cap[sc]['wind']  * 1000  # GW → MW
    target_solar = national_cap[sc]['solar'] * 1000
    target_hydro = national_cap[sc]['hydro'] * 1000

    for pi, prov in enumerate(PROVINCE_ORDER):
        if prov in cn50_provincial:
            p = cn50_provincial[prov]
            # 省级比例 × 目标全国总量
            cap_wind[pi]  = (p['wind']  / cn50_total_wind  * target_wind)  if cn50_total_wind  > 0 else 0
            cap_solar[pi] = (p['solar'] / cn50_total_solar * target_solar) if cn50_total_solar > 0 else 0
            cap_hydro[pi] = (p['hydro'] / cn50_total_hydro * target_hydro) if cn50_total_hydro > 0 else 0

    cap_by_scenario[sc] = {
        'wind':  cap_wind,   # (31,) MW
        'solar': cap_solar,
        'hydro': cap_hydro
    }

    print(f'\n  {sc} 省级装机 (GW):')
    print(f'    风电: {cap_wind.sum()/1000:.0f}, 光伏: {cap_solar.sum()/1000:.0f}, '
          f'水电: {cap_hydro.sum()/1000:.0f}')
    # 显示前3大省
    top3_wind = np.argsort(cap_wind)[::-1][:3]
    print(f'    风电前3: {", ".join(f"{PROVINCE_ORDER[i]}={cap_wind[i]/1000:.0f}GW" for i in top3_wind)}')

# ============================================================
# 2. 读取 CF 数据（从 step3A3 日度QDM校正结果聚合为月度）
# ============================================================
print('\n[2/5] 读取 CF 数据（step3A3 daily → monthly aggregation）...')

GCM_LIST = [
    'ACCESS-CM2', 'EC-Earth3', 'GFDL-ESM4', 'MRI-ESM2-0', 'NorESM2-MM'
]

cf_data = {}
for ssp in SSP_LIST:
    # 逐GCM读取日度CF，聚合为月度，再取multi-model mean
    gcm_wind_monthly = []   # list of (12, 31) per GCM
    gcm_solar_monthly = []

    for gcm in GCM_LIST:
        fpath = DATA_DIR / f'CMIP6_daily_CF_corrected_{ssp}_{gcm}.npz'
        d = np.load(fpath)
        cf_w = d['cf_wind']     # (n_days, 31)
        cf_s = d['cf_solar']    # (n_days, 31)
        months_d = d['months']  # (n_days,) 0-indexed

        # Daily → monthly mean
        wind_monthly = np.zeros((12, 31))
        solar_monthly = np.zeros((12, 31))
        for m in range(12):
            mm = months_d == m
            if mm.sum() > 0:
                wind_monthly[m] = cf_w[mm].mean(axis=0)
                solar_monthly[m] = cf_s[mm].mean(axis=0)

        gcm_wind_monthly.append(wind_monthly)
        gcm_solar_monthly.append(solar_monthly)

    # Multi-model mean (5 GCMs)
    gcm_wind_arr = np.array(gcm_wind_monthly)    # (5, 12, 31)
    gcm_solar_arr = np.array(gcm_solar_monthly)  # (5, 12, 31)

    cf_data[ssp] = {
        'wind': gcm_wind_arr.mean(axis=0),        # (12, 31)
        'solar': gcm_solar_arr.mean(axis=0),       # (12, 31)
        'wind_gcms': gcm_wind_arr,                 # (5, 12, 31)
        'solar_gcms': gcm_solar_arr
    }
    print(f'  {ssp}: wind_CF={cf_data[ssp]["wind"].mean():.4f}, '
          f'solar_CF={cf_data[ssp]["solar"].mean():.4f} '
          f'({len(GCM_LIST)} GCMs)')

# Also compute historical CF for reference
hist_wind_monthly_gcms = []
hist_solar_monthly_gcms = []
for gcm in GCM_LIST:
    fpath = DATA_DIR / f'CMIP6_daily_CF_corrected_historical_{gcm}.npz'
    d = np.load(fpath)
    cf_w = d['cf_wind']
    cf_s = d['cf_solar']
    months_d = d['months']
    wind_m = np.zeros((12, 31))
    solar_m = np.zeros((12, 31))
    for m in range(12):
        mm = months_d == m
        if mm.sum() > 0:
            wind_m[m] = cf_w[mm].mean(axis=0)
            solar_m[m] = cf_s[mm].mean(axis=0)
    hist_wind_monthly_gcms.append(wind_m)
    hist_solar_monthly_gcms.append(solar_m)

hist_wind_mean = np.mean(hist_wind_monthly_gcms, axis=0)  # (12, 31)
hist_solar_mean = np.mean(hist_solar_monthly_gcms, axis=0)
print(f'  historical: wind_CF={hist_wind_mean.mean():.4f}, '
      f'solar_CF={hist_solar_mean.mean():.4f}')

# 水电CF（历史气候态，假设不变）
hydro_data = np.load(HYDRO_CF)
cf_hydro = hydro_data['cf_mean']  # (12, 31)
print(f'  水电 CF: mean={cf_hydro.mean():.4f}')

# ============================================================
# 3. 计算供需匹配（6种情景）
# ============================================================
print('\n[3/5] 计算供需匹配 (6种情景)...')

results = {}

for scenario in SCENARIOS:
    # 该情景的装机容量
    cap = cap_by_scenario[scenario]
    cap_wind  = cap['wind']    # (31,) MW
    cap_solar = cap['solar']
    cap_hydro = cap['hydro']

    for ssp in SSP_LIST:
        key = f'{scenario}_{ssp}'
        file_key = f'{scenario.replace("/","_")}_{ssp}'

        # 读取用电量
        dem_data = np.load(DATA_DIR / f'demand_2050_{file_key}.npz')
        monthly_demand = dem_data['monthly_demand']  # (12, 31) TWh

        # 读取CF
        cf_wind  = cf_data[ssp]['wind']    # (12, 31)
        cf_solar = cf_data[ssp]['solar']

        # 月发电量 = 装机(MW) × CF × 月小时数 / 1e6 → TWh
        gen_wind  = np.zeros((12, 31))
        gen_solar = np.zeros((12, 31))
        gen_hydro = np.zeros((12, 31))

        for m in range(12):
            gen_wind[m]  = cap_wind  * cf_wind[m]  * HOURS_PER_MONTH[m] / 1e6
            gen_solar[m] = cap_solar * cf_solar[m] * HOURS_PER_MONTH[m] / 1e6
            gen_hydro[m] = cap_hydro * cf_hydro[m] * HOURS_PER_MONTH[m] / 1e6

        gen_renewable = gen_wind + gen_solar + gen_hydro  # (12, 31) TWh

        # 供需缺口 = 可再生能源 - 总需求（负值=需要火电/核电补充）
        gap = gen_renewable - monthly_demand  # (12, 31) TWh

        # 可再生能源渗透率
        penetration_monthly = np.where(
            monthly_demand > 0,
            gen_renewable / monthly_demand * 100,
            0
        )  # (12, 31) %

        # 全国年度汇总
        total_gen_re = gen_renewable.sum()
        total_demand = monthly_demand.sum()
        total_penetration = total_gen_re / total_demand * 100

        results[key] = {
            'gen_wind': gen_wind.astype(np.float32),
            'gen_solar': gen_solar.astype(np.float32),
            'gen_hydro': gen_hydro.astype(np.float32),
            'gen_renewable': gen_renewable.astype(np.float32),
            'monthly_demand': monthly_demand.astype(np.float32),
            'gap': gap.astype(np.float32),
            'penetration_monthly': penetration_monthly.astype(np.float32),
            'total_penetration': total_penetration
        }

        # 保存
        np.savez_compressed(
            DATA_DIR / f'supply_demand_2050_{file_key}.npz',
            **results[key],
            cap_wind=cap_wind,
            cap_solar=cap_solar,
            cap_hydro=cap_hydro,
            provinces=PROVINCE_ORDER,
            scenario=scenario,
            ssp=ssp
        )

        print(f'  {key}: RE={total_gen_re:.0f} TWh, Demand={total_demand:.0f} TWh, '
              f'Penetration={total_penetration:.1f}%')

# ============================================================
# 4. 可视化
# ============================================================
print('\n[4/5] 生成图表...')
months = np.arange(1, 13)

# ---- Fig1: 可再生能源发电量 vs 需求（2×3矩阵）----
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for si, ssp in enumerate(SSP_LIST):
    for sci, scenario in enumerate(SCENARIOS):
        ax = axes[si, sci]
        key = f'{scenario}_{ssp}'
        r = results[key]

        wind_nat  = r['gen_wind'].sum(axis=1)
        solar_nat = r['gen_solar'].sum(axis=1)
        hydro_nat = r['gen_hydro'].sum(axis=1)
        demand_nat = r['monthly_demand'].sum(axis=1)

        ax.stackplot(months, hydro_nat, wind_nat, solar_nat,
                     colors=['#2166ac', '#4393c3', '#f4a582'],
                     labels=['Hydro', 'Wind', 'Solar'], alpha=0.8)
        ax.plot(months, demand_nat, 'k-', lw=2.5, label='Demand')

        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.set_ylabel('TWh', fontsize=10)
        ax.set_title(f'{scenario} — {ssp.upper()}', fontsize=11, fontweight='bold')
        if si == 0 and sci == 0:
            ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

fig.suptitle('Monthly Renewable Generation vs Demand (2050)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_renewable_generation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  已保存: fig_renewable_generation.png')

# ---- Fig2: 可再生能源渗透率 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for si, ssp in enumerate(SSP_LIST):
    ax = axes[si]
    for scenario in SCENARIOS:
        key = f'{scenario}_{ssp}'
        r = results[key]
        nat_re = r['gen_renewable'].sum(axis=1)
        nat_dem = r['monthly_demand'].sum(axis=1)
        nat_pen = nat_re / nat_dem * 100

        ax.plot(months, nat_pen, 'o-', color=COLORS_SCENARIO[scenario],
                lw=2, ms=6, label=f'{scenario} ({r["total_penetration"]:.1f}%)')

    ax.axhline(y=100, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_ylabel('Renewable Penetration (%)', fontsize=11)
    ax.set_title(f'Penetration Rate — {ssp.upper()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_penetration_rate.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  已保存: fig_penetration_rate.png')

# ---- Fig3: 供需缺口（全国月度 + 省级年度）----
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 上排：全国月度缺口（SSP245, SSP370, 气候情景对比）
for si, ssp in enumerate(SSP_LIST):
    ax = fig.add_subplot(gs[0, si])
    for scenario in SCENARIOS:
        key = f'{scenario}_{ssp}'
        gap_nat = results[key]['gap'].sum(axis=1)
        ax.plot(months, gap_nat, 'o-', color=COLORS_SCENARIO[scenario],
                lw=2, ms=5, label=scenario)

    ax.axhline(y=0, color='k', lw=1)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_ylabel('Gap (TWh)', fontsize=11)
    ax.set_title(f'National Supply-Demand Gap — {ssp.upper()}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 右上：SSP245 vs SSP370 对比（CN2050情景）
ax = fig.add_subplot(gs[0, 2])
for ssp in SSP_LIST:
    key = f'CN2050_{ssp}'
    gap_nat = results[key]['gap'].sum(axis=1)
    ax.plot(months, gap_nat, 'o-', lw=2, ms=5, label=ssp.upper())
ax.axhline(y=0, color='k', lw=1)
ax.set_xticks(months)
ax.set_xticklabels(MONTH_LABELS, fontsize=9)
ax.set_ylabel('Gap (TWh)', fontsize=11)
ax.set_title('SSP245 vs SSP370 (CN2050)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 下排：省级年度缺口（NDC, CN2050, 对比）
for si, (scenario, ssp) in enumerate([
    ('NDC', 'ssp245'), ('CN2050', 'ssp245'), ('CN2050', 'ssp370')
]):
    ax = fig.add_subplot(gs[1, si])
    key = f'{scenario}_{ssp}'
    gap_annual = results[key]['gap'].sum(axis=0)  # (31,) TWh

    colors_bar = ['#d73027' if g < 0 else '#4393c3' for g in gap_annual]
    ax.barh(range(31), gap_annual, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(31))
    ax.set_yticklabels(PROVINCE_ORDER, fontsize=7)
    ax.axvline(x=0, color='k', lw=0.5)
    ax.set_xlabel('Annual Gap (TWh)', fontsize=10)
    ax.set_title(f'Provincial Gap — {scenario} {ssp.upper()}',
                 fontsize=11, fontweight='bold')
    ax.invert_yaxis()

fig.savefig(FIG_DIR / 'fig_supply_demand_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  已保存: fig_supply_demand_gap.png')

# ============================================================
# 5. Excel 导出
# ============================================================
print('\n[5/5] 导出 Excel...')

with pd.ExcelWriter(EXCEL_DIR / 'supply_demand_results.xlsx') as writer:
    # 汇总表
    summary_rows = []
    for scenario in SCENARIOS:
        for ssp in SSP_LIST:
            key = f'{scenario}_{ssp}'
            r = results[key]
            c = cap_by_scenario[scenario]
            summary_rows.append({
                'Scenario': scenario,
                'Climate': ssp,
                'Cap_Wind_GW': c['wind'].sum() / 1000,
                'Cap_Solar_GW': c['solar'].sum() / 1000,
                'Cap_Hydro_GW': c['hydro'].sum() / 1000,
                'Wind_Gen_TWh': r['gen_wind'].sum(),
                'Solar_Gen_TWh': r['gen_solar'].sum(),
                'Hydro_Gen_TWh': r['gen_hydro'].sum(),
                'RE_Total_TWh': r['gen_renewable'].sum(),
                'Demand_TWh': r['monthly_demand'].sum(),
                'Gap_TWh': r['gap'].sum(),
                'Penetration_pct': r['total_penetration']
            })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    # 各情景省级详细
    for scenario in SCENARIOS:
        for ssp in SSP_LIST:
            key = f'{scenario}_{ssp}'
            file_key = f'{scenario.replace("/","_")}_{ssp}'
            r = results[key]
            c = cap_by_scenario[scenario]

            df = pd.DataFrame({
                'Province': PROVINCE_ORDER,
                'Cap_Wind_GW': c['wind'] / 1000,
                'Cap_Solar_GW': c['solar'] / 1000,
                'Cap_Hydro_GW': c['hydro'] / 1000,
                'Wind_TWh': r['gen_wind'].sum(axis=0),
                'Solar_TWh': r['gen_solar'].sum(axis=0),
                'Hydro_TWh': r['gen_hydro'].sum(axis=0),
                'RE_Total_TWh': r['gen_renewable'].sum(axis=0),
                'Demand_TWh': r['monthly_demand'].sum(axis=0),
                'Gap_TWh': r['gap'].sum(axis=0),
                'Penetration_pct': np.where(
                    r['monthly_demand'].sum(axis=0) > 0,
                    r['gen_renewable'].sum(axis=0) / r['monthly_demand'].sum(axis=0) * 100,
                    0
                )
            })
            sheet = f'{file_key}'[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

print(f'  已保存: {EXCEL_DIR / "supply_demand_results.xlsx"}')

print('\n' + '=' * 60)
print('Step3C 完成！')
print(f'  数据: {DATA_DIR}')
print(f'  图表: {FIG_DIR}')
print(f'  Excel: {EXCEL_DIR}')
print('下一步: Step3D — 输电优化')
print('=' * 60)
