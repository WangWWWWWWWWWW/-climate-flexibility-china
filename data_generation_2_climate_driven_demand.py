"""
step3B_climate_driven_demand.py

Step3B: 气候驱动的未来用电量预测

方法（参考 Zheng Nature 2025）：
1. 读取历史省级月度用电量，建立月度分布模式
2. 用 CMIP6 未来温度计算温度驱动的用电量变化
   - 采暖阈值: 12.5°C → 每降1°C，日用电量+2.6%
   - 制冷阈值: 19.6°C → 每升1°C，日用电量+3.5%
3. 结合 Zhuo 年度总量预测，得到未来省级月度用电量

时间窗口：只做 2050 年
情景：3 装机情景(NDC, GM2.0, CN2050) × 2 气候情景(SSP245, SSP370)

输入：
- 历史月度用电量: 省级用电量_当月值.xlsx (2006-2025, 亿千瓦时)
- Zhuo 年度预测: Zhuo_Load_Table21_22_23.xlsx (2050, TWh)
- CMIP6 temperature input: set LSLW_CMIP6_DIR when rerunning from raw data.
- Grid coordinates: data/processed_results/grid_coords.npz

输出：
- data/demand_2050_[scenario]_[ssp].npz     月度省级用电量 (12, 31)
- data/demand_historical_monthly.npz         历史月度分布模式
- figure/fig_climate_driven_demand.png
- excel/demand_comparison.xlsx
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
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
# Raw CMIP6 data are external; configure with LSLW_CMIP6_DIR if rerunning this step.
ZHUO_FILE = ZHUO_DIR / 'Zhuo_Load_Table21_22_23.xlsx'
GRID_COORDS = GRID_COORDS
# Grid-province map is external; configure with LSLW_GRID_MAP if rerunning this step.

DATA_DIR = DATA_DIR
FIG_DIR = FIGURES_DIR
EXCEL_DIR = SOURCE_DATA_DIR

# ============================================================
# 常量
# ============================================================
GCM_LIST = [
    'ACCESS-CM2', 'BCC-CSM2-MR', 'CNRM-CM6-1', 'EC-Earth3',
    'GFDL-ESM4', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM'
]
SSP_LIST = ['ssp245', 'ssp370']
HIST_YEARS   = list(range(2010, 2015))
FUTURE_YEARS = list(range(2046, 2051))

SCENARIOS = ['NDC', 'GM2.0', 'CN2050']
ZHUO_SHEETS = {
    'NDC': None,
    'GM2.0':   'Table22_GM2.0',
    'CN2050':  'Table23_CN2050'
}


def resolve_zhuo_sheet(scenario):
    if scenario != 'NDC':
        return ZHUO_SHEETS[scenario]
    sheets = pd.ExcelFile(ZHUO_FILE).sheet_names
    matches = [s for s in sheets if s.startswith('Table21_')]
    if not matches:
        raise ValueError('Could not find the Zhuo Table21 demand sheet for NDC')
    return matches[0]

PROVINCE_ORDER = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia',
    'Liaoning', 'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu',
    'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong',
    'Henan', 'Hubei', 'Hunan', 'Guangdong', 'Guangxi',
    'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 'Yunnan',
    'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]

PROVINCE_CN = [
    '北京','天津','河北','山西','内蒙古','辽宁','吉林','黑龙江',
    '上海','江苏','浙江','安徽','福建','江西','山东','河南',
    '湖北','湖南','广东','广西','海南','重庆','四川','贵州',
    '云南','西藏','陕西','甘肃','青海','宁夏','新疆'
]

# 温度驱动参数（Zheng Nature 2025）
T_HEAT = 12.5   # 采暖阈值 (°C)
T_COOL = 19.6   # 制冷阈值 (°C)
HEAT_RATE = 0.026  # 每降1°C，日用电量增加2.6%
COOL_RATE = 0.035  # 每升1°C，日用电量增加3.5%

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_dpm(year):
    dpm = DAYS_PER_MONTH.copy()
    if is_leap(year):
        dpm[1] = 29
    return dpm

# ============================================================
print('=' * 60)
print('Step3B: 气候驱动的未来用电量预测')
print('=' * 60)

# ============================================================
# 1. 读取历史月度用电量，建立月度分布模式
# ============================================================
print('\n[1/5] 读取历史月度用电量...')

df_load = pd.read_excel(LOAD_FILE, header=0)

# 第一行是单位（亿千瓦时），跳过
# 第一列是日期
cn2en = dict(zip(PROVINCE_CN, PROVINCE_ORDER))

# 解析数据
dates = pd.to_datetime(df_load.iloc[1:, 0])
load_data = {}

for pi, prov_en in enumerate(PROVINCE_ORDER):
    col_idx = pi + 1  # 第0列是日期
    vals = pd.to_numeric(df_load.iloc[1:, col_idx], errors='coerce')
    load_data[prov_en] = vals.values

# 转为 DataFrame
load_df = pd.DataFrame(load_data, index=dates)
load_df = load_df.sort_index()

# 取 2015-2020 的数据计算月度分布模式
mask_hist = (load_df.index.year >= 2015) & (load_df.index.year <= 2020)
load_hist = load_df[mask_hist].copy()
load_hist['month'] = load_hist.index.month

# 月度分布因子：每月用电量 / 年总量
# monthly_profile[m, p] = 该月占全年的比例
monthly_profile = np.zeros((12, 31), dtype=np.float64)

for pi, prov in enumerate(PROVINCE_ORDER):
    annual_total = load_hist.groupby(load_hist.index.year)[prov].sum()
    monthly_mean = load_hist.groupby('month')[prov].mean()
    year_mean = annual_total.mean()
    if year_mean > 0:
        monthly_profile[:, pi] = monthly_mean.values / year_mean

print(f'  历史数据范围: {load_hist.index.min().date()} → {load_hist.index.max().date()}')
print(f'  月度分布因子 shape: {monthly_profile.shape}')
print(f'  因子总和检查 (应≈1.0): {monthly_profile.sum(axis=0).mean():.4f}')

# 保存
np.savez_compressed(
    DATA_DIR / 'demand_historical_monthly.npz',
    monthly_profile=monthly_profile,
    provinces=PROVINCE_ORDER
)

# ============================================================
# 2. 读取 Zhuo 年度总量预测 (2050)
# ============================================================
print('\n[2/5] 读取 Zhuo 年度总量预测 (2050)...')

zhuo_2050 = {}  # scenario → (31,) TWh

for scenario in SCENARIOS:
    sheet = resolve_zhuo_sheet(scenario)
    df = pd.read_excel(ZHUO_FILE, sheet_name=sheet, header=0)

    # 第0行是列名（Province Name, 2020, 2025, ..., 2050）
    # 实际数据从第1行开始
    provinces_zhuo = df.iloc[1:, 0].values
    load_2050 = pd.to_numeric(df.iloc[1:, -1], errors='coerce').values  # 最后一列是2050

    # 按 PROVINCE_ORDER 排序
    demand = np.zeros(31, dtype=np.float64)
    for i, pz in enumerate(provinces_zhuo):
        pz_str = str(pz).strip()
        if pz_str in PROVINCE_ORDER:
            idx = PROVINCE_ORDER.index(pz_str)
            demand[idx] = load_2050[i]

    zhuo_2050[scenario] = demand
    print(f'  {scenario}: 全国总量 = {demand.sum():.1f} TWh')

# ============================================================
# 3. 计算 CMIP6 未来温度的省级月均值
# ============================================================
print('\n[3/5] 计算 CMIP6 未来温度 (2046-2050 省级月均)...')

# 加载网格信息
coords = np.load(GRID_COORDS)
era5_lat = coords['era5_lat']
era5_lon = coords['era5_lon']
era5_nlat, era5_nlon = len(era5_lat), len(era5_lon)

grid_map = np.load(GRID_MAP)
province_masks = [grid_map == pi for pi in range(31)]


def regrid_to_era5(data_2d, src_lat, src_lon):
    """将 CMIP6 格点数据重采样到 ERA5 网格"""
    era5_lat_asc = era5_lat[::-1]
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data_2d = data_2d[::-1, :]
    interp = RegularGridInterpolator(
        (src_lat, src_lon), data_2d,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    lon_grid, lat_grid = np.meshgrid(era5_lon, era5_lat_asc)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    result = interp(points).reshape(era5_nlat, era5_nlon)
    return result[::-1].astype(np.float32)


# 同时计算历史期温度（用于对比）
def compute_monthly_temp(gcm, scenario, years):
    """计算省级月均温度 (12, 31)，单位°C"""
    temp_sum = np.zeros((12, 31), dtype=np.float64)
    temp_cnt = np.zeros(12, dtype=np.int32)

    for year in years:
        base = CMIP6_DIR / gcm / scenario
        tas_files = list((base / 'tas').glob(f'tas_day_{gcm}_{scenario}_*_{year}.nc'))
        if not tas_files:
            continue

        ds = xr.open_dataset(tas_files[0])
        tas = ds['tas'].values  # (n_days, lat, lon), K
        c_lat = ds['lat'].values
        c_lon = ds['lon'].values
        ds.close()

        dpm = get_dpm(year)
        day_idx = 0
        for m in range(12):
            nd = dpm[m]
            for d in range(nd):
                if day_idx >= tas.shape[0]:
                    break
                # 重采样到ERA5网格，聚合到省级
                t_grid = regrid_to_era5(tas[day_idx], c_lat, c_lon) - 273.15
                for pi in range(31):
                    mask = province_masks[pi]
                    valid = t_grid[mask]
                    valid = valid[~np.isnan(valid)]
                    if len(valid) > 0:
                        temp_sum[m, pi] += valid.mean()
                temp_cnt[m] += 1
                day_idx += 1

    for m in range(12):
        if temp_cnt[m] > 0:
            temp_sum[m] /= temp_cnt[m]

    return temp_sum.astype(np.float32)


# 计算各GCM的历史期和未来期温度
hist_temp_gcms = []   # list of (12, 31)
fut_temp = {}         # ssp → list of (12, 31)
for ssp in SSP_LIST:
    fut_temp[ssp] = []

for gi, gcm in enumerate(GCM_LIST):
    print(f'  [{gi+1}/8] {gcm}')

    # 历史期温度（缓存）
    hist_cache = DATA_DIR / f'temp_historical_{gcm}.npz'
    if hist_cache.exists():
        d = np.load(hist_cache)
        t_hist = d['temp_monthly']
        print(f'    historical: 已缓存, mean={t_hist.mean():.1f}°C')
    else:
        print(f'    historical:')
        t_hist = compute_monthly_temp(gcm, 'historical', HIST_YEARS)
        np.savez_compressed(hist_cache, temp_monthly=t_hist)
        print(f'    → mean={t_hist.mean():.1f}°C')
    hist_temp_gcms.append(t_hist)

    # 未来期温度（缓存）
    for ssp in SSP_LIST:
        fut_cache = DATA_DIR / f'temp_{ssp}_{gcm}.npz'
        if fut_cache.exists():
            d = np.load(fut_cache)
            t_fut = d['temp_monthly']
            print(f'    {ssp}: 已缓存, mean={t_fut.mean():.1f}°C')
        else:
            print(f'    {ssp}:')
            t_fut = compute_monthly_temp(gcm, ssp, FUTURE_YEARS)
            np.savez_compressed(fut_cache, temp_monthly=t_fut)
            print(f'    → mean={t_fut.mean():.1f}°C')
        fut_temp[ssp].append(t_fut)

# 集合平均
hist_temp_ens = np.mean(hist_temp_gcms, axis=0)  # (12, 31)
fut_temp_ens = {}
for ssp in SSP_LIST:
    fut_temp_ens[ssp] = np.mean(fut_temp[ssp], axis=0)  # (12, 31)

print(f'\n  历史期全国年均温: {hist_temp_ens.mean():.1f}°C')
for ssp in SSP_LIST:
    delta = fut_temp_ens[ssp].mean() - hist_temp_ens.mean()
    print(f'  {ssp} 2050 全国年均温: {fut_temp_ens[ssp].mean():.1f}°C (Δ={delta:+.1f}°C)')

# ============================================================
# 4. 温度驱动的用电量变化 + 合成未来月度用电量
# ============================================================
print('\n[4/5] 计算温度驱动的用电量变化...')

# 对每个 scenario × ssp 组合
demand_results = {}

for scenario in SCENARIOS:
    for ssp in SSP_LIST:
        key = f'{scenario}_{ssp}'

        # Zhuo 预测的年度总量 (31,) TWh
        annual_total = zhuo_2050[scenario]  # (31,)

        # 基准月度用电量 = 年度总量 × 月度分布
        # monthly_base[m, p] = annual_total[p] * monthly_profile[m, p]  (TWh)
        monthly_base = np.zeros((12, 31), dtype=np.float64)
        for p in range(31):
            monthly_base[:, p] = annual_total[p] * monthly_profile[:, p]

        # 温度驱动的修正因子
        # 对比未来温度和历史温度，计算每月每省的用电量变化
        delta_T = fut_temp_ens[ssp] - hist_temp_ens  # (12, 31) °C

        temp_factor = np.ones((12, 31), dtype=np.float64)
        for m in range(12):
            for p in range(31):
                T_fut = fut_temp_ens[ssp][m, p]

                if T_fut < T_HEAT:
                    # 采暖需求增加（未来比历史更冷 → 更多采暖）
                    # 但一般未来更暖，所以采暖需求可能减少
                    dT = delta_T[m, p]  # 正值=变暖
                    temp_factor[m, p] = 1.0 - dT * HEAT_RATE
                elif T_fut > T_COOL:
                    # 制冷需求增加
                    dT = delta_T[m, p]
                    temp_factor[m, p] = 1.0 + dT * COOL_RATE
                # else: 舒适区，温度变化影响小

        # 确保因子合理
        temp_factor = np.clip(temp_factor, 0.8, 1.3)

        # 最终月度用电量 = 基准 × 温度因子
        monthly_demand = monthly_base * temp_factor  # (12, 31) TWh

        demand_results[key] = {
            'monthly_demand': monthly_demand.astype(np.float32),
            'monthly_base': monthly_base.astype(np.float32),
            'temp_factor': temp_factor.astype(np.float32),
            'delta_T': delta_T.astype(np.float32)
        }

        national_base = monthly_base.sum()
        national_final = monthly_demand.sum()
        pct_change = (national_final / national_base - 1) * 100

        print(f'  {key}: 全国 {national_base:.0f} → {national_final:.0f} TWh '
              f'({pct_change:+.2f}%)')

        # 保存
        np.savez_compressed(
            DATA_DIR / f'demand_2050_{scenario.replace("/","_")}_{ssp}.npz',
            monthly_demand=monthly_demand.astype(np.float32),
            monthly_base=monthly_base.astype(np.float32),
            temp_factor=temp_factor.astype(np.float32),
            delta_T=delta_T.astype(np.float32),
            annual_total=annual_total,
            provinces=PROVINCE_ORDER,
            scenario=scenario,
            ssp=ssp
        )

# ============================================================
# 5. 可视化 + Excel 导出
# ============================================================
print('\n[5/5] 生成图表和 Excel...')

month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
months = np.arange(1, 13)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = {'NDC': '#666666', 'GM2.0': '#74add1', 'CN2050': '#d73027'}

# --- 第1列: 温度变化 ---
for si, ssp in enumerate(SSP_LIST):
    ax = axes[si, 0]
    delta = fut_temp_ens[ssp] - hist_temp_ens  # (12, 31)
    national_delta = delta.mean(axis=1)

    # 各省range
    ax.fill_between(months, delta.min(axis=1), delta.max(axis=1),
                    alpha=0.2, color='#d73027')
    ax.plot(months, national_delta, 'o-', color='#d73027', lw=2, ms=6,
            label=f'National mean')
    ax.axhline(y=0, color='k', lw=0.5, ls='--')
    ax.set_xticks(months); ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_ylabel('ΔT (°C)', fontsize=11)
    ax.set_title(f'Temperature Change — {ssp.upper()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# --- 第2列: 气候驱动的用电量变化 (TWh) ---
for si, ssp in enumerate(SSP_LIST):
    ax = axes[si, 1]
    for scenario in SCENARIOS:
        key = f'{scenario}_{ssp}'
        dem = demand_results[key]['monthly_demand']   # (12, 31)
        base = demand_results[key]['monthly_base']    # (12, 31)
        delta_demand = (dem - base).sum(axis=1)       # (12,) TWh
        ax.plot(months, delta_demand, 'o-', color=colors[scenario],
                lw=2, ms=5, label=scenario)

    ax.axhline(y=0, color='k', lw=0.5, ls='--')
    ax.set_xticks(months); ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_ylabel('ΔDemand (TWh)', fontsize=11)
    ax.set_title(f'Climate-Driven Demand Change — {ssp.upper()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# --- 第3列: 全国月度用电量对比 ---
# --- 第3列: 全国月度用电量对比 ---
for si, ssp in enumerate(SSP_LIST):
    ax = axes[si, 2]
    for scenario in SCENARIOS:
        key = f'{scenario}_{ssp}'
        dem = demand_results[key]['monthly_demand']  # (12, 31) TWh
        national = dem.sum(axis=1)  # (12,) TWh
        ax.plot(months, national, 'o-', color=colors[scenario],
                lw=2, ms=5, label=f'{scenario}')

    # 基准（无温度修正）
    ax.set_xticks(months); ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_ylabel('National Demand (TWh)', fontsize=11)
    ax.set_title(f'Monthly Demand 2050 — {ssp.upper()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_climate_driven_demand.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  图已保存: {FIG_DIR / "fig_climate_driven_demand.png"}')

# ---- Excel 导出 ----
with pd.ExcelWriter(EXCEL_DIR / 'demand_comparison.xlsx') as writer:
    for scenario in SCENARIOS:
        for ssp in SSP_LIST:
            key = f'{scenario}_{ssp}'
            dem = demand_results[key]['monthly_demand']

            sheet_name = f'{scenario.replace("/","_")}_{ssp}'
            df = pd.DataFrame(dem, index=month_labels, columns=PROVINCE_ORDER)
            df.index.name = 'Month'
            df.to_excel(writer, sheet_name=sheet_name[:31])  # sheet名最长31字符

    # 温度变化
    for ssp in SSP_LIST:
        delta = fut_temp_ens[ssp] - hist_temp_ens
        df = pd.DataFrame(delta, index=month_labels, columns=PROVINCE_ORDER)
        df.index.name = 'Month'
        df.to_excel(writer, sheet_name=f'DeltaT_{ssp}')

print(f'  Excel 已保存: {EXCEL_DIR / "demand_comparison.xlsx"}')

print('\n' + '=' * 60)
print('Step3B 完成！')
print(f'  数据: {DATA_DIR}')
print(f'  图表: {FIG_DIR}')
print(f'  Excel: {EXCEL_DIR}')
print('下一步: Step3C — 供需匹配分析')
print('=' * 60)
