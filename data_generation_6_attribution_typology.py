# -*- coding: utf-8 -*-
"""
step3F_attribution_typology.py
==============================

Additional Step 3 analysis:

1. High-emissions sensitivity attribution:
   Decompose SSP3-7.0 minus SSP2-4.5 residual flexibility differences into
   wind, solar, demand, and nonlinear interaction channels for all three
   capacity pathways.

2. Provincial flexibility typology:
   Convert CN2050 provincial residual unmet demand into planning-relevant
   types using annual residual demand, high-emissions sensitivity, daily peak
   burden, and long-duration event metrics.

This script is intentionally independent of step3E_provincial_storage.py.
It imports and reuses the existing dispatch functions, but does not overwrite
existing step3E outputs.

Outputs:
  - data/step3F_attribution_results.csv
  - data/step3F_attribution_per_gcm.csv
  - data/step3F_provincial_typology.csv
  - data/step3F_core_daily_unmet_CN2050.npz
  - excel/step3F_attribution_typology.xlsx
  - figure_new/fig_step3F_attribution_stacked.png
  - figure_new/fig_step3F_provincial_typology_scatter.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from path_config import (
    BASE_DIR, DATA_DIR, FIG_DIR, OUT_DIR, EXCEL_DIR, SOURCE_DATA_DIR,
    INPUT_TABLES_DIR, FIGURES_DIR, CODE_DIR, ERA5_HOURLY, ERA5_DAILY_RSDS,
    CMIP6_DIR, STEP1_DATA, ZHUO_DIR, HYDRO_PATH, TX_PATH, MOESM4_PATH,
    LOAD_FILE, GRID_COORDS, GRID_MAP, ensure_output_dirs,
)
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Make sure the current code directory is importable when launched elsewhere.
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    import step3E_provincial_storage as base
except ModuleNotFoundError:
    import data_generation_5_provincial_dispatch as base


# ---------------------------------------------------------------------
# Constants and labels
# ---------------------------------------------------------------------

BASE_DIR = base.BASE_DIR
DATA_DIR = base.DATA_DIR
FIG_DIR = base.FIG_DIR
EXCEL_DIR = base.EXCEL_DIR

GCM_LIST = base.GCM_LIST
CAP_SCENARIOS = base.CAP_SCENARIOS
PROVINCES = base.PROVINCE_ORDER
PROV_SHORT = base.PROV_SHORT
N_GCM = base.N_GCM
N_PROV = base.N_PROV

CASE_DEFS = {
    "SSP245_all": {
        "wind_ssp": "ssp245",
        "solar_ssp": "ssp245",
        "demand_ssp": "ssp245",
        "label": "SSP2-4.5 all",
    },
    "SSP370_all": {
        "wind_ssp": "ssp370",
        "solar_ssp": "ssp370",
        "demand_ssp": "ssp370",
        "label": "SSP3-7.0 all",
    },
    "wind370_only": {
        "wind_ssp": "ssp370",
        "solar_ssp": "ssp245",
        "demand_ssp": "ssp245",
        "label": "Wind",
    },
    "solar370_only": {
        "wind_ssp": "ssp245",
        "solar_ssp": "ssp370",
        "demand_ssp": "ssp245",
        "label": "Solar",
    },
    "demand370_only": {
        "wind_ssp": "ssp245",
        "solar_ssp": "ssp245",
        "demand_ssp": "ssp370",
        "label": "Demand",
    },
}

ATTR_COMPONENTS = ["wind", "solar", "demand", "interaction"]
ATTR_COLORS = {
    "wind": "#4E79A7",
    "solar": "#E6A23C",
    "demand": "#59A14F",
    "interaction": "#8E6C8A",
}

TYPE_COLORS = {
    "Compound pressure": "#b2182b",
    "Structural flexibility burden": "#ef8a62",
    "Climate-sensitive": "#4393c3",
    "Lower-priority quadrant": "#c9c9c9",
}

TYPE_ORDER = [
    "Compound pressure",
    "Structural flexibility burden",
    "Climate-sensitive",
    "Lower-priority quadrant",
]

TYPE_DESCRIPTIONS = {
    "Compound pressure": "high residual demand and strong high-emissions sensitivity",
    "Structural flexibility burden": "high residual demand, lower high-emissions sensitivity",
    "Climate-sensitive": "strong high-emissions sensitivity, lower residual demand",
    "Lower-priority quadrant": "lower residual demand and lower high-emissions sensitivity",
}

TYPE_DISPLAY_LABELS = {
    "Compound pressure": "Compound pressure",
    "Structural flexibility burden": "Structural flexibility burden",
    "Climate-sensitive": "High-emissions sensitive",
    "Lower-priority quadrant": "Lower near-term priority",
}


# ---------------------------------------------------------------------
# Counterfactual dispatch
# ---------------------------------------------------------------------

def _check_month_alignment(months_a: np.ndarray, months_b: np.ndarray, name: str) -> None:
    if len(months_a) != len(months_b) or not np.array_equal(months_a, months_b):
        raise ValueError(f"Month vectors are not aligned for {name}.")


def load_fixed_capacity_data(cap_scen: str) -> dict[str, np.ndarray]:
    """Load pathway capacity once, independent of climate SSP.

    The capacity pathway is conceptually separate from the climate channel.
    In the current step3C files, cap_wind/cap_solar/cap_hydro are expected to
    be identical between SSP2-4.5 and SSP3-7.0. We use the SSP2-4.5 file as
    the fixed pathway capacity source to avoid mixing capacity changes into
    the attribution.
    """
    d245 = base.load_scenario_data(cap_scen, "ssp245")
    d370 = base.load_scenario_data(cap_scen, "ssp370")
    for k in ["cap_wind", "cap_solar", "cap_hydro"]:
        if not np.allclose(d245[k], d370[k], rtol=1e-5, atol=1e-5):
            print(f"WARNING: {cap_scen} {k} differs between SSP files; using SSP245 capacity.")
    return {k: d245[k] for k in ["cap_wind", "cap_solar", "cap_hydro"]}


def compute_re_residual_counterfactual(
    cap_scen: str,
    wind_ssp: str,
    solar_ssp: str,
    demand_ssp: str,
    gcm: str,
    hydro_cf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute daily residual load with independently chosen climate channels.

    Parameters
    ----------
    cap_scen:
        Capacity pathway: NDC, GM2.0, or CN2050.
    wind_ssp, solar_ssp, demand_ssp:
        Climate source for wind CF, solar CF, and temperature-adjusted demand.
    gcm:
        GCM name.
    hydro_cf:
        Monthly hydropower capacity factor climatology.

    Returns
    -------
    residual, cf_wind, cf_solar, months
        residual is daily provincial residual load before fossil dispatch (GW).
    """
    cf_wind, _, months_w = base.load_daily_cf(wind_ssp, gcm)
    _, cf_solar, months_s = base.load_daily_cf(solar_ssp, gcm)
    _check_month_alignment(months_w, months_s, f"{wind_ssp}/{solar_ssp}/{gcm}")

    demand_data = base.load_scenario_data(cap_scen, demand_ssp)
    cap_data = load_fixed_capacity_data(cap_scen)

    n_days = len(months_w)
    demand = np.zeros((n_days, N_PROV), dtype=np.float64)
    hydro_d = np.zeros((n_days, N_PROV), dtype=np.float64)
    for i, m in enumerate(months_w):
        demand[i] = demand_data["monthly_demand"][m] / base.DAYS_PER_MONTH[m] * 1000 / 24
        hydro_d[i] = hydro_cf[m]

    gen_wind = cf_wind * cap_data["cap_wind"][None, :] / 1000
    gen_solar = cf_solar * cap_data["cap_solar"][None, :] / 1000
    gen_hydro = hydro_d * cap_data["cap_hydro"][None, :] / 1000

    residual = demand - gen_wind - gen_solar - gen_hydro - base.NUCLEAR_GEN_GW[None, :]
    return residual, cf_wind, cf_solar, months_w


def run_dispatch_counterfactual(
    cap_scen: str,
    case_name: str,
    gcm: str,
    hydro_cf: np.ndarray,
    regions: np.ndarray,
    inter_links: np.ndarray,
    region_map: dict,
    unique_regions: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Run four-layer dispatch for one counterfactual case."""
    cfg = CASE_DEFS[case_name]
    residual, cf_w, cf_s, months = compute_re_residual_counterfactual(
        cap_scen=cap_scen,
        wind_ssp=cfg["wind_ssp"],
        solar_ssp=cfg["solar_ssp"],
        demand_ssp=cfg["demand_ssp"],
        gcm=gcm,
        hydro_cf=hydro_cf,
    )

    n_days = len(months)
    n_years = n_days / 365.0
    to_TWh_yr = 24 / 1000 / n_years

    # Fossil capacity allocation follows the demand SSP in the counterfactual.
    # This preserves the original SSP245_all and SSP370_all model behavior
    # while keeping wind/solar-only cases anchored to SSP2-4.5 demand.
    gas_MW, coal_MW = base.get_provincial_fossil(cap_scen, cfg["demand_ssp"])
    gas_GW = gas_MW / 1000
    coal_GW = coal_MW / 1000

    post_fossil, gas_used, coal_used = base.provincial_fossil_dispatch(
        residual, gas_GW, coal_GW
    )
    nat_deficit, prov_post_tx = base.constrained_rebalance_provincial(
        post_fossil, regions, inter_links, region_map, unique_regions
    )

    deficit_autarky = np.maximum(residual, 0).sum(axis=1)
    deficit_post_fossil = np.maximum(post_fossil, 0).sum(axis=1)
    gas_total = gas_used.sum(axis=1)
    coal_total = coal_used.sum(axis=1)
    tx_value = deficit_post_fossil - nat_deficit

    return {
        "national_unmet_TWh": float(nat_deficit.sum() * to_TWh_yr),
        "prov_unmet_TWh": (prov_post_tx.sum(axis=0) * to_TWh_yr).astype(np.float32),
        "deficit_autarky_TWh": float(deficit_autarky.sum() * to_TWh_yr),
        "gas_TWh": float(gas_total.sum() * to_TWh_yr),
        "coal_TWh": float(coal_total.sum() * to_TWh_yr),
        "tx_value_TWh": float(tx_value.sum() * to_TWh_yr),
        "prov_post_tx_daily": prov_post_tx.astype(np.float32),
        "months": months,
        "cf_w": cf_w,
        "cf_s": cf_s,
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def event_metrics_for_series(u_gw: np.ndarray) -> tuple[float, float]:
    """Return max event energy (TWh) and max duration (days) for U>0 runs."""
    positive = u_gw > 1e-6
    max_energy = 0.0
    max_duration = 0
    i = 0
    n = len(u_gw)
    while i < n:
        if not positive[i]:
            i += 1
            continue
        j = i
        while j < n and positive[j]:
            j += 1
        event = u_gw[i:j]
        energy = float(event.sum() * 24 / 1000)  # TWh over the event, not annualized
        duration = int(j - i)
        max_energy = max(max_energy, energy)
        max_duration = max(max_duration, duration)
        i = j
    return max_energy, float(max_duration)


def provincial_metrics_from_daily(
    daily_245: list[np.ndarray],
    daily_370: list[np.ndarray],
) -> pd.DataFrame:
    """Compute CN2050 provincial typology metrics from daily arrays.

    daily_245 and daily_370 are lists of arrays with shape (n_days, N_PROV).
    Some CMIP6 models have 9,125 days and others have 9,132 days in the
    processed 2036--2060 windows, so metrics are computed GCM by GCM rather
    than by stacking the daily arrays.
    """
    n_gcm = len(daily_370)

    rows = []
    for p, prov in enumerate(PROVINCES):
        annual245_g, annual370_g = [], []
        p95_pos, p99_pos, max_daily, max_event_e, max_event_d = [], [], [], [], []
        for gi in range(n_gcm):
            u245 = daily_245[gi][:, p]
            u370 = daily_370[gi][:, p]
            n_years_245 = len(u245) / 365.0
            n_years_370 = len(u370) / 365.0
            annual245_g.append(float(u245.sum() * 24 / 1000 / n_years_245))
            annual370_g.append(float(u370.sum() * 24 / 1000 / n_years_370))

            u = u370
            pos = u[u > 1e-6]
            if len(pos) == 0:
                p95_pos.append(0.0)
                p99_pos.append(0.0)
            else:
                p95_pos.append(float(np.percentile(pos, 95)))
                p99_pos.append(float(np.percentile(pos, 99)))
            max_daily.append(float(np.max(u)))
            e, d = event_metrics_for_series(u)
            max_event_e.append(e)
            max_event_d.append(d)

        annual245_g = np.array(annual245_g)
        annual370_g = np.array(annual370_g)

        rows.append({
            "province": prov,
            "province_short": PROV_SHORT[p],
            "region": base.PROV_TO_REGION.get(prov, ""),
            "annual_residual_TWh_245": float(np.mean(annual245_g)),
            "annual_residual_TWh_245_std": float(np.std(annual245_g)),
            "annual_residual_TWh_370": float(np.mean(annual370_g)),
            "annual_residual_TWh_370_std": float(np.std(annual370_g)),
            "climate_increment_TWh": float(np.mean(annual370_g - annual245_g)),
            "climate_increment_TWh_std": float(np.std(annual370_g - annual245_g)),
            "p95_positive_daily_residual_GW": float(np.mean(p95_pos)),
            "p99_positive_daily_residual_GW": float(np.mean(p99_pos)),
            "max_daily_residual_GW": float(np.mean(max_daily)),
            "max_event_energy_TWh": float(np.mean(max_event_e)),
            "max_event_duration_days": float(np.mean(max_event_d)),
        })

    df = pd.DataFrame(rows)

    # Add simple net renewable surplus/deficit from existing monthly supply-demand gap.
    sd = np.load(DATA_DIR / "supply_demand_2050_CN2050_ssp370.npz", allow_pickle=True)
    net_gap = sd["gap"].sum(axis=0).astype(float)  # TWh/yr, positive = net RE surplus
    df["net_RE_gap_TWh_370"] = net_gap
    df["net_RE_surplus_TWh_370"] = np.maximum(net_gap, 0)

    return df


def assign_typology(df: pd.DataFrame) -> pd.DataFrame:
    """Assign transparent four-quadrant provincial flexibility types.

    The primary typology is based only on two axes:
    annual residual flexibility demand and high-emissions sensitivity. This
    makes the classification easy to defend. Persistent residual-deficit
    pressure is encoded separately by bubble size, and transmission-support
    provinces are encoded as an overlay flag rather than a peer category.
    """
    out = df.copy()

    p75_annual = out["annual_residual_TWh_370"].quantile(0.75)
    p75_increment = out["climate_increment_TWh"].quantile(0.75)
    p75_event = out["max_event_energy_TWh"].quantile(0.75)
    p75_export = out["net_RE_surplus_TWh_370"].quantile(0.75)

    out["high_annual_residual"] = out["annual_residual_TWh_370"] > p75_annual
    out["high_climate_increment"] = out["climate_increment_TWh"] > p75_increment
    out["high_event_energy"] = out["max_event_energy_TWh"] > p75_event
    out["high_export_role"] = (out["net_RE_surplus_TWh_370"] > max(p75_export, 1.0))
    out["transmission_support_overlay"] = out["high_export_role"] & ~out["high_annual_residual"]

    types = []
    for _, r in out.iterrows():
        if r["high_annual_residual"] and r["high_climate_increment"]:
            t = "Compound pressure"
        elif r["high_annual_residual"]:
            t = "Structural flexibility burden"
        elif r["high_climate_increment"]:
            t = "Climate-sensitive"
        else:
            t = "Lower-priority quadrant"
        types.append(t)
    out["flexibility_type"] = types

    # Store thresholds as attrs for logging.
    out.attrs["thresholds"] = {
        "P75 annual_residual_TWh_370": float(p75_annual),
        "P75 climate_increment_TWh": float(p75_increment),
        "P75 max_event_energy_TWh": float(p75_event),
        "P75 net_RE_surplus_TWh_370": float(p75_export),
    }
    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 150,
    })


def plot_attribution(attr_df: pd.DataFrame) -> None:
    """Publication-style signed stacked bar chart of attribution components."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.55), constrained_layout=True)

    df = attr_df.set_index("pathway").loc[CAP_SCENARIOS]
    x = np.arange(len(df))
    width = 0.46
    pos_bottom = np.zeros(len(df))
    neg_bottom = np.zeros(len(df))
    component_labels = {
        "wind": "Wind",
        "solar": "Solar",
        "demand": "Demand",
        "interaction": "Nonlinear\ncoupling",
    }

    for comp in ATTR_COMPONENTS:
        vals = df[f"{comp}_TWh_yr"].values.astype(float)
        bottoms = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(x, vals, width, bottom=bottoms, color=ATTR_COLORS[comp],
               edgecolor="white", linewidth=0.45, label=component_labels[comp],
               zorder=3)
        pos_bottom += np.where(vals > 0, vals, 0)
        neg_bottom += np.where(vals < 0, vals, 0)

    totals = df["total_TWh_yr"].values.astype(float)
    total_std = df["total_std_TWh_yr"].values.astype(float)
    ax.errorbar(x, totals, yerr=total_std, fmt="o", color="black",
                markersize=4.2, markeredgewidth=0, elinewidth=0.75,
                capsize=2.2, capthick=0.75, zorder=6, label="Total")
    for i, v in enumerate(totals):
        ax.text(x[i], v + total_std[i] + 10, f"{v:.0f}",
                ha="center", va="bottom", fontsize=7.2)

    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([base.SCENARIO_LABELS[c] for c in df.index])
    ax.set_ylabel("Residual flexibility increment\n(TWh yr$^{-1}$)")
    ax.set_ylim(-45, 445)
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_xlim(-0.55, len(df) - 0.45)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8, zorder=0)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.margins(x=0.08)

    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_labels = ["Wind", "Solar", "Demand", "Nonlinear\ncoupling", "Total"]
    ax.legend([handle_by_label[label] for label in legend_labels], legend_labels,
              frameon=False, ncol=5, fontsize=6.2, loc="upper left",
              bbox_to_anchor=(-0.02, 1.08), handlelength=1.15,
              columnspacing=0.75, handletextpad=0.35)

    out = FIG_DIR / "fig_step3F_attribution_stacked.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    for suffix in [".pdf", ".svg"]:
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_typology_scatter(typology_df: pd.DataFrame) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    df = typology_df.copy()
    size_raw = df["max_event_energy_TWh"].values
    if np.nanmax(size_raw) > 0:
        sizes = 28 + 280 * size_raw / np.nanmax(size_raw)
    else:
        sizes = np.full(len(df), 50.0)

    # Draw background provinces first and priority classes later.
    for t in TYPE_ORDER[::-1]:
        sub = df[df["flexibility_type"] == t]
        if sub.empty:
            continue
        idx = sub.index.values
        is_other = t == "Lower-priority quadrant"
        ax.scatter(
            sub["annual_residual_TWh_370"],
            sub["climate_increment_TWh"],
            s=np.where(is_other, 36, sizes[idx]),
            color=TYPE_COLORS.get(t, "#999999"),
            edgecolor="white" if not is_other else "#eeeeee",
            linewidth=0.6 if not is_other else 0.35,
            alpha=0.9 if not is_other else 0.55,
            label=t,
        )

    # Overlay transmission-support provinces with a green ring. This is a
    # system role, not a separate quadrant category.
    support = df[df.get("transmission_support_overlay", False)]
    if not support.empty:
        ax.scatter(
            support["annual_residual_TWh_370"],
            support["climate_increment_TWh"],
            s=95,
            facecolor="none",
            edgecolor="#1b9e77",
            linewidth=1.3,
            zorder=5,
        )

    thresholds = df.attrs.get("thresholds", {})
    x_thr = thresholds.get("P75 annual_residual_TWh_370")
    y_thr = thresholds.get("P75 climate_increment_TWh")
    if x_thr is not None:
        ax.axvline(x_thr, color="0.35", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.text(x_thr + 1, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else 0,
                "P75 residual", rotation=90, va="bottom", ha="left",
                fontsize=6.5, color="0.35")
    if y_thr is not None:
        ax.axhline(y_thr, color="0.35", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < 0 else 0, y_thr + 0.6,
                "P75 sensitivity", va="bottom", ha="left",
                fontsize=6.5, color="0.35")

    # Label all non-background quadrant provinces, plus only the largest
    # transmission-support overlays. Labelling every low-left overlay province
    # makes the origin unreadable without adding much planning information.
    priority_labels = df[df["flexibility_type"] != "Lower-priority quadrant"]
    if "transmission_support_overlay" in df.columns:
        support_labels = (
            df[df["transmission_support_overlay"]]
            .sort_values("net_RE_surplus_TWh_370", ascending=False)
            .head(4)
        )
    else:
        support_labels = df.iloc[0:0]
    label_df = (
        pd.concat([priority_labels, support_labels], axis=0)
        .drop_duplicates(subset=["province"])
        .copy()
    )
    label_offsets = {
        "Guangdong": (3.0, 1.0), "Henan": (2.0, 1.2), "Shandong": (2.0, -0.4),
        "Guangxi": (2.0, -0.8), "Hunan": (2.0, 0.8), "Guizhou": (2.0, 0.6),
        "Jiangxi": (2.0, 0.7), "Shaanxi": (2.0, -0.8), "Anhui": (2.0, 0.8),
        "Inner Mongolia": (5.6, 1.9), "Gansu": (5.0, 1.3),
        "Sichuan": (4.2, 2.1), "Tibet": (4.8, -1.2),
    }
    for _, r in label_df.iterrows():
        dx, dy = label_offsets.get(r["province"], (2.0, 0.5))
        x = r["annual_residual_TWh_370"]
        yv = r["climate_increment_TWh"]
        tx = x + dx
        ty = yv + dy
        is_support = bool(r.get("transmission_support_overlay", False))
        if is_support and r["flexibility_type"] == "Lower-priority quadrant":
            ax.annotate(
                r["province_short"],
                xy=(x, yv),
                xytext=(tx, ty),
                textcoords="data",
                fontsize=6.8,
                color="black",
                arrowprops=dict(arrowstyle="-", color="0.45", linewidth=0.45,
                                shrinkA=0, shrinkB=2),
            )
        else:
            ax.text(tx, ty, r["province_short"], fontsize=6.8, color="black")

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Annual residual flexibility demand under CN2050 SSP3-7.0 (TWh yr$^{-1}$)")
    ax.set_ylabel("High-emissions sensitivity\nSSP3-7.0 minus SSP2-4.5 (TWh yr$^{-1}$)")
    ax.grid(alpha=0.25)

    # Legend: fixed-size markers only encode the four quadrant classes.
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=6.5,
            markerfacecolor=TYPE_COLORS[lab],
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=0.9 if lab != "Lower-priority quadrant" else 0.55,
        )
        for lab in TYPE_ORDER
    ]
    labels = [TYPE_DISPLAY_LABELS.get(lab, lab) for lab in TYPE_ORDER]
    ax.legend(handles, labels, frameon=False, fontsize=7.2, loc="upper left",
              bbox_to_anchor=(0.01, 0.99), borderaxespad=0.0,
              handletextpad=0.4, labelspacing=0.45)

    out = FIG_DIR / "fig_step3F_provincial_typology_scatter.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def compute_attribution_and_daily() -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
    print("=" * 72)
    print("Step3F: high-emissions sensitivity attribution and provincial typology")
    print("=" * 72)

    hydro_cf = base.load_hydro_cf()
    tx_cap = base.load_transmission_capacity()
    regions = base.find_grid_regions(tx_cap)
    inter_cap, unique_regions, region_map = base.build_inter_region_links(tx_cap, regions)
    n_e = int((inter_cap > 0).sum())
    print(f"Transmission network: {len(unique_regions)} AC regions, {n_e} directed edges")

    # Raw national unmet for each counterfactual.
    national = {
        cap: {case: np.zeros(N_GCM, dtype=float) for case in CASE_DEFS}
        for cap in CAP_SCENARIOS
    }

    # For provincial typology, keep CN2050 daily U for core all-climate cases.
    cn2050_daily_245 = []
    cn2050_daily_370 = []

    # Optional annual provincial counterfactuals for later inspection.
    prov_counterfactual = {}

    for cap in CAP_SCENARIOS:
        print(f"\nCapacity pathway: {base.SCENARIO_LABELS[cap]}")
        for gi, gcm in enumerate(GCM_LIST):
            print(f"  GCM {gi+1}/{N_GCM}: {gcm}")
            for case in CASE_DEFS:
                res = run_dispatch_counterfactual(
                    cap_scen=cap,
                    case_name=case,
                    gcm=gcm,
                    hydro_cf=hydro_cf,
                    regions=regions,
                    inter_links=inter_cap,
                    region_map=region_map,
                    unique_regions=unique_regions,
                )
                national[cap][case][gi] = res["national_unmet_TWh"]
                prov_counterfactual.setdefault((cap, case), []).append(res["prov_unmet_TWh"])

                if cap == "CN2050" and case == "SSP245_all":
                    cn2050_daily_245.append(res["prov_post_tx_daily"])
                if cap == "CN2050" and case == "SSP370_all":
                    cn2050_daily_370.append(res["prov_post_tx_daily"])

    cn2050_daily_245 = [a.astype(np.float32) for a in cn2050_daily_245]
    cn2050_daily_370 = [a.astype(np.float32) for a in cn2050_daily_370]

    # Attribution tables
    summary_rows = []
    pergcm_rows = []
    for cap in CAP_SCENARIOS:
        u245 = national[cap]["SSP245_all"]
        u370 = national[cap]["SSP370_all"]
        uwind = national[cap]["wind370_only"]
        usolar = national[cap]["solar370_only"]
        udemand = national[cap]["demand370_only"]

        total = u370 - u245
        wind = uwind - u245
        solar = usolar - u245
        demand = udemand - u245
        interaction = total - wind - solar - demand

        comp = {
            "total": total,
            "wind": wind,
            "solar": solar,
            "demand": demand,
            "interaction": interaction,
        }
        row = {"pathway": cap, "pathway_label": base.SCENARIO_LABELS[cap]}
        for k, arr in comp.items():
            row[f"{k}_TWh_yr"] = float(np.mean(arr))
            row[f"{k}_std_TWh_yr"] = float(np.std(arr))
        row["SSP245_all_unmet_TWh_yr"] = float(np.mean(u245))
        row["SSP370_all_unmet_TWh_yr"] = float(np.mean(u370))
        summary_rows.append(row)

        for gi, gcm in enumerate(GCM_LIST):
            pergcm_rows.append({
                "pathway": cap,
                "pathway_label": base.SCENARIO_LABELS[cap],
                "gcm": gcm,
                "SSP245_all_unmet_TWh_yr": float(u245[gi]),
                "SSP370_all_unmet_TWh_yr": float(u370[gi]),
                "total_TWh_yr": float(total[gi]),
                "wind_TWh_yr": float(wind[gi]),
                "solar_TWh_yr": float(solar[gi]),
                "demand_TWh_yr": float(demand[gi]),
                "interaction_TWh_yr": float(interaction[gi]),
            })

    attr_df = pd.DataFrame(summary_rows)
    pergcm_df = pd.DataFrame(pergcm_rows)

    # Save annual provincial counterfactuals as a compact npz for reproducibility.
    npz_dict = {
        "provinces": np.array(PROVINCES),
        "gcm_list": np.array(GCM_LIST),
    }
    for gi, gcm in enumerate(GCM_LIST):
        safe_gcm = gcm.replace("-", "_").replace(".", "_")
        npz_dict[f"cn2050_ssp245_all_daily_unmet_GW_{safe_gcm}"] = cn2050_daily_245[gi]
        npz_dict[f"cn2050_ssp370_all_daily_unmet_GW_{safe_gcm}"] = cn2050_daily_370[gi]
    for (cap, case), vals in prov_counterfactual.items():
        npz_dict[f"{cap}_{case}_prov_unmet_TWh_each"] = np.stack(vals, axis=0).astype(np.float32)
    np.savez_compressed(DATA_DIR / "step3F_core_daily_unmet_CN2050.npz", **npz_dict)
    print(f"\nSaved: {DATA_DIR / 'step3F_core_daily_unmet_CN2050.npz'}")

    return attr_df, pergcm_df, cn2050_daily_245, cn2050_daily_370


def save_outputs(attr_df: pd.DataFrame, pergcm_df: pd.DataFrame, typology_df: pd.DataFrame) -> None:
    attr_csv = DATA_DIR / "step3F_attribution_results.csv"
    pergcm_csv = DATA_DIR / "step3F_attribution_per_gcm.csv"
    typ_csv = DATA_DIR / "step3F_provincial_typology.csv"

    attr_df.to_csv(attr_csv, index=False, encoding="utf-8-sig")
    pergcm_df.to_csv(pergcm_csv, index=False, encoding="utf-8-sig")
    typology_df.to_csv(typ_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {attr_csv}")
    print(f"Saved: {pergcm_csv}")
    print(f"Saved: {typ_csv}")

    xlsx = EXCEL_DIR / "step3F_attribution_typology.xlsx"
    try:
        with pd.ExcelWriter(xlsx) as writer:
            attr_df.to_excel(writer, sheet_name="Attribution_summary", index=False)
            pergcm_df.to_excel(writer, sheet_name="Attribution_per_GCM", index=False)
            typology_df.to_excel(writer, sheet_name="Provincial_typology", index=False)
            pd.DataFrame([typology_df.attrs.get("thresholds", {})]).to_excel(
                writer, sheet_name="Typology_thresholds", index=False
            )
        print(f"Saved: {xlsx}")
    except Exception as exc:
        print(f"WARNING: failed to write Excel file {xlsx}: {exc}")


def print_key_results(attr_df: pd.DataFrame, typology_df: pd.DataFrame) -> None:
    print("\nAttribution summary (TWh/yr):")
    cols = ["pathway_label", "total_TWh_yr", "wind_TWh_yr", "solar_TWh_yr",
            "demand_TWh_yr", "interaction_TWh_yr"]
    print(attr_df[cols].round(1).to_string(index=False))

    print("\nTypology counts:")
    print(typology_df["flexibility_type"].value_counts().to_string())

    print("\nTop provinces by CN2050 SSP3-7.0 annual residual demand:")
    top = typology_df.sort_values("annual_residual_TWh_370", ascending=False).head(10)
    show_cols = ["province", "annual_residual_TWh_370", "climate_increment_TWh",
                 "max_event_energy_TWh", "max_event_duration_days", "flexibility_type"]
    print(top[show_cols].round(2).to_string(index=False))


def main() -> None:
    attr_df, pergcm_df, daily245, daily370 = compute_attribution_and_daily()

    typology_base = provincial_metrics_from_daily(daily245, daily370)
    typology_df = assign_typology(typology_base)

    save_outputs(attr_df, pergcm_df, typology_df)
    plot_attribution(attr_df)
    plot_typology_scatter(typology_df)
    print_key_results(attr_df, typology_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
