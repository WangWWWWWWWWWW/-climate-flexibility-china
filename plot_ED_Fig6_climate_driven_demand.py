# -*- coding: utf-8 -*-
"""Plot Extended Data Fig. 6 from deposited source data.

The full data-generation workflow depends on external raw load and CMIP6 files.
This plotting script uses the deposited source-data workbook so the manuscript
figure can be regenerated from the submission package alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from path_config import FIGURES_DIR, SOURCE_DATA_DIR, ensure_output_dirs


SCENARIOS = ["NDC", "GM2.0", "CN2050"]
SSPS = ["ssp245", "ssp370"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COLORS = {"NDC": "#666666", "GM2.0": "#74add1", "CN2050": "#d73027"}
SOURCE_FILE = SOURCE_DATA_DIR / "table_ED_Fig6_climate_driven_demand_source_data.xlsx"


def load_source_data() -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for scenario in SCENARIOS:
        for ssp in SSPS:
            sheet = f"{scenario}_{ssp}"
            out[(scenario, ssp)] = pd.read_excel(SOURCE_FILE, sheet_name=sheet)
    return out


def monthly_series(df: pd.DataFrame, column: str, reducer: str) -> np.ndarray:
    grouped = df.groupby("month", sort=False)[column]
    if reducer == "sum":
        values = grouped.sum()
    elif reducer == "mean":
        values = grouped.mean()
    elif reducer == "min":
        values = grouped.min()
    elif reducer == "max":
        values = grouped.max()
    else:
        raise ValueError(f"Unknown reducer: {reducer}")
    return values.reindex(MONTHS).to_numpy(dtype=float)


def main() -> None:
    ensure_output_dirs()
    data = load_source_data()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })

    x = np.arange(1, 13)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panel_labels = ["a", "b", "c", "d", "e", "f"]

    for si, ssp in enumerate(SSPS):
        temp_df = data[("NDC", ssp)]

        ax = axes[si, 0]
        temp_mean = monthly_series(temp_df, "delta_T_C", "mean")
        temp_min = monthly_series(temp_df, "delta_T_C", "min")
        temp_max = monthly_series(temp_df, "delta_T_C", "max")
        ax.fill_between(x, temp_min, temp_max, alpha=0.2, color="#d73027")
        ax.plot(x, temp_mean, "o-", color="#d73027", lw=2, ms=6,
                label="National mean")
        ax.axhline(y=0, color="k", lw=0.5, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(MONTHS, fontsize=9)
        ax.set_ylabel("Delta T (deg C)", fontsize=11)
        ax.set_title(f"Temperature change - {ssp.upper()}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[si, 1]
        for scenario in SCENARIOS:
            df = data[(scenario, ssp)].copy()
            df["delta_demand_TWh"] = df["monthly_demand_TWh"] - df["monthly_base_TWh"]
            delta_demand = monthly_series(df, "delta_demand_TWh", "sum")
            ax.plot(x, delta_demand, "o-", color=COLORS[scenario],
                    lw=2, ms=5, label=scenario)
        ax.axhline(y=0, color="k", lw=0.5, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(MONTHS, fontsize=9)
        ax.set_ylabel("Delta demand (TWh)", fontsize=11)
        ax.set_title(f"Climate-driven demand change - {ssp.upper()}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[si, 2]
        for scenario in SCENARIOS:
            national = monthly_series(data[(scenario, ssp)], "monthly_demand_TWh", "sum")
            ax.plot(x, national, "o-", color=COLORS[scenario],
                    lw=2, ms=5, label=scenario)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTHS, fontsize=9)
        ax.set_ylabel("National demand (TWh)", fontsize=11)
        ax.set_title(f"Monthly demand in 2050 - {ssp.upper()}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for label, ax in zip(panel_labels, axes.ravel()):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    for name in ["fig_climate_driven_demand.png", "ED_Fig6_climate_driven_demand.png"]:
        fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
