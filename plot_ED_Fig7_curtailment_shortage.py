# -*- coding: utf-8 -*-
"""Plot Extended Data Fig. 7 from deposited source data.

The original LP optimization uses third-party raw inputs that are not deposited
in this submission package. This script regenerates the manuscript figure from
the package-local ED Fig. 7 source-data workbook.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from path_config import FIGURES_DIR, SOURCE_DATA_DIR, ensure_output_dirs


SOURCE_FILE = SOURCE_DATA_DIR / "table_ED_Fig7_curtailment_shortage_source_data.xlsx"
OUT_FILE = FIGURES_DIR / "ED_Fig7_curtailment_shortage.png"

SCENARIOS = ["NDC", "GM2.0", "CN2050"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TICK_LABELS = ["Jan", "", "Mar", "", "May", "", "Jul", "", "Sep", "", "Nov", ""]
REGIONS = [
    "Northwest",
    "Northeast",
    "North China",
    "East China",
    "Central China",
    "Southwest",
    "South China",
]
REGION_COLORS = {
    "Northwest": "#F39B7F",
    "Northeast": "#4DBBD5",
    "North China": "#E64B35",
    "East China": "#00A087",
    "Central China": "#3C5488",
    "Southwest": "#8491B4",
    "South China": "#91D1C2",
}


def load_region_data() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_FILE, sheet_name="Monthly_region")
    required = {"scenario", "transmission_case", "metric", "month", "region", "value_TWh"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SOURCE_FILE.name}: {sorted(missing)}")
    return df


def monthly_values(df: pd.DataFrame, scenario: str, case: str, metric: str,
                   region: str | None = None) -> np.ndarray:
    sub = df[
        (df["scenario"] == scenario)
        & (df["transmission_case"] == case)
        & (df["metric"] == metric)
    ]
    if region is not None:
        sub = sub[sub["region"] == region]
    values = sub.groupby("month", sort=False)["value_TWh"].sum()
    return values.reindex(MONTHS, fill_value=0).to_numpy(dtype=float)


def main() -> None:
    ensure_output_dirs()
    df = load_region_data()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.5,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(
        2, 3, figsize=(183 / 25.4, 100 / 25.4),
        gridspec_kw={"hspace": 0.4, "wspace": 0.3},
    )
    x = np.arange(12)
    region_handles = None
    without_handle = None

    for si, scenario in enumerate(SCENARIOS):
        ax = axes[0, si]
        bottom = np.zeros(12)
        handles_this_axis = []
        for region in REGIONS:
            vals = monthly_values(
                df, scenario, "with_transmission", "curtailment_TWh", region
            )
            handle = ax.fill_between(
                x, bottom, bottom + vals, alpha=0.7,
                color=REGION_COLORS[region], label=region,
            )
            handles_this_axis.append(handle)
            bottom += vals
        line, = ax.plot(
            x,
            monthly_values(df, scenario, "without_transmission", "curtailment_TWh"),
            "k--", lw=1.2, alpha=0.7, label="Without trans.",
        )
        if region_handles is None:
            region_handles = handles_this_axis
        if without_handle is None:
            without_handle = line
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_TICK_LABELS, fontsize=6)
        ax.set_ylabel("Curtailment (TWh)")
        ax.set_title(f"{scenario} - Curtailment", fontsize=7, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = axes[1, si]
        bottom = np.zeros(12)
        for region in REGIONS:
            vals = monthly_values(
                df, scenario, "with_transmission", "shortage_TWh", region
            )
            ax.fill_between(
                x, bottom, bottom + vals, alpha=0.7,
                color=REGION_COLORS[region],
            )
            bottom += vals
        ax.plot(
            x,
            monthly_values(df, scenario, "without_transmission", "shortage_TWh"),
            "k--", lw=1.2, alpha=0.7, label="Without trans.",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_TICK_LABELS, fontsize=6)
        ax.set_ylabel("Shortage (TWh)")
        ax.set_title(f"{scenario} - Shortage", fontsize=7, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if si == 2 and without_handle is not None:
            ax.legend(
                handles=[without_handle], labels=["Without trans."],
                fontsize=4.8, loc="upper right", framealpha=0.9,
                handlelength=1.6, borderpad=0.25, labelspacing=0.2,
            )

    if region_handles is not None:
        fig.legend(
            region_handles, REGIONS, loc="upper center",
            bbox_to_anchor=(0.5, 1.02), ncol=7, fontsize=4.8,
            framealpha=0.9, handlelength=1.2, columnspacing=0.8,
        )
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
