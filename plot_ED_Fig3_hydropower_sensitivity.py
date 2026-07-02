# -*- coding: utf-8 -*-
"""Plot Extended Data Fig. 3 hydropower sensitivity."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from path_config import FIGURES_DIR, SOURCE_DATA_DIR, ensure_output_dirs

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

SOURCE_FILE = SOURCE_DATA_DIR / "table_ED_Fig3_hydropower_sensitivity_source_data.xlsx"
OUT_FILE = FIGURES_DIR / "ED_Fig3_hydropower_sensitivity.png"
OUT_PDF = FIGURES_DIR / "ED_Fig3_hydropower_sensitivity.pdf"

SCENARIOS = ["NDC", "GM2.0", "CN2050"]
SSPS = ["SSP2-4.5", "SSP3-7.0"]
SSP_STYLE = {
    "SSP2-4.5": {"color": "#4292c6", "linestyle": "--", "label": "SSP2-4.5"},
    "SSP3-7.0": {"color": "#de2d26", "linestyle": "-", "label": "SSP3-7.0"},
}


def main() -> None:
    ensure_output_dirs()

    df = pd.read_excel(SOURCE_FILE, sheet_name="Summary")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.83, wspace=0.27)

    for ax, scenario in zip(axes, SCENARIOS):
        sub_scen = df[df["cap_scenario"] == scenario]
        for ssp in SSPS:
            sub = sub_scen[sub_scen["ssp"] == ssp].sort_values("hydro_change_pct")
            x = sub["hydro_change_pct"].to_numpy(dtype=float)
            y = sub["unmet_TWh_mean"].to_numpy(dtype=float)
            sd = sub["unmet_TWh_std"].to_numpy(dtype=float)
            style = SSP_STYLE[ssp]

            ax.fill_between(x, y - sd, y + sd, color=style["color"], alpha=0.12, linewidth=0)
            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                label=style["label"],
            )

        ax.set_title(scenario, fontweight="bold")
        ax.set_xlabel("Hydropower CF change (%)")
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("National unmet demand\n(TWh/yr)")
    axes[0].legend(frameon=False, loc="upper right")

    fig.savefig(OUT_FILE, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
