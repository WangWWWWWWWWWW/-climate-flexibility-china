# -*- coding: utf-8 -*-
"""Generate only the figures referenced by the manuscript TeX file.

This runner intentionally excludes legacy, diagnostic and alternative-layout
plotting scripts so that ``figures/`` stays aligned with the manuscript.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "plot_Fig2_nature_supply_demand.py",
    "plot_Fig3a_nature_LSLW_dotplot.py",
    "plot_Fig3b_nature_sync_curve.py",
    "plot_Fig4_penetration_by_day_type.py",  # refreshes Fig. 4 intermediate data
    "plot_Fig4_nature_penetration.py",
    "plot_Fig5a_nature_waterfall.py",
    "plot_Fig5b_nature_provincial.py",
    "plot_Fig6_combined.py",
    "plot_ED_Fig3_hydropower_sensitivity.py",
    "plot_ED_Fig6_climate_driven_demand.py",
    "plot_ED_Fig7_curtailment_shortage.py",
    "plot_ED_attribution.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"=== {script} ===", flush=True)
        subprocess.run([sys.executable, script], cwd=CODE_DIR, check=True)


if __name__ == "__main__":
    main()
