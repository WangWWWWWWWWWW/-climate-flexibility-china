# -*- coding: utf-8 -*-
"""Shared paths for the deposited data and code package.

By default, scripts read processed data from this package and write figures back
to the package's ``figures`` directory. Raw third-party data are not deposited;
set the environment variables below when rerunning the full data-generation
pipeline from original ERA5, CMIP6 and Zhuo et al. inputs.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(os.environ.get("LSLW_PACKAGE_ROOT", Path(__file__).resolve().parents[1])).resolve()

PROCESSED_RESULTS_DIR = PACKAGE_ROOT / "data" / "processed_results"
SOURCE_DATA_DIR = PACKAGE_ROOT / "data" / "source_data"
INPUT_TABLES_DIR = PACKAGE_ROOT / "data" / "input_tables"
FIGURES_DIR = PACKAGE_ROOT / "figures"
DOCS_DIR = PACKAGE_ROOT / "docs"
CODE_DIR = PACKAGE_ROOT / "code"

# Backwards-compatible names used by the original analysis scripts.
BASE_DIR = PACKAGE_ROOT
DATA_DIR = PROCESSED_RESULTS_DIR
FIG_DIR = FIGURES_DIR
OUT_DIR = FIGURES_DIR
EXCEL_DIR = SOURCE_DATA_DIR

# Optional local paths for raw or third-party inputs that are not redistributed.
ERA5_HOURLY = Path(os.environ.get("LSLW_ERA5_HOURLY", PACKAGE_ROOT / "external_data" / "ERA5_hourly"))
ERA5_DAILY_RSDS = Path(os.environ.get("LSLW_ERA5_DAILY_RSDS", PACKAGE_ROOT / "external_data" / "ERA5_daily_rsds"))
CMIP6_DIR = Path(os.environ.get("LSLW_CMIP6_DIR", PACKAGE_ROOT / "external_data" / "CMIP6"))
STEP1_DATA = Path(os.environ.get("LSLW_STEP1_DATA", PACKAGE_ROOT / "external_data" / "step1_data"))
ZHUO_DIR = Path(os.environ.get("LSLW_ZHUO_DIR", PACKAGE_ROOT / "external_data" / "zhuo_2022"))
HYDRO_PATH = Path(os.environ.get("LSLW_HYDRO_PATH", INPUT_TABLES_DIR / "input_hydro_climatology.xlsx"))
TX_PATH = Path(os.environ.get("LSLW_TX_PATH", INPUT_TABLES_DIR / "input_transmission_network.xlsx"))
MOESM4_PATH = Path(os.environ.get("LSLW_MOESM4_PATH", PACKAGE_ROOT / "external_data" / "41467_2022_30747_MOESM4_ESM.xlsx"))
LOAD_FILE = Path(os.environ.get("LSLW_LOAD_FILE", PACKAGE_ROOT / "external_data" / "provincial_monthly_load.xlsx"))
GRID_COORDS = PROCESSED_RESULTS_DIR / "grid_coords.npz"
GRID_MAP = Path(os.environ.get("LSLW_GRID_MAP", PACKAGE_ROOT / "external_data" / "grid_province_map.npy"))


def ensure_output_dirs() -> None:
    """Create package-local output directories used by plotting scripts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
