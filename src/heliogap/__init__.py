"""
Heliogap: A Python library for space physics data extraction and time-series gap analysis.
"""

from .metrics import (
    calculate_wmape,
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_mda
)
from .engine import run_exhaustive_analysis
from .omni import load_omni_data, clean_nasa_fill_values
from .plotter import plot_metric_matrix
from .embrace import (
    clean_embrace_data, 
    download_embrace_data, 
    load_local_embrace_data,
    fetch_inpe_stations
)

__all__ = [
    "calculate_wmape",
    "calculate_mae",
    "calculate_rmse",
    "calculate_r2",
    "calculate_mda",
    "run_exhaustive_analysis",
    "load_omni_data",
    "clean_nasa_fill_values",
    "plot_metric_matrix",
    "clean_embrace_data",
    "download_embrace_data",
    "load_local_embrace_data",
    "fetch_inpe_stations"
]