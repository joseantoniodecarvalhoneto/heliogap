"""
Heliogap: Space Geophysics and Heliophysics Python Library.

Time-series gap simulation, data ingestion, and interpolation benchmarking
for solar wind, interplanetary magnetic field, and geomagnetic observatory datasets.
"""

__version__ = "0.2.0"
__author__ = "José Antonio de Carvalho Neto"

# Metrics
from .metrics import (
    calculate_wmape,
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_mda,
    calculate_dbt,
    wmape,
    mae,
    rmse,
    r2,
    mda,
    dbt,
    calculate_dbt_simples,
    calcular_dbt,
    calcular_dbt_simples,
    METRIC_FUNCTIONS
)

# Core Engine
from .engine import (
    evaluate_gaps,
    run_gap_analysis,
    run_exhaustive_analysis,
    downcast_dataframe,
    extract_continuous_segments,
    get_optimal_chunk_size
)

# Plotter
from .plotter import (
    plot_metric_matrix,
    FEATURE_CONFIG
)

# NASA OMNI Ingestion
from .omni import (
    clean_omni_fill_values,
    clean_nasa_fill_values,
    load_omni_data
)

# INPE EMBRACE Ingestion
from .embrace import (
    clean_embrace_data,
    download_embrace_data,
    load_local_embrace_data,
    fetch_inpe_stations,
    parse_inpe_file,
    get_session_with_retries
)

# INTERMAGNET Ingestion
from .intermagnet import (
    download_intermagnet_vss,
    download_intermagnet_data,
    clean_intermagnet_data,
    parse_iaga2002,
    load_local_intermagnet_data
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    # Metrics
    "calculate_wmape",
    "calculate_mae",
    "calculate_rmse",
    "calculate_r2",
    "calculate_mda",
    "calculate_dbt",
    "wmape",
    "mae",
    "rmse",
    "r2",
    "mda",
    "dbt",
    "calculate_dbt_simples",
    "calcular_dbt",
    "calcular_dbt_simples",
    "METRIC_FUNCTIONS",
    # Engine
    "evaluate_gaps",
    "run_gap_analysis",
    "run_exhaustive_analysis",
    "downcast_dataframe",
    "extract_continuous_segments",
    "get_optimal_chunk_size",
    # Plotter
    "plot_metric_matrix",
    "FEATURE_CONFIG",
    # OMNI
    "clean_omni_fill_values",
    "clean_nasa_fill_values",
    "load_omni_data",
    # EMBRACE
    "clean_embrace_data",
    "download_embrace_data",
    "load_local_embrace_data",
    "fetch_inpe_stations",
    "parse_inpe_file",
    "get_session_with_retries",
    # INTERMAGNET
    "download_intermagnet_vss",
    "download_intermagnet_data",
    "clean_intermagnet_data",
    "parse_iaga2002",
    "load_local_intermagnet_data",
]