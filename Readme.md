<div align="center">

# Heliogap

**High-Performance Python Library for Heliophysics and Space Geophysics Gap Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

</div>

**Heliogap** is a specialized, high-performance Python library tailored for Space Geophysics and Heliophysics. It enables automated ingestion, physics-informed data cleaning, and systematic interpolation benchmarking across time-series datasets from interplanetary space and ground-based observatory networks.

Built for multi-million-row datasets, Heliogap provides seamless integration with **NASA OMNI**, **INPE EMBRACE**, and **INTERMAGNET**, paired with a parallelized gap simulation engine designed to evaluate interpolation algorithms across simulated data gaps without memory bottlenecks.

---

## Key Features

- **Automated Multi-Source Data Ingestion:**
  - **NASA OMNI:** High-resolution (1-minute) solar wind plasma and interplanetary magnetic field (IMF) parameters via CDAWeb and `pyspedas`.
  - **INPE EMBRACE:** Terrestrial magnetometer telemetry across South America with automatic directory parsing and station discovery.
  - **INTERMAGNET:** Global geomagnetic observatory network ingestion in IAGA-2002 ASCII format via BGS GIN web services.
- **Self-Healing Incremental Caching:**
  - Automatically identifies missing temporal ranges, retrieves only the delta years, sanitizes values, and synchronizes local disk caches.
- **Physics-Informed Data Cleaning:**
  - Strips NASA/INTERMAGNET fill codes (e.g., `99999.0`, `88888.0`), clips unphysical values, and applies **boundary trimming** to isolate sensor transitions around gaps.
- **HPC Gap Evaluation Engine (`evaluate_gaps`):**
  - GIL-bypassing multi-core execution via `joblib`.
  - Fast run-length segment extraction (NumPy zero-copy views).
  - Dynamic chunking governed by available system RAM to prevent Out-Of-Memory (OOM) faults.
- **Geophysical Metrics Suite:**
  - WMAPE, MAE, RMSE, R² Score, Mean Directional Accuracy (MDA), and rate of change of magnetic field ($\mathrm{d}B/\mathrm{d}t$).
- **Standardized Scientific Visualizations:**
  - Ready-to-publish multi-panel metric grids configured with physical units and logarithmic gap axes.

---

## Installation

### Prerequisites

Heliogap requires **Python 3.9+** and standard scientific libraries:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `joblib`
- `pyspedas`
- `cdflib`
- `requests`
- `urllib3`

### Install from Source

```bash
git clone https://github.com/joseantoniodecarvalhoneto/heliogap.git
cd heliogap
pip install -e .
```

For development and test dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. NASA OMNI Solar Wind Gap Benchmark

```python
import heliogap as hg

# 1. Ingest historical OMNI dataset (automatically downloads and caches)
df = hg.load_omni_data(cache_filepath="omni_data.pkl", start_year=2020)

# 2. Select variables, metric, and simulated gap dimensions (in minutes)
features = ['flow_speed', 'proton_density', 'BZ_GSM']
metric = 'rmse'
gap_sizes = [1, 5, 15, 30, 60, 120]

results = {}

# 3. Run gap interpolation evaluation
for var in features:
    if var in df.columns and df[var].notna().any():
        gaps, errors = hg.evaluate_gaps(
            df=df,
            feature_name=var,
            metric_name=metric,
            gap_sizes=gap_sizes,
            interpolation_method='linear',
            n_jobs=-1  # Multi-core CPU parallelization
        )
        results[var] = errors

# 4. Generate publication-ready figure
hg.plot_metric_matrix(
    gap_sizes=gap_sizes,
    results_dict=results,
    metric_name=metric,
    save_path="omni_benchmark.png"
)
```

---

### 2. INPE EMBRACE Ground Magnetometer Analysis

```python
import heliogap as hg

# Ingest Vassouras station (VSS) magnetometer telemetry
df = hg.download_embrace_data(station='VSS', start_year=2024)

# Evaluate interpolation on Horizontal Field component (H)
gap_sizes, mae_errors = hg.evaluate_gaps(
    df=df,
    feature_name='H',
    metric_name='mae',
    gap_sizes=[1, 2, 5, 10, 15, 30, 60]
)

# Plot results
hg.plot_metric_matrix(
    gap_sizes=gap_sizes,
    results_dict={'H': mae_errors},
    metric_name='mae',
    save_path="embrace_vss_mae.png"
)
```

---

### 3. INTERMAGNET Observatory Ingestion (IAGA-2002)

```python
import heliogap as hg

# Download 1-minute cadence geomagnetic data for Vassouras (VSS)
df = hg.download_intermagnet_vss(start_year=2023)

# Evaluate cubic spline interpolation on North component (X)
gaps, r2_scores = hg.evaluate_gaps(
    df=df,
    feature_name='X',
    metric_name='r2',
    gap_sizes=[1, 5, 15, 30, 60],
    interpolation_method='cubic'
)
```

---

### 4. Computing Magnetic Field Rate of Change ($\mathrm{d}B/\mathrm{d}t$)

In space weather and geomagnetism, $\mathrm{d}B/\mathrm{d}t$ is critical for assessing Geomagnetically Induced Currents (GICs):

```python
import heliogap as hg

# Compute time derivative of geomagnetic horizontal component (H)
# dt=60.0s for 1-minute cadence data
d_h_dt = hg.calculate_dbt(df['H'], dt=60.0)
```

---

## Supported Metrics

| Metric | Function / Alias | Description | Ideal For |
| :--- | :--- | :--- | :--- |
| **WMAPE** | `calculate_wmape`, `wmape` | Weighted Mean Absolute Percentage Error | Relative error comparison across features with different scales. |
| **MAE** | `calculate_mae`, `mae` | Mean Absolute Error | Robust evaluation with large baseline fields (e.g., Earth's 24,000 nT field). |
| **RMSE** | `calculate_rmse`, `rmse` | Root Mean Squared Error | Penalizes large deviations and extreme outliers during geomagnetic storms. |
| **R²** | `calculate_r2`, `r2` | Coefficient of Determination | Quantifies variance explained relative to dataset baseline mean. |
| **MDA** | `calculate_mda`, `mda` | Mean Directional Accuracy (%) | Evaluates whether interpolation preserves the sign of physical derivatives. |
| **$\mathrm{d}B/\mathrm{d}t$** | `calculate_dbt`, `dbt` | Time Derivative of Magnetic Field | GIC modeling and rapid geomagnetic impulse detection. |

---

## Supported Interpolation Methods

Heliogap supports pure NumPy and SciPy 1D interpolation algorithms via the `interpolation_method` parameter:

- **Standard Methods:** `'linear'`, `'nearest'`, `'zero'`
- **Advanced Methods:** `'cubic'`, `'cubicspline'`, `'pchip'`, `'akima'`, `'slinear'`, `'quadratic'`, `'polynomial'` (with `order=N`).

---

## Architecture & Module Structure

```text
heliogap/
├── __init__.py         # Package root exposing public API and version
├── engine.py           # Core evaluation engine (evaluate_gaps, downcast, segment extraction)
├── metrics.py          # Mathematical loss functions and dB/dt computation
├── omni.py             # NASA OMNI CDAWeb downloader and physics cleaner
├── embrace.py          # INPE EMBRACE MagNet ingestion and station scraper
├── intermagnet.py      # INTERMAGNET BGS GIN downloader and IAGA-2002 parser
└── plotter.py          # Multi-panel publication plotter
```

---

## Running Tests

Unit tests are written using `pytest`:

```bash
pytest
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```text
MIT License
Copyright (c) 2026 José Antonio de Carvalho Neto
```

---

## Author & Citation

- **Author:** José Antonio de Carvalho Neto
- **Affiliation:** Instituto Nacional de Pesquisas Espaciais (INPE)
- **Email:** [joseadecn@gmail.com](mailto:joseadecn@gmail.com)
- **Repository:** [https://github.com/joseantoniodecarvalhoneto/heliogap](https://github.com/joseantoniodecarvalhoneto/heliogap)

If you use Heliogap in your research, please cite:

```bibtex
@software{heliogap2026,
  author = {de Carvalho Neto, José Antonio},
  title = {Heliogap: High-Performance Python Library for Heliophysics and Space Geophysics Gap Analysis},
  year = {2026},
  url = {https://github.com/joseantoniodecarvalhoneto/heliogap}
}
```
