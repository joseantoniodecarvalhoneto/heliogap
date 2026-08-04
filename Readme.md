# Heliogap

**Heliogap** is a specialized Python library for Space Geophysics and Heliophysics, designed for robust extraction, cleaning, and time-series gap analysis of space weather datasets. It provides built-in support for downloading and analyzing massive datasets from NASA's OMNI database and INPE's EMBRACE MagNet network.

---

## Key Features

- **NASA OMNI Integration:** Seamlessly fetch and clean high-resolution (1-min) solar wind and interplanetary magnetic field data via `pyspedas`.
- **INPE EMBRACE MagNet Integration:** Download and process ground magnetometer station data across Brazil natively.
- **Exhaustive Gap Engine:** A highly optimized, sequential interpolation engine to simulate and evaluate various gap sizes across historical time-series datasets without memory crashes.
- **Comprehensive Metrics:** Built-in evaluation functions including WMAPE, MAE, RMSE, R², and MDA (Mean Directional Accuracy).
- **Standardized Visualization:** Ready-to-use plotter to automatically generate publication-quality metric matrices for multiple physics features.

---

## Installation

Heliogap requires Python 3.8+ and standard scientific libraries (`numpy`, `pandas`, `matplotlib`, `pyspedas`, `cdflib`).

Clone the repository and install it locally:

```bash
git clone https://github.com/joseantoniodecarvalhoneto/heliogap.git
cd heliogap
pip install -e .
```

---

##  Quick Start

Heliogap makes it incredibly easy to start analyzing massive space physics datasets.

### 1. NASA OMNI Analysis (Solar Wind & IMF)

```python
import heliogap as hg

# 1. Load historical OMNI data (automatically downloads and caches)
df = hg.load_omni_data(cache_filepath="omni_dados.pkl")

# 2. Define features, metrics, and gap sizes to simulate
features = ['flow_speed', 'proton_density', 'BZ_GSM']
metric = 'rmse'
gap_sizes = [1, 5, 15, 30, 60, 120] # in minutes

results_dict = {}

# 3. Run the Exhaustive Engine
for var in features:
    if var in df.columns and df[var].notna().any():
        gaps, errors = hg.run_exhaustive_analysis(
            df=df,
            feature_name=var,
            metric_name=metric,
            gap_sizes=gap_sizes,
            interpolation_method='linear'
        )
        results_dict[var] = errors

# 4. Generate the Visualization Matrix
hg.plot_metric_matrix(
    gap_sizes=gap_sizes, 
    results_dict=results_dict, 
    metric_name=metric,
    save_path="omni_analysis.png"
)
```

### 2. INPE EMBRACE Analysis (Ground Magnetometers)

```python
import heliogap as hg

# 1. Download data for a specific station (e.g., Vassouras - VSS)
df = hg.download_embrace_data(station='VSS', start_year=2024)

# 2. Run analysis on the Horizontal Magnetic Field (H)
gap_sizes, errors = hg.run_exhaustive_analysis(
    df=df,
    feature_name='H',
    metric_name='mae',
    gap_sizes=[1, 2, 5, 10, 15, 30, 60]
)

# 3. Plot
hg.plot_metric_matrix(
    gap_sizes=gap_sizes,
    results_dict={'H': errors},
    metric_name='mae',
    save_path="embrace_VSS_mae.png"
)
```

---

## Available Metrics

The exhaustive engine supports the following metrics to evaluate the performance of interpolations across gaps:

- **`wmape`**: Weighted Mean Absolute Percentage Error
- **`mae`**: Mean Absolute Error (Great for massive baselines like Earth's magnetic field)
- **`rmse`**: Root Mean Squared Error (Penalizes larger deviations)
- **`r2`**: Coefficient of Determination (R² Score)
- **`mda`**: Mean Directional Accuracy (Percentage of correctly predicted trend directions)

---

## Supported Interpolation Methods

The engine is highly flexible and accepts multiple interpolation techniques via the `interpolation_method` parameter in `run_exhaustive_analysis()`.

- **Basic Methods:** `'linear'` (default), `'nearest'`, `'zero'`
- **Advanced Methods (via pandas):** `'polynomial'`, `'spline'`, `'cubic'`, `'barycentric'`, `'krogh'`, `'pchip'`, `'akima'`, etc. (Note: Some of these require the `order` parameter, e.g., `order=2`).

---

## Architecture / Modules

- `heliogap.engine`: Core gap simulation logic (`run_exhaustive_analysis`).
- `heliogap.omni`: NASA OMNI data loader and cleaner.
- `heliogap.embrace`: INPE EMBRACE MagNet network scraper and cleaner.
- `heliogap.metrics`: Mathematical definitions of the evaluation metrics.
- `heliogap.plotter`: Automated multi-subplot generator for gap metric visualization.

---

## License & Authors

**Author:** José Antonio de Carvalho Neto  
**Contact:** joseadecn@gmail.com  
**Homepage:** [GitHub - heliogap](https://github.com/joseantoniodecarvalhoneto/heliogap)
