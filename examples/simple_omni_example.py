"""
Simple NASA OMNI Solar Wind and IMF Gap Analysis Example.

Demonstrates loading historical OMNI data, simulating gaps,
and visualizing interpolation benchmark errors.
"""

import os
import heliogap as hg


def main():
    print("Initializing NASA OMNI Gap Analysis...")

    # 1. Load data from local cache or download if unavailable
    cache_path = "omni_data.pkl" if os.path.exists("omni_data.pkl") else "../omni_data.pkl"
    df = hg.load_omni_data(cache_filepath=cache_path)

    if df is None or df.empty:
        print("Error: Could not load OMNI dataset.")
        return

    # 2. Selected solar wind and magnetic field parameters
    features = [
        'flow_speed', 'proton_density', 'T',
        'F', 'BX_GSE', 'BY_GSE', 'BZ_GSE',
        'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz'
    ]

    metric = 'wmape'
    gap_sizes = [1, 5, 15, 30, 60, 120]
    results = {}

    print(f"Evaluating gap interpolation performance using metric: {metric.upper()}...")

    # 3. Gap simulation across active features
    for feat in features:
        if feat in df.columns and df[feat].notna().any():
            gaps, errors = hg.evaluate_gaps(
                df=df,
                feature_name=feat,
                metric_name=metric,
                gap_sizes=gap_sizes,
                interpolation_method='linear'
            )
            results[feat] = errors
        else:
            print(f"Notice: Feature '{feat}' not present or contains only NaNs. Skipping.")

    # 4. Generate multi-panel visualization
    output_plot = "omni_gap_analysis.png"
    print(f"Generating benchmark plot: {output_plot}...")
    hg.plot_metric_matrix(
        gap_sizes=gap_sizes,
        results_dict=results,
        metric_name=metric,
        save_path=output_plot
    )
    print(f"Completed! Plot saved to '{output_plot}'.")


if __name__ == "__main__":
    main()
