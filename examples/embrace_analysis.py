"""
INPE EMBRACE Ground Station Gap Analysis Example.

Downloads or loads local EMBRACE magnetometer data and evaluates
interpolation accuracy across geomagnetic components.
"""

import heliogap as hg


def main():
    print("--- INPE EMBRACE Ground Station Gap Analysis ---")

    station_code = 'VSS'
    year = 2024

    print(f"1. Downloading/Loading data for station {station_code} ({year}+)...")
    df = hg.download_embrace_data(station=station_code, start_year=year)

    if df is None or df.empty:
        print("Error: Could not retrieve EMBRACE data.")
        return

    print("2. Running Gap Evaluation Engine...")
    feature = 'H'  # Horizontal Magnetic Field
    metric = 'mae'
    gap_sizes = [1, 2, 5, 10, 15, 30, 60, 120]

    gaps, errors = hg.evaluate_gaps(
        df=df,
        feature_name=feature,
        metric_name=metric,
        gap_sizes=gap_sizes,
        interpolation_method='linear'
    )

    print("3. Generating Output Plot...")
    output_filename = f"embrace_{station_code}_{metric}_analysis.png"
    hg.plot_metric_matrix(
        gap_sizes=gaps,
        results_dict={feature: errors},
        metric_name=metric,
        save_path=output_filename
    )
    print(f"Success! Analysis plot saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
