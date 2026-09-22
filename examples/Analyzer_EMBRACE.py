import os
import time
import heliogap as hg

def main():
    print("=" * 60)
    print("🚀 HELIOGAP: EMBRACE MASTER NETWORK ANALYZER 🚀")
    print("=" * 60)
    
    # Create an output directory to keep the generated plots organized
    output_dir = "embrace_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Configuration
    # We will test all Earth magnetic field components available
    features_to_test = ['H', 'D', 'Z', 'F']
    
    # We will stress-test the data with all mathematical perspectives
    metrics_to_test = ['mae', 'rmse', 'wmape', 'r2', 'mda']
    
    # Gap sizes in minutes
    gaps = [1, 2, 5, 10, 15, 30, 60, 120]
    
    # Start year (2015 is when the network became more robust, 
    # but you can change to 2005 if you want the absolute full history)
    start_year = 2015 
    
    print("\n📡 Fetching available stations from INPE servers...")
    stations = hg.fetch_inpe_stations()
    
    if not stations:
        print("Warning: Could not fetch station list from server. Using default known stations.")
        stations = ['VSS', 'CXP', 'SMS', 'SLZ', 'VJH', 'MAN', 'EUS', 'MED']
        
    print(f"✅ Found {len(stations)} stations to process: {', '.join(stations)}\n")

    for station_code in stations:
        print("=" * 60)
        print(f"🏢 PROCESSING STATION: {station_code}")
        print("=" * 60)
        
        # 1. Download/Load Data for the current station
        df = hg.download_embrace_data(station=station_code, start_year=start_year)
        
        if df is None or df.empty:
            print(f"❌ Skipping {station_code}: No valid data retrieved.")
            continue
            
        print(f"✅ Data ready for {station_code}. Shape: {df.shape}")
        
        for metric in metrics_to_test:
            print(f"\n🧠 Running engine for Metric: {metric.upper()}")
            
            results_dict = {}
            valid_gaps = gaps
            
            for feature in features_to_test:
                # Check if the feature actually exists and has data in this station
                if feature in df.columns and df[feature].notna().any():
                    gap_sizes, errors = hg.evaluate_gaps(
                        df=df,
                        feature_name=feature,
                        metric_name=metric,
                        gap_sizes=gaps
                    )
                    results_dict[feature] = errors
                    valid_gaps = gap_sizes
                else:
                    print(f"⚠️ Feature '{feature}' has no valid data in {station_code}. Skipping...")
            
            if results_dict:
                output_filename = os.path.join(
                    output_dir, 
                    f"embrace_{station_code}_{metric.upper()}.png"
                )
                
                print(f"🎨 Generating plot matrix for {station_code} - {metric.upper()}...")
                hg.plot_metric_matrix(
                    gap_sizes=valid_gaps,
                    results_dict=results_dict,
                    metric_name=metric,
                    save_path=output_filename
                )
                print(f"💾 Saved to '{output_filename}'")
            else:
                print(f"❌ No valid features were processed for {station_code} ({metric.upper()}).")
                
    print("\n" + "=" * 60)
    print(f"🎉 MASTER ANALYSIS COMPLETE! All plots saved in the '{output_dir}/' folder.")
    print("=" * 60)

if __name__ == "__main__":
    main()