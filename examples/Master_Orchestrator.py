import os
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import heliogap as hg


def ensure_directory(path: str) -> None:
    """Ensures destination directory exists before file generation."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    print("\n" + "="*70)
    print("🚀 HELIOGAP MASTER ORCHESTRATOR - HPC HIGH-PERFORMANCE PARALLEL MODE")
    print("="*70)

    # ==============================================================================
    # 1. CONFIGURATION & BENCHMARK MATRIX
    # ==============================================================================
    gap_sizes = [1, 2, 5, 10, 15, 30, 60, 120]
    metrics = ['wmape', 'mae', 'rmse', 'r2', 'mda']
    
    interpolation_methods = [
        'linear', 'nearest', 'zero', 'slinear', 
        'quadratic', 'cubic', 'pchip', 'akima'
    ]

    # ==============================================================================
    # FASE 1: NASA OMNI HISTORICAL DATABASE (1981 - CURRENT)
    # ==============================================================================
    print("\n📡 [FASE 1] Ingesting NASA OMNI Historical Dataset (1981+)...")
    
    df_omni = hg.load_omni_data(start_year=1981)
    
    if df_omni is not None and not df_omni.empty:
        # Enforce memory downcasting to float32 immediately
        df_omni = hg.downcast_dataframe(df_omni, inplace=True)
        print(f"✅ OMNI Dataset Loaded & Downcasted! Total rows: {len(df_omni):,}")
        
        features_omni = [
            'flow_speed', 'proton_density', 'T', 'F', 
            'BZ_GSM', 'BY_GSM', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 
            'Vx', 'Vy', 'Vz'
        ]

        # Extract 1D float32 arrays to isolate memory footprint
        omni_arrays = {}
        for feat in features_omni:
            if feat in df_omni.columns and df_omni[feat].notna().any():
                omni_arrays[feat] = df_omni[feat].to_numpy(dtype=np.float32, copy=False)
        
        # Free multi-column DataFrame from RAM
        del df_omni
        gc.collect()

        for method in tqdm(interpolation_methods, desc="OMNI Methods"):
            for metric in metrics:
                folder_path = f"Omni_Charts/Interpolation_{method}"
                ensure_directory(folder_path)
                file_path = os.path.join(folder_path, f"omni_{method}_{metric.upper()}.png")
                
                # Smart Skip: bypass compute if plot already rendered
                if os.path.exists(file_path):
                    continue
                
                results_dict_omni = {}
                
                for feature, arr in omni_arrays.items():
                    _, results = hg.evaluate_gaps(
                        df=arr, 
                        feature_name=feature, 
                        metric_name=metric,
                        gap_sizes=gap_sizes, 
                        interpolation_method=method,
                        n_jobs=-1,
                        verbose=False
                    )
                    results_dict_omni[feature] = results
                
                if results_dict_omni:
                    hg.plot_metric_matrix(
                        gap_sizes=gap_sizes, 
                        results_dict=results_dict_omni, 
                        metric_name=metric, 
                        save_path=file_path
                    )
                    plt.close('all')

                del results_dict_omni
                gc.collect()

        del omni_arrays
        gc.collect()
    else:
        print("❌ Failed to load NASA OMNI database.")

    # ==============================================================================
    # FASE 2: INPE EMBRACE NETWORK (2008 - CURRENT)
    # ==============================================================================
    print("\n" + "="*70)
    print("🌍 [FASE 2] Ingesting INPE EMBRACE MagNet Network (Historical)...")
    
    stations = hg.fetch_inpe_stations()
    print(f"📡 Found {len(stations)} stations on INPE MagNet.")
    
    features_embrace = ['H', 'D', 'Z', 'F']
    
    # Process one station at a time with strict memory deallocation protocol
    for st in stations:
        print(f"\n================ Processing Station: {st} ================")
        
        df_st = hg.download_embrace_data(station=st, start_year=2008)
            
        if df_st is not None and not df_st.empty:
            df_st = hg.downcast_dataframe(df_st, inplace=True)
            print(f"✅ Station {st} loaded! Total rows: {len(df_st):,}")
            
            # Extract 1D arrays and immediately free DataFrame
            st_arrays = {}
            for feat in features_embrace:
                if feat in df_st.columns and df_st[feat].notna().any():
                    st_arrays[feat] = df_st[feat].to_numpy(dtype=np.float32, copy=False)
            
            del df_st
            gc.collect()

            # Execute mathematical benchmark per method & metric
            for method in interpolation_methods:
                for metric in metrics:
                    folder_path = os.path.join("embrace_results", st, f"Interpolation_{method}")
                    ensure_directory(folder_path)
                    file_path = os.path.join(folder_path, f"embrace_{st}_{method}_{metric.upper()}.png")
                    
                    if os.path.exists(file_path):
                        continue
                    
                    results_dict_embrace = {}
                    
                    for feature, arr in st_arrays.items():
                        _, results = hg.evaluate_gaps(
                            df=arr, 
                            feature_name=feature, 
                            metric_name=metric,
                            gap_sizes=gap_sizes, 
                            interpolation_method=method,
                            n_jobs=-1,
                            verbose=False
                        )
                        results_dict_embrace[feature] = results
                    
                    if results_dict_embrace:
                        hg.plot_metric_matrix(
                            gap_sizes=gap_sizes, 
                            results_dict=results_dict_embrace, 
                            metric_name=metric, 
                            save_path=file_path
                        )
                        plt.close('all')

                    del results_dict_embrace
                    gc.collect()

            # Strict memory deallocation protocol after station completion
            del st_arrays
            gc.collect()
            
        else:
            print(f"⚠️ Insufficient telemetry data for station {st}.")

    print("\n🎉 HPC MASTER ORCHESTRATION COMPLETE! All data analyzed successfully.")


if __name__ == "__main__":
    main()