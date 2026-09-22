"""
Unified CPU Benchmark Orchestrator for Heliogap.
Races Sequential (1 core) vs Parallel (-1 cores) on real OMNI, EMBRACE, and INTERMAGNET datasets.
"""

import time
import os
import pandas as pd

# Core Engine
from heliogap.engine import evaluate_gaps

# Data Ingestors
from heliogap.omni import load_omni_data
from heliogap.embrace import download_embrace_data
try:
    from heliogap.intermagnet import download_intermagnet_vss
    HAS_INTERMAGNET = True
except ImportError:
    HAS_INTERMAGNET = False

def run_cpu_benchmark(df: pd.DataFrame, feature_name: str, dataset_name: str) -> dict:
    """
    Executes the micro-benchmark racing Parallel vs Sequential modes for a specific dataset.
    """
    print(f"\n{'='*60}")
    print(f"🚀 RACING ON DATASET: {dataset_name.upper()}")
    print(f"Dataset Size: {len(df):,} rows")
    print(f"{'='*60}")
    
    test_gaps = [1, 2, 5, 10, 15, 30, 60, 120] 
    
    # 1. Sequential Execution
    print("\n[Race 1/2] Running Sequential Mode (n_jobs=1)...")
    start_seq = time.perf_counter()
    evaluate_gaps(df, feature_name, gap_sizes=test_gaps, n_jobs=1, verbose=False)
    time_seq = time.perf_counter() - start_seq
    print(f"  ↳ Finished in: {time_seq:.4f} seconds")
    
    # 2. Parallel Execution
    print("\n[Race 2/2] Running Parallel Mode (n_jobs=-1)...")
    start_par = time.perf_counter()
    evaluate_gaps(df, feature_name, gap_sizes=test_gaps, n_jobs=-1, verbose=False)
    time_par = time.perf_counter() - start_par
    print(f"  ↳ Finished in: {time_par:.4f} seconds")
    
    # 3. Analytics
    winner = "Parallel" if time_par < time_seq else "Sequential"
    recommended_jobs = -1 if winner == "Parallel" else 1
    speed_diff = (time_seq / time_par) if winner == "Parallel" else (time_par / time_seq)
    
    print(f"\n🏆 WINNER FOR {dataset_name}: {winner} Mode ({speed_diff:.2f}x faster)")
    
    return {
        "Dataset": dataset_name,
        "Rows": len(df),
        "Sequential_Sec": round(time_seq, 4),
        "Parallel_Sec": round(time_par, 4),
        "Winner": winner,
        "Speed_Multiplier": f"{speed_diff:.2f}x",
        "Rec_n_jobs": recommended_jobs
    }

def export_benchmark_log(results_list: list, filename: str = "cpu_benchmark_log.txt"):
    """
    Exports the benchmark list of dictionaries to a .txt file, mimicking terminal output formatting.
    """
    df_log = pd.DataFrame(results_list)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("====================================================================\n")
        f.write("             HELIOGAP REAL-WORLD CPU BENCHMARK LOG                  \n")
        f.write("====================================================================\n\n")
        f.write(f"Host OS Cores Detected: {os.cpu_count()}\n\n")
        f.write(df_log.to_string(index=False))
        f.write("\n\n====================================================================\n")
        
    print(f"\n[Log] Benchmark results successfully exported to '{filename}'.")

def main():
    """Main orchestration flow for the unified real-data benchmark."""
    print("Initializing Real-Data CPU Benchmark...")
    print("Warning: Initial data loading may take time if caches are missing.\n")
    
    all_results = []

    # ---------------------------------------------------------
    # 1. NASA OMNI DATASET (1996+)
    # ---------------------------------------------------------
    print(">>> Loading NASA OMNI Data...")
    df_omni = load_omni_data(start_year=1996)
    if df_omni is not None and not df_omni.empty and 'F' in df_omni.columns:
        res_omni = run_cpu_benchmark(df_omni, feature_name='F', dataset_name='NASA OMNI (1996+)')
        all_results.append(res_omni)

    # ---------------------------------------------------------
    # 2. INPE EMBRACE DATASET (VSS, 2008+)
    # ---------------------------------------------------------
    print("\n>>> Loading INPE EMBRACE Data (VSS)...")
    df_embrace = download_embrace_data(station='VSS', start_year=2008)
    if df_embrace is not None and not df_embrace.empty and 'F' in df_embrace.columns:
        res_embrace = run_cpu_benchmark(df_embrace, feature_name='F', dataset_name='EMBRACE VSS (2008+)')
        all_results.append(res_embrace)

    # ---------------------------------------------------------
    # 3. INTERMAGNET DATASET (VSS, 1996+)
    # ---------------------------------------------------------
    if HAS_INTERMAGNET:
        print("\n>>> Loading INTERMAGNET Data (VSS)...")
        df_intermagnet = download_intermagnet_vss(start_year=1996)
        if df_intermagnet is not None and not df_intermagnet.empty and 'F' in df_intermagnet.columns:
            res_intermagnet = run_cpu_benchmark(df_intermagnet, feature_name='F', dataset_name='INTERMAGNET VSS (1996+)')
            all_results.append(res_intermagnet)
    else:
        print("\n[Skip] INTERMAGNET module not found. Skipping VSS 1996+ test.")

    # ---------------------------------------------------------
    # EXPORT RESULTS
    # ---------------------------------------------------------
    if all_results:
        export_benchmark_log(all_results)
    else:
        print("\n[Error] No datasets were loaded successfully. Check network or cache files.")
        
    print("\nUnified Orchestration complete.")

if __name__ == "__main__":
    main()