"""
Heliogap NASA OMNI Data Ingestion and Cleaning Module.

Provides automated download, caching, and physical value sanitation
for high-resolution (1-minute) NASA OMNI solar wind and IMF datasets.
"""

from typing import Optional, List
import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


def clean_omni_fill_values(df: pd.DataFrame, trim_edges: bool = True) -> pd.DataFrame:
    """
    Replace NASA OMNI fill values and unphysical outliers with NaN.
    Optionally applies boundary trimming (1-minute edge removal around gaps)
    to isolate sensor transition anomalies.

    Parameters:
        df (pd.DataFrame): DataFrame containing raw OMNI data.
        trim_edges (bool): If True, discards 1 minute before and after identified gaps.

    Returns:
        pd.DataFrame: Sanitized DataFrame.
    """
    if df is None or df.empty:
        return df

    # Physical thresholds for solar wind plasma parameters
    if 'proton_density' in df.columns:
        df['proton_density'] = df['proton_density'].where(df['proton_density'] < 200, np.nan)

    # Interplanetary magnetic field components (nT)
    mag_columns = ['F', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM']
    for col in mag_columns:
        if col in df.columns:
            df[col] = df[col].where(df[col].abs() < 200, np.nan)

    # Plasma velocity components (km/s)
    vel_columns = ['flow_speed', 'Vx', 'Vy', 'Vz']
    for col in vel_columns:
        if col in df.columns:
            df[col] = df[col].where(df[col].abs() < 5000, np.nan)

    # Temperature (K)
    if 'T' in df.columns:
        df['T'] = df['T'].where(df['T'] < 8000000, np.nan)

    # Boundary trimming around missing segments
    if trim_edges:
        cols_to_trim = [
            'proton_density', 'F', 'BX_GSE', 'BY_GSE', 'BZ_GSE',
            'BY_GSM', 'BZ_GSM', 'flow_speed', 'Vx', 'Vy', 'Vz', 'T'
        ]
        for col in cols_to_trim:
            if col in df.columns:
                is_gap = df[col].isna()
                widened_gap = is_gap | is_gap.shift(1) | is_gap.shift(-1)
                df.loc[widened_gap, col] = np.nan

    return df


# Backwards compatibility alias
clean_nasa_fill_values = clean_omni_fill_values


def load_omni_data(
    cache_filepath: str = "omni_data.pkl",
    start_year: int = 1981,
    trim_edges: bool = True
) -> Optional[pd.DataFrame]:
    """
    Load or incrementally download historical NASA OMNI datasets.

    Checks local cache for existing years, downloads missing temporal ranges
    from NASA CDAWeb via pyspedas, extracts CDF variables, sanitizes values,
    and updates the local cache.

    Parameters:
        cache_filepath (str): Local file path for data cache (.pkl).
        start_year (int): Initial year of extraction (default: 1981).
        trim_edges (bool): Whether to apply edge trimming on fill values.

    Returns:
        pd.DataFrame | None: Cleaned DataFrame with 'Time' and physical features.
    """
    current_year = datetime.now().year
    cached_df = None

    expected_years = set(range(start_year, current_year + 1))
    years_to_download = list(expected_years)

    if os.path.exists(cache_filepath):
        print(f"📦 Loading cached OMNI dataset from {cache_filepath}...")
        try:
            cached_df = pd.read_pickle(cache_filepath)
            time_col = 'Time' if 'Time' in cached_df.columns else ('Tempo' if 'Tempo' in cached_df.columns else None)
            if time_col is not None:
                cached_years = set(cached_df[time_col].dt.year.unique())
                missing_years = expected_years - cached_years
                years_to_download = sorted(list(missing_years))
        except Exception as e:
            print(f"⚠️ Cache read error ({e}). Rebuilding dataset...")
            years_to_download = sorted(list(expected_years))

    if not years_to_download:
        if cached_df is not None:
            # Ensure both Time and Tempo columns are available
            if 'Time' not in cached_df.columns and 'Tempo' in cached_df.columns:
                cached_df['Time'] = cached_df['Tempo']
            elif 'Tempo' not in cached_df.columns and 'Time' in cached_df.columns:
                cached_df['Tempo'] = cached_df['Time']
        return clean_omni_fill_values(cached_df, trim_edges=trim_edges)

    print(f"📡 Downloading OMNI data for years: {years_to_download}")
    downloaded_files: List[str] = []

    # 1. Download CDF files via pyspedas
    import pyspedas
    for year in years_to_download:
        print(f"  ↳ Requesting NASA CDAWeb data for year {year}...", end=" ", flush=True)
        try:
            time_range = [f'{year}-01-01', f'{year+1}-01-01']
            files = pyspedas.omni.data(
                trange=time_range,
                datatype='1min',
                downloadonly=True,
                no_update=False
            )
            if files:
                for f in files:
                    if f not in downloaded_files:
                        downloaded_files.append(f)
                print("OK")
            else:
                print("No files found")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")

    # 2. Extract CDF files
    if downloaded_files:
        import cdflib
        print(f"⚙️ Parsing {len(downloaded_files)} CDF files...")
        dataframes = []

        for idx, cdf_path in enumerate(downloaded_files):
            filename = os.path.basename(cdf_path)
            print(f"\r  [{idx+1}/{len(downloaded_files)}] Extracting {filename}...", end="", flush=True)

            try:
                cdf_file = cdflib.CDF(cdf_path)
                raw_times = cdf_file.varget('Epoch')
                datetime_times = pd.to_datetime(cdflib.cdfepoch.unixtime(raw_times), unit='s')
                data_dict = {
                    'Time': datetime_times,
                    'Tempo': datetime_times
                }

                cdf_info = cdf_file.cdf_info()
                for var_name in cdf_info.zVariables:
                    if var_name != 'Epoch':
                        var_data = cdf_file.varget(var_name)
                        if isinstance(var_data, np.ndarray) and len(var_data) == len(raw_times) and var_data.ndim == 1:
                            data_dict[var_name] = var_data

                dataframes.append(pd.DataFrame(data_dict))
            except Exception:
                continue

        print("\n✅ Extraction complete. Merging records...")
        new_df = pd.concat(dataframes, ignore_index=True)

        if cached_df is not None:
            final_df = pd.concat([cached_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        time_col = 'Time' if 'Time' in final_df.columns else 'Tempo'
        final_df.sort_values(time_col, inplace=True)
        final_df.drop_duplicates(subset=[time_col], keep='last', inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        if 'Time' not in final_df.columns:
            final_df['Time'] = final_df['Tempo']
        if 'Tempo' not in final_df.columns:
            final_df['Tempo'] = final_df['Time']

        final_df = clean_omni_fill_values(final_df, trim_edges=trim_edges)
        from .engine import downcast_dataframe
        final_df = downcast_dataframe(final_df, inplace=True)

        print(f"💾 Saving updated cache to {cache_filepath}...")
        final_df.to_pickle(cache_filepath)
        return final_df

    elif cached_df is not None:
        if 'Time' not in cached_df.columns and 'Tempo' in cached_df.columns:
            cached_df['Time'] = cached_df['Tempo']
        elif 'Tempo' not in cached_df.columns and 'Time' in cached_df.columns:
            cached_df['Tempo'] = cached_df['Time']
        cleaned = clean_omni_fill_values(cached_df, trim_edges=trim_edges)
        from .engine import downcast_dataframe
        return downcast_dataframe(cleaned, inplace=True)

    return None