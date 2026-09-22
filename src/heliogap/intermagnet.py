"""
Heliogap INTERMAGNET Data Ingestion Module.

High-resolution (1-minute) geomagnetic data ingestion from the INTERMAGNET
global observatory network via the BGS GIN Web Services.
"""

from typing import Optional, List
import os
import io
import time
from datetime import datetime
import pandas as pd
import numpy as np

from .embrace import get_session_with_retries
from .engine import downcast_dataframe

# INTERMAGNET GIN Web Service Base URL (British Geological Survey)
INTERMAGNET_GIN_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"


def parse_iaga2002(content: str) -> Optional[pd.DataFrame]:
    """
    Parse raw IAGA-2002 formatted ASCII text into a cleaned pandas DataFrame.

    Extracts header metadata, parses datetime into 'Time' and 'Tempo',
    identifies geomagnetic component columns (X, Y, Z, F, G, H, D),
    and converts standard INTERMAGNET fill values (88888.00, 99999.00) to np.nan.

    Parameters:
        content (str): Raw string content in IAGA-2002 format.

    Returns:
        pd.DataFrame | None: Parsed DataFrame with 'Time'/'Tempo' and magnetic columns.
    """
    if not content or not isinstance(content, str):
        return None

    lines = content.splitlines()
    header_idx = None

    for i, line in enumerate(lines):
        clean_line = line.strip()
        if clean_line.upper().startswith("DATE") and "TIME" in clean_line.upper():
            header_idx = i
            break

    if header_idx is None:
        return None

    header_line = lines[header_idx].replace("|", "").strip()
    raw_col_names = header_line.split()

    data_lines = [
        l for l in lines[header_idx + 1:]
        if l.strip() and not l.strip().startswith("#") and not l.strip().endswith("|")
    ]

    if not data_lines:
        return None

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=r"\s+",
            header=None,
            engine="c"
        )
    except Exception:
        return None

    if df.empty or df.shape[1] < 4:
        return None

    # Construct datetime timestamp from Date (col 0) and Time (col 1)
    timestamp = pd.to_datetime(df[0].astype(str) + " " + df[1].astype(str), errors="coerce")
    df["Time"] = timestamp
    df["Tempo"] = timestamp

    comp_names = raw_col_names[3:] if len(raw_col_names) > 3 else []
    cleaned_cols = {}

    for idx, col_idx in enumerate(range(3, min(df.shape[1], 3 + len(comp_names)))):
        orig_name = comp_names[idx] if idx < len(comp_names) else f"C{idx+1}"
        comp_letter = orig_name[-1].upper() if orig_name else f"C{idx+1}"
        cleaned_cols[col_idx] = comp_letter

    df.rename(columns=cleaned_cols, inplace=True)

    selected_cols = ["Time", "Tempo"] + list(cleaned_cols.values())
    df = df[[c for c in selected_cols if c in df.columns]]

    # Replace INTERMAGNET missing/error values
    numeric_cols = [c for c in df.columns if c not in ("Time", "Tempo")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(df[col].abs() < 88888.0, np.nan)

    return df


def clean_intermagnet_data(df: pd.DataFrame, trim_edges: bool = True) -> pd.DataFrame:
    """
    Clean geomagnetic data from INTERMAGNET observatories and apply edge trimming.

    Parameters:
        df (pd.DataFrame): DataFrame containing geomagnetic measurements.
        trim_edges (bool): If True, discards 1 minute before and after any data gap.

    Returns:
        pd.DataFrame: Sanitized DataFrame.
    """
    if df is None or df.empty:
        return df

    numeric_cols = [c for c in df.columns if c not in ("Time", "Tempo")]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(df[col].abs() < 88888.0, np.nan)

        if trim_edges:
            is_gap = df[col].isna()
            widened_gap = is_gap | is_gap.shift(1) | is_gap.shift(-1)
            df.loc[widened_gap, col] = np.nan

    return df


def _fetch_intermagnet_year(
    session,
    station: str,
    year: int,
    trim_edges: bool = True
) -> List[pd.DataFrame]:
    """
    Retrieve one full year of 1-minute data in monthly batches.

    Parameters:
        session: Active requests.Session configured with retries and backoff.
        station (str): 3-letter IAGA observatory code.
        year (int): Target year.
        trim_edges (bool): Whether to apply edge trimming to each batch.

    Returns:
        list[pd.DataFrame]: List of parsed monthly DataFrames.
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    max_month = current_month if year == current_year else 12

    monthly_dfs = []

    for month in range(1, max_month + 1):
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"

        print(f"\r📡 [{station}] Downloading {year} - Month {month:02d}/{max_month:02d} ({start_date})...", end="", flush=True)

        params = {
            "request": "GetData",
            "observatoryIagaCode": station.upper(),
            "dataStartDate": start_date,
            "dataEndDate": end_date,
            "format": "iaga2002",
            "samplesPerDay": "minute"
        }

        try:
            res = session.get(INTERMAGNET_GIN_URL, params=params, timeout=30)
            if res.status_code == 200:
                df_month = parse_iaga2002(res.text)
                if df_month is not None and not df_month.empty:
                    df_month = clean_intermagnet_data(df_month, trim_edges=trim_edges)
                    monthly_dfs.append(df_month)
            else:
                print(f"\n⚠️ [{station}] HTTP {res.status_code} for {start_date}")
        except Exception as e:
            print(f"\n⚠️ [{station}] Error fetching {start_date}: {e}")

        time.sleep(0.3)

    print()
    return monthly_dfs


def download_intermagnet_data(
    station: str = "VSS",
    start_year: int = 1996,
    end_year: Optional[int] = None,
    cache_filepath: Optional[str] = None,
    trim_edges: bool = True
) -> pd.DataFrame:
    """
    Download and aggregate historical geomagnetic time series from INTERMAGNET.

    Parameters:
        station (str): 3-letter IAGA observatory code (default 'VSS').
        start_year (int): Initial year (default 1996).
        end_year (int, optional): Final year (defaults to current year).
        cache_filepath (str, optional): Target cache path. Defaults to '{station}_intermagnet_cache.pkl'.
        trim_edges (bool): Apply boundary gap trimming.

    Returns:
        pd.DataFrame: Cleaned, downcasted DataFrame with 'Time'/'Tempo' and magnetic fields.
    """
    current_year = datetime.now().year
    target_end_year = end_year if end_year is not None else current_year

    if start_year > target_end_year:
        raise ValueError(f"start_year ({start_year}) cannot be greater than end_year ({target_end_year}).")

    station_code = station.upper()
    if cache_filepath is None:
        cache_filepath = f"{station_code.lower()}_intermagnet_cache.pkl"

    expected_years = set(range(start_year, target_end_year + 1))
    cached_df = None

    if os.path.exists(cache_filepath):
        print(f"📦 Loading cached INTERMAGNET dataset from {cache_filepath}...")
        try:
            cached_df = pd.read_pickle(cache_filepath)
            time_col = 'Time' if 'Time' in cached_df.columns else ('Tempo' if 'Tempo' in cached_df.columns else None)
            if time_col is not None:
                cached_years = set(cached_df[time_col].dt.year.unique())
                missing_years = sorted(list(expected_years - cached_years))
            else:
                missing_years = sorted(list(expected_years))
        except Exception as e:
            print(f"⚠️ Cache read error ({e}). Rebuilding dataset...")
            missing_years = sorted(list(expected_years))
    else:
        missing_years = sorted(list(expected_years))

    if not missing_years and cached_df is not None:
        if 'Time' not in cached_df.columns and 'Tempo' in cached_df.columns:
            cached_df['Time'] = cached_df['Tempo']
        elif 'Tempo' not in cached_df.columns and 'Time' in cached_df.columns:
            cached_df['Tempo'] = cached_df['Time']
        return downcast_dataframe(cached_df, inplace=True)

    print(f"📡 [{station_code}] Downloading missing data for years: {missing_years}")
    session = get_session_with_retries()
    downloaded_dfs = []

    for yr in missing_years:
        year_data = _fetch_intermagnet_year(session, station_code, yr, trim_edges=trim_edges)
        downloaded_dfs.extend(year_data)

    if downloaded_dfs:
        new_df = pd.concat(downloaded_dfs, ignore_index=True)

        if cached_df is not None:
            final_df = pd.concat([cached_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        time_col = 'Time' if 'Time' in final_df.columns else 'Tempo'
        final_df.sort_values(time_col, inplace=True)
        final_df.drop_duplicates(subset=[time_col], keep="last", inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        if 'Time' not in final_df.columns:
            final_df['Time'] = final_df['Tempo']
        if 'Tempo' not in final_df.columns:
            final_df['Tempo'] = final_df['Time']

        final_df = downcast_dataframe(final_df, inplace=True)
        print(f"💾 Saving updated cache to {cache_filepath}...")
        final_df.to_pickle(cache_filepath)
        return final_df

    elif cached_df is not None:
        if 'Time' not in cached_df.columns and 'Tempo' in cached_df.columns:
            cached_df['Time'] = cached_df['Tempo']
        elif 'Tempo' not in cached_df.columns and 'Time' in cached_df.columns:
            cached_df['Tempo'] = cached_df['Time']
        return downcast_dataframe(cached_df, inplace=True)

    return pd.DataFrame(columns=["Time", "Tempo"])


def download_intermagnet_vss(
    start_year: int = 1996,
    end_year: Optional[int] = None,
    cache_filepath: str = "vss_intermagnet_cache.pkl",
    trim_edges: bool = True
) -> pd.DataFrame:
    """
    Download and process 1-minute geomagnetic data for Vassouras Observatory (VSS).

    Strict Temporal Filter:
    Enforces a minimum start year of 1996 to align with the NASA OMNI dataset.

    Parameters:
        start_year (int): Initial year (strictly >= 1996).
        end_year (int, optional): Final year (defaults to current year).
        cache_filepath (str): Path to local cache file.
        trim_edges (bool): Apply boundary trimming (default True).

    Returns:
        pd.DataFrame: Cleaned DataFrame for Vassouras.
    """
    effective_start = max(1996, start_year)
    return download_intermagnet_data(
        station="VSS",
        start_year=effective_start,
        end_year=end_year,
        cache_filepath=cache_filepath,
        trim_edges=trim_edges
    )


def load_local_intermagnet_data(filepath: str, trim_edges: bool = True) -> Optional[pd.DataFrame]:
    """
    Load and parse a local IAGA-2002 file.

    Parameters:
        filepath (str): Path to local IAGA-2002 text file.
        trim_edges (bool): Whether to apply edge trimming.

    Returns:
        pd.DataFrame | None: Processed DataFrame or None if file not found/invalid.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        df = parse_iaga2002(content)
        if df is not None and not df.empty:
            time_col = 'Time' if 'Time' in df.columns else 'Tempo'
            df.sort_values(time_col, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = clean_intermagnet_data(df, trim_edges=trim_edges)
            return downcast_dataframe(df, inplace=True)
        return None
    except Exception as e:
        print(f"Error loading local file {filepath}: {e}")
        return None
