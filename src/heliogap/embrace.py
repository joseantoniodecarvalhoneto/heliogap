"""
Heliogap INPE EMBRACE Data Ingestion and Cleaning Module.

Provides automated ingestion, file parsing, and cache management for ground-based
magnetometer stations across South America operated by INPE's EMBRACE network.
"""

from typing import Optional, List
import os
import re
import time
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def clean_embrace_data(df: pd.DataFrame, trim_edges: bool = True) -> pd.DataFrame:
    """
    Clean and format magnetic field data from the EMBRACE MagNet network.
    Filters unphysical magnetic anomalies (> 50,000 nT) and applies optional
    1-minute boundary trimming around identified data gaps.

    Parameters:
        df (pd.DataFrame): Raw DataFrame containing telemetry data.
        trim_edges (bool): If True, discards 1 minute before and after failures/spikes.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df is None or df.empty:
        return df

    magnet_columns = ['H', 'D', 'Z', 'F']

    for col in magnet_columns:
        if col in df.columns:
            # Coerce numeric values and filter unphysical anomalies
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].abs() < 50000.0, np.nan)

            # Boundary trimming: extend gap mask by 1 minute in both directions
            if trim_edges:
                is_gap = df[col].isna()
                widened_gap = is_gap | is_gap.shift(1) | is_gap.shift(-1)
                df.loc[widened_gap, col] = np.nan

    return df


def parse_inpe_file(file_content: str) -> Optional[pd.DataFrame]:
    """
    Parse raw ASCII text files from INPE EMBRACE MagNet data servers.

    Parameters:
        file_content (str): The raw text content of the file.

    Returns:
        pd.DataFrame | None: Parsed data with 'Time'/'Tempo' and magnetic components.
    """
    if not file_content:
        return None

    lines = file_content.splitlines()
    parsed_data = []

    for line in lines:
        clean_l = line.strip()
        if not clean_l or not clean_l[0].isdigit():
            continue

        parts = clean_l.split()

        try:
            if len(parts) == 10 and len(parts[2]) == 4:
                day, month, year = parts[0], parts[1], parts[2]
                hour, minute = parts[3], parts[4]
                timestamp = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
                d_val, h_val, z_val, f_val = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[9])
                parsed_data.append({
                    'Time': timestamp, 'Tempo': timestamp,
                    'H': h_val, 'D': d_val, 'Z': z_val, 'F': f_val
                })

            elif len(parts) >= 9 and len(parts[0]) == 4:
                timestamp = pd.to_datetime(f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}")
                h_val, d_val, z_val = float(parts[6]), float(parts[7]), float(parts[8])
                parsed_data.append({
                    'Time': timestamp, 'Tempo': timestamp,
                    'H': h_val, 'D': d_val, 'Z': z_val
                })

            elif len(parts) >= 5 and ('-' in parts[0] or '/' in parts[0]):
                timestamp = pd.to_datetime(f"{parts[0]} {parts[1]}")
                h_val, d_val, z_val = float(parts[2]), float(parts[3]), float(parts[4])
                parsed_data.append({
                    'Time': timestamp, 'Tempo': timestamp,
                    'H': h_val, 'D': d_val, 'Z': z_val
                })

        except (ValueError, IndexError):
            continue

    if parsed_data:
        return pd.DataFrame(parsed_data)
    return None


# Backwards compatibility alias
_parse_inpe_file = parse_inpe_file


def load_local_embrace_data(filepath: str, trim_edges: bool = True) -> Optional[pd.DataFrame]:
    """
    Load EMBRACE MagNet data from a local ASCII text file.

    Parameters:
        filepath (str): Path to the local file.
        trim_edges (bool): Apply boundary trimming around missing samples.

    Returns:
        pd.DataFrame | None: Processed DataFrame or None if file not found.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        df = parse_inpe_file(content)
        if df is not None:
            time_col = 'Time' if 'Time' in df.columns else 'Tempo'
            df.sort_values(time_col, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = clean_embrace_data(df, trim_edges=trim_edges)
            from .engine import downcast_dataframe
            return downcast_dataframe(df, inplace=True)
        return None
    except Exception as e:
        print(f"Error loading local EMBRACE file: {e}")
        return None


def get_session_with_retries(retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """
    Create a requests Session configured with automatic retries and exponential backoff
    to handle transient DNS or network connection drops.

    Parameters:
        retries (int): Maximum number of retry attempts.
        backoff_factor (float): Multiplier for exponential backoff delay.

    Returns:
        requests.Session: Configured session object.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def fetch_inpe_stations() -> List[str]:
    """
    Retrieve the list of active magnetometer station identifiers from the INPE data portal.

    Returns:
        list[str]: Alphabetically sorted 3-letter station codes.
    """
    base_url = "https://embracedata.inpe.br/magnetometer/"
    session = get_session_with_retries()

    try:
        response = session.get(base_url, timeout=15)
        if response.status_code == 200:
            stations = re.findall(r'href="([A-Z]{3})/"', response.text)
            return sorted(list(set(stations)))
    except Exception as e:
        print(f"Error fetching station list: {e}")
    return []


def download_embrace_data(
    station: str = 'VSS',
    start_year: int = 2008,
    end_year: Optional[int] = None,
    trim_edges: bool = True
) -> Optional[pd.DataFrame]:
    """
    Download historical magnetometer data for a specific station from INPE EMBRACE servers.
    Implements incremental cache synchronization with local storage.

    Parameters:
        station (str): Station code (e.g. 'VSS', 'CXP', 'SMS', 'SLZ').
        start_year (int): Initial year of extraction.
        end_year (int, optional): Final year of extraction (defaults to current year).
        trim_edges (bool): Apply boundary trimming on gap edges.

    Returns:
        pd.DataFrame | None: Aggregated historical dataset.
    """
    station_code = station.upper()
    current_year = datetime.now().year
    target_end_year = end_year if end_year is not None else current_year
    cache_filename = f"embrace_magnet_{station_code}_hist.pkl"

    cached_df = None
    expected_years = set(range(start_year, target_end_year + 1))

    if os.path.exists(cache_filename):
        try:
            cached_df = pd.read_pickle(cache_filename)
            time_col = 'Time' if 'Time' in cached_df.columns else ('Tempo' if 'Tempo' in cached_df.columns else None)
            if time_col is not None:
                cached_years = set(cached_df[time_col].dt.year.unique())
                missing_years = sorted(list(expected_years - cached_years))
            else:
                missing_years = sorted(list(expected_years))
        except Exception:
            missing_years = sorted(list(expected_years))
    else:
        missing_years = sorted(list(expected_years))

    if not missing_years:
        if cached_df is not None:
            if 'Time' not in cached_df.columns and 'Tempo' in cached_df.columns:
                cached_df['Time'] = cached_df['Tempo']
            elif 'Tempo' not in cached_df.columns and 'Time' in cached_df.columns:
                cached_df['Tempo'] = cached_df['Time']
            from .engine import downcast_dataframe
            return downcast_dataframe(cached_df, inplace=True)
        return None

    print(f"📡 [{station_code}] Missing data detected. Initiating download for years: {missing_years}")
    annual_dataframes = []
    session = get_session_with_retries()

    for year in missing_years:
        year_url = f"https://embracedata.inpe.br/magnetometer/{station_code}/{year}/"

        try:
            response = session.get(year_url, timeout=15)

            if response.status_code == 404:
                print(f"  ↳ [{station_code}] ({year}): Skipped (Directory not found on INPE server - 404)")
                continue
            elif response.status_code != 200:
                print(f"  ↳ [{station_code}] ({year}): Skipped (Server returned HTTP {response.status_code})")
                continue

            links = re.findall(r'href="([^"]+)"', response.text)
            file_links = [
                lnk for lnk in links
                if not lnk.startswith('?') and not lnk.endswith('/') and not lnk.startswith('/')
            ]

            if not file_links:
                print(f"  ↳ [{station_code}] ({year}): Directory exists but is empty.")
                continue

            for idx, filename in enumerate(file_links):
                print(f"\r  [{station_code} - {year}] File {idx+1}/{len(file_links)}: {filename}...", end="", flush=True)

                file_url = year_url + filename
                file_res = session.get(file_url, timeout=15)

                if file_res.status_code == 200:
                    df_file = parse_inpe_file(file_res.text)
                    if df_file is not None and not df_file.empty:
                        df_file = clean_embrace_data(df_file, trim_edges=trim_edges)
                        annual_dataframes.append(df_file)
            print()

        except Exception as e:
            print(f"\nConnection error for year {year} at station {station_code}: {e}")

    if annual_dataframes:
        new_df = pd.concat(annual_dataframes, ignore_index=True)

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

        from .engine import downcast_dataframe
        final_df = downcast_dataframe(final_df, inplace=True)
        final_df.to_pickle(cache_filename)
        return final_df

    elif cached_df is not None:
        if 'Time' not in cached_df.columns and 'Tempo' in cached_df.columns:
            cached_df['Time'] = cached_df['Tempo']
        elif 'Tempo' not in cached_df.columns and 'Time' in cached_df.columns:
            cached_df['Tempo'] = cached_df['Time']
        from .engine import downcast_dataframe
        return downcast_dataframe(cached_df, inplace=True)

    return None