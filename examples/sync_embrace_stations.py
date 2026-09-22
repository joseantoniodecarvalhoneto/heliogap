"""
Synchronize INPE EMBRACE Stations Datasets.

Checks local cache files, identifies missing stations or out-of-date records,
and downloads the missing temporal deltas from the server.
"""

from datetime import datetime
import os
import pandas as pd
import heliogap as hg


def sync_stations(start_year: int = 2015) -> None:
    """
    Synchronize all available stations from INPE EMBRACE network.

    Parameters:
        start_year (int): Initial year of extraction.
    """
    print("Initiating EMBRACE Network Synchronization...")

    current_year = datetime.now().year
    server_stations = hg.fetch_inpe_stations()

    if not server_stations:
        print("Warning: Could not fetch stations from server. Keeping local files.")
        return

    for station in server_stations:
        cache_filename = f"embrace_magnet_{station}_hist.pkl"

        # Case 1: Station does not exist locally
        if not os.path.exists(cache_filename):
            print(f"[{station}] Not cached locally. Downloading from {start_year}...")
            try:
                hg.download_embrace_data(station=station, start_year=start_year)
            except Exception as e:
                print(f"Error downloading {station}: {e}")
            continue

        # Case 2: Station exists -> verify freshness
        try:
            df_cache = pd.read_pickle(cache_filename)
            time_col = 'Time' if 'Time' in df_cache.columns else ('Tempo' if 'Tempo' in df_cache.columns else None)

            if df_cache.empty or time_col is None:
                print(f"[{station}] Cache file invalid or empty. Re-downloading...")
                hg.download_embrace_data(station=station, start_year=start_year)
                continue

            last_year = int(df_cache[time_col].dt.year.max())
            if last_year < current_year:
                print(f"[{station}] Updating from {last_year} to {current_year}...")
                hg.download_embrace_data(station=station, start_year=last_year)
            else:
                print(f"[{station}] Up to date (latest: {last_year}).")

        except Exception as e:
            print(f"Error verifying cache for {station}: {e}")


if __name__ == "__main__":
    sync_stations(start_year=2015)
