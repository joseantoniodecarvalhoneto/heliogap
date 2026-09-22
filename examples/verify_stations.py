"""
Verify INPE EMBRACE Stations Availability.

Queries active magnetometer stations on the INPE server and
compares them against locally cached datasets.
"""

import os
import heliogap as hg


def main():
    print("Fetching available stations from INPE EMBRACE server...")
    server_stations = hg.fetch_inpe_stations()

    local_stations = [
        f.split('_')[2] for f in os.listdir()
        if f.startswith('embrace_magnet_') and f.endswith('_hist.pkl')
    ]

    missing = sorted(list(set(server_stations) - set(local_stations)))

    print(f"Total stations on server: {len(server_stations)}")
    print(f"Total downloaded stations: {len(local_stations)}")

    if not missing:
        print("All server stations are present locally.")
    else:
        print(f"Stations pending download: {missing}")


if __name__ == "__main__":
    main()
