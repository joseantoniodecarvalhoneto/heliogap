import os
import re
import time
from datetime import datetime
import pandas as pd
import numpy as np
import requests

def clean_embrace_data(df):
    """
    Clean and format magnetic field data from the EMBRACE MagNet network.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame containing telemetry data.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    magnet_columns = ['H', 'D', 'Z', 'F']
    
    for col in magnet_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].abs() < 50000, np.nan)
            
    return df

def _parse_inpe_file(file_content):
    """
    Parse raw text files from INPE, ignoring headers and extracting timeseries data.
    
    Parameters:
        file_content (str): The raw text content of the file.
        
    Returns:
        pd.DataFrame: Parsed data.
    """
    lines = file_content.split('\n')
    parsed_data = []
    
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
            
        parts = line.split()
        
        try:
            if len(parts) == 10 and len(parts[2]) == 4:
                day, month, year = parts[0], parts[1], parts[2]
                hour, minute = parts[3], parts[4]
                timestamp = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
                d_val, h_val, z_val, f_val = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[9])
                parsed_data.append({'Tempo': timestamp, 'H': h_val, 'D': d_val, 'Z': z_val, 'F': f_val})
                
            elif len(parts) >= 9 and len(parts[0]) == 4:
                timestamp = pd.to_datetime(f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}")
                h_val, d_val, z_val = float(parts[6]), float(parts[7]), float(parts[8])
                parsed_data.append({'Tempo': timestamp, 'H': h_val, 'D': d_val, 'Z': z_val})
                
            elif len(parts) >= 5 and ('-' in parts[0] or '/' in parts[0]):
                timestamp = pd.to_datetime(f"{parts[0]} {parts[1]}")
                h_val, d_val, z_val = float(parts[2]), float(parts[3]), float(parts[4])
                parsed_data.append({'Tempo': timestamp, 'H': h_val, 'D': d_val, 'Z': z_val})
                
        except (ValueError, IndexError):
            continue
            
    if parsed_data:
        return pd.DataFrame(parsed_data)
    return None

def load_local_embrace_data(filepath):
    """
    Load EMBRACE MagNet data from a local text file.
    
    Parameters:
        filepath (str): Path to the local file.
        
    Returns:
        pd.DataFrame: Processed local data.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        df = _parse_inpe_file(content)
        
        if df is not None:
            df.sort_values('Tempo', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = clean_embrace_data(df)
            return df
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def fetch_inpe_stations():
    """
    Retrieve available magnetic stations from the INPE server.
    
    Returns:
        list: Sorted list of station codes.
    """
    base_url = "https://embracedata.inpe.br/magnetometer/"
    
    try:
        response = requests.get(base_url, timeout=15)
        if response.status_code == 200:
            stations = re.findall(r'href="([A-Z]{3})/"', response.text)
            return sorted(list(set(stations)))
    except Exception as e:
        print(f"Error connecting to server: {e}")
    return []

def download_embrace_data(station='VSS', start_year=2015, end_year=None):
    """
    Download historical magnetic data for a specific station from INPE servers.
    
    Parameters:
        station (str): Station code (e.g., 'VSS').
        start_year (int): Initial year of extraction.
        end_year (int): Final year of extraction.
        
    Returns:
        pd.DataFrame: Aggregated historical data.
    """
    current_year = datetime.now().year
    target_end_year = end_year if end_year is not None else current_year
    cache_filename = f"embrace_magnet_{station}_hist.pkl"
    
    if os.path.exists(cache_filename):
        df_cache = pd.read_pickle(cache_filename)
        last_year = df_cache['Tempo'].dt.year.max()
        
        if last_year >= current_year:
            return df_cache
        else:
            start_year = last_year
    else:
        df_cache = None

    annual_dataframes = []
    
    for year in range(start_year, target_end_year + 1):
        year_url = f"https://embracedata.inpe.br/magnetometer/{station}/{year}/"
        print(f"Fetching directory: {year_url}")
        
        try:
            response = requests.get(year_url, timeout=15)
            if response.status_code != 200:
                continue
                
            links = re.findall(r'href="([^"]+)"', response.text)
            file_links = [lnk for lnk in links if not lnk.startswith('?') and not lnk.endswith('/') and not lnk.startswith('/')]
            
            if not file_links:
                continue
                
            for idx, filename in enumerate(file_links):
                print(f"\rDownloading file {idx+1}/{len(file_links)}: {filename}...", end="", flush=True)
                
                file_url = year_url + filename
                file_res = requests.get(file_url, timeout=10)
                
                if file_res.status_code == 200:
                    df_file = _parse_inpe_file(file_res.text)
                    if df_file is not None and not df_file.empty:
                        df_file = clean_embrace_data(df_file)
                        annual_dataframes.append(df_file)
            print() 
            
        except Exception as e:
            print(f"\nConnection error for year {year}: {e}")

    if annual_dataframes:
        new_df = pd.concat(annual_dataframes, ignore_index=True)
        
        if df_cache is not None:
            final_df = pd.concat([df_cache, new_df], ignore_index=True)
            final_df.drop_duplicates(subset=['Tempo'], keep='last', inplace=True)
        else:
            final_df = new_df
            
        final_df.sort_values('Tempo', inplace=True)
        final_df.to_pickle(cache_filename)
        return final_df
        
    elif df_cache is not None:
        return df_cache
        
    return None