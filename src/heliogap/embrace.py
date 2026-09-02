import os
import re
import time
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def clean_embrace_data(df, trim_edges=True):
    """
    Clean and format magnetic field data from the EMBRACE MagNet network.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame containing telemetry data.
        trim_edges (bool): Se True, descarta 1 minuto antes e depois de falhas/ruídos.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    magnet_columns = ['H', 'D', 'Z', 'F']
    
    for col in magnet_columns:
        if col in df.columns:
            # 1. Converte para numérico e limpa falhas bizarras (> 50000)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].abs() < 50000, np.nan)
            
            # 2. SISTEMA DE TRIMMING (Retirada de bordas do gap)
            if trim_edges:
                # Cria uma "máscara" apontando onde estão os buracos/erros atuais
                is_gap = df[col].isna()
                
                # Desloca a máscara para frente (+1 min) e para trás (-1 min)
                widened_gap = is_gap | is_gap.shift(1) | is_gap.shift(-1)
                
                # Aplica a nova máscara alargada de volta na coluna
                df.loc[widened_gap, col] = np.nan
            
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
            from .engine import downcast_dataframe
            df = downcast_dataframe(df, inplace=True)
            return df
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def get_session_with_retries():
    """
    Creates a robust requests session with automatic retries and backoff to handle
    temporary DNS or network failures (e.g. Temporary failure in name resolution).
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))
    return session

def fetch_inpe_stations():
    """
    Retrieve available magnetic stations from the INPE server.
    
    Returns:
        list: Sorted list of station codes.
    """
    base_url = "https://embracedata.inpe.br/magnetometer/"
    session = get_session_with_retries()
    
    try:
        response = session.get(base_url, timeout=15)
        if response.status_code == 200:
            stations = re.findall(r'href="([A-Z]{3})/"', response.text)
            return sorted(list(set(stations)))
    except Exception as e:
        print(f"Error connecting to server: {e}")
    return []

def download_embrace_data(station='VSS', start_year=2008, end_year=None):
    """
    Download historical magnetic data for a specific station from INPE servers.
    Features a Self-Healing Cache and a Robust Network Session.
    
    Parameters:
        station (str): Station code (default is 'VSS', overridden by the orchestrator).
        start_year (int): Initial year of extraction.
        end_year (int): Final year of extraction (defaults to current year).
        
    Returns:
        pd.DataFrame: Aggregated and self-healed historical data.
    """
    current_year = datetime.now().year
    target_end_year = end_year if end_year is not None else current_year
    cache_filename = f"embrace_magnet_{station}_hist.pkl"
    
    cached_df = None
    expected_years = set(range(start_year, target_end_year + 1))
    
    if os.path.exists(cache_filename):
        df_cache = pd.read_pickle(cache_filename)
        cached_df = df_cache
        
        cached_years = set(cached_df['Tempo'].dt.year.unique())
        missing_years = expected_years - cached_years
        years_to_download = sorted(list(missing_years))
    else:
        years_to_download = sorted(list(expected_years))
            
    if not years_to_download:
        return cached_df

    print(f"📡 {station}: Missing data detected. Initiating targeted download for years: {years_to_download}")
    annual_dataframes = []
    
    session = get_session_with_retries()
    
    for year in years_to_download:
        year_url = f"https://embracedata.inpe.br/magnetometer/{station}/{year}/"
        
        try:
            response = session.get(year_url, timeout=15)
            
            if response.status_code == 404:
                print(f"  ↳ ⚠️ {station} ({year}): Pulado (Pasta não existe no servidor do INPE - Erro 404)")
                continue
            elif response.status_code != 200:
                print(f"  ↳ ⚠️ {station} ({year}): Pulado (Falha de conexão com o INPE - Status {response.status_code})")
                continue
                
            links = re.findall(r'href="([^"]+)"', response.text)
            file_links = [lnk for lnk in links if not lnk.startswith('?') and not lnk.endswith('/') and not lnk.startswith('/')]
            
            if not file_links:
                print(f"  ↳ ⚠️ {station} ({year}): Pulado (A pasta existe, mas está VAZIA no servidor)")
                continue
                
            for idx, filename in enumerate(file_links):
                print(f"\rDownloading {station} ({year}) - file {idx+1}/{len(file_links)}: {filename}...", end="", flush=True)
                
                file_url = year_url + filename
                file_res = session.get(file_url, timeout=15)
                
                if file_res.status_code == 200:
                    df_file = _parse_inpe_file(file_res.text)
                    if df_file is not None and not df_file.empty:
                        # O Trimming é aplicado AQUI durante o download do novo arquivo
                        df_file = clean_embrace_data(df_file, trim_edges=True)
                        annual_dataframes.append(df_file)
            print() 
            
        except Exception as e:
            print(f"\nConnection error for year {year} at station {station}: {e}")

    if annual_dataframes:
        new_df = pd.concat(annual_dataframes, ignore_index=True)
        
        if cached_df is not None:
            final_df = pd.concat([cached_df, new_df], ignore_index=True)
            final_df.sort_values('Tempo', inplace=True)
            final_df.drop_duplicates(subset=['Tempo'], keep='last', inplace=True)
            final_df.reset_index(drop=True, inplace=True)
        else:
            final_df = new_df
            final_df.sort_values('Tempo', inplace=True)
            final_df.reset_index(drop=True, inplace=True)
            
        from .engine import downcast_dataframe
        final_df = downcast_dataframe(final_df, inplace=True)
        final_df.to_pickle(cache_filename)
        return final_df
        
    elif cached_df is not None:
        from .engine import downcast_dataframe
        return downcast_dataframe(cached_df, inplace=True)
        
    return None