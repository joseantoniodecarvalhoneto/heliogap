import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pyspedas
import cdflib
import warnings

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

def clean_nasa_fill_values(df):
    """
    Replace NASA OMNI fill values with NaN to prevent interpolation bias.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing raw OMNI data.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if 'proton_density' in df.columns:
        df['proton_density'] = df['proton_density'].where(df['proton_density'] < 200, np.nan)

    mag_columns = ['F', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM']
    for col in mag_columns:
        if col in df.columns:
            df[col] = df[col].where(df[col].abs() < 200, np.nan)
            
    vel_columns = ['flow_speed', 'Vx', 'Vy', 'Vz']
    for col in vel_columns:
        if col in df.columns:
            df[col] = df[col].where(df[col].abs() < 5000, np.nan)
            
    if 'T' in df.columns:
        df['T'] = df['T'].where(df['T'] < 8000000, np.nan)
        
    return df

def load_omni_data(cache_filepath="omni_data.pkl", start_year=1981):
    """
    Load or download historical NASA OMNI datasets.
    
    Parameters:
        cache_filepath (str): Path to the cache file.
        start_year (int): Year to start the download if cache is missing.
        
    Returns:
        pd.DataFrame: DataFrame ready for mathematical processing.
    """
    current_year = datetime.now().year
    cached_df = None
    download_start = start_year
    
    if os.path.exists(cache_filepath):
        print(f"Loading cached dataset from {cache_filepath}")
        cached_df = pd.read_pickle(cache_filepath)
        last_cache_year = cached_df['Tempo'].dt.year.max()
        
        if last_cache_year == current_year:
            download_start = current_year
        else:
            download_start = last_cache_year
            
    downloaded_files = []
    
    for year in range(download_start, current_year + 1):
        try:
            time_range = [f'{year}-01-01', f'{year+1}-01-01']
            skip_update = (year < current_year)
            
            files = pyspedas.omni.data(
                trange=time_range, 
                datatype='1min', 
                downloadonly=True, 
                no_update=skip_update
            )
            
            if files:
                for f in files:
                    if f not in downloaded_files:
                        downloaded_files.append(f)
            time.sleep(1)
        except Exception as e:
            print(f"Error downloading data for year {year}: {e}")

    if downloaded_files:
        dataframes = []
        for cdf_path in downloaded_files:
            try:
                cdf_file = cdflib.CDF(cdf_path)
                raw_times = cdf_file.varget('Epoch')
                datetime_times = pd.to_datetime(cdflib.cdfepoch.unixtime(raw_times), unit='s')
                data_dict = {'Tempo': datetime_times}
                
                cdf_info = cdf_file.cdf_info()
                for var_name in cdf_info.zVariables:
                    if var_name != 'Epoch':
                        var_data = cdf_file.varget(var_name)
                        if isinstance(var_data, np.ndarray) and len(var_data) == len(raw_times) and var_data.ndim == 1: 
                            data_dict[var_name] = var_data
                
                dataframes.append(pd.DataFrame(data_dict))
            except:
                continue
                
        new_df = pd.concat(dataframes, ignore_index=True)
        
        if cached_df is not None:
            final_df = pd.concat([cached_df, new_df], ignore_index=True)
            final_df.drop_duplicates(subset=['Tempo'], keep='last', inplace=True)
        else:
            final_df = new_df
        
        final_df = clean_nasa_fill_values(final_df)
        final_df.to_pickle(cache_filepath)
        return final_df
    
    elif cached_df is not None:
        return clean_nasa_fill_values(cached_df)
    
    return None