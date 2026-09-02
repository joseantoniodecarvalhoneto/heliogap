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

def clean_nasa_fill_values(df, trim_edges=True):
    """
    Replace NASA OMNI fill values with NaN to prevent interpolation bias.
    Applies margin trimming (edge removal) to isolate sensor noise.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing raw OMNI data.
        trim_edges (bool): Se True, descarta 1 minuto antes e depois de falhas.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with widened gaps.
    """
    # 1. LIMPEZA DOS CÓDIGOS DE ERRO DA NASA E OUTLIERS FÍSICOS
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
        
    # 2. SISTEMA DE TRIMMING (Retirada de bordas espaciais)
    if trim_edges:
        # Colunas que queremos higienizar as bordas
        cols_to_trim = ['proton_density', 'F', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 
                        'BY_GSM', 'BZ_GSM', 'flow_speed', 'Vx', 'Vy', 'Vz', 'T']
        
        for col in cols_to_trim:
            if col in df.columns:
                # Localiza onde os buracos e erros da NASA já foram marcados
                is_gap = df[col].isna()
                
                # Desloca a máscara para remover 1 minuto antes e 1 minuto depois
                widened_gap = is_gap | is_gap.shift(1) | is_gap.shift(-1)
                
                # Aplica a tesoura de volta na coluna
                df.loc[widened_gap, col] = np.nan
                
    return df

def load_omni_data(cache_filepath="omni_data.pkl", start_year=1981):
    """
    Load or download historical NASA OMNI datasets.
    Features a Self-Healing Cache with Visual Progress Indicators.
    """
    current_year = datetime.now().year
    cached_df = None
    
    expected_years = set(range(start_year, current_year + 1))
    years_to_download = list(expected_years)
    
    if os.path.exists(cache_filepath):
        print(f"📦 Loading cached dataset from {cache_filepath}")
        cached_df = pd.read_pickle(cache_filepath)
        
        cached_years = set(cached_df['Tempo'].dt.year.unique())
        missing_years = expected_years - cached_years
        years_to_download = sorted(list(missing_years))
        
    if not years_to_download:
        return clean_nasa_fill_values(cached_df, trim_edges=True)
        
    print(f"⚠️ Missing/incomplete data detected. Initiating targeted download for years: {years_to_download}")
    downloaded_files = []
    
    # ---------------------------------------------------------
    # ETAPA 1: INDICADOR VISUAL DE DOWNLOAD DA NASA
    # ---------------------------------------------------------
    for year in years_to_download:
        print(f"📡 Requesting NASA CDAWeb data for year {year}...", end=" ", flush=True)
        try:
            time_range = [f'{year}-01-01', f'{year+1}-01-01']
            
            # TRAVA REMOVIDA: no_update=False garante o uso da internet
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
                print("✅ OK!")
            else:
                print("❌ No files found.")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error: {e}")

    # ---------------------------------------------------------
    # ETAPA 2: INDICADOR VISUAL DE EXTRAÇÃO DOS ARQUIVOS .CDF
    # ---------------------------------------------------------
    if downloaded_files:
        print(f"\n⚙️ Extracting data from {len(downloaded_files)} CDF files...")
        dataframes = []
        
        for idx, cdf_path in enumerate(downloaded_files):
            # Print sobrepondo a mesma linha (usando \r) para efeito de carregamento
            filename = os.path.basename(cdf_path)
            print(f"\r⏳ Processing file {idx+1}/{len(downloaded_files)}: {filename}...", end="", flush=True)
            
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
                
        print("\n✅ CDF Extraction complete! Merging datasets...")
        
        new_df = pd.concat(dataframes, ignore_index=True)
        
        if cached_df is not None:
            final_df = pd.concat([cached_df, new_df], ignore_index=True)
            final_df.sort_values('Tempo', inplace=True)
            final_df.drop_duplicates(subset=['Tempo'], keep='last', inplace=True)
            final_df.reset_index(drop=True, inplace=True)
        else:
            final_df = new_df
            final_df.sort_values('Tempo', inplace=True)
            final_df.reset_index(drop=True, inplace=True)
        
        print("🧹 Cleaning and formatting NASA fill values...")
        final_df = clean_nasa_fill_values(final_df, trim_edges=True)
        from .engine import downcast_dataframe
        final_df = downcast_dataframe(final_df, inplace=True)
        
        print("💾 Saving updated cache to disk...")
        final_df.to_pickle(cache_filepath)
        
        return final_df
    
    elif cached_df is not None:
        cleaned = clean_nasa_fill_values(cached_df, trim_edges=True)
        from .engine import downcast_dataframe
        return downcast_dataframe(cleaned, inplace=True)
    
    return None