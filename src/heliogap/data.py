import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pyspedas
import cdflib

# Ignora avisos desnecessários do pyspedas
import warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

def clean_nasa_fill_values(df):
    """
    Remove os "Fill Values" da NASA (ex: 999.99) que destroem as interpolações.
    
    Args:
        df (pd.DataFrame): DataFrame com dados brutos OMNI.
        
    Returns:
        pd.DataFrame: DataFrame limpo, com NaNs reais onde haviam falsos positivos.
    """
    # 1. Densidade: O vento solar normal tem entre 1 e 50 protões/cm3. 
    # Um valor de 999.99 é o código de erro. Cortamos tudo acima de 200.
    if 'proton_density' in df.columns:
        df['proton_density'] = df['proton_density'].where(df['proton_density'] < 200, np.nan)

    # 2. Campo Magnético: Raramente excede os 50 nT. Cortamos os 999.99.
    mag_columns = ['F', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM']
    for col in mag_columns:
        if col in df.columns:
            df[col] = df[col].where(df[col].abs() < 200, np.nan)
            
    # 3. Velocidade e Temperatura
    vel_columns = ['flow_speed', 'Vx', 'Vy', 'Vz']
    for col in vel_columns:
        if col in df.columns:
            df[col] = df[col].where(df[col].abs() < 5000, np.nan)
            
    if 'T' in df.columns:
        df['T'] = df['T'].where(df['T'] < 8000000, np.nan)
        
    return df

def load_omni_data(cache_filepath="data.pkl", start_year=1981):
    """
    Carrega o dataset histórico OMNI. Se não existir, descarrega da NASA.
    Se já existir, faz a verificação inteligente do ano atual para atualizar.
    
    Args:
        cache_filepath (str): Caminho para o ficheiro de cache.
        start_year (int): Ano de início do download, caso não exista cache.
        
    Returns:
        pd.DataFrame: DataFrame pronto e limpo para processamento matemático.
    """
    current_year = datetime.now().year
    cached_df = None
    download_start = start_year
    
    # Verifica se já temos o ficheiro guardado localmente
    if os.path.exists(cache_filepath):
        print(f"Loading cache from {cache_filepath}...")
        cached_df = pd.read_pickle(cache_filepath)
        last_cache_year = cached_df['Tempo'].dt.year.max()
        
        # Se for do ano atual, já está atualizado
        if last_cache_year == current_year:
            print("Cache is up to date.")
            download_start = current_year
        else:
            print(f"Cache outdated (last year: {last_cache_year}). Resuming download...")
            download_start = last_cache_year
    else:
        print("No cache found. Starting full historical download...")

    downloaded_files = []
    
    # Laço de download por ano para não estourar a memória/timeout
    for year in range(download_start, current_year + 1):
        try:
            time_range = [f'{year}-01-01', f'{year+1}-01-01']
            skip_update = True if year < current_year else False
            
            # Chama a API da NASA via PySpedas
            # downloadonly=True: Apenas faz o download, não abre em memória RAM
            # no_update=skip_update: A mágica local! Se o ficheiro já existir na pasta local
            # da biblioteca (e não precisarmos de atualização do ano atual), ele IGNORA o 
            # servidor da NASA e usa o disco local instantaneamente.
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
            time.sleep(1) # Dá um respiro aos servidores da NASA
        except Exception as e:
            print(f"Warning during {year} download: {e}")

    # Processamento dos ficheiros CDF baixados
    if downloaded_files:
        tables_list = []
        print("Extracting data from CDF files...")
        
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
                        # Só adiciona matrizes de 1 dimensão que caibam perfeitamente na coluna de tempo
                        if isinstance(var_data, np.ndarray) and len(var_data) == len(raw_times) and var_data.ndim == 1: 
                            data_dict[var_name] = var_data
                
                tables_list.append(pd.DataFrame(data_dict))
            except:
                continue
                
        print("Merging and cleaning datasets...")
        new_df = pd.concat(tables_list, ignore_index=True)
        del tables_list 
        
        # Funde o que já tínhamos (cache) com os ficheiros novos baixados
        if cached_df is not None:
            final_df = pd.concat([cached_df, new_df], ignore_index=True)
            final_df.drop_duplicates(subset=['Tempo'], keep='last', inplace=True)
        else:
            final_df = new_df
        
        # Limpa os 999.99 usando a nossa função modular
        final_df = clean_nasa_fill_values(final_df)
            
        final_df.to_pickle(cache_filepath)
        print("Cache updated and saved successfully!")
        return final_df
    
    elif cached_df is not None:
        # Se não baixou nada e já tinha o cache, apenas aplica a limpeza (por precaução)
        # e devolve
        return clean_nasa_fill_values(cached_df)
    
    return None