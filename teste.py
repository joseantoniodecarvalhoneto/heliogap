import pyspedas
import cdflib
import pandas as pd
import numpy as np

print("--- A iniciar a extração híbrida no Google Colab ---")

# 2. O Download Inteligente: Pedimos ao PySpedas para APENAS DESCARREGAR o ficheiro
# Isto impede que o pytplot tente ler o ficheiro e quebre o processo.
arquivos_baixados = pyspedas.omni.data(trange=['2024-05-09', '2024-05-15'], datatype='1min', downloadonly=True)

if arquivos_baixados:
    # O PySpedas devolve uma lista com o caminho do ficheiro no disco
    caminho_do_arquivo = arquivos_baixados[0]
    print(f"\nSucesso! Ficheiro localizado em: {caminho_do_arquivo}")
    
    # 3. Leitura Bruta e Dinâmica: Abrir o ficheiro CDF
    print("A analisar o ficheiro e a extrair TODAS as variáveis disponíveis...")
    arquivo_cdf = cdflib.CDF(caminho_do_arquivo)
    
    # Extrair os tempos primeiro
    tempos_brutos = arquivo_cdf.varget('Epoch')
    tempos_datetime = pd.to_datetime(cdflib.cdfepoch.unixtime(tempos_brutos), unit='s')
    
    # Iniciar o dicionário da tabela com a coluna principal de Tempo
    dicionario_dados = {'Tempo': tempos_datetime}
    
    # 4. Obter a lista de todas as variáveis dentro do ficheiro
    info_cdf = arquivo_cdf.cdf_info()
    todas_variaveis = info_cdf.zVariables
    
    # 5. Laço de Repetição: Percorrer e extrair cada variável automaticamente
    for nome_var in todas_variaveis:
        if nome_var != 'Epoch': # Ignorar o tempo (já o extraímos)
            dados_var = arquivo_cdf.varget(nome_var)
            
            # Garantir que a variável tem o mesmo tamanho do tempo e é 1D (para caber na tabela)
            if isinstance(dados_var, np.ndarray) and len(dados_var) == len(tempos_brutos):
                if dados_var.ndim == 1: 
                    dicionario_dados[nome_var] = dados_var
    
    # 6. Construir a "Super Tabela" com todos os parâmetros
    df = pd.DataFrame(dicionario_dados)
    
    print(f"\n--- SUCESSO! TABELA COMPLETA COM {len(df.columns)} COLUNAS ---")
    
    # Pedir ao Pandas para não esconder colunas ao imprimir
    pd.set_option('display.max_columns', None) 
    print(df.head())
    
else:
    print("\nFalha: Nenhum ficheiro foi descarregado. Verifique a sua ligação à internet.")