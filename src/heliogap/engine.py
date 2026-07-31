import time
import numpy as np
from .metrics import METRICS_MAP

def run_exhaustive_analysis(df, feature_name, metric_name, gap_sizes=[1, 2, 5, 10, 15, 30, 60, 120]):
    """
    O Motor de Força Bruta Vetorizada.
    Aplica a técnica de Janela Deslizante (Sliding Window) sobre todos os dados ininterruptos.
    """
    print(f"\n{'='*50}\nAnalyzing: {feature_name} using {metric_name.upper()}\n{'='*50}")
    
    # 1. Identificar blocos sem buracos (O 'Período de Ouro')
    is_valid = df[feature_name].notna()
    block_ids = (~is_valid).cumsum()[is_valid]
    
    # Extrai os valores crus para matrizes NumPy (Ultra rápido)
    segments = [group.values for _, group in df[feature_name][is_valid].groupby(block_ids)]
    
    print(f"Found {len(segments):,} continuous historical blocks.")
    
    results = []
    metric_func = METRICS_MAP.get(metric_name.lower())
    
    # Opcional: Variáveis globais para métricas que as exigem (como R²)
    global_mean = df[feature_name].mean() if metric_name.lower() == 'r2' else 0
    
    # 2. Iniciar a simulação exaustiva para cada tamanho de apagão
    for k in gap_sizes:
        start_time = time.time()
        
        # Acumuladores de erros
        accumulated_error = 0.0
        accumulated_signal = 0.0
        total_points_tested = 0
        
        for seg in segments:
            n_points = len(seg)
            
            # Só podemos induzir um gap de tamanho 'k' se o bloco for maior que 'k + 2'
            # (Precisamos de 1 ponto de âncora antes e 1 depois do buraco)
            if n_points >= k + 2:
                # Matrizes das âncoras esquerda e direita
                left_anchor = seg[:-k-1]
                right_anchor = seg[k+1:]
                
                # Para o MDA, pré-calculamos a direção da interpolação
                interpolation_dir = np.sign(right_anchor - left_anchor) if metric_name.lower() == 'mda' else None
                
                for step in range(1, k + 1):
                    # Percentagem da distância percorrida no buraco
                    alpha = step / (k + 1)
                    
                    # Interpolação linear vetorizada (milhões de cálculos simultâneos)
                    interpolated_window = left_anchor + (right_anchor - left_anchor) * alpha
                    true_window = seg[step : step + len(left_anchor)]
                    
                    # O motor delega o cálculo acumulativo baseado na métrica escolhida
                    if metric_name.lower() == 'wmape':
                        accumulated_error += np.sum(np.abs(true_window - interpolated_window))
                        accumulated_signal += np.sum(np.abs(true_window))
                        
                    elif metric_name.lower() == 'mae':
                        accumulated_error += np.sum(np.abs(true_window - interpolated_window))
                        
                    elif metric_name.lower() == 'rmse':
                        accumulated_error += np.sum((true_window - interpolated_window)**2)
                        
                    elif metric_name.lower() == 'r2':
                        accumulated_error += np.sum((true_window - interpolated_window)**2) # ss_res
                        accumulated_signal += np.sum((true_window - global_mean)**2)        # ss_tot
                        
                    elif metric_name.lower() == 'mda':
                        true_previous = seg[step - 1 : step - 1 + len(left_anchor)]
                        real_direction = np.sign(true_window - true_previous)
                        accumulated_error += np.sum(real_direction == interpolation_dir) # Acertos
                        
                    total_points_tested += len(true_window)
        
        # 3. Processamento Final após testar todos os milhões de segmentos para este 'k'
        final_result = 0
        if total_points_tested > 0:
            if metric_name.lower() == 'wmape':
                final_result = (accumulated_error / accumulated_signal) * 100 if accumulated_signal > 0 else 0
            elif metric_name.lower() == 'mae':
                final_result = accumulated_error / total_points_tested
            elif metric_name.lower() == 'rmse':
                final_result = np.sqrt(accumulated_error / total_points_tested)
            elif metric_name.lower() == 'r2':
                final_result = 1 - (accumulated_error / accumulated_signal) if accumulated_signal > 0 else 0
            elif metric_name.lower() == 'mda':
                final_result = (accumulated_error / total_points_tested) * 100
                
        results.append(final_result)
        elapsed_time = time.time() - start_time
        
        # Imprime o formato bonito no terminal que você gostou
        print(f"[Gap {k:3} min] Scenarios: {total_points_tested:11,} | {metric_name.upper()}: {final_result:8.2f} | Time: {elapsed_time:.2f}s")
        
    return gap_sizes, results