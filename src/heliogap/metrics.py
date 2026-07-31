import numpy as np

def calculate_wmape(true_values, interpolated_values):
    """
    Calcula o Erro Percentual Absoluto Ponderado (WMAPE).
    Excelente para variáveis que cruzam o zero, como o campo magnético (Bz).
    """
    # Soma dos erros absolutos entre o real e a matemática
    error_sum = np.sum(np.abs(true_values - interpolated_values))
    # Soma absoluta do sinal original para ser o denominador
    signal_sum = np.sum(np.abs(true_values))
    
    # Previne a divisão por zero caso a soma do sinal seja zero
    if signal_sum > 0:
        return (error_sum / signal_sum) * 100
    return 0

def calculate_mae(true_values, interpolated_values):
    """
    Calcula o Erro Absoluto Médio (MAE).
    Devolve o erro nas unidades físicas originais (ex: km/s ou nT).
    """
    return np.mean(np.abs(true_values - interpolated_values))

def calculate_rmse(true_values, interpolated_values):
    """
    Calcula a Raiz do Erro Quadrático Médio (RMSE).
    Pune severamente os erros nos picos das tempestades solares.
    """
    return np.sqrt(np.mean((true_values - interpolated_values)**2))

def calculate_r2(true_values, interpolated_values, global_mean):
    """
    Calcula o Coeficiente de Determinação (R²).
    Se for negativo, a interpolação foi pior que usar simplesmente a média.
    """
    # Soma dos quadrados dos resíduos (erros)
    ss_res = np.sum((true_values - interpolated_values)**2)
    # Soma total dos quadrados (variância real)
    ss_tot = np.sum((true_values - global_mean)**2)
    
    if ss_tot > 0:
        return 1 - (ss_res / ss_tot)
    return 0

def calculate_mda(true_values_current, true_values_previous, interpolation_dir):
    """
    Calcula a Precisão Direcional Média (MDA).
    Verifica se a reta adivinhou a direção correta da subida/descida.
    """
    # Extrai o sinal (+1 ou -1) da diferença entre o ponto atual e o anterior
    real_direction = np.sign(true_values_current - true_values_previous)
    
    # Conta quantos acertos de direção ocorreram
    correct_directions = np.sum(real_direction == interpolation_dir)
    total_points = len(true_values_current)
    
    if total_points > 0:
        return (correct_directions / total_points) * 100
    return 0

# Dicionário de mapeamento para facilitar a chamada das funções por nome no Motor Principal
METRICS_MAP = {
    'wmape': calculate_wmape,
    'mae': calculate_mae,
    'rmse': calculate_rmse,
    # R2 e MDA requerem parâmetros especiais, lidaremos com eles no motor
}