import numpy as np

# Importamos todas as funções que queremos testar da nossa biblioteca
from heliogap.metrics import (
    calculate_wmape,
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_mda
)

def test_calculate_mae():
    """Testa o Erro Absoluto Médio (MAE)."""
    # 1. PREPARAÇÃO (Given)
    y_real = np.array([10, 20, 30])
    y_interpolado = np.array([12, 18, 30]) 
    
    # 2. AÇÃO (When)
    # Erros absolutos: |10-12|=2, |20-18|=2, |30-30|=0. 
    # Média: (2 + 2 + 0) / 3 = 1.3333333...
    resultado = calculate_mae(y_real, y_interpolado)
    
    # 3. VERIFICAÇÃO (Then)
    assert np.isclose(resultado, 1.3333333)

def test_calculate_rmse():
    """Testa a Raiz do Erro Quadrático Médio (RMSE)."""
    y_real = np.array([0, 0, 0])
    y_interpolado = np.array([3, 4, 0])
    
    # Erros quadrados: 9, 16, 0. Soma = 25. 
    # Média = 25 / 3 = 8.333... Raiz = ~2.88675
    resultado = calculate_rmse(y_real, y_interpolado)
    
    assert np.isclose(resultado, 2.88675134)

def test_calculate_wmape():
    """Testa o Erro Percentual Absoluto Ponderado (WMAPE)."""
    y_real = np.array([10, -10, 20])
    y_interpolado = np.array([12, -8, 20])
    
    # Erros absolutos: 2, 2, 0. Soma = 4.
    # Sinal real absoluto: 10, 10, 20. Soma = 40.
    # WMAPE = (4 / 40) * 100 = 10.0%
    resultado = calculate_wmape(y_real, y_interpolado)
    
    assert np.isclose(resultado, 10.0)

def test_calculate_r2():
    """Testa o Coeficiente de Determinação (R²)."""
    y_real = np.array([10, 20, 30])
    y_interpolado = np.array([12, 18, 28])
    media_global = 20.0
    
    # Soma dos quadrados dos resíduos (ss_res) = 4 + 4 + 4 = 12
    # Soma total dos quadrados (ss_tot) = 100 + 0 + 100 = 200
    # R² = 1 - (12 / 200) = 1 - 0.06 = 0.94
    resultado = calculate_r2(y_real, y_interpolado, media_global)
    
    assert np.isclose(resultado, 0.94)

def test_calculate_mda():
    """Testa a Precisão Direcional Média (MDA)."""
    y_verdadeiro_atual = np.array([15, 25, 10])
    y_verdadeiro_anterior = np.array([10, 30, 5])
    
    # Direção Real (sinal da diferença):
    # 15 - 10 = Sobe (+1)
    # 25 - 30 = Desce (-1)
    # 10 - 5 = Sobe (+1)
    # Array de direção: [+1, -1, +1]
    
    # Vamos simular que a interpolação previu tudo a subir (+1)
    dir_interpolacao = 1 
    
    # Acertos: O 1º e o 3º ponto acertaram a direção.
    # Total de acertos = 2 de 3 = 66.666...%
    resultado = calculate_mda(y_verdadeiro_atual, y_verdadeiro_anterior, dir_interpolacao)
    
    assert np.isclose(resultado, 66.6666667)