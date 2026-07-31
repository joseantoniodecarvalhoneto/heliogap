"""
Este é o ficheiro mágico que transforma uma pasta normal numa biblioteca Python.
Tudo o que você importar aqui ficará disponível quando o utilizador fizer `import spacegap`.
"""

# Importamos as funções dos nossos ficheiros internos para a raiz da biblioteca
from .metrics import (
    calculate_wmape,
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_mda
)
from .engine import run_exhaustive_analysis
from .data import load_omni_data, clean_nasa_fill_values
from .plotter import plot_metric_matrix

# Define o que é exportado se alguém fizer `from spacegap import *`
__all__ = [
    "calculate_wmape",
    "calculate_mae",
    "calculate_rmse",
    "calculate_r2",
    "calculate_mda",
    "run_exhaustive_analysis",
    "load_omni_data",
    "clean_nasa_fill_values",
    "plot_metric_matrix"
]