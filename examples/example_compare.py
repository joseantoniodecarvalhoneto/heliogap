import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import heliogap as hg
import warnings

warnings.filterwarnings('ignore')

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_methods_comparison(all_results, metric_name, title_prefix, save_path, gap_sizes):
    """
    Gera um painel com subgráficos (um para cada variável física).
    Em cada gráfico, plota uma linha para cada método de interpolação.
    """
    features = [f for f in all_results.keys() if all_results[f]] # Filtra variáveis vazias
    if not features:
        return
        
    num_features = len(features)
    cols = 2 if num_features > 1 else 1
    rows = (num_features + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    
    # Ajusta o formato do axes para ser sempre iterável
    if num_features == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
        
    # Cores e estilos para diferenciar bem as 8 linhas
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        methods_data = all_results[feature]
        
        for color_idx, (method, errors) in enumerate(methods_data.items()):
            # Plota a linha de cada método para esta variável física
            ax.plot(gap_sizes, errors, marker='o', linewidth=2, 
                    label=method.capitalize(), color=colors[color_idx])
            
        ax.set_title(f"Variável: {feature}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Tamanho do Gap (minutos)", fontsize=12)
        ax.set_ylabel(f"Erro ({metric_name.upper()})", fontsize=12)
        ax.legend(title="Métodos", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Se for R2, o eixo Y vai até 1.0 (melhor cenário). Se for erro, quanto menor, melhor.
        if metric_name.lower() == 'r2':
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=1.05)

    plt.tight_layout()
    plt.suptitle(f"{title_prefix} - Comparativo de Interpolação ({metric_name.upper()})", 
                 y=1.02, fontsize=18, fontweight='bold')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    print("\n" + "="*70)
    print("🚀 HELIOGAP - COMPARATIVE ANALYSIS MODE")
    print("="*70)

    gap_sizes = [1, 2, 5, 10, 15, 30, 60, 120]
    metrics = ['wmape', 'mae', 'rmse', 'r2', 'mda']
    interpolation_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'pchip', 'akima']

    # ==============================================================================
    # FASE 1: OMNI UNIVERSE
    # ==============================================================================
    print("\n📡 [FASE 1] Carregando OMNI Histórica...")
    df_omni = hg.load_omni_data(start_year=1981)
    
    if df_omni is not None and not df_omni.empty:
        features_omni = ['flow_speed', 'proton_density', 'T', 'F', 'BZ_GSM', 'BY_GSM', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'Vx', 'Vy', 'Vz']
        
        for metric in tqdm(metrics, desc="Gerando Comparativos OMNI"):
            folder_path = os.path.join("omni", metric.upper())
            ensure_directory(folder_path)
            file_path = os.path.join(folder_path, f"Omni_{metric.upper()}.png")
            
            if os.path.exists(file_path):
                continue
                
            # Store results across methods
            all_results_omni = {feature: {} for feature in features_omni}
            
            for method in interpolation_methods:
                for feature in features_omni:
                    if feature in df_omni.columns and df_omni[feature].notna().any():
                        _, results = hg.evaluate_gaps(
                            df=df_omni, feature_name=feature, metric_name=metric,
                            gap_sizes=gap_sizes, interpolation_method=method
                        )
                        if results:
                            all_results_omni[feature][method] = results
            
            plot_methods_comparison(all_results_omni, metric, "OMNI Database", file_path, gap_sizes)
    
    # ==============================================================================
    # PHASE 2: EMBRACE NETWORK
    # ==============================================================================
    print("\n🌍 [PHASE 2] Loading EMBRACE Network...")
    stations = hg.fetch_inpe_stations()
    features_embrace = ['H', 'D', 'Z', 'F']
    
    for st in stations:
        print(f"\n================ Comparing Methods: Station {st} ================")
        df_st = hg.download_embrace_data(station=st, start_year=2008)
        
        if df_st is not None and not df_st.empty:
            for metric in metrics:
                folder_path = os.path.join("embrace", st, metric.upper())
                ensure_directory(folder_path)
                file_path = os.path.join(folder_path, f"embrace_{st}_{metric.upper()}.png")
                
                if os.path.exists(file_path):
                    continue
                    
                all_results_embrace = {feature: {} for feature in features_embrace}
                
                for method in interpolation_methods:
                    for feature in features_embrace:
                        if feature in df_st.columns and df_st[feature].notna().any():
                            _, results = hg.evaluate_gaps(
                                df=df_st, feature_name=feature, metric_name=metric,
                                gap_sizes=gap_sizes, interpolation_method=method
                            )
                            if results:
                                all_results_embrace[feature][method] = results
                
                plot_methods_comparison(all_results_embrace, metric, f"EMBRACE ({st})", file_path, gap_sizes)
                
            del df_st
            gc.collect()

    print("\n🎉 GERAÇÃO DE GRÁFICOS COMPARATIVOS CONCLUÍDA!")

if __name__ == "__main__":
    main()