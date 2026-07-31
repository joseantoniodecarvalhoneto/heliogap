import matplotlib.pyplot as plt

def plot_metric_matrix(gap_sizes, results_dict, metric_name="WMAPE"):
    """
    Gera uma matriz de gráficos 3x2 padronizada para as 5 variáveis vitais.
    
    Args:
        gap_sizes (list): O eixo X (tamanhos dos apagões em minutos).
        results_dict (dict): Dicionário com as chaves das variáveis e listas de erros.
        metric_name (str): Nome da métrica ('WMAPE', 'MAE', 'RMSE', 'R2', 'MDA').
    """
    # Dicionário de configuração de cores, nomes e unidades
    variables_config = {
        'flow_speed': {'name': 'Velocity', 'unit': 'km/s', 'color': 'darkmagenta'},
        'proton_density': {'name': 'Proton Density', 'unit': 'N/cm³', 'color': 'darkorange'},
        'T': {'name': 'Temperature', 'unit': 'K', 'color': 'crimson'},
        'F': {'name': 'Magnetic Force (F)', 'unit': 'nT', 'color': 'forestgreen'},
        'BZ_GSM': {'name': 'Magnetic Field (Bz)', 'unit': 'nT', 'color': 'royalblue'}
    }
    
    print(f"\nGenerating 3x2 Matrix Plot for {metric_name}...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Itera sobre o dicionário local e preenche os gráficos
    for i, (col_name, info) in enumerate(variables_config.items()):
        ax = axes[i]
        
        # Se a variável não estiver nos resultados que passámos, salta o gráfico
        if col_name not in results_dict or not results_dict[col_name]:
            continue
            
        errors = results_dict[col_name]
        
        # Desenha a curva
        ax.plot(gap_sizes, errors, color=info['color'], marker='o', linewidth=2.5)
        
        # Se for RMSE, coloca uma área sombreada para dar mais ênfase
        if metric_name.upper() == 'RMSE':
             ax.fill_between(gap_sizes, errors, color=info['color'], alpha=0.1)
             
        # Se for R2 ou MDA, adiciona a linha vermelha de alerta (Threshold)
        if metric_name.upper() == 'R2':
            ax.axhline(0, color='red', linewidth=2, linestyle='--', label='Worse than Mean (0.0)')
            ax.legend(fontsize=9, loc='best')
        elif metric_name.upper() == 'MDA':
            ax.axhline(50, color='red', linewidth=2, linestyle='--', label='Coin Flip (50%)')
            ax.legend(fontsize=9, loc='lower left')
        
        # Títulos e Eixos
        y_label = f"Global Error (%)" if metric_name.upper() in ['WMAPE', 'MDA'] else f"Error ({info['unit']})"
        if metric_name.upper() == 'R2':
            y_label = "R² Score"
            
        ax.set_title(f"{metric_name.upper()}: {info['name']}", fontweight='bold')
        ax.set_ylabel(y_label)
        ax.set_xscale('log')
        ax.set_xticks(gap_sizes)
        ax.set_xticklabels(gap_sizes)
        ax.grid(True, ls="--", alpha=0.4)
        
        # A inteligência de margem para os textos não cortarem
        ax.margins(y=0.15)
        
        # Escreve o número exato em TODOS os pontos, como nos pediu
        for j, val in enumerate(errors):
            # Formatações condicionais consoante o tipo de métrica
            if metric_name.upper() in ['WMAPE', 'MDA']:
                text_val = f"{val:.1f}%"
            elif metric_name.upper() == 'R2':
                text_val = f"{val:.2f}"
            else:
                text_val = f"{val:.2f}"
                
            ax.annotate(text_val, (gap_sizes[j], val), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', va='bottom', color=info['color'], 
                        fontweight='bold', fontsize=10)

    # Esconde o 6º gráfico que sobra na grelha 3x2
    axes[5].set_visible(False)
    
    # Título principal geral (Header)
    main_title = f"Space Weather Metric Analysis: {metric_name.upper()}"
    plt.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()