import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_metric_matrix(gap_sizes, results_dict, metric_name="WMAPE", save_path="metrics_plot.png"):
    """
    Generate a standardized grid of metric plots for evaluated features.
    
    Parameters:
        gap_sizes (list): Simulated data gap dimensions.
        results_dict (dict): Keys as feature names and values as error matrices.
        metric_name (str): Evaluated metric identifier.
        save_path (str): Output image destination path.
    """
    feature_config = {
        'flow_speed': {'name': 'Velocity', 'unit': 'km/s', 'color': 'darkmagenta'},
        'proton_density': {'name': 'Proton Density', 'unit': 'N/cm³', 'color': 'darkorange'},
        'T': {'name': 'Temperature', 'unit': 'K', 'color': 'crimson'},
        'F': {'name': 'Magnetic Force (F)', 'unit': 'nT', 'color': 'forestgreen'},
        'BZ_GSM': {'name': 'Magnetic Field (Bz GSM)', 'unit': 'nT', 'color': 'royalblue'},
        'BY_GSM': {'name': 'Magnetic Field (By GSM)', 'unit': 'nT', 'color': 'cornflowerblue'},
        'BX_GSE': {'name': 'Magnetic Field (Bx GSE)', 'unit': 'nT', 'color': 'teal'},
        'BY_GSE': {'name': 'Magnetic Field (By GSE)', 'unit': 'nT', 'color': 'mediumseagreen'},
        'BZ_GSE': {'name': 'Magnetic Field (Bz GSE)', 'unit': 'nT', 'color': 'dodgerblue'},
        'Vx': {'name': 'Velocity X', 'unit': 'km/s', 'color': 'purple'},
        'Vy': {'name': 'Velocity Y', 'unit': 'km/s', 'color': 'mediumorchid'},
        'Vz': {'name': 'Velocity Z', 'unit': 'km/s', 'color': 'darkviolet'},
        'H': {'name': 'Horizontal Field (H)', 'unit': 'nT', 'color': 'indigo'},
        'D': {'name': 'Declination (D)', 'unit': 'Deg', 'color': 'saddlebrown'},
        'Z': {'name': 'Vertical Field (Z)', 'unit': 'nT', 'color': 'darkslategray'}
    }
    
    active_features = {k: v for k, v in feature_config.items() if k in results_dict and results_dict[k]}
    num_plots = len(active_features)
    
    if num_plots == 0:
        print("No valid data available to plot.")
        return
        
    cols = 1 if num_plots == 1 else (2 if num_plots <= 6 else 3)
    rows = (num_plots + cols - 1) // cols
    
    fig_width = 12 if cols == 1 else cols * 7
    fig_height = max(6, rows * 4.5)
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = [axes] if num_plots == 1 else axes.flatten()
        
    for i, (col_name, info) in enumerate(active_features.items()):
        ax = axes[i]
        errors = results_dict[col_name]
        
        ax.plot(gap_sizes, errors, color=info['color'], marker='o', linewidth=2.5)
        
        if metric_name.upper() == 'RMSE':
            ax.fill_between(gap_sizes, errors, color=info['color'], alpha=0.1)
            
        if metric_name.upper() == 'R2':
            ax.axhline(0, color='red', linewidth=2, linestyle='--', label='Worse than Mean (0.0)')
            ax.legend(fontsize=9, loc='best')
        elif metric_name.upper() == 'MDA':
            ax.axhline(50, color='red', linewidth=2, linestyle='--', label='Coin Flip (50%)')
            ax.legend(fontsize=9, loc='lower left')
        
        y_label = "Global Error (%)" if metric_name.upper() in ['WMAPE', 'MDA'] else f"Error ({info['unit']})"
        if metric_name.upper() == 'R2':
            y_label = "R² Score"
            
        ax.set_title(f"{info['name']} ({col_name})", fontweight='bold')
        ax.set_ylabel(y_label)
        ax.set_xscale('log')
        ax.set_xticks(gap_sizes)
        ax.set_xticklabels(gap_sizes)
        ax.grid(True, ls="--", alpha=0.4)
        ax.margins(y=0.15)
        
        for j, val in enumerate(errors):
            if metric_name.upper() in ['WMAPE', 'MDA']:
                text_val = f"{val:.1f}%"
            elif metric_name.upper() == 'R2':
                text_val = f"{val:.2f}"
            else:
                if col_name == 'D':
                    text_val = f"{val:.4f}"
                else:
                    text_val = f"{val:.2f}"
                
            ax.annotate(text_val, (gap_sizes[j], val), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', va='bottom', color=info['color'], 
                        fontweight='bold', fontsize=9)

    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Interpolation Analysis: {metric_name.upper()}", fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')