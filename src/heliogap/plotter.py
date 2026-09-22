"""
Heliogap Plotter Module.

Generates publication-quality metric matrices and multi-feature comparative
visualizations for space physics gap interpolation studies.
"""

from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FEATURE_CONFIG = {
    'flow_speed': {'name': 'Bulk Velocity', 'unit': 'km/s', 'color': 'darkmagenta'},
    'proton_density': {'name': 'Proton Density', 'unit': 'N/cm³', 'color': 'darkorange'},
    'T': {'name': 'Temperature', 'unit': 'K', 'color': 'crimson'},
    'F': {'name': 'Total Magnetic Field (F)', 'unit': 'nT', 'color': 'forestgreen'},
    'BZ_GSM': {'name': 'Magnetic Field (Bz GSM)', 'unit': 'nT', 'color': 'royalblue'},
    'BY_GSM': {'name': 'Magnetic Field (By GSM)', 'unit': 'nT', 'color': 'cornflowerblue'},
    'BX_GSE': {'name': 'Magnetic Field (Bx GSE)', 'unit': 'nT', 'color': 'teal'},
    'BY_GSE': {'name': 'Magnetic Field (By GSE)', 'unit': 'nT', 'color': 'mediumseagreen'},
    'BZ_GSE': {'name': 'Magnetic Field (Bz GSE)', 'unit': 'nT', 'color': 'dodgerblue'},
    'Vx': {'name': 'Velocity Vx', 'unit': 'km/s', 'color': 'purple'},
    'Vy': {'name': 'Velocity Vy', 'unit': 'km/s', 'color': 'mediumorchid'},
    'Vz': {'name': 'Velocity Vz', 'unit': 'km/s', 'color': 'darkviolet'},
    'H': {'name': 'Horizontal Field (H)', 'unit': 'nT', 'color': 'indigo'},
    'D': {'name': 'Declination (D)', 'unit': 'deg', 'color': 'saddlebrown'},
    'Z': {'name': 'Vertical Field (Z)', 'unit': 'nT', 'color': 'darkslategray'},
    'X': {'name': 'North Component (X)', 'unit': 'nT', 'color': 'navy'},
    'Y': {'name': 'East Component (Y)', 'unit': 'nT', 'color': 'darkcyan'},
}


def plot_metric_matrix(
    gap_sizes: List[int],
    results_dict: Dict[str, List[float]],
    metric_name: str = "WMAPE",
    save_path: str = "metrics_plot.png",
    dpi: int = 300
) -> Optional[plt.Figure]:
    """
    Generate a standardized grid of metric plots for evaluated features.

    Parameters:
        gap_sizes (list[int]): Simulated data gap dimensions.
        results_dict (dict[str, list[float]]): Feature names mapped to error lists.
        metric_name (str): Evaluated metric identifier ('wmape', 'mae', 'rmse', 'r2', 'mda').
        save_path (str): Output destination image path.
        dpi (int): Image resolution dots per inch (default 300).

    Returns:
        plt.Figure | None: Matplotlib figure object or None if no valid features.
    """
    active_features = {
        k: FEATURE_CONFIG.get(k, {'name': k, 'unit': 'a.u.', 'color': 'steelblue'})
        for k in results_dict
        if results_dict[k]
    }
    num_plots = len(active_features)

    if num_plots == 0:
        print("No valid data available to plot.")
        return None

    cols = 1 if num_plots == 1 else (2 if num_plots <= 6 else 3)
    rows = (num_plots + cols - 1) // cols

    fig_width = 12 if cols == 1 else cols * 7
    fig_height = max(6, rows * 4.5)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes_list = [axes] if num_plots == 1 else (axes.flatten() if hasattr(axes, 'flatten') else [axes])

    metric_upper = metric_name.upper()

    for i, (col_name, info) in enumerate(active_features.items()):
        ax = axes_list[i]
        errors = results_dict[col_name]

        ax.plot(gap_sizes, errors, color=info['color'], marker='o', linewidth=2.5)

        if metric_upper == 'RMSE':
            ax.fill_between(gap_sizes, errors, color=info['color'], alpha=0.1)

        if metric_upper == 'R2':
            ax.axhline(0, color='red', linewidth=2, linestyle='--', label='Baseline Mean (0.0)')
            ax.legend(fontsize=9, loc='best')
        elif metric_upper == 'MDA':
            ax.axhline(50, color='red', linewidth=2, linestyle='--', label='Random Guess (50%)')
            ax.legend(fontsize=9, loc='lower left')

        if metric_upper in ['WMAPE', 'MDA']:
            y_label = "Global Score (%)" if metric_upper == 'MDA' else "Global Error (%)"
        elif metric_upper == 'R2':
            y_label = "R² Score"
        else:
            y_label = f"Error ({info['unit']})"

        ax.set_title(f"{info['name']} ({col_name})", fontweight='bold')
        ax.set_ylabel(y_label)
        ax.set_xlabel("Gap Duration (points / minutes)")
        ax.set_xscale('log')
        ax.set_xticks(gap_sizes)
        ax.set_xticklabels([str(g) for g in gap_sizes])
        ax.grid(True, ls="--", alpha=0.4)
        ax.margins(y=0.15)

        for j, val in enumerate(errors):
            if metric_upper in ['WMAPE', 'MDA']:
                text_val = f"{val:.1f}%"
            elif metric_upper == 'R2':
                text_val = f"{val:.2f}"
            else:
                text_val = f"{val:.4f}" if col_name == 'D' else f"{val:.2f}"

            ax.annotate(
                text_val, (gap_sizes[j], val),
                textcoords="offset points", xytext=(0, 10),
                ha='center', va='bottom', color=info['color'],
                fontweight='bold', fontsize=9
            )

    for j in range(num_plots, len(axes_list)):
        axes_list[j].set_visible(False)

    plt.suptitle(f"Heliogap Benchmark: {metric_upper}", fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig