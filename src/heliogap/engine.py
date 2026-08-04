import time
import numpy as np
import pandas as pd
import gc

def run_exhaustive_analysis(df, feature_name, metric_name, gap_sizes=[1, 2, 5, 10, 15, 30, 60, 120], interpolation_method='linear', order=None):
    """
    Core vector engine designed to run sequentially (no multiprocessing) 
    to guarantee stability and prevent memory crashes on notebooks.
    """
    print(f"\n{'='*60}\nAnalyzing: {feature_name} | Metric: {metric_name.upper()} | Method: {interpolation_method.capitalize()}\n{'='*60}")
    
    is_valid = df[feature_name].notna()
    block_ids = (~is_valid).cumsum()[is_valid]
    
    segments = [group.values for _, group in df[feature_name][is_valid].groupby(block_ids)]
    print(f"Extracted {len(segments):,} pure continuous historical blocks.")
    
    results = []
    global_mean = df[feature_name].mean() if metric_name.lower() == 'r2' else 0.0
    
    kwargs = {}
    if order is not None:
        kwargs['order'] = order

    for k in gap_sizes:
        start_time = time.time()
        
        accumulated_error = 0.0
        accumulated_signal = 0.0
        total_points_tested = 0
        
        if interpolation_method in ['linear', 'nearest', 'zero']:
            for seg in segments:
                n_points = len(seg)
                if n_points >= k + 2:
                    y_left = seg[:-k-1]
                    y_right = seg[k+1:]
                    
                    interpolation_dir = np.sign(y_right - y_left) if metric_name.lower() == 'mda' else None
                    
                    for step in range(1, k + 1):
                        alpha = step / (k + 1)
                        
                        if interpolation_method == 'linear':
                            y_pred = y_left + (y_right - y_left) * alpha
                        elif interpolation_method == 'nearest':
                            y_pred = np.where(alpha < 0.5, y_left, y_right)
                        elif interpolation_method == 'zero':
                            y_pred = y_left
                            
                        y_true = seg[step : step + len(y_left)]
                        
                        if metric_name.lower() == 'wmape':
                            accumulated_error += np.sum(np.abs(y_true - y_pred))
                            accumulated_signal += np.sum(np.abs(y_true))
                        elif metric_name.lower() == 'mae':
                            accumulated_error += np.sum(np.abs(y_true - y_pred))
                        elif metric_name.lower() == 'rmse':
                            accumulated_error += np.sum((y_true - y_pred)**2)
                        elif metric_name.lower() == 'r2':
                            accumulated_error += np.sum((y_true - y_pred)**2)
                            accumulated_signal += np.sum((y_true - global_mean)**2)
                        elif metric_name.lower() == 'mda':
                            y_true_prev = seg[step - 1 : step - 1 + len(y_left)]
                            real_direction = np.sign(y_true - y_true_prev)
                            accumulated_error += np.sum(real_direction == interpolation_dir)
                            
                        total_points_tested += len(y_true)
        
        else:
            for seg in segments:
                n_points = len(seg)
                if n_points >= k + 2:
                    num_windows = n_points - k - 1
                    start_anchors = np.arange(num_windows)
                    window_indices = np.arange(k + 2) + start_anchors[:, None]
                    windows = seg[window_indices]
                    
                    y_true = windows[:, 1:k+1].copy()
                    windows[:, 1:k+1] = np.nan

                    df_windows = pd.DataFrame(windows.T)
                    df_windows = df_windows.interpolate(method=interpolation_method, axis=0, **kwargs)
                    y_pred = df_windows.values[1:k+1, :].T

                    if metric_name.lower() == 'wmape':
                         accumulated_error += np.sum(np.abs(y_true - y_pred))
                         accumulated_signal += np.sum(np.abs(y_true))
                    elif metric_name.lower() == 'mae':
                         accumulated_error += np.sum(np.abs(y_true - y_pred))
                    elif metric_name.lower() == 'rmse':
                         accumulated_error += np.sum((y_true - y_pred)**2)
                    elif metric_name.lower() == 'r2':
                         accumulated_error += np.sum((y_true - y_pred)**2)
                         accumulated_signal += np.sum((y_true - global_mean)**2)
                    elif metric_name.lower() == 'mda':
                         true_direction = np.sign(y_true[:, 1:] - y_true[:, :-1])
                         pred_direction = np.sign(y_pred[:, 1:] - y_pred[:, :-1])
                         true_boundary_dir = np.sign(y_true[:, 0] - windows[:, 0])
                         pred_boundary_dir = np.sign(y_pred[:, 0] - windows[:, 0])
                         
                         accumulated_error += np.sum(true_direction == pred_direction)
                         accumulated_error += np.sum(true_boundary_dir == pred_boundary_dir)

                    total_points_tested += y_true.size

        # Limpeza agressiva da RAM ao concluir cada gap
        gc.collect()

        final_result = 0.0
        if total_points_tested > 0:
            if metric_name.lower() == 'wmape':
                final_result = (accumulated_error / accumulated_signal) * 100 if accumulated_signal > 0 else 0.0
            elif metric_name.lower() == 'mae':
                final_result = accumulated_error / total_points_tested
            elif metric_name.lower() == 'rmse':
                final_result = np.sqrt(accumulated_error / total_points_tested)
            elif metric_name.lower() == 'r2':
                final_result = 1.0 - (accumulated_error / accumulated_signal) if accumulated_signal > 0 else 0.0
            elif metric_name.lower() == 'mda':
                final_result = (accumulated_error / total_points_tested) * 100
                
        results.append(final_result)
        elapsed_time = time.time() - start_time
        
        print(f"[Gap {k:3} min] Scenarios: {total_points_tested:11,} | Result: {final_result:8.2f} | Time: {elapsed_time:.2f}s")
        
    return gap_sizes, results