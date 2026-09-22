"""
Heliogap Mathematical Engine - Gap Simulation and Benchmarking Module.

Provides memory-efficient, high-performance gap interpolation evaluation
for space weather and geomagnetic time series (NASA OMNI, INPE EMBRACE, INTERMAGNET).
"""

from typing import Union, List, Tuple, Optional
import os
import time
import gc
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator
from joblib import Parallel, delayed

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def downcast_dataframe(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns in a DataFrame (float64 -> float32, int64 -> int32)
    to reduce memory footprint prior to vectorized numerical evaluation.

    Parameters:
        df (pd.DataFrame): Target DataFrame.
        inplace (bool): Whether to modify the DataFrame columns in-place.

    Returns:
        pd.DataFrame: Memory-optimized DataFrame.
    """
    if df is None or df.empty:
        return df

    target_df = df if inplace else df.copy()

    for col in target_df.columns:
        dtype = target_df[col].dtype
        if pd.api.types.is_float_dtype(dtype) and dtype != np.float32:
            target_df[col] = target_df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dtype) and dtype != np.int32:
            target_df[col] = target_df[col].astype(np.int32)

    return target_df


def extract_continuous_segments(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    feature_name: Optional[str] = None,
    min_length: int = 3
) -> List[np.ndarray]:
    """
    Extract continuous non-NaN slices using fast NumPy run-length transition detection.

    Avoids DataFrame groupby operations for efficient zero-copy array views.

    Parameters:
        data (pd.DataFrame | pd.Series | np.ndarray): Input series or dataset.
        feature_name (str, optional): Target column if a DataFrame is passed.
        min_length (int): Minimum continuous points required to evaluate gaps.

    Returns:
        List[np.ndarray]: Contiguous 1D float32 NumPy array segments.
    """
    if isinstance(data, pd.DataFrame):
        if feature_name is None:
            raise ValueError("feature_name must be specified when data is a DataFrame.")
        arr = data[feature_name].to_numpy(dtype=np.float32, copy=False)
    elif isinstance(data, pd.Series):
        arr = data.to_numpy(dtype=np.float32, copy=False)
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.float32, copy=False)
    else:
        arr = np.asarray(data, dtype=np.float32)

    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return []

    # Run-length transition detection via padded difference
    diff = np.diff(np.concatenate(([False], valid_mask, [False])).astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return [arr[s:e] for s, e in zip(starts, ends) if (e - s) >= min_length]


def get_optimal_chunk_size(window_size: int, n_workers: int = 1, dtype=np.float32) -> int:
    """
    Compute safe sliding-window chunk size based on available system memory
    divided among active worker processes.

    Parameters:
        window_size (int): Total window width (gap size + boundary contexts).
        n_workers (int): Number of parallel worker processes.
        dtype: Data type (default: np.float32).

    Returns:
        int: Maximum number of sliding windows per batch chunk.
    """
    if not HAS_PSUTIL:
        return 200_000

    ram_available = psutil.virtual_memory().available
    effective_workers = max(1, n_workers if n_workers > 0 else (os.cpu_count() or 1))

    # Allocate a fraction of available memory safely per worker
    target_ram_per_worker = (ram_available * 0.35) / effective_workers
    bytes_per_scenario = window_size * np.dtype(dtype).itemsize * 10

    chunk = int(target_ram_per_worker / max(1, bytes_per_scenario))
    return max(50_000, min(chunk, 3_000_000))


def _evaluate_single_gap(
    segments: List[np.ndarray],
    k: int,
    interpolation_method: str,
    metric_name: str,
    global_mean: float = 0.0,
    order: Optional[int] = None,
    chunk_size: int = 200_000
) -> Tuple[int, float, int, float]:
    """
    Worker unit that evaluates gap size k across continuous data segments.

    Parameters:
        segments (list[np.ndarray]): List of continuous 1D float32 segments.
        k (int): Gap size (sampling points).
        interpolation_method (str): Interpolation method identifier.
        metric_name (str): Metric name ('wmape', 'mae', 'rmse', 'r2', 'mda').
        global_mean (float): Baseline mean for R2 metric.
        order (int, optional): Polynomial order if applicable.
        chunk_size (int): Maximum sliding windows evaluated per chunk.

    Returns:
        tuple[int, float, int, float]: (k, metric_result, total_points_tested, elapsed_seconds)
    """
    start_time = time.time()
    accumulated_error = 0.0
    accumulated_signal = 0.0
    total_points_tested = 0
    metric_lower = metric_name.lower()
    method_lower = interpolation_method.lower()

    # --------------------------------------------------------------------------
    # 1. Vectorized NumPy evaluation for basic methods (linear, nearest, zero)
    # --------------------------------------------------------------------------
    if method_lower in ['linear', 'nearest', 'zero']:
        for seg in segments:
            n_points = len(seg)
            if n_points >= k + 2:
                y_left = seg[:-k-1]
                y_right = seg[k+1:]
                interpolation_dir = np.sign(y_right - y_left) if metric_lower == 'mda' else None

                for step in range(1, k + 1):
                    alpha = step / (k + 1)
                    if method_lower == 'linear':
                        y_pred = y_left + (y_right - y_left) * alpha
                    elif method_lower == 'nearest':
                        y_pred = np.where(alpha < 0.5, y_left, y_right)
                    elif method_lower == 'zero':
                        y_pred = y_left

                    y_true = seg[step : step + len(y_left)]

                    if metric_lower == 'wmape':
                        accumulated_error += float(np.sum(np.abs(y_true - y_pred)))
                        accumulated_signal += float(np.sum(np.abs(y_true)))
                    elif metric_lower == 'mae':
                        accumulated_error += float(np.sum(np.abs(y_true - y_pred)))
                    elif metric_lower == 'rmse':
                        accumulated_error += float(np.sum((y_true - y_pred) ** 2))
                    elif metric_lower == 'r2':
                        accumulated_error += float(np.sum((y_true - y_pred) ** 2))
                        accumulated_signal += float(np.sum((y_true - global_mean) ** 2))
                    elif metric_lower == 'mda':
                        y_true_prev = seg[step - 1 : step - 1 + len(y_left)]
                        real_direction = np.sign(y_true - y_true_prev)
                        accumulated_error += float(np.sum(real_direction == interpolation_dir))

                    total_points_tested += len(y_true)

    # --------------------------------------------------------------------------
    # 2. SciPy sliding-window interpolation (spline, cubic, pchip, akima, etc.)
    # --------------------------------------------------------------------------
    else:
        context_pts = 3
        window_size = k + 2 * context_pts

        for seg in segments:
            n_points = len(seg)
            if n_points >= window_size:
                num_windows = n_points - window_size + 1
                x_all = np.arange(window_size, dtype=np.float32)
                known_indices = np.concatenate([
                    np.arange(context_pts),
                    np.arange(context_pts + k, window_size)
                ])
                target_indices = np.arange(context_pts, context_pts + k)

                x_known = x_all[known_indices]
                x_target = x_all[target_indices]

                for start_idx in range(0, num_windows, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_windows)
                    start_anchors = np.arange(start_idx, end_idx)
                    window_indices = np.arange(window_size) + start_anchors[:, None]
                    chunk_windows = seg[window_indices]

                    y_known = chunk_windows[:, known_indices].T
                    y_true = chunk_windows[:, target_indices]

                    try:
                        if method_lower in ['cubic', 'cubicspline', 'spline']:
                            interpolator = CubicSpline(x_known, y_known, axis=0)
                            y_pred = interpolator(x_target).T
                        elif method_lower == 'pchip':
                            interpolator = PchipInterpolator(x_known, y_known, axis=0)
                            y_pred = interpolator(x_target).T
                        elif method_lower == 'akima':
                            interpolator = Akima1DInterpolator(x_known, y_known, axis=0)
                            y_pred = interpolator(x_target).T
                        elif method_lower in ['quadratic', 'slinear', 'polynomial']:
                            if method_lower == 'polynomial':
                                kind = min(order if order else 3, len(x_known) - 1)
                            elif method_lower == 'quadratic':
                                kind = 'quadratic'
                            else:
                                kind = 'slinear'
                            interpolator = interp1d(x_known, y_known, axis=0, kind=kind)
                            y_pred = interpolator(x_target).T
                        else:
                            interpolator = interp1d(x_known, y_known, axis=0, kind='linear')
                            y_pred = interpolator(x_target).T
                    except Exception:
                        continue

                    if metric_lower == 'wmape':
                        accumulated_error += float(np.sum(np.abs(y_true - y_pred)))
                        accumulated_signal += float(np.sum(np.abs(y_true)))
                    elif metric_lower == 'mae':
                        accumulated_error += float(np.sum(np.abs(y_true - y_pred)))
                    elif metric_lower == 'rmse':
                        accumulated_error += float(np.sum((y_true - y_pred) ** 2))
                    elif metric_lower == 'r2':
                        accumulated_error += float(np.sum((y_true - y_pred) ** 2))
                        accumulated_signal += float(np.sum((y_true - global_mean) ** 2))
                    elif metric_lower == 'mda':
                        y_true_prev = chunk_windows[:, target_indices - 1]
                        true_direction = np.sign(y_true - y_true_prev)
                        y_pred_with_anchor = np.column_stack([chunk_windows[:, context_pts - 1], y_pred])
                        pred_direction = np.sign(y_pred_with_anchor[:, 1:] - y_pred_with_anchor[:, :-1])
                        accumulated_error += float(np.sum(true_direction == pred_direction))

                    total_points_tested += y_true.size
                    del chunk_windows, y_known, y_true

    # --------------------------------------------------------------------------
    # 3. Final metric aggregation
    # --------------------------------------------------------------------------
    final_result = 0.0
    if total_points_tested > 0:
        if metric_lower == 'wmape':
            final_result = (accumulated_error / accumulated_signal) * 100.0 if accumulated_signal > 0 else 0.0
        elif metric_lower == 'mae':
            final_result = accumulated_error / total_points_tested
        elif metric_lower == 'rmse':
            final_result = np.sqrt(accumulated_error / total_points_tested)
        elif metric_lower == 'r2':
            final_result = 1.0 - (accumulated_error / accumulated_signal) if accumulated_signal > 0 else 0.0
        elif metric_lower == 'mda':
            final_result = (accumulated_error / total_points_tested) * 100.0

    elapsed_time = time.time() - start_time
    return (k, float(final_result), total_points_tested, elapsed_time)


def evaluate_gaps(
    df: Union[pd.DataFrame, pd.Series, np.ndarray],
    feature_name: Optional[str] = None,
    metric_name: str = 'wmape',
    gap_sizes: Optional[List[int]] = None,
    interpolation_method: str = 'linear',
    order: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Simulate and evaluate data gaps across time-series datasets.

    Supports single-core sequential execution (n_jobs=1) and multi-core
    parallel processing via Joblib (n_jobs=-1 or specific worker count).

    Parameters:
        df (pd.DataFrame | pd.Series | np.ndarray): Input dataset.
        feature_name (str, optional): Target column name if df is a DataFrame.
        metric_name (str): Evaluation metric ('wmape', 'mae', 'rmse', 'r2', 'mda').
        gap_sizes (list[int], optional): Gap dimensions in sampling steps (default: [1, 2, 5, 10, 15, 30, 60, 120]).
        interpolation_method (str): Interpolation method ('linear', 'nearest', 'zero', 'cubic', 'pchip', 'akima', etc.).
        order (int, optional): Polynomial order for interpolation if applicable.
        n_jobs (int): CPU workers (1 = sequential, -1 = all cores).
        verbose (bool): Whether to print progress log to stdout.

    Returns:
        tuple[list[int], list[float]]: (gap_sizes, metric_results)
    """
    if gap_sizes is None:
        gap_sizes = [1, 2, 5, 10, 15, 30, 60, 120]

    feature_label = feature_name if feature_name else 'Feature'
    segments = extract_continuous_segments(df, feature_name=feature_name)

    if verbose:
        print(f"\n{'=' * 60}\nEvaluating: {feature_label} | Metric: {metric_name.upper()} | Method: {interpolation_method.capitalize()}\n{'=' * 60}")
        print(f"Extracted {len(segments):,} continuous data segments.")

    if not segments or not gap_sizes:
        return gap_sizes, [0.0] * len(gap_sizes)

    # Compute global mean if R2 metric is requested
    global_mean = 0.0
    if metric_name.lower() == 'r2':
        if isinstance(df, pd.DataFrame) and feature_name:
            global_mean = float(df[feature_name].mean())
        elif isinstance(df, pd.Series):
            global_mean = float(df.mean())
        else:
            all_pts = np.concatenate(segments)
            global_mean = float(np.mean(all_pts))

    num_cpus = os.cpu_count() or 1
    effective_workers = num_cpus if n_jobs == -1 else max(1, min(n_jobs, num_cpus))
    max_k = max(gap_sizes) if gap_sizes else 120
    chunk_size = get_optimal_chunk_size(max_k + 6, n_workers=effective_workers, dtype=np.float32)

    # Execute gap evaluations (Sequential or Parallel)
    if n_jobs == 1 or len(gap_sizes) == 1:
        raw_results = [
            _evaluate_single_gap(
                segments, k, interpolation_method, metric_name, global_mean, order, chunk_size
            )
            for k in gap_sizes
        ]
    else:
        raw_results = Parallel(
            n_jobs=effective_workers,
            max_nbytes='1M'
        )(
            delayed(_evaluate_single_gap)(
                segments, k, interpolation_method, metric_name, global_mean, order, chunk_size
            )
            for k in gap_sizes
        )

    # Sort results deterministically by gap size
    raw_results_sorted = sorted(raw_results, key=lambda x: x[0])
    results = [r[1] for r in raw_results_sorted]

    if verbose:
        for k, final_res, total_pts, el_time in raw_results_sorted:
            print(f"[Gap {k:3} pts] Evaluated: {total_pts:11,} | {metric_name.upper()}: {final_res:8.2f} | Time: {el_time:.2f}s")

    del segments, raw_results, raw_results_sorted
    gc.collect()

    return gap_sizes, results


# Concise aliases and backwards compatibility
run_gap_analysis = evaluate_gaps
run_exhaustive_analysis = evaluate_gaps