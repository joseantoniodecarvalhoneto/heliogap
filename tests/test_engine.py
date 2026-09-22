"""
Unit tests for the Heliogap mathematical evaluation engine.
"""

import numpy as np
import pandas as pd
import pytest
import heliogap as hg


def test_downcast_dataframe():
    """Test memory downcasting for float64 and int64 columns."""
    df = pd.DataFrame({
        'f64': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'i64': np.array([10, 20, 30], dtype=np.int64),
        'dt': pd.date_range('2026-01-01', periods=3, freq='min')
    })
    df_down = hg.downcast_dataframe(df)
    assert df_down['f64'].dtype == np.float32
    assert df_down['i64'].dtype == np.int32
    assert pd.api.types.is_datetime64_any_dtype(df_down['dt'])


def test_extract_continuous_segments():
    """Test fast extraction of continuous non-NaN slices."""
    arr = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan], dtype=np.float32)
    segments = hg.extract_continuous_segments(arr, min_length=3)
    assert len(segments) == 2
    assert np.array_equal(segments[0], [1.0, 2.0, 3.0])
    assert np.array_equal(segments[1], [4.0, 5.0, 6.0, 7.0])


def test_evaluate_gaps_sequential_vs_parallel():
    """Verify that sequential (n_jobs=1) and parallel (n_jobs=-1) yield identical results."""
    np.random.seed(42)
    x = np.linspace(0, 10, 1000, dtype=np.float32)
    y = (np.sin(x) + 0.05 * np.random.randn(len(x))).astype(np.float32)
    y[100:105] = np.nan
    df = pd.DataFrame({'sig': y})

    gaps, res_seq = hg.evaluate_gaps(
        df, 'sig', 'wmape', gap_sizes=[1, 2, 5],
        interpolation_method='linear', n_jobs=1, verbose=False
    )
    gaps, res_par = hg.evaluate_gaps(
        df, 'sig', 'wmape', gap_sizes=[1, 2, 5],
        interpolation_method='linear', n_jobs=-1, verbose=False
    )

    assert gaps == [1, 2, 5]
    assert np.allclose(res_seq, res_par, atol=1e-5)


def test_evaluate_gaps_methods_and_metrics():
    """Test evaluate_gaps across multiple interpolation methods and metrics."""
    np.random.seed(42)
    x = np.linspace(0, 10, 500, dtype=np.float32)
    y = np.cos(x).astype(np.float32)
    df = pd.DataFrame({'val': y})

    # Test cubic interpolation and RMSE metric
    gaps, res_rmse = hg.evaluate_gaps(
        df, 'val', 'rmse', gap_sizes=[1, 3],
        interpolation_method='cubic', n_jobs=1, verbose=False
    )
    assert len(res_rmse) == 2
    assert res_rmse[0] >= 0.0

    # Test nearest interpolation and MAE metric
    gaps, res_mae = hg.evaluate_gaps(
        df, 'val', 'mae', gap_sizes=[1, 3],
        interpolation_method='nearest', n_jobs=1, verbose=False
    )
    assert len(res_mae) == 2
    assert res_mae[0] >= 0.0

    # Backwards compatibility check
    gaps_compat, res_compat = hg.run_exhaustive_analysis(
        df, 'val', 'mae', gap_sizes=[1, 3],
        interpolation_method='nearest', n_jobs=1, verbose=False
    )
    assert np.allclose(res_mae, res_compat)
