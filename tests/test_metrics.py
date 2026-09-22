"""
Unit tests for Heliogap loss functions and metrics.
"""

import numpy as np
import pandas as pd
from heliogap.metrics import (
    calculate_wmape,
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_mda,
    calculate_dbt,
    wmape,
    mae,
    rmse,
    r2,
    mda,
    dbt
)


def test_calculate_mae():
    """Test Mean Absolute Error (MAE) computation."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    # Absolute errors: |10-12|=2, |20-18|=2, |30-30|=0 -> Mean = 4 / 3 = 1.3333333
    result = calculate_mae(y_true, y_pred)
    assert np.isclose(result, 1.3333333)
    assert np.isclose(mae(y_true, y_pred), 1.3333333)


def test_calculate_rmse():
    """Test Root Mean Squared Error (RMSE) computation."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 4.0, 0.0])
    # Squared errors: 9, 16, 0. Mean = 25 / 3 = 8.3333... Sqrt = ~2.88675
    result = calculate_rmse(y_true, y_pred)
    assert np.isclose(result, 2.88675134)
    assert np.isclose(rmse(y_true, y_pred), 2.88675134)


def test_calculate_wmape():
    """Test Weighted Mean Absolute Percentage Error (WMAPE)."""
    y_true = np.array([10.0, -10.0, 20.0])
    y_pred = np.array([12.0, -8.0, 20.0])
    # Absolute errors: 2 + 2 + 0 = 4
    # Absolute signal: 10 + 10 + 20 = 40
    # WMAPE = (4 / 40) * 100 = 10.0%
    result = calculate_wmape(y_true, y_pred)
    assert np.isclose(result, 10.0)
    assert np.isclose(wmape(y_true, y_pred), 10.0)


def test_calculate_r2():
    """Test Coefficient of Determination (R-squared)."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 28.0])
    global_mean = 20.0
    # SS_res = 4 + 4 + 4 = 12
    # SS_tot = 100 + 0 + 100 = 200
    # R2 = 1 - (12 / 200) = 0.94
    result = calculate_r2(y_true, y_pred, global_mean)
    assert np.isclose(result, 0.94)
    assert np.isclose(r2(y_true, y_pred, global_mean), 0.94)


def test_calculate_mda():
    """Test Mean Directional Accuracy (MDA)."""
    y_true_current = np.array([15.0, 25.0, 10.0])
    y_true_previous = np.array([10.0, 30.0, 5.0])
    # True directions: [15-10=+1, 25-30=-1, 10-5=+1] -> [+1, -1, +1]
    pred_direction = 1
    # Matches: points 0 and 2 -> 2/3 = 66.666...%
    result = calculate_mda(y_true_current, y_true_previous, pred_direction)
    assert np.isclose(result, 66.6666667)
    assert np.isclose(mda(y_true_current, y_true_previous, pred_direction), 66.6666667)


def test_calculate_dbt():
    """Test rate of change of magnetic field dB/dt computation."""
    # Linear ramp: B increases by 60 nT every 60 seconds -> dB/dt = 1.0 nT/s
    b_field = np.array([100.0, 160.0, 220.0, 280.0])
    dt = 60.0
    derivative = calculate_dbt(b_field, dt=dt)
    assert np.allclose(derivative, 1.0)
    assert np.allclose(dbt(b_field, dt=dt), 1.0)

    # DataFrame input test
    df = pd.DataFrame({"B": b_field})
    df_dbt = calculate_dbt(df, dt=dt, column="B")
    assert isinstance(df_dbt, pd.Series)
    assert np.allclose(df_dbt.values, 1.0)