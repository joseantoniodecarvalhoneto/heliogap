"""
Unit tests for data sanitization, parsing, and boundary trimming modules.
"""

import numpy as np
import pandas as pd
import pytest
from heliogap import (
    clean_omni_fill_values,
    clean_embrace_data,
    clean_intermagnet_data,
    parse_iaga2002
)


def test_clean_omni_fill_values():
    """Test filtering of OMNI unphysical outliers and edge trimming."""
    df = pd.DataFrame({
        'proton_density': [5.0, 250.0, 10.0, np.nan, 8.0],
        'flow_speed': [400.0, 6000.0, 450.0, 420.0, 410.0],
        'F': [5.0, 10.0, 300.0, 8.0, 7.0]
    })
    cleaned = clean_omni_fill_values(df, trim_edges=False)
    assert np.isnan(cleaned.loc[1, 'proton_density'])  # 250 > 200 threshold
    assert np.isnan(cleaned.loc[1, 'flow_speed'])      # 6000 > 5000 threshold
    assert np.isnan(cleaned.loc[2, 'F'])               # 300 > 200 threshold


def test_clean_omni_fill_values_trim_edges():
    """Test that boundary trimming removes 1 index before and after NaN."""
    df = pd.DataFrame({
        'proton_density': [5.0, 6.0, np.nan, 8.0, 9.0]
    })
    cleaned = clean_omni_fill_values(df, trim_edges=True)
    # Indices 1, 2, 3 should all be NaN
    assert np.isnan(cleaned.loc[1, 'proton_density'])
    assert np.isnan(cleaned.loc[2, 'proton_density'])
    assert np.isnan(cleaned.loc[3, 'proton_density'])
    # Index 0 and 4 remain valid
    assert cleaned.loc[0, 'proton_density'] == 5.0
    assert cleaned.loc[4, 'proton_density'] == 9.0


def test_clean_embrace_data():
    """Test EMBRACE magnetic field cleaning and outlier removal."""
    df = pd.DataFrame({
        'H': [24000.0, 99999.0, 24050.0, -60000.0, 24100.0]
    })
    cleaned = clean_embrace_data(df, trim_edges=False)
    assert cleaned.loc[0, 'H'] == 24000.0
    assert np.isnan(cleaned.loc[1, 'H'])
    assert cleaned.loc[2, 'H'] == 24050.0
    assert np.isnan(cleaned.loc[3, 'H'])


def test_clean_intermagnet_data():
    """Test INTERMAGNET 88888.00 fill value replacement."""
    df = pd.DataFrame({
        'X': [20000.0, 88888.0, 20010.0],
        'Y': [-5000.0, -5010.0, 99999.0]
    })
    cleaned = clean_intermagnet_data(df, trim_edges=False)
    assert np.isnan(cleaned.loc[1, 'X'])
    assert np.isnan(cleaned.loc[2, 'Y'])


def test_parse_iaga2002():
    """Test IAGA-2002 ASCII parser."""
    sample_iaga = """# Sample IAGA-2002 file
# Observatory: Vassouras
# Co-latitude: 112.4
DATE       TIME         DOY     VSSX      VSSY      VSSZ      VSSF   |
2024-01-01 00:00:00.000 001   20000.00  -5000.00  -10000.00  23000.00
2024-01-01 00:01:00.000 001   20005.00  -5002.00  -10002.00  23005.00
2024-01-01 00:02:00.000 001   88888.00  -5001.00  -10001.00  23003.00
"""
    parsed = parse_iaga2002(sample_iaga)
    assert parsed is not None
    assert "Time" in parsed.columns
    assert "Tempo" in parsed.columns
    assert "X" in parsed.columns
    assert len(parsed) == 3
    assert np.isnan(parsed.loc[2, "X"])  # 88888.00 converted to NaN
