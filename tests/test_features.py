"""
Unit tests for IMU feature extraction pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import compute_spectral_entropy, extract_window_features


def test_spectral_entropy_edge_cases():
    # Constant or empty PSD
    psd_zero = np.zeros(10)
    assert compute_spectral_entropy(psd_zero) == 0.0
    
    # Delta function (single frequency peak -> entropy = 0)
    psd_peak = np.zeros(16)
    psd_peak[4] = 100.0
    entropy = compute_spectral_entropy(psd_peak)
    assert 0.0 <= entropy <= 0.05
    
    # Uniform PSD (maximum entropy -> 1.0)
    psd_uniform = np.ones(16)
    entropy_uniform = compute_spectral_entropy(psd_uniform)
    assert pytest.approx(entropy_uniform, 0.01) == 1.0


def test_extract_window_features():
    fs = 100.0
    n = 200  # 2 seconds
    t = np.linspace(0, 2.0, n, endpoint=False)
    
    df = pd.DataFrame({
        "timestamp": t,
        "accel_x": 0.5 * np.sin(2 * np.pi * 2.0 * t),
        "accel_y": 0.3 * np.cos(2 * np.pi * 2.0 * t),
        "accel_z": 9.81 + 0.1 * np.sin(2 * np.pi * 10.0 * t),
        "gyro_x": 0.02 * np.sin(2 * np.pi * 1.5 * t),
        "gyro_y": 0.01 * np.cos(2 * np.pi * 1.5 * t),
        "gyro_z": 0.03 * np.sin(2 * np.pi * 2.5 * t),
    })
    
    feats = extract_window_features(df, fs=fs)
    
    # Assert expected keys exist
    assert "accel_z_mean" in feats
    assert pytest.approx(feats["accel_z_mean"], 0.1) == 9.81
    assert "accel_z_tremor_ratio" in feats
    assert feats["accel_z_tremor_ratio"] > 0.0
    assert "corr_accel_xy" in feats
    assert "accel_sma" in feats
    
    # Check no NaNs or Infs
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} returned NaN"
        assert not np.isinf(v), f"Feature {k} returned Inf"
