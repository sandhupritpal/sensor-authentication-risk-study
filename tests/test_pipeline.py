"""
Unit tests for data simulation, defense transformations, and attack model pipeline.
"""

import numpy as np
import pytest
from src.dataset import BiometricIMUSimulator
from src.defense import SensorDefenseTransformer
from src.attack_model import ReIdentificationAttackModel


def test_imu_simulator_shape_and_columns():
    sim = BiometricIMUSimulator(sampling_rate_hz=100.0, random_seed=42)
    df = sim.generate_session(user_id=1, session_id=1, duration_sec=5.0, activity="walking")
    
    assert len(df) == 500
    expected_cols = ["timestamp", "user_id", "session_id", "activity", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    for col in expected_cols:
        assert col in df.columns
    assert df["user_id"].iloc[0] == 1


def test_defense_transformer():
    sim = BiometricIMUSimulator(sampling_rate_hz=100.0, random_seed=42)
    df = sim.generate_session(user_id=1, session_id=1, duration_sec=5.0)
    
    # 1. Test DP Noise
    transformer_dp = SensorDefenseTransformer(noise_mechanism="laplace", epsilon=0.5)
    df_dp = transformer_dp.apply(df)
    assert len(df_dp) == len(df)
    assert not np.allclose(df["accel_x"].to_numpy(), df_dp["accel_x"].to_numpy())
    
    # 2. Test Frequency Throttling (100Hz -> 20Hz)
    transformer_throttle = SensorDefenseTransformer(target_fs=20.0, original_fs=100.0)
    df_throttled = transformer_throttle.apply(df)
    assert len(df_throttled) == 100  # 5 seconds at 20Hz = 100 samples


def test_attack_model_train_and_predict():
    # Synthetic small dataset
    n_samples = 40
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.array([1] * 20 + [2] * 20)
    
    model = ReIdentificationAttackModel(model_type="rf", random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert set(preds).issubset({1, 2})
    
    probas = model.predict_proba(X[:5])
    assert probas.shape == (5, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
