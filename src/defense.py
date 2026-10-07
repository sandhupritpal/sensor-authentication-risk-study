"""
Sensor Privacy Mitigations & Countermeasures.
Simulates client-side and OS-level defenses:
1. Differential Privacy (Laplace & Gaussian perturbation)
2. Sampling Frequency Throttling (Downsampling from 100Hz -> 20Hz / 10Hz)
3. Precision Bit Truncation / Quantization
"""

from typing import Optional
import numpy as np
import pandas as pd


class SensorDefenseTransformer:
    """
    Applies privacy-preserving transforms to raw IMU sensor streams to evaluate
    the tradeoff between biometric de-anonymization resistance and utility.
    """

    def __init__(
        self,
        noise_mechanism: Optional[str] = None,  # "laplace", "gaussian", or None
        epsilon: float = 1.0,
        sensitivity: float = 0.5,
        target_fs: Optional[float] = None,      # e.g., 20.0 Hz
        original_fs: float = 100.0,
        quantization_step: Optional[float] = None, # e.g., 0.1 m/s^2
        random_seed: int = 42,
    ):
        self.noise_mechanism = noise_mechanism
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.target_fs = target_fs
        self.original_fs = original_fs
        self.quantization_step = quantization_step
        self.rng = np.random.default_rng(random_seed)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies configured defenses to raw IMU telemetry.
        """
        df_defended = df.copy()
        sensor_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        
        # 1. Frequency Throttling / Decimation
        if self.target_fs is not None and self.target_fs < self.original_fs:
            factor = int(round(self.original_fs / self.target_fs))
            if factor > 1:
                # Decimate rows
                df_defended = df_defended.iloc[::factor].copy().reset_index(drop=True)

        # 2. Differential Privacy Perturbation
        if self.noise_mechanism == "laplace":
            scale = self.sensitivity / max(self.epsilon, 1e-4)
            for col in sensor_cols:
                noise = self.rng.laplace(0, scale, size=len(df_defended))
                df_defended[col] += noise
                
        elif self.noise_mechanism == "gaussian":
            # Calibrated Gaussian noise (delta = 1e-5)
            delta = 1e-5
            sigma = (np.sqrt(2 * np.log(1.25 / delta)) * self.sensitivity) / max(self.epsilon, 1e-4)
            for col in sensor_cols:
                noise = self.rng.normal(0, sigma, size=len(df_defended))
                df_defended[col] += noise

        # 3. Precision Bit Truncation / Quantization
        if self.quantization_step is not None and self.quantization_step > 0:
            for col in sensor_cols:
                df_defended[col] = np.round(df_defended[col] / self.quantization_step) * self.quantization_step

        return df_defended


def evaluate_mitigation_tradeoff(
    raw_df: pd.DataFrame,
    noise_epsilons: list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    throttling_rates: list = [10.0, 20.0, 50.0, 100.0],
) -> pd.DataFrame:
    """
    Simulates sweep over privacy defense parameters to benchmark re-identification degradation.
    """
    from src.features import extract_features_from_dataset
    from src.attack_model import ReIdentificationAttackModel, split_by_session
    from sklearn.metrics import accuracy_score
    
    results = []
    
    # Baseline (no defense)
    X_base, y_base, meta_base, f_names = extract_features_from_dataset(raw_df, fs=100.0)
    X_tr, y_tr, X_te, y_te = split_by_session(X_base, y_base, meta_base)
    base_model = ReIdentificationAttackModel(model_type="rf").fit(X_tr, y_tr)
    base_acc = float(accuracy_score(y_te, base_model.predict(X_te)))
    results.append({
        "defense_type": "None (Baseline)",
        "parameter": "None",
        "reid_accuracy": base_acc,
        "privacy_gain_pct": 0.0,
    })
    
    # 1. Evaluate Laplace DP Noise sweep
    for eps in noise_epsilons:
        transformer = SensorDefenseTransformer(noise_mechanism="laplace", epsilon=eps, sensitivity=0.2)
        df_pert = transformer.apply(raw_df)
        X_p, y_p, meta_p, _ = extract_features_from_dataset(df_pert, fs=100.0)
        X_tr, y_tr, X_te, y_te = split_by_session(X_p, y_p, meta_p)
        model = ReIdentificationAttackModel(model_type="rf").fit(X_tr, y_tr)
        acc = float(accuracy_score(y_te, model.predict(X_te)))
        results.append({
            "defense_type": "Laplace DP",
            "parameter": f"epsilon={eps}",
            "reid_accuracy": acc,
            "privacy_gain_pct": max(0.0, (base_acc - acc) / base_acc * 100.0),
        })
        
    # 2. Evaluate Frequency Throttling sweep
    for fs_target in throttling_rates:
        if fs_target == 100.0:
            continue
        transformer = SensorDefenseTransformer(target_fs=fs_target, original_fs=100.0)
        df_throttled = transformer.apply(raw_df)
        X_t, y_t, meta_t, _ = extract_features_from_dataset(df_throttled, fs=fs_target)
        X_tr, y_tr, X_te, y_te = split_by_session(X_t, y_t, meta_t)
        model = ReIdentificationAttackModel(model_type="rf").fit(X_tr, y_tr)
        acc = float(accuracy_score(y_te, model.predict(X_te)))
        results.append({
            "defense_type": "Frequency Throttling",
            "parameter": f"fs={fs_target}Hz",
            "reid_accuracy": acc,
            "privacy_gain_pct": max(0.0, (base_acc - acc) / base_acc * 100.0),
        })

    return pd.DataFrame(results)
