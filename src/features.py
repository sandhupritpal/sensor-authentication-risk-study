"""
Feature engineering pipeline for 6-DoF IMU telemetry.
Computes comprehensive time-domain, frequency-domain, kinematic jerk, and cross-axis
correlation features over sliding temporal windows to isolate biometric side-channel signatures.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


SENSOR_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def compute_spectral_entropy(psd: np.ndarray, eps: float = 1e-12) -> float:
    """Computes normalized Shannon entropy of normalized power spectral density."""
    psd_norm = psd / (np.sum(psd) + eps)
    psd_norm = psd_norm[psd_norm > eps]
    if len(psd_norm) == 0:
        return 0.0
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def extract_window_features(
    window_df: pd.DataFrame,
    fs: float = 100.0,
) -> Dict[str, float]:
    """
    Extracts statistical, kinematic, and spectral features from a single temporal window.
    """
    feats: Dict[str, float] = {}
    
    # 1. Magnitudes for acceleration and gyroscopic velocity
    ax = window_df["accel_x"].to_numpy()
    ay = window_df["accel_y"].to_numpy()
    az = window_df["accel_z"].to_numpy()
    gx = window_df["gyro_x"].to_numpy()
    gy = window_df["gyro_y"].to_numpy()
    gz = window_df["gyro_z"].to_numpy()
    
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    channels = {
        "accel_x": ax, "accel_y": ay, "accel_z": az, "accel_mag": accel_mag,
        "gyro_x": gx, "gyro_y": gy, "gyro_z": gz, "gyro_mag": gyro_mag,
    }
    
    # 2. Time-Domain Statistical & Kinematic Features
    for ch_name, sig in channels.items():
        n = len(sig)
        if n == 0:
            continue
            
        mean_val = float(np.mean(sig))
        std_val = float(np.std(sig))
        feats[f"{ch_name}_mean"] = mean_val
        feats[f"{ch_name}_std"] = std_val
        feats[f"{ch_name}_rms"] = float(np.sqrt(np.mean(sig**2)))
        feats[f"{ch_name}_p2p"] = float(np.ptp(sig))
        
        # Higher order moments
        feats[f"{ch_name}_skew"] = float(stats.skew(sig)) if std_val > 1e-8 else 0.0
        feats[f"{ch_name}_kurt"] = float(stats.kurtosis(sig)) if std_val > 1e-8 else 0.0
        
        # Zero-crossing rate (centered)
        centered = sig - mean_val
        zero_crossings = np.where(np.diff(np.signbit(centered)))[0]
        feats[f"{ch_name}_zcr"] = float(len(zero_crossings) / n)
        
        # Kinematic Jerk / Angular Acceleration derivative
        jerk = np.diff(sig) * fs
        feats[f"{ch_name}_jerk_mean"] = float(np.mean(np.abs(jerk))) if len(jerk) > 0 else 0.0
        
        # 3. Frequency-Domain (FFT) Features
        # Half-spectrum up to Nyquist frequency
        fft_vals = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = np.abs(fft_vals)**2
        
        total_energy = float(np.sum(psd))
        feats[f"{ch_name}_spectral_energy"] = total_energy
        feats[f"{ch_name}_spectral_entropy"] = compute_spectral_entropy(psd)
        
        if len(freqs) > 1 and total_energy > 1e-8:
            dom_idx = int(np.argmax(psd[1:])) + 1
            feats[f"{ch_name}_dom_freq"] = float(freqs[dom_idx])
            feats[f"{ch_name}_dom_power"] = float(psd[dom_idx])
            feats[f"{ch_name}_spectral_centroid"] = float(np.sum(freqs * psd) / total_energy)
        else:
            feats[f"{ch_name}_dom_freq"] = 0.0
            feats[f"{ch_name}_dom_power"] = 0.0
            feats[f"{ch_name}_spectral_centroid"] = 0.0
            
        # Biometric Tremor Band (8 - 12 Hz) Energy Ratio
        tremor_mask = (freqs >= 8.0) & (freqs <= 12.0)
        tremor_energy = np.sum(psd[tremor_mask])
        feats[f"{ch_name}_tremor_ratio"] = float(tremor_energy / (total_energy + 1e-8))
        
        # Locomotion Band (1 - 3 Hz) Energy Ratio
        gait_mask = (freqs >= 1.0) & (freqs <= 3.0)
        gait_energy = np.sum(psd[gait_mask])
        feats[f"{ch_name}_gait_ratio"] = float(gait_energy / (total_energy + 1e-8))

    # 4. Signal Magnitude Area (SMA)
    feats["accel_sma"] = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    feats["gyro_sma"] = float(np.mean(np.abs(gx) + np.abs(gy) + np.abs(gz)))
    
    # 5. Cross-Axis Pearson Correlations (Physical posture coupling)
    def safe_corr(s1: np.ndarray, s2: np.ndarray) -> float:
        if np.std(s1) < 1e-8 or np.std(s2) < 1e-8:
            return 0.0
        r = np.corrcoef(s1, s2)[0, 1]
        return float(r) if not np.isnan(r) else 0.0

    feats["corr_accel_xy"] = safe_corr(ax, ay)
    feats["corr_accel_xz"] = safe_corr(ax, az)
    feats["corr_accel_yz"] = safe_corr(ay, az)
    feats["corr_gyro_xy"] = safe_corr(gx, gy)
    feats["corr_gyro_xz"] = safe_corr(gx, gz)
    feats["corr_gyro_yz"] = safe_corr(gy, gz)
    
    return feats


def extract_features_from_dataset(
    df: pd.DataFrame,
    window_sec: float = 2.0,
    overlap_ratio: float = 0.5,
    fs: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Segments multi-user IMU sessions into sliding temporal windows and extracts features.
    
    Returns:
        X: Feature matrix of shape (n_windows, n_features)
        y: Array of target user labels
        metadata_df: DataFrame containing session_id, activity, window start time
        feature_names: List of column names corresponding to X
    """
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1.0 - overlap_ratio))
    
    feature_rows = []
    metadata_rows = []
    
    grouped = df.groupby(["user_id", "session_id"])
    
    for (user_id, session_id), session_data in grouped:
        activity = session_data["activity"].iloc[0] if "activity" in session_data else "unknown"
        n_samples = len(session_data)
        
        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            w_df = session_data.iloc[start_idx : start_idx + window_samples]
            feats = extract_window_features(w_df, fs=fs)
            
            feature_rows.append(feats)
            metadata_rows.append({
                "user_id": user_id,
                "session_id": session_id,
                "activity": activity,
                "start_idx": start_idx,
                "start_time_sec": start_idx / fs,
            })
            
    feature_df = pd.DataFrame(feature_rows)
    # Fill any subtle NaNs with median/0
    feature_df = feature_df.fillna(0.0)
    
    feature_names = list(feature_df.columns)
    X = feature_df.to_numpy()
    metadata_df = pd.DataFrame(metadata_rows)
    y = metadata_df["user_id"].to_numpy()
    
    return X, y, metadata_df, feature_names
