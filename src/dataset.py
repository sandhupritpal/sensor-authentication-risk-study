"""
Dataset generation and benchmark ingestion module for IMU sensor telemetry.
Simulates high-fidelity 6-DoF (3-axis accelerometer, 3-axis gyroscope) time-series data
with individual biometric signatures, sensor noise, bias drift, and behavioral activities.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class BiometricIMUSimulator:
    """
    Simulates multi-user 6-DoF IMU sensor telemetry from consumer wearables (smartwatches)
    and XR headsets (Meta Quest, Apple Vision Pro) with individualized biometric micro-signatures.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        random_seed: int = 42,
    ):
        self.fs = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        self.rng = np.random.default_rng(random_seed)

    def _generate_user_profile(self, user_id: int) -> Dict[str, float]:
        """
        Derives invariant physiological and behavioral biometric parameters for a given user:
        - Physiological tremor frequency (8 - 12 Hz) & amplitude
        - Characteristic posture/tilt angles (gravity distribution across axes)
        - Harmonic gait / gesture cadence (1.4 - 2.2 Hz)
        - Axis coupling coefficients and baseline sensor bias
        """
        user_rng = np.random.default_rng(1000 + user_id)
        
        # Biometric tremor (involuntary micro-vibrations)
        tremor_freq = user_rng.uniform(8.0, 12.0)
        tremor_amp = user_rng.uniform(0.015, 0.045)  # m/s^2
        tremor_phase = user_rng.uniform(0, 2 * np.pi, size=3)
        
        # Posture tilt (distribution of 9.81 m/s^2 earth gravity)
        tilt_pitch = np.radians(user_rng.uniform(-15.0, 15.0))
        tilt_roll = np.radians(user_rng.uniform(-20.0, 20.0))
        
        gx = 9.81 * np.sin(tilt_roll)
        gy = 9.81 * np.sin(tilt_pitch) * np.cos(tilt_roll)
        gz = 9.81 * np.cos(tilt_pitch) * np.cos(tilt_roll)
        
        # Behavioral cadence (gait or interaction rhythm)
        cadence_freq = user_rng.uniform(1.5, 2.1)
        cadence_amp = user_rng.uniform(0.8, 1.6)
        
        # Rotational micro-dynamics (gyroscope biometric baseline)
        gyro_tremor_freq = tremor_freq * user_rng.uniform(0.95, 1.05)
        gyro_tremor_amp = user_rng.uniform(0.008, 0.025)  # rad/s
        
        return {
            "user_id": user_id,
            "tremor_freq": tremor_freq,
            "tremor_amp": tremor_amp,
            "tremor_phase": tremor_phase,
            "gravity_vec": np.array([gx, gy, gz]),
            "cadence_freq": cadence_freq,
            "cadence_amp": cadence_amp,
            "gyro_tremor_freq": gyro_tremor_freq,
            "gyro_tremor_amp": gyro_tremor_amp,
        }

    def generate_session(
        self,
        user_id: int,
        session_id: int,
        duration_sec: float = 30.0,
        activity: str = "walking",
        noise_std_accel: float = 0.02,
        noise_std_gyro: float = 0.005,
    ) -> pd.DataFrame:
        """
        Generates a continuous time-series session for a specific user and activity.
        """
        profile = self._generate_user_profile(user_id)
        num_samples = int(duration_sec * self.fs)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        
        # 1. Base gravity vector with slight posture drift
        drift_accel = np.cumsum(self.rng.normal(0, 0.0001, size=(num_samples, 3)), axis=0)
        accel = np.tile(profile["gravity_vec"], (num_samples, 1)) + drift_accel
        
        # 2. Activity-specific macroscopic motion
        if activity == "walking":
            f_cad = profile["cadence_freq"]
            a_cad = profile["cadence_amp"]
            # Primary vertical acceleration + 2nd harmonic
            accel[:, 2] += a_cad * np.sin(2 * np.pi * f_cad * t) + 0.3 * a_cad * np.sin(4 * np.pi * f_cad * t)
            # Anterior-posterior and lateral motion
            accel[:, 0] += 0.4 * a_cad * np.sin(2 * np.pi * f_cad * t + 0.4)
            accel[:, 1] += 0.25 * a_cad * np.sin(np.pi * f_cad * t)  # sway at half-cadence
            
            # Gyroscopic angular velocities (torso / limb rotation)
            gyro = np.zeros((num_samples, 3))
            gyro[:, 0] = 0.4 * np.sin(2 * np.pi * f_cad * t)
            gyro[:, 1] = 0.6 * np.sin(np.pi * f_cad * t + 0.5)
            gyro[:, 2] = 0.2 * np.sin(2 * np.pi * f_cad * t + 1.2)
            
        elif activity == "typing":
            # Rapid micro-keystroke shocks and localized wrist tilts
            accel[:, 2] += 0.15 * np.sin(2 * np.pi * 3.5 * t)
            accel[:, 0] += 0.08 * np.sin(2 * np.pi * 4.2 * t + 0.2)
            gyro = np.zeros((num_samples, 3))
            gyro[:, 0] = 0.05 * np.sin(2 * np.pi * 3.8 * t)
            gyro[:, 1] = 0.04 * np.sin(2 * np.pi * 4.5 * t)
            gyro[:, 2] = 0.03 * np.sin(2 * np.pi * 2.9 * t)
            
        elif activity == "head_browsing":
            # Headset saccades, micro-nodding, smooth pursuit
            f_head = 0.35 + (user_id % 5) * 0.05
            accel[:, 0] += 0.12 * np.sin(2 * np.pi * f_head * t)
            accel[:, 1] += 0.18 * np.cos(2 * np.pi * f_head * 1.3 * t)
            gyro = np.zeros((num_samples, 3))
            gyro[:, 0] = 0.15 * np.cos(2 * np.pi * f_head * t)       # Pitch
            gyro[:, 1] = 0.25 * np.sin(2 * np.pi * f_head * 0.8 * t) # Yaw
            gyro[:, 2] = 0.08 * np.sin(2 * np.pi * f_head * 1.5 * t) # Roll
            
        else:  # "idle_standing"
            # Postural sway (0.2 Hz)
            accel[:, 0] += 0.05 * np.sin(2 * np.pi * 0.2 * t)
            accel[:, 1] += 0.06 * np.cos(2 * np.pi * 0.18 * t)
            gyro = np.zeros((num_samples, 3))
            gyro[:, 0] = 0.02 * np.sin(2 * np.pi * 0.2 * t)
            gyro[:, 1] = 0.02 * np.cos(2 * np.pi * 0.2 * t)

        # 3. Add User-Specific Physiological Micro-Tremor (The Key Biometric Side-Channel)
        f_tremor = profile["tremor_freq"]
        a_tremor = profile["tremor_amp"]
        phases = profile["tremor_phase"]
        
        for i in range(3):
            accel[:, i] += a_tremor * np.sin(2 * np.pi * f_tremor * t + phases[i])
            gyro[:, i] += profile["gyro_tremor_amp"] * np.sin(2 * np.pi * profile["gyro_tremor_freq"] * t + phases[i])
            
        # 4. Add Real-world MEMS Sensor White Noise & Quantization Error
        accel += self.rng.normal(0, noise_std_accel, size=accel.shape)
        gyro += self.rng.normal(0, noise_std_gyro, size=gyro.shape)
        
        df = pd.DataFrame({
            "timestamp": t,
            "user_id": user_id,
            "session_id": session_id,
            "activity": activity,
            "accel_x": accel[:, 0],
            "accel_y": accel[:, 1],
            "accel_z": accel[:, 2],
            "gyro_x": gyro[:, 0],
            "gyro_y": gyro[:, 1],
            "gyro_z": gyro[:, 2],
        })
        return df

    def generate_population_dataset(
        self,
        num_users: int = 15,
        sessions_per_user: int = 4,
        session_duration_sec: float = 25.0,
        activities: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generates a comprehensive multi-user dataset across multiple sessions and activities.
        """
        if activities is None:
            activities = ["walking", "head_browsing", "typing", "idle_standing"]
            
        all_dfs = []
        for user_id in range(1, num_users + 1):
            for s_idx in range(1, sessions_per_user + 1):
                act = activities[(user_id + s_idx) % len(activities)]
                df_session = self.generate_session(
                    user_id=user_id,
                    session_id=s_idx,
                    duration_sec=session_duration_sec,
                    activity=act,
                )
                all_dfs.append(df_session)
                
        return pd.concat(all_dfs, ignore_index=True)


def load_or_generate_dataset(
    cache_path: Optional[str] = None,
    num_users: int = 15,
    sessions_per_user: int = 4,
    duration_sec: float = 25.0,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """
    Loads dataset from disk if cached, or synthesizes a fresh population dataset.
    """
    import os
    if cache_path and os.path.exists(cache_path) and not force_regenerate:
        print(f"Loading cached dataset from: {cache_path}")
        return pd.read_parquet(cache_path) if cache_path.endswith(".parquet") else pd.read_csv(cache_path)
    
    print(f"Generating synthetic population dataset ({num_users} users, {sessions_per_user} sessions/user)...")
    sim = BiometricIMUSimulator(sampling_rate_hz=100.0, random_seed=42)
    df = sim.generate_population_dataset(
        num_users=num_users,
        sessions_per_user=sessions_per_user,
        session_duration_sec=duration_sec,
    )
    
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        if cache_path.endswith(".parquet"):
            df.to_parquet(cache_path, index=False)
        else:
            df.to_csv(cache_path, index=False)
        print(f"Dataset cached to: {cache_path}")
        
    return df
