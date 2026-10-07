"""
Interactive Streamlit Forensic Dashboard:
Sensor-Based Authentication Risk Study & Biometric Re-Identification Explorer.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import BiometricIMUSimulator
from src.features import extract_window_features
from src.defense import SensorDefenseTransformer
from src.attack_model import ReIdentificationAttackModel

st.set_page_config(
    page_title="IMU Side-Channel Risk Visualizer",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Sensor-Based Biometric Side-Channel Explorer")
st.markdown(
    """
    **Side-Channel Biometric Leakage in Consumer Wearables & Headsets**: 
    Demonstrating how unprivileged 6-DoF IMU motion telemetry (accelerometer & gyroscope) 
    leaks unique physiological tremor and behavioral signatures, enabling passive cross-session re-identification.
    """
)

# ----------------- SIDEBAR -----------------
st.sidebar.header("🎯 Session & Target Parameters")
user_id = st.sidebar.number_input("Target User ID", min_value=1, max_value=20, value=3)
activity = st.sidebar.selectbox("Activity Mode", ["walking", "head_browsing", "typing", "idle_standing"])
session_duration = st.sidebar.slider("Session Duration (seconds)", min_value=10, max_value=40, value=20)

st.sidebar.header("🛡️ Client/OS Defense Simulation")
enable_defense = st.sidebar.checkbox("Enable Privacy Defense", value=False)

noise_mech = None
epsilon = 1.0
target_fs = 100.0
quant_step = None

if enable_defense:
    defense_mode = st.sidebar.selectbox("Defense Mechanism", ["Differential Privacy (Laplace)", "Frequency Throttling", "Bit Quantization"])
    if defense_mode == "Differential Privacy (Laplace)":
        noise_mech = "laplace"
        epsilon = st.sidebar.slider("Privacy Budget (Epsilon ε)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    elif defense_mode == "Frequency Throttling":
        target_fs = st.sidebar.select_slider("Throttled Sampling Rate (Hz)", options=[10.0, 20.0, 50.0, 100.0], value=20.0)
    elif defense_mode == "Bit Quantization":
        quant_step = st.sidebar.select_slider("Quantization Step (m/s²)", options=[0.1, 0.25, 0.5, 1.0], value=0.5)

# ----------------- DATA GENERATION -----------------
@st.cache_data
def get_simulated_session(u_id: int, act: str, dur: float):
    sim = BiometricIMUSimulator(sampling_rate_hz=100.0, random_seed=42 + u_id)
    return sim.generate_session(user_id=u_id, session_id=1, duration_sec=dur, activity=act)

raw_df = get_simulated_session(user_id, activity, session_duration)

# Apply Defense if configured
if enable_defense:
    transformer = SensorDefenseTransformer(
        noise_mechanism=noise_mech,
        epsilon=epsilon,
        target_fs=target_fs if target_fs < 100.0 else None,
        quantization_step=quant_step,
        original_fs=100.0,
    )
    active_df = transformer.apply(raw_df)
else:
    active_df = raw_df.copy()

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Raw Telemetry & Spectra",
    "🎯 Live Re-Identification Attack",
    "🛡️ Defense Analysis",
    "📄 Threat Model & Ethics",
])

with tab1:
    st.subheader(f"Motion Telemetry — User {user_id} ({activity.capitalize()})")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_acc, ax_acc = plt.subplots(figsize=(6, 3))
        ax_acc.plot(active_df["timestamp"], active_df["accel_x"], label="Acc X", alpha=0.8, color="#1f77b4")
        ax_acc.plot(active_df["timestamp"], active_df["accel_y"], label="Acc Y", alpha=0.8, color="#ff7f0e")
        ax_acc.plot(active_df["timestamp"], active_df["accel_z"], label="Acc Z", alpha=0.8, color="#2ca02c")
        ax_acc.set_xlabel("Time (seconds)")
        ax_acc.set_ylabel("Acceleration (m/s²)")
        ax_acc.set_title("3-Axis Accelerometer Stream")
        ax_acc.legend(loc="upper right", fontsize=8)
        st.pyplot(fig_acc)
        
    with col2:
        fig_gyro, ax_gyro = plt.subplots(figsize=(6, 3))
        ax_gyro.plot(active_df["timestamp"], active_df["gyro_x"], label="Gyro X", alpha=0.8, color="#d62728")
        ax_gyro.plot(active_df["timestamp"], active_df["gyro_y"], label="Gyro Y", alpha=0.8, color="#9467bd")
        ax_gyro.plot(active_df["timestamp"], active_df["gyro_z"], label="Gyro Z", alpha=0.8, color="#8c564b")
        ax_gyro.set_xlabel("Time (seconds)")
        ax_gyro.set_ylabel("Angular Vel (rad/s)")
        ax_gyro.set_title("3-Axis Gyroscope Stream")
        ax_gyro.legend(loc="upper right", fontsize=8)
        st.pyplot(fig_gyro)

    # Power Spectral Density
    st.subheader("Frequency Spectrum: Isolating Physiological Micro-Tremor (8–12 Hz)")
    sig = active_df["accel_z"].to_numpy()
    sig_centered = sig - np.mean(sig)
    fft_vals = np.abs(np.fft.rfft(sig_centered))**2
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / (target_fs if enable_defense and target_fs < 100.0 else 100.0))
    
    fig_psd, ax_psd = plt.subplots(figsize=(10, 3))
    ax_psd.plot(freqs, fft_vals, color="#333333", lw=1.2)
    ax_psd.axvspan(8.0, 12.0, color="orange", alpha=0.25, label="Biometric Tremor Band (8-12 Hz)")
    ax_psd.axvspan(1.0, 3.0, color="lightblue", alpha=0.25, label="Locomotion Band (1-3 Hz)")
    ax_psd.set_xlim(0, min(50, freqs[-1]))
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power Spectral Density")
    ax_psd.legend(loc="upper right")
    st.pyplot(fig_psd)

with tab2:
    st.subheader("PoC Re-Identification Attack Simulator")
    st.markdown(
        """
        The attacker model extracts 64 statistical and frequency-domain biometric features from a 2.0-second 
        motion window and classifies the candidate user against known profile signatures.
        """
    )
    
    # Check if pre-trained model exists or train on the fly
    model_path = os.path.join("models", "rf_reid_classifier.joblib")
    if os.path.exists(model_path):
        model = ReIdentificationAttackModel.load(model_path)
    else:
        st.info("Training lightweight live model across 10 user profiles...")
        sim = BiometricIMUSimulator(sampling_rate_hz=100.0, random_seed=42)
        pop_df = sim.generate_population_dataset(num_users=10, sessions_per_user=3, session_duration_sec=15.0)
        from src.features import extract_features_from_dataset
        X_live, y_live, meta_live, f_names = extract_features_from_dataset(pop_df)
        model = ReIdentificationAttackModel(model_type="rf").fit(X_live, y_live, feature_names=f_names)
        
    # Extract features from current session
    w_len = int(2.0 * (target_fs if enable_defense and target_fs < 100.0 else 100.0))
    if len(active_df) >= w_len:
        sub_w = active_df.iloc[:w_len]
        feats = extract_window_features(sub_w, fs=target_fs if enable_defense and target_fs < 100.0 else 100.0)
        
        # Align features
        if model.feature_names:
            x_vec = np.array([[feats.get(fn, 0.0) for fn in model.feature_names]])
        else:
            x_vec = np.array([list(feats.values())])
            
        probas = model.predict_proba(x_vec)[0]
        classes = model.classes_
        top_idx = int(np.argmax(probas))
        predicted_user = classes[top_idx]
        confidence = probas[top_idx]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("True Target User", f"User {user_id}")
        c2.metric("Attacker Prediction", f"User {predicted_user}")
        c3.metric("Prediction Confidence", f"{confidence*100:.1f}%")
        
        if predicted_user == user_id:
            st.error(f"🚨 **RE-IDENTIFICATION SUCCESS**: Attacker deanonymized User {user_id} with {confidence*100:.1f}% confidence!")
        else:
            st.success(f"🛡️ **DEANONYMIZATION FAILED**: Classified as User {predicted_user} (Confusion/Defense successful).")
            
        # Top Candidates Bar Chart
        top_k_indices = np.argsort(probas)[-5:][::-1]
        top_users = [f"User {classes[i]}" for i in top_k_indices]
        top_probs = [probas[i] * 100 for i in top_k_indices]
        
        fig_cand, ax_cand = plt.subplots(figsize=(7, 3))
        ax_cand.barh(top_users[::-1], top_probs[::-1], color="#d9534f")
        ax_cand.set_xlabel("Attacker Confidence (%)")
        ax_cand.set_title("Top-5 Re-Identification Candidates")
        st.pyplot(fig_cand)

with tab3:
    st.subheader("Countermeasure Efficacy & Tradeoffs")
    st.markdown(
        """
        Evaluating client-side defenses:
        - **Differential Privacy**: Injects calibrated Laplace/Gaussian noise directly into high-frequency sensor readings.
        - **Frequency Throttling**: Restricts sample rates from 100 Hz down to 20 Hz, filtering out micro-tremor frequencies (>10 Hz).
        """
    )
    st.write(
        "Current Defense Status:",
        f"**{'ACTIVE' if enable_defense else 'DISABLED'}**",
        f"| Mechanism: **{noise_mech or ('Throttling' if target_fs < 100 else 'None')}**"
    )

with tab4:
    st.subheader("Threat Model & Responsible Disclosure")
    st.markdown(
        """
        ### Threat Vector
        Unlike camera and microphone interfaces which require explicit runtime permission dialogs,
        motion sensors (accelerometer, gyroscope) are frequently accessible by unprivileged web scripts
        (e.g., via `DeviceMotionEvent` in WebXR / mobile browsers) or background apps.
        
        ### Risk Analysis
        - **Cross-Session Tracking**: Third-party trackers can compute biometric signatures to correlate anonymous sessions.
        - **Keystroke / PIN Inference**: Wrist-worn smartwatches capture micro-deflections during keypad interaction.
        - **Headset Eavesdropping**: Speech micro-vibrations can be partially recovered via VR/AR headset IMUs.
        
        ### Proposed Standards Hardening
        1. Rate-limit unprompted sensor APIs to $\le 20\\text{ Hz}$.
        2. Introduce explicit permission gates for high-frequency ($\ge 50\\text{ Hz}$) sensor access.
        3. Enforce OS-level differential privacy noise injection when running in background contexts.
        """
    )
