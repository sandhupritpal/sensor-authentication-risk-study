"""
End-to-End Re-Identification Attack Experiment Runner.
Executes the full pipeline:
1. Data generation / loading
2. Feature extraction (time, frequency, kinematic jerk, correlations)
3. Train/test split by unseen session
4. Model training (Random Forest & GBDT)
5. Metric evaluation (Top-1, Top-3, EER, AUC)
6. Publication figure generation (Confusion Matrix, Feature Importances, t-SNE)
"""

import argparse
import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import load_or_generate_dataset
from src.features import extract_features_from_dataset
from src.attack_model import ReIdentificationAttackModel, split_by_session
from src.evaluation import (
    compute_reid_metrics,
    compute_equal_error_rate,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_tsne_biometric_clusters,
)


def run_experiment(
    num_users: int = 15,
    sessions_per_user: int = 4,
    duration_sec: float = 25.0,
    output_dir: str = "reports",
    models_dir: str = "models",
):
    print("=" * 70)
    print("  SENSOR-BASED AUTHENTICATION RISK STUDY: RE-IDENTIFICATION BENCHMARK")
    print("=" * 70)
    
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Dataset Generation
    data_cache = os.path.join("data", "processed", f"imu_population_{num_users}u.parquet")
    df = load_or_generate_dataset(
        cache_path=data_cache,
        num_users=num_users,
        sessions_per_user=sessions_per_user,
        duration_sec=duration_sec,
    )
    print(f"[+] Total IMU raw samples: {len(df):,} across {num_users} users and {sessions_per_user} sessions/user.")

    # 2. Feature Extraction
    print("[+] Extracting temporal & spectral feature vectors (window=2.0s, overlap=50%)...")
    X, y, meta_df, feature_names = extract_features_from_dataset(df, window_sec=2.0, overlap_ratio=0.5, fs=100.0)
    print(f"[+] Extracted {X.shape[0]} feature vectors with {X.shape[1]} features each.")

    # 3. Session-Based Train/Test Split (Unseen Session 4 as Test Target)
    test_session = sessions_per_user
    print(f"[+] Performing strict cross-session split (Train: Sessions 1..{test_session-1}, Test: Session {test_session})...")
    X_train, y_train, X_test, y_test = split_by_session(X, y, meta_df, test_session_id=test_session)
    print(f"[+] Train set: {len(X_train)} windows | Test set: {len(X_test)} windows.")

    # 4. Train Attack Classifier (Random Forest)
    print("\n[+] Training Random Forest Re-Identification Classifier...")
    rf_attack = ReIdentificationAttackModel(model_type="rf", random_state=42)
    rf_attack.fit(X_train, y_train, feature_names=feature_names)
    
    # Save model
    model_save_path = os.path.join(models_dir, "rf_reid_classifier.joblib")
    rf_attack.save(model_save_path)
    print(f"[+] Saved trained model checkpoint to: {model_save_path}")

    # 5. Evaluate Re-Identification Performance
    y_pred = rf_attack.predict(X_test)
    y_proba = rf_attack.predict_proba(X_test)
    classes = rf_attack.classes_

    metrics = compute_reid_metrics(y_test, y_pred, y_proba, classes)
    eer = compute_equal_error_rate(y_test, y_proba, classes)
    metrics["equal_error_rate"] = eer
    metrics["random_baseline_top1"] = round(1.0 / num_users, 4)

    print("\n" + "-" * 45)
    print("  SIDE-CHANNEL THREAT EVALUATION RESULTS")
    print("-" * 45)
    print(f"  Random Chance Baseline:   {metrics['random_baseline_top1']*100:.2f}%")
    print(f"  Top-1 Re-ID Accuracy:     {metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3 Re-ID Accuracy:     {metrics['top3_accuracy']*100:.2f}%")
    print(f"  Macro-Averaged F1 Score:  {metrics['macro_f1']:.4f}")
    print(f"  Macro-Averaged ROC-AUC:   {metrics['macro_auc']:.4f}")
    print(f"  Biometric Equal Error Rate (EER): {metrics['equal_error_rate']*100:.2f}%")
    print("-" * 45)

    # 6. Generate Figures
    print("\n[+] Generating publication-quality figures...")
    
    # Confusion Matrix
    cm_path = os.path.join(output_dir, "figures", "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, classes, cm_path)
    print(f"  -> Generated: {cm_path}")

    # Feature Importances
    imp_df = rf_attack.get_feature_importances(top_n=12)
    imp_path = os.path.join(output_dir, "figures", "feature_importances.png")
    plot_feature_importances(imp_df, imp_path)
    print(f"  -> Generated: {imp_path}")

    # t-SNE Clustering
    tsne_path = os.path.join(output_dir, "figures", "tsne_clusters.png")
    plot_tsne_biometric_clusters(X_test, y_test, tsne_path)
    print(f"  -> Generated: {tsne_path}")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved metrics report to: {metrics_path}")

    print("\n[SUCCESS] Re-identification experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IMU Re-Identification Experiment")
    parser.add_argument("--users", type=int, default=15, help="Number of simulated users")
    parser.add_argument("--sessions", type=int, default=4, help="Sessions per user")
    parser.add_argument("--duration", type=float, default=25.0, help="Duration per session in seconds")
    args = parser.parse_args()

    run_experiment(
        num_users=args.users,
        sessions_per_user=args.sessions,
        duration_sec=args.duration,
    )
