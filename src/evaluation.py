"""
Evaluation metrics and scientific visualization module for biometric re-identification risks.
Computes Top-k accuracy, EER (Equal Error Rate), ROC-AUC, confusion matrices,
t-SNE latent space biometric clustering, and mitigation tradeoff curves.
"""

from typing import Dict, List, Optional, Tuple
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Style settings for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_reid_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    """
    Computes primary biometric identification & security metrics.
    """
    top1_acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    # Top-3 Accuracy
    top3_correct = 0
    top3_indices = np.argsort(y_proba, axis=1)[:, -3:]
    for i, target in enumerate(y_true):
        if target in classes[top3_indices[i]]:
            top3_correct += 1
    top3_acc = float(top3_correct / len(y_true)) if len(y_true) > 0 else 0.0
    
    # Multiclass One-vs-Rest ROC-AUC
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] > 1:
            auc_val = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
        else:
            auc_val = 1.0
    except Exception:
        auc_val = 0.0

    return {
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "macro_auc": auc_val,
        "sample_count": len(y_true),
        "target_count": len(classes),
    }


def compute_equal_error_rate(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    """
    Computes Biometric Equal Error Rate (EER) where FAR == FRR under one-vs-rest verification.
    """
    y_bin = label_binarize(y_true, classes=classes)
    eers = []
    
    for c_idx in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, c_idx], y_proba[:, c_idx])
        fnr = 1.0 - tpr
        # Find index where |fpr - fnr| is minimized
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        eers.append(eer)
        
    return float(np.mean(eers)) if len(eers) > 0 else 0.0


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    output_path: str,
) -> None:
    """Plots and saves normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"U{c}" for c in classes],
        yticklabels=[f"U{c}" for c in classes],
        cbar_kws={"label": "Normalized Identification Rate"},
        ax=ax,
    )
    ax.set_title("Cross-Session Biometric Re-Identification Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted User Identity", fontsize=11)
    ax.set_ylabel("True User Identity", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importances(
    df_imp: pd.DataFrame,
    output_path: str,
    top_n: int = 12,
) -> None:
    """Renders top side-channel leakage features by importance weight."""
    if df_imp.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    top_df = df_imp.head(top_n).sort_values(by="importance", ascending=True)
    
    ax.barh(top_df["feature"], top_df["importance"], color="#1f77b4", edgecolor="#0e4b77")
    ax.set_title(f"Top {top_n} Motion Side-Channel Leakage Features (Gini Importance)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Feature Importance", fontsize=11)
    ax.set_ylabel("Extracted IMU Feature", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_tsne_biometric_clusters(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str,
    perplexity: float = 25.0,
) -> None:
    """Visualizes latent biometric clustering of users in low-dimensional space."""
    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5.0, len(X) / 4)), random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y,
        cmap="tab20",
        alpha=0.85,
        edgecolors="w",
        linewidths=0.5,
        s=45,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("User Identity", fontsize=11)
    ax.set_title("t-SNE Embedding of IMU Feature Signatures (Biometric Clustered)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_defense_benchmark(
    results_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Renders privacy-utility degradation curves under defenses."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    sns.barplot(
        data=results_df,
        x="parameter",
        y="reid_accuracy",
        hue="defense_type",
        palette="viridis",
        ax=ax,
    )
    ax.axhline(1.0 / 15.0, color="red", linestyle="--", label="Random Guess Baseline (1/N)")
    ax.set_title("Impact of Privacy Mitigations on Re-Identification Risk", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Defense Configuration Parameter", fontsize=11)
    ax.set_ylabel("Re-Identification Top-1 Accuracy", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Countermeasure", loc="upper right")
    plt.xticks(rotation=25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
