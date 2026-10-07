"""
Proof-of-Concept biometric re-identification attack classifiers.
Implements Random Forest, Gradient Boosted Decision Trees, and Multi-Layer Perceptron
to deanonymize users across distinct, unseen sessions.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ReIdentificationAttackModel:
    """
    Threat Actor Classifier: Learns biometric signatures from historical IMU telemetry
    to re-identify targets across anonymous future sessions.
    """

    SUPPORTED_MODELS = ["rf", "gbdt", "mlp"]

    def __init__(
        self,
        model_type: str = "rf",
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {self.SUPPORTED_MODELS}")
            
        self.model_type = model_type
        self.random_state = random_state
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=4,
                n_jobs=-1,
                random_state=random_state,
            )
        elif model_type == "gbdt":
            self.model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=10,
                random_state=random_state,
            )
        elif model_type == "mlp":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=300,
                    early_stopping=True,
                    random_state=random_state,
                )),
            ])

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ReIdentificationAttackModel":
        """Trains the attack model on labeled training windows."""
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        elif hasattr(self.model, "named_steps") and hasattr(self.model.named_steps["mlp"], "classes_"):
            self.classes_ = self.model.named_steps["mlp"].classes_
        else:
            self.classes_ = np.unique(y_train)
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the top-1 re-identified user predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns posterior probability distributions over user identities."""
        return self.model.predict_proba(X)

    def predict_top_k(self, X: np.ndarray, k: int = 3) -> np.ndarray:
        """Returns top-k candidate identities for each sample."""
        probas = self.predict_proba(X)
        top_k_indices = np.argsort(probas, axis=1)[:, -k:][:, ::-1]
        return self.classes_[top_k_indices]

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extracts top discriminative side-channel features (available for Tree-based models).
        """
        if self.model_type != "rf":
            return pd.DataFrame()
            
        importances = self.model.feature_importances_
        names = self.feature_names or [f"feat_{i}" for i in range(len(importances))]
        df_imp = pd.DataFrame({"feature": names, "importance": importances})
        return df_imp.sort_values(by="importance", ascending=False).head(top_n)

    def save(self, filepath: str) -> None:
        """Saves trained model weights and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "classes": self.classes_,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ReIdentificationAttackModel":
        """Loads a persisted attack model."""
        payload = joblib.load(filepath)
        instance = cls(model_type=payload["model_type"])
        instance.model = payload["model"]
        instance.feature_names = payload.get("feature_names")
        instance.classes_ = payload.get("classes")
        return instance


def split_by_session(
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    test_session_id: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset strictly by session to evaluate true cross-session biometric leakage
    and prevent intra-session temporal contamination.
    """
    test_mask = (metadata_df["session_id"] == test_session_id).to_numpy()
    train_mask = ~test_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    return X_train, y_train, X_test, y_test
