"""
classifier.py
--------------
Functional Module 3 (part C) / core of "Pattern Analysis" (Syllabus
Module 4): trains and applies a K-Nearest-Neighbour classifier, with
optional PCA dimensionality reduction, over HOG features to recognise
individual characters (digits + uppercase letters).
"""

import os
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from . import config
from .utils import get_logger, PipelineError

logger = get_logger(__name__)


class CharacterClassifier:
    """Wraps a KNN classifier (+ optional PCA) for character recognition."""

    def __init__(self, n_neighbors: int = config.KNN_NEIGHBOURS, use_pca: bool = True, pca_components: int = 40):
        self.n_neighbors = n_neighbors
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        self.pca: Optional[PCA] = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the classifier on a feature matrix X and label vector y.
        Returns a small report dict (train/test accuracy) for logging/report use.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if self.use_pca:
            n_components = min(self.pca_components, X_train.shape[0], X_train.shape[1])
            self.pca = PCA(n_components=n_components, random_state=42)
            X_train = self.pca.fit_transform(X_train)
            X_test = self.pca.transform(X_test)

        self.knn.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        logger.info("Classifier trained. Held-out test accuracy = %.4f", acc)
        return {"test_accuracy": acc, "classification_report": report}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels + confidence (based on neighbour vote fraction)."""
        if not self.is_trained:
            raise PipelineError("Classifier has not been trained or loaded.")
        X_in = self.pca.transform(X) if (self.use_pca and self.pca is not None) else X

        preds = self.knn.predict(X_in)
        proba = self.knn.predict_proba(X_in)
        confidences = proba.max(axis=1)
        return preds, confidences

    def save(self, path: str = config.MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {"knn": self.knn, "pca": self.pca, "use_pca": self.use_pca, "is_trained": self.is_trained},
            path,
        )
        logger.info("Saved trained classifier to %s", path)

    @classmethod
    def load(cls, path: str = config.MODEL_PATH) -> "CharacterClassifier":
        if not os.path.isfile(path):
            raise PipelineError(
                f"No trained model found at {path}. Run scripts/train_classifier.py first."
            )
        payload = joblib.load(path)
        obj = cls(use_pca=payload["use_pca"])
        obj.knn = payload["knn"]
        obj.pca = payload["pca"]
        obj.is_trained = payload["is_trained"]
        logger.info("Loaded trained classifier from %s", path)
        return obj
