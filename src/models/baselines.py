"""Baseline models for next-event price direction prediction.

Two baseline families:

1. **Linear**: logistic regression on engineered features (Cont, Kukanov, Stoikov-
   style OFI regression with a multinomial output).

2. **Tree**: XGBoost classifier, which historically does well on tabular LOB
   features and is a strong baseline before reaching for sequence models.

Both implement the sklearn-compatible interface (`fit`, `predict`, `predict_proba`).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LinearOFIBaseline:
    """Standardized features + multinomial logistic regression.

    Maps target values {-1, 0, +1} to internal classes {0, 1, 2} for sklearn
    compatibility, then maps back on predict.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 200):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
        )
        self._class_map = {-1: 0, 0: 1, 1: 2}
        self._inv_map = {v: k for k, v in self._class_map.items()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearOFIBaseline":
        X_scaled = self.scaler.fit_transform(X)
        y_mapped = np.array([self._class_map[int(v)] for v in y])
        self.clf.fit(X_scaled, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_mapped = self.clf.predict(X_scaled)
        return np.array([self._inv_map[int(v)] for v in y_mapped])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)


class XGBoostBaseline:
    """XGBoost classifier on engineered LOB features.

    Imported lazily so the module loads without xgboost installed if you only
    use the linear baseline first.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required for XGBoostBaseline. `pip install xgboost`.") from e

        self.clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        self._class_map = {-1: 0, 0: 1, 1: 2}
        self._inv_map = {v: k for k, v in self._class_map.items()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        y_mapped = np.array([self._class_map[int(v)] for v in y])
        self.clf.fit(X, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_mapped = self.clf.predict(X)
        return np.array([self._inv_map[int(v)] for v in y_mapped])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


class PersistenceBaseline:
    """Trivial baseline: predict the previous observed direction.

    Useful as a sanity-check floor — any model worth deploying should beat this.
    """

    def __init__(self):
        self._last = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        if len(y) > 0:
            self._last = int(y[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._last, dtype=int)
