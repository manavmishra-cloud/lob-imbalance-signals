"""Deep sequence models for limit order book direction prediction.

Two architectures, both predicting a 3-class target {-1, 0, +1} representing
the sign of the next-N-event mid-price change:

1. **LSTM** — Standard 2-layer LSTM consuming a sequence of LOB features
   (OFI at multiple depths, volume imbalance, microprice deviation, spread).
2. **Transformer (encoder-only)** — Sinusoidal positional encoding +
   self-attention over the same feature sequence.

Both expose an sklearn-compatible interface:
    .fit(X, y) where X is (N, T, F) sequences and y is (N,) labels in {-1, 0, +1}
    .predict(X) returns (N,) integer predictions in {-1, 0, +1}
    .predict_proba(X) returns (N, 3) class probabilities

Targets are mapped internally to {0, 1, 2} for cross-entropy training and
mapped back at predict time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sequence dataset construction
# ---------------------------------------------------------------------------

def build_lob_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (T, F) feature matrix + (T,) targets into sliding windows.

    For row t with t >= seq_len, X[t-seq_len] is the (seq_len, F) window
    ending at t-1, and y[t-seq_len] is the target at row t.
    """
    n = len(features)
    n_samples = n - seq_len
    if n_samples <= 0:
        return np.zeros((0, seq_len, features.shape[1])), np.zeros(0, dtype=np.int64)

    X = np.empty((n_samples, seq_len, features.shape[1]), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)
    for t in range(n_samples):
        X[t] = features[t : t + seq_len]
        y[t] = targets[t + seq_len]
    return X, y


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class LSTMClassifier(nn.Module):
        """2-layer LSTM with dropout, 3-class output."""

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            n_classes: int = 3,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last)  # (B, n_classes) logits


    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1), :]


    class TransformerClassifier(nn.Module):
        """Encoder-only Transformer with sinusoidal positional encoding."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.2,
            seq_len: int = 100,
            n_classes: int = 3,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_proj(x)
            h = self.pos_enc(h)
            h = self.encoder(h)
            last = h[:, -1, :]
            return self.head(last)


# ---------------------------------------------------------------------------
# Training wrapper (sklearn-compatible)
# ---------------------------------------------------------------------------

@dataclass
class DeepConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    early_stopping_patience: int = 5
    device: str = "cpu"
    verbose: bool = False


class _DeepClassifierWrapper:
    """Common training logic for both LSTM and Transformer classifiers.

    Handles target remapping ({-1,0,+1} <-> {0,1,2}), standardization of
    inputs, training with early stopping, and prediction.
    """

    _class_map = {-1: 0, 0: 1, 1: 2}
    _inv_map = {v: k for k, v in _class_map.items()}

    def __init__(self, model_factory, seq_len: int, config: Optional[DeepConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required.")
        self.model_factory = model_factory  # callable(n_features) -> nn.Module
        self.seq_len = seq_len
        self.config = config or DeepConfig()
        self.model = None
        self.feature_mean = None
        self.feature_std = None

    def _to_sequences(self, X_flat: np.ndarray) -> np.ndarray:
        """Convert a 2D feature matrix into sequences via build_lob_sequences-style overlap.

        Expects X_flat shape (N, F) of per-event features. Returns (N - seq_len, seq_len, F).
        """
        return X_flat  # by convention, fit/predict receive already-sequenced X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DeepClassifierWrapper":
        # X is (N, T, F); y is (N,) in {-1, 0, +1}
        device = torch.device(self.config.device)
        n, T, F = X.shape

        # Standardize per-feature
        flat = X.reshape(-1, F)
        self.feature_mean = flat.mean(axis=0)
        self.feature_std = flat.std(axis=0) + 1e-8
        X_n = (X - self.feature_mean) / self.feature_std

        # Map targets
        y_mapped = np.array([self._class_map[int(v)] for v in y], dtype=np.int64)

        # Temporal split for early stopping
        n_val = max(int(n * self.config.val_frac), 1)
        X_train, y_train = X_n[:-n_val], y_mapped[:-n_val]
        X_val, y_val = X_n[-n_val:], y_mapped[-n_val:]

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        self.model = self.model_factory(F).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # Class weights to handle imbalance: inverse frequency on the train split
        class_counts = np.bincount(y_train, minlength=3).astype(np.float32) + 1.0
        class_weight = torch.from_numpy(class_counts.sum() / (3.0 * class_counts)).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)

        best_val = float("inf")
        best_state = None
        patience_left = self.config.early_stopping_patience

        for epoch in range(self.config.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = self.model(xb)
                    val_losses.append(criterion(logits, yb).item())
            val_loss = float(np.mean(val_losses))

            if self.config.verbose and epoch % 5 == 0:
                print(f"  epoch {epoch}: val={val_loss:.4f}")

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.config.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if self.config.verbose:
                        print(f"  early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return np.array([self._inv_map[int(v)] for v in idx])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_n = (X - self.feature_mean) / self.feature_std
        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_n).float().to(device)
            logits = self.model(x_t)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba


# ---------------------------------------------------------------------------
# Public classifier factories
# ---------------------------------------------------------------------------

class LSTMBaseline:
    """LSTM 3-class classifier wrapper for the walk-forward harness."""

    def __init__(self, seq_len: int = 100, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTMBaseline.")
        self.seq_len = seq_len
        factory = lambda n_features: LSTMClassifier(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self._inner = _DeepClassifierWrapper(factory, seq_len=seq_len, config=DeepConfig(**kwargs))

    def fit(self, X, y):
        # X shape is (N, T*F) flat from walk_forward, need to reshape to (N, T, F)
        # Convention: caller knows seq_len, passes already-shaped X. If 2D, reshape.
        if X.ndim == 2:
            n, tf = X.shape
            F = tf // self.seq_len
            X = X.reshape(n, self.seq_len, F)
        self._inner.fit(X, y)
        return self

    def predict(self, X):
        if X.ndim == 2:
            n, tf = X.shape
            F = tf // self.seq_len
            X = X.reshape(n, self.seq_len, F)
        return self._inner.predict(X)


class TransformerBaseline:
    """Transformer 3-class classifier wrapper."""

    def __init__(self, seq_len: int = 100, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2,
                 **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TransformerBaseline.")
        self.seq_len = seq_len
        factory = lambda n_features: TransformerClassifier(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_len=seq_len,
        )
        self._inner = _DeepClassifierWrapper(factory, seq_len=seq_len, config=DeepConfig(**kwargs))

    def fit(self, X, y):
        if X.ndim == 2:
            n, tf = X.shape
            F = tf // self.seq_len
            X = X.reshape(n, self.seq_len, F)
        self._inner.fit(X, y)
        return self

    def predict(self, X):
        if X.ndim == 2:
            n, tf = X.shape
            F = tf // self.seq_len
            X = X.reshape(n, self.seq_len, F)
        return self._inner.predict(X)
