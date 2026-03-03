"""2-layer MLP binary-correctness probe (Section 3.3.2).

Architecture (Eq. 6):
    P(Correct | h_last) = σ(W_2 · ReLU(W_1 · h_last + b_1) + b_2)

where h_last ∈ R^d is the final-token hidden state at a chosen layer
(d = 4096 for LLaVA-7B).

Training details (paper §4):
  - 80/20 stratified train/test split
  - Adam optimizer, lr = 1e-4
  - 50 epochs
  - Binary cross-entropy loss
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np


class MLPProbe(nn.Module):
    """2-layer MLP binary classifier operating on a hidden-state vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit (pre-sigmoid) of shape (batch,)."""
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

def train_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    input_dim: Optional[int] = None,
    hidden_dim: int = 256,
    lr: float = 1e-4,
    epochs: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple[MLPProbe, dict]:
    """Train the MLP probe on (hidden_states, binary_correctness) pairs.

    Parameters
    ----------
    hidden_states : ndarray of shape (N, d)
        Hidden-state vectors collected at the chosen probe layer.
    labels : ndarray of shape (N,)
        Binary ground-truth correctness labels (1 = correct, 0 = incorrect).
    input_dim : int, optional
        Feature dimension.  Inferred from ``hidden_states`` if not provided.
    hidden_dim : int
        Width of the single hidden layer (default 256).
    lr : float
        Adam learning rate (default 1e-4 as in the paper).
    epochs : int
        Training epochs (default 50).
    test_size : float
        Fraction of samples reserved for evaluation (default 0.2).
    random_state : int
        Seed for reproducible stratified split.
    device : str
        PyTorch device string.

    Returns
    -------
    probe : MLPProbe
        Trained probe moved to CPU.
    metrics : dict
        Training and validation accuracy + AUROC on the held-out split.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    if input_dim is None:
        input_dim = hidden_states.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        hidden_states, labels, test_size=test_size,
        stratify=labels, random_state=random_state
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    probe = MLPProbe(input_dim, hidden_dim).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    # ---- Evaluation ----
    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val_t).cpu().numpy()
    val_probs = _sigmoid(val_logits)
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_preds)),
        "val_auroc": float(roc_auc_score(y_val, val_probs)),
    }
    return probe.cpu(), metrics


# ------------------------------------------------------------------ #
#  Persistence                                                         #
# ------------------------------------------------------------------ #

def save_probe(probe: MLPProbe, path: str | Path) -> None:
    """Save probe weights to a .pt file."""
    torch.save(probe.state_dict(), str(path))


def load_probe(path: str | Path, input_dim: int, hidden_dim: int = 256) -> MLPProbe:
    """Load a saved probe from a .pt file."""
    probe = MLPProbe(input_dim, hidden_dim)
    probe.load_state_dict(torch.load(str(path), map_location="cpu"))
    probe.eval()
    return probe


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
