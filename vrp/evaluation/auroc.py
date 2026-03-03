"""Evaluation utilities: AUROC, point-biserial correlation, metric fusion.

Implements Section 3.4 of the paper:
  - AUROC (Eq. 8–9) as the primary evaluation metric
  - Point-Biserial Correlation R_pb as a secondary metric
  - Combined AUROC via logistic regression fusion (Eq. 10)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ------------------------------------------------------------------ #
#  Primary metrics                                                     #
# ------------------------------------------------------------------ #

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute AUROC for a scalar reliability score and binary labels.

    Parameters
    ----------
    scores : sequence of float
        Per-example reliability scores.  Higher = more likely correct.
    labels : sequence of int
        Binary ground-truth correctness (1 = correct, 0 = incorrect).

    Returns
    -------
    auroc_val : float
        Area Under the ROC Curve, in [0.5, 1.0] for useful predictors.
    """
    y_score = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(labels, dtype=int)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


def point_biserial_correlation(
    scores: Sequence[float], labels: Sequence[int]
) -> Tuple[float, float]:
    """Compute Point-Biserial Correlation R_pb between scores and binary labels.

    Returns
    -------
    r_pb : float
        Correlation coefficient in [−1, 1].
    p_value : float
        Two-tailed p-value.
    """
    r_pb, p_value = stats.pointbiserialr(labels, scores)
    return float(r_pb), float(p_value)


# ------------------------------------------------------------------ #
#  Combined AUROC via metric fusion  (Eq. 10)                         #
# ------------------------------------------------------------------ #

class MetricFusion:
    """Logistic-regression fusion of multiple reliability signals (Eq. 10).

    Fits on a training subset and evaluates on a held-out test subset.

    Usage
    -----
    >>> fusion = MetricFusion()
    >>> fusion.fit(train_features, train_labels)
    >>> test_auroc = fusion.auroc(test_features, test_labels)
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        self._fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> "MetricFusion":
        """Fit the fusion model on training features.

        Parameters
        ----------
        features : ndarray of shape (N_train, n_metrics)
            Each column is one reliability metric.
        labels : ndarray of shape (N_train,)
            Binary correctness labels.
        """
        z = self._scaler.fit_transform(features)
        self._clf.fit(z, labels)
        self._fitted = True
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return calibrated P(correct) for each example."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        z = self._scaler.transform(features)
        return self._clf.predict_proba(z)[:, 1]

    def auroc(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Compute combined AUROC on a held-out set."""
        proba = self.predict_proba(features)
        return float(roc_auc_score(labels, proba))


# ------------------------------------------------------------------ #
#  Results summary helper                                              #
# ------------------------------------------------------------------ #

def summarise_metrics(
    metric_scores: Dict[str, List[float]],
    labels: List[int],
) -> Dict[str, Dict[str, float]]:
    """Compute AUROC and R_pb for every metric in *metric_scores*.

    Parameters
    ----------
    metric_scores : dict mapping metric name → list of per-example scores
    labels : list of binary correctness labels

    Returns
    -------
    results : dict mapping metric name → {"auroc": ..., "r_pb": ..., "p_value": ...}
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, scores in metric_scores.items():
        auc = auroc(scores, labels)
        r_pb, p_val = point_biserial_correlation(scores, labels)
        results[name] = {"auroc": auc, "r_pb": r_pb, "p_value": p_val}
    return results
