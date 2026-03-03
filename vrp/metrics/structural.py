"""Structural reliability metrics (Stage 1 of VRP).

Implements Section 3.2 of the paper:
  - Attention Cluster Count C_k  (Eq. 2)  via DBSCAN
  - Spatial Entropy H_s           (Eq. 3)
  - Attention Evolution ΔH_s     (layer-wise change in entropy)

All functions operate on a single aggregated attention map
``M ∈ R^{H_grid × W_grid}`` (reshaped from the 576-dim CLIP grid for LLaVA,
or appropriately resized for other encoders).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


# ------------------------------------------------------------------ #
#  Constants (paper §4, Appendix A.7)                                 #
# ------------------------------------------------------------------ #

_DBSCAN_EPS: float = 1.5
_DBSCAN_MIN_SAMPLES: int = 3
_TOP_PERCENTILE: float = 90.0      # keep top 10 % of activation mass
_ENTROPY_DELTA: float = 1e-9       # numerical stability (δ in Eq. 3)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def aggregate_attention(
    attention_maps: List["torch.Tensor"],
    grid_size: int = 24,
) -> np.ndarray:
    """Average a list of per-layer head-averaged attention vectors into M.

    Parameters
    ----------
    attention_maps:
        Each element is a 1-D tensor of length ``grid_size**2`` (one entry
        per visual token), already averaged over heads for a single layer.
    grid_size:
        Square root of the number of visual tokens (default 24 for LLaVA's
        CLIP ViT-L/14 which yields 24×24 = 576 tokens).

    Returns
    -------
    M : np.ndarray of shape (grid_size, grid_size)
        Aggregated, L1-normalised attention map.
    """
    import torch
    if len(attention_maps) == 0:
        return np.ones((grid_size, grid_size), dtype=np.float32) / (grid_size ** 2)

    stacked = torch.stack(attention_maps, dim=0).mean(0)  # (S,)
    # Truncate or pad to exactly grid_size^2
    s = grid_size ** 2
    vec = stacked[:s].numpy().astype(np.float32)
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec.reshape(grid_size, grid_size)


def spatial_entropy(M: np.ndarray) -> float:
    """Compute Shannon spatial entropy H_s of attention map M (Eq. 3).

    Parameters
    ----------
    M : ndarray of shape (H, W)
        Normalised attention map (values in [0, 1], summing to ~1).

    Returns
    -------
    H_s : float
        Shannon entropy in nats.
    """
    flat = M.flatten().astype(np.float64)
    return float(-np.sum(flat * np.log(flat + _ENTROPY_DELTA)))


def cluster_count(M: np.ndarray) -> int:
    """Compute attention cluster count C_k via DBSCAN (Eq. 2).

    Parameters
    ----------
    M : ndarray of shape (H, W)
        Normalised attention map.

    Returns
    -------
    C_k : int
        Number of DBSCAN clusters found in the top-10 % activation region
        (excluding noise points labelled –1).
    """
    threshold = np.percentile(M, _TOP_PERCENTILE)
    coords = np.argwhere(M >= threshold).astype(np.float32)

    if len(coords) < _DBSCAN_MIN_SAMPLES:
        return 0

    labels = DBSCAN(
        eps=_DBSCAN_EPS, min_samples=_DBSCAN_MIN_SAMPLES
    ).fit_predict(coords)
    n_clusters = len(set(labels) - {-1})
    return n_clusters


def attention_evolution(
    layer_maps: List["torch.Tensor"],
    grid_size: int = 24,
) -> Tuple[List[float], List[float]]:
    """Compute per-layer spatial entropy sequence and its differences ΔH_s.

    Parameters
    ----------
    layer_maps : list of tensors
        Each element is a (S,) tensor, one per layer (already head-averaged).
    grid_size : int
        Spatial grid size (default 24 for LLaVA).

    Returns
    -------
    entropies : list of float
        H_s at each layer.
    deltas : list of float
        ΔH_s[l] = H_s[l] − H_s[l−1], length len(entropies) − 1.
    """
    entropies: List[float] = []
    for tensor in layer_maps:
        import torch
        vec = tensor[:grid_size ** 2].numpy().astype(np.float32)
        if vec.sum() > 0:
            vec /= vec.sum()
        M = vec.reshape(grid_size, grid_size)
        entropies.append(spatial_entropy(M))

    deltas = [entropies[i] - entropies[i - 1] for i in range(1, len(entropies))]
    return entropies, deltas
