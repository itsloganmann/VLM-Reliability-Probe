"""Mechanistic reliability metrics (Stage 2 of VRP).

Implements Section 3.3 of the paper:
  - Token Confidence P_tok  (Eq. 7)  – mean log-prob of generated tokens
  - Logit Lens / Truth Margin ΔM_l   – per-layer correct-vs-incorrect logit gap
  - Hidden-State Probe score          – output of the trained MLP classifier

These functions operate on outputs already collected by the model wrapper's
forward hooks.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Token Confidence  P_tok  (Eq. 7)                                   #
# ------------------------------------------------------------------ #

def token_confidence(logits: torch.Tensor) -> float:
    """Mean log-probability of the generated answer tokens.

    Parameters
    ----------
    logits : Tensor of shape (seq_len, vocab_size)
        Raw (unnormalised) logits for each generated token step.

    Returns
    -------
    P_tok : float
        Average log-probability.  Higher → model is more confident.
    """
    if logits.numel() == 0:
        return float("nan")

    log_probs = F.log_softmax(logits, dim=-1)          # (seq_len, vocab)
    # Greedy token at each step
    chosen = log_probs.max(dim=-1).values               # (seq_len,)
    return float(chosen.mean().item())


# ------------------------------------------------------------------ #
#  Logit Lens / Truth Margin  ΔM_l                                    #
# ------------------------------------------------------------------ #

def truth_margin(
    hidden_states: List[torch.Tensor],
    lm_head: torch.nn.Module,
    correct_token_id: int,
    top_k_incorrect: int = 1,
) -> List[float]:
    """Compute per-layer truth margin ΔM_l (paper §5.2).

    Projects each layer's hidden state through the LM head and computes
    the logit difference between the correct token and the top-1 incorrect
    token.

    Parameters
    ----------
    hidden_states : list of Tensor, shape (d,) each
        One hidden state vector per layer (final token), as collected by
        the model wrapper's hidden-state hooks.
    lm_head : nn.Module
        The language-model head (unembedding matrix + optional bias).
    correct_token_id : int
        Vocabulary index of the ground-truth answer token.
    top_k_incorrect : int
        Number of top incorrect tokens to average over (default 1).

    Returns
    -------
    margins : list of float
        ΔM_l for each layer; positive → model leans correct.
    """
    margins: List[float] = []
    lm_head.eval()

    with torch.no_grad():
        for h in hidden_states:
            logits = lm_head(h.unsqueeze(0)).squeeze(0)  # (vocab,)
            correct_logit = logits[correct_token_id].item()

            # Top-k incorrect logits (excluding the correct token)
            logits_copy = logits.clone()
            logits_copy[correct_token_id] = float("-inf")
            top_incorrect = logits_copy.topk(top_k_incorrect).values.mean().item()

            margins.append(correct_logit - top_incorrect)

    return margins


# ------------------------------------------------------------------ #
#  Hidden-State Probe Score                                            #
# ------------------------------------------------------------------ #

def probe_score(
    hidden_state: torch.Tensor,
    probe: torch.nn.Module,
) -> float:
    """Run the trained MLP probe on a single hidden state.

    Parameters
    ----------
    hidden_state : Tensor of shape (d,)
        Last-token hidden state from the chosen probe layer.
    probe : nn.Module
        A trained ``MLPProbe`` instance (see ``vrp/probes/mlp_probe.py``).

    Returns
    -------
    p_correct : float
        Probability in [0, 1] that the model's answer is correct.
    """
    probe.eval()
    with torch.no_grad():
        out = probe(hidden_state.unsqueeze(0))   # (1, 1)
        return float(torch.sigmoid(out).item())
