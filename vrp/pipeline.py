"""VRP end-to-end diagnostic pipeline.

Ties together all three stages described in Figure 1 of the paper:

  Stage 1 – Visual Encoder
    - Extract cross-attention maps via forward hooks
    - Compute structural metrics: H_s, C_k, ΔH_s

  Stage 2 – LLM Backbone
    - Capture hidden states via forward hooks
    - Compute mechanistic metrics: truth margin ΔM_l, token confidence P_tok,
      hidden-state probe score

  Stage 3 – Generation
    - Sample K stochastic outputs
    - Compute behavioral metric: Self-Consistency S_SC

Usage
-----
>>> from vrp.pipeline import VRPPipeline
>>> from vrp.models.llava import LLaVAModel
>>> model = LLaVAModel()
>>> pipeline = VRPPipeline(model)
>>> result = pipeline.run(image, question, ground_truth_answer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from vrp.metrics.structural import aggregate_attention, spatial_entropy, cluster_count, attention_evolution
from vrp.metrics.mechanistic import token_confidence, probe_score
from vrp.metrics.behavioral import self_consistency


# ------------------------------------------------------------------ #
#  Result dataclass                                                    #
# ------------------------------------------------------------------ #

@dataclass
class VRPResult:
    """All VRP signals for a single (image, question) example."""

    # ---- Input ----
    question: str
    ground_truth: Optional[str] = None
    predicted_answer: str = ""

    # ---- Stage 1: Structural ----
    spatial_entropy: float = float("nan")
    cluster_count: int = 0
    attention_entropies_per_layer: List[float] = field(default_factory=list)
    attention_deltas_per_layer: List[float] = field(default_factory=list)

    # ---- Stage 2: Mechanistic ----
    token_confidence: float = float("nan")
    probe_score: Optional[float] = None
    truth_margins_per_layer: List[float] = field(default_factory=list)

    # ---- Stage 3: Behavioral ----
    self_consistency_score: float = float("nan")
    sc_majority_answer: str = ""
    sc_all_answers: List[str] = field(default_factory=list)

    # ---- Ground-truth alignment ----
    is_correct: Optional[bool] = None


# ------------------------------------------------------------------ #
#  Pipeline                                                            #
# ------------------------------------------------------------------ #

class VRPPipeline:
    """Orchestrate all three VRP stages for a single model.

    Parameters
    ----------
    model :
        A concrete ``VRPModelBase`` instance (LLaVAModel, PaliGemmaModel,
        or Qwen2VLModel).
    probe :
        Optional pre-trained ``MLPProbe`` instance.  If ``None``, the
        probe score is omitted from results.
    probe_layer_idx :
        Index (0-based) into the ordered ``model.hidden_states`` list that
        corresponds to the best probe layer for this architecture.
        Defaults to -1 (last hooked layer).
    lm_head :
        Optional LM head module used for logit-lens truth-margin computation.
        If ``None``, truth-margin analysis is skipped.
    sc_k :
        Number of self-consistency samples (default 10).
    sc_temperature :
        Sampling temperature for self-consistency (default 0.7).
    sc_top_p :
        Nucleus sampling probability (default 0.9).
    grid_size :
        Visual token grid size per dimension (default 24 for LLaVA).
    """

    def __init__(
        self,
        model,
        probe=None,
        probe_layer_idx: int = -1,
        lm_head=None,
        sc_k: int = 10,
        sc_temperature: float = 0.7,
        sc_top_p: float = 0.9,
        grid_size: int = 24,
    ) -> None:
        self.model = model
        self.probe = probe
        self.probe_layer_idx = probe_layer_idx
        self.lm_head = lm_head
        self.sc_k = sc_k
        self.sc_temperature = sc_temperature
        self.sc_top_p = sc_top_p
        self.grid_size = grid_size

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        image: Image.Image,
        question: str,
        ground_truth: Optional[str] = None,
        correct_token_id: Optional[int] = None,
        run_sc: bool = True,
        run_probe: bool = True,
        run_logit_lens: bool = False,
    ) -> VRPResult:
        """Run the full VRP diagnostic for one example.

        Parameters
        ----------
        image :
            Input image.
        question :
            VQA question string.
        ground_truth :
            Expected answer string for correctness labelling.
        correct_token_id :
            Vocabulary index of the ground-truth answer token (needed for
            logit-lens truth-margin computation).
        run_sc :
            Whether to run the self-consistency stage (10× inference cost).
        run_probe :
            Whether to run the hidden-state probe (requires pre-trained probe).
        run_logit_lens :
            Whether to compute per-layer truth margins (requires lm_head).

        Returns
        -------
        VRPResult
        """
        # ---- Stage 1 + 2: greedy forward pass ----
        predicted_answer, logits = self.model.generate(
            image, question,
            do_sample=False, temperature=1.0,
        )

        # Stage 1 – Structural metrics
        M = aggregate_attention(self.model.attention_maps, self.grid_size)
        Hs = spatial_entropy(M)
        Ck = cluster_count(M)
        entropies, deltas = attention_evolution(
            self.model.attention_maps, self.grid_size
        )

        # Stage 2 – Mechanistic metrics
        P_tok = token_confidence(logits)

        probe_val: Optional[float] = None
        if run_probe and self.probe is not None and len(self.model.hidden_states) > 0:
            h = self.model.hidden_states[self.probe_layer_idx]
            probe_val = probe_score(h, self.probe)

        margins: List[float] = []
        if run_logit_lens and self.lm_head is not None and correct_token_id is not None:
            from vrp.metrics.mechanistic import truth_margin
            margins = truth_margin(self.model.hidden_states, self.lm_head, correct_token_id)

        # ---- Stage 3: Self-Consistency ----
        sc_score = float("nan")
        sc_majority = ""
        sc_answers: List[str] = []

        if run_sc:
            sc_majority, sc_score, sc_answers = self_consistency(
                image, question,
                generate_fn=self.model.generate,
                k=self.sc_k,
                temperature=self.sc_temperature,
                top_p=self.sc_top_p,
            )

        # ---- Correctness label ----
        is_correct: Optional[bool] = None
        if ground_truth is not None:
            is_correct = _answers_match(predicted_answer, ground_truth)

        return VRPResult(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer,
            spatial_entropy=Hs,
            cluster_count=Ck,
            attention_entropies_per_layer=entropies,
            attention_deltas_per_layer=deltas,
            token_confidence=P_tok,
            probe_score=probe_val,
            truth_margins_per_layer=margins,
            self_consistency_score=sc_score,
            sc_majority_answer=sc_majority,
            sc_all_answers=sc_answers,
            is_correct=is_correct,
        )

    def run_batch(
        self,
        examples: List[Dict[str, Any]],
        run_sc: bool = True,
        run_probe: bool = True,
        run_logit_lens: bool = False,
    ) -> List[VRPResult]:
        """Run the pipeline over a list of examples.

        Each example dict must have keys: ``"image"``, ``"question"``.
        Optional keys: ``"ground_truth"``, ``"correct_token_id"``.
        """
        results: List[VRPResult] = []
        for ex in examples:
            result = self.run(
                image=ex["image"],
                question=ex["question"],
                ground_truth=ex.get("ground_truth"),
                correct_token_id=ex.get("correct_token_id"),
                run_sc=run_sc,
                run_probe=run_probe,
                run_logit_lens=run_logit_lens,
            )
            results.append(result)
        return results


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _answers_match(predicted: str, ground_truth: str) -> bool:
    """Normalized binary correctness comparison."""
    import re
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return " ".join(s.split())
    return _norm(predicted) == _norm(ground_truth)
