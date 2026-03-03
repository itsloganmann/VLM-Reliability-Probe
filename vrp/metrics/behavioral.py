"""Behavioral reliability metrics (Stage 3 of VRP).

Implements Section 3.3.1 of the paper:
  - Self-Consistency (SC) score S_SC  (Eq. 5)

Self-Consistency samples K stochastic outputs for a single prompt using
non-zero temperature (τ = 0.7) and nucleus sampling (p = 0.9), then
computes the normalised frequency of the majority-vote answer.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, List, Tuple

from PIL import Image


# Paper §3.3.1 / §4 hyperparameters
_DEFAULT_K: int = 10
_DEFAULT_TEMPERATURE: float = 0.7
_DEFAULT_TOP_P: float = 0.9


def self_consistency(
    image: Image.Image,
    question: str,
    generate_fn: Callable[..., Tuple[str, object]],
    k: int = _DEFAULT_K,
    temperature: float = _DEFAULT_TEMPERATURE,
    top_p: float = _DEFAULT_TOP_P,
) -> Tuple[str, float, List[str]]:
    """Compute the Self-Consistency score S_SC (Eq. 5).

    Parameters
    ----------
    image : PIL.Image
        Input image.
    question : str
        VQA question string.
    generate_fn : callable
        A function with signature
        ``generate_fn(image, question, do_sample, temperature, top_p) → (answer, logits)``.
        Typically the ``generate`` method of a VRP model wrapper.
    k : int
        Number of stochastic samples (paper uses K = 10).
    temperature : float
        Sampling temperature (paper uses τ = 0.7).
    top_p : float
        Nucleus sampling probability (paper uses p = 0.9).

    Returns
    -------
    majority_answer : str
        The most frequent answer across K samples.
    s_sc : float
        Normalised frequency of the majority vote, in [1/K, 1].
    all_answers : list of str
        All K sampled answers.
    """
    answers: List[str] = []
    for _ in range(k):
        answer, _ = generate_fn(
            image,
            question,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        answers.append(_normalise_answer(answer))

    counts = Counter(answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    s_sc = majority_count / k

    return majority_answer, s_sc, answers


# ------------------------------------------------------------------ #
#  Answer normalisation                                                #
# ------------------------------------------------------------------ #

def _normalise_answer(answer: str) -> str:
    """Lowercase and strip punctuation for robust majority-vote matching."""
    import re
    answer = answer.lower().strip()
    answer = re.sub(r"[^a-z0-9\s]", "", answer)
    return " ".join(answer.split())
