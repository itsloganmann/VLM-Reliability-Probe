"""Abstract base class for VRP model wrappers.

Each concrete wrapper must:
  1. Load its model and processor from HuggingFace.
  2. Register forward hooks that populate ``self.attention_maps`` and
     ``self.hidden_states`` during inference.
  3. Implement ``generate()`` so the pipeline can obtain token logits and
     sampled outputs from a unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


class VRPModelBase(ABC):
    """Abstract base for all VRP model wrappers."""

    # Populated by forward hooks during each forward pass
    attention_maps: List[torch.Tensor]   # list of (H, S) per layer, head-averaged
    hidden_states: List[torch.Tensor]    # list of (d,) final-token hidden states per layer

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        self.device = device
        self.dtype = dtype
        self.attention_maps = []
        self.hidden_states = []
        self._hooks: List = []
        self._load_model()
        self._register_hooks()

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _load_model(self) -> None:
        """Load model + processor/tokenizer from HuggingFace."""

    @abstractmethod
    def _register_hooks(self) -> None:
        """Register PyTorch forward hooks that fill attention_maps / hidden_states."""

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
    ) -> Tuple[str, torch.Tensor]:
        """Run a single forward pass.

        Returns
        -------
        answer : str
            Decoded answer string.
        logits : torch.Tensor
            Token log-probabilities of shape ``(seq_len, vocab_size)``.
        """

    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer decoder layers."""

    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden-state dimension *d*."""

    # ------------------------------------------------------------------ #
    #  Hook helpers                                                         #
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Reset stored attention maps and hidden states between samples."""
        self.attention_maps = []
        self.hidden_states = []

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
