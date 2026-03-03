"""LLaVA-1.5-7B model wrapper for VRP.

Architecture: 32 transformer layers, 32 attention heads per layer.
Visual encoder: frozen CLIP ViT-L/14 (576 visual tokens, 24×24 grid).
Language backbone: Vicuna-7B.

Attention extraction follows Section 3.1 of the paper:
  - Hooks on MultiheadAttention modules of the Vicuna decoder (layers 16–32).
  - Head-averaged attention is stored as (S,) → reshaped to (24, 24).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from .base import VRPModelBase


_HF_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Layers where visual-linguistic integration is densest (paper Appendix A.7)
_HOOK_LAYER_START = 16


class LLaVAModel(VRPModelBase):
    """Wrapper for LLaVA-1.5-7B with cross-attention and hidden-state hooks."""

    def _load_model(self) -> None:
        self.processor = AutoProcessor.from_pretrained(_HF_MODEL_ID)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            _HF_MODEL_ID,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

    def _register_hooks(self) -> None:
        """Register hooks on each decoder layer's self-attention module."""
        decoder_layers = self.model.language_model.model.layers
        for layer_idx, layer in enumerate(decoder_layers):
            if layer_idx < _HOOK_LAYER_START:
                continue
            # Attention hook – captures output attention weights
            h_attn = layer.self_attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            # Hidden-state hook – captures residual stream after layer
            h_hidden = layer.register_forward_hook(
                self._make_hidden_hook(layer_idx)
            )
            self._hooks.extend([h_attn, h_hidden])

    # ------------------------------------------------------------------ #
    #  Hook factories                                                       #
    # ------------------------------------------------------------------ #

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            # outputs is (attn_output, attn_weights, ...)
            # attn_weights shape: (batch, heads, seq, seq) – may be None if
            # output_attentions=False; we use scores from the raw softmax.
            if isinstance(outputs, tuple) and len(outputs) > 1:
                attn_weights = outputs[1]  # (1, H, T, T)
                if attn_weights is not None:
                    # Average over heads, take last-token row, keep visual cols
                    # Visual tokens start at index 1 (after BOS) in LLaVA
                    avg = attn_weights[0].mean(0)  # (T, T)
                    last_tok = avg[-1, 1:577]       # 576 visual tokens
                    self.attention_maps.append(last_tok.detach().cpu().float())
        return hook

    def _make_hidden_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            # outputs[0] is the hidden state tensor (1, T, d)
            hidden = outputs[0][0, -1, :]  # last token (1, d) → (d,)
            self.hidden_states.append(hidden.detach().cpu().float())
        return hook

    # ------------------------------------------------------------------ #
    #  Generation interface                                                 #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
    ) -> Tuple[str, torch.Tensor]:
        self.clear_cache()

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, self.dtype)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                output_scores=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )

        generated_ids = output.sequences[0, inputs["input_ids"].shape[-1]:]
        answer = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Stack per-token scores to (seq_len, vocab_size)
        if output.scores:
            logits = torch.stack([s[0] for s in output.scores], dim=0).cpu().float()
        else:
            logits = torch.empty(0)

        return answer, logits

    def num_layers(self) -> int:
        return len(self.model.language_model.model.layers)

    def hidden_size(self) -> int:
        return self.model.config.text_config.hidden_size
