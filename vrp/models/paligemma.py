"""PaliGemma-3B model wrapper for VRP.

Architecture: 18 transformer layers, 8 attention heads per layer.
Visual encoder: SigLIP (projected via a linear layer).
Language backbone: Gemma.

Key characteristic (paper §5.2): PaliGemma integrates visual evidence earlier
(peak at L14) relative to LLaVA, leaving fewer late layers with strong
probe separability.
"""

from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

from .base import VRPModelBase


_HF_MODEL_ID = "google/paligemma-3b-pt-224"

# Hook all decoder layers (shallow model, only 18 layers)
_HOOK_LAYER_START = 0


class PaliGemmaModel(VRPModelBase):
    """Wrapper for PaliGemma-3B with attention and hidden-state hooks."""

    def _load_model(self) -> None:
        self.processor = AutoProcessor.from_pretrained(_HF_MODEL_ID)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            _HF_MODEL_ID,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

    def _register_hooks(self) -> None:
        decoder_layers = self.model.language_model.model.layers
        for layer_idx, layer in enumerate(decoder_layers):
            if layer_idx < _HOOK_LAYER_START:
                continue
            h_attn = layer.self_attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            h_hidden = layer.register_forward_hook(
                self._make_hidden_hook(layer_idx)
            )
            self._hooks.extend([h_attn, h_hidden])

    # ------------------------------------------------------------------ #
    #  Hook factories                                                       #
    # ------------------------------------------------------------------ #

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) > 1:
                attn_weights = outputs[1]
                if attn_weights is not None:
                    avg = attn_weights[0].mean(0)  # (T, T)
                    # PaliGemma uses 256 image tokens by default (16×16 grid)
                    n_img_tokens = self._get_num_image_tokens()
                    last_tok = avg[-1, 1 : 1 + n_img_tokens]
                    self.attention_maps.append(last_tok.detach().cpu().float())
        return hook

    def _make_hidden_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            hidden = outputs[0][0, -1, :]
            self.hidden_states.append(hidden.detach().cpu().float())
        return hook

    def _get_num_image_tokens(self) -> int:
        """Return the number of image tokens expected by this model."""
        # PaliGemma-3b-pt-224: 16×16 patches → 256 tokens
        return getattr(self.model.config, "num_image_tokens", 256)

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

        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            padding="longest",
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

        if output.scores:
            logits = torch.stack([s[0] for s in output.scores], dim=0).cpu().float()
        else:
            logits = torch.empty(0)

        return answer, logits

    def num_layers(self) -> int:
        return len(self.model.language_model.model.layers)

    def hidden_size(self) -> int:
        return self.model.config.text_config.hidden_size
