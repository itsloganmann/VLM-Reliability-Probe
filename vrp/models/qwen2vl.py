"""Qwen2-VL-7B-Instruct model wrapper for VRP.

Architecture: 28 transformer layers with Grouped Query Attention (28 heads,
4 KV heads). Native multimodal architecture with interleaved visual tokens
and dynamic resolution support.

Key characteristic (paper §5.5): Qwen2-VL exhibits "Cyclical Refinement"
(re-sharpening attention at layers 17 and 25), producing strong late-stage
probe performance (AUROC = 0.971).
"""

from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from .base import VRPModelBase


_HF_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

_HOOK_LAYER_START = 0


class Qwen2VLModel(VRPModelBase):
    """Wrapper for Qwen2-VL-7B-Instruct with attention and hidden-state hooks."""

    def _load_model(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            _HF_MODEL_ID, trust_remote_code=True
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            _HF_MODEL_ID,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def _register_hooks(self) -> None:
        decoder_layers = self.model.model.layers
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
                    # Qwen2-VL uses dynamic resolution; image tokens follow the
                    # <|vision_start|> special token – store the full row for the
                    # pipeline to slice by actual visual-token count.
                    last_tok = avg[-1, :]
                    self.attention_maps.append(last_tok.detach().cpu().float())
        return hook

    def _make_hidden_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            hidden = outputs[0][0, -1, :]
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
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
        return len(self.model.model.layers)

    def hidden_size(self) -> int:
        return self.model.config.hidden_size
