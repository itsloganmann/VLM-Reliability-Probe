# VLM Reliability Probe (VRP)

> **Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models**

**Papers**

- Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models. ICLR 2026 Workshop on Multimodal Intelligence. [OpenReview](https://openreview.net/forum?id=fyDhQ2dKdw)
- Where Reliability Lives in Vision-Language Models: A Mechanistic Study of Attention, Hidden States, and Causal Circuits. ICML 2026 Mechanistic Interpretability Workshop. [arXiv](https://arxiv.org/abs/2605.08200)

Author: Logan Mann, UC Santa Barbara. Contact: github.com/itsloganmann


---

## Overview

**VLM Reliability Probe (VRP)** is a systematic diagnostic pipeline for evaluating the *reliability signals* of Vision-Language Models (VLMs).  It instruments three computational stages of any VLM forward pass:

| Stage | What we extract | Metrics |
|---|---|---|
| **Stage 1 – Visual Encoder** | Cross-attention maps `A(l,h) ∈ R^{T×S}` | Spatial entropy `H_s`, cluster count `C_k`, attention evolution `ΔH_s` |
| **Stage 2 – LLM Backbone** | Hidden states `h^(ℓ) ∈ R^d` | Truth margin `ΔM_l` (logit lens), token confidence `P_tok`, sparse MLP probe |
| **Stage 3 – Generation** | Output set `Y = {y_1,…,y_K}` | Self-consistency score `S_SC` |

### Core Findings

* **Visuals Lie**: Spatial attention metrics (entropy, clustering) have near-zero correlation with answer correctness across all three VLM families (R ≈ 0.001, R² < 0.08).
* **Consistency Speaks**: Self-consistency (R = 0.43) and hidden-state probes (AUROC > 0.95) are the dominant reliability signals.
* **Symbolic Detachment**: LLaVA-1.5 exhibits "Early Locking" of visual features followed by late diffusion, decoupling visual grounding from the final decision.

---

## Supported Models

| Model | HuggingFace ID | Layers | Visual Encoder |
|---|---|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | 32 | CLIP ViT-L/14 |
| PaliGemma-3B | `google/paligemma-3b-pt-224` | 18 | SigLIP |
| Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` | 28 | Native multimodal |

---

## Repository Structure

```
VLM-Reliability-Probe/
├── vrp/
│   ├── models/          # Per-family model wrappers with attention hooks
│   ├── metrics/
│   │   ├── structural.py    # C_k, H_s, ΔH_s
│   │   ├── mechanistic.py   # Logit lens, token confidence, hidden-state probe
│   │   └── behavioral.py    # Self-consistency
│   ├── probes/
│   │   └── mlp_probe.py     # 2-layer MLP reliability classifier
│   ├── evaluation/
│   │   └── auroc.py         # AUROC, point-biserial correlation, metric fusion
│   └── pipeline.py          # End-to-end VRP pipeline
├── scripts/
│   ├── run_vrp.py           # Single-command evaluation
│   └── train_probe.py       # Probe training workflow
├── configs/                 # Per-model YAML hyperparameters
└── data/
    └── README.md            # Dataset download instructions
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd VLM-Reliability-Probe

# 2. Create a conda environment (recommended)
conda create -n vrp python=3.10
conda activate vrp

# 3. Install dependencies
pip install -r requirements.txt
```

> **Hardware**: All experiments in the paper were conducted on NVIDIA A100 (80 GB VRAM).  At minimum, a 24 GB GPU is needed to run LLaVA-1.5-7B or Qwen2-VL-7B in 16-bit precision.

---

## Quick Start

### Run the full VRP diagnostic on a single model

```bash
python scripts/run_vrp.py \
    --model llava \
    --dataset pope \
    --split adversarial \
    --num_samples 1000 \
    --output_dir results/llava_pope
```

### Train a hidden-state probe

```bash
python scripts/train_probe.py \
    --model llava \
    --dataset pope \
    --split adversarial \
    --probe_layer 21 \
    --output_dir checkpoints/llava_probe
```

### Evaluate a saved probe

```bash
python scripts/run_vrp.py \
    --model llava \
    --dataset pope \
    --split adversarial \
    --probe_checkpoint checkpoints/llava_probe/best_probe.pt \
    --output_dir results/llava_pope_with_probe
```

---

## Datasets

See [`data/README.md`](data/README.md) for download instructions.

| Dataset | Task | Samples |
|---|---|---|
| POPE (Adversarial) | Object hallucination | 1 000 |
| LLaVA-Bench | Open-ended reasoning | Full benchmark |
| Custom Counting | Quantitative reasoning | 1 000 |
| Custom Spatial | Spatial-relation VQA | 1 000 |

---

## Metrics

### Structural (Stage 1)

| Metric | Symbol | Description |
|---|---|---|
| Spatial Entropy | `H_s` | Shannon entropy of aggregated attention map |
| Cluster Count | `C_k` | DBSCAN clusters on top-10% activated patches |
| Attention Evolution | `ΔH_s` | Per-layer change in spatial entropy |

### Mechanistic (Stage 2)

| Metric | Symbol | Description |
|---|---|---|
| Truth Margin | `ΔM_l` | Logit difference (correct token − top incorrect) at layer *l* |
| Token Confidence | `P_tok` | Mean log-probability of generated answer tokens |
| Hidden-State Probe | - | 2-layer MLP trained on `h_last` to predict binary correctness |

### Behavioral (Stage 3)

| Metric | Symbol | Description |
|---|---|---|
| Self-Consistency | `S_SC` | Normalized majority-vote frequency over K = 10 samples |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Self-consistency samples K | 10 |
| Sampling temperature τ | 0.7 |
| Nucleus sampling p | 0.9 |
| DBSCAN ε | 1.5 |
| DBSCAN min_samples | 3 |
| Attention top percentile | 10 % |
| Probe train/test split | 80 / 20 (stratified) |
| Probe optimizer | Adam, lr = 1e-4 |
| Probe training epochs | 50 |

---

## Reproducing Paper Results

All prompts, dataset split definitions, and probe training pipelines are included in this repository.  To reproduce Table 1 from the paper:

```bash
for MODEL in llava paligemma qwen2vl; do
    python scripts/run_vrp.py \
        --model $MODEL \
        --dataset pope \
        --split adversarial \
        --num_samples 1000 \
        --output_dir results/${MODEL}_pope
done
```

---

## Citation

```bibtex
@inproceedings{mann2026visualslie,
  title     = {Visuals Lie, Consistency Speaks: Disentangling Spatial Attention
               from Reliability in Vision-Language Models},
  author    = {Mann, Logan},
  booktitle = {ICLR 2026 Workshop on Multimodal Intelligence},
  year      = {2026},
  url       = {https://openreview.net/forum?id=fyDhQ2dKdw}
}

@inproceedings{mann2026wherereliability,
  title     = {Where Reliability Lives in Vision-Language Models: A Mechanistic
               Study of Attention, Hidden States, and Causal Circuits},
  author    = {Mann, Logan},
  booktitle = {ICML 2026 Mechanistic Interpretability Workshop},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.08200}
}
```

---

## License

This project is released under the MIT License.
