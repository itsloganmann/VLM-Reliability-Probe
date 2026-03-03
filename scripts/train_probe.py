#!/usr/bin/env python3
"""Train the VRP hidden-state MLP probe on collected hidden states.

The probe is a 2-layer MLP that predicts binary answer correctness from the
final-token hidden state h_last at a chosen decoder layer (Section 3.3.2).

Example usage
-------------
# 1. First collect hidden states via run_vrp.py --no_sc (fast).
# 2. Train the probe on the collected data:
python scripts/train_probe.py \\
    --model llava \\
    --dataset pope \\
    --split adversarial \\
    --num_samples 1000 \\
    --probe_layer 21 \\
    --output_dir checkpoints/llava_probe

# The best probe will be saved to checkpoints/llava_probe/best_probe.pt
# together with training metrics in checkpoints/llava_probe/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a VRP hidden-state probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   choices=["llava", "paligemma", "qwen2vl"])
    p.add_argument("--dataset", required=True,
                   choices=["pope", "llava_bench", "counting", "spatial"])
    p.add_argument("--split", default="adversarial")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--probe_layer", type=int, default=21,
                   help="Index of the decoder layer whose hidden state to probe "
                        "(0-based, relative to hooked layers).")
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Width of the probe's hidden layer.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction of data reserved for validation.")
    p.add_argument("--output_dir", default="checkpoints/probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def load_model(model_name: str, device: str, dtype_str: str):
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[dtype_str]
    if model_name == "llava":
        from vrp.models.llava import LLaVAModel
        return LLaVAModel(device=device, dtype=dtype)
    if model_name == "paligemma":
        from vrp.models.paligemma import PaliGemmaModel
        return PaliGemmaModel(device=device, dtype=dtype)
    if model_name == "qwen2vl":
        from vrp.models.qwen2vl import Qwen2VLModel
        return Qwen2VLModel(device=device, dtype=dtype)
    raise ValueError(f"Unknown model: {model_name}")


def load_dataset(dataset: str, split: str, data_dir: str, num_samples=None):
    from vrp.data_utils import load_pope, load_counting_spatial
    if dataset == "pope":
        examples = load_pope(data_dir, split=split)
    elif dataset in ("counting", "spatial"):
        examples = load_counting_spatial(data_dir, task=dataset)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' loader not yet implemented.")
    if num_samples is not None:
        examples = examples[:num_samples]
    return examples


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_probe] Loading model: {args.model}")
    model = load_model(args.model, args.device, args.dtype)

    from vrp.pipeline import VRPPipeline
    pipeline = VRPPipeline(model=model, sc_k=1)  # no SC needed for probe training

    print(f"[train_probe] Loading dataset: {args.dataset} / {args.split}")
    examples = load_dataset(args.dataset, args.split, args.data_dir, args.num_samples)
    print(f"[train_probe] Collecting hidden states for {len(examples)} examples …")

    from tqdm import tqdm
    hidden_vecs = []
    labels = []

    for ex in tqdm(examples, desc="Collect"):
        result = pipeline.run(
            image=ex["image"],
            question=ex["question"],
            ground_truth=ex.get("ground_truth"),
            run_sc=False,
            run_probe=False,
        )
        if result.is_correct is None:
            continue
        if len(model.hidden_states) == 0:
            continue

        # Extract hidden state at the requested layer index
        layer_idx = min(args.probe_layer, len(model.hidden_states) - 1)
        h = model.hidden_states[layer_idx].numpy()
        hidden_vecs.append(h)
        labels.append(int(result.is_correct))

    X = np.stack(hidden_vecs, axis=0)
    y = np.array(labels, dtype=np.float32)
    print(f"[train_probe] Collected {len(y)} labelled examples "
          f"({int(y.sum())} correct, {int((1-y).sum())} incorrect).")

    from vrp.probes.mlp_probe import train_probe, save_probe
    print("[train_probe] Training MLP probe …")
    probe, metrics = train_probe(
        hidden_states=X,
        labels=y,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        test_size=args.test_size,
        device=args.device,
    )

    print(f"[train_probe] Validation accuracy : {metrics['val_accuracy']:.4f}")
    print(f"[train_probe] Validation AUROC     : {metrics['val_auroc']:.4f}")

    probe_path = out_dir / "best_probe.pt"
    save_probe(probe, probe_path)
    print(f"[train_probe] Probe saved to {probe_path}")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "probe_layer": args.probe_layer}, f, indent=2)
    print(f"[train_probe] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
