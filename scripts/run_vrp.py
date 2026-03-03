#!/usr/bin/env python3
"""Run the full VRP diagnostic pipeline on a given model and dataset.

Example usage
-------------
python scripts/run_vrp.py \\
    --model llava \\
    --dataset pope \\
    --split adversarial \\
    --num_samples 1000 \\
    --output_dir results/llava_pope

python scripts/run_vrp.py \\
    --model paligemma \\
    --dataset pope \\
    --num_samples 500 \\
    --no_sc \\
    --output_dir results/paligemma_pope_nsc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from vrp.evaluation.auroc import summarise_metrics, MetricFusion
import numpy as np


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM Reliability Probe – full evaluation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   choices=["llava", "paligemma", "qwen2vl"],
                   help="VLM architecture to evaluate.")
    p.add_argument("--dataset", required=True,
                   choices=["pope", "llava_bench", "counting", "spatial"],
                   help="Evaluation dataset.")
    p.add_argument("--split", default="adversarial",
                   help="Dataset split (e.g. 'adversarial' for POPE).")
    p.add_argument("--data_dir", default="data",
                   help="Root directory containing dataset files.")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Cap the number of samples (None = all).")
    p.add_argument("--output_dir", default="results",
                   help="Directory to write JSON results and metrics.")
    p.add_argument("--probe_checkpoint", default=None,
                   help="Path to a .pt probe checkpoint. If provided, adds probe score.")
    p.add_argument("--probe_layer", type=int, default=-1,
                   help="Index into hooked layers list for probe inference.")
    p.add_argument("--no_sc", action="store_true",
                   help="Skip self-consistency sampling (10× inference cost).")
    p.add_argument("--sc_k", type=int, default=10,
                   help="Number of self-consistency samples.")
    p.add_argument("--sc_temperature", type=float, default=0.7,
                   help="Sampling temperature for self-consistency.")
    p.add_argument("--sc_top_p", type=float, default=0.9,
                   help="Nucleus sampling p for self-consistency.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Model loading                                                       #
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


# ------------------------------------------------------------------ #
#  Dataset loading                                                     #
# ------------------------------------------------------------------ #

def load_dataset(dataset: str, split: str, data_dir: str, num_samples=None):
    """Load (image, question, ground_truth) triples.

    Returns a list of dicts with keys: image, question, ground_truth.
    Raises FileNotFoundError if the dataset is not yet downloaded.
    """
    from vrp.data_utils import load_pope, load_counting_spatial

    if dataset == "pope":
        examples = load_pope(data_dir, split=split)
    elif dataset in ("counting", "spatial"):
        examples = load_counting_spatial(data_dir, task=dataset)
    else:
        raise NotImplementedError(
            f"Dataset '{dataset}' loader not yet implemented. "
            "See data/README.md for download instructions."
        )

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

    print(f"[VRP] Loading model: {args.model}  ({args.device}, {args.dtype})")
    model = load_model(args.model, args.device, args.dtype)

    probe = None
    if args.probe_checkpoint:
        from vrp.probes.mlp_probe import load_probe
        probe = load_probe(args.probe_checkpoint, input_dim=model.hidden_size())
        print(f"[VRP] Loaded probe from {args.probe_checkpoint}")

    from vrp.pipeline import VRPPipeline
    pipeline = VRPPipeline(
        model=model,
        probe=probe,
        probe_layer_idx=args.probe_layer,
        sc_k=args.sc_k,
        sc_temperature=args.sc_temperature,
        sc_top_p=args.sc_top_p,
    )

    print(f"[VRP] Loading dataset: {args.dataset} / {args.split}")
    examples = load_dataset(args.dataset, args.split, args.data_dir, args.num_samples)
    print(f"[VRP] Running on {len(examples)} examples …")

    all_results = []
    from tqdm import tqdm
    for ex in tqdm(examples, desc="VRP"):
        res = pipeline.run(
            image=ex["image"],
            question=ex["question"],
            ground_truth=ex.get("ground_truth"),
            run_sc=not args.no_sc,
            run_probe=(probe is not None),
        )
        all_results.append(res)

    # ---- Collect metric arrays ----
    labels = [int(r.is_correct) for r in all_results if r.is_correct is not None]
    idx = [i for i, r in enumerate(all_results) if r.is_correct is not None]

    metric_scores = {
        "spatial_entropy": [-all_results[i].spatial_entropy for i in idx],
        "cluster_count": [float(all_results[i].cluster_count) for i in idx],
        "token_confidence": [all_results[i].token_confidence for i in idx],
    }
    if not args.no_sc:
        metric_scores["self_consistency"] = [all_results[i].self_consistency_score for i in idx]
    if probe is not None:
        metric_scores["probe_score"] = [all_results[i].probe_score for i in idx]

    summary = summarise_metrics(metric_scores, labels)

    # ---- Metric fusion ----
    feature_names = list(metric_scores.keys())
    feature_matrix = np.column_stack([metric_scores[k] for k in feature_names])
    split_n = int(0.8 * len(labels))
    fusion = MetricFusion()
    fusion.fit(feature_matrix[:split_n], np.array(labels[:split_n]))
    combined_auroc = fusion.auroc(feature_matrix[split_n:], np.array(labels[split_n:]))
    summary["combined"] = {"auroc": combined_auroc}

    # ---- Print summary ----
    print("\n=== VRP Results ===")
    for metric_name, vals in summary.items():
        print(f"  {metric_name:30s}  AUROC={vals['auroc']:.3f}", end="")
        if "r_pb" in vals:
            print(f"  R_pb={vals['r_pb']:+.3f}  p={vals['p_value']:.3e}", end="")
        print()

    # ---- Persist ----
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            [
                {
                    "question": r.question,
                    "predicted": r.predicted_answer,
                    "ground_truth": r.ground_truth,
                    "is_correct": r.is_correct,
                    "spatial_entropy": r.spatial_entropy,
                    "cluster_count": r.cluster_count,
                    "token_confidence": r.token_confidence,
                    "probe_score": r.probe_score,
                    "self_consistency_score": r.self_consistency_score,
                    "sc_majority_answer": r.sc_majority_answer,
                }
                for r in all_results
            ],
            f,
            indent=2,
        )

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[VRP] Results written to {results_path}")
    print(f"[VRP] Metrics  written to {metrics_path}")


if __name__ == "__main__":
    main()
