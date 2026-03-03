"""Dataset loading utilities for VRP.

Supported datasets:
  - POPE (adversarial / random / popular splits)
  - Custom Counting & Spatial tasks

See ``data/README.md`` for download instructions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


# ------------------------------------------------------------------ #
#  POPE                                                               #
# ------------------------------------------------------------------ #

def load_pope(
    data_dir: str,
    split: str = "adversarial",
    image_subdir: str = "val2014",
) -> List[Dict[str, Any]]:
    """Load POPE evaluation examples.

    Expected directory layout (after following data/README.md)::

        data/pope/
            coco_pope_{split}.json
            images/val2014/
                COCO_val2014_<id>.jpg
                …

    Parameters
    ----------
    data_dir :
        Root data directory (e.g. ``"data"``).
    split :
        One of ``"adversarial"``, ``"popular"``, ``"random"``.
    image_subdir :
        Sub-directory containing COCO val2014 images.

    Returns
    -------
    examples : list of dict
        Each dict has keys ``image``, ``question``, ``ground_truth``.
    """
    pope_dir = Path(data_dir) / "pope"
    ann_path = pope_dir / f"coco_pope_{split}.json"
    img_dir = pope_dir / "images" / image_subdir

    if not ann_path.exists():
        raise FileNotFoundError(
            f"POPE annotation file not found: {ann_path}\n"
            "Run: bash data/download_pope.sh  (see data/README.md)"
        )

    with open(ann_path) as f:
        annotations = json.load(f)

    examples: List[Dict[str, Any]] = []
    for ann in annotations:
        img_file = img_dir / ann["image"]
        if not img_file.exists():
            continue
        image = Image.open(img_file).convert("RGB")
        question = ann["text"]
        # POPE ground-truth is "yes" / "no"
        ground_truth = ann["label"].lower().strip()
        examples.append({"image": image, "question": question, "ground_truth": ground_truth})

    return examples


# ------------------------------------------------------------------ #
#  Custom Counting & Spatial tasks                                     #
# ------------------------------------------------------------------ #

def load_counting_spatial(
    data_dir: str,
    task: str = "counting",
) -> List[Dict[str, Any]]:
    """Load custom counting or spatial-relation evaluation examples.

    Expected directory layout::

        data/custom/
            counting.json
            spatial.json
            images/
                <image_id>.jpg
                …

    Each JSON entry has the schema::

        {
            "image_id": "...",
            "question": "...",
            "answer": "..."    // ground truth (str)
        }

    Parameters
    ----------
    data_dir :
        Root data directory.
    task :
        ``"counting"`` or ``"spatial"``.

    Returns
    -------
    examples : list of dict
    """
    custom_dir = Path(data_dir) / "custom"
    ann_path = custom_dir / f"{task}.json"
    img_dir = custom_dir / "images"

    if not ann_path.exists():
        raise FileNotFoundError(
            f"Custom task annotation file not found: {ann_path}\n"
            "See data/README.md for dataset construction details."
        )

    with open(ann_path) as f:
        annotations = json.load(f)

    examples: List[Dict[str, Any]] = []
    for ann in annotations:
        img_file = img_dir / (str(ann["image_id"]) + ".jpg")
        if not img_file.exists():
            continue
        image = Image.open(img_file).convert("RGB")
        examples.append({
            "image": image,
            "question": ann["question"],
            "ground_truth": str(ann["answer"]).lower().strip(),
        })

    return examples
