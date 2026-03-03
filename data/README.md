# Datasets for VLM Reliability Probe (VRP)

This directory should contain the datasets used in the paper.  Below are
download instructions for each dataset.

---

## 1. POPE (Polling-based Object Probing Evaluation)

**Paper reference**: Li et al., *Evaluating Object Hallucination in Large
Vision-Language Models*, EMNLP 2023.

### Download

```bash
# 1. Clone the POPE repository
git clone https://github.com/AoiDragon/POPE.git /tmp/POPE

# 2. Copy annotation files (adversarial, popular, random splits)
mkdir -p data/pope
cp /tmp/POPE/output/coco/coco_pope_adversarial.json data/pope/
cp /tmp/POPE/output/coco/coco_pope_popular.json      data/pope/
cp /tmp/POPE/output/coco/coco_pope_random.json       data/pope/

# 3. Download COCO val2014 images
mkdir -p data/pope/images
wget http://images.cocodataset.org/zips/val2014.zip -P /tmp/
unzip /tmp/val2014.zip -d data/pope/images/
```

### Expected layout

```
data/pope/
    coco_pope_adversarial.json
    coco_pope_popular.json
    coco_pope_random.json
    images/
        val2014/
            COCO_val2014_000000000042.jpg
            …
```

---

## 2. LLaVA-Bench

**Paper reference**: Zhou et al., *LLaVA-Bench: A Benchmark for Visual
Instruction Following*, arXiv 2023.

```bash
# Download via HuggingFace Datasets
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("liuhaotian/llava-bench-in-the-wild")
ds.save_to_disk("data/llava_bench")
EOF
```

---

## 3. Custom Counting & Spatial Tasks

The custom counting and spatial-relation tasks are constructed from
COCO-style images with manually verified integer / object-relation answers.

### Schema

Each JSON file (`data/custom/counting.json`, `data/custom/spatial.json`)
contains a list of entries with the following fields:

```json
{
    "image_id": "COCO_val2014_000000000042",
    "question": "How many people are in the image?",
    "answer": "3"
}
```

Images are stored in `data/custom/images/` as `<image_id>.jpg`.

### Constructing the custom set

The custom set uses COCO val2014 images.  Sample construction script:

```bash
python - <<'EOF'
# Requires COCO annotations already downloaded to data/pope/images/val2014/
import json, random, pathlib

random.seed(42)
ann_path = "path/to/coco/annotations/instances_val2014.json"
with open(ann_path) as f:
    coco = json.load(f)

# ... filtering logic for counting / spatial questions ...
# Refer to Appendix A.4 of the paper for full construction details.
EOF
```

> **Note**: Pre-constructed split files will be released upon de-anonymisation.

---

## Directory Summary

```
data/
    pope/
        coco_pope_adversarial.json
        coco_pope_popular.json
        coco_pope_random.json
        images/val2014/
    llava_bench/          (HuggingFace Datasets format)
    custom/
        counting.json
        spatial.json
        images/
```
