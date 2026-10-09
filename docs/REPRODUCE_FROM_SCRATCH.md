# Reproduce from Source Data and Published Checkpoints

This guide is for a new machine. It distinguishes **CPU data/analysis** from **GPU model inference** and describes the precise frozen evaluation instrument. Run all commands from the repository root. No local developers' paths are required.

## 0. Requirements and storage

- Linux recommended; Python 3.11; a CUDA GPU with enough memory for Qwen2-VL-2B-Instruct / MedVLM-R1 (the pilot used a 24 GB GPU), `git`, ~11 GB for both model checkpoints.
- Official dataset: `images.zip` ~2.5 GB, plus a 900-question JSON; SLAKE ~220 MB; VQA-RAD ~120 MB.
- **PMC-VQA is large**: the original figure archives are ~21 GB. The download step requires an explicit flag. Retrieving/checking archive members and extracting selected figures requires extra time and disk space.
- Install a supported PyTorch+CUDA build appropriate for your GPU before running the inference suite. This repo pins the Transformers version used for the original run (`4.53.3`), but your CUDA/PyTorch wheel selection may be platform-specific.

See also the [original environment versions](ENVIRONMENT_PROVENANCE.md).

## 1. Installation

```bash
git clone https://github.com/bbuller591-sketch/medical-vlm-visual-reliance.git
cd medical-vlm-visual-reliance

# Optional isolated environment
conda create -n med-vlm-repro python=3.11 -y
conda activate med-vlm-repro

# On a CPU-only machine, just the analysis and tests:
pip install -r requirements-analysis.txt

# On a machine with CUDA, install the appropriate PyTorch CUDA wheel first,
# then install the inference libraries and dataset dependencies:
pip install -r requirements-inference.txt
pip install -r requirements-download.txt
```

A successful lightweight check:

```bash
python scripts/reproduce_results.py --check --plots
python -m unittest discover -s tests -v
python scripts/check_release.py --verify
```

All of the above can run on **CPU** from the committed anonymized numeric outputs. They do not download any biomedical data or models.

## 2. Download models and data (GPU experiments only)

The download script uses the original official source accounts, with **all models and dataset source revisions pinned to the Hugging Face revisions recorded during the original experiments**. Review licenses and access terms before downloading. Large datasets and checkpoints are **not** committed to git.

```bash
# Both model checkpoints (about 8–10 GB total)
python scripts/download_assets.py --asset-root assets --models both

# MedVLM-R1 official 900 questions + the original parent evaluation image archive
# Official 900 JSON: fixed MedVLM-R1 GitHub commit + SHA256 verification.
# Image archive: pinned FreedomIntelligence/Medical_Multimodal_Evaluation_Data snapshot.
python scripts/download_assets.py --asset-root assets --datasets official

# SLAKE English closed yes/no; public test + validation and image archive
python scripts/download_assets.py --asset-root assets --datasets slake

# VQA-RAD: official Hugging Face test parquet (images embedded)
python scripts/download_assets.py --asset-root assets --datasets vqarad

# PMC-VQA: test_clean.csv + large figure archives; requires disk space
python scripts/download_assets.py --asset-root assets --datasets pmcvqa --accept-large-download
```

Model references: [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), [MedVLM-R1](https://huggingface.co/JZPeterPan/MedVLM-R1). Official test set: [MedVLM-R1 repository](https://github.com/JZPeterPan/MedVLM-R1) (the JSON is from commit `2eee4210b25b149dcfa68513eb4682ac63b029d2`, SHA256 verified). Its 900 **clean medical images are not contained in the demo GitHub repo**; the downloader obtains the proper [evaluation image archive](https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data) at pinned revision `7d5b870cf7ace833a187a3459d7180a60ce9c5a1`. The JSON and source image archive are independently checked before use.

## 3. Rebuild strict benchmark manifests

```bash
python scripts/prepare_datasets.py --asset-root assets --dataset official900
python scripts/prepare_datasets.py --asset-root assets --dataset official825
python scripts/prepare_datasets.py --asset-root assets --dataset slake_test
python scripts/prepare_datasets.py --asset-root assets --dataset slake_val
python scripts/prepare_datasets.py --asset-root assets --dataset vqarad
python scripts/prepare_datasets.py --asset-root assets --dataset pmcvqa
```

Manifest format: one JSON object per question, including stable item ID, original question, candidate answer strings, correct answer index, image ID, local image file path, and grouping metadata. Manifests and extracted medical images remain under ignored `assets/` and **must not be pushed to the public repository**.

Frozen expected checks:

| Manifest | Questions | Images | Original protocol |
|---|---:|---:|---|
| `official900.jsonl` | 900 | 900 | MRI 300, CT 300, X-ray 300; 2-choice and 4-choice |
| `official825.jsonl` | 825 | 825 | The same official set, **four options only** |
| `slake_test.jsonl` | 355 | 96 | English closed Yes/No, original `test.json` |
| `slake_val.jsonl` | 358 | 96 | English closed Yes/No, `validation.json` |
| `pmcvqa.jsonl` | 1,000 | 852 | Answer-stratified, seed `20260916`, from `test_clean.csv` |
| `vqarad.jsonl` | 251 | 135 | Original Yes/No subset of test parquet |

The dataset adapters were validated against the real original archives on 9 October 2026. For SLAKE and PMC-VQA, frozen item IDs, image IDs, and answer labels were checked to match the original experiment *exactly*. The converter also prevents same-basename collisions across different SLAKE studies.

## 4. Re-run GPU behavior: original vs No-image vs wrong image

The external scorer uses the same question and answer options in all conditions, normalizes logits over candidate A/B/C/D letters only, and maps option permutations back to **semantic** option identities. For each item, a fixed seeded derangement selects a different real image. **Use exactly the same `--donor-plan` file for both checkpoints**; it is created at the first invocation if absent. Do not regenerate the donor mapping per model.

```bash
# Quick test first — validates inputs and generates a frozen donor matching.
python scripts/evaluate_external.py \
  --manifest assets/manifests/slake_test.jsonl \
  --donor-plan outputs/slake_donors.json --dry-run

python scripts/evaluate_external.py \
  --manifest assets/manifests/slake_test.jsonl \
  --donor-plan outputs/slake_donors.json \
  --checkpoint assets/models/Qwen2-VL-2B-Instruct \
  --device cuda:0 --output outputs/slake_base.jsonl --resume

python scripts/evaluate_external.py \
  --manifest assets/manifests/slake_test.jsonl \
  --donor-plan outputs/slake_donors.json \
  --checkpoint assets/models/MedVLM-R1 \
  --device cuda:0 --output outputs/slake_rl.jsonl --resume
```

For PMC-VQA, change only the manifest, donor-plan filename, and output filenames; keep the same two checkpoint paths. For VQA-RAD and SLAKE validation, use their corresponding manifests. For a smoke test, add `--limit 2 --max-orders 1`. **Do not use smoke-test settings in reported results.**

The 900/825 matched candidate-scoring experiment can likewise use this scorer with the official manifest; the official stage's options may be a mix of two and four choices, so for exact 825 figures use `official825.jsonl`. The full `official900.jsonl` is intended for the quality suite below.

The image conditions are:

- `clean`: original paired image;
- `no_image`: no visual tokens at all;
- `shuffled_image`: different real image, fixed over both checkpoints, not necessarily medically contradictory.

## 5. Re-run GPU quality × relevance and confidence calibration diagnostic

The full factorial uses **the same original 900 questions** and the original frozen 3 × 900 same-modality donor assignments stored in `configs/official900_donors.json`. The semantic-conflict arm has 375 label-supported eligible items, **not** clinician-validated counterfactuals. Synthetic Gaussian blur is at `0.005 / 0.015 / 0.030 × min(image width, image height)`; both the correctly paired image and the donor image receive the **same** blur level.

```bash
# Minimal test of donor assignment and arm selection
python scripts/evaluate_quality.py \
  --manifest assets/manifests/official900.jsonl \
  --donors configs/official900_donors.json --dry-run --limit 2

# FULL study: many forward passes. GPU time can be substantial.
python scripts/evaluate_quality.py \
  --manifest assets/manifests/official900.jsonl \
  --donors configs/official900_donors.json \
  --checkpoint assets/models/Qwen2-VL-2B-Instruct \
  --device cuda:0 --output outputs/quality_base.jsonl

python scripts/evaluate_quality.py \
  --manifest assets/manifests/official900.jsonl \
  --donors configs/official900_donors.json \
  --checkpoint assets/models/MedVLM-R1 \
  --device cuda:0 --output outputs/quality_rl.jsonl
```

The quality scorer writes records after each successful forward pass and skips completed item/condition/order combinations when resumed. Use the same 900-image manifest, same donor map, same blur ladder, same processor pixel budget for both models.

## 6. Analyze new inference outputs without GPU

```bash
python scripts/analyze_inference.py --predictions outputs/quality_base.jsonl \
  --output outputs/quality_base_metrics.json \
  --per-item-output outputs/quality_base_per_item.jsonl

python scripts/analyze_inference.py --predictions outputs/quality_rl.jsonl \
  --output outputs/quality_rl_metrics.json

python scripts/analyze_inference.py --predictions outputs/slake_base.jsonl \
  --output outputs/slake_base_metrics.json
```

Metrics include: both mean-per-order and option-order-aggregated accuracy, correct-answer probability, **signed GT probability and log margin** (different units!), candidate-distribution entropy, current-image utility relative to No-image, correct vs donor benefit by blur level, and VES-inspired *sharpen-and-harm* rate. Confidence intervals are resampled over source image IDs to prevent treating many questions about one image as independent.

For an **exact recomputation of the historical published tables and two figures**, which used several frozen scoring protocols, use `python scripts/reproduce_results.py --check --plots` from the checked-in numeric tables. This is the authoritative way to recreate the figures shown in the research brief. A new model inference run is separate and must preserve the source protocol for exact agreement; different model libraries, data versions, or prompt handling can change numbers.

## 7. Representation probing (optional mechanistic extension)

```bash
# GPU: collect each layer's decision-position residual on paired clean/severe inputs
python scripts/probe_image_quality.py --mode extract \
  --manifest assets/manifests/official900.jsonl \
  --checkpoint assets/models/Qwen2-VL-2B-Instruct \
  --device cuda:0 --output outputs/base_clean_vs_severe_layers.npz

# CPU: leakage-safe GroupKFold over source item IDs, train-only normalization
python scripts/probe_image_quality.py --mode fit \
  --features outputs/base_clean_vs_severe_layers.npz \
  --output outputs/base_probe.json
```

Repeat for MedVLM-R1. The portable probing script reproduces the **type of analysis** (layer-wise clean/severe linear decodability), whereas the archived original code includes additional original encoder-side controls and mechanistic intervention scripts. See `experiments/original_sources/README.md`. Probing AUROC is **not** proof that the model uses the encoded quality feature when answering.

## 8. Reproducibility levels

| Level | Status | How to reproduce |
|---|---|---|
| CPU historical table/figure reproduction | **Tested** | `reproduce_results.py --check --plots` |
| Original-source dataset preparation | **Tested on real local archives** | `prepare_datasets.py` |
| Small-sample Qwen and MedVLM-R1 inference | **GPU-tested** | `evaluate_external.py` with `--limit 2` |
| Small-sample quality/mismatch/blur | **GPU-tested** | `evaluate_quality.py` with `--limit 2` |
| Hidden-state extraction | **GPU-tested** | `probe_image_quality.py --mode extract --limit 2` |
| Full 900×quality×donor×order fresh re-run | **Not re-executed from the new portable wrapper** | Run long GPU job above; historical numeric outputs already included |
| Exact original activation patching and visual-stream knockout | **Archived original source; not repackaged as one-command** | See archived scripts and setup dependencies |

The repository intentionally does not claim that a short smoke test proves *bitwise* equivalence across all original model/PyTorch environments. It provides source code, fixed datasets and donor selection, reproducible metrics, and a clear path to conducting full new inference.
