# Visual Reliance in Medical VLMs: Quality Sensitivity and Non-Visual RL Gains

A reproducibility package for a medical-VQA pilot comparing **Qwen2-VL-2B-Instruct** and **MedVLM-R1**. It addresses whether answers depend on the **current image**, how that dependence varies under image degradation, and why a substantial part of the RL-checkpoint performance gain persists without an image.

**Status (2026-10-09):** released checkpoint inference and exploratory analysis; **no RL model was trained by this project**. The main plots and accuracy contrasts below have been independently recomputed from frozen, non-image, per-item evaluation exports. The standalone GPU scorer mirrors the original scoring logic but **has not been rerun end-to-end** on newly downloaded models/datasets as part of this public packaging step; that distinction matters for reproducibility.

## Read the research findings

- **[One-page research brief](docs/ONE_PAGE_SUMMARY.md)** — measured / found / surprised; intended for research discussion.
- **[Supporting experimental report](docs/DETAILED_REPORT.md)** — all main comparisons, error bounds, scope limits, and sources.
- **[Figure: visual-content benefit by blur level](figures/quality_content_gap.png)** and **[Figure: RL uplift with/without an image](figures/post_rl_uplift.png)**.

## Main findings

| Evaluation | MedVLM-R1−Base Clean | MedVLM-R1−Base No-image |
|---|---:|---:|
| MRI 300 | +29.11 pp | +32.11 pp |
| Official four-option 825 | +12.08 pp | +13.82 pp |
| External SLAKE test 355 | −5.78 pp | −2.68 pp |
| External PMC-VQA 1,000 | +0.33 pp | −0.37 pp |

The two main rows use **matched per-option scoring**. The external rows use matching scoring and frozen test sets. Relative changes in visual benefit should be assessed with *paired comparisons*, not by subtracting unrelated accuracy protocols. DeepSeek did not select any main test set.

## Reproduce the **reported analysis and figures** (verified)

Requirements: Python 3.10+, `numpy`, `matplotlib` (no GPU or medical images needed).

```bash
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_results.py --check --plots
python -m unittest discover -s tests -v
```

The analysis script checks 9,900 candidate-scoring rows (825 questions × 2 models × 2 conditions × 3 option orders); 24 frozen quality-content comparison records (2 models × 4 blur levels × 3 mismatched donors); plus the complete 355-item SLAKE and 1,000-item PMC-VQA per-item checkpoint comparisons. The tables are **de-identified numeric scores with synthetic benchmark item IDs**, not patient images or original question texts.

Main data files:

- `results/official825_scores.csv`: per-order Qwen/MedVLM-R1 scores for Clean and No-image (exact numerical recomputation of MRI 300 and 825 totals).
- `results/quality_content_donor_contrasts.csv`: frozen quality × image-pairing GT **log-margin** contrasts across three donors.
- `results/slake_yesno_test_per_item.csv` and `results/pmcvqa_per_item.csv`: per-item, averaged-over-option-orders accuracy for the same Base/RL checkpoint pair.
- `results/clean_both_success_506.json`: aggregated *clean-correct* conditional analysis; the original run used different answer-output protocols and is not causal evidence of equivalent knowledge.
- `results/selected_subsets/*.txt`: **exploratory official benchmark IDs** from an earlier question-only DeepSeek screen. These are not clinician-verified labels. No model or image files are distributed.

**Difference between two margin metrics:** `GT probability margin = p(correct) − max p(wrong)` and `GT log margin = log p(correct) − max log p(wrong)`. Their numerical scales cannot be pooled or directly compared. The quality figure uses **log margin**. The frozen `official825_scores.csv` from the matched stage uses the **probability margin** in `margin_gt_minus_best_wrong`.

## Re-run **model inference** (requires separate assets; packaging smoke-tested only)

Upstream model repositories:

- Qwen2-VL-2B-Instruct: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- MedVLM-R1: https://huggingface.co/JZPeterPan/MedVLM-R1

Upstream datasets:

- MedVLM-R1 official benchmark/project: https://github.com/JZPeterPan/MedVLM-R1
- SLAKE: https://huggingface.co/datasets/BoKelvin/SLAKE
- PMC-VQA: https://huggingface.co/datasets/xmcmic/PMC-VQA
- VQA-RAD: https://huggingface.co/datasets/flaviagiammarino/vqa-rad

Download and extract the relevant dataset according to its license and documentation. Convert your local test items into the JSONL schema documented in `scripts/evaluate_external.py`; a *synthetic schema example* is in `configs/example_manifest.jsonl`. For the four-choice PMC-VQA subset, the historic release fixed a **stratified 1,000-item sample**. Reproducing its exact model-level benchmark numbers requires recovering the same item identities and the original frozen donor matching, not only resampling 1,000 items with the same seed. Contact the authors/retain the original private frozen manifest if exact new inference is required; the numeric per-item CSV supports full *analysis* reproduction in this archive.

```bash
python -m pip install -r requirements-inference.txt
python scripts/evaluate_external.py \
  --manifest /local/path/benchmark.jsonl \
  --donor-plan /local/path/shared_frozen_donors.json \
  --dry-run

python scripts/evaluate_external.py \
  --manifest /local/path/benchmark.jsonl \
  --donor-plan /local/path/shared_frozen_donors.json \
  --checkpoint /local/path/Qwen2-VL-2B-Instruct \
  --device cuda:0 --output /local/path/qwen_scores.jsonl

# Repeat with the same manifest and exact donor plan, changing --checkpoint to MedVLM-R1.
```

The inference script uses the image-and-question chat template, exact `A) ...` answer options, single-token A/B/C/D candidate scoring, softmax normalized over the displayed choices, semantic answer ID inversion, min/max image pixels `3136/401408`, three answer orders for four-choice or two for Yes/No, and a frozen seeded one-cycle image donor permutation. It does not download raw data or train a model; multimodal inference needs CUDA-capable hardware and `qwen-vl-utils`.

## Experimental limitations

- **Matched scoring and model-specific native generation are not interchangeable.** The 825/MRI RL uplift headline uses matched scoring; some secondary visual-perturbation plots were measured with the separate native-output pipeline.
- **No-image measures absence of visual input**, not pure prior knowledge. A substituted real image controls image presence better than removing all tokens, but donor images are not guaranteed to contradict the clinical answer.
- **Probe decodability does not imply causal use** of quality information. Artificial blur is not clinical diagnostic quality.
- **No-image RL** training is a proposal, not a completed control. The pilot does not distinguish new medical knowledge from better use of learned task priors.
- Selected DeepSeek subsets are supplemental and **were not used to filter the main test results**; they should not be treated as ground truth.

## Source works

- [MedVLM-R1, MICCAI 2025](https://papers.miccai.org/miccai-2025/0553-Paper3267.html)
- [VES-RFT, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html)
- [SurgCheck, Int. J. CARS 2026](https://link.springer.com/article/10.1007/s11548-026-03739-4)
- [Reference-free whole-heart MR IQA, Radiology: AI 2020](https://pubs.rsna.org/doi/full/10.1148/ryai.2020190123)
- [PMC-VQA dataset and model](https://github.com/xiaoman-zhang/PMC-VQA)

This code/results release includes no raw medical images, weights, private experiments, original full datasets, or API credentials. Upstream repositories and datasets retain their respective licenses. An explicit open-source license for this *new project code* has not yet been assigned by the project owner.

## Publication

If a new public GitHub repository cannot be created from your connected GitHub integration, this directory is prepared for publication. On a machine with GitHub CLI authenticated using `gh auth login`, run `bash scripts/publish_github.sh`; it attempts to create a **new public repository** named `medical-vlm-visual-reliance`. Do not substitute an unrelated existing repository. If you prefer to create the empty public repo in GitHub's web UI, initialize a local git repository here and push to that URL instead. **Until one of those actions succeeds, this package is not online and no GitHub link should be reported.**
