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

## Full model/dataset download and GPU reproduction

The repository now includes the **full portable behavioral analysis pipeline**, rather than only CPU recomputation. It follows the original checkpoint and dataset setup used by the published [MedVLM-R1 research repository](https://github.com/JZPeterPan/MedVLM-R1): source installation, separately downloaded model weights, original dataset download, explicit evaluation scripts, frozen experiment configs, and post-processing.

**[Start-to-finish reproduction guide with copyable commands →](docs/REPRODUCE_FROM_SCRATCH.md)**  · [Original environment versions](docs/ENVIRONMENT_PROVENANCE.md)

Quick index:

1. `scripts/download_assets.py`: Hugging Face checkpoints and original dataset archives, including verified official 900-question JSON and source image archive.
2. `scripts/prepare_datasets.py`: strict conversion to reproducible local JSONL manifests (official900/825, SLAKE test/val, PMC-VQA, VQA-RAD).
3. `scripts/evaluate_external.py`: original, no-image, and shuffled-image **GPU** inference with frozen option permutations and image donors.
4. `scripts/evaluate_quality.py`: **GPU** 900-question blur × three donors × semantic-conflict experiment using `configs/official900_donors.json`.
5. `scripts/analyze_inference.py`: **CPU** accuracy, GT probability/log margin, entropy, paired source-cluster bootstrap CIs, image-content utility and VES-style confidence/harmfulness analysis.
6. `scripts/probe_image_quality.py`: **GPU** layer-wise residual extraction and **CPU** group-disjoint linear probing.
7. `experiments/original_sources/`: original research experiment sources for extended probing, activation patching, knockout and quality interventions (archived; additional original staging requirements).

All original images, full questions, model files, and large feature arrays are ignored by Git; only safe numeric, item-ID-level experiment records and code are published. PMC-VQA image archives exceed 20 GB and download only when explicitly requested.

**Verification boundary:** The standalone data adapters, two checkpoints' GPU inference on a smoke subset, main quality perturbation scorer on a smoke subset, and representation extraction have been tested. Full-scale GPU re-inference of every arm has *not* been rerun using the new portable wrapper and is not claimed as bitwise reproduced. The exact historical published figures **are** reproduced from frozen numeric CSV exports by `scripts/reproduce_results.py`.

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

This code/results release includes no raw medical images, weights, private experiments, original full datasets, or API credentials. Upstream repositories and datasets retain their respective licenses. An explicit open-source license for this *new project code* has not yet been assigned by the project owner. Please request a license decision before redistributing modified code.

## Repository status and authorship

This is the **public** GitHub repository: https://github.com/bbuller591-sketch/medical-vlm-visual-reliance . It contains the full stand-alone behavioral workflow and an archive of the original mechanistic scripts, with separate reproducibility statuses documented above. No API keys or raw patient images are included. The project owner has not selected a license for newly authored research code; all third-party model/dataset sources retain their own terms.
