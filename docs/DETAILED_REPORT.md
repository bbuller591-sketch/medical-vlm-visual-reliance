# Visual Reliance, Image Quality, and Non-Visual Gains in Medical VLMs

*Supporting experimental report | 9 October 2026*

> This document supports the [one-page research brief](ONE_PAGE_SUMMARY.md). The emphasis is on what was measured, what the results support, and what they cannot establish. All experiments evaluate **released checkpoints**; we did **not** retrain an RL policy.

## 1. Experimental setup and measurements

**Checkpoints:** Qwen2-VL-2B-Instruct (Base) and MedVLM-R1 (RL). **Main evaluation:** official MedVLM-R1 medical-VQA 900 questions (MRI 300, CT 300, X-ray 300), including 825 four-choice questions used for the direct, matched 4-way comparison. **External checks:** VQA-RAD Yes/No (251 questions), SLAKE English Yes/No test (355, 96 images) and validation (358, 96 images), and PMC-VQA four-choice (frozen 1,000, 852 images). External results for SLAKE and PMC-VQA are from **this same Qwen2-VL ↔ MedVLM-R1 checkpoint pair**, not MedVLThinker.

**Image interventions:** Clean retains the image paired with the question; No-image removes the image; Blank supplies an empty visual input; Mismatch replaces the original with a different real image while retaining exactly the same question and options; Blur degrades the *original* image with successively stronger synthetic blur. Mismatch is **not** a guarantee of opposite clinical findings; semantic-conflict donors are separately indicated when used. We kept the questions and displayed-answer sets fixed, counterbalanced option position, and mapped scores back to the same semantic option IDs.

For option-normalized probabilities \(p(y)\), let \(y^*\) be the correct choice. We measure:

- **Accuracy:** \(\mathbb{1}[\arg\max_y p(y)=y^*]\).
- **Signed GT probability margin:** \(M_p=p(y^*)-\max_{y\ne y^*}p(y)\).
- **Signed GT log margin:** \(M_{\log}=\log p(y^*)-\max_{y\ne y^*}\log p(y)\).
- **Answer entropy:** \(H=-\sum_y p(y)\log p(y)\).

Positive GT margins mean the gold answer beats every incorrect choice; they can be negative when the prediction is wrong. **Never mix probability-margin and log-margin scales**, and do not interpret entropy of answer choices as clinical certainty. A plain top-1 minus top-2 gap is *not* the same as signed GT margin when the model is wrong.

## 2. Does the current image matter?

The original 825-item four-choice evaluation, using the earlier checkpoint-specific output paths, produced:

| Condition | MedVLM-R1, native answer generation | Qwen Base, direct candidate scoring |
|---|---:|---:|
| Clean | 81.5% | 64.7% |
| No-image | 55.7% | 43.4% |
| Different real image | 69.6% | 50.2% |
| Severe blur | 72.8% | 54.4% |

**Important protocol boundary:** The columns above were obtained by different answer-production paths. This table describes *within-checkpoint sensitivity only*; it is **not** the matched Base-to-RL uplift estimate in Section 4.

No-image removes visual tokens and changes the multimodal input structure. A stronger content-oriented contrast is \(M_{\log}(\mathrm{Clean})-M_{\log}(\mathrm{Mismatch})\), where both inputs carry real images. Donor replacement still cannot certify that every donor image contradicts the original diagnosis. A subset of 375 questions has label-supported semantic-conflict donors, with 109 in a stronger mutually-exclusive tier and 266 in a weaker different-finding tier; neither tier constitutes clinician-verified counterfactual images.

## 3. Does visual benefit change with image quality?

In a frozen 900-question matched-scoring quality × relevance experiment (both models using matched processing), we compare the gold-answer **log-margin** with the correct image to its log-margin with an unrelated real image at the same blur level:

| Quality | Base correct−mismatch | MedVLM-R1 correct−mismatch |
|---|---:|---:|
| Clean | 0.730 | 0.894 |
| Mild | 0.736 | 0.883 |
| Medium | 0.543 | 0.647 |
| Severe | 0.211 | 0.228 |

![Matched-image advantage decreases under blur](../figures/quality_content_gap.png)

The early Clean→Mild segment changes little; the benefit falls with stronger blur. This is a **quality-conditioned outcome difference**, not direct proof that the model actively estimates when medical evidence is trustworthy. The blur is a technical/perceptual manipulation, not a validated clinical diagnostic-quality rating.

In separate, original 825-item model-specific output analyses, signed **GT log margin** decreased with blur: Base 1.045 (Clean), 1.025 (Mild), 0.840 (Medium), 0.333 (Severe); MedVLM-R1 4.102, 3.982, 3.560, 2.184. These absolute numbers are within-model traces under different inference protocols, **not matched cross-model logit comparisons**.

Answer-choice entropy rose substantially in the matched mechanism run: Base 0.756 (Clean) → 0.972 (Severe); MedVLM-R1 0.220 → 0.362. A layer-wise probe distinguished severe blur from clean inputs with best AUROC approximately 1.000 for both checkpoints. This establishes **decodability**, not that quality information causally governed the final answer.

**What VES-style entropy sensitivity misses.** In the label-supported semantic-conflict arm (375 samples), an image sometimes made the answer distribution sharper (entropy fell relative to No-image), but simultaneously *reduced* the signed correct-answer log margin. This pattern occurred in approximately 30.7% of Base cases and 23.5% of MedVLM-R1 cases in that arm. Confidence sharpening alone therefore cannot certify beneficial visual evidence. This diagnostic is inspired by the visual-entropy question raised by [VES-RFT (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html); it is **not** a reproduction or disproof of the complete VES-RFT training objective, which also uses a correctness reward.

## 4. What improved after medical RL?

Unlike Section 2's legacy native/score comparison, the following results use **matched candidate scoring and three controlled answer-option orders for both checkpoints**. The accuracy shown is the mean of correctness across the orders.

| Evaluation | Base Clean | RL Clean | Base No-image | RL No-image | RL−Base Clean | RL−Base No-image |
|---|---:|---:|---:|---:|---:|---:|
| MRI 300 | 67.33% | 96.44% | 43.56% | 75.67% | **+29.11 pp** | **+32.11 pp** |
| Four-choice 825 | 68.89% | 80.97% | 44.12% | 57.94% | **+12.08 pp** | **+13.82 pp** |

![Matched post-training gains](../figures/post_rl_uplift.png)

The additional value of providing an image, \(\mathrm{Accuracy}_{\mathrm{Clean}}-\mathrm{Accuracy}_{\mathrm{NoImage}}\), changes after RL by **−3.00 pp** on MRI (95% CI approximately [−7.56, +1.67]) and **−1.74 pp** on all 825 four-choice questions (bootstrap CI approximately [−3.76, +0.40]). This is a *difference-in-differences of behaviors*, not a causal partition into visual skill versus knowledge.

The 25% equal-choice baseline is substantially below Base No-image accuracy of **44.12%** on the 825 four-choice items: there are predictable non-visual signals in the question/option/task input, potentially including general medical knowledge and task/answer-distribution regularities. The large No-image uplift suggests RL reinforces **non-visual predictive capability**, but does not prove whether it is learning new knowledge, exploiting existing knowledge better, or fitting task-specific priors.

**Previously successful image-conditioned questions.** We also isolated the original **506 four-choice items that both models answered correctly when the image was present** under the existing model-specific output protocols. On the 499 cases with both No-image results valid, **219** remained correct for both, **82** only for RL, **46** only for Base, **152** neither. This controls for success in the *image-conditioned task*, not for equal pre-existing text-only medical knowledge. It supports examining whether RL reinforced text-based answer tendencies in cases Base could already solve *with an image*. We do not say it proves dataset-prior memorization or excludes all knowledge acquisition.

## 5. Does the gain transfer across datasets?

The external benchmark comparisons below use a matched pair of Qwen2-VL-2B and MedVLM-R1 checkpoints, identical image substitutions, and matched option scoring. Image-clustered paired bootstrap 95% CIs are reported for the new external results.

| Dataset | Base Clean | RL Clean | RL−Base Clean | Base No-image | RL No-image |
|---|---:|---:|---:|---:|---:|
| SLAKE test (355) | 68.0% | 62.3% | **−5.8 pp** (CI [−9.1, −2.6]) | 45.4% | 42.7% |
| SLAKE validation (358) | 69.0% | 64.8% | **−4.2 pp** (CI [−7.2, −1.4]) | 44.1% | 43.2% |
| PMC-VQA (1,000) | 45.1% | 45.4% | **+0.3 pp** (CI [−0.6, +1.2]) | 31.4% | 31.1% |
| VQA-RAD Yes/No (251) | 61.8% | 63.5% | +1.8 pp (not conclusive) | 51.8% | 54.8% |

These experiments do not reproduce the large MRI uplift. Different question types, answer distributions, visual domains, and training exposure may contribute. They do **not** by themselves establish training-set leakage, dataset memorization, or a causal mechanism for the limited gains.

**Auxiliary controls, kept out of the main story.** We performed a question-and-options-only DeepSeek review of the frozen SLAKE test and PMC-VQA samples, and reversed answer order to check stability. It was not an image-necessity gold label and **did not remove any items** from the external primary analysis. For the official 825-item benchmark, earlier question-only answerability-screening outputs are included solely as labeled exploratory subsets; they are not substitutes for the unfiltered results. The selected ID lists and counts are in the reproducibility package.

## 6. What next?

This pilot establishes **behavioral effects of current image pairing and image degradation**, while showing that image-driven certainty can be unhelpful. The unexpected large **No-image RL benefit** motivates a controlled follow-up comparing visual RL with an otherwise matched **image-free RL** condition using the same base, questions, and reward. No image-free RL was trained in this project. A second, distinct research option is to test clinically verified answer-opposed image pairs and separate **adaptive distrust** from passive loss of visual information in unreadable images. Synthetic blur cannot stand in for expert-rated clinical image quality.

### Reproducibility and source papers

The [repository README](../README.md) separates analysis reproduction from GPU inference reproduction and lists dataset/checkpoint access, controlled scoring, and files. Studies motivating this pilot include [VES-RFT (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html), [SurgCheck (2026)](https://link.springer.com/article/10.1007/s11548-026-03739-4), [MedVLM-R1 (MICCAI 2025)](https://papers.miccai.org/miccai-2025/0553-Paper3267.html), and [reference-free whole-heart MR image quality assessment (2020)](https://pubs.rsna.org/doi/full/10.1148/ryai.2020190123).
