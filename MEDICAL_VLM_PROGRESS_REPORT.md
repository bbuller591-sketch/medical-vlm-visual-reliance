# Visual Reliance in Medical VLMs: Image Quality and Post-Training Behavior

*Research progress | 9 October 2026*
[Code, data protocols, and reproducible results](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance)

## Research question and setup

When a medical VLM answers correctly, what role does the **current image** play, and how does that role change with image quality? We compared **Qwen2-VL-2B-Instruct** (Base) and **MedVLM-R1** (RL checkpoint) on the official 900-question medical VQA benchmark (MRI, CT, and X-ray), using its 825 four-choice questions for direct Base–RL comparison. We tested **Clean**, **No-image**, a different real image assigned to the same question (**Mismatch**), and increasing synthetic **Blur**, measuring accuracy, signed ground-truth (GT) log margin, and answer entropy.

## What we found: visual benefit changes with image quality

Both checkpoints benefited from correctly matched visual evidence. Across 900 questions, the mean **GT log-margin advantage of the correct image over mismatched images** decreased with stronger blur:

| Checkpoint | Clean | Mild blur | Medium blur | Severe blur |
|---|---:|---:|---:|---:|
| Qwen2-VL Base | 0.730 | 0.736 | 0.543 | 0.211 |
| MedVLM-R1 | 0.894 | 0.883 | 0.647 | 0.228 |

Stronger blur also reduced GT margin and increased answer entropy; layer-wise probes decoded severe degradation (best AUROC ≈ 1.00). On 375 label-supported semantic-conflict cases, image input **reduced entropy while lowering the gold-answer margin** in 30.7% of Base cases and 23.5% of RL cases. Image-induced confidence and useful visual evidence can therefore diverge.

## What surprised us: RL gains persist without the image

RL accuracy gains persisted with the image removed. Using **matched candidate scoring and option-order controls**:

| Evaluation | RL − Base, Clean | RL − Base, No-image |
|---|---:|---:|
| MRI, 300 questions | +29.1 pp | **+32.1 pp** |
| All four-choice, 825 questions | +12.1 pp | **+13.8 pp** |

The change in the extra image-presence benefit was −3.0 pp on MRI and −1.7 pp on the 825-question set (both confidence intervals spanning zero). Base already achieved **44.1% without images** on 825 four-choice questions versus 25% uniform chance. Among 506 questions both models solved with Clean images, RL retained more correct answers after image removal: **301 versus 265** among 499 paired valid cases.

These patterns suggest that RL may reinforce **existing medical capabilities and task-specific question–answer priors** alongside visual grounding. External Clean gains were −5.8 pp on SLAKE test, −4.2 pp on SLAKE validation, and +0.3 pp on PMC-VQA, illustrating task-dependent transfer.

## Discussion and next steps

We plan to examine whether reduced visual reliance reflects **adaptive trust** or loss of readable information using clinically grounded quality and conflict controls, and to compare visual RL against **matched image-free RL training** to study which non-visual capabilities the reward reinforces.

---

## Appendix

### A. Experimental setup and metrics

**Checkpoints and benchmarks.** We evaluated the released Qwen2-VL-2B-Instruct and MedVLM-R1 checkpoints. The official set contains 900 questions covering MRI (300), CT (300), and X-ray (300); 825 of these questions have four choices. External tests include VQA-RAD closed Yes/No (251 questions), SLAKE English closed Yes/No test (355 questions, 96 images) and validation (358 questions, 96 images), and a fixed 1,000-question PMC-VQA four-choice sample (852 images). SLAKE and PMC-VQA comparisons use the same Qwen2-VL/MedVLM-R1 pair.

**Image interventions.** The question, answer options, and ground-truth label remain fixed while the image changes:

| Condition | Model input | Measurement purpose |
|---|---|---|
| Clean | Original paired image | Ordinary medical VQA |
| No-image | Question and options alone | Non-visual answering performance |
| Blank | Question with an uninformative image | Effect of image presence and formatting |
| Mismatch | Another real medical image paired with the original question | Contribution of the correct image content |
| Blur | Original image at increasing Gaussian blur strengths | Response to image degradation |
| Semantic conflict | Different image with dataset-label-supported conflicting findings | Response to more targeted visual contradictions |

The fixed mismatch mappings use real medical images, with the official 900-question quality experiment incorporating three independently generated donor permutations. For semantic conflict, 375 eligible questions were assembled using dataset labels: 109 stronger mutually exclusive or polarity cases and 266 different-finding cases. These categories identify the strength of available label evidence.

**Candidate-answer probabilities.** For the displayed answer set \(\mathcal{Y}\), we normalize model scores over its options and map all option-order permutations back to their semantic answer identities. Let \(y^*\) be the reference answer. We compute:

\[
\begin{aligned}
\mathrm{Accuracy} &= \mathbb{1}\!\left[\arg\max_{y\in\mathcal{Y}}p(y)=y^*\right],\\
M_{p} &= p(y^*)-\max_{y\ne y^*}p(y),\\
M_{\log} &= \log p(y^*)-\max_{y\ne y^*}\log p(y),\\
H(p)&=-\sum_{y\in\mathcal{Y}}p(y)\log p(y).
\end{aligned}
\]

The signed GT margins are positive when the correct answer ranks first and negative when an alternative ranks first. \(M_p\) and \(M_{\log}\) have different scales. Entropy summarizes uncertainty across the provided choices. We report **GT log margin** for image-content comparisons and explicitly identify the scoring protocol for each result.

**Visual content utility at blur level \(q\).** With the same blur level applied to the paired and mismatched images:

\[
U_{\mathrm{content}}(q)
= M_{\log}(I_{\mathrm{correct}},q)
-\frac{1}{K}\sum_{k=1}^{K}M_{\log}(I_{\mathrm{mismatch},k},q),\qquad K=3.
\]

Option positions and image-donor assignments were fixed across checkpoints. The direct Base–RL accuracy study used matched candidate scoring with three answer-option permutations for four-choice questions. The earlier model-specific intervention run used native answer generation for MedVLM-R1 and candidate scoring for Base; its within-checkpoint changes are presented separately below.

### B. Visual reliance and quality sensitivity

**B.1. Clean, No-image, Mismatch, and Blur.** The initial 825-question experiment produced the following within-checkpoint accuracy results:

| Image condition | Qwen2-VL Base, candidate scoring | MedVLM-R1, native generation |
|---|---:|---:|
| Clean | 64.73% | 81.46% |
| No-image | 43.39% | 55.69% |
| Mismatch | 50.18% | 69.58% |
| Severe blur | 54.42% | 72.82% |

Both checkpoints achieve their highest accuracy with the correctly paired image. The same experiment also shows the distinct effects of removing the image entirely and replacing it with a different real image. Each checkpoint's condition differences describe its response to image intervention; the matched inter-checkpoint comparison appears in Appendix C.

**B.2. Correct-image advantage across quality levels.** The 900-question quality-by-relevance experiment measures \(U_{\mathrm{content}}(q)\) using three fixed real-image donors at each blur level:

| Model | Clean | Mild blur | Medium blur | Severe blur |
|---|---:|---:|---:|---:|
| Qwen2-VL Base | 0.730 | 0.736 | 0.543 | 0.211 |
| MedVLM-R1 | 0.894 | 0.883 | 0.647 | 0.228 |

![Correct-image GT log-margin advantage by synthetic blur level](figures/quality_content_gap.png)

The correct-image advantage remains relatively stable under mild degradation, then decreases more sharply at medium and severe blur. This pattern gives a direct, quality-conditioned measure of the predictive value of the *correctly paired* visual content.

**B.3. Signed GT margin, entropy, and representation probing.** The initial 825-question model-specific runs showed the following mean signed GT **log** margins:

| Input | Qwen2-VL Base | MedVLM-R1 |
|---|---:|---:|
| Clean | 1.045 | 4.102 |
| Mild blur | 1.025 | 3.982 |
| Medium blur | 0.840 | 3.560 |
| Severe blur | 0.333 | 2.184 |
| No-image | −0.181 | 0.295 |

The additional, matched-scoring mechanism analysis produced mean **answer entropies**:

| Input | Qwen2-VL Base | MedVLM-R1 |
|---|---:|---:|
| Clean | 0.756 | 0.220 |
| Mild blur | 0.755 | 0.227 |
| Medium blur | 0.813 | 0.245 |
| Severe blur | 0.972 | 0.362 |
| No-image | 1.098 | 0.473 |

These two complementary measurements show falling gold-answer advantage and increasing uncertainty with stronger degradation. In a separate layer-wise representation study, Clean versus severe blur was linearly decodable with best AUROC approximately 1.000 in both checkpoints. Representation probing measures the visibility of degradation information in the model's hidden states, alongside the behavioral evidence above.

**B.4. Confidence sharpening and visual utility.** Following the visual-evidence sensitivity question raised by VES-RFT, define image-induced entropy reduction and signed gold-margin utility as:

\[
\Delta H=H(p_{\mathrm{NoImage}})-H(p_{\mathrm{Image}}),
\qquad
U_{\mathrm{presence}}=M_{\log}(p_{\mathrm{Image}})-M_{\log}(p_{\mathrm{NoImage}}).
\]

We examined cases with \(\Delta H>0\) together with \(U_{\mathrm{presence}}<0\): the image increases confidence while decreasing support for the reference answer. On the **375 label-supported semantic-conflict questions, Clean image quality**, this combination appeared in **30.7%** of Base predictions and **23.5%** of MedVLM-R1 predictions. The result highlights the value of examining entropy changes together with correctness-aware margins when evaluating visual evidence.

### C. What changes after RL?

**C.1. Matched Clean and No-image comparison.** The same candidate-answer scoring and option-order controls were applied to both checkpoints:

| Evaluation set | Base Clean | RL Clean | Base No-image | RL No-image | RL − Base Clean | RL − Base No-image |
|---|---:|---:|---:|---:|---:|---:|
| MRI, 300 | 67.33% | 96.44% | 43.56% | 75.67% | +29.11 pp | **+32.11 pp** |
| Four-choice, 825 | 68.89% | 80.97% | 44.12% | 57.94% | +12.08 pp | **+13.82 pp** |

![RL checkpoint gains with and without image input](figures/post_rl_uplift.png)

We define the extra image-presence benefit as \(\mathrm{Acc}(\mathrm{Clean})-\mathrm{Acc}(\mathrm{NoImage})\). The RL-versus-Base change in this quantity was **−3.00 pp** for MRI (95% bootstrap CI [−7.56, +1.67] pp) and **−1.74 pp** for all 825 four-choice questions (95% CI [−3.76, +0.40] pp). The image-presence increment was comparable across checkpoints even as their overall accuracy improved.

Qwen2-VL Base reached **44.12% No-image accuracy** on the 825 four-choice questions, versus 25% for uniform random selection. This reveals substantial predictive information in the non-visual task input. RL increased performance in this same text-only setting by **13.82 pp** across the full four-choice set and **32.11 pp** on MRI. Medical knowledge, option wording, and task-specific answer regularities are candidate sources of that gain.

**C.2. Questions both models solved with the original image.** We isolated **506** original four-choice questions answered correctly by both checkpoints under Clean images. Of these, **499** have paired valid No-image outputs:

| No-image outcome on the same 499 questions | Count |
|---|---:|
| Both checkpoints correct | 219 |
| RL checkpoint correct, Base incorrect | **82** |
| Base correct, RL checkpoint incorrect | 46 |
| Both checkpoints incorrect | 152 |

Thus, within the paired 499-question set, MedVLM-R1 retained **301/499 (60.3%)** correct No-image answers, compared with **265/499 (53.1%)** for Base. The initial Clean-success selection used native generation for MedVLM-R1 and candidate scoring for Base, consistent with the original experimental records. This analysis focuses on items that both checkpoints could already solve when seeing the image, then measures the additional non-visual success of the RL checkpoint. Together with the matched full-set comparisons, it motivates a closer study of reinforcement of task priors and existing medical capabilities.

### D. External benchmarks and auxiliary analyses

External evaluations of the same two released checkpoints used the same frozen question sets and controlled candidate-answer scoring:

| Benchmark | Base Clean | RL Clean | RL − Base Clean | Base No-image | RL No-image |
|---|---:|---:|---:|---:|---:|
| SLAKE test, 355 | 68.0% | 62.3% | −5.8 pp | 45.4% | 42.7% |
| SLAKE validation, 358 | 69.0% | 64.8% | −4.2 pp | 44.1% | 43.2% |
| PMC-VQA, 1,000 | 45.1% | 45.4% | +0.3 pp | 31.4% | 31.1% |
| VQA-RAD Yes/No, 251 | 61.8% | 63.5% | +1.8 pp | 51.8% | 54.8% |

Paired image-clustered bootstrap 95% intervals for the new external Clean differences were **[−9.1, −2.6] pp** on SLAKE test, **[−7.2, −1.4] pp** on SLAKE validation, and **[−0.6, +1.2] pp** on PMC-VQA. These experiments demonstrate variation in post-training gains across medical imaging tasks, question formats, and datasets.

In additional SLAKE test analysis, **28 same-question, same-modality groups** contained both Yes- and No-labeled images. Across these groups, the mean difference in predicted Yes probability between Yes-labeled and No-labeled images was **0.203 for Base** and **0.217 for MedVLM-R1**. Both checkpoints respond to image content even for fixed question wording.

We also produced frozen text-only answerability subsets for the official benchmark (684 consensus image-required IDs, 598 with additional question-quality screening, and 205 in a stricter exploratory subset). An auxiliary DeepSeek text-only analysis of SLAKE and PMC-VQA is retained in the experiment records. The headline external comparisons use the complete frozen test samples.

### E. Reproducibility and related work

The [public experiment repository](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance) includes fixed donor assignments, model/data download scripts, dataset preparation, GPU evaluation, CPU metric computation, plots, provenance of the original experiments, and an [end-to-end reproduction guide](docs/REPRODUCE_FROM_SCRATCH.md). The CPU results can be regenerated directly from the released numeric records; the GPU workflow loads the released model checkpoints and official dataset assets.

Related work motivating the experiments:

- [VES-RFT (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html) — visual-evidence sensitivity and image-induced changes in prediction confidence.
- [SurgCheck (2026)](https://link.springer.com/article/10.1007/s11548-026-03739-4) — evaluation of medically grounded image use and possible language shortcuts.
- [MedVLM-R1 (MICCAI 2025)](https://papers.miccai.org/miccai-2025/0553-Paper3267.html) — reinforcement learning for medical visual question answering.
- [Reference-free whole-heart MR image quality assessment (Radiology: AI, 2020)](https://pubs.rsna.org/doi/full/10.1148/ryai.2020190123) — clinically grounded image-quality assessment.
