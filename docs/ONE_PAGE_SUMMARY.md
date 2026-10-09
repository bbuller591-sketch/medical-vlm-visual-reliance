# When Does a Medical VLM Actually Use Its Image?

*Research pilot — 9 October 2026*

### What we measured

We studied whether medical-VQA answers depend on the **current image**, and whether this dependence changes with image quality. We compared **Qwen2-VL-2B-Instruct** with its RL-trained checkpoint **MedVLM-R1** on the official 900-question medical-VQA benchmark (including 825 four-choice questions). We tested **Clean** images, **No-image**, a different real medical image paired with the same question (**Mismatch**), and increasing synthetic **Blur**. We measured accuracy, signed ground-truth (GT) log margin, and answer-choice entropy. Mismatch preserves a real visual input but does not guarantee a clinically contradictory image.

### What we found

**Correct visual evidence helps, but the benefit decreases with degradation.** On 900 questions under matched scoring, the mean **GT log-margin benefit** of the correctly paired image over mismatched images was:

| Model | Clean | Mild blur | Medium | Severe |
|---|---:|---:|---:|---:|
| Qwen Base | 0.730 | 0.736 | 0.543 | 0.211 |
| MedVLM-R1 | 0.894 | 0.883 | 0.647 | 0.228 |

Stronger blur generally reduced GT margin and increased answer entropy. Linear probing decoded severe degradation from both checkpoints' internal states (best AUROC ≈ 1.00), although decodability does not establish causal or clinically appropriate use. Importantly, in a label-supported semantic-conflict test, adding an image sometimes **reduced entropy while also lowering the correct-answer margin** (31% of Base cases; 23% of RL cases). Thus, image-induced confidence alone is not evidence of *useful* visual reliance.

### What surprised us

**The RL checkpoint's advantage remained large without images.** With the **same candidate-scoring protocol for both checkpoints**:

| Evaluation | RL−Base Clean gain | RL−Base No-image gain |
|---|---:|---:|
| MRI, 300 questions | +29.1 pp | **+32.1 pp** |
| All four-choice, 825 | +12.1 pp | **+13.8 pp** |

The additional Clean−No-image benefit did **not** increase significantly after RL. Even Base achieved **44.1% without images** on 825 four-choice questions (25% uniform-guess baseline), indicating exploitable non-visual information. On 506 items both checkpoints answered correctly with images, RL also retained more correct answers after removing the image. This controls prior **image-conditioned success**, not equal prior medical knowledge. A plausible hypothesis is that RL reinforced medical knowledge and/or task-specific question-answer priors, rather than yielding a proportional gain in current-image grounding.

This gain was **not consistent across external tasks**: MedVLM-R1 versus Base was −5.8 pp on SLAKE test, −4.2 pp on SLAKE validation, and +0.3 pp on PMC-VQA. The cause remains unresolved.

**Open question:** Can we separate genuine, quality-aware visual trust from passive loss of readability? A matched **image-free RL** training control could also help test which capabilities the reward reinforces; this control has **not** been trained.

*See [supporting experiments](DETAILED_REPORT.md) for protocols, uncertainty intervals, and interpretation limits.*
