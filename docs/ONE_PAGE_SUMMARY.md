# How Much Does a Medical VLM Rely on Its Image?

*Pilot findings on visual evidence, image quality, and medical RL — 9 October 2026*

### Question and measurements

We asked whether correct medical-VQA answers actually depend on the **current image**, and whether that dependence changes as images deteriorate. We evaluated **Qwen2-VL-2B-Instruct** and its RL-trained counterpart **MedVLM-R1** on the official medical-VQA benchmark (900 MRI/CT/X-ray questions; 825 four-choice questions for the primary comparison). For each question, we tested its original image (**Clean**), no image, and a different real image substituted for the original (**Mismatch**); we also applied mild-to-severe synthetic blur. We measured answer accuracy, **signed ground-truth margin** (correct-answer score minus the strongest incorrect-answer score), and entropy over the answer choices. A mismatched image is not necessarily a clinically contradictory image.

### What we found: visual evidence helps, but its benefit depends on quality

On 900 questions under the same probability-scoring protocol, the *mean correct-image benefit over mismatched images*, measured as the **GT log-margin difference**, fell with degradation:

| Checkpoint | Clean | Mild blur | Medium blur | Severe blur |
|---|---:|---:|---:|---:|
| Qwen2-VL-2B | 0.730 | 0.736 | 0.543 | 0.211 |
| MedVLM-R1 | 0.894 | 0.883 | 0.647 | 0.228 |

Answer uncertainty generally increased under stronger blur, while probing recovered near-perfect information about severe image degradation from both checkpoints' internal representations. **Knowing that an image changed is not the same as appropriately relying on it.** In a label-supported semantic-conflict subset, adding an image sometimes *sharpened* the predicted answer distribution while *lowering* the correct-answer margin (about **31%** of cases for Base; **23%** for MedVLM-R1). Thus, entropy reduction alone is not reliable evidence of *useful* visual grounding.

### What surprised us: large gains survive with no visual input

Under the **same candidate-scoring and option-order protocol** for both checkpoints:

| Benchmark | RL−Base, Clean | RL−Base, No-image | Change in Clean−No-image benefit |
|---|---:|---:|---:|
| MRI, 300 questions | **+29.1 pp** | **+32.1 pp** | −3.0 pp |
| Four-choice, 825 questions | **+12.1 pp** | **+13.8 pp** | −1.7 pp |

Both changes in image-presence benefit have uncertainty intervals spanning zero. On **506 questions both checkpoints answered correctly with Clean images**, the RL checkpoint retained more correct answers after image removal; this controls prior success with an image, **not prior non-visual medical knowledge**. Even Base scored **44.1% without images** on the 825 four-choice questions, above the 25% uniform-guess baseline. RL may therefore be reinforcing useful medical knowledge and/or predictable question/answer patterns without proportionately improving current-image grounding. These are hypotheses, not a causal attribution.

The same RL checkpoint did not reproduce its large MRI gain on external benchmarks: **SLAKE test −5.8 pp**, **SLAKE validation −4.2 pp**, **PMC-VQA +0.3 pp** (latter statistically unresolved). This points to task-dependent benefits rather than a general conclusion that the model does not use images.

### Next question

Can we distinguish genuine quality-aware visual trust from passive degradation of unreadable inputs, and can a **matched image-free RL training control** separate reinforcement of visual understanding from reinforcement of text-based answer priors? These are proposed follow-ups; no new RL training has been performed.

*The four-choice and MRI comparisons use matched scoring. The quality/relevance contrasts use a separate frozen quality-analysis protocol; absolute margins should not be pooled across protocols. See [detailed supporting results](DETAILED_REPORT.md).*
