# Quality-associated state interchange and visual-content reliance (9 Oct 2026)

This experiment extends the official-900 medical VLM visual-reliance study with a controlled, held-out activation-interchange pilot.

Models: `Qwen2-VL-2B-Instruct` and `MedVLM-R1` (frozen). Matched visual processing at `min_pixels=3136`, `max_pixels=401408`. Per model: 450 stratified direction-construction questions and 90 disjoint stratified evaluation questions, seed `20261009`. Tested decoder layers `L20` and `L27`, original option ordering `O0`. The same question, label, checkpoint, and actual image remain fixed during decision-token residual interventions.

Two intervention classes:

- **Full state replacement:** use the same question's state from a severe-blur or random same-modality mismatched image while continuing Clean inference, plus reverse conditions.
- **Direction steering:** derive average `(blur_severe - clean)` decision-state change from 450 held-out-from-test questions, normalize to half the median training displacement, add it to evaluation states. Comparison controls are a same-norm random direction, a mismatch-derived direction, a No-image input arm, and follow-up strength levels `0.5` and `1.5` relative to the fixed base scale.

Results use signed GT log margin, probability margin, entropy, prediction correctness, and the Clean−Mismatch GT log-margin advantage evaluated per question. Differences and paired bootstrap intervals can be recomputed CPU-only with:

```bash
python scripts/analyze.py
python scripts/extend_analysis.py
```

All primary and diagnostic per-question outputs are recorded in `results/`, with frozen evaluation IDs and method parameters in `results/metadata_*.json`. Figures are reproducible with the same analysis scripts.

For **GPU replay** of the interventions, first run the original experiment pipeline to rebuild the official dataset, image conditions, and hidden states under `visual_reliance_mechanism_v2/features`, including `O0/{clean,blur_severe,no_image,random_same_modality_replacement}`. The GPU scripts access those cached features and model checkpoints from `MEDICAL_RESEARCH_ROOT` (default: ancestor directory containing the original mechanism project), and write results into this experiment directory. Example:

```bash
MEDICAL_RESEARCH_ROOT=/path/to/medical_workspace python scripts/run_interchange.py --model qwen2vl --gpu 0 --max-train-per-modality 150 --max-eval-per-modality 30 --layers 20,27
```

The two matched decision-state comparisons are conducted **within each checkpoint**. The quality-associated direction is an observable mixture of image-quality and content/decision effects; its relation to clinical image reliability is a subsequent scientific question.

**Source scripts (GitHub):** [`run_interchange.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/quality_state_interchange_20261009/scripts/run_interchange.py), [`analyze.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/quality_state_interchange_20261009/scripts/analyze.py), [`extend_analysis.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/quality_state_interchange_20261009/scripts/extend_analysis.py), [`dose_response.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/quality_state_interchange_20261009/scripts/dose_response.py), [`diagnostic_noimage.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/quality_state_interchange_20261009/scripts/diagnostic_noimage.py).
