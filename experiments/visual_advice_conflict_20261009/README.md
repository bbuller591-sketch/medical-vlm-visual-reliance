# Text Advice × Visual Quality & Pairing — 2026-10-09

This folder provides the finished, frozen analysis (559 joint-Clean-correct questions; two checkpoints; 27,950 scored prompt–image trials):

- [Complete research report](reports/REPORT.md)
- [Frozen experimental protocol](reports/PROTOCOL.md)
- [Aggregate results](results/summary.json) and [modality breakdown](results/summary_by_modality.json)
- [Exact item selection and model-specific advice confidence](results/selection.json)
- [Per-item response metrics](results/per_item_metrics.csv)
- Raw scored results: `results/advice_qwen2vl.jsonl`, `results/advice_medvlmr1.jsonl`
- Figures under `figures/`; scripts under `scripts/`.

The source medical images and 559 generated blurred mismatch images were not copied into the public-release staging directory. The complete source-image-dependent working directory remains at `<medical_research_workspace>/visual_advice_conflict_20261009/`. The GPU scripts as originally authored also import the local frozen original MedVLM-R1 data and Qwen processor pipeline from `<medical_research_workspace>/visual_reliance_mechanism_v2/` and `<medical_research_workspace>/official900_visual_reliance_v1/`. **The public bundle is intended for auditing the completed outputs, not self-contained image-level model re-execution.** Model-specific advice confidence prevents clean attribution of all between-checkpoint differences to RL; all pairwise quality effects are properly caveated in the report.

Files are staged locally for review and have not been pushed to GitHub.

**Source scripts (GitHub):** [`prepare.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/visual_advice_conflict_20261009/scripts/prepare.py), [`run.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/visual_advice_conflict_20261009/scripts/run.py), [`analyze.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/visual_advice_conflict_20261009/scripts/analyze.py), [`analyze_modality.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/visual_advice_conflict_20261009/scripts/analyze_modality.py), [`plot.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/visual_advice_conflict_20261009/scripts/plot.py), [`verify.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/visual_advice_conflict_20261009/scripts/verify.py), [`write_report.py`](https://github.com/bbuller591-sketch/medical-vlm-visual-reliance/blob/main/experiments/visual_advice_conflict_20261009/scripts/write_report.py).
