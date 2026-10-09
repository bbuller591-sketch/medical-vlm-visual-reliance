#!/usr/bin/env python3
"""Write a readable stand-alone English research report from frozen metrics."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
x=json.load(open(ROOT/'results/summary.json'))
MOD=json.load(open(ROOT/'results/summary_by_modality.json'))
N=x['items_complete']
models={'qwen2vl':'Qwen2-VL Base','medvlmr1':'MedVLM-R1'}
cond={'clean':'Clean','blur_severe':'Severe Blur','random_same_modality_replacement':'Mismatch','blurred_mismatch':'Blur + Mismatch','no_image':'No-image'}
def pp(a):return f'{100*a:.1f}%'
def num(a,d=2):return f'{a:+.{d}f}'
def ci(a,scale=100,d=2):return f"[{a[0]*scale:+.{d}f}, {a[1]*scale:+.{d}f}]"
def value(k,m,c):return x['metrics'][m][c][k]['mean']
def lines():
 L=[]
 def w(t=''):L.append(t)
 w('# Visual Evidence vs. Model-Specific Confidence-Calibrated Text Advice')
 w()
 w('**Status:** '+('Full 559-item matched analysis.' if N==559 else f'Pilot / partial run only ({N}/559 matched items).'))
 w(f'**Cohort:** {N} questions on which both checkpoints predicted the original reference option correctly on Clean with matched four-option candidate scoring (O0). Full eligible cohort has MRI 207, CT 160, X-ray 192.')
 w()
 w('## Question and design')
 w()
 w('We ask whether changing the quality or pairing of visual input increases susceptibility to a fixed, numeric-confidence text suggestion, and whether this **within-checkpoint change** differs between the Base and the medically RL-trained checkpoint. We do not equate text susceptibility or entropy changes with direct proof of an explicit visual-trust mechanism.')
 w()
 w('For each eligible item, the checkpoint-specific Clean O0 highest-option probability is converted to a text-advice confidence by `25 + 0.5 × (100×p_clean−25)` percent. Once computed, the confidence is held **constant for that item/checkpoint across all five visual conditions and all four hinted options**. Each question is scored with no hint, then once each with suggestions of A, B, C, and D, using the otherwise identical answer instruction.')
 w()
 w('The five visual conditions are original Clean, Severe Blur of the original image, a fixed random real **same-modality Mismatch**, Severe Blur of the same mismatched image, and genuinely No-image. For Mismatch, the original label is **not automatically the correct diagnosis of the replacement image**. Matching and image quality are separate interventions.')
 w()
 w('**Important cross-check:** models receive *different* hint confidence values, as requested (their own initial Clean confidence). Absolute Base–R1 following rates thus jointly reflect checkpoint behavior and different prompt strengths. Within-checkpoint visual-condition differences are more interpretable; model interaction comparisons are descriptive, with paired CIs.')
 w()
 w('## Primary results')
 w()
 w('**Following rate** counts the fraction of the three suggestions **different from the current no-hint predicted option** that succeed in moving the final argmax to the suggested option. This is computed separately at every visual condition; it is not a majority vote over four prompted answers.')
 w()
 w('| Visual condition | Base: following | R1: following | Base: mean wrong-option probability lift | R1: lift |')
 w('|---|---:|---:|---:|---:|')
 for c,n in cond.items():
  w(f"| {n} | **{pp(value('follow_nonbaseline','qwen2vl',c))}** | **{pp(value('follow_nonbaseline','medvlmr1',c))}** | {pp(value('wrong_hint_prob_lift','qwen2vl',c))} | {pp(value('wrong_hint_prob_lift','medvlmr1',c))} |")
 w()
 w('Wrong-option probability lift is the change in the probability assigned to the *hinted original non-reference option*, relative to its no-hint probability, averaged over three alternative semantic options. In Mismatch arms this is only an option-level shift relative to the **original** label; it is not necessarily wrong relative to the donor image.')
 w()
 w('![Suggestion following under matched image conditions](../figures/following_by_condition.png)')
 w()
 w('## Within-model quality effects and Base/RL comparison')
 w()
 w('For each question and checkpoint, calculate following-rate change from that checkpoint’s **Clean** condition. The R1 minus Base effect below is the **paired difference in these changes**. All intervals use item-paired bootstrap over the same scored questions.')
 w()
 w('| Visual condition vs Clean | Base change (pp) | R1 change (pp) | R1 − Base interaction (pp, 95% CI) |')
 w('|---|---:|---:|---:|')
 for c in list(cond)[1:]:
  o=x['interactions'][c]['follow_nonbaseline']
  a=o['base_change']['mean']*100;b=o['rl_change']['mean']*100;d=o['rl_minus_base_change']
  w(f'| {cond[c]} | {num(a)} | {num(b)} | **{num(d["mean"]*100)}** {ci(d["ci95"])} |')
 w()
 w('![Changes relative to Clean](../figures/relative_to_clean.png)')
 w()
 w('## Modality-specific change in advice susceptibility')
 w()
 w('The RL-trained checkpoint received medical RL on MRI data, so a pooled contrast can hide substantial between-modality differences. The table below uses **the same paired question IDs within each modality** and gives confidence intervals for the checkpoint interaction in severe-blur sensitivity.')
 w()
 w('| Modality | Shared Clean-correct n | Base: Severe − Clean | R1: Severe − Clean | R1 − Base (95% CI, pp) |')
 w('|---|---:|---:|---:|---:|')
 for modality in ['MRI','CT','Xray']:
  z=MOD[modality];t=z['interactions']['blur_severe']
  w(f"| {modality} | {z['n']} | {num(t['base_change']['mean']*100)} pp | {num(t['rl_change']['mean']*100)} pp | **{num(t['rl_minus_base_change']['mean']*100)}** {ci(t['rl_minus_base_change']['ci95'])} |")
 w()
 w('The strong pooled Base/RL difference in severe-blur response is chiefly driven by MRI. In CT and X-ray the corresponding checkpoint-interaction intervals contain zero; these are exploratory subgroup comparisons, not evidence of equivalence.')
 w()
 w('## Previously correct answers: are they changed by wrong advice?')
 w()
 w('By construction, **every item is originally Clean-correct for both checkpoints**. For each visual arm we additionally restrict to items whose **no-hint response under that arm remains correct**, then calculate how often a suggestion of one of the three *originally wrong* options makes the final answer incorrect. The denominator is `3 × number of still-correct no-hint items`, not all 559×3 for degraded or no-image conditions.')
 w()
 w('| Visual condition | Base no-hint correct (n) | Base: originally correct → wrong | R1 no-hint correct (n) | R1: originally correct → wrong |')
 w('|---|---:|---:|---:|---:|')
 for c in ['clean','blur_severe','no_image']:
  a=x['metrics']['qwen2vl'][c];b=x['metrics']['medvlmr1'][c]
  w(f"| {cond[c]} | {a['baseline_correct_count']}/{N} | {pp(a['correct_to_wrong_conditioned']['mean'])} | {b['baseline_correct_count']}/{N} | {pp(b['correct_to_wrong_conditioned']['mean'])} |")
 w()
 w('We do not interpret original-label accuracy on Mismatch as diagnostic correctness without independent labels for the replacement images. For these arms, susceptibility to prompting is the principal measurement.')
 w()
 w('## Response stability and prompt confidence')
 w()
 w('**Answer stability** is the maximum number of times any single option is selected across four hinted answers divided by four; it is descriptive and does not replace the option-conditioned following rate.')
 w()
 w('| Visual condition | Base answer stability | R1 answer stability |')
 w('|---|---:|---:|')
 for c,n in cond.items():w(f'| {n} | {pp(value("hint_stability","qwen2vl",c))} | {pp(value("hint_stability","medvlmr1",c))} |')
 w()
 w(f"Model-specific mean text confidence in this selection is **{value('confidence_pct','qwen2vl','clean'):.2f}% (Base)** and **{value('confidence_pct','medvlmr1','clean'):.2f}% (R1)**, frozen per question across image conditions. The numeric confidence controls are not calibrated claims about actual advice correctness.")
 w()
 w('## Interpretation and scope')
 w()
 w('The finding measures **behavioral susceptibility to a given text statement when visual conditions change**. It does not by itself prove that the model explicitly recognizes image quality and chooses to reduce visual reliance. Two particular caveats matter: (1) random Mismatch produces an alternative real medical image, which may itself contain meaningful evidence, and (2) the model-specific confidence rule gives Base and R1 different numerical statements. No-hint baseline shifts and probability lifts should accompany following rates; a separate shared-hint control would isolate prompt-strength differences if needed.')
 w()
 w('## Reproduction files')
 w()
 w('All files are relative to the enclosing `visual_advice_conflict_20261009` experiment folder:')
 w()
 w('- `reports/PROTOCOL.md`: the frozen design and metric definitions.')
 w('- `results/selection.json`: exact 559 question IDs, original gold index, each model’s source Clean confidence and derived hint confidence.')
 w('- `results/advice_qwen2vl.jsonl`, `results/advice_medvlmr1.jsonl`: all per-item, per-condition, per-option predictions and probability vectors, with prompt fingerprint.')
 w('- `results/per_item_metrics.csv`, `results/summary.json`: per-item derived metrics, model comparisons and question-paired bootstrap intervals.')
 w('- `scripts/prepare.py`, `scripts/run.py`, `scripts/analyze.py`, `scripts/verify.py`, `scripts/plot.py`: preparation, checkpoint-resumable GPU inference, analysis, validation, figures.')
 w('- For Mismatch+Blur, actual locally generated images are in `images/blurred_mismatch/`; the original clean, blur and donor images come from the existing frozen `medical/official900_visual_reliance_v1` assets.')
 w()
 w(f'**Record audit:** {N} complete question IDs × 5 visual conditions × 5 prompt settings × 2 checkpoints = {N*50:,} recorded predictions in this analysis. The final run expects 27,950 rows.')
 return '\n'.join(L)+'\n'
(ROOT/'reports/REPORT.md').write_text(lines())
print('WRITTEN_REPORT',ROOT/'reports/REPORT.md','ITEMS',N)
