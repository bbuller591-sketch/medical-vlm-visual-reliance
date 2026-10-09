#!/usr/bin/env python3
"""Integrate post-primary diagnostic and exploratory dose-response controls."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures'
SEED=610091; rng=np.random.default_rng(SEED);B=5000

def ci(arr):
 a=np.array(arr,dtype=float);n=len(a)
 if n!=90:raise AssertionError((n,a.shape))
 return {'mean':round(float(a.mean()),5),'ci95':[round(float(x),5) for x in np.percentile(a[rng.integers(0,n,(B,n))].mean(1),[2.5,97.5])],
         'n':n}

all_summary=json.loads((R/'summary.json').read_text())
for model in ['qwen2vl','medvlmr1']:
 def load(name):
  p=R/(name+'_'+model+'.jsonl')
  return list(map(json.loads,filter(None,p.read_text().splitlines())))
 main=load('steering');diag=load('diagnostic');dose=load('dose')
 index=lambda rr:{(r['sample_id'],r['condition'],r.get('layer',0),r.get('intervention',r.get('alpha'))):r for r in rr}
 a=index(main);d=index(diag);z=index(dose)
 ids=json.loads((R/f'metadata_{model}.json').read_text())['eval_ids']
 x={};data=all_summary[model]
 for k in [20,27]:
  for treatment in ['quality_shift','random_shift']:
   x[f'noimage_L{k}_{treatment}']=ci([d[s,'no_image',k,treatment]['log_margin']-d[s,'no_image',0,'baseline']['log_margin'] for s in ids])
   x[f'noimage_L{k}_{treatment}_entropy']=ci([d[s,'no_image',k,treatment]['entropy']-d[s,'no_image',0,'baseline']['entropy'] for s in ids])
  x[f'noimage_L{k}_quality_minus_random']=ci([d[s,'no_image',k,'quality_shift']['log_margin']-d[s,'no_image',k,'random_shift']['log_margin'] for s in ids])
  x[f'content_gap_L{k}_mismatch_direction_change']=ci([
    a[s,'clean',k,'mismatch_shift']['log_margin']-d[s,'random_same_modality_replacement',k,'mismatch_shift']['log_margin']
    -a[s,'clean',0,'baseline']['log_margin']+a[s,'random_same_modality_replacement',0,'baseline']['log_margin'] for s in ids])
 for alpha in [0.5,1.5]:
  x[f'dose_L27_alpha{alpha}_clean_change']=ci([z[s,'clean',27,alpha]['log_margin']-a[s,'clean',0,'baseline']['log_margin'] for s in ids])
  x[f'dose_L27_alpha{alpha}_content_gap_change']=ci([
   z[s,'clean',27,alpha]['log_margin']-z[s,'random_same_modality_replacement',27,alpha]['log_margin']
   -a[s,'clean',0,'baseline']['log_margin']+a[s,'random_same_modality_replacement',0,'baseline']['log_margin'] for s in ids])
 data['diagnostic_controls']=x
 print('\nMODEL',model)
 for k in sorted(x):print(k,x[k])
(R/'summary_extended.json').write_text(json.dumps(all_summary,indent=2,ensure_ascii=False))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7.5,4.2),dpi=180)
for model,label in [('qwen2vl','Qwen2-VL Base'),('medvlmr1','MedVLM-R1')]:
 d=all_summary[model]
 vals=[d['diagnostic_controls'][f'dose_L27_alpha{x}_content_gap_change']['mean'] if x!=1.0 else d['content_gap']['L27|quality_shift|change']['mean'] for x in [0.5,1.0,1.5]]
 ax.plot([0.5,1,1.5],vals,marker='o',label=label)
 ax.annotate(f'{vals[-1]:+.3f}',(1.5,vals[-1]),xytext=(-5,-13),textcoords='offset points',ha='right',fontsize=9)
ax.axhline(0,color='gray',linewidth=.7);ax.set_xticks([.5,1,1.5]);ax.set_xlabel('Quality-shift direction strength (relative to frozen base scale)');ax.set_ylabel('Change in correct − mismatched GT log-margin advantage')
ax.set_title('Held-out quality-direction dose response (L27, n = 90/model)');ax.legend();ax.grid(axis='y',alpha=.16)
fig.tight_layout();F.mkdir(exist_ok=True);fig.savefig(F/'quality_direction_dose_response.png',dpi=180);plt.close(fig)
print('EXTENDED_ANALYSIS_WRITTEN')
