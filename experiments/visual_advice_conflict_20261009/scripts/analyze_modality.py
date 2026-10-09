#!/usr/bin/env python3
"""Paired stratified summary of text-suggestion susceptibility by modality."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/'results/per_item_metrics.csv')))
by=defaultdict(dict)
for r in rows:
 key=(r['sample_id'],r['modality'])
 by[key][r['model'],r['condition']]=r
cats=['MRI','CT','Xray']
conditions=['clean','blur_severe','random_same_modality_replacement','blurred_mismatch','no_image']
rng=np.random.default_rng(20261009)
def cims(x):
 x=np.array(x,dtype=float);n=len(x)
 vals=np.array([x[rng.integers(0,n,n)].mean() for _ in range(4000)])
 return {'n':n,'mean':round(float(x.mean()),7),'ci95':[round(float(v),7) for v in np.quantile(vals,[.025,.975])]}
out={}
for mod in cats:
 ids=sorted(s for s,md in by if md==mod and len(by[s,md])==10)
 out[mod]={'n':len(ids),'models':{},'interactions':{}}
 for model in ['qwen2vl','medvlmr1']:
  out[mod]['models'][model]={}
  for cond in conditions:
   rr=[by[s,mod][model,cond] for s in ids]
   out[mod]['models'][model][cond]={k:cims([float(r[k]) for r in rr]) for k in ['follow_nonbaseline','wrong_hint_prob_lift','hint_stability','baseline_correct','confidence_pct']}
 for cond in conditions[1:]:
  effects={}
  for m in ['qwen2vl','medvlmr1']:
   effects[m]=[float(by[s,mod][m,cond]['follow_nonbaseline'])-float(by[s,mod][m,'clean']['follow_nonbaseline']) for s in ids]
  out[mod]['interactions'][cond]={'base_change':cims(effects['qwen2vl']),
                                   'rl_change':cims(effects['medvlmr1']),
                                   'rl_minus_base_change':cims(np.array(effects['medvlmr1'])-np.array(effects['qwen2vl']))}
(ROOT/'results/summary_by_modality.json').write_text(json.dumps(out,indent=2)+'\n')
print('MODALITY ANALYSIS')
for mod,z in out.items():
 print(mod,'N',z['n'])
 for cond in conditions[1:]:
  x=z['interactions'][cond]
  print(' ',cond,'base',round(100*x['base_change']['mean'],2),'R1',round(100*x['rl_change']['mean'],2),'DIFFERENCE',round(100*x['rl_minus_base_change']['mean'],2),'CI',[round(100*t,2) for t in x['rl_minus_base_change']['ci95']])
