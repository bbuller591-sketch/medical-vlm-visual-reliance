#!/usr/bin/env python3
"""Frozen O0 joint-Clean-correct subset audit: Clean vs No-image.

No new model inference. Results for the identical 559 questions are obtained
from the completed 559-item text-advice experiment's NO-HINT records.
"""
from pathlib import Path
from collections import Counter
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'experiments/visual_advice_conflict_20261009/results'
selection=json.loads((BUNDLE/'selection.json').read_text())['items']
assert len(selection)==559
raw={}
for model in ['qwen2vl','medvlmr1']:
 for line in (BUNDLE/f'advice_{model}.jsonl').open():
  r=json.loads(line)
  if r['hint_letter'] is not None or r['condition'] not in ['clean','no_image']:continue
  key=(model,r['sample_id'],r['condition'])
  assert key not in raw,key
  raw[key]=r

def bootstrap(arr,seed=20261010,b=10000):
 vals=np.asarray(arr,dtype=float)
 g=np.random.default_rng(seed)
 means=np.array([vals[g.integers(0,len(vals),len(vals))].mean() for _ in range(b)])
 return [float(v) for v in np.quantile(means,[.025,.975])]

out={'protocol':'Four-choice normalized candidate scoring; original option order O0; both Clean correct; no-hint Clean/No-image from fully scored 559 experiment','seed':20261010,'bootstrap_draws':10000,'groups':{}}
for modality in ['All 559','MRI 207','CT 160','X-ray 192']:
 kind=modality.split()[0]
 entries=[x for x in selection if kind=='All' or x['modality'].lower()==kind.lower().replace('-','')]
 ids=[x['sample_id'] for x in entries]
 clean={m:np.array([int(raw[m,sid,'clean']['correct']) for sid in ids]) for m in ('qwen2vl','medvlmr1')}
 no={m:np.array([int(raw[m,sid,'no_image']['correct']) for sid in ids]) for m in ('qwen2vl','medvlmr1')}
 assert all(clean[m].sum()==len(ids) for m in clean)
 a=no['qwen2vl'];r=no['medvlmr1'];diff=r-a
 tab=Counter(f'base_{i}_rl_{j}' for i,j in zip(a,r))
 data={'n':len(ids),'clean_accuracy':{'base':float(clean['qwen2vl'].mean()),'rl':float(clean['medvlmr1'].mean()),'rl_minus_base':float((clean['medvlmr1']-clean['qwen2vl']).mean())},
       'no_image_accuracy':{'base':float(a.mean()),'rl':float(r.mean()),'rl_minus_base':float(diff.mean()),'rl_minus_base_ci95':bootstrap(diff)},
       'image_presence_accuracy_benefit':{'base':float((1-a).mean()),'rl':float((1-r).mean()),'rl_minus_base':float((a-r).mean()),'rl_minus_base_ci95':bootstrap(a-r)},
       'no_image_paired_counts':{k:int(v) for k,v in tab.items()},'no_image_correct_counts':{'base':int(a.sum()),'rl':int(r.sum())}}
 out['groups'][kind]=data
 print(modality,'n',len(ids),'clean: base=rl=100.00%',
       'no_image base',f'{100*a.mean():.2f}%', 'rl',f'{100*r.mean():.2f}%',
       'delta',f'{100*diff.mean():+.2f} pp','95CI',[round(100*z,2) for z in data['no_image_accuracy']['rl_minus_base_ci95']], 'counts',data['no_image_paired_counts'])
assert sum(out['groups'][m]['n'] for m in ['MRI','CT','X-ray'])==559
p=ROOT/'results/joint_clean_559_noimage_audit.json'
p.write_text(json.dumps(out,indent=2)+'\n')
print('WRITTEN',p)
