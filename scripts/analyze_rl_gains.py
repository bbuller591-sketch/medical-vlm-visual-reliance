#!/usr/bin/env python3
"""Recompute matched Base/RL image benefit and its item-paired uncertainty.

Unit of resampling is the original question; its O0/O1/O2 scores are grouped.
No external GPU or model calls; reads published candidate-score records.
"""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RECORDS=ROOT/'results/official825_scores.csv'
OUTPUT=ROOT/'results/rl_matched_gains_audit.json'
N_BOOT=10000
SEED=20261009

def load():
    items={}
    with RECORDS.open(newline='') as f:
        for r in csv.DictReader(f):
            assert r['n_options']=='4'
            sid=r['sample_id'];pair=(r['model'],r['condition'])
            key=(sid,pair)
            items.setdefault(sid,{'modality':r['modality'],'vals':{}})
            assert items[sid]['modality']==r['modality']
            assert r['order'] not in items[sid]['vals'].setdefault(pair,{})
            items[sid]['vals'][pair][r['order']]=int(r['correct'])
    assert len(items)==825,len(items)
    keys=[(m,c) for m in ['qwen2vl','medvlmr1'] for c in ['clean','no_image']]
    for sid,v in items.items():
        assert set(v['vals'])==set(keys)
        for pair,d in v['vals'].items():assert set(d)=={'O0','O1','O2'},(sid,pair,d)
    return items,keys

def ci_array(arr,labels,indices,nboot,rng):
    a=arr[indices,:]
    n=len(a)
    means=a.mean(axis=0)
    draws=np.empty((nboot,len(labels)))
    for t in range(nboot):
        idx=rng.integers(0,n,size=n)
        draws[t]=a[idx].mean(axis=0)
    result={}
    for j,k in enumerate(labels):
        result[k]={'mean':round(100*float(means[j]),4),
                   'ci95':[round(100*float(q),4) for q in np.quantile(draws[:,j],[.025,.975])]}
    return result

def main():
    its,keys=load();ids=sorted(its)
    mat=np.array([[np.mean(list(its[sid]['vals'][k].values())) for k in keys] for sid in ids])
    assert mat.shape==(825,4)
    # cells order: Base clean, Base no-image, RL clean, RL no-image
    bc,bn,rc,rn=mat.T
    derived=np.stack([bc,bn,rc,rn,bc-bn,rc-rn,rc-bc,rn-bn,(rc-rn)-(bc-bn)],axis=1)
    labels=['base_clean','base_noimage','rl_clean','rl_noimage','base_image_benefit',
            'rl_image_benefit','rl_gain_clean','rl_gain_noimage','image_benefit_difference_RL_minus_Base']
    result={'method':'Item-paired bootstrap, retaining three option-order scores per original question; 10000 resamples per group, seed=20261009',
            'source':str(RECORDS.relative_to(ROOT)), 'group_scores':{}}
    rng=np.random.default_rng(SEED)
    groups={'all825':np.arange(825),'MRI300':np.array([i for i,sid in enumerate(ids) if its[sid]['modality']=='MRI']),
            'CT237':np.array([i for i,sid in enumerate(ids) if its[sid]['modality']=='CT']),
            'Xray288':np.array([i for i,sid in enumerate(ids) if its[sid]['modality']=='Xray'])}
    for group,indices in groups.items():
        result['group_scores'][group]={'n':len(indices), 'metrics_pp':ci_array(derived,labels,indices,N_BOOT,rng)}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print('WRITTEN',OUTPUT)
    for group,g in result['group_scores'].items():
        print('GROUP',group,'n',g['n'])
        for k in labels:
            entry=g['metrics_pp'][k]
            print(' ',k,'=',entry['mean'],'CI',entry['ci95'])

if __name__=='__main__':main()
