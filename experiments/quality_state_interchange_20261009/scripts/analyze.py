#!/usr/bin/env python3
"""Predeclared paired estimands and per-question bootstrap for quality-state pilot."""
from pathlib import Path
from collections import defaultdict
import csv,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results';FIG=ROOT/'figures'
RNG=np.random.default_rng(927410)
BS=4000
MAPPING={'baseline':'Baseline','quality_shift':'Quality direction','random_shift':'Random control','mismatch_shift':'Mismatch direction','full_blur_state':'Full Blur state','full_mismatch_state':'Full Mismatch state','full_clean_state':'Full Clean state','reverse_quality_shift':'Reverse quality direction','reverse_mismatch_shift':'Reverse mismatch direction'}

def summarize(vals):
 vals=np.asarray(vals,dtype=float)
 n=len(vals)
 if n==0:return {'n':0}
 idx=RNG.integers(0,n,size=(BS,n));means=vals[idx].mean(axis=1)
 return {'n':n,'mean':round(float(vals.mean()),5),'ci95':[round(float(v),5) for v in np.percentile(means,[2.5,97.5])],
         'median':round(float(np.median(vals)),5),'frac_negative':round(float(np.mean(vals<0)),4)}

def analyze_one(model):
 p=RESULTS/f'steering_{model}.jsonl'
 if not p.is_file():raise FileNotFoundError(p)
 meta=json.loads((RESULTS/f'metadata_{model}.json').read_text())
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 # avoid silently accepting partial runs
 expected=29*meta['eval_n']
 if len(rows)!=expected:raise AssertionError((model,len(rows),expected))
 d={(r['sample_id'],r['condition'],r['layer'],r['intervention']):r for r in rows}
 assert len(d)==len(rows)
 sids=meta['eval_ids'];layers=meta['layers']
 result={'meta':{k:meta[k] for k in ['train_n','eval_n','layers','seed','quality_direction','mismatch_direction','direction_diagnostics']},'effects':{},'content_gap':{},'baseline':{},'behavioral_shift':{}}
 for cond in ['clean','blur_severe','random_same_modality_replacement']:
  data=[d[sid,cond,0,'baseline'] for sid in sids]
  result['baseline'][cond]={'accuracy':round(float(np.mean([r['correct'] for r in data])),5),
                            'log_margin':round(float(np.mean([r['log_margin'] for r in data])),5),
                            'prob_margin':round(float(np.mean([r['prob_margin'] for r in data])),5),
                            'entropy':round(float(np.mean([r['entropy'] for r in data])),5)}
  for k in layers:
   for name in sorted(set(r['intervention'] for r in rows if r['condition']==cond and r['layer']==k)):
    a=[d[sid,cond,k,name] for sid in sids]
    difflog=[x['log_margin']-y['log_margin'] for x,y in zip(a,data)]
    diffprob=[x['prob_margin']-y['prob_margin'] for x,y in zip(a,data)]
    diffacc=[x['correct']-y['correct'] for x,y in zip(a,data)]
    delta_entropy=[x['entropy']-y['entropy'] for x,y in zip(a,data)]
    lost=sum(y['correct']==1 and x['correct']==0 for x,y in zip(a,data)); numclean=sum(y['correct'] for y in data)
    result['effects'][f'{cond}|L{k}|{name}']={
        'delta_log_margin':summarize(difflog),'delta_prob_margin':summarize(diffprob),
        'delta_accuracy':summarize(diffacc),'delta_entropy':summarize(delta_entropy),
        'clean_correct_n':numclean,'correct_to_wrong_n':lost,'correct_to_wrong_rate':round(lost/numclean,4) if numclean else None}
 for k in layers:
  base=[d[sid,'clean',0,'baseline']['log_margin']-d[sid,'random_same_modality_replacement',0,'baseline']['log_margin'] for sid in sids]
  result['content_gap'][f'L{k}|baseline']=summarize(base)
  for name in ['quality_shift','random_shift']:
   steered=[d[sid,'clean',k,name]['log_margin']-d[sid,'random_same_modality_replacement',k,name]['log_margin'] for sid in sids]
   result['content_gap'][f'L{k}|{name}']=summarize(steered)
   result['content_gap'][f'L{k}|{name}|change']=summarize(np.asarray(steered)-np.asarray(base))
  # true control comparison: same-item quality vs random intervention changes.
  quality=np.array([d[sid,'clean',k,'quality_shift']['log_margin']-d[sid,'random_same_modality_replacement',k,'quality_shift']['log_margin'] for sid in sids])
  random=np.array([d[sid,'clean',k,'random_shift']['log_margin']-d[sid,'random_same_modality_replacement',k,'random_shift']['log_margin'] for sid in sids])
  result['content_gap'][f'L{k}|quality_minus_random']=summarize(quality-random)
  # effect on clean confidence when clean image is inserted unchanged
  for name in ['quality_shift','random_shift','full_blur_state','full_mismatch_state']:
   rec=[d[sid,'clean',k,name] for sid in sids]
   result['behavioral_shift'][f'L{k}|{name}|clean_entropy_mean'] = summarize([r['entropy'] for r in rec])
 return result,rows

def main():
 FIG.mkdir(exist_ok=True)
 full={}
 for model in ['qwen2vl','medvlmr1']:
  r,rows=analyze_one(model);full[model]=r
  print('MODEL',model,'EVAL N',r['meta']['eval_n'],'TRAIN',r['meta']['train_n'],'layers',r['meta']['layers'])
  print('BASELINE',r['baseline'])
  for k in r['meta']['layers']:
   print('LAYER',k,'QUALITY_ALIGN',r['meta']['direction_diagnostics'][str(k)]['heldout_quality_direction_alignment_fraction'] if str(k) in r['meta']['direction_diagnostics'] else r['meta']['direction_diagnostics'][k]['heldout_quality_direction_alignment_fraction'])
   for name in ['quality_shift','random_shift','full_blur_state','full_mismatch_state']:
    effect=r['effects'][f'clean|L{k}|{name}']
    print('CLEAN',name,'Δ log margin',effect['delta_log_margin'],'flip correct->wrong',effect['correct_to_wrong_n'],'/',effect['clean_correct_n'])
   for name in ['quality_shift','random_shift','quality_minus_random']:
    print('CONT GAP',name,r['content_gap'][f'L{k}|{name}|change' if name!='quality_minus_random' else f'L{k}|{name}'])
  print()
 (RESULTS/'summary.json').write_text(json.dumps(full,indent=2,ensure_ascii=False))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 # Plot two distinct figures, one for clean state effects, one for matched-vs-mismatch.
 labels=['Full Blur','Full Mismatch','Quality direction','Random direction']
 entries=['full_blur_state','full_mismatch_state','quality_shift','random_shift']
 for chart in ['clean_log_margin_change','visual_content_advantage_change']:
  fig,ax=plt.subplots(figsize=(9,4.35),dpi=180)
  if chart=='clean_log_margin_change':
   xx=np.arange(4)
   for i,(m,title) in enumerate([('qwen2vl','Qwen Base'),('medvlmr1','MedVLM-R1')]):
    vals=[full[m]['effects'][f'clean|L27|{x}']['delta_log_margin']['mean'] for x in entries]
    bars=ax.bar(xx+(-.18 if i==0 else .18),vals,width=.35,label=title)
    ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
   ax.set_xticks(xx,labels)
   ax.set_ylabel('Change in GT log margin after patching (L27)')
  else:
   xx=np.arange(2)
   for i,(m,title) in enumerate([('qwen2vl','Qwen Base'),('medvlmr1','MedVLM-R1')]):
    vals=[full[m]['content_gap'][f'L27|{x}|change']['mean'] for x in ['quality_shift','random_shift']]
    bars=ax.bar(xx+(-.18 if i==0 else .18),vals,width=.35,label=title)
    ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=2)
   ax.set_xticks(xx,['Quality direction','Random control'])
   ax.set_ylabel('Change in correct − mismatched GT log-margin advantage (L27)')
  ax.axhline(0,linewidth=.8,color='gray');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.14)
  ax.legend();ax.set_title('Held-out 90-question controlled intervention pilot')
  fig.tight_layout();fig.savefig(FIG/(chart+'.png'),dpi=180);plt.close(fig)
 print('ANALYSIS_COMPLETE',RESULTS/'summary.json')
if __name__=='__main__':main()
