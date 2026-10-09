#!/usr/bin/env python3
"""Full fixed-sample hint influence analysis, with item-paired bootstrap.

Definitions (per item/visual condition):
  follow_nonbaseline = fraction of three hinted alternatives to the local
                       no-hint argmax that become the hinted argmax.
  wrong_follow = fraction of the three non-GT hint choices that become argmax.
  correct_to_wrong = among baseline-correct local visual inputs, fraction of
                     three wrong hints that make final answer incorrect.
  wrong_hint_prob_lift = mean over three wrong semantic options of the increase
                         in hinted-option probability vs the same no-hint arm.
  hint_stability = max fraction of 4 final predictions selecting the same option.
Cross-visual effects use item-level differences, holding per-model confidence fixed.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
VIS=['clean','blur_severe','random_same_modality_replacement','blurred_mismatch','no_image']
MODELS=['qwen2vl','medvlmr1']

def load_rows():
 data={}
 for m in MODELS:
  p=ROOT/'results'/f'advice_{m}.jsonl'
  if not p.is_file():continue
  for line in p.read_text().splitlines():
   if not line.strip():continue
   r=json.loads(line)
   key=(r['sample_id'],r['condition'],r['hint_letter'])
   if (m,key) in data:raise ValueError(('duplicate',m,key))
   data[m,key]=r
 return data

def compute_item(data,m,sid,cond,gt):
 base=data.get((m,(sid,cond,None)))
 responses={j:data.get((m,(sid,cond,j))) for j in 'ABCD'}
 if base is None or any(v is None for v in responses.values()):return None
 for j,v in responses.items():
  assert v['hint_semantic_index']==ord(j)-65
  assert v['sample_id']==sid and v['model']==m and v['gt_index']==gt
 bpred=base['prediction'];b_correct=base['correct']
 follows=[int(responses[j]['prediction']==ord(j)-65) for j in 'ABCD' if ord(j)-65!=bpred]
 assert len(follows)==3
 wrongs=[j for j in 'ABCD' if ord(j)-65!=gt]
 assert len(wrongs)==3
 wrong_follow=[int(responses[j]['prediction']==ord(j)-65) for j in wrongs]
 correct_to_wrong=[int(responses[j]['prediction']!=gt) for j in wrongs]
 wrong_lift=[responses[j]['probs_semantic'][ord(j)-65]-base['probs_semantic'][ord(j)-65] for j in wrongs]
 all_lift=[responses[j]['probs_semantic'][ord(j)-65]-base['probs_semantic'][ord(j)-65] for j in 'ABCD']
 counts=Counter(v['prediction'] for v in responses.values())
 confidence=responses['A']['confidence_pct']
 assert all(v['confidence_pct']==confidence for v in responses.values())
 return dict(sample_id=sid,modality=base['modality'],gt_index=gt,condition=cond,model=m,
     baseline_pred=bpred,baseline_correct=b_correct,confidence_pct=confidence,
     follow_nonbaseline=float(np.mean(follows)),wrong_follow=float(np.mean(wrong_follow)),
     correct_to_wrong=float(np.mean(correct_to_wrong)) if b_correct else None,
     unconditional_wrong_after_hint=float(np.mean(correct_to_wrong)),
     wrong_hint_prob_lift=float(np.mean(wrong_lift)),all_hint_prob_lift=float(np.mean(all_lift)),
     hint_stability=max(counts.values())/4,
     wrong_hint_argmax_count=int(sum(wrong_follow)),
     base_pgt=base['p_gt'],base_entropy=base['entropy'])

def bootstrap_mean(x,seed=20261009,b=4000):
 arr=np.asarray(x,dtype=float);assert len(arr)>1
 rng=np.random.default_rng(seed)
 draws=np.empty(b)
 for i in range(b):draws[i]=arr[rng.integers(0,len(arr),len(arr))].mean()
 return {'n':len(arr),'mean':float(arr.mean()),'ci95':[float(z) for z in np.quantile(draws,[.025,.975])]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
 sel=json.load(open(ROOT/'results/selection.json'))['items'];gt_by_id={x['sample_id']:x['gt_index'] for x in sel}
 rec=load_rows();eligible=[];computed={}
 for entry in sel:
  sid=entry['sample_id'];v={}
  for m in MODELS:
   for cond in VIS:v[(m,cond)]=compute_item(rec,m,sid,cond,entry['gt_index'])
  if all(x is not None for x in v.values()):eligible.append(sid);computed[sid]=v
 print('COMPLETE',len(eligible),'OF',len(sel),'INTERVENTION_ROWS',len(rec))
 if args.require_complete:assert len(eligible)==len(sel),len(eligible)
 if len(eligible)<10:return
 output={'protocol':'same questions both correct clean O0; per-checkpoint adaptive self-confidence; four semantic hint choices; same visual and hint options per item; no-image is no visual channel','items_total':len(sel),'items_complete':len(eligible),'modalities':dict(Counter(x['modality'] for x in sel)),'metrics':{},'interactions':{}}
 itemrows=[]
 for sid in eligible:
  for m in MODELS:
   for cond in VIS:itemrows.append(computed[sid][(m,cond)])
 p=ROOT/'results'/'per_item_metrics.csv'
 with p.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(itemrows[0]));writer.writeheader();writer.writerows(itemrows)
 for m in MODELS:
  output['metrics'][m]={}
  for c in VIS:
   rows=[computed[sid][(m,c)] for sid in eligible]
   val={}
   for metric in ['baseline_correct','follow_nonbaseline','wrong_follow','unconditional_wrong_after_hint','wrong_hint_prob_lift','all_hint_prob_lift','hint_stability','confidence_pct','base_entropy']:
    val[metric]=bootstrap_mean([r[metric] for r in rows])
   correct_rows=[r for r in rows if r['baseline_correct']]
   val['baseline_correct_count']=len(correct_rows)
   val['correct_to_wrong_conditioned']=bootstrap_mean([r['correct_to_wrong'] for r in correct_rows]) if len(correct_rows)>1 else None
   output['metrics'][m][c]=val
 for comparison in ['blur_severe','random_same_modality_replacement','blurred_mismatch','no_image']:
  output['interactions'][comparison]={}
  for metric in ['follow_nonbaseline','wrong_hint_prob_lift','unconditional_wrong_after_hint','hint_stability']:
   base=np.array([computed[sid][('qwen2vl',comparison)][metric]-computed[sid][('qwen2vl','clean')][metric] for sid in eligible]);rl=np.array([computed[sid][('medvlmr1',comparison)][metric]-computed[sid][('medvlmr1','clean')][metric] for sid in eligible])
   output['interactions'][comparison][metric]={'base_change':bootstrap_mean(base),'rl_change':bootstrap_mean(rl),'rl_minus_base_change':bootstrap_mean(rl-base)}
  for m in MODELS:
   change=np.array([computed[sid][(m,comparison)]['wrong_hint_prob_lift']-computed[sid][(m,'clean')]['wrong_hint_prob_lift'] for sid in eligible])
   output['interactions'][comparison][f'{m}_wrong_hint_lift_change']=bootstrap_mean(change)
 (ROOT/'results'/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
 print('MAIN METRICS (percentages)')
 for m in MODELS:
  print('MODEL',m)
  for c in VIS:
   a=output['metrics'][m][c]
   print(' ',c,'correct',round(a['baseline_correct']['mean']*100,2),'FOLLOW',round(a['follow_nonbaseline']['mean']*100,2),
       'WRONG_HINT_FOLLOW',round(a['wrong_follow']['mean']*100,2),'WRONG_HINT_PROB_LIFT',round(a['wrong_hint_prob_lift']['mean']*100,2),'CONDITIONAL_CORRECT_TO_WRONG',round(100*a['correct_to_wrong_conditioned']['mean'],2) if a['correct_to_wrong_conditioned'] else 'na')
 for c in output['interactions']:
  t=output['interactions'][c]['follow_nonbaseline']
  print('RELATIVE_TO_CLEAN',c,'Base',round(100*t['base_change']['mean'],2),'RL',round(100*t['rl_change']['mean'],2),'RL_minus_Base',round(100*t['rl_minus_base_change']['mean'],2),'CI',[round(100*x,2) for x in t['rl_minus_base_change']['ci95']])

if __name__=='__main__':main()
