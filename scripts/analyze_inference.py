#!/usr/bin/env python3
"""GPU-independent analysis of fresh model predictions (quality or external).

Computes per-item order-average probabilities, mean-per-order accuracy, signed
GT probability/log margin, entropy, matched-image utility, VES-style entropy
sharpening vs useful/harmful margin, and image-clustered confidence intervals.
Raw input predictions use synthetic item IDs and answer-choice probabilities only.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np


def cluster_ci(rows,values,n_boot=1000,seed=20261009):
 if not rows:return [None,None]
 groups=defaultdict(list)
 for r,v in zip(rows,values):groups[r['image_id']].append(float(v))
 keys=list(groups)
 rng=np.random.default_rng(seed)
 means=[]
 for _ in range(n_boot):
  picked=rng.integers(0,len(keys),len(keys))
  vals=[v for j in picked for v in groups[keys[j]]]
  means.append(float(np.mean(vals)))
 return [round(float(x),5) for x in np.quantile(means,[.025,.975])]

def accumulate(prediction_file):
 groups=defaultdict(list)
 with Path(prediction_file).open(encoding='utf-8') as f:
  for line in f:
   if not line.strip():continue
   rec=json.loads(line)
   groups[(rec['item_id'],rec['condition'])].append(rec)
 output={}
 for (sid,cond),rows in groups.items():
  # Exactly one row per permutation ID, avoiding double-counting an accidentally retried order.
  orders={r['order']:r for r in rows}
  if len(orders)!=len(rows):raise ValueError(f'duplicate prediction record: {sid} {cond}')
  gt=int(rows[0]['gt_index']);p=np.mean([np.asarray(r['semantic_probs'],float) for r in orders.values()],axis=0)
  if not np.isclose(p.sum(),1,atol=1e-6):raise ValueError(f'not normalized {sid} {cond}')
  other=np.delete(p,gt)
  output[(sid,cond)]={'item_id':sid,'condition':cond,'image_id':rows[0]['image_id'],
   'gt_index':gt,'n_options':len(p),
   'mean_per_order_accuracy':float(np.mean([r['correct'] for r in orders.values()])),
   'argmax_mean_prob_accuracy':int(np.argmax(p)==gt),
   'p_gt':float(p[gt]),
   'gt_prob_margin':float(p[gt]-other.max()),
   'gt_log_margin':float(np.log(max(p[gt],1e-300))-np.log(max(other.max(),1e-300))),
   'entropy':float(-np.sum(p*np.log(np.maximum(p,1e-300)))),
   'semantic_probs':p.tolist(),'orders':sorted(orders)}
 return output

def analyze(predictions,bootstrap=1000):
 data=accumulate(predictions)
 bycond=defaultdict(list)
 for r in data.values():bycond[r['condition']].append(r)
 result={'file':str(predictions),'n_unique_items':len({x[0] for x in data}),
         'method':'order-average semantic answer probabilities + separate per-order accuracy',
         'conditions':{},'contrasts':{}}
 for c,rows in sorted(bycond.items()):
  result['conditions'][c]={'n_items':len(rows),'n_images':len({r['image_id'] for r in rows}),
     'mean_per_order_accuracy':round(float(np.mean([r['mean_per_order_accuracy'] for r in rows])),6),
     'argmax_mean_prob_accuracy':round(float(np.mean([r['argmax_mean_prob_accuracy'] for r in rows])),6),
     'mean_p_gt':round(float(np.mean([r['p_gt'] for r in rows])),6),
     'mean_gt_prob_margin':round(float(np.mean([r['gt_prob_margin'] for r in rows])),6),
     'mean_gt_log_margin':round(float(np.mean([r['gt_log_margin'] for r in rows])),6),
     'mean_entropy':round(float(np.mean([r['entropy'] for r in rows])),6)}
 for cond in sorted(bycond):
  if cond=='no_image' or not ('no_image' in bycond):continue
  pairs=[(data[sid,cond],data[sid,'no_image']) for (sid,c) in data if c==cond and (sid,'no_image') in data]
  if not pairs:continue
  rows=[a for a,_ in pairs]
  U=[a['gt_log_margin']-b['gt_log_margin'] for a,b in pairs]
  sharpen=[b['entropy']-a['entropy'] for a,b in pairs]
  harmful=[int(delta>0 and util<0) for delta,util in zip(sharpen,U)]
  result['contrasts'][cond+'_vs_noimage']={'n':len(pairs),
    'mean_GT_log_margin_utility':round(float(np.mean(U)),6),
    'GT_log_margin_utility_95CI':cluster_ci(rows,U,bootstrap),
    'mean_entropy_reduction':round(float(np.mean(sharpen)),6),
    'sharpen_and_harm_rate':round(float(np.mean(harmful)),6),
    'sharpen_and_harm_95CI':cluster_ci(rows,harmful,bootstrap)}
 for q in ['clean','mild','medium','severe']:
  target='clean' if q=='clean' else f'blur_{q}'
  donors=[f'pm_{q}_k{k}' for k in (1,2,3)]
  pairs=[]
  for (sid,cond),x in data.items():
   if cond!=target:continue
   matches=[data.get((sid,c)) for c in donors]
   if not all(matches):continue
   pairs.append((x,np.mean([m['gt_log_margin'] for m in matches])))
  if not pairs:continue
  benefit=[x['gt_log_margin']-v for x,v in pairs]
  row=[x for x,_ in pairs]
  result['contrasts'][f'content_{q}']={'n':len(pairs),
   'mean_correct_minus_donor_GT_log_margin':round(float(np.mean(benefit)),6),
   '95CI_cluster_by_image':cluster_ci(row,benefit,bootstrap)}
 return result,data

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--predictions',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--per-item-output',type=Path,default=None)
 ap.add_argument('--bootstrap',type=int,default=1000)
 args=ap.parse_args()
 result,data=analyze(args.predictions,args.bootstrap)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(result,indent=2))
 if args.per_item_output:
  args.per_item_output.parent.mkdir(parents=True,exist_ok=True)
  with args.per_item_output.open('w') as f:
   for key in sorted(data):f.write(json.dumps(data[key])+'\n')
 print('ANALYSIS SAVED',args.output,'items',result['n_unique_items'])
 for k,v in result['contrasts'].items():print(k,v)
if __name__=='__main__':main()
