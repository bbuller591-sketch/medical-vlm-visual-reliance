#!/usr/bin/env python3
"""Recompute all headline numeric results from released non-image CSV files.
No GPU or medical images required. `--check` tests frozen values and selected counts.
"""
from pathlib import Path
import argparse,csv,json,math
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'

def load_csv(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def close(v,x,tol=.006):
 assert abs(v-x)<tol,(v,x)
def metric(x):return round(100*float(x),3)

def analyze_official(check):
 rows=load_csv(R/'official825_scores.csv')
 assert len(rows)==9900,len(rows)
 d=defaultdict(list)
 for r in rows:
  assert r['n_options']=='4' and r['condition'] in ('clean','no_image')
  d[(r['model'],r['modality'],r['condition'])].append(int(r['correct']))
 outcomes={}
 for label,modality,n in [('all',None,825),('mri','MRI',300)]:
  res={}
  for m in ['qwen2vl','medvlmr1']:
   for cond in ['clean','no_image']:
    v=[int(r['correct']) for r in rows if r['model']==m and r['condition']==cond and (modality is None or r['modality']==modality)]
    assert len(v)==n*3,(label,m,cond,len(v))
    res[f'{m}_{cond}']=float(np.mean(v))
  res['uplift_clean']=res['medvlmr1_clean']-res['qwen2vl_clean']
  res['uplift_no_image']=res['medvlmr1_no_image']-res['qwen2vl_no_image']
  res['interaction']=res['uplift_clean']-res['uplift_no_image']
  outcomes[label]=res
 if check:
  close(outcomes['all']['qwen2vl_clean'],.688889)
  close(outcomes['all']['medvlmr1_clean'],.809697)
  close(outcomes['all']['uplift_no_image'],.138182)
  close(outcomes['mri']['uplift_clean'],.291111)
  close(outcomes['mri']['uplift_no_image'],.321111)
 return outcomes

def analyze_quality(check):
 rows=load_csv(R/'quality_content_donor_contrasts.csv')
 assert len(rows)==24
 out={}
 for m in ['qwen2vl','medvlmr1']:
  out[m]={}
  for q in ['clean','mild','medium','severe']:
   values=[float(r['value']) for r in rows if r['model']==m and r['quality_level']==q]
   assert len(values)==3,(m,q,values)
   out[m][q]=float(np.mean(values))
 if check:
  close(out['qwen2vl']['clean'],.7301,.002)
  close(out['medvlmr1']['clean'],.8942,.002)
  close(out['medvlmr1']['severe'],.228,.002)
 return out

def analyze_external(check):
 out={}
 for ds,n in [('slake_yesno_test',355),('pmcvqa',1000)]:
  rows=load_csv(R/f'{ds}_per_item.csv')
  assert len(rows)==n
  res={}
  for m in ['qwen2vl','medvlmr1']:
   for c in ['clean','no_image','shuffled_image']:
    col=f'{m}__{c}__acc_order_mean'
    res[f'{m}_{c}']=float(np.mean([float(r[col]) for r in rows]))
  res['uplift_clean']=res['medvlmr1_clean']-res['qwen2vl_clean']
  res['content_base']=res['qwen2vl_clean']-res['qwen2vl_shuffled_image']
  res['content_rl']=res['medvlmr1_clean']-res['medvlmr1_shuffled_image']
  out[ds]=res
 if check:
  close(out['slake_yesno_test']['uplift_clean'],-.05775,.001)
  close(out['pmcvqa']['uplift_clean'],.00333,.001)
 return out

def plot_all(a,q):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 F=ROOT/'figures';F.mkdir(exist_ok=True)
 labels=['Clean','Mild','Medium','Severe']
 fig,ax=plt.subplots(figsize=(6.7,3.65),dpi=180)
 for m,title in [('qwen2vl','Qwen2-VL Base'),('medvlmr1','MedVLM-R1')]:
  ax.plot(labels,[q[m][l.lower()] for l in labels],marker='o',linewidth=2,label=title)
 ax.set_ylabel('GT log-margin advantage: correct − mismatched')
 ax.set_xlabel('Image quality (synthetic blur)')
 ax.set_ylim(0,1.05);ax.legend(frameon=False);ax.grid(alpha=.15)
 fig.tight_layout();fig.savefig(F/'quality_content_gap.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.7,3.6),dpi=180)
 x=np.arange(2);w=.35
 ax.bar(x-w/2,[a[k]['uplift_clean']*100 for k in ['mri','all']],w,label='Clean')
 ax.bar(x+w/2,[a[k]['uplift_no_image']*100 for k in ['mri','all']],w,label='No-image')
 ax.set_xticks(x,['MRI 300','Four-choice 825']);ax.set_ylabel('MedVLM-R1 − Base accuracy (pp)')
 ax.set_ylim(0,39);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
 for rect in ax.patches:
  ax.annotate(f'{rect.get_height():.1f}',(rect.get_x()+rect.get_width()/2,rect.get_height()),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9)
 fig.tight_layout();fig.savefig(F/'post_rl_uplift.png');plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--plots',action='store_true');args=ap.parse_args()
 a=analyze_official(args.check);q=analyze_quality(args.check);ext=analyze_external(args.check)
 if args.plots:plot_all(a,q)
 result={'official_matched':a,'quality_content':q,'external':ext,'note':'QWEN and RL matched per-order candidate-scoring; quality in separate frozen log-margin dataset'}
 out=R/'headline_recomputed.json';out.write_text(json.dumps(result,indent=2))
 print('REPRODUCED',out)
 for name,d in a.items():print('MAIN',name,'Clean uplift pp',metric(d['uplift_clean']),'No-image uplift pp',metric(d['uplift_no_image']),'Interaction pp',metric(d['interaction']))
 for name,d in ext.items():print('EXTERNAL',name,'Clean uplift pp',metric(d['uplift_clean']))
 for model,d in q.items():print('QUALITY',model,'clean',round(d['clean'],4),'severe',round(d['severe'],4))
 if args.check:print('CHECKS PASSED')
if __name__=='__main__':main()
