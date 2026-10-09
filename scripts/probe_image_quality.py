#!/usr/bin/env python3
"""Reproducible layer-wise linear probing: clean vs severe-blur decision states.

GPU extraction: --mode extract --manifest official900.jsonl --checkpoint MODEL
CPU inference: --mode fit --features FILE.npz

Paired image/question items always stay in the same GroupKFold fold;
normalization is train-only. This quantifies *decodability*, not causal use.
The exact original advanced probe used additional encoder-side controls; original
source files are also archived in experiments/original_sources/ for audit.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_external as E
from evaluate_quality import transform_image

def extract(args):
 import torch
 from qwen_vl_utils import process_vision_info
 items=E.load_manifest(args.manifest)
 if args.limit:items=items[:args.limit]
 model,proc,cand_ids=E.load_checkpoint(args.checkpoint,args.device)
 data=[];ids=[]
 for i,item in enumerate(items):
  x=[]
  n=len(item['options'])
  prompt=item['question']+'\n'+'\n'.join(f'{"ABCD"[j]}) {v}' for j,v in enumerate(item['options']))
  prompt+='\n\nAnswer with the single letter of the correct option. Reply with only that letter.'
  for cond in ['clean','severe']:
   ip=item['image_path'] if cond=='clean' else transform_image(item['image_path'],'severe',args.cache_dir)
   content=[{'type':'image','image':str(ip)},{'type':'text','text':prompt}]
   msg=[{'role':'user','content':content}]
   txt=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
   img,vid=process_vision_info(msg)
   enc=proc(text=[txt],images=img,videos=vid,return_tensors='pt',padding=True)
   enc={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in enc.items()}
   with torch.inference_mode():
    out=model(**enc,output_hidden_states=True,use_cache=False)
   layers=np.stack([h[0,-1,:].float().cpu().numpy() for h in out.hidden_states],axis=0).astype('float16')
   x.append(layers)
  ids.append(item['item_id']);data.append(np.stack(x,axis=0))
  if (i+1)%25==0:print('EXTRACTED',i+1,'/',len(items),flush=True)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(args.output,items=np.array(ids),decision_hidden=np.stack(data,axis=0),conditions=np.array(['clean','severe']))
 print('SAVED DECISION FEATURES',args.output,'shape',np.stack(data).shape)

def fit(args):
 from sklearn.metrics import roc_auc_score
 from sklearn.model_selection import GroupKFold
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 z=np.load(args.features)
 ids=z['items'];X=z['decision_hidden'].astype('float32')
 assert X.ndim==4 and X.shape[1]==2 and len(ids)==X.shape[0]
 n,t=X.shape[0],X.shape[2]
 if n<10:raise ValueError('Need at least ten distinct paired items for grouped cross validation')
 # Observations ordered as [all clean, all severe]; same group ID is used twice.
 groups=np.tile(ids,2);y=np.repeat([0,1],n)
 result={'n_items':n,'folds':min(5,n),'layers':{},'evidence':'decodable representations only'}
 for k in range(t):
  D=np.concatenate([X[:,0,k,:],X[:,1,k,:]],axis=0)
  oof=np.zeros(len(y),dtype=float)
  for train,test in GroupKFold(n_splits=min(5,n)).split(D,y,groups):
   scaler=StandardScaler().fit(D[train])
   lr=LogisticRegression(max_iter=1500).fit(scaler.transform(D[train]),y[train])
   oof[test]=lr.predict_proba(scaler.transform(D[test]))[:,1]
  result['layers'][str(k)]={'auroc':round(float(roc_auc_score(y,oof)),6)}
 vals={int(k):v['auroc'] for k,v in result['layers'].items()}
 result['best_layer']=max(vals,key=vals.get);result['best_auroc']=max(vals.values())
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
 print('GROUPED PROBE',n,'images','BEST_LAYER',result['best_layer'],'AUROC',result['best_auroc'])

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=['extract','fit'],required=True)
 p.add_argument('--manifest',type=Path)
 p.add_argument('--checkpoint',type=Path)
 p.add_argument('--features',type=Path)
 p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0)
 p.add_argument('--cache-dir',type=Path,default=Path('outputs/degraded_cache'))
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.mode=='extract':
  if not a.manifest or not a.checkpoint:p.error('GPU extraction needs --manifest and --checkpoint')
  extract(a)
 else:
  if not a.features:p.error('Probe fitting requires --features .npz')
  fit(a)
if __name__=='__main__':main()
