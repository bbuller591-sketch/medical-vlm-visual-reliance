#!/usr/bin/env python3
"""Post-primary diagnostic: does the quality direction also change text-only answers?

Uses EXACT same frozen splits, direction recipe, and chosen layers as main run.
Adds No-image arm and applies content direction to matched Mismatch arm.
"""
from pathlib import Path
import os,sys,json,time,argparse
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np,torch
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
def find_research_root():
    env=os.getenv('MEDICAL_RESEARCH_ROOT')
    if env and (Path(env)/'visual_reliance_mechanism_v2'/'features').is_dir():
        return Path(env).resolve()
    for folder in Path(__file__).resolve().parents:
        if (folder/'visual_reliance_mechanism_v2'/'features').is_dir():
            return folder
    raise RuntimeError('Set MEDICAL_RESEARCH_ROOT to a workspace containing visual_reliance_mechanism_v2/features')
MED=find_research_root()
ROOT=Path(__file__).resolve().parents[1]
BASE=MED/'visual_reliance_mechanism_v2'
sys.path.insert(0,str(BASE/'scripts'))
import common_v2 as V
import m2_extract as X
from run_interchange import load_states,safe_probability

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--gpu',type=int,required=True)
 a=p.parse_args()
 meta=json.loads((ROOT/'results'/f'metadata_{a.model}.json').read_text());train=meta['train_ids'];test=meta['eval_ids'];layers=meta['layers']
 output=ROOT/'results'/f'diagnostic_{a.model}.jsonl'
 if output.exists():raise RuntimeError('exists '+str(output))
 states=load_states(a.model)
 clean={}
 for file in sorted((BASE/'features'/a.model/'no_image'/'O0').glob('shard_*.npz')):
  with np.load(file) as x:
   for sid,vec in zip(x['sample_ids'],x['decision_hidden']):clean[str(sid)]=vec
 states['no_image']=clean
 assert all(sid in clean for sid in test)
 dirs={};rng=np.random.default_rng(20261009+1)
 for k in layers:
  qa=np.stack([states['blur_severe'][s][k].astype('float32')-states['clean'][s][k].astype('float32') for s in train]);q=qa.mean(0)
  ca=np.stack([states['random_same_modality_replacement'][s][k].astype('float32')-states['clean'][s][k].astype('float32') for s in train]);c=ca.mean(0)
  norm=float(np.median(np.linalg.norm(qa,axis=1)))*.5
  q=q/(np.linalg.norm(q)+1e-12)*norm;c=c/(np.linalg.norm(c)+1e-12)*norm
  random=rng.normal(size=q.shape).astype('float32');random=random/np.linalg.norm(random)*norm
  dirs[k]=(q,c,random)
 device=f'cuda:{a.gpu}'
 proc=AutoProcessor.from_pretrained(str(V.MODELS[a.model]));ip=proc.image_processor
 ip.min_pixels=V.MATCHED_MIN_PIXELS;ip.max_pixels=V.MATCHED_MAX_PIXELS
 if isinstance(getattr(ip,'size',None),dict):ip.size.update(longest_edge=V.MATCHED_MAX_PIXELS,shortest_edge=V.MATCHED_MIN_PIXELS)
 tok=[proc.tokenizer.encode(x,add_special_tokens=False)[0] for x in 'ABCD']
 model=Qwen2VLForConditionalGeneration.from_pretrained(str(V.MODELS[a.model]),torch_dtype=torch.bfloat16,
 attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
 dec=X._resolve(model,['model.language_model.layers','model.layers','language_model.layers'],'layers')[0]
 def forward(enc,order,it,k=None,v=None):
  hooks=[]
  if k is not None:
   def hook(m,inp,out):
    y=out[0].clone();y[:,-1,:]=torch.as_tensor(v,device=y.device,dtype=y.dtype)
    return (y,)+tuple(out[1:])
   hooks.append(dec[k-1].register_forward_hook(hook))
  try:
   with torch.no_grad():o=model(**enc,logits_to_keep=1,use_cache=False)
   return safe_probability(o.logits[0,-1,tok].float().cpu().numpy(),it['n_options'],order['permutation'],it['gt_index'])
  finally:
   for x in hooks:x.remove()
 items=V.load_items();n=0;t0=time.monotonic()
 with output.open('w') as f:
  for i,sid in enumerate(test):
   it=items[sid]
   for cond in ['no_image','random_same_modality_replacement']:
    enc,order,_,err=X.build_inputs(model,proc,it,cond,'O0',device);assert not err
    if cond=='no_image':
     r=forward(enc,order,it);f.write(json.dumps(dict(sample_id=sid,condition=cond,layer=0,intervention='baseline',**r))+'\n');n+=1
    for k in layers:
     q,c,random=dirs[k];h=states[cond][sid][k].astype('float32')
     changes={'quality_shift':h+q,'random_shift':h+random} if cond=='no_image' else {'mismatch_shift':h+c}
     for name,v in changes.items():
      r=forward(enc,order,it,k,v)
      f.write(json.dumps(dict(sample_id=sid,condition=cond,layer=k,intervention=name,**r))+'\n');n+=1
   if (i+1)%10==0:
    f.flush();print('DIAG',a.model,i+1,'/',len(test),'sec',round(time.monotonic()-t0,1),flush=True)
 print('DONE_DIAG',a.model,'n',n,'secs',round(time.monotonic()-t0,1),flush=True)
if __name__=='__main__':main()
