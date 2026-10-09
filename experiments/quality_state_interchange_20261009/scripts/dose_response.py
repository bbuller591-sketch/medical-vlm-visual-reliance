#!/usr/bin/env python3
"""Exploratory dose-response for held-out quality direction at L27.

Uses fixed train/eval split, pretrained models, no fitting on evaluation output.
Doses are multiples of previously frozen half-median-scale quality direction.
"""
import argparse,json,os,sys,time
from pathlib import Path
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
MED=find_research_root();ROOT=Path(__file__).resolve().parents[1];BASE=MED/'visual_reliance_mechanism_v2'
sys.path.insert(0,str(BASE/'scripts'))
import common_v2 as V
import m2_extract as X
from run_interchange import load_states,safe_probability
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
meta=json.loads((ROOT/'results'/f'metadata_{a.model}.json').read_text())
train=meta['train_ids'];test=meta['eval_ids'];k=27
states=load_states(a.model)
qdiff=np.stack([states['blur_severe'][s][k].astype('float32')-states['clean'][s][k].astype('float32') for s in train])
q=qdiff.mean(0);q=q/(np.linalg.norm(q)+1e-12)*np.median(np.linalg.norm(qdiff,axis=1))*.5
output=ROOT/'results'/f'dose_{a.model}.jsonl'
if output.exists():raise RuntimeError(str(output)+' already exists')
device=f'cuda:{a.gpu}';proc=AutoProcessor.from_pretrained(str(V.MODELS[a.model]));ip=proc.image_processor
ip.min_pixels=V.MATCHED_MIN_PIXELS;ip.max_pixels=V.MATCHED_MAX_PIXELS
if isinstance(getattr(ip,'size',None),dict):ip.size.update(longest_edge=V.MATCHED_MAX_PIXELS,shortest_edge=V.MATCHED_MIN_PIXELS)
tok=[proc.tokenizer.encode(l,add_special_tokens=False)[0] for l in 'ABCD']
model=Qwen2VLForConditionalGeneration.from_pretrained(str(V.MODELS[a.model]),torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
dec=X._resolve(model,['model.language_model.layers','model.layers','language_model.layers'],'layers')[0]
items=V.load_items();t0=time.monotonic();n=0
with output.open('w') as f:
 for i,sid in enumerate(test):
  it=items[sid]
  for cond in ['clean','random_same_modality_replacement']:
   enc,order,_,error=X.build_inputs(model,proc,it,cond,'O0',device);assert not error,error
   current=states[cond][sid][k].astype('float32')
   for alpha in [0.5,1.5]:
    vector=current+alpha*q
    def hook(m,inp,out):
     h=out[0].clone();h[:,-1,:]=torch.as_tensor(vector,device=h.device,dtype=h.dtype)
     return (h,)+tuple(out[1:])
    hh=dec[k-1].register_forward_hook(hook)
    try:
     with torch.no_grad():output_tensor=model(**enc,logits_to_keep=1,use_cache=False)
     r=safe_probability(output_tensor.logits[0,-1,tok].float().cpu().numpy(),it['n_options'],order['permutation'],it['gt_index'])
    finally:hh.remove()
    f.write(json.dumps(dict(sample_id=sid,condition=cond,layer=k,alpha=alpha,**r))+'\n');n+=1
  if (i+1)%15==0:
   f.flush();print('DOSE',a.model,i+1,'/',len(test),'sec',round(time.monotonic()-t0,1),flush=True)
print('DOSE_DONE',a.model,'records',n,'secs',round(time.monotonic()-t0,1),flush=True)
