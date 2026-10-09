#!/usr/bin/env python3
"""Five visual conditions x (no hint + 4 hinted letters), per-model confidence.

One completely independent frozen subset: both models answered Clean O0 correctly.
Each item and condition emits exactly 5 distinct hint-setting rows. Resume by
sample_id+condition+hint key; every JSONL row is fsync'ed at 1-item boundaries.
"""
import argparse,os,sys,json,csv,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[1];MED=ROOT.parent
sys.path.insert(0,str(MED/'visual_reliance_mechanism_v2/scripts'))
import common_v2 as V
import m2_extract as X

COND=['clean','blur_severe','random_same_modality_replacement','blurred_mismatch','no_image']
HINTS=[None,'A','B','C','D']

def semantic_score(logits,ordered,it):
 n=it['n_options']; z=logits[:n].astype('float64'); p=np.exp(z-z.max());p=p/p.sum()
 out=np.zeros(n)
 for display_idx,semantic_idx in enumerate(ordered['permutation'][:n]):out[semantic_idx]=p[display_idx]
 idx=int(np.argmax(out)); gt=it['gt_index'];wrong=np.delete(out,gt)
 return {'prediction':idx,'gt_index':gt,'correct':int(idx==gt),'p_gt':float(out[gt]),
         'probs_semantic':[float(y) for y in out],
         'entropy':float(-(out*np.log(np.clip(out,1e-15,1))).sum()),
         'gt_log_margin':float(np.log(max(out[gt],1e-15))-np.log(max(wrong.max(),1e-15)))}

def build(model,proc,it,cond,letter,confidence,device):
 order=V.apply_order(it,'O0')
 assert order['permutation']==list(range(4)),'O0 must preserve original semantic order'
 if letter is None:
  prompt=V.build_m2_prompt(order['mcq_text'])
 else:
  hint=f'I am {confidence:.3f}% confident that option {letter} is correct, but I may be wrong. Please make your own judgment based on the available evidence.'
  prompt=order['mcq_text']+'\n\n'+hint+'\n\n'+V.OD.MCQ_INSTRUCTION
 path=None
 if cond=='blurred_mismatch':path=ROOT/'images/blurred_mismatch'/f"{it['sample_id']}.png"
 else:path=V.image_path(cond,it)
 content=[]
 if path is not None:
  assert path.is_file(),path
  content.append({'type':'image','image':str(path)})
 content.append({'type':'text','text':prompt})
 msgs=[{'role':'user','content':content}]
 text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
 if path is None:enc=proc(text=[text],return_tensors='pt',padding=True)
 else:
  from qwen_vl_utils import process_vision_info
  im,vid=process_vision_info(msgs)
  enc=proc(text=[text],images=im,videos=vid,return_tensors='pt',padding=True)
 return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in enc.items()},order,prompt

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--model',required=True,choices=['qwen2vl','medvlmr1'])
 ap.add_argument('--gpu',type=int,required=True)
 ap.add_argument('--stage',choices=['pilot','main'],required=True)
 ap.add_argument('--order',default='O0',choices=['O0'])
 ap.add_argument('--max-per-modality',type=int,default=10)
 opts=ap.parse_args()
 os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
 device=f'cuda:{opts.gpu}'
 selection=json.load(open(ROOT/'results/selection.json'))['items']
 # First 10/each for the smoke/pilot. Main runs full list, resumes from pilot by key.
 rng=np.random.default_rng(20261009)
 pilotset=[]
 for modality in ('MRI','CT','Xray'):
  pool=sorted(x['sample_id'] for x in selection if x['modality']==modality)
  pilotset+=list(rng.choice(pool,size=min(opts.max_per_modality,len(pool)),replace=False))
 keep=set(pilotset)
 current=[x for x in selection if opts.stage=='main' or x['sample_id'] in keep]
 out=ROOT/'results'/f'advice_{opts.model}.jsonl'
 done={}
 if out.is_file():
  for line in out.read_text().splitlines():
   if not line.strip():continue
   r=json.loads(line);key=(r['sample_id'],r['condition'],r['hint_letter'])
   assert key not in done,key
   done[key]=r
 expected=len(current)*len(COND)*len(HINTS)
 print('BEGIN',opts.model,opts.stage,'n_items',len(current),'expected_records',expected,'already_done',len(done),'GPU',opts.gpu,flush=True)
 proc=AutoProcessor.from_pretrained(str(V.MODELS[opts.model]))
 proc.image_processor.min_pixels=V.MATCHED_MIN_PIXELS
 proc.image_processor.max_pixels=V.MATCHED_MAX_PIXELS
 if isinstance(getattr(proc.image_processor,'size',None),dict):
  proc.image_processor.size['longest_edge']=V.MATCHED_MAX_PIXELS
  proc.image_processor.size['shortest_edge']=V.MATCHED_MIN_PIXELS
 ids=[proc.tokenizer.encode(letter,add_special_tokens=False)[0] for letter in 'ABCD']
 assert len(set(ids))==4,ids
 model=Qwen2VLForConditionalGeneration.from_pretrained(str(V.MODELS[opts.model]),torch_dtype=torch.bfloat16,
         attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
 print('MODEL_READY',opts.model,'allocated_GB',round(torch.cuda.memory_allocated(opts.gpu)/2**30,3),flush=True)
 its=V.load_items();start=time.time();executed=0
 with out.open('a',encoding='utf8') as f:
  for j,entry in enumerate(current):
   sid=entry['sample_id'];it=its[sid];conf=entry[opts.model+'_hint_confidence_pct']
   for cond in COND:
    for letter in HINTS:
     key=(sid,cond,letter)
     if key in done:continue
     enc,order,prompt=build(model,proc,it,cond,letter,conf,device)
     with torch.inference_mode():
      result=model(**enc,logits_to_keep=1,use_cache=False)
     pred=semantic_score(result.logits[0,-1,ids].float().cpu().numpy(),order,it)
     row={'sample_id':sid,'model':opts.model,'modality':entry['modality'],'condition':cond,
          'hint_letter':letter,'hint_semantic_index':None if letter is None else ord(letter)-65,
          'confidence_pct':conf if letter is not None else None,'source_clean_pmax':entry[opts.model+'_clean_pmax'],
          'prompt_hash':hashlib.sha256(prompt.encode()).hexdigest()[:16],
          'option_order':'O0',**pred}
     f.write(json.dumps(row,ensure_ascii=False)+'\n');executed+=1
     del result,enc
    f.flush()
   if (j+1)%10==0 or j+1==len(current):
    print('PROGRESS',opts.model,j+1,'/',len(current),'new_records',executed,'elapsed_min',round((time.time()-start)/60,2),flush=True)
 print('DONE',opts.model,'new_records',executed,'total_records',len(done)+executed,'elapsed_min',round((time.time()-start)/60,2),flush=True)

if __name__=='__main__':main()
