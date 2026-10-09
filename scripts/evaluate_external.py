#!/usr/bin/env python3
"""Standalone external MCQ scoring for released Qwen2-VL and MedVLM-R1 models.

Requires locally obtained checkpoints and medical benchmark images. Do not commit
checkpoint weights, any raw images, or hospital/patient data to this repository.

Manifest JSONL: {"item_id": "...", "image_id": "...", "image_path": "...",
"question": "...", "options": ["...", ...], "gt_index": 0, "group": "..."}.

Output: one JSONL record per (item, input condition, option order), containing
probabilities in *semantic* choice order. Use a frozen donor-plan JSON across
both checkpoints. This runner reproduces the scoring/inversion rules used in
x1_external.py, but raw model-inference reproduction has not been retested in
this packaging step; see README for verified reproducibility level.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np

SEED=20260916
MIN_PIXELS=3136
MAX_PIXELS=401408

def stable_seed(*parts):
 return int(hashlib.sha256('|'.join(map(str,parts)).encode('utf-8')).hexdigest()[:16],16)

def option_order(n,order_id,item_id):
 base=list(range(n))
 if order_id=='O0':return base
 rot=base[1:]+base[:1]
 if order_id=='O1':return rot
 if order_id=='O2':
  rng=np.random.default_rng(stable_seed('option_order',item_id,order_id,SEED))
  p=base[:]
  for _ in range(200):
   rng.shuffle(p)
   if p!=base and p!=rot:return p
  return rot if rot!=base else base
 raise ValueError('unknown option order '+order_id)

def load_manifest(path):
 items=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
 assert len(items)>1,'Need at least two items'
 ids=[it['item_id'] for it in items]
 assert len(ids)==len(set(ids)),'Duplicate item_id'
 for it in items:
  assert 2<=len(it['options'])<=4
  assert all(isinstance(v,str) and v.strip() for v in it['options'])
  assert 0<=int(it['gt_index'])<len(it['options'])
  for k in ['image_path','image_id','question']:assert it.get(k)
 return items

def build_or_load_donor_plan(items,path):
 path=Path(path)
 ids=[it['item_id'] for it in items]
 if path.exists():
  data=json.loads(path.read_text())
  assert data.get('ordered_item_ids')==ids,'Donor plan no longer matches manifest ordering!'
  assert data.get('seed')==SEED
  donor=data['donor_item_by_target']
 else:
  n=len(items);rng=np.random.default_rng(SEED);order=rng.permutation(n);perm=np.empty(n,dtype=int)
  perm[order]=np.roll(order,1)
  donor={ids[i]:ids[int(perm[i])] for i in range(n)}
  data={'seed':SEED,'ordered_item_ids':ids,'donor_item_by_target':donor,
        'note':'single-cycle shuffled pairing; preserve this exact plan for both model runs'}
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2))
 assert set(donor)==set(ids)
 for target,source in donor.items():assert target!=source and source in ids
 return donor

def load_checkpoint(checkpoint,device):
 import torch
 import transformers
 from transformers import AutoProcessor
 path=Path(checkpoint);info=json.loads((path/'config.json').read_text())
 mtype=info.get('model_type')
 classes={'qwen2_vl':'Qwen2VLForConditionalGeneration',
          'qwen2_5_vl':'Qwen2_5_VLForConditionalGeneration'}
 assert mtype in classes,f'Unsupported model type: {mtype}'
 cls=getattr(transformers,classes[mtype])
 proc=AutoProcessor.from_pretrained(str(path))
 proc.image_processor.min_pixels=MIN_PIXELS
 proc.image_processor.max_pixels=MAX_PIXELS
 tok=proc.tokenizer
 label_ids={l:tok.encode(l,add_special_tokens=False) for l in 'ABCD'}
 assert all(len(v)==1 for v in label_ids.values()),'candidate labels require single token'
 model=cls.from_pretrained(str(path),torch_dtype=torch.bfloat16,
                           attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
 return model,proc,{k:v[0] for k,v in label_ids.items()}

def score(model,proc,tok_ids,item,image_path,perm,device):
 import torch
 from qwen_vl_utils import process_vision_info
 n=len(item['options']);letters=list('ABCD'[:n])
 mcq=item['question']+'\n'+'\n'.join(f'{L}) {item["options"][perm[k]]}' for k,L in enumerate(letters))
 prompt=mcq+'\n\nAnswer with the single letter of the correct option. Reply with only that letter.'
 content=([{'type':'image','image':str(image_path)}] if image_path is not None else [])+[{'type':'text','text':prompt}]
 messages=[{'role':'user','content':content}]
 text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 if image_path is None:enc=proc(text=[text],return_tensors='pt',padding=True)
 else:
  img,vid=process_vision_info(messages)
  enc=proc(text=[text],images=img,videos=vid,return_tensors='pt',padding=True)
 enc={k:v.to(device) if torch.is_tensor(v) else v for k,v in enc.items()}
 with torch.inference_mode():logits=model(**enc,logits_to_keep=1).logits[0,-1,:].float()
 z=np.array([float(logits[tok_ids[l]]) for l in letters],dtype=np.float64)
 ex=np.exp(z-z.max());p_display=ex/ex.sum()
 p_sem=np.zeros(n)
 for displayed,semantic in enumerate(perm):p_sem[semantic]=p_display[displayed]
 return [float(v) for v in p_sem]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--manifest',required=True,help='JSONL file of local benchmark questions and image paths')
 ap.add_argument('--donor-plan',required=True,help='Frozen donor JSON; share across both model runs')
 ap.add_argument('--checkpoint',help='Local directory containing model + processor')
 ap.add_argument('--device',default='cuda:0')
 ap.add_argument('--output',default='evaluation_records.jsonl')
 ap.add_argument('--dry-run',action='store_true',help='Only validate inputs and freeze donor pairing')
 ap.add_argument('--limit',type=int,default=0,help='Smoke test only; freezes donors on FULL manifest first')
 ap.add_argument('--max-orders',type=int,choices=[1,2,3],default=3)
 ap.add_argument('--conditions',default='clean,no_image,shuffled_image')
 ap.add_argument('--resume',action='store_true',help='Append missing item/condition/order combinations')
 args=ap.parse_args()
 items=load_manifest(args.manifest);donor=build_or_load_donor_plan(items,args.donor_plan)
 lookup={it['item_id']:it for it in items}
 if args.limit:items=items[:args.limit]
 nopts={len(it['options']) for it in items}
 assert len(nopts)==1,'One run must use a fixed choice count'
 if args.dry_run:
  print('DRY RUN OK',len(items),'items',len({it['image_id'] for it in items}),'images',len(donor),'different donor items')
  return
 assert args.checkpoint,'--checkpoint required for GPU inference'
 model,proc,tok_ids=load_checkpoint(args.checkpoint,args.device)
 orders=(['O0','O1'] if next(iter(nopts))==2 else ['O0','O1','O2'])[:args.max_orders]
 conditions=[x.strip() for x in args.conditions.split(',') if x.strip()]
 assert conditions and all(c in ('clean','no_image','shuffled_image') for c in conditions)
 outfile=Path(args.output);outfile.parent.mkdir(parents=True,exist_ok=True)
 completed=set()
 if args.resume and outfile.exists():
  for line in outfile.open():
   try:
    x=json.loads(line);completed.add((x['item_id'],x['condition'],x['order']))
   except json.JSONDecodeError:pass
 with outfile.open('a' if args.resume else 'w',encoding='utf-8') as f:
  for index,it in enumerate(items):
   for cond in conditions:
    if all((it['item_id'],cond,o) in completed for o in orders):continue
    image_path=(it['image_path'] if cond=='clean' else
                lookup[donor[it['item_id']]]['image_path'] if cond=='shuffled_image' else None)
    if image_path is not None and not Path(image_path).is_file():raise FileNotFoundError(image_path)
    for o in orders:
     if (it['item_id'],cond,o) in completed:continue
     perm=option_order(len(it['options']),o,it['item_id'])
     p=score(model,proc,tok_ids,it,image_path,perm,args.device)
     g=int(it['gt_index'])
     rec={'item_id':it['item_id'],'image_id':it['image_id'],'condition':cond,
          'order':o,'semantic_probs':p,'correct':int(int(np.argmax(p))==g),
          'gt_index':g,'donor_item_id':donor[it['item_id']] if cond=='shuffled_image' else None}
     f.write(json.dumps(rec)+'\n');f.flush()
   if (index+1)%50==0:print('completed',index+1,'/',len(items),flush=True)
 print('Saved',outfile)
if __name__=='__main__':main()
