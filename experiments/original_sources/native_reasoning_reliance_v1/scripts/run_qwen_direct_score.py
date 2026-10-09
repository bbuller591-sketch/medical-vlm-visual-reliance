#!/usr/bin/env python
from __future__ import annotations
import argparse,json,math,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; MED=ROOT.parent
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(MED/'quality_reliance_v1'/'scripts'))
import common_qr as Q
from run_native import NativeRunner,SIMPLE_TEMPLATE,resolve_image

def stable_int(*parts): return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')
def blank(item,kind,max_pixels):
 src=Image.open(Q.source_image_path(item)).convert('RGB');w,h=src.size;scale=min(1.0,math.sqrt(max_pixels/max(1,w*h)));nw,nh=max(8,int(round(w*scale))),max(8,int(round(h*scale)));src=src.resize((nw,nh),Image.Resampling.BILINEAR);a=np.asarray(src,dtype=np.float32);mean=a.reshape(-1,3).mean(0)
 if kind=='blank_white': arr=np.full((nh,nw,3),255,dtype=np.uint8)
 else:
  v=np.clip(np.round(mean),0,255).astype(np.uint8);arr=np.broadcast_to(v,(nh,nw,3)).copy()
 return Image.fromarray(arr,'RGB')
def noise(item,seed,max_pixels):
 src=Image.open(Q.source_image_path(item)).convert('RGB');w,h=src.size;scale=min(1.0,math.sqrt(max_pixels/max(1,w*h)));nw,nh=max(8,int(round(w*scale))),max(8,int(round(h*scale)));src=src.resize((nw,nh),Image.Resampling.BILINEAR);a=np.asarray(src,dtype=np.float32);mean=a.reshape(-1,3).mean(0);std=np.maximum(a.reshape(-1,3).std(0),12.0);rng=np.random.default_rng(stable_int('qwen_direct_noise_v1',item['sample_id'],seed));arr=np.clip(rng.normal(mean,std,size=a.shape),0,255).astype(np.uint8);return Image.fromarray(arr,'RGB')
def score(runner,prompt,image,letters):
 enc,chat_text,ii,vi=runner._encode_message(prompt,image)
 with runner.torch.inference_mode(): out=runner.model(**enc,use_cache=False)
 logits=out.logits[0,-1].float();lpv=runner.torch.log_softmax(logits,dim=-1); cand={}; tids={}
 for L in letters:
  ids=runner.tok.encode(L,add_special_tokens=False)
  if len(ids)!=1:return {'status':'candidate_tokenization_fail','candidate':L,'candidate_token_ids':ids}
  cand[L]=float(lpv[ids[0]]);tids[L]=ids[0]
 z=max(cand.values());den=sum(math.exp(v-z) for v in cand.values());probs={k:math.exp(v-z)/den for k,v in cand.items()};top=max(probs,key=probs.get)
 return {'status':'ok','candidate_logprobs':cand,'candidate_probs':probs,'pred_letter_from_probs':top,'candidate_token_ids':tids,'n_visual_tokens':int((enc.get('image_grid_thw').prod(-1)//((getattr(runner.processor.image_processor,'merge_size',1) or 1)**2)).sum()) if enc.get('image_grid_thw') is not None else 0}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ids',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=5);ap.add_argument('--worker',required=True);ap.add_argument('--noise-seeds',type=int,default=10);ap.add_argument('--arms',default='clean,no_image,blank_white,blank_mean,pm_clean_k1,sc_clean,blur_mild,blur_mid_extra,blur_medium,blur_severe,sc_mild,sc_medium,sc_severe');args=ap.parse_args()
 items=Q.load_items();ids=[x.strip() for x in open(args.ids) if x.strip()];ids=sorted(ids)[args.shard::args.nshards]
 arms=[a.strip() for a in args.arms.split(',') if a.strip()]
 out=ROOT/'results'/'qwen_direct';out.mkdir(parents=True,exist_ok=True);p=out/f'QWENDIRECT__{args.worker}.jsonl';done=set()
 if p.exists():
  for ln in p.open():
   try:r=json.loads(ln);done.add((r['sample_id'],r['arm'],r.get('noise_seed')))
   except:pass
 runner=NativeRunner('qwen2vl',args.device);ok=err=miss=0;t0=time.time()
 with p.open('a',encoding='utf-8') as fw:
  for sid in ids:
   it=items[sid];prompt=SIMPLE_TEMPLATE.format(problem=it['problem']); jobs=[(a,None) for a in arms]+[('pure_noise_ms',s) for s in range(args.noise_seeds)]
   for arm,seed in jobs:
    if (sid,arm,seed) in done:continue
    rec={'sample_id':sid,'model':'qwen2vl','arm':arm,'noise_seed':seed,'gt_letter':it['gt_letter'],'gt_text':it['gt_text'],'n_options':it['n_options'],'prompt_type':'direct_candidate_score_v1','timestamp':Q.now()}
    try:
     if arm=='blank_white' or arm=='blank_mean': im=blank(it,arm,Q.MATCHED_MAX_PIXELS)
     elif arm=='pure_noise_ms': im=noise(it,seed,Q.MATCHED_MAX_PIXELS)
     else:
      im,_=resolve_image(arm,it,Q.MATCHED_MAX_PIXELS)
     if arm.startswith('sc_') and im is None: rec['status']='missing_donor';miss+=1
     else:
      rr=score(runner,prompt,im,it['letters']);rec.update(rr)
      if rr['status']=='ok':
       gt=it['gt_letter'];wrong=[L for L in it['letters'] if L!=gt];pr=rr['candidate_probs'];lp=rr['candidate_logprobs'];rec['p_gt']=pr[gt];rec['gt_log_margin']=lp[gt]-max(lp[L] for L in wrong);rec['gt_prob_margin']=pr[gt]-max(pr[L] for L in wrong);rec['entropy']=-sum(q*math.log(max(q,1e-15)) for q in pr.values());rec['pred_correct']=rr['pred_letter_from_probs']==gt;ok+=1
    except Exception as e: rec.update(status='error',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()[-1200:]);err+=1
    fw.write(json.dumps(rec,ensure_ascii=False)+'\n');fw.flush()
    n=ok+err+miss
    if n and n%100==0:print(json.dumps({'worker':args.worker,'done':n,'ok':ok,'missing':miss,'error':err,'elapsed_min':round((time.time()-t0)/60,2)}),flush=True)
 print(json.dumps({'worker':args.worker,'ok':ok,'missing':miss,'error':err,'elapsed_min':(time.time()-t0)/60}),flush=True)
if __name__=='__main__':main()
