#!/usr/bin/env python3
"""Held-out quality/content steering plus complete-state within-sample interchange.

Frozen pretrained checkpoints and matched image processor from 2026-09-16 V2.
No training of the VLM. Directions are means over disjoint train subjects.
A single intervention changes the decision-token decoder residual at layer k.
"""
import argparse,csv,json,os,sys,time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
import torch
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
BASE=MED/'visual_reliance_mechanism_v2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'scripts'))
import common_v2 as V
import m2_extract as X

CONDITIONS=('clean','blur_severe','random_same_modality_replacement')

def load_states(model):
    states={}
    for cond in CONDITIONS:
        cache={}
        for f in sorted((BASE/'features'/model/cond/'O0').glob('shard_*.npz')):
            with np.load(f) as z:
                for sid,hidden in zip(z['sample_ids'],z['decision_hidden']):
                    cache[str(sid)]=hidden
        states[cond]=cache
    return states

def safe_probability(logits,options,permutation,gt):
    l=logits[:options].astype(np.float64)
    e=np.exp(l-l.max());p=e/e.sum()
    ps=np.zeros(options,dtype=float)
    for letter_idx,sem_idx in enumerate(permutation[:options]):ps[sem_idx]=p[letter_idx]
    wrong=np.delete(ps,gt)
    return {'log_margin':float(np.log(max(ps[gt],1e-300))-np.log(max(wrong.max(),1e-300))),
            'prob_margin':float(ps[gt]-wrong.max()),'p_gt':float(ps[gt]),
            'entropy':float(-(ps*np.log(np.clip(ps,1e-300,None))).sum()),
            'correct':int(ps.argmax()==gt),'prediction':int(ps.argmax())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=list(V.MODELS),required=True)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--max-eval-per-modality',type=int,default=15)
    ap.add_argument('--max-train-per-modality',type=int,default=150)
    ap.add_argument('--layers',default='20,27')
    ap.add_argument('--output',default='')
    opt=ap.parse_args()
    seed=20261009
    layers=[int(x) for x in opt.layers.split(',')]
    if any(x<1 or x>=28 for x in layers):raise ValueError('layers must be 1..27')
    output=Path(opt.output) if opt.output else ROOT/'results'/f'steering_{opt.model}.jsonl'
    output.parent.mkdir(parents=True,exist_ok=True)
    meta_path=ROOT/'results'/f'metadata_{opt.model}.json'
    if output.exists():raise RuntimeError('Output already exists: '+str(output))
    print('MODEL',opt.model,'GPU',opt.gpu,'TIME',time.ctime(),flush=True)
    all_items=V.load_items()
    states=load_states(opt.model)
    elig=set(all_items)
    for s in states.values():elig.intersection_update(s)
    rng=np.random.default_rng(seed)
    tr=[];ev=[]
    for mod in ['MRI','CT','Xray']:
        pool=sorted(s for s in elig if all_items[s]['modality']==mod)
        perm=rng.permutation(len(pool))
        tr += [pool[i] for i in perm[:opt.max_train_per_modality]]
        ev += [pool[i] for i in perm[opt.max_train_per_modality:opt.max_train_per_modality+opt.max_eval_per_modality]]
    assert len(tr)==opt.max_train_per_modality*3 and len(ev)==opt.max_eval_per_modality*3
    assert not (set(tr)&set(ev))
    rng=np.random.default_rng(seed+1)
    # Direction calibration is independent of all held-out inference outputs.
    directions={};direction_diagnostics={}
    for k in layers:
        quality=np.stack([states['blur_severe'][sid][k].astype('float32')-states['clean'][sid][k].astype('float32') for sid in tr])
        content=np.stack([states['random_same_modality_replacement'][sid][k].astype('float32')-states['clean'][sid][k].astype('float32') for sid in tr])
        q=quality.mean(0);c=content.mean(0)
        # Shrink mean direction to 50% of the median observed within-sample displacement.
        # Fixed without inspecting held-out predictions; moderate magnitude helps
        # avoid turning generic corruptions into giant answer-state overrides.
        typical=float(np.median(np.linalg.norm(quality,axis=1)))
        q=q/(np.linalg.norm(q)+1e-12)*typical*.5
        c=c/(np.linalg.norm(c)+1e-12)*typical*.5
        random_vec=rng.normal(size=q.shape).astype('float32')
        random_vec=random_vec/(np.linalg.norm(random_vec)+1e-12)*np.linalg.norm(q)
        heldout_deltas=np.stack([states['blur_severe'][sid][k].astype('float32')-states['clean'][sid][k].astype('float32') for sid in ev])
        proj=heldout_deltas@q
        directions[k]={'quality':q,'mismatch':c,'random':random_vec}
        direction_diagnostics[k]={'typical_training_quality_delta_norm':typical,
                                  'intervention_norm':float(np.linalg.norm(q)),
                                  'heldout_quality_direction_alignment_fraction':float(np.mean(proj>0)),
                                  'training_count':len(tr)}
    meta={'model':opt.model,'seed':seed,'layers':layers,'train_n':len(tr),'eval_n':len(ev),
          'train_ids':tr,'eval_ids':ev,'test_split_method':'random per modality independent of outcomes',
          'representation':'decision token decoder residual, matched processor O0',
          'patching':'only within checkpoint; no model weights changed',
          'quality_direction':'mean(blur_severe - clean) over train, normalized to half median train per-sample delta norm',
          'mismatch_direction':'mean(random_same_modality_replacement - clean) over train, same norm as quality',
          'random_direction':'isotropic random vector with same norm as quality direction',
          'direction_diagnostics':direction_diagnostics,
          'gpu':opt.gpu,'start':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    device=f'cuda:{opt.gpu}'
    proc=AutoProcessor.from_pretrained(str(V.MODELS[opt.model]))
    proc.image_processor.min_pixels=V.MATCHED_MIN_PIXELS
    proc.image_processor.max_pixels=V.MATCHED_MAX_PIXELS
    if hasattr(proc.image_processor,'size') and isinstance(proc.image_processor.size,dict):
        proc.image_processor.size['longest_edge']=V.MATCHED_MAX_PIXELS
        proc.image_processor.size['shortest_edge']=V.MATCHED_MIN_PIXELS
    cand_ids=[proc.tokenizer.encode(x,add_special_tokens=False)[0] for x in 'ABCD']
    print('loading model, peak allocated before:',round(torch.cuda.memory_allocated(opt.gpu)/2**30,2),'GB',flush=True)
    model=Qwen2VLForConditionalGeneration.from_pretrained(str(V.MODELS[opt.model]),torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
    decoder=X._resolve(model,['model.language_model.layers','model.layers','language_model.layers'],'decoder layers')[0]
    assert len(decoder)==28,len(decoder)
    print('model loaded: allocated',round(torch.cuda.memory_allocated(opt.gpu)/2**30,2),'GB',flush=True)
    patch={'v':None}
    def hook(_m,_inp,out):
        if patch['v'] is None:return out
        h=out[0].clone()
        h[:,-1,:]=torch.as_tensor(patch['v'],device=h.device,dtype=h.dtype)
        return (h,)+tuple(out[1:])
    def inference(enc,order,item,k=None,v=None):
        handles=[]
        try:
            if k is not None:
                patch['v']=v
                handles.append(decoder[k-1].register_forward_hook(hook))
            with torch.no_grad():
                output=model(**enc,logits_to_keep=1,use_cache=False)
            lg=output.logits[0,-1,cand_ids].float().cpu().numpy()
            return safe_probability(lg,item['n_options'],order['permutation'],item['gt_index'])
        finally:
            for h in handles:h.remove()
            patch['v']=None
    # self-patch identity verifies using saved same-condition state at exact row.
    sid=ev[0];it=all_items[sid]
    enc,order,_,error=X.build_inputs(model,proc,it,'clean','O0',device)
    assert not error,error
    ref=inference(enc,order,it)
    for k in layers:
        ident=inference(enc,order,it,k,states['clean'][sid][k].astype('float32'))
        delta=ident['log_margin']-ref['log_margin']
        print('IDENTITY',k,'delta log margin',delta,flush=True)
        if abs(delta)>0.02:raise AssertionError(('self-patch failed',k,delta))
    del enc
    torch.cuda.empty_cache()
    count=0;t0=time.monotonic()
    with output.open('w',encoding='utf-8') as f:
        for ix,sid in enumerate(ev):
            item=all_items[sid]
            for cond in CONDITIONS:
                enc,ordered,_,err=X.build_inputs(model,proc,item,cond,'O0',device)
                if err:raise RuntimeError(err)
                baseline=inference(enc,ordered,item)
                row={'sample_id':sid,'modality':item['modality'],'condition':cond,
                     'layer':0,'intervention':'baseline','method':'none',**baseline}
                f.write(json.dumps(row)+'\n');count+=1
                for k in layers:
                    vectors=directions[k]
                    current=states[cond][sid][k].astype('float32')
                    patch_conditions=[('quality_shift',current+vectors['quality']),
                         ('random_shift',current+vectors['random'])]
                    if cond=='clean':
                        patch_conditions += [('mismatch_shift',current+vectors['mismatch']),
                             ('full_blur_state',states['blur_severe'][sid][k].astype('float32')),
                             ('full_mismatch_state',states['random_same_modality_replacement'][sid][k].astype('float32'))]
                    elif cond=='blur_severe':
                        patch_conditions += [('reverse_quality_shift',current-vectors['quality']),
                             ('full_clean_state',states['clean'][sid][k].astype('float32'))]
                    else:
                        patch_conditions += [('reverse_mismatch_shift',current-vectors['mismatch']),
                             ('full_clean_state',states['clean'][sid][k].astype('float32'))]
                    for label,vec in patch_conditions:
                        r=inference(enc,ordered,item,k,vec)
                        row={'sample_id':sid,'modality':item['modality'],'condition':cond,
                             'layer':k,'intervention':label,'method':'full-patch' if label.startswith('full_') else 'directional',**r}
                        f.write(json.dumps(row)+'\n');count+=1
                del enc
                f.flush()
            if (ix+1)%5==0:
                print('PROGRESS',ix+1,'/',len(ev),'records',count,'sec',round(time.monotonic()-t0,1),flush=True)
        f.flush()
    meta['end']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    meta['output_records']=count;meta['elapsed_seconds']=round(time.monotonic()-t0,2)
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print('DONE',opt.model,count,'records','seconds',round(time.monotonic()-t0,2),flush=True)
if __name__=='__main__':main()
