#!/usr/bin/env python3
"""End-to-end GPU inference for official 900 quality × relevance (or a small pilot).

Uses the ORIGINAL frozen 900-item donor assignments from configs/official900_donors.json.
Conditions: clean, no_image, blank, blur_mild/medium/severe, pm_<quality>_k1/2/3,
sc_<quality> (semantic conflict, only 375 label-supported items).

GPU inference is potentially lengthy: 21+ arms × 3 option orders × 2 checkpoints.
Results saved one record per (item, arm, order) and can resume after interruption.
Model/processor/prompt and option normalization mirror the original frozen scorer;
quality option-order seed is 20260914, different from external datasets.
"""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
from PIL import Image,ImageFilter
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_external as E

BLUR={'mild':0.005,'medium':0.015,'severe':0.030}
DEFAULT=['clean','no_image','blank']+[f'blur_{q}' for q in BLUR]+[f'pm_{q}_k{k}' for q in ('clean',*BLUR) for k in (1,2,3)]+[f'sc_{q}' for q in ('clean',*BLUR)]

def load_items(path):
 return E.load_manifest(path)

def transform_image(source,level,cache:Path):
 if level=='clean':return source
 source=Path(source)
 key=hashlib.sha256((str(source.resolve())+'|'+level+'|'+str(source.stat().st_size)).encode()).hexdigest()[:16]
 out=cache/f'{key}.png'
 if out.is_file() and out.stat().st_size>0:return str(out)
 cache.mkdir(parents=True,exist_ok=True)
 with Image.open(source) as im:
  im=im.convert('RGB')
  if level=='blank': im=Image.new('RGB',im.size,'white')
  else:im=im.filter(ImageFilter.GaussianBlur(radius=BLUR[level]*min(im.size)))
  im.save(out)
 return str(out)

def determine_image(item,arm,lookup,donors,cache):
 if arm=='no_image':return None
 if arm=='blank':return transform_image(item['image_path'],'blank',cache)
 if arm=='clean':return item['image_path']
 if arm.startswith('blur_'):return transform_image(item['image_path'],arm.removeprefix('blur_'),cache)
 if arm.startswith('pm_'):
  q,k=arm.removeprefix('pm_').split('_k')
  donor=donors['derangements']['D'+k][item['item_id']]
  return transform_image(lookup[donor]['image_path'],q,cache)
 if arm.startswith('sc_'):
  conf=donors['semantic_conflict_label_supported_not_clinician_verified']
  if item['item_id'] not in conf:return None
  donor=conf[item['item_id']]['donor']
  q=arm.removeprefix('sc_')
  return transform_image(lookup[donor]['image_path'],q,cache)
 raise ValueError('Unknown image arm: '+arm)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--manifest',type=Path,required=True,help='Full official900 manifest produced by prepare_datasets.py')
 ap.add_argument('--donors',type=Path,default=Path('configs/official900_donors.json'))
 ap.add_argument('--checkpoint',type=Path)
 ap.add_argument('--output',type=Path,default=Path('outputs/quality.jsonl'))
 ap.add_argument('--cache-dir',type=Path,default=Path('outputs/degraded_image_cache'))
 ap.add_argument('--device',default='cuda:0')
 ap.add_argument('--arms',default=','.join(DEFAULT))
 ap.add_argument('--limit',type=int,default=0)
 ap.add_argument('--max-orders',type=int,choices=[1,2,3],default=3)
 ap.add_argument('--dry-run',action='store_true')
 args=ap.parse_args()
 items=load_items(args.manifest)
 all_ids={it['item_id'] for it in items}
 if len(items)!=900 and not args.limit:raise ValueError('Full quality run requires full 900 manifest; use --limit only for smoke')
 donors=json.loads(args.donors.read_text())
 if len(items)==900:
  for k,v in donors['derangements'].items():
   assert set(v)==all_ids==set(v.values())
 lookup={it['item_id']:it for it in items}
 arms=[a.strip() for a in args.arms.split(',') if a.strip()]
 assert len(arms)==len(set(arms)) and all(a in DEFAULT for a in arms)
 if args.limit:items=items[:args.limit]
 if args.dry_run:
  print('QUALITY PLAN VALID',len(items),'items',len(arms),'arms',args.max_orders,'orders')
  print('ARM_NAMES',','.join(arms));return
 if not args.checkpoint:ap.error('Supply --checkpoint to run GPU inference')
 E.SEED=20260914  # frozen quality control uses original official900 order seed
 model,proc,token_ids=E.load_checkpoint(args.checkpoint,args.device)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 done=set()
 if args.output.exists():
  for line in args.output.open():
   try:
    r=json.loads(line);done.add((r['item_id'],r['condition'],r['order']))
   except json.JSONDecodeError:pass
 with args.output.open('a') as f:
  for i,it in enumerate(items):
   for arm in arms:
    if arm.startswith('sc_') and it['item_id'] not in donors['semantic_conflict_label_supported_not_clinician_verified']:continue
    ids=['O0','O1','O2'][:args.max_orders]
    needed=[o for o in ids if (it['item_id'],arm,o) not in done]
    if not needed:continue
    img=determine_image(it,arm,lookup,donors,args.cache_dir)
    if img is not None and not Path(img).is_file():raise FileNotFoundError(img)
    for o in needed:
     perm=E.option_order(len(it['options']),o,it['item_id'])
     p=E.score(model,proc,token_ids,it,img,perm,args.device)
     g=int(it['gt_index']);wrong=[x for j,x in enumerate(p) if j!=g]
     pg=float(p[g]);log_margin=float(np.log(max(pg,1e-300))-np.log(max(wrong)))
     entropy=-sum(float(x)*np.log(max(float(x),1e-300)) for x in p)
     rec={'item_id':it['item_id'],'image_id':it['image_id'],'modality':it.get('group'),
          'condition':arm,'order':o,'n_options':len(p),'gt_index':g,
          'semantic_probs':p,'correct':int(np.argmax(p)==g),'p_gt':pg,
          'gt_prob_margin':pg-max(wrong),'gt_log_margin':log_margin,
          'entropy':float(entropy),'processing':'min3136-max401408','order_seed':E.SEED}
     f.write(json.dumps(rec)+'\n');f.flush()
   if (i+1)%25==0:print('QUALITY',i+1,'/',len(items),flush=True)
 print('QUALITY COMPLETE',args.output)
if __name__=='__main__':main()
