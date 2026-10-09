#!/usr/bin/env python3
"""Convert official public sources into one portable local JSONL inference manifest.

Original benchmark labels, option text and images stay local (never in git).
This script enforces strict coverage and supports the original fixed PMC-VQA
1000-item answer-stratified test set, not a different random test set.
"""
from __future__ import annotations
import argparse,json,re,zipfile,hashlib,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=20260916
OFFICIAL_SHA='9daad1bfd5c3181a0b91292f70418522d7b96ffc4e2d03e5d8e4ed489646b1c1'

def valid_save(items,dest,expected):
 assert len(items)==expected, f'expected {expected}, found {len(items)}'
 ids=[x['item_id'] for x in items]
 assert len(set(ids))==len(ids),'duplicate prediction IDs'
 for item in items:
  assert len(item['options']) in (2,4) and 0<=int(item['gt_index'])<len(item['options']),item['item_id']
  assert Path(item['image_path']).is_file(),f'missing image: {item["image_path"]}'
 dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text(''.join(json.dumps(it,ensure_ascii=False)+'\n' for it in items),encoding='utf-8')
 print('PREPARED',len(items),'items',len(set(it['image_id'] for it in items)),'unique images',dest)
 print('FIRST LAST',items[0]['item_id'],items[-1]['item_id'])

def extract_zip_entries(zpath:Path, names:list[str],target:Path)->dict[str,Path]:
 if not zpath.exists():raise FileNotFoundError(zpath)
 target.mkdir(parents=True,exist_ok=True)
 result={}
 with zipfile.ZipFile(zpath) as archive:
  names_set=set(archive.namelist())
  missing=set(names)-names_set
  if missing:raise RuntimeError(f'{len(missing)} images not found in {zpath.name}: {list(sorted(missing))[:5]}')
  for inner in sorted(set(names)):
   # Do not trust zip-internal paths: map only to basename with collision check.
   basename=Path(inner).name
   if not basename:raise ValueError('invalid archive member')
   # Different SLAKE studies often share the name source.jpg. Never alias images.
   uniquename=hashlib.sha256(inner.encode('utf-8')).hexdigest()[:12]+'_'+basename
   p=target/uniquename
   if p.is_file() and p.stat().st_size>0:result[inner]=p;continue
   with archive.open(inner) as fi,open(p,'wb') as fo:
    import shutil
    shutil.copyfileobj(fi,fo)
   if p.stat().st_size==0:raise ValueError(f'empty extracted image {inner}')
   result[inner]=p
 return result

def prepare_official(src:Path,out:Path,only_four:bool,json_override:Path|None):
 j=(json_override if json_override is not None else src/'MRI_CT_XRAY_300each_dataset.json')
 if not j.is_file():raise FileNotFoundError(f'Official JSON missing: {j}. Run download_assets.py --datasets official')
 digest=hashlib.sha256(j.read_bytes()).hexdigest()
 if digest!=OFFICIAL_SHA:raise ValueError(f'Official JSON SHA mismatch: {digest}')
 data=json.loads(j.read_text())
 if len(data)!=900:raise ValueError(f'Official JSON expected 900, got {len(data)}')
 imgs=[str(it['image'][0]) for it in data]
 all_paths=extract_zip_entries(src/'images.zip',imgs,out.parent/'images'/'official')
 items=[]
 for i,it in enumerate(data):
  mod='MRI' if i<300 else ('CT' if i<600 else 'Xray');local_i=i if i<300 else (i-300 if i<600 else i-600)
  opts=[str(x) for x in it['options']];letter=str(it['solution']).strip()
  gt=ord(letter)-ord('A')
  if not 0<=gt<len(opts):raise ValueError(f'Invalid label at {i}')
  if only_four and len(opts)!=4:continue
  img=imgs[i]
  items.append({'item_id':f'{mod}_{local_i:03d}','image_id':img,'image_path':str(all_paths[img].resolve()),
                'question':str(it['question']),'options':opts,'gt_index':gt,'group':mod,
                'meta':{'modality':mod,'dataset':'official900','official_index':i}})
 valid_save(items,out,825 if only_four else 900)

def prepare_slake(src:Path,out:Path,split:str):
 data=json.loads((src/f'{split}.json').read_text(encoding='utf-8'))
 records=[r for r in data if r.get('q_lang')=='en' and r.get('answer_type')=='CLOSED'
          and str(r.get('answer','')).strip().lower() in ('yes','no')]
 img=[f'imgs/{r["img_name"]}' for r in records]
 paths=extract_zip_entries(src/'imgs.zip',img,out.parent/'images'/('slake_'+split))
 items=[]
 for r,inner in zip(records,img):
  label=str(r['answer']).strip().lower()
  items.append({'item_id':f'slake_{split}_{r["qid"]}','image_id':str(r['img_name']),
                'image_path':str(paths[inner].resolve()),'question':str(r['question']),
                'options':['Yes','No'],'gt_index':0 if label=='yes' else 1,
                'group':str(r.get('content_type') or 'unknown'),
                'meta':{'modality':str(r.get('modality')),'split':split}})
 valid_save(items,out,355 if split=='test' else 358)

def prepare_pmc(src:Path,out:Path):
 import pandas as pd
 df=pd.read_csv(src/'test_clean.csv')
 cols=['Figure_path','Question','Answer_label']+[f'Choice {x}' for x in 'ABCD']
 for k in cols:
  if k not in df.columns:raise KeyError(k)
 df=df.dropna(subset=cols).reset_index(drop=True)
 clean=lambda x:re.sub(r'^\s*[A-D]\s*[:.)]\s*','',str(x)).strip()
 options=np.array([[clean(r[f'Choice {L}']) for L in 'ABCD'] for _,r in df.iterrows()],dtype=object)
 labels=df['Answer_label'].astype(str).str.strip().str.upper().to_numpy()
 keep_valid=np.isin(labels,list('ABCD'))
 df,options,labels=df[keep_valid].reset_index(drop=True),options[keep_valid],labels[keep_valid]
 gt=np.array([list('ABCD').index(x) for x in labels])
 # Original 2026-09-16 x1_external.py deterministic per-label subsampling.
 rng=np.random.default_rng(SEED)
 selection=[]
 for L in 'ABCD':
  idx=np.flatnonzero(labels==L)
  take=max(1,int(round(1000*len(idx)/len(df))))
  selection += list(rng.choice(idx,size=min(take,len(idx)),replace=False))
 selection=sorted(selection)
 df,options,gt,labels=df.iloc[selection].reset_index(drop=True),options[selection],gt[selection],labels[selection]
 if len(df)!=1000:raise RuntimeError(f'PMC stratified subset expected 1000, got {len(df)}')
 needed={Path(str(v)).name for v in df['Figure_path']}
 # An existing upstream extraction cache is permitted. It saves many minutes and
 # avoids re-reading 20+ GB of ZIP archives on repeat experiments.
 local_cache=src/'_x1_cache'
 local_hit={name:(local_cache/name).resolve() for name in needed if (local_cache/name).is_file() and (local_cache/name).stat().st_size>0}
 if len(local_hit)==len(needed):
  paths=local_hit
  print('Reused existing PMC-VQA source image cache:',len(paths),'figures')
 else:
  print('PMC cache hits:',len(local_hit),'of',len(needed),'scanning archives for remaining figures',flush=True)
  paths=dict(local_hit)
  missing_names=needed-set(paths)
  zips=sorted(src.glob('images*.zip'))
  if not zips:raise FileNotFoundError('Expected PMC images.zip and optional images_2.zip')
  member={}
  for zp in zips:
   with zipfile.ZipFile(zp) as z:
    for name in z.namelist():
     if Path(name).name in missing_names:member.setdefault(Path(name).name,(zp,name))
  missing=missing_names-set(member)
  if missing:raise FileNotFoundError(f'{len(missing)} selected figures missing from PMC zip archives: {sorted(missing)[:5]}')
  cache=out.parent/'images'/'pmcvqa';cache.mkdir(parents=True,exist_ok=True)
  for name,(zp,inner) in member.items():
   p=cache/name
   if not p.exists() or p.stat().st_size==0:
    with zipfile.ZipFile(zp) as z:p.write_bytes(z.read(inner))
   paths[name]=p.resolve()
 items=[]
 for i,r in df.iterrows():
  name=Path(str(r['Figure_path'])).name
  items.append({'item_id':f'pmcvqa_{i}','image_id':str(r['Figure_path']),
                'image_path':str(paths[name]),'question':str(r['Question']).strip(),
                'options':[str(v) for v in options[i]],'gt_index':int(gt[i]),
                'group':str(r['Figure_path']).split('_')[0],
                'meta':{'answer_label':str(labels[i]),'source':'test_clean.csv'}})
 valid_save(items,out,1000)

def prepare_vqarad(src:Path,out:Path):
 import pandas as pd
 d=pd.read_parquet(src/'data/test-00000-of-00001.parquet')
 d=d[d['answer_type']=='CLOSED']
 d=d[d['answer_normalized'].astype(str).str.strip().str.lower().isin(['yes','no'])]
 cache=out.parent/'images'/'vqarad';cache.mkdir(parents=True,exist_ok=True)
 items=[]
 for _,r in d.iterrows():
  name=str(r['image_name']);p=cache/name
  if not p.exists():
   image=r['image'];bytes_=image['bytes'] if isinstance(image,dict) else None
   if not bytes_:raise ValueError(f'No embedded VQA-RAD image bytes for {name}')
   p.write_bytes(bytes_)
  answer=str(r['answer_normalized']).strip().lower()
  items.append({'item_id':f'vqarad_{int(r["qid"])}','image_id':name,
                'image_path':str(p.resolve()),'question':str(r['question']),
                'options':['yes','no'],'gt_index':0 if answer=='yes' else 1,
                'group':str(r['image_organ'])})
 valid_save(items,out,251)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',required=True,choices=['official900','official825','slake_test','slake_val','pmcvqa','vqarad'])
 p.add_argument('--asset-root',type=Path,default=Path('assets'))
 p.add_argument('--source-dir',type=Path,default=None,help='Optional existing dataset directory (to avoid re-downloading)')
 p.add_argument('--official-json',type=Path,default=None,help='Optional existing official 900 JSON')
 p.add_argument('--output',type=Path,default=None,help='Default: <asset-root>/manifests/<dataset>.jsonl')
 a=p.parse_args();root=a.asset_root.resolve()
 source_key=('official' if a.dataset.startswith('official') else 'slake' if a.dataset.startswith('slake') else a.dataset)
 src=a.source_dir.resolve() if a.source_dir else root/'datasets'/source_key
 out=a.output.resolve() if a.output else root/'manifests'/f'{a.dataset}.jsonl'
 if a.dataset.startswith('official'):prepare_official(src,out,a.dataset=='official825',a.official_json)
 elif a.dataset.startswith('slake'):prepare_slake(src,out,'test' if a.dataset=='slake_test' else 'validation')
 elif a.dataset=='pmcvqa':prepare_pmc(src,out)
 elif a.dataset=='vqarad':prepare_vqarad(src,out)
if __name__=='__main__':main()
