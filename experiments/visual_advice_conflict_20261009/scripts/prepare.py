#!/usr/bin/env python3
"""Freeze O0 both-Clean-correct selection, per-checkpoint confidence and image cases."""
import csv,json,hashlib,sys
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image,ImageFilter
ROOT=Path(__file__).resolve().parents[1]
MED=ROOT.parent
sys.path.insert(0,str(MED/'visual_reliance_mechanism_v2/scripts'))
import common_v2 as V

scorefile=MED/'briefing_20261009/public_release_20261009/results/official825_scores.csv'
byid=defaultdict(dict)
for r in csv.DictReader(scorefile.open()):
 if r['condition']=='clean' and r['order']=='O0':byid[r['sample_id']][r['model']]=r
items=V.load_items();selection=[]
for sid,rs in sorted(byid.items()):
 if set(rs)!={'qwen2vl','medvlmr1'} or not all(int(rs[m]['correct']) for m in rs):continue
 item=items[sid];assert item['n_options']==4
 entry={'sample_id':sid,'modality':item['modality'],'gt_index':item['gt_index'],'option_order':'O0'}
 for m in ('qwen2vl','medvlmr1'):
  r=rs[m];assert int(r['gt_index'])==item['gt_index']
  x=float(r['p_gt']);assert x>0.25
  c=25.0+0.5*(x*100-25)
  entry[m+'_clean_pmax']=x
  entry[m+'_hint_confidence_pct']=round(c,3)
 selection.append(entry)
assert len(selection)>=400,len(selection)
by_mod=Counter(x['modality'] for x in selection)
# Generate only the new combined condition; existing sources never modified.
for entry in selection:
 sid=entry['sample_id'];mod=entry['modality']
 src=V.image_path('random_same_modality_replacement',items[sid]);assert src.is_file()
 dst=ROOT/'images/blurred_mismatch'/f'{sid}.png'
 if not dst.is_file():
  with Image.open(src) as im:
   sigma=V.OD.blur_sigma(im.size,V.OD.BLUR_FRACTION_MIN_SIDE['severe'])
   out=im.convert('RGB').filter(ImageFilter.GaussianBlur(radius=sigma))
   out.save(dst,format='PNG')
 assert dst.is_file()
 for cond in ('clean','blur_severe','random_same_modality_replacement'):
  assert V.image_path(cond,items[sid]).is_file(),(sid,cond)
sel=ROOT/'results/selection.json'
sel.write_text(json.dumps({'selection_rule':'Both original models correctly predict GT at Clean O0, matched candidate scoring','confidence_rule':'25 + (100*p_clean_O0_max-25)*0.5, each model separately; hold fixed across all visual arms and hinted options','n':len(selection),'modality_counts':dict(by_mod),'items':selection},indent=2)+'\n')
manifest={'generated_combined_condition':'Severe Gaussian blur of cached same-modality random mismatch image','blur_fraction_short_side':V.OD.BLUR_FRACTION_MIN_SIDE['severe'], 'n_images':len(selection), 'sample_ids_sha256':hashlib.sha256('\n'.join(x['sample_id'] for x in selection).encode()).hexdigest()}
(ROOT/'results/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('SELECTION_FROZEN',len(selection),dict(by_mod))
for m in ('qwen2vl','medvlmr1'):
 vals=[x[m+'_hint_confidence_pct'] for x in selection]
 print(m,'confidence_range',round(min(vals),2),round(max(vals),2),'mean',round(sum(vals)/len(vals),2))
print('COMBINED_BLUR_IMAGES_READY',len(selection))
