#!/usr/bin/env python3
"""Strict file integrity, baseline parity and result completeness checks."""
from pathlib import Path
import csv,json,collections,sys
ROOT=Path(__file__).resolve().parents[1];MED=ROOT.parent
select=json.load(open(ROOT/'results/selection.json'))
items={r['sample_id']:r for r in select['items']};assert len(items)==559
assert sum(int(r['modality']=='MRI') for r in items.values())==207
assert sum(int(r['modality']=='CT') for r in items.values())==160
assert sum(int(r['modality']=='Xray') for r in items.values())==192
src={}
for r in csv.DictReader(open(MED/'briefing_20261009/public_release_20261009/results/official825_scores.csv')):
 if r['order']=='O0' and r['condition']=='clean':src[(r['model'],r['sample_id'])]=r
for m in ('qwen2vl','medvlmr1'):
 path=ROOT/'results'/f'advice_{m}.jsonl'
 data=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 ids=collections.Counter((d['sample_id'],d['condition'],d['hint_letter']) for d in data)
 assert max(ids.values())==1
 for r in data:
  s=items[r['sample_id']]
  assert 25<=s[m+'_hint_confidence_pct']<=62.5
  assert r['modality']==s['modality']
  assert r['gt_index']==s['gt_index']
  assert len(r['probs_semantic'])==4
  assert abs(sum(r['probs_semantic'])-1)<1e-6
  if r['hint_letter'] is not None:
   assert abs(r['confidence_pct']-s[m+'_hint_confidence_pct'])<1e-6
   assert r['hint_semantic_index']==ord(r['hint_letter'])-65
  if r['condition']=='clean' and r['hint_letter'] is None:
   old=src[m,r['sample_id']]
   assert r['prediction']==int(old['pred_index'])
   assert abs(r['p_gt']-float(old['p_gt']))<1e-5,(m,r['sample_id'],r['p_gt'],old['p_gt'])
 complete=all(ids[k]==1 for k in ((s,c,h) for s in items for c in ('clean','blur_severe','random_same_modality_replacement','blurred_mismatch','no_image') for h in [None,'A','B','C','D']))
 print('MODEL',m,'RECORDS',len(data),'DUPLICATES',sum(count-1 for count in ids.values()),'COMPLETE',complete,'EXPECTED',13975)
 if '--complete' in sys.argv:assert complete and len(data)==13975
print('PREPARED_SELECTION_AND_BASELINES_CHECKED')
