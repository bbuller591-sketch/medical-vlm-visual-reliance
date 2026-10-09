#!/usr/bin/env python3
"""Unfiltered MedVLM-R1/Qwen clean-correct intersection and intervention retention.
Only original MedVLM-R1 official900 4-option items; no DeepSeek labels or filtering.
The selection criterion is fixed as BOTH CLEAN CORRECT (Med generated; Qwen direct).
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,glob,json,math,random,statistics
import numpy as np

ROOT=Path('<LOCAL_PROJECT_ROOT>/medical')
OUT=ROOT/'briefing_20261009'
OUT.mkdir(exist_ok=True,parents=True)
BASE=ROOT/'native_reasoning_reliance_v1'
items={x['sample_id']:x for x in json.loads((ROOT/'official900_visual_reliance_v1/manifests/dataset_manifest.json').read_text())['items'] if x['n_options']==4}
assert len(items)==825

def read_records(globs):
 out={}
 for pat in globs:
  for path in glob.glob(str(pat)):
   for line in open(path):
    try:r=json.loads(line)
    except:continue
    key=(r['sample_id'],r.get('arm'))
    if key not in out or (out[key].get('status')!='ok' and r.get('status')=='ok'):
     out[key]=r
 return out

med=read_records([BASE/'results/raw/NATIVE__med_canon_s*.jsonl',
                  BASE/'results/raw/NATIVE__med_blurmid_s*.jsonl',
                  BASE/'results/blank_controls/BLANK__blank_smoke.jsonl'])
qwen=read_records([BASE/'results/qwen_direct/QWENDIRECT__all4_core.jsonl',
                   BASE/'results/qwen_direct/QWENDIRECT__blank_full825_s*.jsonl'])
arms=['clean','no_image','blank_white','blank_mean','pure_noise','pm_clean_k1','sc_clean',
      'blur_mild','blur_mid_extra','blur_medium','blur_severe','sc_mild','sc_medium','sc_severe']
arm_names={'clean':'Clean','no_image':'No-image','blank_white':'Blank white image',
'blank_mean':'Blank mean-color image','pure_noise':'Pure noise (Med only)',
'pm_clean_k1':'Plausible mismatch image','sc_clean':'Semantic-conflict image',
'blur_mild':'Mild blur','blur_mid_extra':'Intermediate blur (Med only)',
'blur_medium':'Medium blur','blur_severe':'Severe blur',
'sc_mild':'Semantic-conflict (mild blur)','sc_medium':'Semantic-conflict (medium blur)',
'sc_severe':'Semantic-conflict (severe blur)'}
def correct(r,model):
 return bool(r['generated_correct'] if model=='med' else r['pred_correct'])
def getvalid(dic,sid,arm):
 r=dic.get((sid,arm))
 return r if r is not None and r.get('status')=='ok' else None
clean_both={sid for sid in items if getvalid(med,sid,'clean') and getvalid(qwen,sid,'clean') and correct(med[sid,'clean'],'med') and correct(qwen[sid,'clean'],'qwen')}
assert len(clean_both)==506
rng=np.random.default_rng(900506)

def ci_means(vals,reps=6000):
 if not vals:return [None,None,None]
 a=np.asarray(vals,dtype=float)
 if len(a)==1:return [round(float(a.mean()),6),None,None]
 samples=a[rng.integers(0,len(a),size=(reps,len(a)))].mean(1)
 return [round(float(a.mean()),6),round(float(np.quantile(samples,.025)),6),round(float(np.quantile(samples,.975)),6)]

def stats_arm(ss,arm,model,dic):
 v=[getvalid(dic,sid,arm) for sid in ss]
 v=[r for r in v if r is not None]
 return {'valid_n':len(v),'missing_n':len(ss)-len(v),
         'retained_correct_n':sum(correct(r,model) for r in v),
         'retained_correct_rate':round(sum(correct(r,model) for r in v)/len(v),6) if v else None,
         'p_gt_mean':round(float(np.mean([r['p_gt'] for r in v])),6) if v else None,
         'gt_log_margin_mean':round(float(np.mean([r['gt_log_margin'] for r in v])),6) if v else None,
         'accuracy_ci95':ci_means([int(correct(r,model)) for r in v]) if v else None}

def bothvalid_pair(ss,arm):
 a=[];b=[];cont=Counter()
 for sid in ss:
  x=getvalid(med,sid,arm);y=getvalid(qwen,sid,arm)
  if x is None or y is None:continue
  ok_m=correct(x,'med');ok_q=correct(y,'qwen')
  a.append(int(ok_m));b.append(int(ok_q))
  cont[(ok_m,ok_q)]+=1
 return {'common_valid_n':len(a),'both_right':cont[(True,True)],'med_right_qwen_wrong':cont[(True,False)],
         'med_wrong_qwen_right':cont[(False,True)],'both_wrong':cont[(False,False)],
         'med_retention_on_same_n':round(float(np.mean(a)),6) if a else None,
         'qwen_retention_on_same_n':round(float(np.mean(b)),6) if b else None,
         'paired_med_minus_qwen_ci95':ci_means([x-y for x,y in zip(a,b)]) if a else None}

def evaluate(ss):
 s=sorted(ss)
 return {'n_clean_both_right':len(s),'modalities':dict(Counter(items[sid]['modality'] for sid in s)),
   'med':{arm:stats_arm(s,arm,'med',med) for arm in arms},
   'qwen':{arm:stats_arm(s,arm,'qwen',qwen) for arm in arms},
   'same_rows':{arm:bothvalid_pair(s,arm) for arm in arms if arm!='pure_noise'}}
all_both=evaluate(clean_both)

# What is the relationship of no-image performance to conditioning on clean correctness?
groups=defaultdict(list)
for sid in sorted(items):
 m=getvalid(med,sid,'clean');q=getvalid(qwen,sid,'clean')
 if not(m and q):continue
 b1,b2=correct(m,'med'),correct(q,'qwen')
 key='both_clean_correct' if b1 and b2 else 'med_only_clean_correct' if b1 else 'qwen_only_clean_correct' if b2 else 'neither_clean_correct'
 groups[key].append(sid)
group_table=[]
for group,ss in groups.items():
 row={'group':group,'n':len(ss)}
 for label,table in [('med',med),('qwen',qwen)]:
  rr=[getvalid(table,sid,'no_image') for sid in ss]
  rr=[r for r in rr if r is not None]
  row[f'{label}_no_image_n']=len(rr)
  row[f'{label}_no_image_correct']=sum(correct(r,label) for r in rr)
  row[f'{label}_no_image_acc']=round(sum(correct(r,label) for r in rr)/len(rr),6) if rr else None
 group_table.append(row)

# Modality-specific same frozen clean-correct intersection
modal={mod:evaluate({sid for sid in clean_both if items[sid]['modality']==mod}) for mod in ['MRI','CT','Xray']}
summary={'scope':'original unfiltered official900 4-option items only',
 'criteria':'MedVLM-R1 native generated_correct Clean==True AND Qwen2-VL direct-score pred_correct Clean==True; neither no-image nor corruption outcomes used in selection',
 'n_original_four_option':825,'n_shared_clean_correct':len(clean_both),
 'n_original_clean_valid':sum(bool(getvalid(med,sid,'clean') and getvalid(qwen,sid,'clean')) for sid in items),
 'clean_retained_fraction':len(clean_both)/825,
 'four_groups_on_joint_valid_clean':group_table,
 'all_shared_clean_correct':all_both,'by_modality':modal,
 'limitations':['Clean-correct may be achieved by current image evidence, non-visual knowledge, answer prior, or guessing, so not a knowledge-controlled subset',
 'Conditioning on Clean correctness fixes Clean accuracy to 100% by design and raises baseline relative to full set',
 'Med native generated correctness and Qwen direct next-token scoring differ; cross-model percentage difference is diagnostic not RL causal',
 'Missing/format-failed nonclean arms excluded and valid denominators shown; donors exist only on restricted subsets',
 'Qwen pure noise and intermediate blur are not covered by full 825; earlier outputs come from screened subsets and are deliberately excluded']}
(OUT/'CLEAN_BOTH_CORRECT_506_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
(OUT/'CLEAN_BOTH_CORRECT_4OPT_IDS.txt').write_text('\n'.join(sorted(clean_both))+'\n')

def write_csv(path,records):
 with open(path,'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]))
  w.writeheader();w.writerows(records)

per_item=[]
for sid in sorted(clean_both):
 it=items[sid];d={'sample_id':sid,'modality':it['modality'],'question':it['question'],
                  'gt_letter':it['gt_letter'],'gt_text':it['gt_text']}
 for model,lookup in [('med',med),('qwen',qwen)]:
  for arm in arms:
   r=getvalid(lookup,sid,arm)
   d[f'{model}__{arm}__correct']=str(bool(correct(r,model))) if r else ''
   d[f'{model}__{arm}__pred']=str(r.get('generated_answer_letter' if model=='med' else 'pred_letter_from_probs','')) if r else ''
   d[f'{model}__{arm}__p_gt']=round(r['p_gt'],6) if r else ''
 per_item.append(d)
write_csv(OUT/'CLEAN_BOTH_CORRECT_506_PER_ITEM.csv',per_item)

# Non-model-filtered all4 result on noimage for comparison; report only clean-both group at forefront.
def fmt(x,n=1,suffix='%'):
 return '—' if x is None else f'{x*n:.1f}{suffix}'
lines=['# Clean-correct conditional analysis — unfiltered official MedVLM-R1 4-option MCQs','',
'**Design:** From 825 original four-option questions, select 506 that BOTH MedVLM-R1 native generated answer and Qwen2-VL direct next-token argmax answer correctly in Clean. The sample set is fixed before inspecting interventions. This is not a DeepSeek selection or a language-knowledge-controlled design.','',
'## Main per-condition retention on the fixed 506 items','',
'Rate is percent correctly answered among valid records in that condition. Invalid-format or missing-donor rows are missing; denominators shown.','',
'| Condition | MedVLM-R1 correct (valid n) | Qwen direct correct (valid n) | Same-valid items | Both still correct | Both now wrong |',
'|---|---:|---:|---:|---:|---:|']
for arm in arms:
 m=all_both['med'][arm];q=all_both['qwen'][arm]
 common=all_both['same_rows'].get(arm,{})
 lines.append(f'| {arm_names[arm]} | {fmt(m["retained_correct_rate"],100)} ({m["valid_n"]}) | {fmt(q["retained_correct_rate"],100)} ({q["valid_n"]}) | {common.get("common_valid_n","—")} | {common.get("both_right","—")} | {common.get("both_wrong","—")} |')
lines+=['','### Paired no-image joint outcomes on **same items**','']
pair=all_both['same_rows']['no_image']
for k in ['common_valid_n','both_right','med_right_qwen_wrong','med_wrong_qwen_right','both_wrong','med_retention_on_same_n','qwen_retention_on_same_n','paired_med_minus_qwen_ci95']:
 lines.append(f'- {k}: {pair[k]}')
lines+=['','## By modality: No-image and two image controls','',
'| Modality | n clean-both | Med noimage | Qwen noimage | Med blank-white | Qwen blank-white | Med plausible mismatch | Qwen plausible mismatch |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for mod,raw in modal.items():
 x=lambda model,arm:fmt(raw[model][arm]['retained_correct_rate'],100)
 lines.append(f'| {mod} | {raw["n_clean_both_right"]} | {x("med","no_image")} | {x("qwen","no_image")} | {x("med","blank_white")} | {x("qwen","blank_white")} | {x("med","pm_clean_k1")} | {x("qwen","pm_clean_k1")} |')
lines+=['','## How sample selection affects interpretation','',
'| Joint Clean-correctness group | n | Med no-image acc | Qwen no-image acc |',
'|---|---:|---:|---:|']
for r in group_table:
 lines.append(f'| {r["group"]} | {r["n"]} | {fmt(r["med_no_image_acc"],100)} (n={r["med_no_image_n"]}) | {fmt(r["qwen_no_image_acc"],100)} (n={r["qwen_no_image_n"]}) |')
lines+=['','## Interpretation',
'- Both-clean-correct does not imply the model already knew the correct diagnosis without seeing the image. A clean-correct answer can be caused by image grounding, general medical knowledge, answer frequency, or chance.',
'- In this fixed subset Clean accuracy is 100% **by construction**; No-image retention measures how often an originally successful answer survives image removal, not how much knowledge was independently acquired.',
'- Blank images control image-token presence but may introduce an unnatural visual input; shuffling/mismatch is better for specific image content, but does not guarantee semantically opposite ground truth.',
'- Failure when image removed may be due to absent visual data, altered prompt/image-token format, or both.',
'- Use same-row conditional rates and McNemar/paired bootstrap for robust differences; med native generation and qwen direct scoring must not be interpreted as a causal RL contrast.',
'- This was conducted with no DeepSeek filtering. The 506 IDs were selected only from Clean correctness.','',
'## Files','- CLEAN_BOTH_CORRECT_506_SUMMARY.json — complete counts, conditional outcome patterns, bootstrap CIs and modality splits.',
'- CLEAN_BOTH_CORRECT_506_PER_ITEM.csv — each ID, question, ground-truth label, every arm correctness, predicted letter, and p(GT).',
'- CLEAN_BOTH_CORRECT_4OPT_IDS.txt — exact 506 IDs.'
]
(OUT/'CLEAN_BOTH_CORRECT_506_REPORT.md').write_text('\n'.join(lines))
print('COUNTS_506',len(clean_both),'modalities',all_both['modalities'])
print('ARMS')
for arm in arms:
 m,q=all_both['med'][arm],all_both['qwen'][arm]
 print(arm,'Med',m['retained_correct_n'],'/',m['valid_n'],m['retained_correct_rate'],'Q',q['retained_correct_n'],'/',q['valid_n'],q['retained_correct_rate'])
print('JOINT_NOIMAGE',json.dumps(all_both['same_rows']['no_image'],ensure_ascii=False))
print('MODALITIES',[(k,v['n_clean_both_right'],v['med']['no_image']['retained_correct_rate'],v['qwen']['no_image']['retained_correct_rate']) for k,v in modal.items()])
print('GROUPS',group_table)
print('REPORT',OUT/'CLEAN_BOTH_CORRECT_506_REPORT.md')
