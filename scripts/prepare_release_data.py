#!/usr/bin/env python3
"""Local-only export from full experiment outputs to lightweight non-image tables.
Run from the original private project root; does NOT copy medical images, weights,
private paths, question strings, API responses, or tokens.
"""
from pathlib import Path
import csv,json,shutil
ROOT=Path(__file__).resolve().parents[3]  # medical/
OUT=Path(__file__).resolve().parents[1]/'results'
OUT.mkdir(exist_ok=True)

def export(infile,outfile,keys,filter_fn=None):
 with open(infile,encoding='utf-8') as f:
  rows=list(csv.DictReader(f))
 chosen=[r for r in rows if filter_fn is None or filter_fn(r)]
 with open(outfile,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in chosen])
 print(outfile.relative_to(OUT.parent),len(chosen))

export(ROOT/'visual_reliance_mechanism_v2/RESULTS/behaviour_per_example.csv',OUT/'official825_scores.csv',
 ['model','condition','order','sample_id','modality','n_options','gt_index','pred_index','correct','p_gt','margin_gt_minus_best_wrong','entropy_semantic'],
 lambda r:r['n_options']=='4' and r['condition'] in ('clean','no_image'))
export(ROOT/'quality_reliance_v1/RESULTS/QUALITY_RELEVANCE_SUMMARY.csv',OUT/'quality_content_donor_contrasts.csv',
 ['model','cohort','stratum','quality_level','donor_k','quantity','value','n'],
 lambda r:r['cohort']=='all' and r['stratum']=='ALL' and r['contrast_family']=='plausible_mismatch' and r['quantity']=='U_content')
for ds in ['slake_yesno_test','pmcvqa']:
 inp=ROOT/f'briefing_20261009/external_extension/{ds}_per_item_joined.csv'
 dest=OUT/f'{ds}_per_item.csv'
 shutil.copyfile(inp,dest);print(dest.relative_to(OUT.parent))
counts=json.load(open(ROOT/'briefing_20261009/CLEAN_BOTH_CORRECT_506_SUMMARY.json'))
row=next(x for x in counts['four_groups_on_joint_valid_clean'] if x['group']=='both_clean_correct')
summary={'n_shared_clean_correct':506,'n_both_no_image_valid':499,
 'both_correct':219,'rl_only_correct':82,'base_only_correct':46,'neither_correct':152,
 'med_no_image_correct_in_valid':301,'base_no_image_correct_in_valid':265,
 'base_definition':'Qwen direct candidate scoring on Clean; RL native generated answer on Clean',
 'limitations':'Clean conditioned and output protocols differ; not a controlled equal-knowledge subset'}
# Verify 219+82=301; 219+46=265
assert 219+82==301 and 219+46==265 and 219+82+46+152==499
assert row['n']==506 and row['med_no_image_correct']==301
(OUT/'clean_both_success_506.json').write_text(json.dumps(summary,indent=2))
for name in ['consensus_image_required_ids.txt','consensus_image_quality_clear_ids.txt','quality_and_textnotstrong_40pct_ids.txt']:
 src=ROOT/'native_reasoning_reliance_v1/analysis/deepseek_answerability_judge_v1'/name
 dst=OUT/'selected_subsets'/name
 dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(src,dst)
 print('selected_subsets',name,'n',len(dst.read_text().splitlines()))
