"""Self-contained checks for the incrementally authored research report.

Review phase: Parts 0–6 with Appendices 0–4 and checked results.
"""
from __future__ import annotations
import csv,re,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'MEDICAL_VLM_PROGRESS_REPORT.md'

class ReportIntegrity(unittest.TestCase):
 def setUp(self):
  self.s=REPORT.read_text(encoding='utf-8')
  self.front,self.appendix=self.s.split('\n---\n')

 def test_summary_and_appendix_sections_are_paired(self):
  self.assertEqual(self.s.count('\n---\n'),1)
  for fragment in ['## 0. Research Questions and Experimental Setup',
                   '## 1. Does the Model Use the Current Image?',
                   '## 2. How Does Image Quality Affect Visual Reliance?',
                   '## 3. What Did RL Actually Improve?',
                   '## 4. Does Medical RL Generalize Beyond the Training Benchmark?',
                   '## 5. Discussion and Research Implications',
                   '## 6. Next Steps']:
   self.assertEqual(self.front.count(fragment),1)
  for fragment in ['### Appendix 0.','### Appendix 1.','### Appendix 2.','### Appendix 3.','### Appendix 4.']:
   self.assertEqual(self.appendix.count(fragment),1)
  part0=self.front.split('## 0.',1)[-1].split('## 1.',1)[0]
  part1=self.front.split('## 1.',1)[-1].split('## 2.',1)[0]
  part2=self.front.split('## 2.',1)[-1]
  self.assertTrue(170<len(part0.split())<300)
  self.assertTrue(140<len(part1.split())<260)
  self.assertIn('reinforcement learning (RL)',part0)
  self.assertIn('the current image',part0)
  self.assertIn('506 questions',part1)
  self.assertIn('825 four-choice',part1)
  self.assertIn('| Model | Clean | No-image | Mismatch |',part1)
  self.assertIn('900 official questions',part2)
  self.assertIn('559 four-choice questions',part2)
  self.assertIn('Decision-Token Intervention',part2)
  self.assertIn('Appendix 2.6–2.9',part2)
  self.assertNotIn('Blur + Mismatch',part2.split('### 2.2.',1)[1].split('### 2.3.',1)[0])
  part3=self.front.split('## 3.',1)[-1]
  self.assertIn('**825-question, four-choice**',part3)
  self.assertIn('| Clean image | 68.89% | **80.97%** | **+12.08 pp** |',part3)
  self.assertIn('| No-image | 44.12% | **57.94%** | **+13.82 pp** |',part3)
  self.assertIn('**66** questions are answered correctly only by R1',part3)
  self.assertIn('versus **6** only by Base',part3)
  self.assertIn('MRI (n=300)',part3)
  self.assertIn('Appendix 3.3–3.4',part3)
  self.assertIn('### Appendix 3.',self.appendix)

 def test_original_option_counts_and_part0(self):
  for x in ['| MRI | 300 | 0 | 300 |','| CT | 237 | 63 | 300 |',
            '| X-ray | 288 | 12 | 300 |','| **Total** | **825** | **75** | **900** |']:
   self.assertIn(x,self.s)
  self.assertIn('the third ID repeats',self.s)
  self.assertIn('minimum 3,136 pixels',self.s)
  self.assertIn('maximum 401,408 pixels',self.s)
  self.assertNotIn('**Illustrative four-choice example',self.front)
  self.assertIn('**Illustrative four-choice example',self.appendix)

 def test_part1_tables_against_published_aggregate_files(self):
  with (ROOT/'results/official825_visual_interventions.csv').open(newline='') as f:
   arms={(r['model'],r['condition']):r for r in csv.DictReader(f)}
  for model,cond,label in [('Qwen2-VL Base','Clean','64.73%'),
                           ('MedVLM-R1','Clean','81.46%'),
                           ('Qwen2-VL Base','No-image','43.39%'),
                           ('MedVLM-R1','No-image','55.69%'),
                           ('Qwen2-VL Base','Mismatch','50.18%'),
                           ('MedVLM-R1','Mismatch','69.58%')]:
   r=arms[model,cond];self.assertEqual(f"{float(r['accuracy_pct']):.2f}%",label)
   self.assertIn(label,self.appendix)
  for model in ['Qwen2-VL Base','MedVLM-R1']:
   acc={cond:float(arms[model,cond]['accuracy_pct']) for cond in ['Clean','No-image','Mismatch']}
   for difference in [acc['Clean']-acc['No-image'],acc['Clean']-acc['Mismatch'],acc['Mismatch']-acc['No-image']]:
    self.assertIn(f'+{difference:.2f} pp',self.appendix)

  with (ROOT/'results/clean_correct_506_interventions.csv').open(newline='') as f:
   rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),7)
  for r in rows:
   for col in ['base','rl']:
    if not r[f'{col}_valid'] or int(r[f'{col}_valid'])==0:continue
    numerator=int(r[f'{col}_correct']);denominator=int(r[f'{col}_valid'])
    self.assertIn(f'{numerator}/{denominator}',self.appendix)
    self.assertAlmostEqual(100*numerator/denominator,float(r[f'{col}_retention_pct']),delta=.006)
  self.assertIn('265/499',self.appendix)
  self.assertIn('301/499',self.appendix)

 def test_math_figures_links_and_tables(self):
  self.assertNotRegex(self.s,r'\\\(|\\\)|\\\[|\\\]')
  self.assertEqual(self.s.count('$$'),20)
  for uri in re.findall(r'\]\(([^)]+)\)',self.s):
   if uri.startswith(('http://','https://','#')):continue
   self.assertTrue((ROOT/uri).exists(),uri)
  self.assertTrue((ROOT/'figures/clean_correct_506_retention.png').exists())
  tables=[];group=[]
  for line in self.s.splitlines()+['']:
   if line.startswith('|') and line.endswith('|'):group.append(line)
   elif group:tables.append(group);group=[]
  self.assertGreaterEqual(len(tables),5)
  for t in tables:
   self.assertGreaterEqual(len(t),3)
   self.assertEqual(len({x.count('|') for x in t}),1)

 def test_heldout_quality_interchange_matches_frozen_results(self):
  import json
  bundle=ROOT/'experiments/quality_state_interchange_20261009'
  data=json.loads((bundle/'results/summary_extended.json').read_text())
  ids=[]
  for model,expected_full,expected_quality,expected_gap in [
      ('qwen2vl',-0.41806,-0.12778,-0.13056),
      ('medvlmr1',-0.77639,-0.31528,-0.15694)]:
   m=json.loads((bundle/f'results/metadata_{model}.json').read_text())
   self.assertEqual((m['train_n'],m['eval_n'],m['layers']),(450,90,[20,27]))
   self.assertEqual(len(set(m['train_ids']) & set(m['eval_ids'])),0)
   ids.append(m['eval_ids'])
   d=data[model]
   self.assertAlmostEqual(d['effects']['clean|L27|full_blur_state']['delta_log_margin']['mean'],expected_full,places=4)
   self.assertAlmostEqual(d['effects']['clean|L27|quality_shift']['delta_log_margin']['mean'],expected_quality,places=4)
   self.assertAlmostEqual(d['content_gap']['L27|quality_shift|change']['mean'],expected_gap,places=4)
   self.assertIn(f'{abs(expected_gap):.3f}',self.s)
   for kind in ['steering','diagnostic','dose']:
    rows=[r for r in (bundle/f'results/{kind}_{model}.jsonl').read_text().splitlines() if r.strip()]
    self.assertEqual(len(rows),{'steering':2610,'diagnostic':630,'dose':360}[kind])
  self.assertEqual(ids[0],ids[1])
  for figure in ['quality_content_gap.png','clean_log_margin_change.png',
                 'visual_content_advantage_change.png','quality_direction_dose_response.png']:
   self.assertTrue((ROOT/'figures'/figure).is_file())

 def test_pairwise_recomputation_interchange_advantage(self):
  import json
  import numpy as np
  b=ROOT/'experiments/quality_state_interchange_20261009/results'
  for model in ('qwen2vl','medvlmr1'):
   rows=[json.loads(x) for x in (b/f'steering_{model}.jsonl').read_text().splitlines()]
   d={(r['sample_id'],r['condition'],r['layer'],r['intervention']):r for r in rows}
   ids=json.loads((b/f'metadata_{model}.json').read_text())['eval_ids']
   diffs=[]
   for sid in ids:
    base=d[sid,'clean',0,'baseline']['log_margin']-d[sid,'random_same_modality_replacement',0,'baseline']['log_margin']
    shifted=d[sid,'clean',27,'quality_shift']['log_margin']-d[sid,'random_same_modality_replacement',27,'quality_shift']['log_margin']
    diffs.append(shifted-base)
   observed=float(np.mean(diffs))
   summary=json.loads((b/'summary_extended.json').read_text())[model]
   self.assertAlmostEqual(observed,summary['content_gap']['L27|quality_shift|change']['mean'],places=4)

 def test_matched_rl_gain_recomputation(self):
  import json
  x=json.loads((ROOT/'results/rl_matched_gains_audit.json').read_text())
  groups=x['group_scores']
  self.assertEqual(groups['all825']['n'],825)
  self.assertEqual(groups['MRI300']['n'],300)
  self.assertEqual(groups['CT237']['n'],237)
  self.assertEqual(groups['Xray288']['n'],288)
  for group,expected in [('all825',(12.0808,13.8182,-1.7374)),
                         ('MRI300',(29.1111,32.1111,-3.0))]:
   y=groups[group]['metrics_pp']
   for name,num in zip(['rl_gain_clean','rl_gain_noimage','image_benefit_difference_RL_minus_Base'],expected):
    self.assertAlmostEqual(y[name]['mean'],num,places=3)
  for val in ['+29.11 pp','+32.11 pp','+12.08 pp','+13.82 pp',
              '600 MRI','82','46','219','152','26.30%','−3.00','−1.74']:
   self.assertIn(val,self.s)
  for v in ['[−7.56, +1.67]','[−3.80, +0.32]','[+2.81, +11.62]']:
   self.assertIn(v,self.s)
  self.assertIn('format component',self.s.lower())
  self.assertIn('accuracy component',self.s.lower())

 def test_joint_clean_559_image_free_comparison(self):
  import json
  d=json.loads((ROOT/'results/joint_clean_559_noimage_audit.json').read_text())
  g=d['groups']['All']
  self.assertEqual(g['n'],559)
  self.assertEqual(g['clean_accuracy']['rl_minus_base'],0.0)
  self.assertEqual(g['no_image_correct_counts'],{'base':288,'rl':348})
  self.assertAlmostEqual(g['no_image_accuracy']['rl_minus_base']*100,10.73345259,places=5)
  self.assertEqual(g['no_image_paired_counts'],{'base_0_rl_0':205,'base_1_rl_1':282,'base_0_rl_1':66,'base_1_rl_0':6})
  self.assertEqual(d['groups']['MRI']['no_image_correct_counts'],{'base':95,'rl':148})
  part33=self.front.split('### 3.3.',1)[1].split('### 3.4.',1)[0]
  app33=self.appendix.split('#### 3.3.',1)[1].split('#### 3.4.',1)[0]
  self.assertNotIn('**0.00 pp**',part33)
  self.assertNotIn('Clean accuracy',part33)
  self.assertNotIn('100.00%',app33)
  self.assertIn('**All 559 questions**',part33)
  self.assertIn('**MRI 207 questions**',part33)
  self.assertIn('**+10.73 pp**',self.front)
  self.assertIn('**66**',self.front)
  self.assertIn('**6**',self.front)
  self.assertIn('**[Appendix 3.3–3.4]',self.front)

 def test_advice_559_integrity_and_main_table(self):
  import json
  folder=ROOT/'experiments/visual_advice_conflict_20261009'
  selected=json.loads((folder/'results/selection.json').read_text())
  summary=json.loads((folder/'results/summary.json').read_text())
  self.assertEqual(selected['n'],559)
  self.assertEqual(summary['items_complete'],559)
  self.assertEqual(selected['modality_counts'],{'CT':160,'MRI':207,'Xray':192})
  for row in selected['items']:
   for m in ('qwen2vl','medvlmr1'):
    p=row[m+'_clean_pmax']
    c=row[m+'_hint_confidence_pct']
    self.assertLessEqual(abs(c-(25+0.5*(100*p-25))),0.00051)
  part2=self.front.split('### 2.2.',1)[-1].split('### 2.3.',1)[0]
  for cond in ('Clean','Severe Blur','Mismatch','No-image'):
   self.assertRegex(part2,r'(?m)^\| '+cond+r' \|')
  for cond, base, rl in [('clean','66.1%','40.5%'),
                          ('blur_severe','82.3%','66.3%'),
                          ('random_same_modality_replacement','70.5%','48.7%'),
                          ('no_image','89.5%','81.8%')]:
   b=100*summary['metrics']['qwen2vl'][cond]['follow_nonbaseline']['mean']
   r=100*summary['metrics']['medvlmr1'][cond]['follow_nonbaseline']['mean']
   self.assertEqual(f'{b:.1f}%',base)
   self.assertEqual(f'{r:.1f}%',rl)
   self.assertIn(base,part2)
   self.assertIn(rl,part2)
  self.assertIn('27,950 scored predictions',self.appendix)
  for m in ('qwen2vl','medvlmr1'):
   raw=folder/'results'/f'advice_{m}.jsonl'
   with raw.open() as f:
    self.assertEqual(sum(1 for _ in f),13975)

 def test_all_blur_levels_in_compact_quality_table(self):
  part2=self.front.split('### 2.1.',1)[-1].split('### 2.2.',1)[0]
  for row in ['| **Clean** | 0 px | **0.730** | **0.894** |',
              '| Mild | 2 px | 0.736 | 0.883 |',
              '| Additional sensitivity check | 4 px | 0.599 | 0.742 |',
              '| Medium | 6 px | 0.543 | 0.647 |',
              '| **Severe** | 12 px | **0.211** | **0.228** |']:
   self.assertIn(row,part2)

 def test_external_base_rl_generalization_and_compact_discussion(self):
  import json
  data=json.loads((ROOT/'results/external_base_rl_generalization_audit.json').read_text())
  sl=data['dataset_summaries']['slake_yesno_test']
  pm=data['dataset_summaries']['pmcvqa']
  self.assertEqual((sl['n'],sl['n_unique_images']),(355,96))
  self.assertEqual((pm['n'],pm['n_unique_images']),(1000,852))
  self.assertAlmostEqual(100*sl['checkpoint_effect']['clean']['mean'],-5.7746,places=3)
  self.assertAlmostEqual(100*pm['checkpoint_effect']['clean']['mean'],0.3333,places=3)
  self.assertEqual(sl['clean_both_correct']['n'],209)
  self.assertEqual(pm['clean_both_correct']['n'],448)
  self.assertEqual(data['visual_conflict_paired_slake']['n_pairs'],47)
  self.assertEqual(data['visual_conflict_paired_slake']['models']['qwen2vl']['counts']['positive_image_switch'],40)
  self.assertEqual(data['visual_conflict_paired_slake']['models']['medvlmr1']['counts']['positive_image_switch'],38)
  for fragment in ['**−5.77 pp**','**+0.33 pp**','**+25.60 pp**','40/47 (85.11%)','38/47 (80.85%)']:
   self.assertIn(fragment,self.front)
  for fragment in ['### 4.1.','### 4.2.','### 4.3.','### 5.1.','### 5.2.','### 5.3.','## 6. Next Steps']:
   self.assertIn(fragment,self.front)
  for fragment in ['#### 4.1.','#### 4.2.','#### 4.3.','#### 4.4.','#### 4.5.']:
   self.assertIn(fragment,self.appendix)
  self.assertIn('No-image accuracy',self.appendix)
  self.assertIn('VQA-RAD',self.appendix)

 def test_no_old_sections_or_defensive_phrasing(self):
  self.assertNotIn('## 7.',self.s)
  self.assertNotIn('### Appendix 5.',self.s)
  for phrase in ['not proof','cannot establish','we do not claim','important protocol boundary']:
   self.assertNotIn(phrase,self.s.lower())

if __name__=='__main__':unittest.main()
