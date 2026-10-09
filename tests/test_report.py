"""Check that the user-facing report is one self-contained, data-consistent Markdown file."""
from __future__ import annotations
import json,re,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'MEDICAL_VLM_PROGRESS_REPORT.md'

class ReportIntegrity(unittest.TestCase):
 def setUp(self):
  self.s=REPORT.read_text(encoding='utf-8')

 def test_one_main_document_with_appendix(self):
  self.assertEqual(self.s.count('\n## Appendix\n'),1)
  front=self.s.split('\n---\n',1)[0]
  self.assertGreater(len(front.split()),300)
  self.assertLessEqual(len(front.split()),490)
  self.assertIn('## What we found',front)
  self.assertIn('## What surprised us',front)
  self.assertTrue(all(f'### {c}.' in self.s for c in 'ABCDE'))
  self.assertFalse((ROOT/'docs/ONE_PAGE_SUMMARY.md').exists())
  self.assertFalse((ROOT/'docs/DETAILED_REPORT.md').exists())

 def test_no_defensive_boilerplate(self):
  patterns=[r'we do not (?:claim|say)',r'cannot (?:prove|establish)',
            r'not a (?:causal|proof|reproduction)',r'important protocol boundary',
            r'not (?:yet|trained|a guarantee)',r'one.page summary',
            r'supporting report']
  for exp in patterns:
   self.assertNotRegex(self.s.lower(),exp)

 def test_local_links_and_images(self):
  for uri in re.findall(r'\]\(([^)]+)\)',self.s):
   if uri.startswith(('https://','http://','#')):continue
   self.assertTrue((ROOT/uri).exists(),uri)

 def test_headline_values_match_reproduced_records(self):
  j=json.loads((ROOT/'results/headline_recomputed.json').read_text())
  rows=j['official_matched']
  self.assertAlmostEqual(rows['mri']['uplift_clean']*100,29.11,delta=.02)
  self.assertAlmostEqual(rows['mri']['uplift_no_image']*100,32.11,delta=.02)
  self.assertAlmostEqual(rows['all']['uplift_clean']*100,12.08,delta=.02)
  self.assertAlmostEqual(rows['all']['uplift_no_image']*100,13.82,delta=.02)
  for token in ['+29.1 pp','+32.1 pp','+12.1 pp','+13.8 pp','0.730','0.894','0.211','0.228',
                '506','499','301','265','30.7%','23.5%']:
   self.assertIn(token,self.s)

 def test_markdown_tables_have_consistent_column_counts(self):
  tables=[];curr=[]
  for line in self.s.splitlines()+['']:
   if line.startswith('|') and line.endswith('|'):
    curr.append(line)
   else:
    if curr:tables.append(curr);curr=[]
  self.assertGreaterEqual(len(tables),5)
  for table in tables:
   n=[len(x.split('|')) for x in table]
   self.assertEqual(len(set(n)),1,table)
   self.assertGreaterEqual(len(table),3)

if __name__=='__main__':unittest.main()
