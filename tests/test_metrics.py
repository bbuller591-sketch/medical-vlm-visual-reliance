import importlib.util,unittest,tempfile,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evaluation',ROOT/'scripts/evaluate_external.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
class TestEvaluation(unittest.TestCase):
 def test_option_orders(self):
  for n in [2,4]:
   for o in (['O0','O1'] if n==2 else ['O0','O1','O2']):
    p=evaluation.option_order(n,o,'CT_001')
    self.assertEqual(sorted(p),list(range(n)))
  self.assertEqual(evaluation.option_order(4,'O0','CT_001'),[0,1,2,3])
  self.assertEqual(evaluation.option_order(4,'O1','CT_001'),[1,2,3,0])
 def test_donor_plan_is_reusable(self):
  items=[{'item_id':f'i{i}','image_id':f'im{i}','image_path':f'/tmp/im{i}.png','question':'q','options':['yes','no'],'gt_index':i%2} for i in range(7)]
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'plan.json'
   a=evaluation.build_or_load_donor_plan(items,path)
   b=evaluation.build_or_load_donor_plan(items,path)
   self.assertEqual(a,b)
   self.assertTrue(all(k!=v for k,v in a.items()))
   self.assertEqual(set(a.values()),set(a.keys()))
 def test_signed_prob_margin(self):
  p=np.array([.1,.2,.7])
  self.assertAlmostEqual(float(p[2]-np.max(p[[0,1]])),.5)
  self.assertAlmostEqual(float(p[0]-np.max(p[[1,2]])),-.6)
if __name__=='__main__':unittest.main()
