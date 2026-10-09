import importlib.util,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

D=module('prepare_datasets')
Q=module('evaluate_quality')
A=module('analyze_inference')

class FullPipelineTests(unittest.TestCase):
 def test_zip_images_same_filename_must_remain_distinct(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);z=d/'imgs.zip'
   with zipfile.ZipFile(z,'w') as f:
    f.writestr('imgs/study1/source.jpg',b'first image contents')
    f.writestr('imgs/study2/source.jpg',b'second image contents')
   extracted=D.extract_zip_entries(z,['imgs/study1/source.jpg','imgs/study2/source.jpg'],d/'extract')
   self.assertNotEqual(extracted['imgs/study1/source.jpg'],extracted['imgs/study2/source.jpg'])
   self.assertNotEqual(extracted['imgs/study1/source.jpg'].read_bytes(),extracted['imgs/study2/source.jpg'].read_bytes())
 def test_official_donor_maps_full_permutations(self):
  donor=json.loads((ROOT/'configs/official900_donors.json').read_text())
  self.assertEqual(donor['n_items'],900)
  for k,v in donor['derangements'].items():
   self.assertEqual(len(v),900)
   self.assertEqual(set(v),set(v.values()))
   self.assertFalse(any(x==y for x,y in v.items()))
  self.assertEqual(len(donor['semantic_conflict_label_supported_not_clinician_verified']),375)
 def test_blur_and_blank_are_deterministic(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d)
   a=np.zeros((24,24,3),dtype=np.uint8);a[4:20,4:20]=255
   img=d/'image.png';Image.fromarray(a).save(img)
   orig=Q.transform_image(str(img),'clean',d/'cache')
   blank=Q.transform_image(str(img),'blank',d/'cache')
   mild=Q.transform_image(str(img),'mild',d/'cache')
   severe=Q.transform_image(str(img),'severe',d/'cache')
   self.assertEqual(orig,str(img))
   self.assertEqual(Path(mild).read_bytes(),Path(Q.transform_image(str(img),'mild',d/'cache')).read_bytes())
   with Image.open(blank) as im:self.assertTrue(np.all(np.array(im)==255))
   with Image.open(severe) as im:self.assertFalse(np.array_equal(np.array(im),a))
 def test_entropy_and_margin_with_paired_conditions(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);inp=d/'predict.jsonl'
   with inp.open('w') as f:
    for sid,g in [('a',0),('b',1)]:
     for cond,p in [('clean',[.85,.15] if g==0 else [.15,.85]),('no_image',[.65,.35] if g==0 else [.65,.35])]:
      for order in ['O0','O1']:
       rec={'item_id':sid,'image_id':sid,'condition':cond,'order':order,
            'gt_index':g,'semantic_probs':p,'correct':int(np.argmax(p)==g)}
       f.write(json.dumps(rec)+'\n')
   summary,data=A.analyze(inp,bootstrap=25)
   self.assertEqual(summary['n_unique_items'],2)
   self.assertAlmostEqual(summary['conditions']['clean']['mean_per_order_accuracy'],1)
   self.assertAlmostEqual(summary['conditions']['no_image']['mean_per_order_accuracy'],.5)
   self.assertGreater(summary['contrasts']['clean_vs_noimage']['mean_GT_log_margin_utility'],0)
   self.assertGreater(data['b','no_image']['entropy'],0)

if __name__=='__main__':unittest.main()
