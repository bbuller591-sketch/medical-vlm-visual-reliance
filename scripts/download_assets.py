#!/usr/bin/env python3
"""Official-source model and benchmark downloader. It does not bundle datasets.

Usage: python scripts/download_assets.py --models both --datasets official slake
       python scripts/download_assets.py --datasets pmcvqa --accept-large-download

Uses only Hugging Face Hub and the cited *original* MedVLM-R1 GitHub release.
All paths are relative to --asset-root, never tied to the developers' computers.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request
from pathlib import Path

MODELS={'base':'Qwen/Qwen2-VL-2B-Instruct','rl':'JZPeterPan/MedVLM-R1'}
MODEL_PINNED_REVISIONS={'base':'895c3a49bc3fa70a340399125c650a463535e71c',
                        'rl':'d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9'}
DATASETS={
 'official':('FreedomIntelligence/Medical_Multimodal_Evaluation_Data',['images.zip'], '7d5b870cf7ace833a187a3459d7180a60ce9c5a1'),
 'slake':('BoKelvin/SLAKE',['imgs.zip','test.json','validation.json'],'a9083ce6c34ac3ffb17671a605962924d8a8f9e9'),
 'pmcvqa':('xmcmic/PMC-VQA',['test_clean.csv','images.zip','images_2.zip'],'b56ae594f794867893143b337b4118a835794647'),
 'vqarad':('flaviagiammarino/vqa-rad',['data/test-00000-of-00001.parquet'],'4812b887ee2266c262d43600597b498d27672762'),
}
OFFICIAL_SHA='9daad1bfd5c3181a0b91292f70418522d7b96ffc4e2d03e5d8e4ed489646b1c1'
OFFICIAL_URL=('https://raw.githubusercontent.com/JZPeterPan/MedVLM-R1/'
 '2eee4210b25b149dcfa68513eb4682ac63b029d2/MRI_CT_XRAY_300each_dataset.json')

def download_official_manifest(dest:Path):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.is_file():
  with urllib.request.urlopen(OFFICIAL_URL,timeout=90) as source:
   dest.write_bytes(source.read())
 digest=hashlib.sha256(dest.read_bytes()).hexdigest()
 if digest != OFFICIAL_SHA:
  raise ValueError(f'Official 900 manifest checksum mismatch: {digest} != {OFFICIAL_SHA}')
 print('OFFICIAL 900 JSON SHA256 verified',digest[:18],dest)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--asset-root',type=Path,default=Path('assets'))
 p.add_argument('--models',choices=['none','base','rl','both'],default='none')
 p.add_argument('--datasets',nargs='*',choices=tuple(DATASETS),default=[])
 p.add_argument('--accept-large-download',action='store_true',help='PMC-VQA image archives can exceed 20 GB')
 p.add_argument('--revision-base',default=MODEL_PINNED_REVISIONS['base'],help='Optional git commit SHA of Qwen checkpoint')
 p.add_argument('--revision-rl',default=MODEL_PINNED_REVISIONS['rl'],help='Optional git commit SHA of MedVLM-R1 checkpoint')
 args=p.parse_args();root=args.asset_root.resolve();root.mkdir(parents=True,exist_ok=True)
 if args.models=='none' and not args.datasets:p.error('Choose models and/or datasets')
 if 'pmcvqa' in args.datasets and not args.accept_large_download:
  p.error('PMC-VQA archives require --accept-large-download (~21 GB)')
 try:from huggingface_hub import snapshot_download
 except ImportError as e:raise SystemExit('Install requirements-download.txt first') from e
 requests=['base','rl'] if args.models=='both' else ([] if args.models=='none' else [args.models])
 for key in requests:
  target=root/'models'/('Qwen2-VL-2B-Instruct' if key=='base' else 'MedVLM-R1')
  print('DOWNLOADING CHECKPOINT',MODELS[key],'to',target,flush=True)
  snapshot_download(MODELS[key],repo_type='model',local_dir=target,
                    revision=(args.revision_base if key=='base' else args.revision_rl))
  assert (target/'config.json').exists()
 for key in args.datasets:
  dataset,patterns,revision=DATASETS[key]
  target=root/'datasets'/key
  print('DOWNLOADING DATASET',dataset,'files',patterns,'to',target,flush=True)
  snapshot_download(dataset,repo_type='dataset',local_dir=target,allow_patterns=patterns,
                    revision=revision)
  if key=='official': download_official_manifest(target/'MRI_CT_XRAY_300each_dataset.json')
  print('DATASET DOWNLOAD FINISHED',key,flush=True)
 print('ASSET DOWNLOAD FINISHED',root)
if __name__=='__main__':main()
