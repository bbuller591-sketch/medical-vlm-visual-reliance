#!/usr/bin/env python3
"""Deterministic file integrity inventory (excludes the inventory file itself)."""
from pathlib import Path
import json,hashlib,sys,re
P=Path(__file__).resolve().parents[1]
files=[]
for f in sorted(P.rglob('*')):
 if not f.is_file() or '.git' in f.parts or '__pycache__' in f.parts or f.name=='SHA256.json':continue
 relative=f.relative_to(P).as_posix()
 if f.suffix.lower() in ('.csv','.md','.py','.sh','.txt','.json','.jsonl'):
  t=f.read_text(encoding='utf-8')
  assert '/data2/' + 'zhuangcy/' not in t,f'Nonportable absolute path in {relative}'
  assert '-----BEGIN ' + 'PRIVATE' not in t,f'Sensitive marker in {relative}'
 files.append({'file':relative,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size})
out=P/'results/SHA256.json'
if '--verify' in sys.argv:
 old=json.loads(out.read_text());assert old==files, 'File inventory differs from saved SHA256';print('INTEGRITY VERIFIED',len(files),'files')
else:
 out.write_text(json.dumps(files,indent=2));print('INVENTORY CREATED',len(files),'files')
