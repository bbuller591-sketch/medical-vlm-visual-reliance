#!/usr/bin/env python3
"""Plot published 506 Clean-both-correct conditional intervention retention.

This CPU-only script reads the public aggregate count table, uses each model's
arm-specific valid denominator and stores a reproducible grouped bar chart.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,default=Path('results/clean_correct_506_interventions.csv'))
 p.add_argument('--output',type=Path,default=Path('figures/clean_correct_506_retention.png'))
 args=p.parse_args()
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 with args.data.open(newline='',encoding='utf-8') as f:rows={r['condition']:r for r in csv.DictReader(f)}
 conditions=['Clean','No-image','White blank','Mismatch']
 label={'Clean':'Clean','No-image':'No image','White blank':'Blank','Mismatch':'Mismatch'}
 values=[]
 for model in ['base','rl']:
  ratios=[]
  for cond in conditions:
   r=rows[cond];a=int(r[model+'_correct']);n=int(r[model+'_valid']);ratios.append(100*a/n)
  values.append(ratios)
 fig,ax=plt.subplots(figsize=(8.4,4.2),dpi=180)
 xs=np.arange(len(conditions));w=.34
 for i,(name,data) in enumerate(zip(['Qwen2-VL Base','MedVLM-R1'],values)):
  bars=ax.bar(xs+(-w/2 if i==0 else w/2),data,width=w,label=name)
  ax.bar_label(bars,fmt='%.1f%%',fontsize=9,padding=3)
 ax.set_ylim(0,110);ax.set_ylabel('Correct-answer retention among valid outputs (%)')
 ax.set_xticks(xs,[label[x] for x in conditions]);ax.legend(loc='upper right')
 ax.set_title('Original 506 questions correctly answered by both models (Clean)')
 ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 fig.tight_layout()
 args.output.parent.mkdir(exist_ok=True,parents=True)
 fig.savefig(args.output,dpi=180)
 plt.close(fig)
 print('PLOT_SAVED',args.output,'conditions',len(conditions))
if __name__=='__main__':main()
