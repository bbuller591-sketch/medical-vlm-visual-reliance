#!/usr/bin/env python3
"""Create main comparison figures only after full per-item analysis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
x=json.load(open(ROOT/'results/summary.json'))
conds=['clean','blur_severe','random_same_modality_replacement','blurred_mismatch','no_image']
labels=['Clean','Severe blur','Mismatch','Blur + mismatch','No image']
fig,ax=plt.subplots(figsize=(9,4.7),dpi=160)
idx=np.arange(5);w=.34
for i,(m,label) in enumerate([('qwen2vl','Qwen2-VL Base'),('medvlmr1','MedVLM-R1')]):
 vals=[100*x['metrics'][m][c]['follow_nonbaseline']['mean'] for c in conds]
 offs=(-w/2 if i==0 else w/2)
 bars=ax.bar(idx+offs,vals,w,label=label)
 ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
ax.set_ylabel('Suggested-option following rate (%)')
ax.set_title('When the suggested answer differs from the current no-hint answer')
ax.set_xticks(idx,labels,rotation=9)
ax.set_ylim(0,108);ax.legend(loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();p=ROOT/'figures/following_by_condition.png';fig.savefig(p,dpi=160);plt.close(fig);print('SAVED',p)

fig,ax=plt.subplots(figsize=(9,4.7),dpi=160)
sub=conds[1:]; idx=np.arange(4)
for i,(m,label) in enumerate([('qwen2vl','Qwen2-VL Base'),('medvlmr1','MedVLM-R1')]):
 values=[100*x['interactions'][c]['follow_nonbaseline'][m.replace('qwen2vl','base').replace('medvlmr1','rl')+'_change']['mean'] for c in sub]
 bars=ax.bar(idx+(-w/2 if i==0 else w/2),values,w,label=label)
 ax.bar_label(bars,fmt='%+.1f',fontsize=8,padding=2)
ax.axhline(0,linewidth=1,alpha=.4)
ax.set_ylabel('Following-rate change vs Clean (percentage points)')
ax.set_xticks(idx,labels[1:],rotation=9)
ax.set_title('Paired quality/relevance effect, model-specific confidence fixed per question')
ax.legend();ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();p=ROOT/'figures/relative_to_clean.png';fig.savefig(p,dpi=160);plt.close(fig);print('SAVED',p)
