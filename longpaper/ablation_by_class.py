"""Ablation of the order direction (Qwen2.5-VL-7B, layer 21, App. G) split by harness frame-pairing class. CPU only.
  uv run --no-project --with numpy python longpaper/ablation_by_class.py"""
import csv, sys, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'longpaper')
import analyze_v3 as av
from within_class import cls4, auc
av.RECORDED.clear(); av.load_recorded('results_v3'); meta=av.load_meta('results_v3/videos_v3_metadata.csv')
R=list(csv.DictReader(open('results_v3/ablate2_qwen2.5-vl.csv')))
by={}
for r in R: by.setdefault(r['direction'],{})[r['path']]=float(r['margin'])
paths=list(by['none']); gt={r['path']:int(r['gt']) for r in R}
cl={p:cls4(meta[p],av.sampled(meta[p],'qwen2.5-vl')) for p in paths}
rng=np.random.default_rng(7)
for c in ('in/btw','in/in','btw/in','btw/btw','ALL'):
    P=[p for p in paths if c=='ALL' or cl[p]==c]; y=np.array([gt[p] for p in P])
    a0=np.array([by['none'][p] for p in P]); a1=np.array([by['order'][p] for p in P])
    d=auc(a1,y)-auc(a0,y); bs=[]
    for _ in range(2000):
        i=rng.integers(0,len(P),len(P)); bs.append(auc(a1[i],y[i])-auc(a0[i],y[i]))
    print(c,len(P),round(auc(a0,y),3),round(auc(a1,y),3),round(d,3),[round(x,3) for x in np.nanpercentile(bs,[2.5,97.5])])
