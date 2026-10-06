"""Summarise steer_v3.py output: per (layer, direction, alpha) mean margin by class, P(YES), BAcc of the
answers, AUC of the margin with a bootstrap CI. Usage: uv run --no-project --with pandas --with numpy python steer_summary.py"""
import numpy as np
import pandas as pd

import analyze_v3 as av

d = pd.read_csv("results_v3/steer_qwen2.5-vl.csv")
d["pred"] = pd.to_numeric(d["pred"], errors="coerce")
rng = np.random.default_rng(0)
rows = []
for (L, dr, a), g in d.groupby(["layer", "direction", "alpha"]):
    pos, neg = list(g.loc[g["gt"] == 1, "margin"]), list(g.loc[g["gt"] == 0, "margin"])
    lo, hi = av.boot_ci(pos, neg, 2000, rng)
    ans = g.dropna(subset=["pred"])
    h = ans.loc[ans["gt"] == 1, "pred"].mean()
    f = ans.loc[ans["gt"] == 0, "pred"].mean()
    rows.append(dict(layer=L, dir=dr, alpha=a, n=len(g), m1=np.mean(pos), m0=np.mean(neg),
                     gap=np.mean(pos) - np.mean(neg), py=ans["pred"].mean(), bacc=(h + 1 - f) / 2,
                     auc=av.auc(pos, neg), lo=lo, hi=hi, parse=len(ans) / len(g)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
