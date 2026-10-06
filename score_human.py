"""Score the human ratings from human_kit/rate_order.html against the key. CPU only.

Usage: uv run --no-project --with numpy python score_human.py order_ratings_*.csv
Prints, per rater and family (shape / real footage): n, P(YES), balanced accuracy with a bootstrap 95% CI,
and accuracy by confidence; then agreement between raters (Cohen's kappa) when there are two or more.
Writes human_kit/human_scores.json.
"""
import csv
import json
import sys

import numpy as np

key = json.load(open("human_kit/key.json"))
rng = np.random.default_rng(0)


def bacc(g, a):
    return ((a[g == 1] == 1).mean() + (a[g == 0] == 0).mean()) / 2


out, by = {}, {}
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f)):
        by.setdefault(r["rater"].strip('"'), {})[r["id"]] = (int(r["answer"]), int(r["confidence"]))
for rater, ans in by.items():
    out[rater] = {}
    for fam in ("shape", "real"):
        ids = [i for i in ans if i.startswith(fam)]
        g = np.array([key[i]["gt"] for i in ids]); a = np.array([ans[i][0] for i in ids]); c = np.array([ans[i][1] for i in ids])
        bs = [bacc(g[j], a[j]) for j in (rng.integers(0, len(g), len(g)) for _ in range(5000)) if 0 < g[j].sum() < len(j)]
        out[rater][fam] = dict(n=len(ids), py=float(a.mean()), bacc=float(bacc(g, a)), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               acc_by_conf={int(k): float((a[c == k] == g[c == k]).mean()) for k in sorted(set(c))})
        print(f"{rater:12} {fam:5} n={len(ids)} P(YES)={a.mean():.2f} BAcc={out[rater][fam]['bacc']:.3f} "
              f"[{out[rater][fam]['ci'][0]:.3f},{out[rater][fam]['ci'][1]:.3f}] by confidence {out[rater][fam]['acc_by_conf']}")
raters = list(by)
for i in range(len(raters)):
    for j in range(i + 1, len(raters)):
        ids = sorted(set(by[raters[i]]) & set(by[raters[j]]))
        x = np.array([by[raters[i]][k][0] for k in ids]); y = np.array([by[raters[j]][k][0] for k in ids])
        po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
        print(f"kappa {raters[i]} vs {raters[j]}: {(po - pe) / (1 - pe):.3f} on {len(ids)} items")
json.dump(out, open("human_kit/human_scores.json", "w"), indent=1)
