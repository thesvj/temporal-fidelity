"""Probe vs. margin within each frame-pairing class, Qwen2.5-VL-7B, shape set, harness frames. CPU only.

Answers the review point that "decodable yet unused" may be a pooling artefact: a pooled margin threshold cancels a margin
that is right in one alignment class and reversed in another, while the probe is free to use either sign.
Within each of the four harness alignment classes (first/second jump inside a fused pair or between pairs) we compare,
on the same visible trials and the same outer folds:
  probe_pooled   nested ridge probe trained on all trials (as in probe_stats.py), its scores restricted to the class
  probe_within   nested ridge probe trained and tested inside the class (layer and strength chosen in inner folds)
  margin_signfree the first-token margin with its sign fitted on the training folds (so a reversed margin counts)
all as AUC, with a paired bootstrap of probe minus margin, a 200-permutation null for probe_within, and the margin's
held-out threshold BAcc (sign free; 200 half splits) against the .60 bar.

  uv run --no-project --with numpy python longpaper/within_class.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "longpaper"))
import analyze_v3 as av  # noqa: E402
import probe_stats as ps  # noqa: E402

MODEL, RES, SEED = "qwen2.5-vl", ROOT / "results_v3", 20261005
CLASSES = ("in/btw", "in/in", "btw/in", "btw/btw")


def cls4(meta_row, idx):
    a, b = sorted((int(float(meta_row["f_first"])), int(float(meta_row["f_second"]))))
    s = next((k for k, i in enumerate(idx) if i >= a), None)
    e = next((k for k, i in enumerate(idx) if i >= b), None)
    if s is None or e is None:
        return "na"
    return ("in" if s % 2 else "btw") + "/" + ("in" if e % 2 else "btw")


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    allv = np.concatenate([p, n]); order = np.argsort(allv, kind="mergesort"); rk = np.empty(len(allv))
    sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def heldout_bacc(m, y, rng, splits=200):
    out = []
    for _ in range(splits):
        i = rng.permutation(len(y)); a, b = i[: len(y) // 2], i[len(y) // 2:]
        sg = 1.0 if auc(m[a], y[a]) >= .5 else -1.0
        ma, mb = sg * m[a], sg * m[b]
        best, t = -1, 0.0
        for c in np.unique(ma):
            h = np.mean(ma[y[a] == 1] >= c); f = np.mean(ma[y[a] == 0] >= c); v = (h + 1 - f) / 2
            if v > best:
                best, t = v, c
        h = np.mean(mb[y[b] == 1] >= t); f = np.mean(mb[y[b] == 0] >= t); out.append((h + 1 - f) / 2)
    return float(np.mean(out))


def main():
    z = np.load(ROOT / "longpaper/feats_v3" / f"{MODEL}.npz", allow_pickle=True)
    paths = [str(p) for p in z["path"]]; y = z["gt"].astype(int)
    av.RECORDED.clear(); av.load_recorded(str(RES))
    meta = av.load_meta(str(RES / "videos_v3_metadata.csv"))
    idx = {p: av.sampled(meta[p], MODEL) for p in paths}
    vis = np.array([bool(av.visible(meta[p], MODEL)) for p in paths])
    cl = np.array([cls4(meta[p], idx[p]) for p in paths])
    marg = {r["path"]: float(r["margin"]) for r in csv.DictReader(open(RES / f"{MODEL}_expg3.csv")) if r["item"] == "order_real"}
    m = np.array([marg[p] for p in paths])
    keys = sorted([k for k in z.files if k.startswith("real_last_L")], key=lambda k: int(k.rsplit("L", 1)[1]))
    X = {int(k.rsplit("L", 1)[1]): z[k].astype(np.float32) for k in keys}
    layers = list(X)
    groups = np.arange(len(y)); yf = y.astype(float)
    metric = lambda s_, i_: float(np.mean(((s_ > .5) == (yf[i_] > .5))))

    # pooled nested probe (same construction as probe_stats.py)
    fo, fin, outer, inner = ps.build(X, yf, groups, 5, 4, np.random.default_rng(SEED))
    s_pool, _ = ps.run(yf, fin, outer, inner, layers, 4, metric)
    # pooled check
    out = dict(pooled=dict(n_visible=int(vis.sum()), probe_auc=auc(s_pool[vis], y[vis]), margin_auc=auc(m[vis], y[vis]),
                           probe_acc=float(np.mean(((s_pool > .5) == (y > .5))[vis]))))
    rng = np.random.default_rng(SEED)
    for c in CLASSES:
        sel = np.where(vis & (cl == c))[0]
        yc = y[sel]; mc = m[sel]
        # margin, sign fitted on training folds of the pooled outer split
        fc = fo[sel]; ms = np.empty(len(sel))
        for f in np.unique(fc):
            tr, te = fc != f, fc == f
            sg = 1.0 if auc(mc[tr], yc[tr]) >= .5 else -1.0
            ms[te] = sg * mc[te]
        # within-class nested probe
        Xc = {l: X[l][sel] for l in layers}
        fo2, fin2, out2, in2 = ps.build(Xc, yc.astype(float), np.arange(len(sel)), 5, 4, np.random.default_rng(SEED + 1))
        mt = lambda s_, i_, _y=yc.astype(float): float(np.mean(((s_ > .5) == (_y[i_] > .5))))
        s_in, chosen = ps.run(yc.astype(float), fin2, out2, in2, layers, 4, mt)
        null = []
        prng = np.random.default_rng(SEED + 2)
        for _ in range(200):
            yp = prng.permutation(yc).astype(float)
            mp = lambda s_, i_, _y=yp: float(np.mean(((s_ > .5) == (_y[i_] > .5))))
            sp, _ = ps.run(yp, fin2, out2, in2, layers, 4, mp)
            null.append(auc(sp, yp.astype(int)))
        a_pp, a_pw, a_m = auc(s_pool[sel], yc), auc(s_in, yc), auc(ms, yc)
        d1, d2 = [], []
        for _ in range(2000):
            i = rng.integers(0, len(sel), len(sel))
            d1.append(auc(s_pool[sel][i], yc[i]) - auc(ms[i], yc[i])); d2.append(auc(s_in[i], yc[i]) - auc(ms[i], yc[i]))
        q = lambda d: [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        out[c] = dict(n=len(sel), probe_pooled_auc=a_pp, probe_within_auc=a_pw, probe_within_null95=float(np.percentile(null, 95)),
                      margin_raw_auc=auc(mc, yc), margin_signfree_auc=a_m,
                      diff_pooled_minus_margin=a_pp - a_m, diff_pooled_ci=q(d1),
                      diff_within_minus_margin=a_pw - a_m, diff_within_ci=q(d2),
                      margin_heldout_bacc_signfree=heldout_bacc(mc, yc, np.random.default_rng(SEED + 3)),
                      layers_chosen=[int(a[0]) for a in chosen])
        print(c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[c].items()}), flush=True)
    print("pooled", out["pooled"])
    Path(ROOT / "longpaper/results_stats/within_class_qwen2.5-vl.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
