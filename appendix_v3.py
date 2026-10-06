"""Appendix numbers for the v3 run: interval estimation, prompt variants, invisible-trial readouts,
per-interval order BAcc and no-evidence answers. Reuses analyze_v3's loaders so visibility is the same
recorded-frame rule as Table 1. Writes results_v3/appendix_v3.json and results_v3/appendix_rows.tex.

Usage: uv run --no-project --with numpy --with scipy python appendix_v3.py
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import norm

import analyze_v3 as av

RES, META = "results_v3", "results_v3/videos_v3_metadata.csv"
BINS = "ABCD"


def f3(x):
    return av.f3(x)


def sdt(rs):
    a = [r for r in rs if r["pred"] in ("0", "1")]
    if not a:
        return None
    s = [r for r in a if r["gt"] == "1"]
    n = [r for r in a if r["gt"] == "0"]
    h = sum(r["pred"] == "1" for r in s)
    fa = sum(r["pred"] == "1" for r in n)
    hr, far = (h + .5) / (len(s) + 1), (fa + .5) / (len(n) + 1)
    py = sum(r["pred"] == "1" for r in a) / len(a)
    degenerate = py in (0.0, 1.0)
    return dict(n=len(a), py=py, bacc=av.bacc(a), c=None if degenerate else -.5 * (norm.ppf(hr) + norm.ppf(far)),
                dprime=None if degenerate else norm.ppf(hr) - norm.ppf(far))


def kappa_w(gt, pr):
    k = len(BINS)
    idx = {b: i for i, b in enumerate(BINS)}
    O = np.zeros((k, k))
    for g, p in zip(gt, pr):
        O[idx[g], idx[p]] += 1
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den > 0 else 0.0


def main():
    meta = av.load_meta(META)
    av.load_recorded(RES)
    out, tex = {}, []
    rng = np.random.default_rng(20260924)
    for m in av.MODELS:
        rows = av.load_rows(RES, m, meta)
        if not rows:
            continue
        t = out.setdefault(m, {})
        # ---- interval estimation
        iv = [r for r in av.by_item(rows, "interval") if r["pred"] in tuple(BINS)]
        if iv:
            cnt = Counter(r["pred"] for r in iv)
            p = {b: cnt[b] / len(iv) for b in BINS}
            gt = [r["gt"] for r in iv]
            pr = [r["pred"] for r in iv]
            rec = [np.mean([x == g for x, y in zip(pr, gt) if y == g]) for g in BINS if g in gt]
            H = -sum(v * math.log2(v) for v in p.values() if v > 0)
            gcnt = Counter(gt)
            conf = {g: {b: sum(1 for x, y in zip(pr, gt) if y == g and x == b) / gcnt[g] for b in BINS}
                    for g in BINS if gcnt[g]}
            t["interval"] = dict(n=len(iv), p=p, acc=float(np.mean([x == y for x, y in zip(pr, gt)])),
                                 bacc=float(np.mean(rec)), kappa_w=float(kappa_w(gt, pr)), H=H,
                                 majority=max(gcnt.values()) / len(gt), gold=dict(gcnt), confusion=conf)
        # ---- prompt variants
        t["variants"] = {}
        for item in ("order_real", "order_rev", "order_forced", "order_32f"):
            s = sdt(av.by_item(rows, item))
            if s:
                t["variants"][item] = s
        # ---- visible / invisible readouts on the order question
        o = [r for r in av.by_item(rows, "order_real") if r["gt"] in ("0", "1")]
        inv = [r for r in o if not av.visible(r["_meta"], m)]
        vis = [r for r in o if av.visible(r["_meta"], m)]
        t["n_invisible"] = len(inv)
        if inv:
            mi = [r for r in inv if r["_m"] is not None]
            pos = [r["_m"] for r in mi if r["gt"] == "1"]
            neg = [r["_m"] for r in mi if r["gt"] == "0"]
            a = av.auc(pos, neg)
            p, c, n = av.perm_p(pos, neg, 2000, rng) if a is not None else (None, None, None)
            t["invisible"] = dict(n=len(inv), bacc=av.bacc([r for r in inv if r["pred"] in ("0", "1")]),
                                  auc=a, p=p)
        # ---- per-interval order BAcc (all trials, and visible only)
        per = {}
        for ms in sorted({int(r["interval_ms"]) for r in o}):
            g = [r for r in o if int(r["interval_ms"]) == ms and r["pred"] in ("0", "1")]
            gv = [r for r in g if av.visible(r["_meta"], m)]
            per[ms] = dict(bacc=av.bacc(g), bacc_vis=av.bacc(gv) if len(gv) >= 10 else None, n_vis=len(gv))
        t["per_interval"] = per
        # ---- Molmo2-style: BAcc by number of received between-jumps frames
        nb = {}
        for r in vis:
            got = av.sampled(r["_meta"], m)
            a_, b_ = int(float(r["_meta"]["f_first"])), int(float(r["_meta"]["f_second"]))
            k = sum(min(a_, b_) <= i < max(a_, b_) for i in got)
            nb.setdefault(min(k, 2), []).append(r)
        t["bacc_by_nbetween"] = {k: (len(v), av.bacc([r for r in v if r["pred"] in ("0", "1")]))
                                 for k, v in sorted(nb.items())}
        # ---- no-evidence answers
        for item in ("order_blank", "order_twin"):
            b = [r for r in av.by_item(rows, item) if r["pred"] in ("0", "1")]
            if b:
                t[item + "_py"] = sum(r["pred"] == "1" for r in b) / len(b)
        # ---- appendix rows
        if "interval" in t:
            q = t["interval"]
            tex.append(f"% interval {m}\n{av.NICE[m]:17} & " + " & ".join(f3(q['p'][b]) for b in BINS) +
                       f" & {f3(q['acc'])} & {f3(q['bacc'])} & {q['kappa_w']:.2f} & {q['H']:.2f} \\\\")
        for item, lab in (("order_real", "Original"), ("order_rev", "Paraphrase"), ("order_forced", "Forced"),
                          ("order_32f", "32 frames")):
            s = t["variants"].get(item)
            if s:
                c = "degen." if s["c"] is None else f"${s['c']:+.2f}$".replace("-", "$-$").replace("$$", "")
                tex.append(f"% variant {m} {lab}: P(Y) {f3(s['py'])} BAcc {f3(s['bacc'])} c {c} n {s['n']}")
    Path(RES, "appendix_v3.json").write_text(json.dumps(out, indent=1, default=float))
    Path(RES, "appendix_rows.tex").write_text("\n".join(tex) + "\n")
    for m, t in out.items():
        inv = t.get("invisible", {})
        print(f"{av.NICE[m]:17} inv n={inv.get('n')} bacc={f3(inv.get('bacc'))} auc={f3(inv.get('auc'))} "
              f"p={inv.get('p')}  blank P(Y)={f3(t.get('order_blank_py'))} twin P(Y)={f3(t.get('order_twin_py'))}"
              f"  nbetween={ {k: (n, f3(b)) for k, (n, b) in t['bacc_by_nbetween'].items()} }")
    print(open(Path(RES, "appendix_rows.tex")).read())


if __name__ == "__main__":
    main()
