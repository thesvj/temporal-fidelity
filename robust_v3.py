"""Robustness analyses requested by the round-4 critics (2026-09-24). CPU only.

1. Same-answer conditional AUC: the order margin's AUC among trials that received the SAME answer. This is
   not bounded below by the answers' BAcc (all trials sit at one operating point), so it answers the
   "AUC >= BAcc is arithmetic" objection directly. Controlled set (visible trials) and TempCompass.
2. Recency confound of the move question: the "did N1 move?" margin on the 240 twin trials, split by
   whether N1 moved first or second, and the order margin's AUC after partialling out the move margin.
3. Stage stability across phrasings: AUC of the paraphrase (order_rev) and forced-choice margins.
4. Holm across the twin and solo tests (each a family over models).
Writes results_v3/robust_v3.json.
"""
import json
from pathlib import Path

import numpy as np

import analyze_v3 as av

RES, META = "results_v3", "results_v3/videos_v3_metadata.csv"
B = 5000


def auc_test(pos, neg, rng, pairs=None):
    a = av.auc(pos, neg)
    if a is None:
        return dict(auc=None)
    lo, hi = av.boot_ci(pos, neg, 2000, rng, pairs=pairs)
    p, c, n = av.perm_p(pos, neg, B, rng, pairs=pairs)
    return dict(auc=a, ci=[lo, hi], p=p, n_pos=len(pos), n_neg=len(neg))


def main():
    meta = av.load_meta(META)
    av.load_recorded(RES)
    rng = np.random.default_rng(20260924)
    out = {}
    for m in av.MODELS:
        rows = av.load_rows(RES, m, meta)
        if not rows:
            continue
        t = out.setdefault(m, {})
        o = [r for r in av.by_item(rows, "order_real") if r["gt"] in ("0", "1") and r["_m"] is not None
             and av.visible(r["_meta"], m)]
        # 1. conditional on the answer
        for ans in ("0", "1"):
            g = [r for r in o if r["pred"] == ans]
            pos = [r["_m"] for r in g if r["gt"] == "1"]
            neg = [r["_m"] for r in g if r["gt"] == "0"]
            if len(pos) >= 20 and len(neg) >= 20:
                t[f"cond_auc_ans{ans}"] = auc_test(pos, neg, rng)
        # 3. other phrasings (all visible trials)
        for item in ("order_rev", "order_forced"):
            g = [r for r in av.by_item(rows, item) if r["gt"] in ("0", "1") and r["_m"] is not None
                 and av.visible(r["_meta"], m)]
            if len(g) >= av.MIN_N:
                t[f"auc_{item}"] = auc_test([r["_m"] for r in g if r["gt"] == "1"],
                                            [r["_m"] for r in g if r["gt"] == "0"], rng)
        # 2. recency: move margin on real clips split by whether the queried (first-named) shape moved first
        mv = {r["path"]: r for r in av.by_item(rows, "moved_real") if r["_m"] is not None}
        od = {r["path"]: r for r in o}
        common = [p for p in mv if p in od]
        if len(common) >= 40:
            first = [mv[p]["_m"] for p in common if od[p]["gt"] == "1"]
            second = [mv[p]["_m"] for p in common if od[p]["gt"] == "0"]
            t["move_margin_by_order"] = dict(queried_first=float(np.mean(first)),
                                             queried_second=float(np.mean(second)),
                                             **auc_test(first, second, rng))
            x = np.array([mv[p]["_m"] for p in common])
            y = np.array([od[p]["_m"] for p in common])
            g = np.array([int(od[p]["gt"]) for p in common])
            r_xy = float(np.corrcoef(x, y)[0, 1])
            beta = np.polyfit(x, y, 1)
            resid = y - np.polyval(beta, x)
            t["order_margin_on_twin_subset"] = auc_test(list(y[g == 1]), list(y[g == 0]), rng)
            t["order_margin_partial_move"] = dict(r_move_order=r_xy,
                                                  **auc_test(list(resid[g == 1]), list(resid[g == 0]), rng))
            # twin-separation split by recency: does AUC_twin depend on whether the queried shape moved last?
            tw = {r["path"]: r for r in av.by_item(rows, "moved_twin") if r["_m"] is not None}
            for lab, gv in (("queried_first", "1"), ("queried_second", "0")):
                pr = [(mv[p]["_m"], tw_r["_m"]) for p in common if od[p]["gt"] == gv
                      for tw_r in [next((v for k, v in tw.items() if v["_meta"] and v["_meta"]["path"] == p), None)]
                      if tw_r is not None]
                if len(pr) >= 20:
                    t[f"auc_twin_{lab}"] = auc_test([a for a, _ in pr], [b for _, b in pr], rng, pairs=pr)
    # 4. Holm within twin and within solo families, from numbers_v3.json
    num = json.loads(Path(RES, "numbers_v3.json").read_text())
    for fam in ("auc_twin_p", "auc_solo_p"):
        for scale in (False, True):   # 7-8B and 72-78B are separate families, as for the order test
            ks = [k for k in num if num[k].get(fam) is not None and (k in av.SCALE_CHECK) == scale]
            for k, v in zip(ks, av.holm([num[k][fam] for k in ks])):
                out.setdefault(k, {})[fam + "_holm"] = v
    Path(RES, "robust_v3.json").write_text(json.dumps(out, indent=1, default=float))
    for m, t in out.items():
        print(f"== {av.NICE[m]}")
        for k, v in t.items():
            if isinstance(v, dict):
                print(f"   {k:32s} " + " ".join(f"{a}={(round(b, 3) if isinstance(b, float) else b)}" for a, b in v.items()))
            else:
                print(f"   {k:32s} {v:.4f}")


if __name__ == "__main__":
    main()
