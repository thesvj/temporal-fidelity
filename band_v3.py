"""Which model resolves the order at which inter-event interval. CPU only.

Pools the 12 intervals into the four bins the interval-estimation question already uses
(A: <1 s, B: [1, 2) s, C: [2, 5) s, D: >=5 s), so the bands are not chosen after seeing the result.
Bands are named by the intervals actually generated: 67-500 ms, 1-1.5 s, 2-3 s, 5-10 s.
Per model and band, on visible trials: n, AUC of the first-token YES-NO margin with bootstrap 95% CI,
two-sided asymptotic Mann-Whitney p (no Monte Carlo error), Holm over every model x band cell of the
family (six 7-8B models; the 72-78B pair separately, as in analyze_v3), and BAcc of the answers.

  margin resolves a band : Holm p < .05 and AUC > .5
  answers resolve a band : BAcc >= bar (.60, the bar for `answered` in analyze_v3.stage)

Writes results_v3/band_v3.json and results_v3/band_rows.tex.
Usage: uv run --no-project --with numpy python band_v3.py [--boot 10000]
"""
import argparse
import json

import numpy as np

import analyze_v3 as av
from profile_v3 import mw_p

BANDS = [("67--500\\,ms", 0, 1000), ("1--1.5\\,s", 1000, 2000), ("2--3\\,s", 2000, 5000), ("5--10\\,s", 5000, 10 ** 9)]
BAR = .60


def f2(x):
    return "--" if x is None else f"{x:.2f}".replace("0.", ".", 1) if x < 1 else "1.00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v3")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    meta = av.load_meta(f"{args.res}/videos_v3_metadata.csv")
    av.load_recorded(args.res)

    out = {}
    for m in av.MODELS:
        rows = av.load_rows(args.res, m, meta)
        if not rows:
            continue
        v = [r for r in av.by_item(rows, "order_real")
             if r["gt"] in ("0", "1") and r["_m"] is not None and av.visible(r["_meta"], m)]
        if len(v) < av.MIN_N:
            continue
        out[m] = {}
        for name, lo, hi in BANDS:
            s = [r for r in v if lo <= int(r["interval_ms"]) < hi]
            pos = [r["_m"] for r in s if r["gt"] == "1"]; neg = [r["_m"] for r in s if r["gt"] == "0"]
            ci = av.boot_ci(pos, neg, args.boot, rng)
            out[m][name] = dict(n=len(s), auc=av.auc(pos, neg), lo=ci[0], hi=ci[1],
                                p=mw_p(np.array(pos), np.array(neg)),
                                bacc=av.bacc([r for r in s if r["pred"] in ("0", "1")]))

    for fam in ([m for m in out if m not in av.SCALE_CHECK], [m for m in out if m in av.SCALE_CHECK]):
        cells = [(m, b) for m in fam for b, _, _ in BANDS]
        for (m, b), ph in zip(cells, av.holm([out[m][b]["p"] for m, b in cells])):
            c = out[m][b]
            c["p_holm"] = ph
            c["margin"] = bool(ph < .05 and c["auc"] > .5)
            c["answer"] = bool((c["bacc"] or 0) >= BAR)

    names = [b for b, _, _ in BANDS]
    print(f"{'model':18} " + " ".join(f"{b:>30}" for b in ("67-500 ms", "1-1.5 s", "2-3 s", "5-10 s")))
    lines = []
    for m, d in out.items():
        print(f"{av.NICE[m]:18} " + " ".join(
            f"{d[b]['auc']:.2f} [{d[b]['lo']:.2f},{d[b]['hi']:.2f}] pH={d[b]['p_holm']:.3f} "
            f"B={d[b]['bacc']:.2f} n={d[b]['n']:<4}" for b in names))
        cell = []
        for b in names:
            c = d[b]
            a = f2(c["auc"]) + ("" if c["margin"] else "$^{\\dagger}$")
            ba = f"\\textbf{{{f2(c['bacc'])}}}" if c["answer"] else f2(c["bacc"])
            cell.append(f"{a} & {ba}")
        ans = [b for b in names if d[b]["answer"]]; mar = [b for b in names if d[b]["margin"]]
        span = lambda x: "all" if len(x) == 4 else "none" if not x else ", ".join(x)
        lines.append(f"{av.NICE[m]:17} & " + " & ".join(cell) + f" & {span(mar)} & {span(ans)} \\\\")
    json.dump(out, open(f"{args.res}/band_v3.json", "w"), indent=1)
    open(f"{args.res}/band_rows.tex", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
