"""Analysis for the long-paper experiments (stdlib only).

  python3 analyze_long.py --fetch      # scp *_expg.csv, *_tc_yesno.csv, *_tc_margin.csv from $REMOTE (user@host:/path/to/repo) into results_long/
  python3 analyze_long.py              # print Exp G + TempCompass yes/no tables

Exp G: order BAcc / margin AUC (all, visible-analytic), 32-frame, motion detection, flipped null, no-evidence nulls.
TempCompass: per condition Acc, P(YES), H, F, BAcc [video-bootstrap CI], d', c; real - static.
"""
import argparse
import csv
import glob
import math
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

Z = NormalDist().inv_cdf
OUT = Path("results_long")
MODELS = ["molmo2", "videochat-flash", "qwen2.5-vl", "internvl2.5", "llava-next-video", "video-llama2",
          "qwen2.5-vl-72b", "internvl2.5-78b"]


def fetch():
    import os
    OUT.mkdir(exist_ok=True)
    rem = os.environ["REMOTE"]  # e.g. user@host:/path/to/repo
    subprocess.run(f"scp -q '{rem}/results/*_expg*.csv' '{rem}/results/*_tc_yesno.csv' "
                   f"'{rem}/results/*_tc_margin.csv' results_long/", shell=True)


def auc(pos, neg):
    pos = [x for x in pos if x is not None]; neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def sdt(rows, gk="gt", pk="pred"):
    rows = [r for r in rows if str(r[pk]) in ("0", "1") and str(r[gk]) in ("0", "1")]
    s = [r for r in rows if str(r[gk]) == "1"]; n = [r for r in rows if str(r[gk]) == "0"]
    if not s or not n:
        return None
    h = sum(str(r[pk]) == "1" for r in s); fa = sum(str(r[pk]) == "1" for r in n)
    H, F = h / len(s), fa / len(n)
    Hc, Fc = (h + .5) / (len(s) + 1), (fa + .5) / (len(n) + 1)
    return dict(n=len(rows), acc=sum(str(r[pk]) == str(r[gk]) for r in rows) / len(rows),
                pyes=(h + fa) / len(rows), H=H, F=F, bacc=(H + 1 - F) / 2,
                d=Z(Hc) - Z(Fc), c=-(Z(Hc) + Z(Fc)) / 2)


def f(x, k=3):
    return "--" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))


def interval_of(path):
    m = re.search(r"int_(\d+)ms", path)
    return int(m.group(1)) if m else None


SAMPLERS = {                      # frames each pipeline receives, as verified in the adapters
    "uniform8": lambda T: [int(i * T / 8) for i in range(8)],
    "midpoint8": lambda T: [int((i + 0.5) * T / 8) for i in range(8)],
    "molmo2": lambda T: sorted({min(T - 1, int(round(t * 30))) for t in
                                [x * 0.5 for x in range(int((T / 30) / 0.5) + 1)] + [T / 30]}),
    "vcf64": lambda T: [int(i * T / 64) for i in range(64)] if T >= 64 else list(range(T)),
}
MODEL_SAMPLER = {"molmo2": "molmo2", "videochat-flash": "vcf64", "video-llama2": "midpoint8",
                 "qwen2.5-vl": "uniform8", "qwen2.5-vl-72b": "uniform8", "internvl2.5": "uniform8",
                 "internvl2.5-78b": "uniform8", "llava-next-video": "uniform8"}


_V2META = None


def v2meta():
    global _V2META
    if _V2META is None:
        import csv as _csv
        _V2META = {r["path"]: r for r in _csv.DictReader(open(OUT / "videos_v2_metadata.csv"))}
    return _V2META


def visible_v2(path, model):
    """v2 stimulus metadata carries the exact jump frames, so visibility is exact per model."""
    row = v2meta().get(str(path))
    if row is None:
        return None
    T = int(float(row["n_frames"])); a = int(float(row["f_first"])); b = int(float(row["f_second"]))
    lo, hi = min(a, b), max(a, b)
    idx = SAMPLERS[MODEL_SAMPLER.get(model, "uniform8")](T)
    return any(lo <= i < hi for i in idx)


def visible_8(dt_ms):
    """base.load_frames rule, 30 fps stimuli: idx=int(i*total/8); visible if first<=idx<second."""
    total = int((1.0 + max(dt_ms / 1000, 0.1) + 1.5) * 30)
    a, b = 30, 30 + int(dt_ms / 1000 * 30)
    return any(a <= int(i * total / 8) < b for i in range(8))


def margin(r):
    try:
        return float(r["margin"])
    except (TypeError, ValueError):
        return None


def cv_threshold_bacc(rows, folds=2, seed=0):
    """Threshold on the YES-NO margin chosen on one half (maximising BAcc, sign fixed: YES = larger margin),
    evaluated on the other half; averaged over folds. Recovers what a recalibrated decision would score."""
    rows = [r for r in rows if margin(r) is not None and r["gt"] in ("0", "1")]
    rng = random.Random(seed); idx = list(range(len(rows))); rng.shuffle(idx)
    parts = [idx[i::folds] for i in range(folds)]
    def bacc_at(rs, t):
        pos = [r for r in rs if r["gt"] == "1"]; neg = [r for r in rs if r["gt"] == "0"]
        if not pos or not neg:
            return None
        return (sum(margin(r) > t for r in pos) / len(pos) + sum(margin(r) <= t for r in neg) / len(neg)) / 2
    out = []
    for k in range(folds):
        train = [rows[i] for j, pt in enumerate(parts) if j != k for i in pt]
        test = [rows[i] for i in parts[k]]
        cands = sorted({margin(r) for r in train})
        if not cands:
            continue
        best = max(cands, key=lambda t: bacc_at(train, t) or 0)
        v = bacc_at(test, best)
        if v is not None:
            out.append(v)
    return sum(out) / len(out) if out else None


def exp_g():
    print("\n=== Exp G ===")
    hdr = ("model", "cov", "ordBAcc", "ordAUC", "ordAUC_vis", "AUC32", "BAcc32", "movBAcc", "movY_real", "movY_twin",
           "movAUC", "stillY_twin", "stillY_real", "ordY_twin", "ordY_blank")
    print("\t".join(hdr))
    for m in MODELS:
        p2 = OUT / f"{m}_expg2.csv"
        p = p2 if p2.exists() else OUT / f"{m}_expg.csv"       # prefer the unique-stimulus re-run
        if not p.exists():
            continue
        if p is p2:
            print(f"[{m}] v2 (unique stimuli)")
        R = list(csv.DictReader(open(p)))
        by = defaultdict(list)
        for r in R:
            by[r["item"]].append(r)
        cov = sum(margin(r) is not None for r in R) / max(1, len(R))
        o = by["order_real"]
        s_o = sdt(o)
        a_all = auc([margin(r) for r in o if r["gt"] == "1"], [margin(r) for r in o if r["gt"] == "0"])
        ov = [r for r in o if visible_8(interval_of(r["path"]))]
        a_vis = auc([margin(r) for r in ov if r["gt"] == "1"], [margin(r) for r in ov if r["gt"] == "0"])
        o32 = by["order_32f"]
        a32 = auc([margin(r) for r in o32 if r["gt"] == "1"], [margin(r) for r in o32 if r["gt"] == "0"]) if o32 else None
        s32 = sdt(o32) if o32 else None
        mv = by["moved_real"] + by["moved_twin"]
        s_mv = sdt(mv)
        pY = lambda rows: (sum(r["pred"] == "1" for r in rows) / len(rows)) if rows else None
        a_mv = auc([margin(r) for r in by["moved_real"]], [margin(r) for r in by["moved_twin"]])
        print("\t".join([m, f(cov, 2), f(s_o["bacc"] if s_o else None), f(a_all), f(a_vis), f(a32),
                         f(s32["bacc"] if s32 else None), f(s_mv["bacc"] if s_mv else None),
                         f(pY(by["moved_real"])), f(pY(by["moved_twin"])), f(a_mv),
                         f(pY(by["still_twin"])), f(pY(by["still_real"])), f(pY(by["order_twin"])),
                         f(pY(by["order_blank"]))]))
        per = defaultdict(list)
        for r in o:
            per[interval_of(r["path"])].append(r)
        print("   CV-threshold BAcc order all / visible:", f(cv_threshold_bacc(o)), f(cv_threshold_bacc(ov)),
              "| moved:", f(cv_threshold_bacc(mv)))
        print("   per-interval order AUC:", {k: f(auc([margin(r) for r in v if r["gt"] == "1"],
                                                    [margin(r) for r in v if r["gt"] == "0"]), 2)
                                           for k, v in sorted(per.items())})
        print("   mean margin A-first / B-first / twin / blank:",
              [f(sum(x) / len(x) if x else None, 2) for x in (
                  [margin(r) for r in o if r["gt"] == "1" and margin(r) is not None],
                  [margin(r) for r in o if r["gt"] == "0" and margin(r) is not None],
                  [margin(r) for r in by["order_twin"] if margin(r) is not None],
                  [margin(r) for r in by["order_blank"] if margin(r) is not None])])


def boot_bacc(rows, key="video_id", B=1000, seed=0):
    vids = defaultdict(list)
    for r in rows:
        vids[r[key]].append(r)
    ks = list(vids)
    rng = random.Random(seed); vals = []
    for _ in range(B):
        samp = [r for k in (rng.choice(ks) for _ in ks) for r in vids[k]]
        s = sdt(samp, gk="gold")
        if s:
            vals.append(s["bacc"])
    vals.sort()
    return (vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]) if vals else (None, None)


def tempcompass():
    print("\n=== TempCompass yes/no ===")
    print("model\tcond\tn\tparse\tacc\tpyes\tH\tF\tbacc\tCI\td'\tc")
    summary = {}
    for m in MODELS:
        p = OUT / f"{m}_tc_yesno.csv"
        if not p.exists():
            continue
        R = list(csv.DictReader(open(p)))
        seen = set(); dedup = []
        for r in R:                      # two jobs may append the same item; keep the first
            k = (r["condition"], r["video_id"], r["question"])
            if k not in seen:
                seen.add(k); dedup.append(r)
        R = dedup
        for cond in ("real", "static", "blank"):
            rows = [r for r in R if r["condition"] == cond]
            if not rows:
                continue
            parse = sum(r["pred"] in ("0", "1") for r in rows) / len(rows)
            s = sdt(rows, gk="gold")
            if not s:
                continue
            lo, hi = boot_bacc([r for r in rows if r["pred"] in ("0", "1")])
            summary[(m, cond)] = s
            print("\t".join([m, cond, str(len(rows)), f(parse, 2), f(s["acc"]), f(s["pyes"]), f(s["H"]), f(s["F"]),
                             f(s["bacc"]), f"[{f(lo)},{f(hi)}]", f(s["d"], 2), f(s["c"], 2)]))
            for dim in ("order", "direction"):
                sd = sdt([r for r in rows if r["dim"] == dim], gk="gold")
                if sd:
                    print(f"\t  {dim}: bacc={f(sd['bacc'])} pyes={f(sd['pyes'])} d'={f(sd['d'], 2)}")
    ms = [m for m in MODELS if (m, "real") in summary]
    if len(ms) >= 3:
        def kendall(a, b):
            c = d = 0
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    s = (a[i] - a[j]) * (b[i] - b[j])
                    c += s > 0; d += s < 0
            return (c - d) / max(1, c + d)
        acc = [summary[(m, "real")]["acc"] for m in ms]; dp = [summary[(m, "real")]["d"] for m in ms]
        print("Kendall tau(acc, d') real:", f(kendall(acc, dp), 2), "n=", len(ms))
        for m in ms:
            if (m, "static") in summary:
                print(f"  {m}: real-static BAcc = {summary[(m,'real')]['bacc'] - summary[(m,'static')]['bacc']:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    exp_g()
    tempcompass()
