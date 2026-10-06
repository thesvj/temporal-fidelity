"""Does Qwen2.5-VL's order margin depend on how the delivered frames pair up in its temporal patches? CPU only.

Qwen2.5-VL fuses delivered frames in pairs (temporal patch size 2): slots (0,1), (2,3), ... Let s be the slot of
the first delivered frame at or after the first event and e the same for the second event. A change that
falls INSIDE a patch (s odd: the patch holds one frame before and one after) is fused into one token; a
change that falls BETWEEN patches (s even) separates two tokens. Classes: s_odd / s_even_e_odd / s_even_e_even.

Arms (run_arms.sh; models/qwen25_vl.arm_frames), order question only:
  harness  the paper's input            shift  same frames one slot later (alignment only)
  ts       same frames, true fps        native 2 fps sampling (compound)
For each arm: margin AUC on arm-visible trials, pooled, by the arm's own alignment class, and by the class
the trial has under the harness; on real footage the paired AUC with a cluster bootstrap over authors.
Predictions fixed before the run: reviews/mainpush_1002/plan_v2.md.

Usage: uv run --no-project --with numpy python analyze_arms.py [--model qwen2.5-vl] [--boot 5000]
Writes results_arms/<model>.json.
"""
import argparse
import json
from pathlib import Path

import zlib

import numpy as np

import analyze_v3 as av
import analyze_real as ar

def rng_for(*key):
    """One generator per statistic, so adding an arm or a class never changes another interval."""
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


CLASSES = ("s_odd", "s_even_e_odd", "s_even_e_even")


def klass(meta_row, idx):
    a, b = sorted((int(float(meta_row["f_first"])), int(float(meta_row["f_second"]))))
    s = next((k for k, i in enumerate(idx) if i >= a), None)
    e = next((k for k, i in enumerate(idx) if i >= b), None)
    if s is None or e is None:
        return "na"
    return "s_odd" if s % 2 else "s_even_e_odd" if e % 2 else "s_even_e_even"


def auc(rows):
    p = [r["_m"] for r in rows if r["gt"] == "1"]; n = [r["_m"] for r in rows if r["gt"] == "0"]
    return av.auc(p, n) if p and n else None


def boot(rows, B, rng):
    g = np.array([int(r["gt"]) for r in rows]); m = np.array([r["_m"] for r in rows])
    out = []
    for _ in range(B):
        i = rng.integers(0, len(g), len(g))
        v = av.auc(m[i][g[i] == 1], m[i][g[i] == 0])
        if v is not None:
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def load(res, model):
    av.RECORDED.clear(); av.load_recorded(res)
    meta = av.load_meta(f"{res}/videos_v3_metadata.csv")
    rows = av.load_rows(res, model, meta)
    if not rows:
        return None, None
    key = "qwen2.5-vl"                                   # arm_frames logs under this name for both sizes
    idx = {p: av.RECORDED[(key, p)] for (k, p) in av.RECORDED if k == key}
    return [r for r in ar.order_rows(rows)], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    out = {}
    for SET, full in (("v3", "results_v3"), ("real", "results_real")):
        base, bidx = load(full, args.model)
        if base is None:
            continue
        hcls = {r["path"]: klass(r["_meta"], bidx[r["path"]]) for r in base}
        hm = {r["path"]: r["_m"] for r in base}
        out[SET] = {}
        for arm in ("harness", "shift", "shift2", "ts", "native"):
            res = full if arm == "harness" else f"results_arm_{arm}_{SET}"
            if not Path(res, f"{args.model}_expg3.csv").exists():
                continue
            rows, idx = load(res, args.model)
            d = dict(n=len(rows))
            if arm == "harness" and Path(f"results_arm_harness_{SET}", f"{args.model}_expg3.csv").exists():
                rep, _ = load(f"results_arm_harness_{SET}", args.model)     # reproduction of the stored run
                d["reproduction_max_abs_margin_diff"] = float(max(abs(r["_m"] - hm[r["path"]]) for r in rep))
                d["reproduction_n"] = len(rep)
            K = int(float(rows[0]["_meta"].get("burst_frames") or 0))

            def visible(r):
                a, b = sorted((int(float(r["_meta"]["f_first"])), int(float(r["_meta"]["f_second"]))))
                return any(a <= i < b or (K and a + K - 1 <= i < b + K - 1) for i in idx[r["path"]])
            vis = [r for r in rows if visible(r)]
            d.update(n_visible=len(vis), auc=auc(vis), ci=boot(vis, args.boot, rng_for(args.model, SET, arm, "pooled")),
                     bacc=av.bacc([r for r in vis if r["pred"] in ("0", "1")]),
                     py=float(np.mean([r["pred"] == "1" for r in vis])))
            d["by_own_class"] = {c: dict(n=len(s), auc=auc(s), ci=boot(s, args.boot, rng_for(args.model, SET, arm, "own", c))) for c in CLASSES
                                 if len(s := [r for r in vis if klass(r["_meta"], idx[r["path"]]) == c]) >= 20}
            d["by_harness_class"] = {c: dict(n=len(s), auc=auc(s), ci=boot(s, args.boot, rng_for(args.model, SET, arm, "hcls", c))) for c in CLASSES
                                     if len(s := [r for r in vis if hcls.get(r["path"]) == c]) >= 20}
            # finer: harness class crossed with harness e parity for s_odd (the reviewer's four-way split)
            if SET == "v3":
                def four(p):
                    a, b = sorted((int(float(base_meta[p]["f_first"])), int(float(base_meta[p]["f_second"]))))
                    s = next((k for k, i in enumerate(bidx[p]) if i >= a), None); e = next((k for k, i in enumerate(bidx[p]) if i >= b), None)
                    return None if s is None or e is None else f"s{'odd' if s % 2 else 'even'}_e{'odd' if e % 2 else 'even'}"
                base_meta = {r["path"]: r["_meta"] for r in base}
                d["by_harness_four"] = {c: dict(n=len(s), auc=auc(s), ci=boot(s, args.boot, rng_for(args.model, SET, arm, "four", c))) for c in ("sodd_eeven", "sodd_eodd", "seven_eodd", "seven_eeven")
                                        if len(s := [r for r in vis if four(r["path"]) == c]) >= 20}
                # the same, on trials that still have a delivered frame after the second jump under this arm
                d["by_harness_four_kept"] = {c: dict(n=len(s), auc=auc(s), ci=boot(s, args.boot, rng_for(args.model, SET, arm, "kept", c))) for c in ("sodd_eeven", "sodd_eodd", "seven_eodd", "seven_eeven")
                                             if len(s := [r for r in vis if four(r["path"]) == c and klass(r["_meta"], idx[r["path"]]) != "na"]) >= 20}
                d["one_jump"] = dict(n=len(s := [r for r in vis if klass(r["_meta"], idx[r["path"]]) == "na"]), auc=auc(s))

                def s_par(r):                            # parity of the slot of the first delivered frame after the first jump
                    a = min(int(float(r["_meta"]["f_first"])), int(float(r["_meta"]["f_second"])))
                    k = next((k for k, i in enumerate(idx[r["path"]]) if i >= a), None)
                    return None if k is None else "inside" if k % 2 else "between"
                d["one_jump_by"] = {c: dict(n=len(q), auc=auc(q)) for c in ("inside", "between")
                                    if len(q := [r for r in s if s_par(r) == c]) >= 20}
            if arm != "harness":                         # paired difference from the harness on the same trials
                common = [r for r in vis if r["path"] in hm]
                g = np.array([int(r["gt"]) for r in common]); ma = np.array([r["_m"] for r in common]); mh = np.array([hm[r["path"]] for r in common])
                ds = []; rng = rng_for(args.model, SET, arm, "diff")
                for _ in range(args.boot):
                    i = rng.integers(0, len(g), len(g))
                    x, y = av.auc(ma[i][g[i] == 1], ma[i][g[i] == 0]), av.auc(mh[i][g[i] == 1], mh[i][g[i] == 0])
                    if x is not None and y is not None:
                        ds.append(x - y)
                d["minus_harness_same_trials"] = dict(n=len(common), arm=av.auc(ma[g == 1], ma[g == 0]), harness=av.auc(mh[g == 1], mh[g == 0]),
                                                      diff_ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))])
            if SET == "real":
                pa, pci, pp, npair = ar.paired_auc(ar.pairs_of(vis), args.boot, rng_for(args.model, SET, arm, "paired"))
                d.update(paired_auc=pa, paired_ci=pci, paired_p=pp, n_pairs=npair)
            out[SET][arm] = d
            print(f"{SET:4} {arm:8} n_vis {d['n_visible']:4} AUC {d['auc']:.3f} {d['ci']} BAcc {d['bacc']:.3f} P(Y) {d['py']:.3f}"
                  + (f" | paired {d['paired_auc']:.3f} {d['paired_ci']}" if SET == "real" else "")
                  + (f" | repro max|dm| {d.get('reproduction_max_abs_margin_diff')}" if arm == "harness" else ""))
            print("      own class:    ", {c: (v["n"], round(v["auc"], 3)) for c, v in d["by_own_class"].items()})
            print("      harness class:", {c: (v["n"], round(v["auc"], 3)) for c, v in d["by_harness_class"].items()})
            if "by_harness_four" in d:
                print("      harness 4-way:", {c: (v["n"], round(v["auc"], 3)) for c, v in d["by_harness_four"].items()})
            if "by_harness_four_kept" in d:
                print("      4-way, kept:  ", {c: (v["n"], round(v["auc"], 3)) for c, v in d["by_harness_four_kept"].items()}, "| one-jump", d["one_jump"]["n"], d["one_jump"]["auc"] and round(d["one_jump"]["auc"], 3), {c: (v["n"], round(v["auc"], 3)) for c, v in d["one_jump_by"].items()})
            if "minus_harness_same_trials" in d:
                print("      vs harness, same trials:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d["minus_harness_same_trials"].items()})
    Path("results_arms").mkdir(exist_ok=True)
    Path("results_arms", f"{args.model}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
