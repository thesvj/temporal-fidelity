"""Stage profile per model: where the order evidence is lost, on one scale. CPU only.

Requested by the round-6/7 critics (2026-09-25). For every model, on visible trials:

  1. answers: P(YES), BAcc, its ceiling .5 + P(YES) (balanced labels, P(YES) < .5), d' with bootstrap CI,
     criterion c, and Fisher's exact test of YES-rate on positives vs negatives (are the answers above
     chance even when BAcc looks like chance?);
  2. loss budget: probe accuracy on the last-token state (best layer by all-trial accuracy, as in the
     paper's probe table; and the final layer), held-out threshold on the margin, answers;
  3. matched criterion, on the trials visible to BOTH models of a pair: BAcc of each margin thresholded
     so that P(YES) equals the other model's (threshold set by the answer rate, never by the label), with
     the ceiling that rate imposes, and the paired AUC difference;
  4. stage under each phrasing (original, paraphrase, forced choice); p from the asymptotic Mann-Whitney
     test (no Monte Carlo error), Holm within phrasing over the 7-8B family; forced-choice margins only on
     trials whose first generated token is A or B, since otherwise lpA - lpB is not the answer's score;
  5. frozen-twin order control; blank-video order answers;
  6. margin AUC by inter-event interval (the 72B's profile is U-shaped);
  7. TempCompass criterion; first-token YES/NO (or A/B) rates;
  8. the range of bars that leaves every stage label unchanged.

Writes results_v3/profile_v3.json. Usage: uv run --no-project --with numpy python profile_v3.py [--boot 10000]
"""
import argparse
import csv
import json
import math
from statistics import NormalDist

import numpy as np

import analyze_v3 as av

N = NormalDist()
SMALL = [m for m in av.MODELS if m not in av.SCALE_CHECK]
PROBES = "longpaper/results_v3"
PHRASINGS = ("order_real", "order_rev", "order_forced")


# ------------------------------------------------------------------ statistics
def mw_p(pos, neg):
    """Two-sided asymptotic Mann-Whitney p with tie correction."""
    x = np.r_[pos, neg]; n1, n0 = len(pos), len(neg); n = n1 + n0
    order = np.argsort(x, kind="mergesort"); xs = x[order]; rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    u = rk[:n1].sum() - n1 * (n1 + 1) / 2
    _, cnt = np.unique(x, return_counts=True)
    var = n1 * n0 / 12 * ((n + 1) - (cnt ** 3 - cnt).sum() / (n * (n - 1)))
    return 2 * (1 - N.cdf(abs((u - n1 * n0 / 2) / math.sqrt(var))))


def fisher_p(a, b, c, d):
    """Two-sided Fisher exact test for [[a, b], [c, d]] (a = YES on positives, c = YES on negatives)."""
    lf = lambda k: math.lgamma(k + 1)
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def lp(x):
        return lf(r1) + lf(r2) + lf(c1) + lf(n - c1) - lf(n) - lf(x) - lf(r1 - x) - lf(c1 - x) - lf(r2 - c1 + x)
    obs = lp(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(math.exp(lp(x)) for x in range(lo, hi + 1) if lp(x) <= obs + 1e-9))


def sdt(g, p):
    H = (p[g == 1].sum() + .5) / ((g == 1).sum() + 1); F = (p[g == 0].sum() + .5) / ((g == 0).sum() + 1)
    return N.inv_cdf(H) - N.inv_cdf(F), -(N.inv_cdf(H) + N.inv_cdf(F)) / 2


def bacc_at_rate(m, g, q):
    """BAcc of `margin > t` with t set so that a fraction q of trials is called YES."""
    pr = m > (np.quantile(m, 1 - q) if q > 0 else np.inf)
    return (pr[g == 1].mean() + (1 - pr[g == 0].mean())) / 2


def auc(pos, neg):
    return av.auc(list(pos), list(neg))


def stage(t):
    """Same rule as analyze_v3.stage(): answered if BAcc >= bar; else criterion / readout by whether the
    held-out margin threshold reaches the bar (margin significant after Holm); else below the order."""
    if (t.get("bacc") or 0) >= t.get("bar", .6):
        return "answered"
    if t.get("p_holm", 1) < .05:
        return "criterion" if (t.get("recal") or 0) >= t.get("bar", .6) else "readout"
    return "below order"


def probe_state(m):
    """Probe accuracy on visible trials: best real_last layer by all-trial accuracy (paper's probe
    table), and the final layer. Label = the question's answer (gt)."""
    try:
        rows = [r for r in csv.DictReader(open(f"{PROBES}/{m}_order_probe.csv"))
                if r["label"] == "gt" and r["feat"].startswith("real_last_L")]
    except FileNotFoundError:
        return None, None, None
    best = max(rows, key=lambda r: float(r["acc"]))
    last = max(rows, key=lambda r: int(r["feat"].rsplit("L", 1)[1]))
    return float(best["acc_visible"]), float(last["acc_visible"]), best["feat"].rsplit("_", 1)[1]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v3")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    meta = av.load_meta(f"{args.res}/videos_v3_metadata.csv")
    av.load_recorded(args.res)
    out = {k: {} for k in ("answers", "budget", "matched", "phrasing", "frozen_order", "blank", "by_interval",
                           "tc", "first_token", "bar_range")}
    rows_by = {m: r for m in av.MODELS if (r := av.load_rows(args.res, m, meta))}

    def vis(m, item):
        return [r for r in av.by_item(rows_by[m], item) if r["gt"] in ("0", "1") and av.visible(r["_meta"], m)]

    V = {m: [r for r in vis(m, "order_real") if r["_m"] is not None] for m in rows_by}
    V = {m: v for m, v in V.items() if len(v) >= av.MIN_N}

    # 1-2. answers and loss budget
    print("=== answers and loss budget, visible trials ===")
    print(f"{'model':18} {'P(Y)':>6} {'BAcc':>6} {'ceil':>6} {'YES+/YES-':>10} {'Fisher p':>9} {'dprime [95% CI]':>19} "
          f"{'c':>6} | {'state':>6} {'final':>6} {'margin':>7} {'readout':>8} {'crit':>6}")
    for m, v in V.items():
        g = np.array([int(r["gt"]) for r in v]); p = np.array([r["pred"] == "1" for r in v])
        d, c = sdt(g, p)
        ds = []
        for _ in range(args.boot // 5):
            i = rng.integers(0, len(g), len(g)); ds.append(sdt(g[i], p[i])[0])
        dlo, dhi = np.percentile(ds, [2.5, 97.5])
        a, b = int(p[g == 1].sum()), int((g == 1).sum() - p[g == 1].sum())
        cc, dd = int(p[g == 0].sum()), int((g == 0).sum() - p[g == 0].sum())
        fp = fisher_p(a, b, cc, dd)
        py = float(p.mean()); ba = av.bacc([r for r in v if r["pred"] in ("0", "1")])
        ceil = .5 + py if py < .5 else 1.5 - py
        degen = py in (0.0, 1.0)
        mg = av.recal(v, rng)[0]
        st, fin, layer = probe_state(m)
        out["answers"][m] = dict(py=py, bacc=ba, ceiling=ceil, yes_pos=a, n_pos=a + b, yes_neg=cc, n_neg=cc + dd,
                                 fisher_p=fp, dprime=None if degen else d, dprime_ci=None if degen else [dlo, dhi],
                                 c=None if degen else c, n=len(v))
        out["budget"][m] = dict(state=st, state_final=fin, state_layer=layer, margin=mg, answer=ba,
                                readout_loss=(st - mg) if st else None, criterion_loss=mg - ba)
        print(f"{av.NICE[m]:18} {py:6.3f} {ba:6.3f} {ceil:6.3f} {a:>4}/{cc:<4} {fp:9.1e} "
              + (f"{d:5.2f} [{dlo:4.2f},{dhi:4.2f}]" if not degen else f"{'deg.':>19}")
              + (f" {c:6.2f}" if not degen else f" {'deg.':>6}")
              + f" | {st if st else float('nan'):6.3f} {fin if fin else float('nan'):6.3f} {mg:7.3f} "
              f"{(st - mg) if st else float('nan'):8.3f} {mg - ba:6.3f}")

    # 3. matched criterion on common trials
    print("\n=== matched criterion, trials visible to both models ===")
    for a_, b_ in (("videochat-flash", "qwen2.5-vl-72b"), ("qwen2.5-vl", "qwen2.5-vl-72b"),
                   ("molmo2", "videochat-flash")):
        if a_ not in V or b_ not in V:
            continue
        A = {r["path"]: r for r in V[a_]}; B_ = {r["path"]: r for r in V[b_]}
        common = sorted(set(A) & set(B_))
        g = np.array([int(A[k]["gt"]) for k in common])
        ma = np.array([A[k]["_m"] for k in common]); mb = np.array([B_[k]["_m"] for k in common])
        pa = np.mean([A[k]["pred"] == "1" for k in common]); pb = np.mean([B_[k]["pred"] == "1" for k in common])
        ba_ = av.bacc([A[k] for k in common]); bb_ = av.bacc([B_[k] for k in common])
        res = dict(n=len(common), auc_a=auc(ma[g == 1], ma[g == 0]), auc_b=auc(mb[g == 1], mb[g == 0]),
                   py_a=pa, py_b=pb, bacc_a=ba_, bacc_b=bb_,
                   a_at_b_rate=bacc_at_rate(ma, g, pb), b_at_a_rate=bacc_at_rate(mb, g, pa),
                   ceil_at_a_rate=.5 + pa, ceil_at_b_rate=.5 + pb, rank_corr=float(np.corrcoef(
                       np.argsort(np.argsort(ma)), np.argsort(np.argsort(mb)))[0, 1]))
        diffs, ab = [], []
        for _ in range(args.boot // 5):
            i = rng.integers(0, len(g), len(g))
            if g[i].min() == g[i].max():
                continue
            diffs.append(auc(mb[i][g[i] == 1], mb[i][g[i] == 0]) - auc(ma[i][g[i] == 1], ma[i][g[i] == 0]))
            ab.append(bacc_at_rate(ma[i], g[i], pb))
        res["auc_diff_b_minus_a"] = [res["auc_b"] - res["auc_a"], *np.percentile(diffs, [2.5, 97.5])]
        res["a_at_b_rate_ci"] = list(np.percentile(ab, [2.5, 97.5]))
        out["matched"][f"{a_}|{b_}"] = res
        print(f"{av.NICE[a_]} vs {av.NICE[b_]} (n={len(common)}): AUC {res['auc_a']:.3f} vs {res['auc_b']:.3f}, "
              f"paired diff {res['auc_diff_b_minus_a'][0]:+.3f} [{res['auc_diff_b_minus_a'][1]:+.3f}, "
              f"{res['auc_diff_b_minus_a'][2]:+.3f}], rank r {res['rank_corr']:.2f}\n"
              f"   answers BAcc {ba_:.3f} (P(Y) {pa:.3f}) vs {bb_:.3f} (P(Y) {pb:.3f}); "
              f"{av.NICE[a_]} margin at {av.NICE[b_]}'s rate {res['a_at_b_rate']:.3f} "
              f"[{res['a_at_b_rate_ci'][0]:.3f}, {res['a_at_b_rate_ci'][1]:.3f}] (ceiling {.5 + pb:.3f}); "
              f"reverse {res['b_at_a_rate']:.3f} (ceiling {.5 + pa:.3f})")

    # 4. stage per phrasing
    print("\n=== stage per phrasing (visible; asymptotic MW p, Holm within phrasing per family) ===")
    for item in PHRASINGS:
        res = {}
        for m in rows_by:
            v = vis(m, item)
            if len(v) < av.MIN_N:
                continue
            ft_ok = (lambda r: r["top_token"].strip().upper() in ("A", "B")) if item == "order_forced" else \
                    (lambda r: r["top_token"].strip().upper() in ("YES", "NO"))
            ft = float(np.mean([ft_ok(r) for r in v]))
            vm = [r for r in v if r["_m"] is not None and ft_ok(r)]
            pos = np.array([r["_m"] for r in vm if r["gt"] == "1"]); neg = np.array([r["_m"] for r in vm if r["gt"] == "0"])
            ok = len(vm) >= av.MIN_N and len(pos) and len(neg)
            ans = [r for r in v if r["pred"] in ("0", "1")]
            res[m] = dict(bacc=av.bacc(ans), py=float(np.mean([r["pred"] == "1" for r in ans])), first_token=ft,
                          n_margin=len(vm), auc=auc(pos, neg) if ok else None, p=mw_p(pos, neg) if ok else 1.0,
                          recal=av.recal(vm, rng)[0] if ok else None, n=len(v))
        for fam in (SMALL, sorted(av.SCALE_CHECK)):
            ks = [m for m in fam if m in res]
            for k, h in zip(ks, av.holm([res[k]["p"] for k in ks])):
                res[k]["p_holm"] = h
        for m, t in res.items():
            t["stage"] = stage(t)
            print(f"{item:13} {av.NICE[m]:18} BAcc {t['bacc']:.3f} first-tok {t['first_token']:.3f} n_m {t['n_margin']:4} "
                  f"AUC {t['auc'] if t['auc'] is not None else float('nan'):.3f} p {t['p']:.4f} Holm {t['p_holm']:.4f} "
                  f"held-out {t['recal'] if t['recal'] is not None else float('nan'):.3f} -> {t['stage']}")
        out["phrasing"][item] = res

    # 8. bar range: the set of bars leaving every label (Table 1 + phrasings) unchanged
    vals = []
    for item, res in out["phrasing"].items():
        for m, t in res.items():
            vals.append(t["bacc"])
            if t.get("p_holm", 1) < .05 and t["recal"] is not None:
                vals.append(t["recal"])
    below = max(x for x in vals if x < .6); above = min(x for x in vals if x >= .6)
    out["bar_range"] = [below, above]
    print(f"\nbars in ({below:.3f}, {above:.3f}] leave every stage label unchanged")

    # 5. frozen-twin order control and blank-video order answers
    print("\n=== order question on frozen twins (vs source label) and on blank video ===")
    for m, rows in rows_by.items():
        src = {r["path"]: r for r in av.by_item(rows, "order_real")}
        tw = [r for r in av.by_item(rows, "order_twin") if r["_m"] is not None and r["_meta"]
              and r["_meta"]["path"] in src]
        bl = [r for r in av.by_item(rows, "order_blank") if r["pred"] in ("0", "1")]
        out["blank"][m] = dict(py=float(np.mean([r["pred"] == "1" for r in bl])) if bl else None, n=len(bl))
        if len(tw) < av.MIN_N:
            print(f"{av.NICE[m]:18} frozen n={len(tw)} (skipped); blank P(Y) {out['blank'][m]['py']} n={len(bl)}")
            continue
        pos = np.array([r["_m"] for r in tw if src[r["_meta"]["path"]]["gt"] == "1"])
        neg = np.array([r["_m"] for r in tw if src[r["_meta"]["path"]]["gt"] == "0"])
        out["frozen_order"][m] = dict(auc=auc(pos, neg), p=mw_p(pos, neg), n=len(tw))
        print(f"{av.NICE[m]:18} frozen AUC {auc(pos, neg):.3f} p={mw_p(pos, neg):.3f} n={len(tw)}; "
              f"blank P(Y) {out['blank'][m]['py']:.3f} n={len(bl)}")

    # 6. margin AUC by interval
    print("\n=== order-margin AUC by interval (visible trials) ===")
    ints = sorted({int(r["interval_ms"]) for r in V[next(iter(V))]})
    print(f"{'model':18} " + " ".join(f"{i:>6}" for i in ints))
    for m, v in V.items():
        row = {}
        for i in ints:
            s = [r for r in v if int(r["interval_ms"]) == i]
            p_ = [r["_m"] for r in s if r["gt"] == "1"]; n_ = [r["_m"] for r in s if r["gt"] == "0"]
            b_ = av.bacc([r for r in s if r["pred"] in ("0", "1")])
            row[i] = dict(n=len(s), auc=av.auc(p_, n_) if p_ and n_ else None, bacc=b_)
        out["by_interval"][m] = row
        print(f"{av.NICE[m]:18} " + " ".join(f"{row[i]['auc']:6.2f}" if row[i]["auc"] is not None else f"{'--':>6}"
                                           for i in ints))

    # 7. TempCompass
    print("\n=== TempCompass yes/no, real clips (margin run) ===")
    for m in av.MODELS:
        try:
            f = open(f"results_long/{m}_tc_margin.csv")
        except FileNotFoundError:
            continue
        rows = []
        for r in csv.DictReader(f):
            if r["gold"] in ("0", "1"):
                try:
                    mg = float(r["margin"])
                except ValueError:
                    mg = None
                rows.append(dict(gt=r["gold"], pred=r["pred"], _m=mg, top=r.get("top_token", "")))
        ans = [r for r in rows if r["pred"] in ("0", "1")]
        g = np.array([int(r["gt"]) for r in ans]); p = np.array([r["pred"] == "1" for r in ans])
        d, c = sdt(g, p)
        rc = av.recal([r for r in rows if r["_m"] is not None], rng)[0]
        ft = float(np.mean([r["top"].strip().upper() in ("YES", "NO") for r in rows]))
        out["tc"][m] = dict(c=c, dprime=d, py=float(p.mean()), bacc=av.bacc(ans), recal=rc, first_token_yn=ft)
        print(f"{av.NICE[m]:18} P(Y) {p.mean():.3f} c {c:+.2f} d' {d:.2f} BAcc {av.bacc(ans):.3f} held-out {rc:.3f} "
              f"first-token Y/N {ft:.3f}")

    print("\n=== first generated token is YES/NO (A/B for forced), all trials ===")
    for m, rows in rows_by.items():
        rr = {}
        for item in PHRASINGS:
            o = av.by_item(rows, item)
            if len(o) < av.MIN_N:
                continue
            ok = ("A", "B") if item == "order_forced" else ("YES", "NO")
            rr[item] = float(np.mean([r["top_token"].strip().upper() in ok for r in o]))
        out["first_token"][m] = rr
        print(f"{av.NICE[m]:18} " + "  ".join(f"{k} {v:.3f}" for k, v in rr.items()))

    with open(f"{args.res}/profile_v3.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.res}/profile_v3.json")


if __name__ == "__main__":
    main()
