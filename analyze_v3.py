"""Analysis for the v3 run. Replaces analyze_long.py, whose exp_g() called visible_8() — an
interval-based rule that assumes the fixed 1.0 s lead-in v2 had already stopped using — while the exact
per-trial visible_v2() sat unused. Every number the paper quotes comes from here.

What this fixes relative to the previous analysis:
  * visibility is exact and per model (jump frames from the stimulus metadata), never inferred from the
    interval;
  * AUC is reported on ALL trials and on the visible subset, because the previous Table 1 silently mixed
    them (BAcc over 1,200 trials beside AUC over a model-specific subset);
  * the move question resamples by matched pair, since a clip and its twin are not independent;
  * recalibration is reported over 200 re-splits with an interval, not a single seed-0 split;
  * p-values are Holm-corrected within each family of tests and report exact permutation counts.

Usage: uv run --no-project --with numpy python analyze_v3.py [--res results_v3] [--boot 2000]
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SCALE_CHECK = {"qwen2.5-vl-72b", "internvl2.5-78b"}
MIN_N = 200   # conditions with fewer rows (smoke tests, unfinished runs) are not reported
MODELS = ["molmo2", "videochat-flash", "qwen2.5-vl", "internvl2.5", "llava-next-video",
          "video-llama2", "qwen2.5-vl-72b", "internvl2.5-78b"]
NICE = {"molmo2": "Molmo2", "videochat-flash": "VideoChat-Flash", "qwen2.5-vl": "Qwen2.5-VL-7B",
        "internvl2.5": "InternVL2.5-8B", "llava-next-video": "LLaVA-NeXT-Video",
        "video-llama2": "Video-LLaMA2", "qwen2.5-vl-72b": "Qwen2.5-VL-72B",
        "internvl2.5-78b": "InternVL2.5-78B"}
SAMPLERS = {
    "uniform8": lambda T: [int(i * T / 8) for i in range(8)],
    "uniform32": lambda T: [int(i * T / 32) for i in range(32)],
    "midpoint8": lambda T: [int((i + 0.5) * T / 8) for i in range(8)],
    "molmo2": lambda T: sorted({min(T - 1, int(round(t * 30))) for t in
                                [x * 0.5 for x in range(int((T / 30) / 0.5) + 1)] + [T / 30]}),
    "vcf64": lambda T: [int(i * T / 64) for i in range(64)] if T >= 64 else list(range(T)),
}
MODEL_SAMPLER = {"molmo2": "molmo2", "videochat-flash": "vcf64", "video-llama2": "midpoint8"}


# ----------------------------------------------------------------- statistics
def auc(pos, neg):
    """Mann-Whitney AUC with mid-ranks for ties."""
    if len(pos) == 0 or len(neg) == 0:   # `not pos` is ambiguous once these are numpy arrays
        return None
    sc = np.asarray(list(pos) + list(neg), float)
    lab = np.asarray([1] * len(pos) + [0] * len(neg))
    order = np.argsort(sc, kind="mergesort")
    s = sc[order]
    r = np.arange(1, len(sc) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks = np.empty(len(sc))
    ranks[order] = r
    n1 = int(lab.sum())
    n0 = len(lab) - n1
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(pos, neg, B, rng, pairs=None):
    """Bootstrap CI for AUC. With `pairs` the resampling unit is the matched pair (a clip and its twin
    are the same stimulus), which is the correct unit and gives a narrower, honest interval."""
    out = []
    if pairs is not None:
        idx = np.arange(len(pairs))
        for _ in range(B):
            take = rng.choice(idx, len(idx), replace=True)
            p = [pairs[i][0] for i in take]
            n = [pairs[i][1] for i in take]
            a = auc(p, n)
            if a is not None:
                out.append(a)
    else:
        P, N = np.asarray(pos, float), np.asarray(neg, float)
        for _ in range(B):
            a = auc(rng.choice(P, len(P), replace=True), rng.choice(N, len(N), replace=True))
            if a is not None:
                out.append(a)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def perm_p(pos, neg, B, rng, pairs=None):
    """Two-sided permutation test for AUC = .5. Returns (p, exceedances, B) so the exact count is
    reportable; the previous analysis printed p = .002 for every significant test, which is 1/500 and
    cannot come from the 2,000 draws the paper claimed."""
    obs = auc(pos, neg)
    if obs is None:
        return None, 0, B
    c = 0
    if pairs is not None:
        # within-pair sign flips: the null is "the label is unrelated to which member of the pair"
        arr = np.asarray(pairs, float)
        for _ in range(B):
            flip = rng.random(len(arr)) < 0.5
            p = np.where(flip, arr[:, 1], arr[:, 0])
            n = np.where(flip, arr[:, 0], arr[:, 1])
            if abs(auc(p, n) - .5) >= abs(obs - .5) - 1e-12:
                c += 1
    else:
        allv = np.asarray(list(pos) + list(neg), float)
        k = len(pos)
        for _ in range(B):
            rng.shuffle(allv)
            if abs(auc(allv[:k], allv[k:]) - .5) >= abs(obs - .5) - 1e-12:
                c += 1
    return (c + 1) / (B + 1), c, B


def holm(pvals):
    """Holm-Bonferroni within a family. Returns adjusted p in the original order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    prev = 0.0
    for rank, i in enumerate(idx):
        val = min(1.0, (len(pvals) - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def bacc(rows):
    pos = [r for r in rows if r["gt"] == "1"]
    neg = [r for r in rows if r["gt"] == "0"]
    if not pos or not neg:
        return None
    h = sum(r["pred"] == "1" for r in pos) / len(pos)
    f = sum(r["pred"] == "1" for r in neg) / len(neg)
    return (h + 1 - f) / 2


def crit(rows):
    """SDT criterion c (log-linear correction); positive = biased toward NO."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    pos = [r for r in rows if r["gt"] == "1"]; neg = [r for r in rows if r["gt"] == "0"]
    if not pos or not neg:
        return None
    h = (sum(r["pred"] == "1" for r in pos) + .5) / (len(pos) + 1)
    f = (sum(r["pred"] == "1" for r in neg) + .5) / (len(neg) + 1)
    return -(z(h) + z(f)) / 2


def recal(rows, rng, splits=200):
    """Held-out threshold on the margin, over many random halves. The single seed-0 split the previous
    analysis used put InternVL2.5-8B's reported gain at the top of its own split distribution."""
    rows = [r for r in rows if r.get("_m") is not None and r["gt"] in ("0", "1")]
    if len(rows) < 20:
        return None, None, None
    def at(rs, t):
        p = [r for r in rs if r["gt"] == "1"]
        n = [r for r in rs if r["gt"] == "0"]
        if not p or not n:
            return None
        return (sum(r["_m"] > t for r in p) / len(p) + sum(r["_m"] <= t for r in n) / len(n)) / 2
    vals = []
    order = np.arange(len(rows))
    for _ in range(splits):
        rng.shuffle(order)
        half = len(order) // 2
        tr = [rows[i] for i in order[:half]]
        te = [rows[i] for i in order[half:]]
        cands = sorted({r["_m"] for r in tr})
        if not cands:
            continue
        best = max(cands, key=lambda t: at(tr, t) or 0)
        v = at(te, best)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, None
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------- data
def load_meta(path):
    """Index every derived clip (twin, flipped, one-shape-moves) back to its source trial, so a clip and
    its twin can be matched as a pair. Twins live in their own directory, so a path-rewrite rule is not
    enough to find them."""
    idx = {}
    for r in csv.DictReader(open(path)):
        idx[r["path"]] = r
        for k in ("twin_path", "flip_path", "solo_a_path", "solo_b_path"):
            v = r.get(k)
            if v:
                idx[v] = r
    return idx


# Frame indices each adapter actually decoded (record_sampled_frames.py), keyed (sampler, path). The rules
# in SAMPLERS are only a fallback: re-implementing a loader got Video-LLaMA2's endpoints and
# VideoChat-Flash's 64-frame grid wrong, and both change which trials count as visible.
RECORDED = {}
RECORDED_SOURCE = {"molmo2": "molmo2", "videochat-flash": "videochat-flash", "video-llama2": "video-llama2",
                   "uniform8": "qwen2.5-vl"}   # every uniform8 adapter calls models/base.load_frames(n=8)


def load_recorded(res):
    for f in sorted(Path(res).glob("sampled_frames_*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            RECORDED[(d["model"], d["path"])] = d["idx"]


def sampled(meta_row, model, n_frames=8):
    T = int(float(meta_row["n_frames"]))
    key = MODEL_SAMPLER.get(model, "uniform32" if int(n_frames) == 32 else "uniform8")
    rec = RECORDED.get((RECORDED_SOURCE.get(model if model in MODEL_SAMPLER else key, ""), meta_row["path"]))
    return rec if rec is not None else SAMPLERS[key](T)


def visible(meta_row, model, n_frames=8):
    if meta_row is None:
        return None
    a = int(float(meta_row["f_first"]))
    b = int(float(meta_row["f_second"]))
    lo, hi = min(a, b), max(a, b)
    # Real-footage stimuli (generate_real.py): each event is a burst of K played frames, so the order is
    # also shown by a frame in which the first panel has finished and the second has not. Frames in
    # which both panels are mid-burst are not counted: telling the order there needs the phase of each.
    K = int(float(meta_row.get("burst_frames") or 0))
    # A panel differs from its first still on frames [onset, ...) and equals its last still from onset + K - 1.
    return any(lo <= i < hi or (K and lo + K - 1 <= i < hi + K - 1) for i in sampled(meta_row, model, n_frames))


def load_rows(res, model, meta):
    f = Path(res) / f"{model}_expg3.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        try:
            r["_m"] = float(r["margin"])
        except (TypeError, ValueError):
            r["_m"] = None
        src = r["path"].replace("_flip", "").replace("_soloa", "").replace("_solob", "")
        r["_meta"] = meta.get(r["path"]) or meta.get(src)
    return rows


def by_item(rows, item):
    return [r for r in rows if r["item"] == item]


def paired(rows_pos, rows_neg, key=lambda r: r["_meta"]["path"] if r["_meta"] else r["path"]):
    """Match positives to negatives by their source stimulus."""
    neg = {key(r): r for r in rows_neg}
    out = []
    for r in rows_pos:
        m = neg.get(key(r))
        if m is not None and r["_m"] is not None and m["_m"] is not None:
            out.append((r["_m"], m["_m"]))
    return out


# ----------------------------------------------------------------- reports
def report(res, meta_path, B, seed):
    meta = load_meta(meta_path)
    load_recorded(res)
    print(f"recorded frame indices: {len(RECORDED)} (model, clip) pairs; rules used for the rest")
    trials = {r["path"]: r for r in meta.values()}.values()
    print(f"{'sampler':16} {'invisible (recorded)':>21} {'invisible (rule)':>17} {'disagree':>9}")
    for model, key in (("qwen2.5-vl", "uniform8"), ("molmo2", "molmo2"), ("video-llama2", "midpoint8"),
                       ("videochat-flash", "vcf64")):
        rec_n = rule_n = dis = have = 0
        for t in trials:
            a, b = int(float(t["f_first"])), int(float(t["f_second"]))
            rule = any(min(a, b) <= i < max(a, b) for i in SAMPLERS[key](int(float(t["n_frames"]))))
            got = RECORDED.get((RECORDED_SOURCE[model if model in MODEL_SAMPLER else key], t["path"]))
            rule_n += not rule
            if got is not None:
                have += 1
                v = any(min(a, b) <= i < max(a, b) for i in got)
                rec_n += not v
                dis += v != rule
        print(f"{key:16} {f'{rec_n}/{have}' if have else 'n/a':>21} {rule_n:>17} {dis if have else '':>9}")
    rng = np.random.default_rng(seed)
    print(f"\n=== v3 controlled results ({res}) ===")
    print(f"{'model':18} {'P(Y)':>6} {'BAcc':>6} {'BAcc_v':>7} {'AUC_all':>8} {'AUC_vis':>8} "
          f"{'[95% CI]':>16} {'perm p':>9} {'recal':>15}")
    order_p, order_names = [], []
    for m in MODELS:
        rows = load_rows(res, m, meta)
        if not rows:
            continue
        o = [r for r in by_item(rows, "order_real") if r["gt"] in ("0", "1")]
        if not o:
            continue
        ans = [r for r in o if r["pred"] in ("0", "1")]
        py = sum(r["pred"] == "1" for r in ans) / len(ans) if ans else float("nan")
        vis = [r for r in o if visible(r["_meta"], m)]
        mo = [r for r in o if r["_m"] is not None]
        mv = [r for r in vis if r["_m"] is not None]
        a_all = auc([r["_m"] for r in mo if r["gt"] == "1"], [r["_m"] for r in mo if r["gt"] == "0"])
        pos = [r["_m"] for r in mv if r["gt"] == "1"]
        neg = [r["_m"] for r in mv if r["gt"] == "0"]
        a_vis = auc(pos, neg)
        lo, hi = boot_ci(pos, neg, B, rng)
        p, c, n = perm_p(pos, neg, B, rng)
        rc, rlo, rhi = recal(mv, rng)
        order_p.append(p if p is not None else 1.0)
        order_names.append(m)
        blank = [r for r in by_item(rows, "order_blank") if r["pred"] in ("0", "1")]
        TAB[m].update(c_v=crit([r for r in vis if r["pred"] in ("0", "1")]), py=py, bacc=bacc(ans), bacc_v=bacc([r for r in vis if r["pred"] in ("0", "1")]),
                      auc_all=a_all, auc_vis=a_vis, auc_vis_ci=[lo, hi], auc_vis_p=p, recal=rc,
                      recal_ci=[rlo, rhi], n_vis=len(vis),
                      blank_py=(sum(r["pred"] == "1" for r in blank) / len(blank)) if blank else None)
        print(f"{NICE[m]:18} {py:6.3f} {bacc(ans) or float('nan'):6.3f} "
              f"{bacc([r for r in vis if r['pred'] in ('0','1')]) or float('nan'):7.3f} "
              f"{(a_all if a_all is not None else float('nan')):8.3f} "
              f"{(a_vis if a_vis is not None else float('nan')):8.3f} "
              f"[{lo:.3f}, {hi:.3f}] {p:9.4f}  "
              f"{rc:.3f} [{rlo:.3f}, {rhi:.3f}]" if a_vis is not None else f"{NICE[m]:18} (no margin)")
        print(f"{'':18} visible {len(vis)}/{len(o)}, permutation exceedances {c}/{n}")
    # Holm families (decided 2026-09-25, before the 78B result): the 7-8B stage tests form
    # one family; the 72-78B scale check is a separate family, so no 7-8B placement
    # depends on whether a large model is added.
    for fam in (SCALE_CHECK, None):
        idx = [i for i, k in enumerate(order_names) if (k in SCALE_CHECK) == (fam is not None)]
        if not idx:
            continue
        adj = holm([order_p[i] for i in idx])
        print("\nHolm-adjusted order-margin p (" + ("72-78B family" if fam else "7-8B family") + "):",
              {NICE[order_names[i]]: round(v, 4) for i, v in zip(idx, adj)})
        for i, v in zip(idx, adj):
            TAB[order_names[i]]["auc_vis_p_holm"] = v

    # --- move question: twin control AND the object-binding control -------------
    print(f"\n=== move question ({'twin = any change'}, {'solo = object binding'}) ===")
    print(f"{'model':18} {'P(Y)real':>9} {'P(Y)twin':>9} {'AUC_twin':>9} {'[95% CI]':>16} {'p':>8} "
          f"{'AUC_solo':>9} {'[95% CI]':>16} {'p':>8}")
    for m in MODELS:
        rows = load_rows(res, m, meta)
        if not rows:
            continue
        real, twin = by_item(rows, "moved_real"), by_item(rows, "moved_twin")
        hit, fa = by_item(rows, "moved_solo_hit"), by_item(rows, "moved_solo_fa")
        if not real:
            continue
        def py(rs):
            a = [r for r in rs if r["pred"] in ("0", "1")]
            return sum(r["pred"] == "1" for r in a) / len(a) if a else float("nan")
        pr_t = paired(real, twin)
        pr_s = paired(hit, fa)
        line = f"{NICE[m]:18} {py(real):9.3f} {py(twin):9.3f}"
        TAB[m].update(move_real=py(real), move_twin=py(twin), solo_hit=py(hit), solo_fa=py(fa))
        for tag, pr in (("twin", pr_t), ("solo", pr_s)):
            if len(pr) < 10:
                line += f" {'--':>9} {'--':>16} {'--':>8}"
                continue
            a = auc([x for x, _ in pr], [y for _, y in pr])
            lo, hi = boot_ci(None, None, B, rng, pairs=pr)
            p, c, n = perm_p([x for x, _ in pr], [y for _, y in pr], B, rng, pairs=pr)
            line += f" {a:9.3f} [{lo:.3f}, {hi:.3f}] {p:8.4f}"
            TAB[m].update({f"auc_{tag}": a, f"auc_{tag}_ci": [lo, hi], f"auc_{tag}_p": p})
        print(line)
    for fam in (SCALE_CHECK, None):
        ks = [m for m in MODELS if "auc_twin_p" in TAB.get(m, {}) and (m in SCALE_CHECK) == (fam is not None)]
        for k, v in zip(ks, holm([TAB[k]["auc_twin_p"] for k in ks])):
            TAB[k]["auc_twin_p_holm"] = v
    print("  AUC_twin separates a moving clip from a frozen one: a change detector achieves this.")
    print("  AUC_solo separates 'the queried shape moved' from 'the other shape moved': needs binding.")

    # --- polarity control, every model -----------------------------------------
    print(f"\n=== polarity control: 'did both shapes stay still?' (gold: real NO, twin YES) ===")
    print(f"{'model':18} {'P(Y)real':>9} {'P(Y)twin':>9} {'BAcc':>7}   verdict")
    for m in MODELS:
        rows = load_rows(res, m, meta)
        if not rows:
            continue
        sr, st = by_item(rows, "still_real"), by_item(rows, "still_twin")
        if not sr:
            continue
        def py(rs):
            a = [r for r in rs if r["pred"] in ("0", "1")]
            return sum(r["pred"] == "1" for r in a) / len(a) if a else float("nan")
        b = bacc([r for r in sr + st if r["pred"] in ("0", "1")])
        pr = py(sr)
        pt = py(st)
        TAB[m].update(still_real=pr, still_twin=pt, still_bacc=b)
        # Pass only if the answer tracks the clip in this polarity too; a flat NO to both is a bias,
        # not detection (Video-LLaMA2: .046 / .087).
        if b is not None and b >= .7:
            verdict = "passes: answer tracks motion in this polarity"
        elif pt - pr < .1 and pr > .5:
            verdict = "fails: says 'still' whether or not a shape moved"
        elif pt - pr < .1:
            verdict = "fails: says 'not still' whether or not a shape moved"
        elif pr > .5:
            verdict = "fails: says 'still' about most clips where a shape moved"
        else:
            verdict = "partial: tracks motion, criterion far toward NO"
        print(f"{NICE[m]:18} {pr:9.3f} {py(st):9.3f} {(b or float('nan')):7.3f}   {verdict}")

    # --- prompt variants and interval ------------------------------------------
    print(f"\n=== prompt variants and interval estimation (v3 stimuli) ===")
    print(f"{'model':18} {'orig BAcc':>10} {'rev BAcc':>9} {'forced BAcc':>12} {'forced AUC':>11} "
          f"{'interval acc':>13} {'bins used':>10}")
    for m in MODELS:
        rows = load_rows(res, m, meta)
        if not rows:
            continue
        o, rv, fo = by_item(rows, "order_real"), by_item(rows, "order_rev"), by_item(rows, "order_forced")
        iv = by_item(rows, "interval")
        rv, fo, iv = [x if len(x) >= MIN_N else [] for x in (rv, fo, iv)]
        fm = [r for r in fo if r["_m"] is not None]
        fa = auc([r["_m"] for r in fm if r["gt"] == "1"], [r["_m"] for r in fm if r["gt"] == "0"])
        ivp = [r for r in iv if r["pred"] in ("A", "B", "C", "D")]
        acc = sum(r["pred"] == r["gt"] for r in ivp) / len(ivp) if ivp else float("nan")
        bins = len({r["pred"] for r in ivp})
        print(f"{NICE[m]:18} {(bacc([r for r in o if r['pred'] in ('0','1')]) or float('nan')):10.3f} "
              f"{(bacc([r for r in rv if r['pred'] in ('0','1')]) or float('nan')):9.3f} "
              f"{(bacc([r for r in fo if r['pred'] in ('0','1')]) or float('nan')):12.3f} "
              f"{(fa if fa is not None else float('nan')):11.3f} {acc:13.3f} {bins:10d}")

    # --- 32 frames, now stratified across all intervals -------------------------
    print(f"\n=== 8 vs 32 frames on the SAME stimuli (v3 stratifies these across all 12 intervals) ===")
    for m in MODELS:
        rows = load_rows(res, m, meta)
        if not rows:
            continue
        f32 = by_item(rows, "order_32f")
        if len(f32) < MIN_N:
            continue
        paths = {r["path"] for r in f32}
        f8 = [r for r in by_item(rows, "order_real") if r["path"] in paths]
        b32 = bacc([r for r in f32 if r["pred"] in ("0", "1")])
        b8 = bacc([r for r in f8 if r["pred"] in ("0", "1")])
        if b32 is None or b8 is None:
            print(f"{NICE[m]:18} no parseable answers at 32 frames")
            continue
        ints = sorted({int(r["interval_ms"]) for r in f32})
        print(f"{NICE[m]:18} 8f {b8:.3f} -> 32f {b32:.3f}  (delta {b32-b8:+.3f}, n={len(f32)}, "
              f"{len(ints)} intervals)")


TAB = defaultdict(dict)


def f3(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return "1.00" if x >= .9995 else f"{x:.3f}".replace("0.", ".", 1)


def stage(t):
    """Where the evidence stops, by a fixed rule on the table's own columns (no judgement per model):
    answered if visible-trial answers are clearly above chance; else, if the order margin is significant
    after Holm, 'criterion' when a held-out threshold on it reaches the same bar (only the operating point
    loses the order) and 'readout' when it does not; else 'ordering' if the margin at least separates
    moving from frozen video; else 'registration'. (Until 2026-09-25 criterion and readout were one
    'decision' stage; split after the round-6 critics showed they are different losses.)"""
    if (t.get("bacc_v") or 0) >= .6:
        return "answered"
    # p is compared with `is not None`: an asymptotic test can return exactly 0.0, which `or 1` would turn
    # into "not significant". A margin significantly BELOW chance is not order evidence either.
    p_ord, p_tw = t.get("auc_vis_p_holm"), t.get("auc_twin_p_holm")
    if p_ord is not None and p_ord < .05 and (t.get("auc_vis") or 0) > .5:
        return "criterion" if (t.get("recal") or 0) >= .6 else "readout"
    if p_tw is not None and p_tw < .05 and (t.get("auc_twin") or 0) > .5:
        return "ordering"
    return "registration"


def write_tables(res):
    """Numbers for the paper, written by the analysis rather than copied by hand."""
    Path(res, "numbers_v3.json").write_text(json.dumps(TAB, indent=1, default=float))
    lines = []
    for m in MODELS:
        t = TAB.get(m)
        if not t or "auc_vis" not in t:
            continue
        ns = "" if (t.get("auc_vis_p_holm", 1) < .05) else "$^{\\dagger}$"
        ci = t["auc_vis_ci"]
        c = t.get("c_v") if 0 < t["py"] < 1 else None   # one answer throughout: c is a correction artefact
        lines.append(f"{NICE[m]:17} & {f3(t['py'])} & {'deg.' if c is None else f'{c:.2f}'} & {f3(t['bacc_v'])} & "
                     f"{f3(t['auc_vis'])} [{f3(ci[0])}, {f3(ci[1])}]{ns} & {f3(t['recal'])} & "
                     f"{f3(t.get('move_real'))} & {f3(t.get('move_twin'))} & {f3(t.get('auc_twin'))} & "
                     f"{f3(t.get('auc_solo'))} & {f3(t.get('still_bacc'))} & {stage(t)} \\\\")
    Path(res, "table_main_rows.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {res}/numbers_v3.json and {res}/table_main_rows.tex ({len(lines)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v3")
    ap.add_argument("--meta", default="results_v3/videos_v3_metadata.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()
    report(args.res, args.meta, args.boot, args.seed)
    write_tables(args.res)


if __name__ == "__main__":
    main()
