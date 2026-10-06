"""Analysis of the real-footage run (generate_real.py stimuli), set against the shape run. CPU only.

Same measurements and rules as analyze_v3.py / band_v3.py, nothing re-tuned: first-token YES-NO margin,
Holm over the six 7-8B models, bar .60, analyze_v3.stage(). Three things differ, all because the stimuli do.

1. Paired AUC is the sensitivity measure. Every trial exists in both orders with the same footage, layout,
   timing and question, so the margin's order sensitivity is P(margin on the gt=1 member > margin on the
   gt=0 member) over the flipped pairs. The ordinary (unpaired) AUC is also reported, because that is what
   the shape run has; but on real footage the two nouns ("the horse", "the boxer") shift the margin by a
   different amount for every pair, which lowers the unpaired AUC without any change in order sensitivity.
   The share of margin variance explained by the noun pair is printed for both stimulus families.
2. Resampling unit. Trials share footage, and one person may have filmed several clips, so confidence
   intervals come from a two-way cluster bootstrap over uploaders: each uploader gets a resampling count
   and a trial (or pair) is weighted by the product of the counts of its two panels. Tests against chance
   are per pair (sign test), valid because each pair's order is the only thing that differs within it.
3. Trial classes come from the pixels (verify_real.py -> evidence.csv), per sampler. state: a delivered
   frame shows one panel changed and the other not (the paper's "visible"). phase: the delivered frames
   differ between the two orders only while both panels are mid-burst. blind: identical for both orders.

Pre-registered outcomes: reviews/realclip_plan_1001/plan_v3_amendments.md.
Writes <res>/numbers_real.json, real_rows.tex, real_band_rows.tex.
Usage: uv run --no-project --with numpy python analyze_real.py [--res results_real] [--shape results_v3]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

import analyze_v3 as av
from band_v3 import BANDS
from profile_v3 import mw_p, sdt

SMALL = [m for m in av.MODELS if m not in av.SCALE_CHECK]
SAMPLER = {"molmo2": "molmo2", "videochat-flash": "vcf64", "video-llama2": "midpoint8"}   # others: uniform8
BAR = .60
Z = NormalDist().inv_cdf


def f3(x, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.3f}" if (signed or x < 0) else f"{x:.3f}"
    return s.replace("0.", ".", 1)


def ci(x):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)])
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def sign_p(k, n):
    """Two-sided exact binomial test of k successes in n at p = .5. Ties are dropped by the caller."""
    if n == 0:
        return 1.0
    k = min(int(k), n - int(k))
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def who(r, side):
    m = r["_meta"]
    return m.get(f"who_{side}") or m[f"src_{side}"]


def cluster_counts(rows, rng):
    ids = sorted({who(r, s) for r in rows for s in "ab"})
    ix = {u: i for i, u in enumerate(ids)}
    a = np.array([ix[who(r, "a")] for r in rows]); b = np.array([ix[who(r, "b")] for r in rows])
    return lambda: (lambda c: (c[a] * c[b]).astype(float))(np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)))


def pairs_of(rows):
    """One entry per flipped pair with both members present: (gt=1 row, gt=0 row)."""
    by = defaultdict(dict)
    for r in rows:
        by[r["_meta"]["pair_id"]][r["gt"]] = r
    return [(d["1"], d["0"]) for d in by.values() if len(d) == 2]


def paired_auc(pairs, B, rng):
    if not pairs:
        return None, [None, None], 1.0, 0
    ind = np.array([(p["_m"] > n["_m"]) + .5 * (p["_m"] == n["_m"]) for p, n in pairs])
    draw = cluster_counts([p for p, _ in pairs], rng)
    bs = []
    for _ in range(B):
        w = draw()
        if w.sum() > 0:
            bs.append(float((w * ind).sum() / w.sum()))
    return float(ind.mean()), ci(bs), sign_p(int((ind == 1).sum()), int((ind != .5).sum())), len(ind)


def unpaired_boot(rows, B, rng, cluster):
    g = np.array([int(r["gt"]) for r in rows]); m = np.array([r["_m"] for r in rows], float)
    C = (m[g == 1][:, None] > m[g == 0][None, :]) + .5 * (m[g == 1][:, None] == m[g == 0][None, :])
    draw = cluster_counts(rows, rng) if cluster else None
    out = []
    for _ in range(B):
        w = draw() if cluster else np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)).astype(float)
        d = w[g == 1].sum() * w[g == 0].sum()
        if d > 0:
            out.append(float(w[g == 1] @ C @ w[g == 0] / d))
    return np.array(out)


def crit_boot(rows, B, rng, cluster):
    g = np.array([int(r["gt"]) for r in rows]); p = np.array([r["pred"] == "1" for r in rows], float)
    draw = cluster_counts(rows, rng) if cluster else None
    out = []
    for _ in range(B):
        w = draw() if cluster else np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)).astype(float)
        if w[g == 1].sum() == 0 or w[g == 0].sum() == 0:
            continue
        H = ((w * p)[g == 1].sum() + .5) / (w[g == 1].sum() + 1); F = ((w * p)[g == 0].sum() + .5) / (w[g == 0].sum() + 1)
        out.append(-(Z(min(max(H, 1e-6), 1 - 1e-6)) + Z(min(max(F, 1e-6), 1 - 1e-6))) / 2)
    return np.array(out)


def var_share(rows, key):
    """Adjusted share of margin variance explained by the noun pair (how the question names the two objects)."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r["_m"])
    m = np.array([r["_m"] for r in rows]); n, k = len(m), len(groups)
    if k < 2 or n <= k:
        return None
    ssw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups.values())
    r2 = 1 - ssw / ((m - m.mean()) ** 2).sum()
    return float(max(0.0, 1 - (1 - r2) * (n - 1) / (n - k)))


def answers(rows):
    """P(YES), BAcc, d', c on rows with a parsed answer."""
    rr = [r for r in rows if r["pred"] in ("0", "1")]
    g = np.array([int(r["gt"]) for r in rr]); p = np.array([r["pred"] == "1" for r in rr])
    d, c = sdt(g, p)
    degenerate = p.sum() == 0 or p.sum() == len(p)
    return dict(py=float(p.mean()), bacc=av.bacc(rr), dprime=d, c=c, degenerate=bool(degenerate), unparsed=len(rows) - len(rr)), rr


def order_rows(rows):
    return [r for r in av.by_item(rows, "order_real") if r["gt"] in ("0", "1") and r["_m"] is not None and r["_meta"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_real")
    ap.add_argument("--shape", default="results_v3")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- shape run (reference)
    av.RECORDED.clear(); av.load_recorded(args.shape)
    meta_s = av.load_meta(f"{args.shape}/videos_v3_metadata.csv")
    shape, sboot = {}, {}
    for m in SMALL:
        rows = av.load_rows(args.shape, m, meta_s)
        if not rows:
            continue
        o = order_rows(rows); vis = [r for r in o if av.visible(r["_meta"], m)]
        a, vp = answers(vis)
        g = np.array([int(r["gt"]) for r in vis]); mm = np.array([r["_m"] for r in vis])
        sboot[m] = (unpaired_boot(vis, args.boot, rng, False), crit_boot(vp, args.boot, rng, False))
        noun = lambda r: tuple((r["_meta"][f"color_{s}"], r["_meta"][f"shape_{s}"]) for s in
                               (("a", "b") if r["_meta"]["name_first"] == "a" else ("b", "a")))
        shape[m] = dict(auc=av.auc(mm[g == 1], mm[g == 0]), c=a["c"], py=answers(o)[0]["py"], bacc=a["bacc"],
                        degenerate=a["degenerate"], noun_var=var_share(vis, noun), n_vis=len(vis))

    # ---------------------------------------------------------------- real-footage run
    av.RECORDED.clear(); av.load_recorded(args.res)
    meta = av.load_meta(f"{args.res}/videos_v3_metadata.csv")
    ev = defaultdict(dict)
    if Path(args.res, "evidence.csv").exists():
        for r in csv.DictReader(open(Path(args.res, "evidence.csv"))):
            ev[r["sampler"]][r["path"]] = (r["cls"], float(r["evidence"]))
    T = {}
    for m in SMALL:
        rows = av.load_rows(args.res, m, meta)
        if not rows or len(av.by_item(rows, "order_real")) < 40:
            continue
        E = ev.get(SAMPLER.get(m, "uniform8"), {})
        o = order_rows(rows)
        cls = lambda r: E[r["path"]][0] if r["path"] in E else ("state" if av.visible(r["_meta"], m) else "phase")
        mism = sum((cls(r) == "state") != bool(av.visible(r["_meta"], m)) for r in o)
        vis = [r for r in o if cls(r) == "state"]; pho = [r for r in o if cls(r) == "phase"]
        a_all, _ = answers(o); a, vp = answers(vis)
        g = np.array([int(r["gt"]) for r in vis]); mm = np.array([r["_m"] for r in vis])
        t = T[m] = dict(n=len(o), n_vis=len(vis), n_phase=len(pho), n_blind=sum(cls(r) == "blind" for r in o),
                        class_mismatch=mism, py=a_all["py"], py_vis=a["py"], bacc_v=a["bacc"], dprime=a["dprime"],
                        c_v=a["c"], degenerate=a["degenerate"], unparsed=a_all["unparsed"],
                        auc_vis=av.auc(mm[g == 1], mm[g == 0]), auc_vis_p=mw_p(mm[g == 1], mm[g == 0]))
        ub = unpaired_boot(vis, args.boot, rng, True); cb = crit_boot(vp, args.boot, rng, True)
        t.update(auc_vis_ci=ci(ub), c_ci=ci(cb))
        pr = pairs_of(vis)
        t["pauc"], t["pauc_ci"], t["pauc_p"], t["n_pairs"] = paired_auc(pr, args.boot, rng)
        rc, rlo, rhi = av.recal(vis, rng)
        t.update(recal=rc, recal_ci=[rlo, rhi])
        noun = lambda r: (r["_meta"]["noun_a"], r["_meta"]["noun_b"]) if r["_meta"]["name_first"] == "a" else \
                         (r["_meta"]["noun_b"], r["_meta"]["noun_a"])
        t["noun_var"] = var_share(vis, noun)
        # the offset belongs to the pair's content (footage, layout, nouns), of which the nouns are a part
        t["pair_var"] = var_share(vis, lambda r: r["_meta"]["pair_id"])
        tw_m = {r["_meta"]["pair_id"]: r["_m"] for r in av.by_item(rows, "order_twin") if r["_m"] is not None and r["_meta"]}
        pm = defaultdict(list)
        for r in vis:
            pm[r["_meta"]["pair_id"]].append(r["_m"])
        both = [k for k in pm if k in tw_m]
        t["twin_corr"] = float(np.corrcoef([np.mean(pm[k]) for k in both], [tw_m[k] for k in both])[0, 1]) if len(both) > 10 else None
        ph_ans = [r for r in pho if r["pred"] in ("0", "1")]
        if pho:
            pp = pairs_of(pho)
            t["phase_pauc"], t["phase_pauc_ci"], t["phase_p"], t["phase_pairs"] = paired_auc(pp, args.boot, rng)
            t["phase_bacc"] = av.bacc([r for r in pho if r["pred"] in ("0", "1")])

        # move question: real clip against its frozen twin, matched by trial (same test as on shapes)
        real, twin = av.by_item(rows, "moved_real"), av.by_item(rows, "moved_twin")
        if not real or not twin:
            raise SystemExit(f"{m}: move items missing; stage cannot be computed")
        pairs = av.paired(real, twin)
        t.update(move_real=float(np.mean([r["pred"] == "1" for r in real])),
                 move_twin=float(np.mean([r["pred"] == "1" for r in twin])),
                 auc_twin=av.auc([x for x, _ in pairs], [y for _, y in pairs]),
                 auc_twin_ci=list(av.boot_ci(None, None, args.boot, rng, pairs=pairs)),
                 auc_twin_p=av.perm_p([x for x, _ in pairs], [y for _, y in pairs], 2000, rng, pairs=pairs)[0])
        # binding: is the first-named noun in the top half? asked on the frozen twin
        bd = [r for r in av.by_item(rows, "bind_twin") if r["_m"] is not None]
        if bd:
            gb = np.array([int(r["gt"]) for r in bd]); mb = np.array([r["_m"] for r in bd])
            t.update(bind_bacc=av.bacc([r for r in bd if r["pred"] in ("0", "1")]), bind_auc=av.auc(mb[gb == 1], mb[gb == 0]),
                     bind_py=float(np.mean([r["pred"] == "1" for r in bd])), n_bind=len(bd))
        # paraphrase, order on the twin, order on blank video
        rv = [r for r in av.by_item(rows, "order_rev") if r["_m"] is not None and r["_meta"] and cls(dict(r, path=r["path"])) == "state"]
        if rv:
            gr = np.array([int(r["gt"]) for r in rv]); mr = np.array([r["_m"] for r in rv])
            t.update(rev_bacc_v=av.bacc([r for r in rv if r["pred"] in ("0", "1")]), rev_auc_v=av.auc(mr[gr == 1], mr[gr == 0]),
                     rev_py=float(np.mean([r["pred"] == "1" for r in rv])), n_rev=len(rv))
        for item, key in (("order_twin", "twin"), ("order_blank", "blank")):
            rr = [r for r in av.by_item(rows, item) if r["pred"] in ("0", "1")]
            if rr:
                t[f"{key}_py"] = float(np.mean([r["pred"] == "1" for r in rr]))
        # leave one uploader / one category out, on the paired AUC
        for name, keyf in (("who", lambda r: {who(r, "a"), who(r, "b")}),
                           ("cat", lambda r: {r["_meta"]["cat_a"], r["_meta"]["cat_b"]})):
            vals = []
            for u in sorted({x for p, _ in pr for x in keyf(p)}):
                keep = [(p, n) for p, n in pr if u not in keyf(p)]
                if len(keep) >= 30:
                    vals.append(float(np.mean([(p["_m"] > n["_m"]) + .5 * (p["_m"] == n["_m"]) for p, n in keep])))
            t[f"loo_{name}"] = [min(vals), max(vals)] if vals else [None, None]
        # evidence terciles (pairs share their evidence value)
        if E:
            e = np.array([E[p["path"]][1] for p, _ in pr]); q = np.quantile(e, [1 / 3, 2 / 3])
            t["evidence_terciles"] = [float(np.mean([(p["_m"] > n["_m"]) + .5 * (p["_m"] == n["_m"])
                                                    for (p, n), x in zip(pr, e) if lo <= x < hi]))
                                      for lo, hi in ((-1, q[0]), (q[0], q[1]), (q[1], 1e9))]
        # bands, and the outer-minus-middle contrast (P4)
        t["bands"] = {}
        band_pairs = {}
        for bname, lo, hi in BANDS:
            s = [r for r in vis if lo <= int(r["interval_ms"]) < hi]
            bp = band_pairs[bname] = pairs_of(s)
            if len(bp) < 10:
                t["bands"][bname] = dict(n=len(s)); continue
            pa, pci, pp_, npair = paired_auc(bp, args.boot // 2, rng)
            gs = np.array([int(r["gt"]) for r in s]); ms = np.array([r["_m"] for r in s])
            t["bands"][bname] = dict(n=len(s), pauc=pa, pauc_ci=pci, p=pp_, auc=av.auc(ms[gs == 1], ms[gs == 0]),
                                     bacc=av.bacc([r for r in s if r["pred"] in ("0", "1")]))
        names = [b for b, _, _ in BANDS]
        outer = band_pairs[names[0]] + band_pairs[names[3]]; mid = band_pairs[names[1]] + band_pairs[names[2]]
        if len(outer) >= 20 and len(mid) >= 20:
            allp = outer + mid; isout = np.array([1] * len(outer) + [0] * len(mid), bool)
            ind = np.array([(p["_m"] > n["_m"]) + .5 * (p["_m"] == n["_m"]) for p, n in allp])
            draw = cluster_counts([p for p, _ in allp], rng); ds = []
            for _ in range(args.boot):
                w = draw()
                if w[isout].sum() > 0 and w[~isout].sum() > 0:
                    ds.append((w * ind)[isout].sum() / w[isout].sum() - (w * ind)[~isout].sum() / w[~isout].sum())
            t["dip"] = float(ind[isout].mean() - ind[~isout].mean()); t["dip_lo95"] = float(np.percentile(ds, 5))

        if m in shape:
            s = shape[m]; n = min(len(ub), len(sboot[m][0])); nc = min(len(cb), len(sboot[m][1]))
            t.update(shape_auc=s["auc"], shape_c=None if s["degenerate"] else s["c"], shape_py=s["py"], shape_bacc=s["bacc"],
                     shape_noun_var=s["noun_var"], d_auc=t["auc_vis"] - s["auc"], d_auc_ci=ci(ub[:n] - sboot[m][0][:n]))
            if not s["degenerate"] and not t["degenerate"]:
                t.update(d_c=t["c_v"] - s["c"], d_c_ci=ci(cb[:nc] - sboot[m][1][:nc]))

    ks = list(T)
    for key in ("auc_vis_p", "pauc_p", "auc_twin_p"):
        for k, v in zip(ks, av.holm([T[k][key] for k in ks])):
            T[k][key + "_holm"] = v
    cells = [(m, b) for m in ks for b in T[m]["bands"] if "p" in T[m]["bands"][b]]
    for (m, b), ph in zip(cells, av.holm([T[m]["bands"][b]["p"] for m, b in cells])):
        c = T[m]["bands"][b]
        c.update(p_holm=ph, margin=bool(ph < .05 and c["pauc"] > .5), answer=bool((c["bacc"] or 0) >= BAR))
    for m in ks:
        t = T[m]
        t["stage"] = av.stage(t)                                        # the shape run's rule, unpaired test
        t["stage_paired"] = av.stage(dict(t, auc_vis=t["pauc"], auc_vis_p_holm=t["pauc_p_holm"]))
        t["rev_n_asked"] = len(av.by_item(av.load_rows(args.res, m, meta), "order_rev"))

    # ---------------------------------------------------------------- report
    print(f"{'model':17} {'nvis':>4} {'P(Y)':>5} {'BAcc':>5} {'c':>5} {'AUC [95% CI]':>19} {'pAUC [95% CI]':>19} {'pHolm':>7} "
          f"{'recal':>5} {'mvR':>4} {'mvT':>4} {'AUCtw':>5} {'bind':>5} {'stage':>12} {'stage(pair)':>12} | "
          f"{'shpAUC':>6} {'dAUC':>6} {'shp c':>5} {'dc':>6} | noun var real/shape")
    for m in ks:
        t = T[m]
        print(f"{av.NICE[m]:17} {t['n_vis']:4d} {t['py']:5.3f} {t['bacc_v']:5.3f} {t['c_v']:5.2f} "
              f"{t['auc_vis']:.3f} [{t['auc_vis_ci'][0]:.3f},{t['auc_vis_ci'][1]:.3f}] "
              f"{t['pauc']:.3f} [{t['pauc_ci'][0]:.3f},{t['pauc_ci'][1]:.3f}] {t['pauc_p_holm']:7.4f} {(t['recal'] or 0):5.3f} "
              f"{t['move_real']:4.2f} {t['move_twin']:4.2f} {t['auc_twin']:5.3f} {f3(t.get('bind_bacc')):>5} "
              f"{t['stage']:>12} {t['stage_paired']:>12} | {f3(t.get('shape_auc')):>6} {f3(t.get('d_auc'), True):>6} "
              f"{'deg.' if t.get('shape_c') is None else format(t['shape_c'], '.2f'):>5} {f3(t.get('d_c'), True):>6} | "
              f"{f3(t.get('noun_var'))} / {f3(t.get('shape_noun_var'))} | pair var {f3(t.get('pair_var'))} twin corr {f3(t.get('twin_corr'))} | pAUC p {t['pauc_p']:.3g} holm {t['pauc_p_holm']:.3g}")
    print("\nclasses, phase-only trials, paraphrase, twin/blank, leave-one-out, evidence terciles, dip")
    for m in ks:
        t = T[m]
        print(f"{av.NICE[m]:17} state {t['n_vis']} phase {t['n_phase']} blind {t['n_blind']} (rule mismatch {t['class_mismatch']}) | "
              f"phase pAUC {f3(t.get('phase_pauc'))} {t.get('phase_pauc_ci')} | rev BAcc {f3(t.get('rev_bacc_v'))} AUC {f3(t.get('rev_auc_v'))} | "
              f"P(Y) twin {f3(t.get('twin_py'))} blank {f3(t.get('blank_py'))} | LOO uploader {t['loo_who']} category {t['loo_cat']} | "
              f"evidence {t.get('evidence_terciles')} | dip {f3(t.get('dip'), True)} (5th pct {f3(t.get('dip_lo95'), True)}) | unparsed {t['unparsed']}")
    print("\nbands: paired AUC (* = significant after Holm) / BAcc / trials")
    for m in ks:
        print(f"{av.NICE[m]:17} " + "  ".join(
            f"{c['pauc']:.2f}{'*' if c.get('margin') else ' '}/{c['bacc']:.2f}/{c['n']:<4}" if "pauc" in c else f"-- n={c['n']}"
            for c in (T[m]["bands"][b] for b, _, _ in BANDS)))

    Path(args.res, "numbers_real.json").write_text(json.dumps(dict(real=T, shape=shape), indent=1, default=float))
    lines, blines = [], []
    names = [b for b, _, _ in BANDS]
    for m in ks:
        t = T[m]
        dag = "" if (t["pauc_p_holm"] < .05 and t["pauc"] > .5) else "$^{\\dagger}$"
        cv = "\\emph{deg.}" if t["degenerate"] else f"{t['c_v']:.2f}"
        sc = "\\emph{deg.}" if t.get("shape_c") is None else f"{t['shape_c']:.2f}"
        lines.append(f"{av.NICE[m]:17} & {f3(t['py'])} & {f3(t['bacc_v'])} & {cv} & {f3(t['auc_vis'])} & "
                     f"{f3(t['pauc'])} [{f3(t['pauc_ci'][0])}, {f3(t['pauc_ci'][1])}]{dag} & {f3(t['recal'])} & "
                     f"{f3(t['auc_twin'])} & {f3(t.get('bind_bacc'))} & \\emph{{{t['stage']}}} & "
                     f"{f3(t.get('shape_py'))} & {f3(t.get('shape_bacc'))} & {sc} & {f3(t.get('shape_auc'))} \\\\")
        cell = []
        for b in names:
            c = t["bands"][b]
            if "pauc" not in c:
                cell.append("-- & --"); continue
            a = f"{c['pauc']:.2f}".replace("0.", ".", 1) + ("" if c["margin"] else "$^{\\dagger}$")
            ba = f"{c['bacc']:.2f}".replace("0.", ".", 1)
            cell.append(f"{a} & " + (f"\\textbf{{{ba}}}" if c["answer"] else ba))
        span = lambda x: "all" if len(x) == 4 else "none" if not x else ", ".join(x)
        blines.append(f"{av.NICE[m]:17} & " + " & ".join(cell) + " & " +
                      span([b for b in names if t["bands"][b].get("margin")]) + " & " +
                      span([b for b in names if t["bands"][b].get("answer")]) + " \\\\")
    Path(args.res, "real_rows.tex").write_text("\n".join(lines) + "\n")
    Path(args.res, "real_band_rows.tex").write_text("\n".join(blines) + "\n")


if __name__ == "__main__":
    main()
