"""Order probes without selection optimism, with a real permutation null. CPU only.

Replaces the "best layer on the same folds, 20 permutations" numbers. The probe is a ridge classifier on
standardised last-token states, in dual form: for a (layer, strength, fold) the map from training labels to
test scores is one matrix, so a label permutation costs a matrix-vector product and the whole procedure,
including the choice of layer and strength, is rerun for each of 1,000 permutations.

  nested   outer 5 folds; inside each training set, 4 inner folds pick (layer, ridge strength); the outer
           test fold is scored once with that choice. No test trial influences the choice.
  fixed    a layer named in advance (for footage: the layer the shape run selected for that model).

Shape run (trials independent): folds stratified by label; accuracy on all / visible / invisible trials;
null = labels permuted. Also by temporal-patch alignment class for uniform-8 models.

Real-footage run (order-swapped pairs share nearly all pixels and have opposite labels): folds keep pairs
together; the score is the paired accuracy, the share of held-out visible pairs in which the probe scores
the gt=1 member above the gt=0 member (same unit as the margin's paired AUC); null = labels swapped within
random pairs. Also an author-disjoint scheme (train on pairs sharing neither author with the test pair) at
the fixed layer, and the per-pair difference between probe and margin with a cluster-bootstrap interval.

Usage:
  uv run --no-project --with numpy python longpaper/probe_stats.py --set v3   --model qwen2.5-vl
  uv run --no-project --with numpy python longpaper/probe_stats.py --set real --model qwen2.5-vl
Writes longpaper/results_stats/<set>_<model>.json.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze_v3 as av  # noqa: E402

ALPHAS = (0.1, 1.0, 10.0)      # ridge strength as a multiple of the feature dimension


def folds(groups, y, k, rng):
    """k folds over groups, stratified by label when every group is one trial."""
    ug = np.unique(groups)
    if len(ug) == len(groups):                       # independent trials: stratify
        out = np.empty(len(groups), int)
        for c in np.unique(y):
            i = np.where(y == c)[0]; rng.shuffle(i)
            out[i] = np.arange(len(i)) % k
        return out
    perm = rng.permutation(len(ug))
    fold_of = {g: j % k for j, g in enumerate(ug[perm])}
    return np.array([fold_of[g] for g in groups])


def dual_map(Xtr, Xte, alpha):
    """Matrix P with scores(test) = P @ (y_train - mean) + mean, for ridge on standardised features."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    K = A @ A.T
    lam = alpha * A.shape[1]
    return (B @ A.T) @ np.linalg.inv(K + lam * np.eye(len(K)))


def score(P, ytr):
    m = ytr.mean()
    return P @ (ytr - m) + m


def paired_acc(s, y, pair):
    """Share of pairs (both members present) in which the gt=1 member scores higher; and the indicators."""
    by = defaultdict(dict)
    for si, yi, pi in zip(s, y, pair):
        by[pi][int(yi)] = si
    ind = {p: float(d[1] > d[0]) + .5 * float(d[1] == d[0]) for p, d in by.items() if len(d) == 2}
    return (float(np.mean(list(ind.values()))) if ind else np.nan), ind


def build(X, y, groups, k_out, k_in, rng):
    """All dual maps needed: outer[(f, l, a)] and inner[(f, l, a, g)] with their index sets."""
    fo = folds(groups, y, k_out, rng)
    outer, inner, fin = {}, {}, {}
    for f in range(k_out):
        tr, te = np.where(fo != f)[0], np.where(fo == f)[0]
        fi = folds(groups[tr], y[tr], k_in, rng); fin[f] = (tr, te, fi)
        for l, Xl in X.items():
            for a in ALPHAS:
                outer[(f, l, a)] = dual_map(Xl[tr], Xl[te], a).astype(np.float32)
                for g in range(k_in):
                    itr, ite = tr[fi != g], tr[fi == g]
                    inner[(f, l, a, g)] = dual_map(Xl[itr], Xl[ite], a).astype(np.float32)
    return fo, fin, outer, inner


def run(y, fin, outer, inner, layers, k_in, metric, fixed=None):
    """Scores for every trial under nested selection (or a fixed layer, strength still chosen inside)."""
    s = np.empty(len(y)); chosen = []
    for f, (tr, te, fi) in fin.items():
        best, arg = -1, None
        for l in ([fixed] if fixed is not None else layers):
            for a in ALPHAS:
                si = np.empty(len(tr))
                for g in range(k_in):
                    si[fi == g] = score(inner[(f, l, a, g)], y[tr][fi != g])
                v = metric(si, tr)
                if v > best:
                    best, arg = v, (l, a)
        s[te] = score(outer[(f, arg[0], arg[1])], y[tr]); chosen.append(arg)
    return s, chosen


def cluster_ci(ind, who, B, rng):
    """Two-way cluster bootstrap over authors for a mean of per-pair values. who: pair -> (author_a, author_b)."""
    pairs = list(ind); v = np.array([ind[p] for p in pairs])
    ids = sorted({u for p in pairs for u in who[p]}); ix = {u: i for i, u in enumerate(ids)}
    a = np.array([ix[who[p][0]] for p in pairs]); b = np.array([ix[who[p][1]] for p in pairs])
    out = []
    for _ in range(B):
        c = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)); w = c[a] * c[b]
        if w.sum() > 0:
            out.append((w * v).sum() / w.sum())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["v3", "real"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cond", default="real", help="input condition of the features: real / shuf / drop")
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feats", default=None, help="feature dir under longpaper/ (default: the set's own)")
    ap.add_argument("--suffix", default="", help="appended to the output name, e.g. _sys")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    real = args.set == "real"
    res = ROOT / ("results_real" if real else "results_v3")
    z = np.load((ROOT / "longpaper" / (("feats_real" if args.cond == "real" else "feats_real_ctl") if real else "feats_v3") if not args.feats else ROOT / "longpaper" / args.feats) / f"{args.model}.npz", allow_pickle=True)
    av.RECORDED.clear(); av.load_recorded(str(res)); meta = av.load_meta(str(res / "videos_v3_metadata.csv"))
    paths = [str(p) for p in z["path"]]
    vis = np.array([bool(av.visible(meta[p], args.model)) for p in paths])
    keys = sorted((k for k in z.files if re.match(rf"{args.cond}_last_L\d+$", k)), key=lambda k: int(k.rsplit("L", 1)[1]))
    layers = [int(k.rsplit("L", 1)[1]) for k in keys]
    X = {l: z[k].astype(np.float64) for l, k in zip(layers, keys)}
    n = len(paths)
    groups = z["pair_id"].astype(int) if real else np.arange(n)
    k_out, k_in = 5, 4
    out = dict(model=args.model, set=args.set, cond=args.cond, n=n, n_visible=int(vis.sum()), layers=layers)

    # layer fixed in advance: the shape run's selected layer (from its published probe table)
    fixed = None
    f = ROOT / "longpaper" / "results_v3" / f"{args.model}_order_probe.csv"
    if f.exists():
        rows = [r for r in csv.DictReader(open(f)) if r["label"] == "gt" and r["feat"].startswith("real_last_L") and r["acc"]]
        fixed = int(max(rows, key=lambda r: float(r["acc"]))["feat"].rsplit("L", 1)[1])
        out["fixed_layer"] = fixed

    for lab in (("gt", "a_first") if real else ("gt",)):
        y = z[lab].astype(float)
        pair = groups

        def metric(s, idx, _y=y):
            if real:
                m = vis[idx]
                return paired_acc(s[m], _y[idx][m], pair[idx][m])[0]
            return float(np.mean((s > .5) == (_y[idx] > .5)))

        fo, fin, outer, inner = build(X, y, groups, k_out, k_in, np.random.default_rng(args.seed))
        r = {}
        for name, fx in (("nested", None), ("fixed", fixed)):
            if name == "fixed" and fx is None:
                continue
            s, chosen = run(y, fin, outer, inner, layers, k_in, metric, fixed=fx)
            if real:
                acc, ind = paired_acc(s[vis], y[vis], pair[vis])
                r[name] = dict(paired_acc=acc, n_pairs=len(ind), chosen=[list(c) for c in chosen])
                r[name]["_ind"] = ind
            else:
                hit = (s > .5) == (y > .5)
                r[name] = dict(acc=float(hit.mean()), acc_visible=float(hit[vis].mean()),
                               acc_invisible=float(hit[~vis].mean()) if (~vis).any() else None,
                               chosen=[list(c) for c in chosen])
                r[name]["_hit"] = hit
            # null: the whole procedure, selection included, on permuted labels
            null = []
            for _ in range(args.perm):
                if real:     # swap the labels of the two members in a random half of the pairs
                    flip = {g: rng.random() < .5 for g in np.unique(groups)}
                    yp = np.array([1 - yi if flip[g] else yi for yi, g in zip(y, groups)])
                else:
                    yp = rng.permutation(y)
                mp = (lambda s_, idx, _y=yp: paired_acc(s_[vis[idx]], _y[idx][vis[idx]], pair[idx][vis[idx]])[0]) if real else \
                     (lambda s_, idx, _y=yp: float(np.mean((s_ > .5) == (_y[idx] > .5))))
                sp, _ = run(yp, fin, outer, inner, layers, k_in, mp, fixed=fx)
                null.append(paired_acc(sp[vis], yp[vis], pair[vis])[0] if real else float(np.mean(((sp > .5) == (yp > .5))[vis])))
            obs = r[name]["paired_acc"] if real else r[name]["acc_visible"]
            null = np.array(null)
            r[name].update(null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)), null_max=float(null.max()),
                           p=float((1 + (null >= obs).sum()) / (1 + len(null))))
        # per-layer curve (strength chosen inside, layer fixed): for the appendix figure/table
        curve = {}
        for l in layers:
            s, _ = run(y, fin, outer, inner, layers, k_in, metric, fixed=l)
            curve[l] = paired_acc(s[vis], y[vis], pair[vis])[0] if real else float(np.mean(((s > .5) == (y > .5))[vis]))
        r["curve_visible"] = curve

        if real:
            who = {int(p): (str(a), str(b)) for p, a, b in zip(z["pair_id"], z["who_a"], z["who_b"])}
            for name in ("nested", "fixed"):
                if name in r:
                    r[name]["ci"] = cluster_ci(r[name]["_ind"], who, 5000, rng)
            # margin on the same pairs, and the per-pair difference
            marg = {}
            for row in csv.DictReader(open(res / f"{args.model}_expg3.csv")):
                if row["item"] == "order_real" and row["margin"] != "":
                    marg[row["path"]] = float(row["margin"])
            mi = paired_acc(np.array([marg[p] for p, v in zip(paths, vis) if v]), z["gt"].astype(float)[vis], pair[vis])[1]
            if lab == "gt" and "fixed" in r:
                pi = r["fixed"]["_ind"]; common = [p for p in pi if p in mi]
                diff = {p: pi[p] - mi[p] for p in common}
                r["probe_minus_margin"] = dict(probe=float(np.mean([pi[p] for p in common])), margin=float(np.mean([mi[p] for p in common])),
                                               diff=float(np.mean(list(diff.values()))), ci=cluster_ci(diff, who, 5000, rng), n_pairs=len(common))
                # author-disjoint: train only on pairs that share neither author with the test pairs
                Xl = X[fixed]; a = max(set(c[1] for c in r["fixed"]["chosen"]), key=[c[1] for c in r["fixed"]["chosen"]].count)
                au = np.array([frozenset(who[int(p)]) for p in pair], dtype=object)
                ind = {}
                for combo in set(au):
                    te = np.where(au == combo)[0]
                    tr = np.array([i for i in range(n) if not (au[i] & combo)])
                    s = score(dual_map(Xl[tr], Xl[te], a), y[tr])
                    m = vis[te]
                    ind.update(paired_acc(s[m], y[te][m], pair[te][m])[1])
                r["author_disjoint"] = dict(paired_acc=float(np.mean(list(ind.values()))), n_pairs=len(ind), ci=cluster_ci(ind, who, 5000, rng))
            # agreement of the extraction pass with the behavioural run
            if lab == "gt":
                beh = {row["path"]: row["pred"] for row in csv.DictReader(open(res / f"{args.model}_expg3.csv")) if row["item"] == "order_real"}
                yn = lambda t: "1" if re.search(r"\bYES\b", str(t).upper()) else "0" if re.search(r"\bNO\b", str(t).upper()) else ""
                if "real_resp" in z.files:               # control-feature archives hold no real-input responses
                    out["answer_agreement"] = float(np.mean([yn(rp) == beh.get(p, "?") for p, rp in zip(paths, z["real_resp"])]))
        else:
            # by temporal-patch alignment class (uniform-8 loaders): slot of the first delivered frame after each jump
            cls = []
            for p in paths:
                idx = av.sampled(meta[p], args.model); a_, b_ = sorted((int(float(meta[p]["f_first"])), int(float(meta[p]["f_second"]))))
                s_ = next((k for k, i in enumerate(idx) if i >= a_), None); e_ = next((k for k, i in enumerate(idx) if i >= b_), None)
                cls.append("na" if s_ is None or e_ is None else "s_odd" if s_ % 2 else "s_even_e_odd" if e_ % 2 else "s_even_e_even")
            cls = np.array(cls)
            if len(av.sampled(meta[paths[0]], args.model)) == 8:
                r["nested"]["by_alignment"] = {c: dict(n=int(((cls == c) & vis).sum()), acc=float(r["nested"]["_hit"][(cls == c) & vis].mean()))
                                              for c in ("s_odd", "s_even_e_odd", "s_even_e_even") if ((cls == c) & vis).any()}
        for name in ("nested", "fixed"):
            if name in r:
                r[name].pop("_ind", None); r[name].pop("_hit", None)
        out[lab] = r
        print(json.dumps({lab: r}, indent=1, default=float)[:2500], flush=True)

    od = ROOT / "longpaper" / "results_stats"; od.mkdir(exist_ok=True)
    tag = f"{args.set}_{args.model}" + ("" if args.cond == "real" else f"_{args.cond}") + args.suffix
    (od / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", od / f"{tag}.json")


if __name__ == "__main__":
    main()
