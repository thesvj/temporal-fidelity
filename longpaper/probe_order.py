"""
CPU analysis of features from extract_order_feats.py.

For every (condition in real/shuf/drop) x (pool in mean/last) x layer, and encoder layers:
  labels: gt (the answer to the question as asked; v3 names either shape first), a_first (the world fact:
          did shape A move first) and interval4
  - 5-fold StratifiedKFold(shuffle=True, seed 0)  -> acc, balanced acc
  - 5-fold unshuffled (the paper's cross_val_score default) -> acc (comparability)
  - majority-class baseline
  - acc split by VISIBILITY: does any frame the adapter actually decodes fall between the two jumps?
    Taken from analyze_v3.visible(), i.e. from the recorded frame indices and the stimulus metadata.
    (The previous visible_intermediate() assumed a fixed 1.0 s lead-in that v2/v3 do not use, so its
    "invisible" split contained visible trials.)
  - label-permutation null (n_perm) at the best layer per (cond,pool,label)
Behaviour: parses real/shuf responses -> P(YES), BAcc, d' (loglinear), overall and by visibility;
compares real responses with results_v3/<model>_expg3.csv order_real (reproducibility).

Usage: python longpaper/probe_order.py --model molmo2 [--npz ...] [--n-perm 5] [--n-jobs 4]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ACL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_v3 as av  # noqa: E402  exact per-trial visibility from recorded frame indices


def parse_yn(t):
    t = str(t).upper()
    y, no = bool(re.search(r"\bYES\b", t)), bool(re.search(r"\bNO\b", t))
    if y and not no:
        return 1
    if no and not y:
        return 0
    return 1 if y else np.nan


def sdt(gt, pred):
    m = ~np.isnan(pred)
    gt, pred = gt[m].astype(int), pred[m].astype(int)
    if len(gt) == 0:
        return dict(n=0)
    s = gt == 1
    h, fa, ns, nn = ((pred == 1) & s).sum(), ((pred == 1) & ~s).sum(), s.sum(), (~s).sum()
    hr, far = (h + .5) / (ns + 1), (fa + .5) / (nn + 1)
    return dict(n=len(gt), p_yes=pred.mean(), bacc=((h / max(ns, 1)) + ((nn - fa) / max(nn, 1))) / 2,
                dprime=norm.ppf(hr) - norm.ppf(far), c=-.5 * (norm.ppf(hr) + norm.ppf(far)))


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--meta", default=str(ACL / "data" / "videos_v3" / "metadata.csv"))
    ap.add_argument("--frames", default=str(ACL / "results_v3"), help="dir with sampled_frames_*.jsonl")
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--outdir", default=str(ACL / "longpaper" / "results"))
    ap.add_argument("--smoke", action="store_true", help="shape/sanity checks only, no CV")
    args = ap.parse_args()

    npz = Path(args.npz or ACL / "longpaper" / "feats" / f"{args.model}.npz")
    z = np.load(npz, allow_pickle=True)
    iv = z["interval_ms"].astype(int)
    y_order = z["a_first"].astype(int)
    y_gt = z["gt"].astype(int) if "gt" in z.files else y_order
    y_int = np.array(["A" if x < 1000 else "B" if x < 2000 else "C" if x < 5000 else "D" for x in iv])
    meta = av.load_meta(args.meta)
    av.load_recorded(args.frames)
    vv = [av.visible(meta.get(str(p)), args.model) for p in z["path"]]
    if any(v is None for v in vv):
        raise SystemExit(f"{sum(v is None for v in vv)} feature rows have no metadata row in {args.meta}")
    vis = np.array(vv, dtype=bool)
    feat_keys = sorted(k for k in z.files if re.match(r"(real|shuf|drop|enc)_(mean|last)_L\d+$", k))
    print(f"{args.model}: n={len(iv)} keys={len(feat_keys)} visible={vis.mean():.3f} "
          f"order-ceiling(visible->1, invisible->0.5)={vis.mean() + .5 * (~vis).mean():.3f}", flush=True)

    # ---------------- behaviour
    beh = []
    for cond in ("real", "shuf", "drop"):
        if f"{cond}_resp" not in z.files:
            continue
        pred = np.array([parse_yn(r) for r in z[f"{cond}_resp"]], dtype=float)
        for name, m in (("all", np.ones_like(vis)), ("visible", vis), ("invisible", ~vis)):
            r = sdt(y_gt[m], pred[m])
            r.update(model=args.model, cond=cond, subset=name, parse_rate=float(np.mean(~np.isnan(pred[m]))) if m.any() else np.nan)
            beh.append(r)
        if cond == "real":
            orig = Path(args.frames) / f"{args.model}_expg3.csv"   # the behavioural v3 run, same prompts
            if orig.exists():
                o = pd.read_csv(orig)
                o = o[o["item"] == "order_real"].set_index("path")["response"].astype(str)
                paths = z["path"].astype(str)
                hit = [p in o.index for p in paths]
                same = np.mean([parse_yn(o[p]) == parse_yn(r)
                                for p, r, h in zip(paths, z["real_resp"], hit) if h]) if any(hit) else np.nan
                print(f"  reproducibility vs exp_g3 order_real ({sum(hit)} matched, same parsed answer): {same:.3f}",
                      flush=True)
    for key in ("real_seqlen", "shuf_seqlen"):
        if key in z.files:
            sl = z[key].astype(int)
            print(f"  {key}: unique={len(np.unique(sl))} min={sl.min()} max={sl.max()} "
                  f"corr(seqlen, interval)={np.corrcoef(sl, iv)[0, 1] if sl.std() > 0 else float('nan'):.3f}", flush=True)
    beh = pd.DataFrame(beh)
    print(beh.to_string(index=False), flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        for k in feat_keys:
            X = z[k].astype(np.float32)
            print(f"  {k}: shape={X.shape} finite={np.isfinite(X).all()} mean_feature_std={X.std(0).mean():.4f}")
        beh.to_csv(outdir / f"{npz.stem}_behaviour.csv", index=False)
        return

    rows = []
    cv_s = StratifiedKFold(5, shuffle=True, random_state=0)
    for k in feat_keys:
        X = z[k].astype(np.float32)
        const = X.std(0).mean() < 1e-6
        for lab_name, y in (("gt", y_gt), ("a_first", y_order), ("interval4", y_int)):
            maj = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=cv_s).mean()
            if const:
                rows.append(dict(model=args.model, feat=k, label=lab_name, acc=np.nan, majority=maj, note="constant features"))
                continue
            pr = cross_val_predict(clf(), X, y, cv=cv_s, n_jobs=args.n_jobs)
            acc_unshuf = cross_val_score(clf(), X, y, cv=5, n_jobs=args.n_jobs).mean()
            rows.append(dict(model=args.model, feat=k, label=lab_name,
                             acc=np.mean(pr == y), bacc=balanced_accuracy_score(y, pr),
                             acc_visible=np.mean(pr[vis] == y[vis]) if vis.any() else np.nan,
                             acc_invisible=np.mean(pr[~vis] == y[~vis]) if (~vis).any() else np.nan,
                             acc_unshuffled_cv=acc_unshuf, majority=maj,
                             feature_std=float(X.std(0).mean())))
            print(f"  {k:18s} {lab_name:9s} acc={rows[-1]['acc']:.3f} bacc={rows[-1]['bacc']:.3f} "
                  f"vis={rows[-1]['acc_visible']:.3f} invis={rows[-1]['acc_invisible']:.3f} "
                  f"unshufCV={acc_unshuf:.3f} maj={maj:.3f}", flush=True)
    res = pd.DataFrame(rows)

    # permutation null at best layer per (cond/pool prefix, label)
    rng = np.random.default_rng(0)
    null_rows = []
    res["prefix"] = res["feat"].str.replace(r"_L\d+$", "", regex=True)
    for (prefix, lab), g in res.dropna(subset=["acc"]).groupby(["prefix", "label"]):
        best = g.loc[g["acc"].idxmax(), "feat"]
        X = z[best].astype(np.float32)
        y = {"gt": y_gt, "a_first": y_order}.get(lab, y_int)
        accs = [cross_val_score(clf(), X, rng.permutation(y), cv=cv_s, n_jobs=args.n_jobs).mean() for _ in range(args.n_perm)]
        null_rows.append(dict(model=args.model, feat=best, label=lab, null_mean=np.mean(accs), null_max=np.max(accs)))
        print(f"  NULL {best} {lab}: mean={np.mean(accs):.3f} max={np.max(accs):.3f}", flush=True)

    res.to_csv(outdir / f"{args.model}_order_probe.csv", index=False)
    pd.DataFrame(null_rows).to_csv(outdir / f"{args.model}_order_probe_null.csv", index=False)
    beh.to_csv(outdir / f"{args.model}_behaviour_shuffle.csv", index=False)
    print(f"PROBE_DONE {args.model}", flush=True)


if __name__ == "__main__":
    main()
