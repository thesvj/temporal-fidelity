"""Steering on v3, reading the margin rather than only the answer.

The v1 steering run (InternVL2.5, v1 stimuli) recorded only the parsed answer, so with a criterion as
extreme as that model's it could not tell "the direction does nothing" from "the direction moves the
YES-NO margin without crossing the threshold". This run records the first-token margin.

Direction: difference of class means (gt=1 minus gt=0) of last-prefill-token states at layer L, from
the order-question forward pass (feats_v3/<model>.npz, extract_order_feats.py), on a training half of
the visible trials. It is added, scaled by alpha times the mean state norm, to the output of decoder
layer L at every position. Control: a random unit direction, same scaling. Evaluated on held-out
visible trials.

Readouts per (layer, direction, alpha): mean margin, P(YES), AUC of the margin for gt, and the mean
margin separately for gt=1 and gt=0 (a direction that carries the order should move the two apart
in the direction of its sign only if it is used downstream).

Usage (inside the GPU shell): .venv-qwen2.5-vl/bin/python steer_v3.py --model qwen2.5-vl
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ACL = Path(__file__).resolve().parent
sys.path.insert(0, str(ACL))

import exp_g3 as eg  # noqa: E402  (patches generate to capture first-token logits)
import analyze_v3 as av  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 21])
    ap.add_argument("--alphas", type=float, nargs="+", default=[-1, -.5, -.25, .25, .5, 1])
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--out", default="results_v3/steer_qwen2.5-vl.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controls", action="store_true", help="add PC2 and label-permuted directions")
    ap.add_argument("--mode", choices=["add", "ablate"], default="add",
                    help="add: h + alpha*nbar*v; ablate: h - (h.v)v, i.e. remove the direction (test of use)")
    args = ap.parse_args()

    z = np.load(ACL / "longpaper/feats_v3" / f"{args.model}.npz", allow_pickle=True)
    meta = pd.read_csv(ACL / "data/videos_v3/metadata.csv").set_index("path")
    mrow = av.load_meta(str(ACL / "data/videos_v3/metadata.csv"))
    av.load_recorded(str(ACL / "results_v3"))
    paths = [str(p) for p in z["path"]]
    gt = z["gt"].astype(int)
    vis = np.array([av.visible(mrow.get(p), args.model) for p in paths], dtype=bool)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(np.where(vis)[0])
    test, train = idx[: args.n_test], idx[args.n_test:]
    print(f"visible {vis.sum()} train {len(train)} test {len(test)} test gt=1 {gt[test].mean():.3f}", flush=True)

    model = eg.load_model(args.model)
    tok = eg.find_tokenizer(model)
    yes = eg.variant_ids(tok, ["YES", "Yes", "yes"])
    no = eg.variant_ids(tok, ["NO", "No", "no"])
    sh = set(yes) & set(no)
    yes, no = [i for i in yes if i not in sh], [i for i in no if i not in sh]
    blocks = None
    for name, m in model._model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == model.n_llm_layers and "visual" not in name:
            blocks = m
    state = {"vec": None, "unit": None}

    def edit(h):
        if state["unit"] is not None:   # projection-out ablation
            u = state["unit"].to(h.dtype)
            return h - (h @ u).unsqueeze(-1) * u
        return h + state["vec"].to(h.dtype)

    def hook(_m, _i, out):
        if state["vec"] is None and state["unit"] is None:
            return out
        if isinstance(out, tuple):
            return (edit(out[0]),) + tuple(out[1:])
        return edit(out)

    handles = [None]
    new = not Path(args.out).exists()
    fh = open(args.out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["layer", "direction", "alpha", "path", "gt", "interval_ms", "margin", "pred", "response"])
    for L in args.layers:
        key = f"real_last_L{L}"
        if key not in z.files:
            print(f"skip layer {L}: {key} not in npz", flush=True)
            continue
        X = z[key].astype(np.float32)
        d = X[train][gt[train] == 1].mean(0) - X[train][gt[train] == 0].mean(0)
        nbar = float(np.linalg.norm(X[train], axis=1).mean())
        r = rng.standard_normal(d.shape).astype(np.float32)
        dirs = {"order": d / np.linalg.norm(d), "random": r / np.linalg.norm(r)}
        if args.controls:   # variance-matched controls: PC2 of train states (not order-related), label-permuted mean differences
            Xt = X[train] - X[train].mean(0)
            _, _, Vt = np.linalg.svd(Xt, full_matrices=False)
            dirs["pc2"] = Vt[1] / np.linalg.norm(Vt[1])
            for s in range(1, 4):
                g = np.random.default_rng(s).permutation(gt[train])
                dp = X[train][g == 1].mean(0) - X[train][g == 0].mean(0)
                dirs[f"perm{s}"] = dp / np.linalg.norm(dp)
            Xc = X[test] - X[test].mean(0)
            tot = float((Xc ** 2).sum(1).mean())
            print("  variance fraction along each direction (held-out):",
                  {k: round(float(((Xc @ v) ** 2).mean()) / tot, 4) for k, v in dirs.items()}, flush=True)
        # how far apart the class means are along the direction, in units of nbar (for scale)
        proj = X[test] @ dirs["order"]
        print(f"L{L}: nbar={nbar:.1f} class-mean gap along dir={(np.linalg.norm(d) / nbar):.4f} nbar; "
              f"held-out projection AUC={av.auc(list(proj[gt[test]==1]), list(proj[gt[test]==0])):.3f}", flush=True)
        if handles[0] is not None:
            handles[0].remove()
        handles[0] = blocks[L].register_forward_hook(hook)
        if args.mode == "add":
            conds = [("none", 0.0)] + [(k, a) for k in dirs for a in args.alphas]
        else:
            conds = [("none", 0.0)] + [(k, 0.0) for k in dirs]
        for dname, a in conds:
            dev = next(model._model.parameters()).device
            state["vec"] = state["unit"] = None
            if dname != "none" and args.mode == "add":
                state["vec"] = torch.tensor(a * nbar * dirs[dname], device=dev)
            elif dname != "none":
                state["unit"] = torch.tensor(dirs[dname], device=dev)
            ms = []
            for i in test:
                p = paths[i]
                r_ = meta.loc[p]
                names, _ = eg.naming(r_)
                eg.CAP["first"] = None
                resp = model.ask(Path(p), eg.P_ORDER.format(**names))
                lp = torch.log_softmax(eg.CAP["first"], -1)
                mg = float(torch.logsumexp(lp[yes], 0) - torch.logsumexp(lp[no], 0))
                pred = eg.parse_yn(resp)
                w.writerow([L, dname, a, p, int(gt[i]), int(r_["interval_ms"]), round(mg, 4),
                            "" if pred is None else pred, resp[:60].replace("\n", " ")])
                ms.append((mg, int(gt[i]), pred))
            fh.flush()
            m = np.array([x[0] for x in ms]); g = np.array([x[1] for x in ms])
            py = np.mean([x[2] == 1 for x in ms])
            print(f"  L{L} {dname:6s} a={a:+.2f}: margin {m.mean():+.2f} (gt1 {m[g==1].mean():+.2f} / gt0 {m[g==0].mean():+.2f}) "
                  f"P(Y)={py:.3f} AUC={av.auc(list(m[g==1]), list(m[g==0])):.3f}", flush=True)
        state["vec"] = state["unit"] = None
    fh.close()
    print("STEER_DONE", flush=True)


if __name__ == "__main__":
    main()
