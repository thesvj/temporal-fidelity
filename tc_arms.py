"""TempCompass yes/no, Qwen2.5-VL-7B: margin AUC with the harness frames vs. the same frames one slot later (QWEN_ARM=shift).
Paired by item; intervals by video bootstrap. CPU only.

  uv run --no-project --with numpy python tc_arms.py results_tc_arms
"""
import csv
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "longpaper")
from within_class import auc  # noqa: E402


def load(f):
    out = {}
    for r in csv.DictReader(open(f)):
        try:
            m = float(r["margin"])
        except (TypeError, ValueError):
            continue
        out[(r["video_id"], r["question"])] = (m, 1 if str(r["gold"]).strip().lower() in ("1", "yes") else 0, r["dim"])
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results_tc_arms"
    H, S = load(f"{d}/qwen2.5-vl_tc_margin_harness.csv"), load(f"{d}/qwen2.5-vl_tc_margin_shift.csv")
    keys = sorted(set(H) & set(S))
    for dim in ("all", "order", "direction"):
        k = [x for x in keys if dim == "all" or H[x][2] == dim]
        y = np.array([H[x][1] for x in k]); h = np.array([H[x][0] for x in k]); s = np.array([S[x][0] for x in k])
        vids = defaultdict(list)
        for i, x in enumerate(k):
            vids[x[0]].append(i)
        V = list(vids); rng = np.random.default_rng(20261005); bs = []
        for _ in range(5000):
            idx = np.concatenate([vids[V[j]] for j in rng.integers(0, len(V), len(V))])
            bs.append(auc(s[idx], y[idx]) - auc(h[idx], y[idx]))
        print(f"{dim:9} n={len(k)} videos={len(V)} harness={auc(h, y):.3f} shift={auc(s, y):.3f} "
              f"diff={auc(s, y) - auc(h, y):+.3f} [{np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
