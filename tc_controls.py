"""TempCompass yes/no: margin AUC on real, static and blank clips, and the differences real-static and
real-blank with a bootstrap over VIDEOS (items cluster within videos). Blank = question text alone, i.e.
the text-plausibility prior; static = one frame. Usage: uv run --no-project --with pandas --with numpy python tc_controls.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_v3 as av

MODELS = ["molmo2", "qwen2.5-vl-72b", "qwen2.5-vl", "videochat-flash", "internvl2.5", "llava-next-video", "video-llama2"]
B = 2000


def load(m, cond):
    f = Path("results_long" if cond == "real" else "results_tc") / (f"{m}_tc_margin.csv" if cond == "real" else f"{m}_tc_margin_{cond}.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype={"video_id": str})
    return d.set_index(["video_id", "question"])[["gold", "margin", "pred"]]


def auc_of(d):
    d = d.dropna(subset=["margin"])
    return av.auc(list(d.margin[d.gold == 1]), list(d.margin[d.gold == 0]))


def main():
    rng = np.random.default_rng(0)
    out = {}
    for m in MODELS:
        R = load(m, "real")
        if R is None:
            continue
        t = out.setdefault(m, {"real": auc_of(R)})
        vids = R.index.get_level_values(0).unique().to_numpy()
        for c in ("static", "blank"):
            C = load(m, c)
            if C is None:
                continue
            j = R.join(C, rsuffix="_c", how="inner")
            t[c] = auc_of(C)
            diffs, lvl = [], []
            groups = {v: j.xs(v, level=0) for v in j.index.get_level_values(0).unique()}
            keys = np.array(list(groups))
            for _ in range(B):
                s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
                a = av.auc(list(s.margin[s.gold == 1].dropna()), list(s.margin[s.gold == 0].dropna()))
                b = av.auc(list(s.margin_c[s.gold == 1].dropna()), list(s.margin_c[s.gold == 0].dropna()))
                if a is not None and b is not None:
                    diffs.append(a - b); lvl.append(b)
            diffs, lvl = np.array(diffs), np.array(lvl)
            t[f"{c}_ci"] = [float(np.percentile(lvl, 2.5)), float(np.percentile(lvl, 97.5))]
            t[f"real_minus_{c}"] = float(t["real"] - t[c])
            t[f"real_minus_{c}_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        print(m, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in t.items()})
    Path("results_v3/tc_controls.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
