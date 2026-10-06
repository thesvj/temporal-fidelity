"""The YES-rate cap on balanced accuracy, on the three stimulus sets. CPU only.

With labels of prevalence pi, a model that says YES on a fraction q of trials cannot score above
  cap(q, pi) = .5 + q / (2 pi)            if q <= pi      (every YES a hit)
             = .5 + (1 - q) / (2 (1 - pi))  otherwise      (every NO a correct rejection)
whatever it perceives; with pi = .5 this is .5 + min(q, 1 - q). The table gives, per model and set, the rate
of YES, the cap, the balanced accuracy of the answers and how much of the room below the cap they use,
(BAcc - .5) / (cap - .5). Sets: shape videos and composited real footage (order question, visible trials)
and TempCompass yes/no (all 876 items).

Writes results_v3/cap_v3.json and results_v3/cap_rows.tex.
Usage: uv run --no-project --with numpy python cap_v3.py
"""
import csv
import json
from pathlib import Path

import analyze_v3 as av
import analyze_real as ar


def cap(q, pi):
    return .5 + q / (2 * pi) if q <= pi else .5 + (1 - q) / (2 * (1 - pi))


def row(rows):
    rr = [r for r in rows if r["pred"] in ("0", "1")]
    q = sum(r["pred"] == "1" for r in rr) / len(rr)
    pi = sum(r["gt"] == "1" for r in rr) / len(rr)
    b = av.bacc(rr); c = cap(q, pi)
    return dict(n=len(rr), py=q, pi=pi, cap=c, bacc=b, used=(b - .5) / (c - .5) if c > .5 + 1e-9 else None)


def controlled(res, model):
    av.RECORDED.clear(); av.load_recorded(res)
    meta = av.load_meta(f"{res}/videos_v3_metadata.csv")
    rows = av.load_rows(res, model, meta)
    return row([r for r in ar.order_rows(rows) if av.visible(r["_meta"], model)]) if rows else None


def main():
    out = {}
    for m in av.MODELS:
        d = {}
        if (s := controlled("results_v3", m)):
            d["shapes"] = s
        if m in ar.SMALL and Path(f"results_real/{m}_expg3.csv").exists():
            d["footage"] = controlled("results_real", m)
        f = Path(f"results_long/{m}_tc_margin.csv")
        if f.exists():
            d["tempcompass"] = row([dict(gt=r["gold"], pred=r["pred"]) for r in csv.DictReader(open(f)) if r["gold"] in ("0", "1")])
        out[m] = d
    Path("results_v3/cap_v3.json").write_text(json.dumps(out, indent=1))
    f3 = lambda x: "--" if x is None else f"{x:.3f}".replace("0.", ".", 1) if x < 1 else "1.00"
    lines = []
    print(f"{'model':18} " + " | ".join(f"{s:^31}" for s in ("shapes", "footage", "tempcompass")))
    for m, d in out.items():
        cells, tex = [], []
        for s in ("shapes", "footage", "tempcompass"):
            c = d.get(s)
            cells.append(f"q {c['py']:.3f} cap {c['cap']:.3f} B {c['bacc']:.3f} u {(c['used'] if c['used'] is not None else float('nan')):.2f}" if c else " " * 31)
            tex.append(f"{f3(c['py'])} & {f3(c['cap'])} & {f3(c['bacc'])}" if c else "-- & -- & --")
        print(f"{av.NICE[m]:18} " + " | ".join(cells))
        lines.append(f"{av.NICE[m]:17} & " + " & ".join(tex) + " \\\\")
    Path("results_v3/cap_rows.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
