"""All data figures for the v3 stimuli, drawn from the per-trial CSVs and numbers_v3.json that
analyze_v3.py writes. Nothing here is typed in by hand.

Run from the repository root:
  python figs/make_figs_v3.py
Writes figs/fig1_protocol.pdf, fig_margin_summary.pdf, fig_interval_bacc.pdf, fig_confusion.pdf.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

RES = Path("results_v3")
OUT = Path("figs")
NUM = json.loads((RES / "numbers_v3.json").read_text())
MODELS = [("molmo2", "Molmo2"), ("videochat-flash", "VideoChat-Flash"), ("qwen2.5-vl", "Qwen2.5-VL-7B"),
          ("internvl2.5", "InternVL2.5-8B"), ("video-llama2", "Video-LLaMA2"),
          ("llava-next-video", "LLaVA-NeXT-Video")]
MODELS = [(k, n) for k, n in MODELS if (RES / f"{k}_expg3.csv").exists()]
COL = {"molmo2": "#2a78d6", "videochat-flash": "#eb6834", "qwen2.5-vl": "#4a3aa7",
       "internvl2.5": "#1baf7a", "video-llama2": "#c9a227", "llava-next-video": "#9a9994",
       "qwen2.5-vl-72b": "#b8336a", "internvl2.5-78b": "#0f6e6e"}
INK, MUTE, GRID = "#222222", "#8a8a8a", "#e6e6e6"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 7.5, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555", "axes.linewidth": .6,
                     "xtick.major.width": .6, "ytick.major.width": .6, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def rows(model):
    return list(csv.DictReader(open(RES / f"{model}_expg3.csv")))


def bacc(rs):
    p = [r for r in rs if r["gt"] == "1"]
    n = [r for r in rs if r["gt"] == "0"]
    if not p or not n:
        return np.nan
    return (sum(r["pred"] == "1" for r in p) / len(p) + 1 - sum(r["pred"] == "1" for r in n) / len(n)) / 2


def sig(m, key):
    t = NUM[m]
    p = t.get(f"{key}_p_holm", t.get(f"{key}_p", 1))
    return p is not None and p < .05


# ------------------------------------------------------------------ Figure 1
def fig1():
    fig = plt.figure(figsize=(7.0, 2.25))
    # (a) delivered frames, pairs, visibility --------------------------------
    ax = fig.add_axes([0.0, 0.02, 0.31, 0.86])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(a) What reached the model?", x=0.02)
    ax.text(.2, 9.25, "8 uniform frames, as our harness delivers them", fontsize=6, color=MUTE)
    fr = [2.4 + i * 1.0 for i in range(8)]

    def track(y, j1, j2, label, note, vis):
        ax.plot([2.0, 9.9], [y, y], color="#bbb", lw=.8)
        for k in range(4):                                   # fused pairs (Qwen2.5-VL)
            ax.add_patch(FancyBboxPatch((fr[2 * k] - .3, y + .45), 1.6, .95,
                                        boxstyle="round,pad=0,rounding_size=.18", fc="none",
                                        ec="#c9c9c9", lw=.6, ls=(0, (2, 1.5))))
        if vis:
            ax.add_patch(Rectangle((j1, y - .45), j2 - j1, .9, color="#1baf7a", alpha=.18, lw=0))
        for x in fr:
            ax.add_patch(Rectangle((x - .12, y + .7), .24, .45,
                                   color="#1baf7a" if j1 <= x < j2 else "#8a8a8a", lw=0))
        ax.plot([j1, j1], [y - .5, y + .5], color="#e34948", lw=1.6)
        ax.plot([j2, j2], [y - .5, y + .5], color="#2a78d6", lw=1.6)
        ax.text(.2, y + .1, label, va="center", fontsize=7)
        ax.text(2.0, y - 1.05, note, fontsize=6.1, color="#13845c" if vis else MUTE, va="center")

    track(6.55, 5.75, 6.05, "Δt = 0.1 s", "no delivered frame between the jumps", False)
    track(3.3, 4.85, 8.85, "Δt = 2 s", "order visible; first jump inside a fused pair", True)
    for i, (c, t) in enumerate([("#e34948", "first jump"), ("#2a78d6", "second jump"),
                                ("#8a8a8a", "delivered frame")]):
        ax.add_patch(Rectangle((.3 + i * 3.1, .55), .22, .5, color=c, lw=0))
        ax.text(.65 + i * 3.1, .8, t, fontsize=6, va="center")
    ax.add_patch(FancyBboxPatch((.3, -.6), .7, .5, clip_on=False, boxstyle="round,pad=0,rounding_size=.15", fc="none",
                                ec="#c9c9c9", lw=.6, ls=(0, (2, 1.5))))
    ax.text(1.2, -.35, "frames fused into one token (Qwen2.5-VL)", fontsize=6, va="center")

    # (b) margin histogram, InternVL2.5-8B move question ---------------------
    m = "internvl2.5"
    rs = rows(m)
    real = [float(r["margin"]) for r in rs if r["item"] == "moved_real" and r["margin"]]
    twin = [float(r["margin"]) for r in rs if r["item"] == "moved_twin" and r["margin"]]
    ax = fig.add_axes([0.355, 0.2, 0.255, 0.6])
    lo, hi = min(real + twin), max(real + twin)
    bins = np.linspace(np.floor(lo), max(1.0, np.ceil(hi)), 34)
    ax.hist(twin, bins=bins, color="#9a9994", alpha=.85, label=f"frozen twin ({len(twin)})")
    ax.hist(real, bins=bins, color="#2a78d6", alpha=.75, label=f"moving video ({len(real)})")
    ax.axvline(0, color=INK, ls="--", lw=.8)
    ax.text(.12, ax.get_ylim()[1] * .98, "answer is YES\nright of this line", fontsize=5.8, va="top")
    t = NUM[m]
    ax.text(.70, .50, f"AUC {t['auc_twin']:.3f}".replace("0.", ".", 1), transform=ax.transAxes,
            fontsize=10, fontweight="bold")
    ax.text(.70, .40, f"[{t['auc_twin_ci'][0]:.2f}, {t['auc_twin_ci'][1]:.2f}]".replace("0.", "."),
            transform=ax.transAxes, fontsize=6.5, color=MUTE)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("first-token margin, log P(YES) − log P(NO)", fontsize=6.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.text(0.345, 0.93, "(b) Answer vs. margin", fontsize=8.5, fontweight="bold")
    fig.text(0.345, 0.87, f"InternVL2.5-8B: YES on {100 * t['move_real']:.1f}% of moving videos",
             fontsize=6, color=MUTE)

    # (c) frame pairing: harness vs one-slot shift ------------------------------
    ax = fig.add_axes([0.715, 0.3, 0.285, 0.5])
    cls = [("sodd_eeven", "in/btw"), ("sodd_eodd", "in/in"), ("seven_eodd", "btw/in"), ("seven_eeven", "btw/btw")]
    for m, dx, name in [("qwen2.5-vl-72b", -.12, "72B"), ("qwen2.5-vl", .12, "7B")]:
        d = json.loads(Path(f"results_arms/{m}.json").read_text())["v3"]
        for j, (k, _) in enumerate(cls):
            a, b = d["harness"]["by_harness_four"][k]["auc"], d["shift"]["by_harness_four"][k]["auc"]
            ax.annotate("", xy=(j + dx, b), xytext=(j + dx, a),
                        arrowprops=dict(arrowstyle="-|>", color=COL[m], lw=.8, mutation_scale=6,
                                        shrinkA=2.5, shrinkB=1.5, alpha=.8))
            ax.plot(j + dx, a, "o", color=COL[m], ms=3.8, zorder=3)
            ax.plot(j + dx, b, "o", mfc="white", mec=COL[m], mew=.9, ms=3.8, zorder=3)
        ax.plot([], [], "o", color=COL[m], ms=3.8, label=f"Qwen2.5-VL-{name}")
    ax.plot([], [], "o", color=MUTE, ms=3.8, label="harness frames")
    ax.plot([], [], "o", mfc="white", mec=MUTE, ms=3.8, label="same frames, one slot later")
    ax.axhline(.5, color="#bbb", lw=.7)
    ax.set_xticks(range(4)); ax.set_xticklabels([c for _, c in cls], fontsize=6.3)
    ax.set_xlim(-.5, 3.5); ax.set_ylim(0, 1.04)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", ".25", ".5", ".75", "1"], fontsize=6.3)
    ax.set_ylabel("order-margin AUC", fontsize=6.5)
    ax.set_xlabel("first / second jump: inside a pair or between pairs", fontsize=6, labelpad=2)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center", bbox_to_anchor=(.45, -.27), ncol=2,
              handletextpad=.2, labelspacing=.2, columnspacing=.8)
    ax.grid(axis="y", color=GRID, lw=.5)
    fig.text(0.68, 0.93, "(c) Shifting the frames by one slot", fontsize=8.5, fontweight="bold")
    fig.text(0.68, 0.87, "Qwen2.5-VL, shape videos, visible trials", fontsize=6, color=MUTE)
    fig.savefig(OUT / "fig1_protocol.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ margin summary
def margin_summary():
    ms = sorted([m for m in MODELS if "auc_vis" in NUM.get(m[0], {})], key=lambda m: -NUM[m[0]]["auc_vis"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"wspace": .55})
    ax = axs[0]
    for i, (m, name) in enumerate(ms):
        t = NUM[m]
        y = len(ms) - 1 - i
        s = sig(m, "auc_vis")
        c = "#4a3aa7" if s else "#b9b3dd"
        ax.plot([t["bacc_v"], t["auc_vis"]], [y, y], color="#d0d0d0", lw=1.6, zorder=1)
        ax.errorbar(t["auc_vis"], y, xerr=[[t["auc_vis"] - t["auc_vis_ci"][0]], [t["auc_vis_ci"][1] - t["auc_vis"]]],
                    fmt="o", color=c, ms=4, lw=.8, capsize=0, zorder=3)
        ax.plot(t["bacc_v"], y, "o", color="#8a8a8a", ms=3.5, zorder=2)
        if not s:
            ax.text(t["auc_vis_ci"][1] + .012, y, "n.s.", fontsize=6, va="center", color=MUTE)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([n for _, n in ms][::-1])
    ax.axvline(.5, color="#bbb", lw=.7)
    ax.set_xlim(.4, 1.02)
    ax.set_xlabel("visible trials: balanced accuracy (gray) / margin AUC (purple)")
    ax.set_title("(a) Temporal order: answers vs. answer logits")
    ax.grid(axis="x", color=GRID, lw=.5)
    ax = axs[1]
    for i, (m, name) in enumerate(ms):
        t = NUM[m]
        y = len(ms) - 1 - i
        ax.barh(y + .17, t["move_real"], height=.32, color="#2a78d6")
        ax.barh(y - .17, t["move_twin"], height=.32, color="#9a9994")
        lab = f"twin {t['auc_twin']:.2f}  solo {t['auc_solo']:.2f}".replace("0.", ".")
        ax.text(1.03, y, lab, fontsize=6.3, va="center",
                color="#4a3aa7" if sig(m, "auc_twin") else MUTE, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([n for _, n in ms][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(YES) to “did the shape move?”: real (blue) / frozen twin (gray)")
    ax.set_title("(b) “Did the shape move?”")
    ax.text(1.03, len(ms) - .35, "margin AUC", fontsize=6.3, color="#4a3aa7", transform=ax.get_yaxis_transform())
    ax.grid(axis="x", color=GRID, lw=.5)
    fig.savefig(OUT / "fig_margin_summary.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ loss budget + matched criterion
def budget():
    """(a) Order evidence on one scale (BAcc, visible trials): hidden state (probe) -> margin (held-out
    threshold) -> answers. (b) BAcc of each margin thresholded at every rate of YES, with the model's own
    operating point: VideoChat-Flash and Qwen2.5-VL-72B have the same curve and different points."""
    import sys
    sys.path.insert(0, ".")
    import analyze_v3 as av
    prof = json.loads((RES / "profile_v3.json").read_text())
    B = prof["budget"]
    PY = {m: t["py"] for m, t in prof["answers"].items()}
    order = ["molmo2", "qwen2.5-vl-72b", "videochat-flash", "qwen2.5-vl", "internvl2.5", "internvl2.5-78b",
             "video-llama2", "llava-next-video"]
    order = [m for m in order if m in B]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"wspace": .5, "width_ratios": [1.05, 1]})
    ax = axs[0]
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        t = B[m]
        xs = [v for v in (t["state"], t["margin"], t["answer"]) if v is not None]
        ax.plot([min(xs), max(xs)], [y, y], color="#d0d0d0", lw=1.6, zorder=1)
        if t["state"] is not None:
            ax.plot(t["state"], y, "s", color=COL[m], ms=4, mfc="white", mew=1, zorder=3)
        ax.plot(t["margin"], y, "o", color=COL[m], ms=4.2, zorder=3)
        ax.plot(t["answer"], y, "o", color="#8a8a8a", ms=3.4, zorder=2)
    ax.axvline(.6, color=INK, ls=":", lw=.8)
    ax.axvline(.5, color="#bbb", lw=.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([av.NICE[m] for m in reversed(order)], fontsize=6.8)
    ax.set_xlim(.47, 1.0)
    ax.set_xlabel("balanced accuracy, visible trials", fontsize=6.8)
    ax.grid(axis="x", color=GRID, lw=.5)
    ax.plot([], [], "s", color=INK, mfc="white", ms=4, label="hidden state (probe accuracy)")
    ax.plot([], [], "o", color=INK, ms=4, label="margin, held-out threshold")
    ax.plot([], [], "o", color="#8a8a8a", ms=3.4, label="answers")
    ax.legend(fontsize=5.8, frameon=False, loc="lower right", handletextpad=.3, borderaxespad=.1)
    ax.set_title("(a) Where the order is lost")

    ax = axs[1]
    meta = av.load_meta(str(RES / "videos_v3_metadata.csv"))
    av.load_recorded(str(RES))
    qs = np.geomspace(.004, .8, 120)
    for m in ["qwen2.5-vl-72b", "videochat-flash", "qwen2.5-vl"]:
        rs = [r for r in av.by_item(av.load_rows(str(RES), m, meta), "order_real")
              if r["gt"] in ("0", "1") and av.visible(r["_meta"], m) and r["_m"] is not None]
        mg = np.array([r["_m"] for r in rs]); g = np.array([int(r["gt"]) for r in rs])
        curve = []
        for q in qs:
            pr = mg > np.quantile(mg, 1 - q)
            curve.append((pr[g == 1].mean() + 1 - pr[g == 0].mean()) / 2)
        ax.plot(qs, curve, color=COL[m], lw=1.1)
        ax.plot(PY[m], B[m]["answer"], "o", color=COL[m], ms=5, mec="white", mew=.7, zorder=4)
        ax.plot([], [], color=COL[m], lw=1.1, label=av.NICE[m])
    ax.plot([], [], "o", color=MUTE, ms=4.5, mec="white", label="the model's own answers")
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", handlelength=1.4, handletextpad=.4,
              labelspacing=.3, borderaxespad=.1)
    ax.plot(qs, .5 + qs, color="#8a8a8a", ls="--", lw=.8)
    ax.text(.29, .735, "ceiling .5 + P(YES)", fontsize=5.8, color="#8a8a8a", ha="left", va="center")
    ax.axhline(.6, color=INK, ls=":", lw=.8)
    ax.axhline(.5, color="#bbb", lw=.7)
    ax.set_xscale("log")
    ax.set_xticks([.01, .03, .1, .3]); ax.set_xticklabels(["1%", "3%", "10%", "30%"])
    ax.set_ylim(.48, .75)
    ax.set_xlabel("rate of YES set by the threshold on the margin", fontsize=6.8)
    ax.set_ylabel("balanced accuracy", fontsize=6.8)
    ax.grid(color=GRID, lw=.5)
    ax.set_title("(b) Same margin, different rate of YES")
    fig.savefig(OUT / "fig_budget.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ BAcc by interval
def interval():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    big = [("qwen2.5-vl-72b", "Qwen2.5-VL-72B")] if (RES / "qwen2.5-vl-72b_expg3.csv").exists() else []
    for m, name in MODELS + big:
        by = defaultdict(list)
        for r in rows(m):
            if r["item"] == "order_real" and r["pred"] in ("0", "1"):
                by[int(r["interval_ms"])].append(r)
        xs = sorted(by)
        ax.plot(xs, [bacc(by[x]) for x in xs], marker="o", ms=2.3, lw=1.1, color=COL[m], label=name)
    ax.axhline(.5, color="#bbb", lw=.6)
    ax.set_xscale("log")
    ax.set_xticks([100, 300, 1000, 3000, 10000]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    ax.set_xlabel("interval between jumps (s)"); ax.set_ylabel("balanced accuracy")
    ax.set_ylim(.3, 1.02)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2)
    fig.savefig(OUT / "fig_interval_bacc.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ interval confusion
def confusion():
    bins = "ABCD"
    names = ["<1", "1–2", "2–5", "≥5"]
    fig, axs = plt.subplots(1, len(MODELS), figsize=(7.0, 1.6), gridspec_kw={"wspace": .12})
    for ax, (m, name) in zip(axs, MODELS):
        M = np.zeros((4, 4))
        for r in rows(m):
            if r["item"] == "interval" and r["gt"] in bins and r["pred"] in bins:
                M[bins.index(r["gt"]), bins.index(r["pred"])] += 1
        M = M / np.maximum(M.sum(1, keepdims=True), 1)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        acc = np.mean([M[i, i] for i in range(4)])
        ax.set_title(f"{name}\n(mean recall {acc:.2f})".replace("0.", "."), fontsize=6.3, loc="center",
                     fontweight="normal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(names, fontsize=5.6, rotation=0); ax.set_yticklabels(names if ax is axs[0] else [], fontsize=5.6)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("true interval (s)", fontsize=6)
    fig.text(.5, -.02, "answered interval (s)", ha="center", fontsize=6)
    fig.savefig(OUT / "fig_confusion.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); budget(); interval(); confusion()
    print("wrote", [str(p) for p in sorted(OUT.glob("fig*_*.pdf")) if p.stat().st_mtime > 0][:0] or "4 figures")
