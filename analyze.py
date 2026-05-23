"""
Publication-quality analysis for the Temporal Fidelity paper.

Produces LaTeX tables (results/) and PDF figures (figs/) with bias-corrected
metrics (d-prime, balanced accuracy), statistical tests (McNemar, Spearman,
Cohen's kappa), and bootstrap confidence intervals.

Usage:
  uv run python analyze.py                    # everything
  uv run python analyze.py --tables-only      # LaTeX tables only
  uv run python analyze.py --figures-only     # figures only
  uv run python analyze.py --format png       # PNG instead of PDF
  uv run python analyze.py --no-bootstrap     # skip bootstrap CIs (faster)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERVALS_MS = [33, 67, 100, 200, 333, 500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]
TICK_MS = [33, 100, 500, 1000, 5000, 10000]
SIM_OFFSETS_MS = [0, 33, 67, 100, 200, 333]
CHANCE = {"order": 0.5, "interval": 0.25, "simultaneity": 0.5}

MODEL_ORDER = [
    "molmo2", "videochat-flash", "qwen2.5-vl",
    "internvl2.5", "llava-next-video", "video-llama2", "videollama3",
]

MODEL_DISPLAY = {
    "llava-next-video": "LLaVA-NeXT-Video",
    "video-llama2":     "Video-LLaMA2",
    "qwen2.5-vl":       "Qwen2.5-VL",
    "molmo2":           "Molmo2",
    "internvl2.5":      "InternVL2.5",
    "videollama3":      "VideoLLaMA3",
    "videochat-flash":  "VideoChat-Flash",
}

MODEL_ENCODER = {
    "llava-next-video": "CLIP ViT-L",
    "video-llama2":     "CLIP ViT-L",
    "qwen2.5-vl":       "ViT (mRoPE)",
    "molmo2":           "SigLIP2",
    "internvl2.5":      "InternViT-300M",
    "videollama3":      "SigLIP (DiffFP)",
    "videochat-flash":  "InternVideo2",
}

MODEL_LLM = {
    "llava-next-video": "Mistral-7B",
    "video-llama2":     "Mistral-7B",
    "qwen2.5-vl":       "Qwen2.5-7B",
    "molmo2":           "Qwen3-8B",
    "internvl2.5":      "InternLM2-8B",
    "videollama3":      "Qwen2.5-7B",
    "videochat-flash":  "Qwen2-7B",
}

MODEL_COLORS = {
    "molmo2":           "#9C27B0",
    "videochat-flash":  "#795548",
    "qwen2.5-vl":       "#4CAF50",
    "internvl2.5":      "#FF9800",
    "llava-next-video": "#2196F3",
    "video-llama2":     "#FF5722",
    "videollama3":      "#00BCD4",
}

MODEL_MARKERS = {
    "molmo2": "o", "videochat-flash": "s", "qwen2.5-vl": "D",
    "internvl2.5": "^", "llava-next-video": "v", "video-llama2": "P",
    "videollama3": "X",
}

FIG_EXT = "pdf"

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_task(results_dir: Path, task: str) -> list[tuple[str, pd.DataFrame]]:
    seen: set[str] = set()
    out: list[tuple[str, pd.DataFrame]] = []

    def _ingest(csv: Path, strip_suffix: str) -> None:
        if csv.stem.startswith(("summary_", "thresholds_", "table")):
            return
        model = csv.stem[: -len(strip_suffix)]
        if model in seen:
            return
        seen.add(model)
        df = pd.read_csv(csv)
        if "correct" in df.columns:
            df = df.dropna(subset=["correct"])
        out.append((model, df))

    for csv in sorted(results_dir.glob(f"*_{task}.csv")):
        _ingest(csv, f"_{task}")
    for csv in sorted(results_dir.glob(f"*_{task}_fps30.csv")):
        _ingest(csv, f"_{task}_fps30")

    return [(m, d) for m, d in out if m in MODEL_ORDER]


def _ordered(pairs: list[tuple[str, pd.DataFrame]]) -> list[tuple[str, pd.DataFrame]]:
    lookup = {m: d for m, d in pairs}
    return [(m, lookup[m]) for m in MODEL_ORDER if m in lookup]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _save(fig, path: Path, *, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> {path}")


def _display(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


# ---------------------------------------------------------------------------
# Signal Detection Theory
# ---------------------------------------------------------------------------

def compute_sdt(df: pd.DataFrame) -> dict:
    """d-prime and criterion with Hautus (1995) loglinear correction."""
    valid = df.dropna(subset=["pred", "gt"])
    if len(valid) == 0:
        return {"d_prime": np.nan, "criterion": np.nan, "hit_rate": np.nan,
                "fa_rate": np.nan, "balanced_acc": np.nan, "accuracy": np.nan,
                "p_yes": np.nan, "n": 0}

    signal = valid[valid["gt"] == 1]
    noise = valid[valid["gt"] == 0]

    hits = (signal["pred"] == 1).sum()
    fa = (noise["pred"] == 1).sum()
    n_signal = len(signal)
    n_noise = len(noise)

    hr = (hits + 0.5) / (n_signal + 1)
    far = (fa + 0.5) / (n_noise + 1)

    d_prime = stats.norm.ppf(hr) - stats.norm.ppf(far)
    criterion = -0.5 * (stats.norm.ppf(hr) + stats.norm.ppf(far))

    tpr = hits / max(n_signal, 1)
    tnr = (n_noise - fa) / max(n_noise, 1)

    return {
        "d_prime": d_prime, "criterion": criterion,
        "hit_rate": tpr, "fa_rate": fa / max(n_noise, 1),
        "balanced_acc": (tpr + tnr) / 2,
        "accuracy": valid["correct"].mean() if "correct" in valid.columns else np.nan,
        "p_yes": (valid["pred"] == 1).mean(),
        "n": len(valid),
    }


def compute_balanced_acc_curve(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["interval_ms", "bacc", "d_prime", "ci_lo", "ci_hi"])
    rows = []
    for iv, g in df.groupby("interval_ms"):
        sdt = compute_sdt(g)
        n = len(g)
        k = int(sdt["balanced_acc"] * n) if not np.isnan(sdt["balanced_acc"]) else 0
        lo, hi = _wilson_ci(k, n)
        rows.append({"interval_ms": iv, "bacc": sdt["balanced_acc"],
                      "d_prime": sdt["d_prime"], "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def bootstrap_dprime(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    valid = df.dropna(subset=["pred", "gt"])
    if len(valid) < 10:
        return (np.nan, np.nan)
    d_primes = []
    for _ in range(n_boot):
        sample = valid.sample(n=len(valid), replace=True, random_state=rng)
        rng = np.random.default_rng(rng.integers(0, 2**31))
        m = compute_sdt(sample)
        d_primes.append(m["d_prime"])
    d_primes = sorted(d_primes)
    return d_primes[int(0.025 * n_boot)], d_primes[int(0.975 * n_boot)]


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def spearman_with_ci(x, y, n_boot=1000, seed=42):
    rho, p = stats.spearmanr(x, y)
    if n_boot <= 0:
        return rho, p, np.nan, np.nan
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        r, _ = stats.spearmanr(np.array(x)[idx], np.array(y)[idx])
        rhos.append(r)
    rhos = sorted(rhos)
    lo = rhos[int(0.025 * n_boot)]
    hi = rhos[int(0.975 * n_boot)]
    return rho, p, lo, hi


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(stat, df=1)
    return stat, p


def response_entropy(pred_series: pd.Series) -> float:
    counts = pred_series.value_counts(normalize=True)
    return stats.entropy(counts.values, base=2)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def _to_latex(df: pd.DataFrame, path: Path, caption: str, label: str,
              col_fmt: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if col_fmt is None:
        col_fmt = "l" + "c" * len(df.columns)
    body = df.to_latex(index=True, escape=False, column_format=col_fmt, na_rep="--")
    wrapped = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )
    path.write_text(wrapped)
    print(f"  LaTeX -> {path}")


def _fmt(v, fmt=".3f"):
    if pd.isna(v):
        return "--"
    return f"{v:{fmt}}"


def _bold_max(series: pd.Series, fmt=".3f") -> pd.Series:
    if series.isna().all():
        return series.apply(lambda _: "--")
    max_val = series.max()
    return series.apply(lambda v: f"\\textbf{{{v:{fmt}}}}" if v == max_val and not pd.isna(v)
                        else _fmt(v, fmt))


def _pval(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<$.001"
    return f"{p:.3f}"


# ===================================================================
# TABLE 1: Main Results (Order — E001)
# ===================================================================

def table_main_results(results_dir: Path, out_dir: Path, n_boot: int):
    pairs = _ordered(_load_task(results_dir, "order"))
    rows = []
    for model, df in pairs:
        sdt = compute_sdt(df)
        acc_by_iv = df.groupby("interval_ms")["correct"].mean()
        rho, p, rlo, rhi = spearman_with_ci(np.log(acc_by_iv.index), acc_by_iv.values, n_boot)
        if np.isnan(sdt["d_prime"]):
            dp_str = "--"
        elif n_boot > 0:
            dp_lo, dp_hi = bootstrap_dprime(df, n_boot)
            dp_str = f"{sdt['d_prime']:.2f} [{dp_lo:.2f}, {dp_hi:.2f}]"
        else:
            dp_str = f"{sdt['d_prime']:.2f}"
        rows.append({
            "Model": _display(model),
            "Encoder": MODEL_ENCODER.get(model, ""),
            "LLM": MODEL_LLM.get(model, ""),
            "$P(\\text{YES})$": _fmt(sdt["p_yes"]),
            "Acc": _fmt(sdt["accuracy"]),
            "BAcc": _fmt(sdt["balanced_acc"]),
            "$d'$": dp_str,
            "$c$": _fmt(sdt["criterion"], ".2f"),
            "$\\rho$": _fmt(rho, ".2f"),
            "$p$": _pval(p),
        })

    # Add videollama3 if missing
    seen = {r["Model"] for r in rows}
    if _display("videollama3") not in seen:
        rows.append({
            "Model": _display("videollama3"),
            "Encoder": MODEL_ENCODER["videollama3"],
            "LLM": MODEL_LLM["videollama3"],
            "$P(\\text{YES})$": "--", "Acc": "--", "BAcc": "--",
            "$d'$": "--", "$c$": "--", "$\\rho$": "--", "$p$": "--",
        })

    tbl = pd.DataFrame(rows).set_index("Model")
    _to_latex(tbl, out_dir / "table1_main_results.tex",
              caption="Temporal order discrimination (E001). $d'$: sensitivity with 95\\% bootstrap CI. "
                      "$c$: response criterion (positive = conservative/NO bias). "
                      "$\\rho$: Spearman correlation between log-interval and accuracy.",
              label="tab:main")

    print("\n=== TABLE 1: MAIN RESULTS (ORDER) ===")
    print(tbl.to_string())
    return {r["Model"]: r for r in rows}


# ===================================================================
# TABLE 2: Interval Estimation (E002)
# ===================================================================

def table_interval(results_dir: Path, out_dir: Path):
    pairs = _ordered(_load_task(results_dir, "interval"))
    rows = []
    for model, df in pairs:
        valid = df.dropna(subset=["pred"])
        if len(valid) == 0:
            continue
        dist = valid["pred"].value_counts(normalize=True)
        acc = valid["correct"].mean() if "correct" in valid.columns else np.nan
        bacc = balanced_accuracy_score(valid["gt"], valid["pred"]) if len(valid) > 0 else np.nan
        try:
            kappa = cohen_kappa_score(valid["gt"], valid["pred"],
                                      labels=["A", "B", "C", "D"], weights="quadratic")
        except Exception:
            kappa = np.nan
        h = response_entropy(valid["pred"])
        rows.append({
            "Model": _display(model),
            "$P(A)$": _fmt(dist.get("A", 0)),
            "$P(B)$": _fmt(dist.get("B", 0)),
            "$P(C)$": _fmt(dist.get("C", 0)),
            "$P(D)$": _fmt(dist.get("D", 0)),
            "Acc": _fmt(acc),
            "BAcc": _fmt(bacc),
            "$\\kappa_w$": _fmt(kappa, ".2f"),
            "$H$": _fmt(h, ".2f"),
        })

    tbl = pd.DataFrame(rows).set_index("Model")
    _to_latex(tbl, out_dir / "table2_interval.tex",
              caption="Interval estimation (E002). $\\kappa_w$: quadratic-weighted Cohen's kappa. "
                      "$H$: response entropy in bits (max 2.0 for 4 classes).",
              label="tab:interval")

    print("\n=== TABLE 2: INTERVAL ESTIMATION ===")
    print(tbl.to_string())


# ===================================================================
# TABLE 3: Representation-Behavior Gap
# ===================================================================

def table_rep_behavior_gap(results_dir: Path, out_dir: Path):
    pairs_order = _ordered(_load_task(results_dir, "order"))
    rows = []
    for model, df in pairs_order:
        sdt = compute_sdt(df)
        bacc = sdt["balanced_acc"]

        enc_acc = np.nan
        llm_acc = np.nan

        enc_path = results_dir / f"{model}_encoder_layers.csv"
        if enc_path.exists():
            enc_df = pd.read_csv(enc_path)
            enc_acc = enc_df["accuracy"].max()

        llm_path = results_dir / f"{model}_llm_layers.csv"
        if llm_path.exists():
            llm_df = pd.read_csv(llm_path)
            llm_acc = llm_df["accuracy"].max()

        enc_gap = enc_acc - bacc if not (np.isnan(enc_acc) or np.isnan(bacc)) else np.nan
        llm_gap = llm_acc - bacc if not (np.isnan(llm_acc) or np.isnan(bacc)) else np.nan

        rows.append({
            "Model": _display(model),
            "Behav. BAcc": _fmt(bacc),
            "Probe (Enc)": _fmt(enc_acc),
            "Probe (LLM)": _fmt(llm_acc),
            "$\\Delta$ Enc": _fmt(enc_gap, "+.3f") if not np.isnan(enc_gap) else "--",
            "$\\Delta$ LLM": _fmt(llm_gap, "+.3f") if not np.isnan(llm_gap) else "--",
        })

    tbl = pd.DataFrame(rows).set_index("Model")
    _to_latex(tbl, out_dir / "table3_rep_behavior_gap.tex",
              caption="Representation--behavior gap. Probe accuracy (max across layers, E005) vs.\\ "
                      "behavioral balanced accuracy (E001). $\\Delta$: gap (positive = information "
                      "present in representations but unused behaviorally).",
              label="tab:gap")

    print("\n=== TABLE 3: REPRESENTATION-BEHAVIOR GAP ===")
    print(tbl.to_string())


# ===================================================================
# TABLE 4: Frame Rate Effect (E004)
# ===================================================================

def table_fps(results_dir: Path, out_dir: Path):
    rows = []
    for model in MODEL_ORDER:
        fps_data = {}
        for fps in [8, 16, 30]:
            path = results_dir / f"{model}_order_fps{fps}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).dropna(subset=["correct"])
            if len(df) == 0:
                continue
            sdt = compute_sdt(df)
            fps_data[fps] = {"df": df, "sdt": sdt}

        if not fps_data:
            continue

        row = {"Model": _display(model)}
        for fps in [8, 16, 30]:
            if fps in fps_data:
                row[f"BAcc@{fps}"] = _fmt(fps_data[fps]["sdt"]["balanced_acc"])
                row[f"$d'$@{fps}"] = _fmt(fps_data[fps]["sdt"]["d_prime"], ".2f")
            else:
                row[f"BAcc@{fps}"] = "--"
                row[f"$d'$@{fps}"] = "--"

        if 8 in fps_data and 30 in fps_data:
            df8 = fps_data[8]["df"].sort_values(["interval_ms", "path"]).reset_index(drop=True)
            df30 = fps_data[30]["df"].sort_values(["interval_ms", "path"]).reset_index(drop=True)
            n = min(len(df8), len(df30))
            _, p = mcnemar_test(df8["correct"].values[:n], df30["correct"].values[:n])
            row["McNemar $p$"] = _pval(p)
        else:
            row["McNemar $p$"] = "--"

        rows.append(row)

    tbl = pd.DataFrame(rows).set_index("Model")
    _to_latex(tbl, out_dir / "table4_fps.tex",
              caption="Frame rate control (E004). Balanced accuracy and $d'$ for temporal order "
                      "at 8, 16, and 30\\,fps. McNemar's test compares 8 vs.\\ 30\\,fps.",
              label="tab:fps")

    print("\n=== TABLE 4: FRAME RATE ===")
    print(tbl.to_string())


# ===================================================================
# TABLE 5: Simultaneity (E003)
# ===================================================================

def table_simultaneity(results_dir: Path, out_dir: Path):
    pairs = _ordered(_load_task(results_dir, "simultaneity"))
    rows = []
    for model, df in pairs:
        sdt = compute_sdt(df)
        p_yes = sdt["p_yes"]
        if pd.isna(p_yes):
            strat = "--"
        elif p_yes < 0.05:
            strat = "Always NO"
        elif p_yes > 0.95:
            strat = "Always YES"
        elif p_yes < 0.2 or p_yes > 0.8:
            strat = "Strong bias"
        else:
            strat = "Mixed"

        rows.append({
            "Model": _display(model),
            "$P(\\text{YES})$": _fmt(sdt["p_yes"]),
            "Acc": _fmt(sdt["accuracy"]),
            "BAcc": _fmt(sdt["balanced_acc"]),
            "$d'$": _fmt(sdt["d_prime"], ".2f"),
            "Strategy": strat,
        })

    tbl = pd.DataFrame(rows).set_index("Model")
    _to_latex(tbl, out_dir / "table5_simultaneity.tex",
              caption="Simultaneity detection (E003). Strategy describes the dominant response pattern.",
              label="tab:simult")

    print("\n=== TABLE 5: SIMULTANEITY ===")
    print(tbl.to_string())


# ===================================================================
# FIGURE 1: Bias-Corrected Order Curves
# ===================================================================

def fig_order_corrected(results_dir: Path, figs_dir: Path):
    pairs = _ordered(_load_task(results_dir, "order"))
    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance", zorder=0)

    for model, df in pairs:
        if len(df) == 0:
            continue
        curve = compute_balanced_acc_curve(df)
        if len(curve) == 0:
            continue
        sdt = compute_sdt(df)
        color = MODEL_COLORS[model]
        label = f"{_display(model)} ($d'$={sdt['d_prime']:.1f})"
        ax.plot(curve["interval_ms"], curve["bacc"], marker=MODEL_MARKERS[model],
                markersize=4, label=label, color=color, lw=1.2)
        ax.fill_between(curve["interval_ms"], curve["ci_lo"], curve["ci_hi"],
                        alpha=0.10, color=color)

    ax.set_xscale("log")
    ax.set_xticks(TICK_MS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Inter-event interval (ms)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.set_title("Temporal order discrimination (bias-corrected)")
    _save(fig, figs_dir / f"fig1_order_corrected.{FIG_EXT}")


# ===================================================================
# FIGURE 2: SDT Summary (ROC scatter + d' bars)
# ===================================================================

def fig_sdt_summary(results_dir: Path, figs_dir: Path, n_boot: int):
    pairs = _ordered(_load_task(results_dir, "order"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8))

    # Panel (a): ROC-style scatter with iso-d' curves
    for dp_val in [0, 1, 2, 3, 4]:
        far_range = np.linspace(0.001, 0.999, 200)
        hr_range = stats.norm.cdf(dp_val + stats.norm.ppf(far_range))
        ax1.plot(far_range, hr_range, color="lightgray", lw=0.6, ls="--")
        x_pos = 0.02 if dp_val > 0 else 0.4
        y_pos = stats.norm.cdf(dp_val + stats.norm.ppf(x_pos))
        if 0 < y_pos < 1:
            ax1.text(x_pos, min(y_pos + 0.02, 0.98), f"$d'$={dp_val}",
                     fontsize=5, color="gray")

    for model, df in pairs:
        if len(df) == 0:
            continue
        sdt = compute_sdt(df)
        if np.isnan(sdt["fa_rate"]):
            continue
        ax1.scatter(sdt["fa_rate"], sdt["hit_rate"], color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model], s=50, zorder=5,
                    label=_display(model), edgecolors="white", linewidth=0.5)

    ax1.plot([0, 1], [0, 1], color="gray", ls=":", lw=0.5)
    ax1.set_xlabel("False alarm rate")
    ax1.set_ylabel("Hit rate")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect("equal")
    ax1.legend(fontsize=5, loc="lower right")
    ax1.set_title("(a) ROC space", fontsize=8)

    # Panel (b): d' bar chart
    models_sorted = []
    for model, df in pairs:
        if len(df) == 0:
            continue
        sdt = compute_sdt(df)
        ci = bootstrap_dprime(df, n_boot) if n_boot > 0 else (np.nan, np.nan)
        models_sorted.append((model, sdt["d_prime"], ci))

    models_sorted.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    y_pos = np.arange(len(models_sorted))
    for i, (model, dp, ci) in enumerate(models_sorted):
        xerr_lo = dp - ci[0] if not np.isnan(ci[0]) else 0
        xerr_hi = ci[1] - dp if not np.isnan(ci[1]) else 0
        ax2.barh(i, dp, color=MODEL_COLORS[model], height=0.6,
                 xerr=[[xerr_lo], [xerr_hi]], capsize=3, ecolor="gray")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([_display(m) for m, _, _ in models_sorted], fontsize=6)
    ax2.axvline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("$d'$ (sensitivity)")
    ax2.set_title("(b) Temporal sensitivity", fontsize=8)
    ax2.invert_yaxis()

    fig.tight_layout(w_pad=3)
    _save(fig, figs_dir / f"fig2_sdt_summary.{FIG_EXT}")


# ===================================================================
# FIGURE 3: Representation-Behavior Gap
# ===================================================================

def fig_gap(results_dir: Path, figs_dir: Path):
    pairs = _ordered(_load_task(results_dir, "order"))
    pair_lookup = {m: df for m, df in pairs}
    models, bacc_vals, enc_vals, llm_vals = [], [], [], []
    shuf_enc_vals, shuf_llm_vals = [], []

    for model in MODEL_ORDER:
        enc_path = results_dir / f"{model}_encoder_layers.csv"
        llm_path = results_dir / f"{model}_llm_layers.csv"
        if not enc_path.exists() and not llm_path.exists():
            continue

        if model in pair_lookup:
            df = pair_lookup[model]
            sdt = compute_sdt(df)
            bacc = sdt["balanced_acc"] if not np.isnan(sdt["balanced_acc"]) else np.nan
        else:
            bacc = np.nan

        enc_acc = pd.read_csv(enc_path)["accuracy"].max() if enc_path.exists() else np.nan
        llm_acc = pd.read_csv(llm_path)["accuracy"].max() if llm_path.exists() else np.nan

        shuf_enc_path = results_dir / f"{model}_shuffled_probe_encoder.csv"
        shuf_llm_path = results_dir / f"{model}_shuffled_probe_llm.csv"
        shuf_enc = pd.read_csv(shuf_enc_path)["shuffled_mean"].mean() if shuf_enc_path.exists() else np.nan
        shuf_llm = pd.read_csv(shuf_llm_path)["shuffled_mean"].mean() if shuf_llm_path.exists() else np.nan

        models.append(model)
        bacc_vals.append(bacc)
        enc_vals.append(enc_acc)
        llm_vals.append(llm_acc)
        shuf_enc_vals.append(shuf_enc)
        shuf_llm_vals.append(shuf_llm)

    if not models:
        return
    x = np.arange(len(models))
    w = 0.19
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    bacc_plot = [v if not np.isnan(v) else 0 for v in bacc_vals]
    ax.bar(x - 1.5 * w, bacc_plot, w, label="Behavioral BAcc",
           color=[MODEL_COLORS[m] for m in models], alpha=0.4, edgecolor="black", lw=0.5)
    for i, v in enumerate(bacc_vals):
        if np.isnan(v):
            ax.bar(x[i] - 1.5 * w, 0.08, w, bottom=0, color="none",
                   edgecolor="gray", lw=0.5, hatch="///")
            ax.text(x[i] - 1.5 * w, 0.04, "N/A", ha="center", va="center", fontsize=4, color="gray")
    ax.bar(x - 0.5 * w, enc_vals, w, label="Probe: Encoder",
           color=[MODEL_COLORS[m] for m in models], alpha=0.7, edgecolor="black", lw=0.5)
    ax.bar(x + 0.5 * w, llm_vals, w, label="Probe: LLM",
           color=[MODEL_COLORS[m] for m in models], alpha=1.0, edgecolor="black", lw=0.5)

    shuf_avg = []
    for i in range(len(models)):
        vals = [v for v in [shuf_enc_vals[i], shuf_llm_vals[i]] if not np.isnan(v)]
        shuf_avg.append(np.mean(vals) if vals else np.nan)
    shuf_plot = [v if not np.isnan(v) else 0 for v in shuf_avg]
    ax.bar(x + 1.5 * w, shuf_plot, w, label="Shuffled control",
           color="lightgray", edgecolor="black", lw=0.5, hatch="xxx")
    for i, v in enumerate(shuf_avg):
        if np.isnan(v):
            ax.bar(x[i] + 1.5 * w, 0.08, w, bottom=0, color="none",
                   edgecolor="gray", lw=0.5, hatch="///")
            ax.text(x[i] + 1.5 * w, 0.04, "N/A", ha="center", va="center", fontsize=4, color="gray")

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance (binary)")
    ax.axhline(0.25, color="gray", ls=":", lw=0.8, label="Chance (4-class)")
    ax.set_xticks(x)
    ax.set_xticklabels([_display(m) for m in models], fontsize=6, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=5.5, ncol=3, loc="upper right")
    ax.set_title("")

    for i, m in enumerate(models):
        b = bacc_vals[i]
        l = llm_vals[i]
        if not np.isnan(l) and not np.isnan(b) and b > 0:
            gap = l - b
            if gap > 0.3:
                ax.annotate("", xy=(i + 0.5 * w, l), xytext=(i - 1.5 * w, b),
                            arrowprops=dict(arrowstyle="<->", color="red", lw=1.0))
                ax.text(i + 0.5 * w + 0.12, (l + b) / 2,
                        f"$\\Delta$={gap:.2f}", fontsize=5, color="red", va="center")

    fig.tight_layout()
    _save(fig, figs_dir / f"fig3_gap.{FIG_EXT}")


# ===================================================================
# FIGURE 4: Layer Probe (normalized depth)
# ===================================================================

def fig_layer_probe(results_dir: Path, figs_dir: Path):
    stages = {}
    for csv in sorted(results_dir.glob("*_layers.csv")):
        df = pd.read_csv(csv)
        stage = df["stage"].iloc[0] if "stage" in df.columns else "llm"
        model = csv.stem.replace(f"_{stage}_layers", "").replace("_layers", "")
        if model not in MODEL_ORDER:
            continue
        stages.setdefault(stage, []).append((model, df))

    if not stages:
        return

    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(3.25 * n, 2.8), squeeze=False)

    for ax, (stage, pairs) in zip(axes[0], sorted(stages.items())):
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
        for model, df in _ordered(pairs):
            max_layer = df["layer"].max()
            norm_depth = df["layer"] / max_layer if max_layer > 0 else df["layer"]
            ax.plot(norm_depth, df["accuracy"],
                    marker=MODEL_MARKERS.get(model, "o"),
                    label=_display(model), color=MODEL_COLORS.get(model),
                    markersize=3, lw=1.0)
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel("Probe accuracy (5-fold CV)")
        ax.set_title(f"({chr(97 + list(sorted(stages.keys())).index(stage))}) "
                     f"{stage.upper()} layers", fontsize=8)
        ax.set_ylim(0.35, 1.05)
        ax.legend(fontsize=5, loc="lower right")

    fig.suptitle("")
    fig.tight_layout()
    _save(fig, figs_dir / f"fig4_layer_probe.{FIG_EXT}")


# ===================================================================
# FIGURE 5: Psychometric Curves (selective fitting)
# ===================================================================

def _psychometric(log_x, threshold, slope, floor):
    return floor + (1.0 - floor) / (1.0 + np.exp(-slope * (log_x - np.log(threshold))))


def fig_psychometric(results_dir: Path, figs_dir: Path):
    pairs = _ordered(_load_task(results_dir, "order"))
    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
    x_dense = np.logspace(np.log10(25), np.log10(15000), 300)

    for model, df in pairs:
        if len(df) == 0:
            continue
        sdt_overall = compute_sdt(df)
        curve = compute_balanced_acc_curve(df)
        if len(curve) == 0:
            continue
        color = MODEL_COLORS[model]

        if sdt_overall["d_prime"] > 1.0:
            ax.scatter(curve["interval_ms"], curve["bacc"], color=color, s=20,
                       marker=MODEL_MARKERS[model], zorder=5)
            try:
                popt, _ = curve_fit(
                    _psychometric,
                    np.log(curve["interval_ms"].values),
                    curve["bacc"].values,
                    p0=[500.0, 1.0, 0.5],
                    bounds=([20.0, 0.01, 0.0], [12000.0, 10.0, 0.55]),
                    maxfev=10000,
                )
                threshold = popt[0]
                if threshold < 11000:
                    y_fit = _psychometric(np.log(x_dense), *popt)
                    ax.plot(x_dense, y_fit, color=color, lw=1.2,
                            label=f"{_display(model)} ($\\theta$={threshold:.0f} ms)")
                else:
                    ax.plot(curve["interval_ms"], curve["bacc"], color=color, ls="--",
                            lw=0.8, label=f"{_display(model)} (no threshold)")
            except RuntimeError:
                ax.plot(curve["interval_ms"], curve["bacc"], color=color, ls="--",
                        lw=0.8, label=f"{_display(model)} (fit failed)")
        else:
            pass

    ax.set_xscale("log")
    ax.set_xticks(TICK_MS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Inter-event interval (ms)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.set_title("")
    _save(fig, figs_dir / f"fig5_psychometric.{FIG_EXT}")


# ===================================================================
# FIGURE 6: Frame Rate Control (focused 2-panel)
# ===================================================================

def fig_fps(results_dir: Path, figs_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8))

    # Panel (a): molmo2 at 3 fps values
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
    fps_colors = {8: "#E57373", 16: "#64B5F6", 30: "#81C784"}
    for fps in [8, 16, 30]:
        path = results_dir / f"molmo2_order_fps{fps}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).dropna(subset=["correct"])
        curve = compute_balanced_acc_curve(df)
        ax1.plot(curve["interval_ms"], curve["bacc"], marker="o", markersize=3,
                 lw=1.2, label=f"{fps} fps", color=fps_colors[fps])

    ax1.set_xscale("log")
    ax1.set_xticks(TICK_MS)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlabel("Interval (ms)")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("(a) Molmo2: frame rate effect", fontsize=8)
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(fontsize=6)

    # Panel (b): all models at 30fps — only show models with d'>0.5
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    for model in MODEL_ORDER:
        path = results_dir / f"{model}_order_fps30.csv"
        if not path.exists():
            path = results_dir / f"{model}_order.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).dropna(subset=["correct"])
        if len(df) == 0:
            continue
        sdt = compute_sdt(df)
        curve = compute_balanced_acc_curve(df)
        if sdt["d_prime"] > 0.5:
            ax2.plot(curve["interval_ms"], curve["bacc"], marker=MODEL_MARKERS[model],
                     markersize=3, lw=1.2, label=_display(model), color=MODEL_COLORS[model])
        else:
            ax2.plot(curve["interval_ms"], curve["bacc"], marker=MODEL_MARKERS[model],
                     markersize=2, lw=0.4, alpha=0.3, label=f"{_display(model)} ($d'$<0.5)",
                     color=MODEL_COLORS[model])

    ax2.set_xscale("log")
    ax2.set_xticks(TICK_MS)
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2.set_xlabel("Interval (ms)")
    ax2.set_ylabel("Balanced accuracy")
    ax2.set_title("(b) All models at 30 fps", fontsize=8)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=5, ncol=2)

    fig.tight_layout(w_pad=2)
    _save(fig, figs_dir / f"fig6_fps.{FIG_EXT}")


# ===================================================================
# FIGURE 7: Confusion Matrix Grid (E002)
# ===================================================================

def fig_confusion_grid(results_dir: Path, figs_dir: Path):
    pairs = _ordered(_load_task(results_dir, "interval"))
    pairs = [(m, d) for m, d in pairs if len(d) > 0]
    if not pairs:
        return

    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 2.2 * nrows), squeeze=False)

    labels = ["A", "B", "C", "D"]
    for idx, (model, df) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        valid = df.dropna(subset=["pred", "gt"])
        mat = pd.crosstab(valid["gt"], valid["pred"], normalize="index")
        mat = mat.reindex(index=labels, columns=labels, fill_value=0)

        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels, fontsize=6)
        for i in range(4):
            for j in range(4):
                v = mat.values[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5, color="white" if v > 0.6 else "black")
        ax.set_title(_display(model), fontsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("True", fontsize=7)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Predicted", fontsize=7)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("")
    fig.tight_layout()
    _save(fig, figs_dir / f"fig7_confusion.{FIG_EXT}")


# ===================================================================
# FIGURE 8: Bias Dashboard
# ===================================================================

def fig_bias_dashboard(results_dir: Path, figs_dir: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.75, 2.8))

    # Panel (a): P(YES) for order
    pairs_order = _ordered(_load_task(results_dir, "order"))
    models_o, pyes_o = [], []
    for model, df in pairs_order:
        if len(df) == 0:
            continue
        sdt = compute_sdt(df)
        if not np.isnan(sdt["p_yes"]):
            models_o.append(model)
            pyes_o.append(sdt["p_yes"])

    y = np.arange(len(models_o))
    ax1.barh(y, pyes_o, color=[MODEL_COLORS[m] for m in models_o], height=0.6)
    ax1.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([_display(m) for m in models_o], fontsize=6)
    ax1.set_xlabel("$P(\\mathrm{YES})$")
    ax1.set_xlim(0, 1.05)
    ax1.set_title("(a) Order bias", fontsize=8)
    ax1.invert_yaxis()

    # Panel (b): P(YES) for simultaneity
    pairs_sim = _ordered(_load_task(results_dir, "simultaneity"))
    models_s, pyes_s = [], []
    for model, df in pairs_sim:
        if len(df) == 0:
            continue
        sdt = compute_sdt(df)
        if not np.isnan(sdt["p_yes"]):
            models_s.append(model)
            pyes_s.append(sdt["p_yes"])

    y2 = np.arange(len(models_s))
    ax2.barh(y2, pyes_s, color=[MODEL_COLORS[m] for m in models_s], height=0.6)
    ax2.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([_display(m) for m in models_s], fontsize=6)
    ax2.set_xlabel("$P(\\mathrm{YES})$")
    ax2.set_xlim(0, 1.05)
    ax2.set_title("(b) Simultaneity bias", fontsize=8)
    ax2.invert_yaxis()

    # Panel (c): Stacked bar for interval prediction distribution
    pairs_int = _ordered(_load_task(results_dir, "interval"))
    models_i = []
    dist_data = {"A": [], "B": [], "C": [], "D": []}
    for model, df in pairs_int:
        valid = df.dropna(subset=["pred"])
        if len(valid) == 0:
            continue
        models_i.append(model)
        counts = valid["pred"].value_counts(normalize=True)
        for letter in "ABCD":
            dist_data[letter].append(counts.get(letter, 0))

    y3 = np.arange(len(models_i))
    bottom = np.zeros(len(models_i))
    bin_colors = {"A": "#66BB6A", "B": "#42A5F5", "C": "#FFA726", "D": "#EF5350"}
    for letter in "ABCD":
        vals = np.array(dist_data[letter])
        ax3.barh(y3, vals, left=bottom, height=0.6, label=letter, color=bin_colors[letter])
        bottom += vals

    ax3.set_yticks(y3)
    ax3.set_yticklabels([_display(m) for m in models_i], fontsize=6)
    ax3.set_xlabel("Proportion")
    ax3.set_xlim(0, 1.05)
    ax3.set_title("(c) Interval class dist.", fontsize=8)
    ax3.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax3.invert_yaxis()

    fig.tight_layout(w_pad=2)
    _save(fig, figs_dir / f"fig8_bias.{FIG_EXT}")


# ===================================================================
# Legacy CSV outputs (backward compatibility)
# ===================================================================

def summary_tables_csv(results_dir: Path):
    for task in ("order", "interval", "simultaneity"):
        pairs = _load_task(results_dir, task)
        if not pairs:
            continue
        frames = []
        for model, df in _ordered(pairs):
            acc = df.groupby("interval_ms")["correct"].mean().rename(_display(model))
            frames.append(acc)
        if frames:
            table = pd.concat(frames, axis=1).T
            table.index.name = "model"
            out = results_dir / f"summary_{task}.csv"
            table.to_csv(out, float_format="%.3f")
            print(f"  CSV -> {out}")


def response_bias_csv(results_dir: Path):
    rows = []
    for task in ("order", "simultaneity"):
        for model, df in _load_task(results_dir, task):
            valid = df.dropna(subset=["pred"])
            yes_rate = (valid["pred"] == 1).mean() if len(valid) else np.nan
            rows.append({"model": model, "task": task,
                         "metric": "P(YES)", "value": yes_rate, "n": len(valid)})
    for model, df in _load_task(results_dir, "interval"):
        valid = df.dropna(subset=["pred"])
        for letter in "ABCD":
            rate = (valid["pred"] == letter).mean() if len(valid) else np.nan
            rows.append({"model": model, "task": "interval",
                         "metric": f"P({letter})", "value": rate, "n": len(valid)})
    if rows:
        bias = pd.DataFrame(rows)
        out = results_dir / "response_bias.csv"
        bias.to_csv(out, index=False)
        print(f"  CSV -> {out}")


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Publication-quality analysis for Temporal Fidelity paper")
    p.add_argument("--results", default="results")
    p.add_argument("--figs", default="figs")
    p.add_argument("--format", default="pdf", choices=["pdf", "png"])
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--no-bootstrap", action="store_true")
    args = p.parse_args()

    rdir = Path(args.results)
    fdir = Path(args.figs)
    FIG_EXT = args.format
    n_boot = 0 if args.no_bootstrap else 1000

    if not args.figures_only:
        print("\n" + "=" * 60)
        print("TABLES")
        print("=" * 60)
        table_main_results(rdir, rdir, n_boot)
        table_interval(rdir, rdir)
        table_rep_behavior_gap(rdir, rdir)
        table_fps(rdir, rdir)
        table_simultaneity(rdir, rdir)
        summary_tables_csv(rdir)
        response_bias_csv(rdir)

    if not args.tables_only:
        print("\n" + "=" * 60)
        print("FIGURES")
        print("=" * 60)
        fig_order_corrected(rdir, fdir)
        fig_sdt_summary(rdir, fdir, n_boot)
        fig_gap(rdir, fdir)
        fig_layer_probe(rdir, fdir)
        fig_psychometric(rdir, fdir)
        fig_fps(rdir, fdir)
        fig_confusion_grid(rdir, fdir)
        fig_bias_dashboard(rdir, fdir)

    print("\nDone. Tables -> results/  Figures -> figs/")
