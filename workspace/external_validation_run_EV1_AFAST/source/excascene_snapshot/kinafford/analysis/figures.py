"""Paper figures from the aggregated results.

    python -m kinafford.analysis.figures --results results --analysis results/analysis --out paper/figures
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics import N_ECE_BINS
from ..decision import aurc

STYLE = {"G1_perceiver": ("tab:gray", "--"), "G2_static": ("tab:red", "-."), "G3_dyn": ("tab:blue", "-"),
         "G4_late": ("tab:cyan", ":"), "G5_cam": ("tab:olive", ":"), "G6_lidar": ("tab:brown", ":"),
         "R0_plain": ("tab:purple", "--"), "R2_noaff": ("tab:pink", "-."), "R2_full": ("tab:green", "-"),
         "R2_full_ens": ("black", "-"), "R2_full_gooseinit": ("tab:orange", "-")}


METRIC_LABELS = {
    "H_mae_vis": "Height MAE, visible cells (m)",
    "regret_direct": "Dig-point regret\n(affordance units)",
    "H_nll_vis": "Height NLL (nats)",
    "cp_H_cov": "Conformal coverage\nof height intervals",
}

def style(tag):
    return STYLE.get(tag, ("k", "-"))


def fig_severity(sev: pd.DataFrame, out, metric="H_mae_vis", tags=None):
    fams = ["cam", "lidar", "ext", "joint"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, fam in zip(axes, fams):
        sub = sev[(sev.family == fam) & (sev.metric == metric)]
        for tag in (tags or sorted(sub.tag.unique())):
            s = sub[sub.tag == tag].sort_values("level")
            if len(s) == 0:
                continue
            c, ls = style(tag)
            ax.plot(s.level, s["mean"], color=c, ls=ls, marker="o", label=tag)
            ax.fill_between(s.level, s.lo, s.hi, color=c, alpha=0.12)
        ax.set_title({"cam": "Camera faults", "lidar": "LiDAR faults",
                      "ext": "Mount drift", "joint": "Joint-report bias"}[fam])
        ax.set_xlabel("severity level")
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric))
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(out, f"fig_severity_{metric}.pdf"))
    fig.savefig(os.path.join(out, f"fig_severity_{metric}.png"), dpi=150)
    plt.close(fig)


def fig_articulation(df: pd.DataFrame, out, tags=("G1_perceiver", "G2_static", "G3_dyn", "R2_full")):
    sub = df[df.suite.isin(["clean", "iid", "ood_artic"]) & df.tag.isin(tags)].copy()
    if len(sub) == 0:
        return
    bins = np.quantile(sub.dq_nom, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sub["bin"] = np.clip(np.digitize(sub.dq_nom, bins[1:-1]), 0, 4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric in zip(axes, ["H_mae_vis", "regret_direct"]):
        for tag in tags:
            g = sub[sub.tag == tag].groupby("bin")[metric].agg(["mean", "sem"])
            if len(g) == 0:
                continue
            c, ls = style(tag)
            ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], color=c, ls=ls, marker="o", label=tag, capsize=2)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(5)], rotation=30, fontsize=7)
        ax.set_xlabel("||q_true - q_nom|| (rad), quintile bins")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(os.path.join(out, "fig_articulation.pdf"))
    fig.savefig(os.path.join(out, "fig_articulation.png"), dpi=150)
    plt.close(fig)


def fig_risk_coverage(df: pd.DataFrame, out, suite="all_s2", tag="R2_full"):
    g = df[(df.tag == tag) & (df.suite == suite)]
    if len(g) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for sel, lab in [("comp", "composed affordance"), ("direct", "direct affordance head"), ("risk", "risk-aware (mu - sigma)")]:
        if f"conf_{sel}" not in g or g[f"conf_{sel}"].isna().all():
            continue
        conf, risk = g[f"conf_{sel}"].values, g[f"regret_{sel}"].values
        order = np.argsort(-conf)
        cum = np.cumsum(risk[order]) / (np.arange(len(risk)) + 1)
        cov = (np.arange(len(risk)) + 1) / len(risk)
        a, _, _ = aurc(conf, risk)
        ax.plot(cov, cum, label=f"{lab} (AURC={a:.3f})")
    ax.set_xlabel("coverage (fraction of frames acted on)")
    ax.set_ylabel("mean dig-point regret")
    ax.set_title(f"{tag} on {suite}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_risk_coverage.pdf"))
    fig.savefig(os.path.join(out, "fig_risk_coverage.png"), dpi=150)
    plt.close(fig)


def fig_reliability(df: pd.DataFrame, out, suite="all_s2", tags=("G1_perceiver", "G2_static", "G3_dyn", "R2_full", "R2_full_ens")):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for tag in tags:
        g = df[(df.tag == tag) & (df.suite == suite)]
        if len(g) == 0:
            continue
        c = g[[f"ece_c{i}" for i in range(N_ECE_BINS)]].sum().values
        a = g[[f"ece_a{i}" for i in range(N_ECE_BINS)]].sum().values
        n = g[[f"ece_n{i}" for i in range(N_ECE_BINS)]].sum().values
        m = n > 0
        col, ls = style(tag)
        ax.plot((c[m] / n[m]), (a[m] / n[m]), color=col, ls=ls, marker="o", ms=3, label=tag)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"semantic reliability ({suite})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_reliability.pdf"))
    fig.savefig(os.path.join(out, "fig_reliability.png"), dpi=150)
    plt.close(fig)


def fig_gates(df: pd.DataFrame, out, tag="R2_full"):
    g = df[df.tag == tag]
    if len(g) == 0 or "gate_0" not in g:
        return
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.5),
                             gridspec_kw={"height_ratios": [3, 1.35]})
    names = ["cab cam", "boom cam", "stick cam", "LiDAR"]
    rng = np.random.default_rng(0)  # Display jitter only; statistics use original targets.
    for i, ax in enumerate(axes[0]):
        if f"gate_{i}" not in g or f"rel_{i}" not in g:
            continue
        rel, gate = g[f"rel_{i}"].to_numpy(), g[f"gate_{i}"].to_numpy()
        valid = np.isfinite(rel) & np.isfinite(gate)
        positive = valid & (rel > 0)
        r_all = np.corrcoef(rel[valid], gate[valid])[0, 1]
        r_nz = np.corrcoef(rel[positive], gate[positive])[0, 1]
        ax.scatter(rel + rng.uniform(-0.02, 0.02, len(g)), gate, s=3, alpha=0.3)
        ax.set_title(f"{names[i]}\nr_all={r_all:.2f}; r_target>0={r_nz:.2f}", fontsize=10)
        ax.set_xlabel("Injected reliability target")
        ax.set_ylabel("Global gate after presence mask")
        ax.grid(alpha=0.3)
        means = [(float(t), float(gate[valid & (rel == t)].mean()))
                 for t in np.unique(rel[valid])]
        axes[1, i].axis("off")
        table = axes[1, i].table(cellText=[[f"{t:.2f}", f"{v:.3f}"] for t, v in means],
                                 colLabels=["Target", "Mean gate"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.25)
        print(f"[gates] {names[i]} r_all={r_all:.3f} r_target_positive={r_nz:.3f} means={means}")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    fig.text(0.5, 0.025, "Target > 0 excludes dropped cameras AND other zero-target faults. "
             "LiDAR has no presence mask. Means include all frames at each target.",
             ha="center", fontsize=10)
    fig.savefig(os.path.join(out, "fig_gates.pdf"))
    fig.savefig(os.path.join(out, "fig_gates.png"), dpi=150)
    plt.close(fig)


def fig_qualitative(results, out, tags=("G2_static", "G3_dyn", "R2_full"), suite="suite_all_s2", idx=0):
    panels = []
    for tag in tags:
        fs = sorted(glob.glob(os.path.join(results, "preds", f"{tag}_s0", f"{suite}_{idx}.npz")))
        if fs:
            panels.append((tag, np.load(fs[0])))
    if not panels:
        return
    cols = ["gt_H", "H", "H_sigma", "gt_aff", "aff"]
    fig, axes = plt.subplots(len(panels), len(cols), figsize=(2.6 * len(cols), 2.6 * len(panels)))
    axes = np.atleast_2d(axes)
    for r, (tag, d) in enumerate(panels):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            if key in d:
                ax.imshow(d[key], origin="lower", cmap="viridis" if "aff" in key else ("magma" if "sigma" in key else "terrain"))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(key, fontsize=8)
        axes[r, 0].set_ylabel(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_qualitative.pdf"))
    fig.savefig(os.path.join(out, "fig_qualitative.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.results, "*.csv")))], ignore_index=True)
    sev = pd.read_csv(os.path.join(args.analysis, "severity_curves.csv"))
    main_tags = [t for t in ("G1_perceiver", "G2_static", "G3_dyn", "R0_plain", "R2_full", "R2_full_ens") if t in df.tag.unique()]
    for metric in ("H_mae_vis", "regret_direct", "cp_H_cov", "H_nll_vis"):
        if metric in sev.metric.unique():
            fig_severity(sev, args.out, metric, tags=main_tags)
    fig_articulation(df, args.out)
    fig_risk_coverage(df, args.out)
    fig_reliability(df, args.out)
    fig_gates(df, args.out)
    fig_qualitative(args.results, args.out)
    print(f"[figures] written to {args.out}")


if __name__ == "__main__":
    main()
