"""Aggregate per-frame result CSVs into tables and pre-registered hypothesis tests.

    python -m kinafford.analysis.aggregate --results results --out results/analysis

Outputs: summary_by_suite.csv, main_table.{md,tex}, calibration_table.{md,tex}, modality_table.{md,tex},
paired_tests.{csv,md}, miou_by_suite.csv, severity_curves.csv, RESULTS_SUMMARY.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..sim.scene import SEM_CLASSES, N_BEV_SEM
from ..metrics import N_ECE_BINS, ece_from_bins

MAIN_SUITES = ["clean", "iid", "cam_s2", "lidar_s2", "ext_s2", "joint_s2", "all_s2", "all_s3", "ood_artic", "ood_terrain"]
MAIN_METRICS = ["H_mae_vis", "sem_miou", "haz_iou", "aff_mae", "regret_direct", "unsafe_direct", "regret_comp"]
CAL_METRICS = ["H_nll_vis", "H_cov90_vis", "cp_H_cov", "cp_H_width", "sem_ece", "err_auroc", "cp_aff_cov", "regret_risk", "aurc_direct"]
VARIANT_ORDER = ["G1_perceiver", "G2_static", "G3_dyn", "G4_late", "G5_cam", "G6_lidar", "R0_plain", "R2_noaff", "R2_full",
                 "R2_full_ens", "R2_full_gooseinit"]
PRETTY = {"G1_perceiver": "Raw-token Perceiver (implicit geometry)", "G2_static": "Static-extrinsic BEV fusion",
          "G3_dyn": "Kinematics-aligned BEV fusion (dynamic)", "G4_late": "Late fusion (dynamic)", "G5_cam": "Cameras only (dynamic)",
          "G6_lidar": "LiDAR only", "R0_plain": "Dynamic fusion, no uncertainty/affordance heads",
          "R2_noaff": "KinAfford-Ex w/o direct affordance head", "R2_full": "KinAfford-Ex (full)",
          "R2_full_ens": "KinAfford-Ex ensemble (3 seeds)", "R2_full_gooseinit": "KinAfford-Ex, GOOSE-Ex-initialised encoder"}

# pre-registered comparisons: (name, variant_a, variant_b, metric_a, metric_b, suites, direction)  a better than b if a-b<0
HYPOTHESES = [
    ("H1a kinematic alignment vs static extrinsics", "G3_dyn", "G2_static", "H_mae_vis", "H_mae_vis", ["iid", "ood_artic", "clean"], "lower"),
    ("H1a (decision)", "G3_dyn", "G2_static", "regret_direct", "regret_direct", ["iid", "ood_artic"], "lower"),
    ("H1b kinematic alignment vs implicit fusion", "G3_dyn", "G1_perceiver", "H_mae_vis", "H_mae_vis", ["iid", "ood_artic", "clean"], "lower"),
    ("H1b (decision)", "G3_dyn", "G1_perceiver", "regret_direct", "regret_direct", ["iid", "ood_artic"], "lower"),
    ("H2 direct vs composed affordance (same model)", "R2_full", "R2_full", "regret_direct", "regret_comp", ["iid", "all_s2", "ood_terrain"], "lower"),
    ("H2 direct head vs no-affordance-head model", "R2_full", "R2_noaff", "regret_direct", "regret_comp", ["iid", "all_s2", "ood_terrain"], "lower"),
    ("H3a gating+uncertainty vs plain dynamic fusion (regret under faults)", "R2_full", "G3_dyn", "regret_risk", "regret_direct", ["cam_s2", "lidar_s2", "all_s2", "all_s3"], "lower"),
    ("H3a (height error under faults)", "R2_full", "G3_dyn", "H_mae_vis", "H_mae_vis", ["cam_s2", "lidar_s2", "all_s2", "all_s3"], "lower"),
    ("H3b calibration: NLL under faults", "R2_full", "G3_dyn", "H_nll_vis", "H_nll_vis", ["all_s2", "all_s3", "ood_artic"], "lower"),
    ("H3c risk-aware vs risk-neutral selection (unsafe picks)", "R2_full", "R2_full", "unsafe_risk", "unsafe_direct", ["all_s2", "all_s3"], "lower"),
    ("H3d ensemble vs single model (NLL)", "R2_full_ens", "R2_full", "H_nll_vis", "H_nll_vis", ["iid", "all_s2", "ood_artic"], "lower"),
    ("H4a multimodal vs cameras only", "R2_full", "G5_cam", "H_mae_vis", "H_mae_vis", ["iid", "cam_s2", "all_s2"], "lower"),
    ("H4b multimodal vs LiDAR only", "R2_full", "G6_lidar", "regret_direct", "regret_direct", ["iid", "lidar_s2", "all_s2"], "lower"),
    ("H4c early/feature fusion vs late fusion", "R2_full", "G4_late", "H_mae_vis", "H_mae_vis", ["iid", "all_s2"], "lower"),
]


def load_results(results_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        df = pd.read_csv(f)
        frames.append(df)
    if not frames:
        raise SystemExit("no result CSVs")
    df = pd.concat(frames, ignore_index=True)
    # derived per-frame quantities
    return df


def suite_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Suite-level quantities that do not average over frames (mIoU, ECE, AURC) per (tag, seed, suite)."""
    rows = []
    cm_cols = [f"cm_{i}" for i in range(N_BEV_SEM ** 2)]
    for (tag, seed, suite), g in df.groupby(["tag", "seed", "suite"]):
        cm = g[cm_cols].sum().values.reshape(N_BEV_SEM, N_BEV_SEM)
        ious = []
        for c in range(N_BEV_SEM):
            tp = cm[c, c]
            denom = cm[c, :].sum() + cm[:, c].sum() - tp
            if cm[c, :].sum() > 0:
                ious.append(tp / max(denom, 1))
        csum = g[[f"ece_c{i}" for i in range(N_ECE_BINS)]].sum().values
        asum = g[[f"ece_a{i}" for i in range(N_ECE_BINS)]].sum().values
        cnt = g[[f"ece_n{i}" for i in range(N_ECE_BINS)]].sum().values
        row = dict(tag=tag, seed=seed, suite=suite, sem_miou=float(np.mean(ious)), sem_ece=ece_from_bins(csum, asum, cnt))
        for c in range(N_BEV_SEM):
            tp = cm[c, c]
            denom = cm[c, :].sum() + cm[:, c].sum() - tp
            row[f"iou_{SEM_CLASSES[c]}"] = float(tp / denom) if denom > 0 else np.nan
        # selective prediction (risk-coverage) for the direct and risk-aware selectors
        from ..decision import aurc
        for sel in ("direct", "risk", "comp"):
            if f"conf_{sel}" in g and g[f"conf_{sel}"].notna().any():
                a, r50, r80 = aurc(g[f"conf_{sel}"].values, g[f"regret_{sel}"].values)
                row[f"aurc_{sel}"], row[f"regret{sel}_cov50"], row[f"regret{sel}_cov80"] = a, r50, r80
        rows.append(row)
    return pd.DataFrame(rows)


def hier_bootstrap(g: pd.DataFrame, metric: str, n_boot: int = 1000, seed: int = 0):
    """Hierarchical bootstrap over seeds and scenes for the mean of `metric` in a (tag, suite) group."""
    rng = np.random.default_rng(seed)
    seeds = g["seed"].unique()
    scenes = g["scene_id"].unique()
    piv = g.pivot_table(index="scene_id", columns="seed", values=metric, aggfunc="mean")
    M = piv.values  # scenes x seeds
    if M.size == 0 or np.all(np.isnan(M)):
        return np.nan, (np.nan, np.nan)
    est = np.nanmean(M)
    stats = []
    for _ in range(n_boot):
        si = rng.integers(0, M.shape[1], size=M.shape[1])
        ci = rng.integers(0, M.shape[0], size=M.shape[0])
        stats.append(np.nanmean(M[np.ix_(ci, si)]))
    return float(est), (float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5)))


def summarize(df: pd.DataFrame, suite_df: pd.DataFrame, metrics, n_boot=600) -> pd.DataFrame:
    rows = []
    for (tag, suite), g in df.groupby(["tag", "suite"]):
        for m in metrics:
            if m in g.columns:
                est, (lo, hi) = hier_bootstrap(g, m, n_boot=n_boot)
                rows.append(dict(tag=tag, suite=suite, metric=m, mean=est, lo=lo, hi=hi, n_seeds=g["seed"].nunique(),
                                 n_frames=len(g) // max(g["seed"].nunique(), 1)))
    for (tag, suite), g in suite_df.groupby(["tag", "suite"]):
        for m in [c for c in suite_df.columns if c not in ("tag", "seed", "suite")]:
            v = g[m].dropna().values
            if len(v):
                rows.append(dict(tag=tag, suite=suite, metric=m, mean=float(v.mean()),
                                 lo=float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
                                 hi=float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
                                 n_seeds=len(v), n_frames=np.nan))
    return pd.DataFrame(rows)


def fmt(mean, lo, hi, digits=3):
    if pd.isna(mean):
        return "--"
    if pd.isna(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table(summary: pd.DataFrame, metrics, suites, tags, digits=3):
    md, tex = [], []
    header = ["Variant", "Suite"] + metrics
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "---|" * len(header))
    tex.append("\\begin{tabular}{ll" + "c" * len(metrics) + "}\\toprule")
    tex.append(" & ".join(h.replace("_", "\\_") for h in header) + " \\\\ \\midrule")
    for tag in tags:
        for suite in suites:
            cells = []
            for m in metrics:
                r = summary[(summary.tag == tag) & (summary.suite == suite) & (summary.metric == m)]
                cells.append(fmt(r["mean"].iloc[0], r["lo"].iloc[0], r["hi"].iloc[0], digits) if len(r) else "--")
            md.append("| " + " | ".join([PRETTY.get(tag, tag), suite] + cells) + " |")
            tex.append(" & ".join([PRETTY.get(tag, tag), suite.replace("_", "\\_")] + cells) + " \\\\")
    tex.append("\\bottomrule\\end{tabular}")
    return "\n".join(md), "\n".join(tex)


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None]).sum()
    lt = (x[:, None] < y[None]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def paired_tests(df: pd.DataFrame, n_boot=2000):
    rows = []
    for name, ta, tb, ma, mb, suites, direction in HYPOTHESES:
        a = df[(df.tag == ta) & (df.suite.isin(suites))]
        b = df[(df.tag == tb) & (df.suite.isin(suites))]
        if len(a) == 0 or len(b) == 0 or ma not in a.columns or mb not in b.columns:
            rows.append(dict(hypothesis=name, a=ta, b=tb, metric_a=ma, metric_b=mb, suites="+".join(suites), status="missing"))
            continue
        # seed-average per frame, then pair on (suite, scene, frame)
        ka = a.groupby(["suite", "scene_id", "frame_id"])[ma].mean()
        kb = b.groupby(["suite", "scene_id", "frame_id"])[mb].mean()
        j = pd.concat([ka.rename("a"), kb.rename("b")], axis=1).dropna()
        if len(j) < 10:
            rows.append(dict(hypothesis=name, a=ta, b=tb, metric_a=ma, metric_b=mb, suites="+".join(suites), status="too few pairs"))
            continue
        d = j["a"].values - j["b"].values
        rng = np.random.default_rng(0)
        # cluster (scene) bootstrap of the mean difference
        scenes = j.index.get_level_values("scene_id").values
        uniq = np.unique(scenes)
        groups = {s: d[scenes == s] for s in uniq}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            boots.append(np.concatenate([groups[s] for s in pick]).mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        try:
            p = wilcoxon(d, alternative="less" if direction == "lower" else "greater").pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = np.nan
        rows.append(dict(hypothesis=name, a=ta, b=tb, metric_a=ma, metric_b=mb, suites="+".join(suites), status="ok",
                         n_pairs=len(d), mean_a=float(j["a"].mean()), mean_b=float(j["b"].mean()), mean_diff=float(d.mean()),
                         ci_lo=float(lo), ci_hi=float(hi), rel_change=float(d.mean() / (abs(j["b"].mean()) + 1e-9)),
                         wilcoxon_p=float(p), cliffs_delta=cliffs_delta(j["a"].values, j["b"].values)))
    out = pd.DataFrame(rows)
    ok = out["status"] == "ok"
    if ok.any():
        ps = out.loc[ok, "wilcoxon_p"].values
        order = np.argsort(ps)
        adj = np.empty_like(ps)
        m = len(ps)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * ps[i])
            adj[i] = min(running, 1.0)
        out.loc[ok, "holm_p"] = adj
        out.loc[ok, "supported"] = (adj < 0.05) & (out.loc[ok, "ci_hi"] < 0)
    return out


def severity_curves(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in ("cam", "lidar", "ext", "joint"):
        for lvl in (0, 1, 2, 3):
            suite = "clean" if lvl == 0 else f"{fam}_s{lvl}"
            sub = summary[summary.suite == suite]
            for _, r in sub.iterrows():
                rows.append(dict(family=fam, level=lvl, tag=r.tag, metric=r.metric, mean=r["mean"], lo=r.lo, hi=r.hi))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--n_boot", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_results(args.results)
    suite_df = suite_level_metrics(df)
    suite_df.to_csv(os.path.join(args.out, "miou_by_suite.csv"), index=False)
    all_metrics = sorted(set(MAIN_METRICS + CAL_METRICS + ["regret_direct_norm", "regret_comp_norm", "regret_risk_norm", "unsafe_comp",
                                                          "unsafe_risk", "top5_regret_direct", "top5_regret_comp", "H_mae_all",
                                                          "H_mae_reach", "mat_acc", "haz_miss_rate", "haz_brier", "affc_mae",
                                                          "aff_spearman", "H_cov90_all", "H_sigma_mean_vis", "H_sigma_mean_unvis",
                                                          "cp_aff_width", "lb_pick_true_aff"]) & set(df.columns))
    summary = summarize(df, suite_df, all_metrics, n_boot=args.n_boot)
    summary.to_csv(os.path.join(args.out, "summary_by_suite.csv"), index=False)
    tags = [t for t in VARIANT_ORDER if t in df.tag.unique()] + sorted(set(df.tag.unique()) - set(VARIANT_ORDER))
    suites = [s for s in MAIN_SUITES if s in df.suite.unique()]
    for name, mets in [("main_table", MAIN_METRICS), ("calibration_table", CAL_METRICS),
                       ("modality_table", ["H_mae_vis", "sem_miou", "regret_direct", "unsafe_direct", "H_nll_vis"])]:
        mets = [m for m in mets if m in summary.metric.unique()]
        md, tex = table(summary, mets, suites, tags)
        open(os.path.join(args.out, f"{name}.md"), "w").write(md)
        open(os.path.join(args.out, f"{name}.tex"), "w").write(tex)
    tests = paired_tests(df)
    tests.to_csv(os.path.join(args.out, "paired_tests.csv"), index=False)
    open(os.path.join(args.out, "paired_tests.md"), "w").write(tests.to_markdown(index=False) if hasattr(tests, "to_markdown") else tests.to_string())
    severity_curves(summary).to_csv(os.path.join(args.out, "severity_curves.csv"), index=False)
    # latency / params
    meta = []
    for f in glob.glob(os.path.join(args.results, "*_s[0-9]*.json")):
        try:
            meta.append(json.load(open(f)))
        except Exception:
            pass
    pd.DataFrame(meta).to_csv(os.path.join(args.out, "run_meta.csv"), index=False)
    # summary document
    lines = ["# RESULTS SUMMARY (auto-generated)", "", f"Variants: {tags}", f"Suites: {sorted(df.suite.unique())}",
             f"Seeds per variant: {df.groupby('tag').seed.nunique().to_dict()}", "", "## Pre-registered hypothesis tests", ""]
    for _, r in tests.iterrows():
        if r.get("status") == "ok":
            lines.append(f"- **{r.hypothesis}**: {r.a}.{r.metric_a} vs {r.b}.{r.metric_b} on {r.suites}: "
                         f"diff = {r.mean_diff:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] ({100 * r.rel_change:+.1f}%), "
                         f"Holm p = {r.holm_p:.2e}, Cliff's d = {r.cliffs_delta:+.3f} -> {'SUPPORTED' if r.supported else 'NOT supported'}")
        else:
            lines.append(f"- **{r.hypothesis}**: {r.status}")
    lines += ["", "## Main table", "", open(os.path.join(args.out, "main_table.md")).read(), "",
              "## Calibration table", "", open(os.path.join(args.out, "calibration_table.md")).read()]
    open(os.path.join(args.out, "RESULTS_SUMMARY.md"), "w").write("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"[aggregate] wrote analysis to {args.out}")


if __name__ == "__main__":
    main()
