import numpy as np
import pandas as pd
from stage4.registry import REGISTRY


def holm_fixed(pvalues):
    p = np.asarray(pvalues, dtype=float)
    book = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(book, kind='stable')
    adjusted = np.empty(len(book))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(book)-np.arange(len(book))) * book[order]))
    return adjusted


def paired_values(df, row):
    _, a, b, ma, mb, suites, _ = row
    keys = ['suite', 'scene_id', 'frame_id']
    x = df[(df.tag == a) & df.suite.isin(suites)]
    y = df[(df.tag == b) & df.suite.isin(suites)]
    if ma not in x or mb not in y or not len(x) or not len(y):
        return pd.DataFrame(columns=['a', 'b'])
    return pd.concat([x.groupby(keys)[ma].mean().rename('a'), y.groupby(keys)[mb].mean().rename('b')], axis=1).dropna()


def registered_tests(df):
    import kinafford.analysis.aggregate as original
    previous = original.HYPOTHESES
    try:
        original.HYPOTHESES = REGISTRY
        tests = original.paired_tests(df, n_boot=2000)
    finally:
        original.HYPOTHESES = previous
    for col in ['wilcoxon_p', 'mean_a', 'mean_b', 'mean_diff', 'ci_lo', 'ci_hi', 'cliffs_delta', 'n_pairs']:
        if col not in tests:
            tests[col] = np.nan
    tests['holm_p'] = holm_fixed(tests.wilcoxon_p)
    computable = (tests.status == 'ok') & np.isfinite(tests.wilcoxon_p)
    tests['supported'] = computable & (tests.holm_p < 0.05) & (tests.ci_hi < 0)
    tests['holm_bookkeeping_p'] = tests.wilcoxon_p.fillna(1.0)
    tests.loc[~computable, 'holm_p'] = np.nan
    for i, row in enumerate(REGISTRY):
        j = paired_values(df, row)
        d = j.a-j.b
        tests.loc[i, 'median_diff'] = d.median() if len(d) else np.nan
        tests.loc[i, 'improved_fraction'] = (d < 0).mean() if len(d) else np.nan
        tests.loc[i, 'sign_negative'] = bool(len(d) and d.mean() < 0)
        for side, tag in [('a', row[1]), ('b', row[2])]:
            selected = df[(df.tag == tag) & df.suite.isin(row[5])]
            tests.loc[i, f'seeds_{side}'] = ','.join(str(int(s)) for s in sorted(selected.seed.unique()))
    return tests


def descriptive_mean_ci(frame, metric, n_boot=2000):
    if metric not in frame or frame[metric].notna().sum() == 0:
        return (np.nan, np.nan, np.nan)
    x = frame.groupby(['scene_id', 'frame_id'])[metric].mean().dropna()
    groups = [g.to_numpy() for _, g in x.groupby(level='scene_id')]
    rng = np.random.default_rng(0)
    estimates = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)
