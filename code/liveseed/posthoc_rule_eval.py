#!/usr/bin/env python
"""posthoc_rule_eval.py: evaluates a score computed on finished images as a seed-selection rule on a pool.

The example in the paper is CAS, the condition alignment score of Hong et al. (2024), used as a post-hoc rule (App. C).
For each prompt the candidate with the highest (or, with --lower-is-better, the lowest) score is picked. The report gives,
for every evaluator, the gain of the pick over the pool mean with a 95% bootstrap interval over prompts, the fraction of
the oracle gap closed and the win-rate, exactly as seed_select_eval.py does for the rules read during generation.

Input:   --scores    score_images.py output for the pool's finished images (image_id, seed, evaluator columns)
         --rule      a CSV with image_id, seed and the rule's score in the column --rule-col
Output (--out): posthoc_rule.csv, posthoc_rule.md and posthoc_rule.json, one row per evaluator: random_mean (pool mean),
         oracle_mean, gain, ci_lo, ci_hi, gap_closed, win_rate, n_prompts.

Usage: python posthoc_rule_eval.py --scores runs/sd14_pool/scores_finals.csv \\
           --rule runs/sd14_pool/scores_cas.csv --rule-col cas_score --name "CAS (SD1.5 likelihood)" \\
           --out runs/sd14_cas
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

METRICS = ["imagereward", "hpsv2", "pickscore", "clip_l14", "vqascore"]


def bootstrap_ci(x, n=2000, seed=0):
    """95% percentile bootstrap interval of the mean of x (NaNs dropped; n resamples, fixed seed): (lo, hi)."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def norm_id(col):
    """Image ids as canonical strings, so that '1.0' (an id column read as floats) and '1' match."""
    num = pd.to_numeric(col, errors="coerce")
    out = col.astype(str)
    ok = num.notna() & (num == num.round())
    out[ok] = num[ok].astype(int).astype(str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--rule-col", required=True)
    ap.add_argument("--name", default="post-hoc rule")      # the label used in the report
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # join the evaluator scores and the rule's score on (prompt, seed); a prompt needs at least two candidates to rank
    sc = pd.read_csv(a.scores, on_bad_lines="skip").drop_duplicates(["image_id", "seed"], keep="last")
    sc["image_id"] = norm_id(sc["image_id"])
    ru = pd.read_csv(a.rule, on_bad_lines="skip")
    ru["image_id"] = norm_id(ru["image_id"])
    ru = ru.dropna(subset=[a.rule_col]).drop_duplicates(["image_id", "seed"], keep="last")[["image_id", "seed", a.rule_col]]
    t = sc.merge(ru, on=["image_id", "seed"], how="inner")
    t = t[t.groupby("image_id").seed.transform("size") >= 2]
    metrics = [m for m in METRICS if m in t.columns]

    # the rule's pick: the candidate of each prompt with the best score
    sign = -1.0 if a.lower_is_better else 1.0
    idx = (sign * t[a.rule_col]).groupby(t.image_id).idxmax()
    pick = t.loc[idx.values].set_index("image_id")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = [f"# Post-hoc rule as a seed selector: {a.name}; scores {a.scores}; prompts = {t.image_id.nunique()}, candidates = {len(t)}", "",
             "| metric | random (pool mean) | oracle | top-1 gain [95% CI] | gap closed | win-rate |", "|---|---|---|---|---|---|"]
    for m in metrics:
        base = t.groupby("image_id")[m].mean()      # a random draw: the pool mean
        orc = t.groupby("image_id")[m].max()        # the oracle: the best candidate under this evaluator
        g = (pick[m] - base.loc[pick.index])
        lo, hi = bootstrap_ci(g.values)
        # fraction of the oracle gap closed, averaged over prompts; prompts with a zero gap are left out
        gap = float(np.nanmean(g / (orc.loc[pick.index] - base.loc[pick.index]).replace(0, np.nan)))
        win = float((g > 0).mean())
        rows.append(dict(metric=m, random_mean=float(base.mean()), oracle_mean=float(orc.mean()), gain=float(g.mean()), ci_lo=lo, ci_hi=hi,
                         gap_closed=gap, win_rate=win, n_prompts=int(len(g))))
        lines.append(f"| {m} | {base.mean():.4f} | {orc.mean():.4f} | {g.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {gap:+.2f} | {win:.2f} |")
    pd.DataFrame(rows).to_csv(out / "posthoc_rule.csv", index=False)
    (out / "posthoc_rule.md").write_text("\n".join(lines) + "\n")
    json.dump(dict(name=a.name, rule_col=a.rule_col, rows=rows), open(out / "posthoc_rule.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
