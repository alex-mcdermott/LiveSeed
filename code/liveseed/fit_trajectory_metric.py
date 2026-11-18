#!/usr/bin/env python
"""fit_trajectory_metric.py: offline calibration and held-out evaluation of the two LiveSeed forms on logged pools.

A reading is the normalised attention entropy of one (block, step) (Sec. 3.1). This file chooses, on data, how a set of
readings is combined into one score per image, and reports how well that score does on prompts it was not chosen on.
It needs no GPU: every candidate of a pool was generated to completion with its readings logged.

Forms (Sec. 3.2; the calibration procedure is Sec. 3.3)
  single   the best single reading with its sign.
  greedy   LiveSeed-Fast: forward selection of up to --max-k readings with signs, combined with equal weights
           (functions `greedy` and `combo_score`).
  ridge    LiveSeed-Max: a ridge regression over all candidate readings, its penalty chosen by an inner two-fold
           cross-validation (functions `ridge` and `ridge_fit`).
Every form is chosen on one half of the prompts and scored on the other; the folds are then swapped, so every prompt is
scored once by a rule chosen without it.

Modes
  selection    Ranking seeds of one prompt (Sec. 4.1). Readings are z-scored within each prompt and the objective is the
               mean gain of the top-scored seed over its prompt's mean, on each evaluator in the scores file.
               Input:  --run, a pool folder of seed_harness.py (results*.csv: one row per image_id, seed and step, one column
                       per reading, named sa_mean_<block> for U-Net self-attention and ae_mean_<block> for the joint
                       attention of the DiTs (diffusion transformers) and Hunyuan-DiT's self-attention);
                       --scores, the output of score_images.py for the finished images (image_id, seed, evaluator columns).
               Output: trajectory_selection.md and .json (per evaluator and form: held-out gain with a 95% bootstrap
                       interval, fraction of the oracle gap closed, win-rate, the readings chosen in each fold) and
                       picks_cv_<form>_<evaluator>.csv (the held-out pick: image_id, seed).
  correlation  Predicting human ratings across prompts (Sec. 4.4). --run holds the rated images, one per prompt,
               regenerated with their readings logged; --human holds the mean ratings (columns Dataset, ID and, per model,
               <MODEL>_Prompt_Alignment_Rating and <MODEL>_Image_Quality_Rating). Readings are z-scored across all images,
               the objective is Spearman's rho with the rating, and the folds alternate over images.
               Output: trajectory_correlation.md and .json (held-out rho per fold and pooled, with a bootstrap interval).
  transfer     Chooses the forms on ALL prompts of --run, freezes them, and scores them on --test-run (another prompt
               set of the same model; App. G). Output: trajectory_transfer.md and .json.
With --ca-stats the candidates also include cross-attention token statistics read from <run>/stats/*.npz (`ca_wide`).

Usage:
  python fit_trajectory_metric.py selection --run runs/sd14_pool --scores runs/sd14_pool/scores_finals.csv --model sd14 --out runs/sd14_selection
  python fit_trajectory_metric.py correlation --run runs/sdxl_rated --model sdxl --out runs/sdxl_correlation
  python fit_trajectory_metric.py transfer --run runs/sd14_pool --scores runs/sd14_pool/scores_finals.csv \\
         --test-run runs/sd14_pool_b --test-scores runs/sd14_pool_b/scores_finals.csv --model sd14 --out runs/sd14_transfer
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_select_eval import bootstrap_ci, reading_wide   # noqa: E402

# Rating columns of the --human file per model: PA = prompt alignment, IQ = image quality.
RATING = {"sdxl": dict(PA="SDXL_Prompt_Alignment_Rating", IQ="SDXL_Image_Quality_Rating"),
          "sd35": dict(PA="SD35_Prompt_Alignment_Rating", IQ="SD35_Image_Quality_Rating"),
          # FLUX.1 and SD1.4 have no expert ratings; a file of automatic-judge grades uses the same column convention
          "flux": dict(PA="FLUX_Prompt_Alignment_Rating", IQ="FLUX_Image_Quality_Rating"),
          "sd14": dict(PA="SD14_Prompt_Alignment_Rating", IQ="SD14_Image_Quality_Rating")}


# ----------------------------------------------------------------------------- reading the logs
def _read_results(run):
    """Returns all results*.csv of a pool folder as one table, one row per (image_id, seed, step).

    A pool may be written by several jobs at once, one results file each. Lines damaged by concurrent writers are
    skipped, the statistic columns are coerced to numbers, and image_id is returned as a string.
    """
    import pandas as _pd
    frames = [_pd.read_csv(f, on_bad_lines="skip", low_memory=False) for f in sorted(run.glob("results*.csv"))]
    r = _pd.concat(frames, ignore_index=True)
    for c in r.columns:
        if c.startswith(("sa_mean_", "ae_mean_", "ae_img_", "txtmass_")) or c in ("step", "seed", "gen_seconds"):
            r[c] = _pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["step", "seed"])
    r["step"] = r["step"].astype(int)
    r["seed"] = r["seed"].astype(int)
    num = _pd.to_numeric(r["image_id"], errors="coerce")          # damaged lines can turn the id column into floats ("1.0")
    if num.notna().mean() > 0.99:     # numeric ids: write them as integers, so that "1.0" and "1" are one prompt
        r = r[num.notna()].copy()
        r["image_id"] = num[num.notna()].astype(int).astype(str)
    else:
        r["image_id"] = r["image_id"].astype(str)
    return r


def zscore_within(df, cols, by="image_id"):
    """Z-scores each reading within its prompt's pool (mean and sample standard deviation over the prompt's seeds).

    Selection compares seeds of one prompt, so offsets shared by a prompt's seeds are removed and every reading is put
    on one scale. A reading that is constant within a prompt becomes NaN, which the fits below treat as 0.
    """
    g = df.groupby(by)[cols]
    return (df[cols] - g.transform("mean")) / (g.transform("std").replace(0, np.nan) + 1e-9)


# ----------------------------------------------------------------------------- objectives
def sel_objective(df, score, metric):
    """Selection objective (the within-prompt gain of Sec. 3.3): for each prompt, the metric of the candidate with the
    highest score minus the mean of the prompt's pool. Returns the per-prompt gains, indexed by image_id."""
    d = df.assign(_s=score.values)
    idx = d.groupby("image_id")["_s"].idxmax().dropna()
    pick = d.loc[idx.values].set_index("image_id")[metric]
    base = d.groupby("image_id")[metric].mean()
    return pick - base.loc[pick.index]          # per-prompt gains


def corr_objective(df, score, target):
    """Prediction objective: Spearman's rho between the score and the rating column `target`, over all rows."""
    return spearmanr(score, df[target], nan_policy="omit").correlation


# ----------------------------------------------------------------------------- LiveSeed-Fast: greedy selection of readings
def greedy(dfc, feats, objective, max_k, min_gain=1e-4):
    """Forward selection of up to max_k signed readings on the calibration table `dfc`.

    The score of a subset is the mean of its signed, z-scored readings (equal weights). Starting from the empty set,
    each round tries every remaining reading with both signs and adds the one that most raises the objective; it
    stops after max_k readings or when no addition raises the objective by more than min_gain. The sign lets a reading
    count either way: +1 scores higher entropy higher, -1 lower entropy higher.

    feats: the candidate columns "<statistic>_<block>@s<step>", already z-scored. objective(dfc, score) returns a scalar
    or per-prompt values, whose mean is used. Returns ([(reading, sign), ...] in the order chosen, the objective reached).
    """
    chosen, best_val = [], -np.inf
    cur = np.zeros(len(dfc))                  # the score of the subset chosen so far
    for _ in range(max_k):
        best = None
        for f in feats:
            if any(f == c for c, _ in chosen):
                continue
            x = dfc[f].values
            if np.isnan(x).mean() > 0.5:      # a reading missing on most rows is not a candidate
                continue
            x = np.nan_to_num(x)
            for sgn in (+1, -1):
                s = (cur * len(chosen) + sgn * x) / (len(chosen) + 1)     # the running mean with this reading added
                v = objective(dfc, pd.Series(s, index=dfc.index))
                v = float(np.nanmean(v)) if not np.isscalar(v) else float(v)
                if v > best_val + min_gain and (best is None or v > best[0]):
                    best = (v, f, sgn, s)
        if best is None:                      # no remaining reading improves the objective
            break
        best_val, f, sgn, cur = best
        chosen.append((f, sgn))
    return chosen, best_val


def combo_score(df, chosen):
    """Applies a chosen subset to a table: the mean of its signed readings, one score per row (0 for an empty subset)."""
    if not chosen:
        return pd.Series(np.zeros(len(df)), index=df.index)
    m = np.mean([sgn * np.nan_to_num(df[f].values) for f, sgn in chosen], axis=0)
    return pd.Series(m, index=df.index)


# ----------------------------------------------------------------------------- LiveSeed-Max: ridge regression
def ridge_fit(X, y, alpha):
    """Closed-form ridge without intercept: w = (X'X + alpha I)^-1 X'y. X: (rows, readings), missing values read as 0."""
    Xc = np.nan_to_num(X)
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ y)
    return w


# The grid of ridge penalties, shared by every mode of this file. The grid in the code is 1 to 10^4; the paper's
# rating-prediction results (Sec. 4.4) were computed with the grid 10^-1 to 10^3, as the appendix's calibration-settings
# table states.
RIDGE_ALPHAS = (1, 10, 100, 1000, 10000)


def ridge(dfc, feats, target_col, objective, alphas=RIDGE_ALPHAS):
    """Fits the ridge weights of LiveSeed-Max on the calibration table `dfc`; returns (weights, chosen alpha).

    Centring: the caller passes in `target_col` a target that is already centred (within the prompt for selection,
    across images for prediction), and the readings are z-scored the same way, so no intercept is fitted. With the
    target centred within the prompt, the weights explain differences between seeds of one prompt and not which
    prompts score high.

    Penalty: alpha is chosen by an inner two-fold cross-validation over the calibration prompts (alternate prompts in
    sorted order). For each alpha the ridge is fitted on one inner half and its scores are judged on the other by the
    same objective the form is evaluated with (the selection gain or rho), not by squared error. The alpha with the best
    mean over the two halves is kept and the ridge is refitted on all calibration prompts.
    """
    prompts = sorted(dfc.image_id.unique())
    inner = {p: i % 2 for i, p in enumerate(prompts)}
    m = dfc.image_id.map(inner).values        # inner fold (0 or 1) of every row
    y = dfc[target_col].values
    best_a, best_v = None, -np.inf
    for a in alphas:
        vals = []
        for k in (0, 1):
            tr, te = m != k, m == k
            w = ridge_fit(dfc.loc[tr, feats].values, y[tr], a)
            s = pd.Series(np.nan_to_num(dfc.loc[te, feats].values) @ w, index=dfc.index[te])
            v = objective(dfc[te], s)
            vals.append(float(np.nanmean(v)) if not np.isscalar(v) else float(v))
        if np.mean(vals) > best_v:
            best_v, best_a = np.mean(vals), a
    return ridge_fit(dfc[feats].values, y, best_a), best_a


# ----------------------------------------------------------------------------- mode: selection
def run_selection(args):
    """Builds the selection table of a pool (one row per prompt and seed: evaluator scores and z-scored readings) and
    evaluates the three forms on it with two-fold cross-validation over prompts."""
    from seed_select_eval import METRICS
    run = Path(args.run)
    if args.ca_stats:
        tbl, feats, metrics = _load_selection_table(args.run, args.scores, args.model, args.max_step, ca_stats=True)
        return _run_selection_table(args, run, tbl, feats, metrics)
    # A (prompt, seed, step) can be logged twice (a requeued job repeats the pair it was interrupted in): the last row is kept.
    res = _read_results(run).drop_duplicates(subset=["image_id", "seed", "step"], keep="last")
    res["image_id"] = res["image_id"].astype(str)
    sc = pd.read_csv(args.scores, on_bad_lines="skip", low_memory=False)
    sc["image_id"] = _norm_id(sc["image_id"])   # damaged csv lines can turn ids into floats ("1.0")
    sc["seed"] = pd.to_numeric(sc["seed"], errors="coerce")
    sc = sc.dropna(subset=["seed"])
    sc["seed"] = sc["seed"].astype(int)
    for m_ in METRICS:
        if m_ in sc.columns:
            sc[m_] = pd.to_numeric(sc[m_], errors="coerce")
    sc = sc.drop_duplicates(subset=["image_id", "seed"], keep="last")
    metrics = [m for m in METRICS if m in sc.columns]
    # the reading of the paper: joint attention (ae_mean_) on the transformers, self-attention (sa_mean_) on the U-Nets
    prefixes = ("ae_mean_",) if args.model.startswith(("sd3", "flux", "hunyuan", "qwen")) else ("sa_mean_",)
    wide = reading_wide(res, max_step=args.max_step, prefixes=prefixes)     # one column per (block, step): "<col>@s<step>"
    feats = [c for c in wide.columns if "@s" in c]
    tbl = sc[["image_id", "seed"] + metrics].merge(wide, on=["image_id", "seed"], how="inner")
    tbl = tbl[tbl.groupby("image_id").seed.transform("size") >= 2].reset_index(drop=True)   # a prompt needs two seeds to rank
    Z = zscore_within(tbl, feats)
    tbl[feats] = Z   # within-prompt z-scores: the score ranks seeds
    return _run_selection_table(args, run, tbl, feats, metrics)


def _top1(tst, score):
    """held-out top-1 candidate per prompt under a selection score (image_id, seed)."""
    d = tst[["image_id", "seed"]].copy()
    d["_s"] = score.values
    return d.sort_values("_s", ascending=False).drop_duplicates("image_id")[["image_id", "seed"]]


def _run_selection_table(args, run, tbl, feats, metrics):
    """Two-fold cross-validation over prompts of the three forms, for every evaluator; writes the selection report.

    Folds: prompt ids in sorted order, alternate positions. For each fold the forms are chosen on the other fold's
    prompts (`cal`) and scored on this fold's (`tst`); the held-out per-prompt gains of the two folds are then pooled.
    """
    prompts = sorted(tbl.image_id.unique())
    fold = {p: i % 2 for i, p in enumerate(prompts)}
    tbl["fold"] = tbl.image_id.map(fold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    lines = [f"# Trajectory metric, selection objective: {args.model} ({run.name}); candidates = {len(feats)} (block x step, steps <= {args.max_step}); N per prompt = {tbl.groupby('image_id').size().median():.0f}", ""]
    for metric in metrics:
        tbl["_t"] = tbl[metric] - tbl.groupby("image_id")[metric].transform("mean")   # centred target for ridge
        obj = lambda d, s, m=metric: sel_objective(d, s, m)
        held = {"single": [], "greedy": [], "ridge": []}            # held-out per-prompt gains, one entry per fold
        chosen_desc = {"single": [], "greedy": [], "ridge": []}     # what was chosen in each fold
        picks = {"single": [], "greedy": [], "ridge": []}   # held-out top-1 (image_id, seed) per prompt, for re-scoring under other judges
        for k in (0, 1):
            cal, tst = tbl[tbl.fold != k], tbl[tbl.fold == k]
            # the best single reading
            ch1, _ = greedy(cal, feats, obj, 1)
            held["single"].append(obj(tst, combo_score(tst, ch1)))
            chosen_desc["single"].append(ch1)
            picks["single"].append(_top1(tst, combo_score(tst, ch1)))
            # LiveSeed-Fast: the greedy subset of up to --max-k readings
            chg, _ = greedy(cal, feats, obj, args.max_k)
            held["greedy"].append(obj(tst, combo_score(tst, chg)))
            chosen_desc["greedy"].append(chg)
            picks["greedy"].append(_top1(tst, combo_score(tst, chg)))
            # LiveSeed-Max: the ridge over all candidates
            w, a = ridge(cal, feats, "_t", obj)
            s = pd.Series(np.nan_to_num(tst[feats].values) @ w, index=tst.index)
            held["ridge"].append(obj(tst, s))
            picks["ridge"].append(_top1(tst, s))
            top = np.argsort(-np.abs(w))[:5]              # the five largest weights, for the report
            chosen_desc["ridge"].append([(feats[i], round(float(w[i]), 4)) for i in top] + [("alpha", a)])
        rows = []
        for name in ("single", "greedy", "ridge"):
            g = pd.concat(held[name])
            lo, hi = bootstrap_ci(g.values)
            # oracle gap: what picking each prompt's best seed would gain; the fraction closed is averaged over prompts
            oracle = tbl.groupby("image_id")[metric].max() - tbl.groupby("image_id")[metric].mean()
            gap = float(np.nanmean(g / oracle.loc[g.index].replace(0, np.nan)))
            rows.append(dict(metric=metric, method=name, gain=float(g.mean()), ci_lo=lo, ci_hi=hi, gap_closed=gap,
                             win_rate=float((g > 0).mean()), n=int(len(g)), chosen=chosen_desc[name]))
        report[metric] = rows
        for name in ("single", "greedy", "ridge"):
            pd.concat(picks[name]).to_csv(out / f"picks_cv_{name}_{metric}.csv", index=False)
        lines += [f"## {metric}", "",
                  "| method | held-out top-1 gain [95% CI] | gap closed | win-rate | chosen (fold 0 / fold 1) |",
                  "|---|---|---|---|---|"]
        for r in rows:
            desc = " / ".join("+".join(f"{'-' if s < 0 else ''}{f.replace('sa_mean_', '').replace('ae_mean_', '').replace('_processor', '')}" for f, s in ch) if r["method"] != "ridge"
                              else ", ".join(f"{f.replace('sa_mean_', '').replace('ae_mean_', '').replace('_processor', '')}:{w}" for f, w in ch[:3]) for ch in r["chosen"])
            lines.append(f"| {r['method']} | {r['gain']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['gap_closed']:+.2f} | {r['win_rate']:.2f} | {desc[:200]} |")
        lines.append("")
    (out / "trajectory_selection.md").write_text("\n".join(lines))
    json.dump(report, open(out / "trajectory_selection.json", "w"), indent=1, default=str)
    print("\n".join(lines))


# ----------------------------------------------------------------------------- mode: correlation (Sec. 4.4)
def run_correlation(args):
    """Held-out Spearman correlation of each form with the human ratings, one image per prompt.

    Unlike selection, the readings are z-scored across all images and the target is centred across images, because the
    task is to predict a rating across prompts. Folds alternate over images.
    """
    run = Path(args.run)
    res = _read_results(run).drop_duplicates(subset=["image_id", "seed", "step"], keep="last")
    res["image_id"] = res["image_id"].astype(str)
    h = pd.read_csv(args.human, encoding="utf-8-sig").rename(columns={"Dataset": "dataset", "ID": "image_id"})
    h["image_id"] = h["image_id"].astype(str)
    prefixes = ("ae_mean_",) if args.model.startswith(("sd3", "flux", "hunyuan", "qwen")) else ("sa_mean_",)
    wide = reading_wide(res, max_step=args.max_step, prefixes=prefixes)
    if args.ca_stats:
        keys = list(wide[["image_id", "seed"]].itertuples(index=False, name=None))
        caw = ca_wide(run, keys, args.max_step, args.model)
        caw["seed"] = caw["seed"].astype(wide["seed"].dtype)
        wide = wide.merge(caw, on=["image_id", "seed"], how="left")
        if args.ca_only:
            wide = wide[["image_id", "seed"] + [c for c in wide.columns if c.startswith("ca_")]]
    feats = [c for c in wide.columns if "@s" in c]
    # one image per prompt: join on dataset+image_id via results' dataset column
    ds = res[["image_id", "seed", "dataset"]].drop_duplicates()
    tbl = wide.merge(ds, on=["image_id", "seed"]).merge(h, on=["dataset", "image_id"], how="inner")
    # global z-score (across prompts) for the correlation objective
    tbl[feats] = (tbl[feats] - tbl[feats].mean()) / (tbl[feats].std() + 1e-9)
    prompts = sorted(tbl.index)
    fold = {p: i % 2 for i, p in enumerate(prompts)}
    tbl["fold"] = [fold[i] for i in tbl.index]
    tbl["image_id"] = tbl.index.astype(str)   # ridge() takes its inner folds from image_id; here each row is its own unit
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Trajectory metric, correlation objective: {args.model}; candidates = {len(feats)} (block x step, steps <= {args.max_step}{', self/joint-attention entropies plus cross-attention token statistics' if args.ca_stats and not args.ca_only else (', cross-attention token statistics only' if args.ca_only else '')}); n = {len(tbl)}", "",
             "| target | method | held-out rho (fold 0 / fold 1 / mean) | pooled held-out rho [95% CI over images] | chosen (fold 0 / fold 1) |",
             "|---|---|---|---|---|"]
    report = {}
    for target, col in RATING[args.model].items():
        obj = lambda d, s, c=col: corr_objective(d, s, c)
        tbl["_t"] = tbl[col] - tbl[col].mean()      # the ridge target, centred across images
        rec = {}
        for name in ("single", "greedy", "ridge"):
            rhos, desc, held = [], [], []
            for k in (0, 1):
                cal, tst = tbl[tbl.fold != k], tbl[tbl.fold == k]
                if name == "ridge":
                    w, a = ridge(cal, feats, "_t", obj)
                    s = pd.Series(np.nan_to_num(tst[feats].values) @ w, index=tst.index)
                    top = np.argsort(-np.abs(w))[:3]
                    desc.append(", ".join(f"{feats[i].replace('sa_mean_', '').replace('ae_mean_', '').replace('_processor', '')}:{w[i]:.3f}" for i in top))
                else:
                    ch, _ = greedy(cal, feats, obj, 1 if name == "single" else args.max_k)
                    s = combo_score(tst, ch)
                    desc.append("+".join(f"{'-' if sg < 0 else ''}{f.replace('sa_mean_', '').replace('ae_mean_', '').replace('_processor', '')}" for f, sg in ch))
                rhos.append(float(obj(tst, s)))
                # held-out scores are rank-normalised within the fold so the two folds can be pooled
                held.append(pd.DataFrame({"s": s.rank(pct=True).values, "y": tst[col].values}))
            pooled = pd.concat(held)
            x, y = pooled.s.values, pooled.y.values
            r_all = float(spearmanr(x, y, nan_policy="omit").correlation)
            # 95% interval of the pooled rho: 1,000 bootstrap resamples of the images
            rng = np.random.RandomState(0)
            bs = [spearmanr(x[i], y[i], nan_policy="omit").correlation for i in (rng.randint(0, len(x), len(x)) for _ in range(1000))]
            lo, hi = float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
            rec[name] = dict(rho_fold0=rhos[0], rho_fold1=rhos[1], rho_mean=float(np.mean(rhos)), rho_pooled=r_all,
                             ci_lo=lo, ci_hi=hi, n=int(len(x)), chosen=desc)
            lines.append(f"| {target} | {name} | {rhos[0]:+.3f} / {rhos[1]:+.3f} / {np.mean(rhos):+.3f} | {r_all:+.3f} [{lo:+.3f}, {hi:+.3f}] | {' / '.join(desc)[:200]} |")
        report[target] = rec
    (out / "trajectory_correlation.md").write_text("\n".join(lines) + "\n")
    json.dump(report, open(out / "trajectory_correlation.json", "w"), indent=1)
    print("\n".join(lines))


# ----------------------------------------------------------------------------- loading a pool as a selection table
# (shared by the transfer mode below, by `selection --ca-stats`, by liveseed_live.py's calibrate and by liveseed_pool_table.py)
def _load_selection_table(run, scores, model, max_step, ca_stats=False, prefixes=None, zscore=True):
    """Returns (tbl, feats, metrics) for a pool.

    tbl      one row per (image_id, seed) of prompts with at least two seeds: the evaluator columns of `scores` and one
             column per reading at steps 1..max_step, named "<statistic>_<block>@s<step>".
    feats    the reading columns. They are z-scored within each prompt unless zscore=False (raw readings).
    metrics  the evaluator columns present.
    prefixes selects the statistic; the default is the paper's reading (ae_mean_ on the transformers, where it is the
             joint attention of the MMDiTs, the multimodal diffusion transformers, or Hunyuan-DiT's self-attention;
             sa_mean_ on the U-Nets). ca_stats adds the cross-attention token statistics of `ca_wide`.
    """
    from seed_select_eval import METRICS
    run = Path(run)
    res = _read_results(run).drop_duplicates(subset=["image_id", "seed", "step"], keep="last")
    res["image_id"] = res["image_id"].astype(str)
    sc = pd.read_csv(scores).drop_duplicates(subset=["image_id", "seed"], keep="last")
    sc["image_id"] = _norm_id(sc["image_id"])
    res["image_id"] = _norm_id(res["image_id"])
    res = res[res.step >= 1]                     # damaged lines can leave a spurious step 0
    metrics = [m for m in METRICS if m in sc.columns]
    if prefixes is None:
        prefixes = ("ae_mean_",) if model.startswith(("sd3", "flux", "hunyuan", "qwen")) else ("sa_mean_",)
    wide = reading_wide(res, max_step=max_step, prefixes=prefixes)
    if ca_stats:
        keys = list(wide[["image_id", "seed"]].itertuples(index=False, name=None))
        caw = ca_wide(run, keys, max_step, model)
        if len(caw):
            caw["seed"] = caw["seed"].astype(wide["seed"].dtype)
            wide = wide.merge(caw, on=["image_id", "seed"], how="left")
    feats = [c for c in wide.columns if "@s" in c]
    tbl = sc[["image_id", "seed"] + metrics].merge(wide, on=["image_id", "seed"], how="inner")
    tbl = tbl[tbl.groupby("image_id").seed.transform("size") >= 2].reset_index(drop=True)
    if zscore:
        tbl[feats] = zscore_within(tbl, feats)
    return tbl, feats, metrics


def _norm_id(col):
    """Image ids as canonical strings: '1.0' (a float-typed column from a damaged csv) and '1' must match."""
    num = pd.to_numeric(col, errors="coerce")
    out = col.astype(str)
    ok = num.notna() & (num == num.round())
    out[ok] = num[ok].astype(int).astype(str)
    return out


def ca_wide(run, keys, max_step, model):
    """Per-layer, per-step cross-attention token statistics from the hook's npz files, one row per (image_id, seed):
    masked entropy over the content tokens (the cross-attention entropy), entropy with the special tokens included
    (the memorisation quantity), <BOS> mass, top-1 content-token mass, subject-token mass; for SD3.5 also the
    mass on the T5 tokens. Columns are named ca_<stat>_<layer>@sK so they mix with the sa_/ae_ readings.

    Each npz holds vec[step, layer, token], the attention mass on every text key. keys: the (image_id, seed) pairs wanted.
    """
    stats_dir = Path(run) / "stats"
    recs = []
    for pid, seed in keys:
        hits = list(stats_dir.glob(f"*_{pid}_seed{seed}.npz"))
        if not hits:
            continue
        z = np.load(hits[0], allow_pickle=True)
        steps, vec = z["steps"], z["vec"]
        layers = [str(l).replace(".attn2.processor", "").replace(".attn.processor", "").replace("_processor", "") for l in z["layers"]]
        if vec.ndim != 3:
            continue
        # Where the special tokens sit depends on the text encoder: CLIP has <BOS> at 0; T5 (FLUX) has none, its content
        # starts at 0 and its EOS follows the last content token.
        from token_layout import layout, body_indices
        L = layout(z)
        n_active, T = L["n_active"], L["T"]
        body = body_indices(z)          # the subject-word tokens stored by the harness
        content = L["content"]          # the prompt's own tokens, without special tokens and padding

        def H(p):
            """Normalised Shannon entropy of a mass vector, renormalised to sum to 1."""
            p = p / max(float(p.sum()), 1e-8)
            p = np.clip(p, 1e-8, None)
            return float(-(p * np.log(p)).sum() / np.log(max(len(p), 2)))

        rec = dict(image_id=str(pid), seed=int(seed))
        for si, s in enumerate(steps):
            if int(s) > max_step:
                continue
            for li, lname in enumerate(layers):
                V = vec[si, li]
                V = V / max(float(V.sum()), 1e-8)
                rec[f"ca_entm_{lname}@s{int(s)}"] = H(V[content]) if content else np.nan
                rec[f"ca_entf_{lname}@s{int(s)}"] = H(V)
                rec[f"ca_bos_{lname}@s{int(s)}"] = float(V[L["bos"]]) if L["bos"] is not None else np.nan
                rec[f"ca_eos_{lname}@s{int(s)}"] = float(V[L["eos"]])
                rec[f"ca_top1_{lname}@s{int(s)}"] = float(V[content].max()) if content else np.nan
                rec[f"ca_body_{lname}@s{int(s)}"] = float(V[body].sum()) if body else np.nan
                if model.startswith("sd3") and T > 77:      # SD3.5: the T5 keys follow the 77 CLIP keys
                    rec[f"ca_t5_{lname}@s{int(s)}"] = float(V[77:].sum())
        recs.append(rec)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- mode: transfer (fit on one prompt set, apply frozen to another)
def run_transfer(args):
    """Choose the readings (and, for ridge, the weights) on EVERY prompt of --run, freeze them, and score EVERY
    prompt of --test-run: no re-tuning on the target prompt set at all. Prompt sets are disjoint by construction."""
    A, fA, mA = _load_selection_table(args.run, args.scores, args.model, args.max_step)
    B, fB, mB = _load_selection_table(args.test_run, args.test_scores, args.model, args.max_step)
    feats = [f for f in fA if f in set(fB)]         # readings logged in both pools
    metrics = [m for m in mA if m in mB]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name_a, name_b = Path(args.run).name, Path(args.test_run).name
    lines = [f"# Trajectory metric, transfer: {args.model}, chosen on {name_a} (all {A.image_id.nunique()} prompts), applied frozen to {name_b} ({B.image_id.nunique()} prompts); candidates = {len(feats)}", "",
             "| metric | method | top-1 gain on the target set [95% CI over prompts] | gap closed | win-rate | chosen on the source set |",
             "|---|---|---|---|---|---|"]
    report = {}
    for metric in metrics:
        A["_t"] = A[metric] - A.groupby("image_id")[metric].transform("mean")
        obj = lambda d, s, m=metric: sel_objective(d, s, m)
        rows = []
        # chosen on the source set A ...
        ch1, _ = greedy(A, feats, obj, 1)
        chg, _ = greedy(A, feats, obj, args.max_k)
        w, a = ridge(A, feats, "_t", obj)
        # ... and applied unchanged to the target set B
        scores_b = {"single": combo_score(B, ch1), "greedy": combo_score(B, chg),
                    "ridge": pd.Series(np.nan_to_num(B[feats].values) @ w, index=B.index)}
        desc = {"single": "+".join(f"{'-' if s < 0 else ''}{f.replace('sa_mean_', '').replace('ae_mean_', '').replace('_processor', '')}" for f, s in ch1),
                "greedy": "+".join(f"{'-' if s < 0 else ''}{f.replace('sa_mean_', '').replace('ae_mean_', '').replace('_processor', '')}" for f, s in chg),
                "ridge": f"alpha={a}; " + ", ".join(f"{feats[i].replace('sa_mean_', '').replace('ae_mean_', '').replace('_processor', '')}:{w[i]:+.4f}" for i in np.argsort(-np.abs(w))[:5])}
        oracle = B.groupby("image_id")[metric].max() - B.groupby("image_id")[metric].mean()
        for nm in ("single", "greedy", "ridge"):
            g = obj(B, scores_b[nm])
            lo, hi = bootstrap_ci(g.values)
            gap = float(np.nanmean(g / oracle.loc[g.index].replace(0, np.nan)))
            rows.append(dict(metric=metric, method=nm, gain=float(g.mean()), ci_lo=lo, ci_hi=hi, gap_closed=gap,
                             win_rate=float((g > 0).mean()), n=int(len(g)), chosen=desc[nm]))
            lines.append(f"| {metric} | {nm} | {g.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {gap:+.2f} | {(g > 0).mean():.2f} | {desc[nm][:160]} |")
        report[metric] = rows
    (out / "trajectory_transfer.md").write_text("\n".join(lines) + "\n")
    json.dump(report, open(out / "trajectory_transfer.json", "w"), indent=1, default=str)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selection", "correlation", "transfer"])
    ap.add_argument("--run", required=True)         # pool folder written by seed_harness.py
    ap.add_argument("--scores")                     # score_images.py output for the pool's finished images (selection, transfer)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-run")                   # transfer: the target pool and its scores
    ap.add_argument("--test-scores")
    ap.add_argument("--ca-stats", action="store_true", help="add per-layer cross-attention token statistics (from the npz stats) to the candidates")
    ap.add_argument("--ca-only", action="store_true", help="with --ca-stats: use only the cross-attention statistics")
    # correlation: the mean human ratings; the default is the expert study's file of this release
    ap.add_argument("--human", default=str(Path(__file__).resolve().parent / "../../human_studies/expert/expert_means_as_analysed.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-k", type=int, default=5)         # size of the greedy subset
    ap.add_argument("--max-step", type=int, default=12)     # readings at steps <= this are candidates
    args = ap.parse_args()
    {"selection": run_selection, "correlation": run_correlation, "transfer": run_transfer}[args.mode](args)


if __name__ == "__main__":
    main()
