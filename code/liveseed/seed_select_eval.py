#!/usr/bin/env python
"""seed_select_eval.py: offline evaluation of seed-selection rules on one pool (Sec. 4.1, App. C).

A pool is a set of prompts with N seeds each, generated to completion by seed_harness.py with the attention statistics
logged, and scored by score_images.py. Every rule sees the same N candidates per prompt and picks its top-1 (and
top-k); the quality of the picked image is read from the evaluator scores. Because all N images were finished, every
rule is evaluated on identical images and the comparison is exact, not sampled.

Fixed rules (one row each in summary.csv)
  random                 expected value: the pool mean
  reading_<t>_s<k>       the attention-entropy reading at a model's hotspot block and step k, with its sign
                         (HOTSPOTS below; <t> names the rating the hotspot was found for)
  reading_<t>_traj_*     variants on the same block: mean over steps 1 to max(5, k), and the slope of a line fitted to them
  reading_<t>_banddist   distance of the hotspot reading from its mean over the whole pool, in standard deviations
  abss_s<k>              an approximation of ABSS (Zhang, Liu & Hong, NeurIPS 2026): cross-attention mass on the prompt's
                         subject tokens, mean over layers, at step k; with and without <BOS>/<EOS> in the token set
  ca_ent_masked_s<k>, ca_ent_full_s<k>
                         cross-attention entropy over the content tokens only, and with special tokens and padding
  preview_<m>_s<k>       evaluator m on the decoded preview of the predicted clean image at step k, in the style of
                         Progressive Seed Pruning (PSP); needs --preview-scores
Cross-validated rule families (sweep_summary.csv): within a family every (statistic, step, sign) is a candidate; the
best one on half of the prompts is scored on the other half and the reverse (`cv_sweep`).

Input:   --run            the pool folder: results*.csv (one row per image_id, seed, step; columns sa_mean_<block> for
                          self-attention entropy and ae_mean_<block> for joint- or cross-attention entropy) and, if
                          present, stats/*.npz (cross-attention mass per text token)
         --scores         score_images.py output for the finished images (image_id, seed, evaluator columns)
         --preview-scores score_images.py output for the previews (image_id, seed, step, evaluator columns), optional
Output (--out):
         candidates.csv   one row per (prompt, seed): evaluator scores and every rule's feature
         summary.csv/.md  per rule and evaluator: top1_mean, topk_mean, random_mean, oracle_mean, gain_top1 with a 95%
                          bootstrap interval over prompts, oracle_gap_closed, win_rate_vs_random
         sweep_summary.csv, sweep_<family>_<model>_<metric>.csv (and _top10.md), picks_cv_<family>_<model>_<metric>.csv,
         top3_cv_<family>_<model>_<metric>.json   the cross-validated families

Usage:
  python seed_select_eval.py --run runs/sd14_pool --scores runs/sd14_pool/scores_finals.csv --model sd14 --out runs/sd14_eval
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Evaluator columns looked for in the scores file. vqascore and soft_tifa_* are present only where a separate scoring
# run has been merged into it.
METRICS = ["imagereward", "hpsv2", "pickscore", "clip_l14", "vqascore", "soft_tifa_gm", "soft_tifa_am", "tifa"]

# Hotspots: for each model and rating, the single reading that correlates best with it (App. F). PA = prompt alignment,
# IQ = image quality. Each entry is (column name in results.csv, step, sign); sign +1 means higher is better, -1 lower.
HOTSPOTS = {
    "sdxl": dict(pa=("sa_mean_down_blocks_1_attentions_0_transformer_blocks_0_attn1_processor", 2, +1),
                 iq=("sa_mean_up_blocks_1_attentions_1_transformer_blocks_1_attn1_processor", 12, +1)),
    "sd35": dict(pa=("ae_mean_transformer_blocks_3_attn_processor", 1, -1),
                 iq=("ae_mean_transformer_blocks_16_attn_processor", 12, +1)),
    # SD1.x/2.x have no hotspot from human ratings: the CV sweep finds the best (layer, step)
    # held-out, and 'pa_xfer' reads the SDXL prompt-alignment hotspot transferred verbatim
    # (same block name, same step, same sign) as an untuned reference.
    "sd14": dict(pa_xfer=("sa_mean_down_blocks_1_attentions_0_transformer_blocks_0_attn1_processor", 2, +1),
                 # SD1.4's own hotspots: the best single reading by |Spearman rho| against an automatic judge's
                 # alignment and quality grades of 400 SD1.4 images; and the SDXL block at step 1, where it correlates on SD1.4.
                 pa_own=("sa_mean_up_blocks_2_attentions_2_transformer_blocks_0_attn1_processor", 10, +1),
                 iq_own=("sa_mean_up_blocks_1_attentions_2_transformer_blocks_0_attn1_processor", 7, +1),
                 pa_xfer1=("sa_mean_down_blocks_1_attentions_0_transformer_blocks_0_attn1_processor", 1, +1)),
    "sd15": dict(pa_xfer=("sa_mean_down_blocks_1_attentions_0_transformer_blocks_0_attn1_processor", 2, +1)),
    "sd21": dict(pa_xfer=("sa_mean_down_blocks_1_attentions_0_transformer_blocks_0_attn1_processor", 2, +1)),
}


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


def load_npz_feats(stats_dir: Path, ids_seeds, abss_steps, include_special: bool):
    """Cross-attention token statistics per (image_id, seed, step), from the harness's stats/*.npz.

    Each npz holds vec[step, layer, token], the attention mass on every text token. Returned columns:
      abss_mass      mass on the subject ("body") tokens, plus the special tokens if include_special
      ca_ent_masked  normalised entropy over the content tokens only
      ca_ent_full    normalised entropy over all positions (special tokens and padding included)
      bos_mass       mass on <BOS> (NaN for an encoder without one)
    Every logged step is returned; `abss_steps` is not used.
    """
    recs = []
    for pid, seed in ids_seeds:
        f = stats_dir / f"*_{pid}_seed{seed}.npz"
        hits = list(stats_dir.glob(f.name))
        if not hits:
            continue
        z = np.load(hits[0], allow_pickle=True)
        steps, vec = z["steps"], z["vec"]            # vec: [S, L, T]
        if vec.ndim != 3:
            continue
        # Where the special tokens sit depends on the text encoder: CLIP has <BOS> at 0; T5 (FLUX) has none, its content
        # starts at 0 and its EOS follows the last content token.
        from token_layout import layout, body_indices
        L = layout(z)
        T = L["T"]
        body = sorted(set(body_indices(z) + (L["special"] if include_special else [])))
        content = L["content"]
        for si, s in enumerate(steps):
            V = vec[si].mean(axis=0)                 # mean over layers -> (T,)
            V = V / max(V.sum(), 1e-8)
            rec = dict(image_id=str(pid), seed=int(seed), step=int(s))
            rec["abss_mass"] = float(V[body].sum()) if body else np.nan

            # ablation of the cross-attention entropy: with and without the special tokens
            def H(p):
                """Normalised Shannon entropy of a mass vector, renormalised to sum to 1."""
                p = p / max(p.sum(), 1e-8)
                p = np.clip(p, 1e-8, None)
                return float(-(p * np.log(p)).sum() / np.log(max(len(p), 2)))

            rec["ca_ent_masked"] = H(V[content]) if content else np.nan
            rec["ca_ent_full"] = H(V)
            rec["bos_mass"] = float(V[L["bos"]]) if L["bos"] is not None else np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- cross-validated choice of a rule
def cv_sweep(tbl: pd.DataFrame, feats, metric: str, out_dir: Path, tag: str, family: str):
    """Generic 2-fold prompt-level cross-validated rule selection.

    tbl: one row per (image_id, seed) with the evaluation metric and candidate feature columns.
    Every feature (both signs) is a candidate rule. The best candidate by mean top-1 gain over
    random on fold A is evaluated on fold B and vice versa, so every prompt is scored exactly once
    by a rule chosen without it. Returns the held-out mean gain with a bootstrap CI over prompts,
    the fraction of the oracle gap closed and the win-rate, plus the chosen rule per fold. Also
    writes the full sweep table (a selection-gain analogue of the correlation heat-maps), the held-out
    picks and the held-out gain of the mean of the top three.
    """
    tbl = tbl.copy()
    prompts = sorted(tbl.image_id.unique())
    fold_of = {p: (i % 2) for i, p in enumerate(prompts)}        # alternate prompts in sorted order
    grp = tbl.groupby("image_id")[metric]
    tbl["_gain_base"] = tbl[metric] - grp.transform("mean")      # a candidate's gain over its prompt's mean
    tbl["_gap"] = grp.transform("max") - grp.transform("mean")   # the oracle gap of its prompt

    def per_prompt(df, f, sign):
        """The top-1 candidate of each prompt under sign * feature: its gain and its prompt's oracle gap."""
        key = sign * df[f]
        idx = key.groupby(df.image_id).idxmax().dropna()
        return df.loc[idx.values, ["image_id", "_gain_base", "_gap"]].set_index("image_id")

    # the sweep: mean gain of every candidate rule on each fold and on all prompts
    rows, cache = [], {}
    for f in feats:
        if f not in tbl or tbl[f].isna().mean() > 0.5:
            continue
        for sign in (+1, -1):
            g = per_prompt(tbl, f, sign)
            cache[(f, sign)] = g
            if len(g) == 0:
                continue
            in0 = np.array([fold_of[p] == 0 for p in g.index], dtype=bool)
            rows.append(dict(feature=f, sign=sign, gain_fold0=g._gain_base[in0].mean(), gain_fold1=g._gain_base[~in0].mean(),
                             gain_all=g._gain_base.mean(), n=len(g)))
    sw = pd.DataFrame(rows)
    if sw.empty:
        return None
    sw.to_csv(out_dir / f"sweep_{family}_{tag}_{metric}.csv", index=False)
    bestA = sw.sort_values("gain_fold0", ascending=False).iloc[0]   # chosen on fold 0 -> evaluated on fold 1
    bestB = sw.sort_values("gain_fold1", ascending=False).iloc[0]   # chosen on fold 1 -> evaluated on fold 0
    gA = cache[(bestA.feature, bestA.sign)]
    gA = gA[[fold_of[p] == 1 for p in gA.index]]
    gB = cache[(bestB.feature, bestB.sign)]
    gB = gB[[fold_of[p] == 0 for p in gB.index]]
    held = pd.concat([gA, gB])
    gains = held._gain_base.values
    # held-out picks: the seed chosen for each prompt by the rule selected on the other fold
    picks = []
    for best, fold_test in ((bestA, 1), (bestB, 0)):
        sub = tbl[tbl.image_id.map(fold_of) == fold_test]
        key = best.sign * sub[best.feature]
        idx = key.groupby(sub.image_id).idxmax().dropna()
        picks.append(sub.loc[idx.values, ["image_id", "seed"]])
    picks = pd.concat(picks)
    picks.to_csv(out_dir / f"picks_cv_{family}_{tag}_{metric}.csv", index=False)
    # held-out top-3 mean under the same chosen rules (ABSS's main protocol keeps the top 3 of 10 and averages them)
    t3 = []
    for best, fold_test in ((bestA, 1), (bestB, 0)):
        sub = tbl[tbl.image_id.map(fold_of) == fold_test].assign(_k=lambda d: best.sign * d[best.feature])
        t3.append(sub.sort_values("_k", ascending=False).groupby("image_id").head(3).groupby("image_id")["_gain_base"].mean())
    t3 = pd.concat(t3)
    t3lo, t3hi = bootstrap_ci(t3.values)
    import json as _json
    _json.dump(dict(family=family, metric=metric, top3_gain=float(t3.mean()), ci_lo=t3lo, ci_hi=t3hi, n=int(len(t3))),
               open(out_dir / f"top3_cv_{family}_{tag}_{metric}.json", "w"))
    lo, hi = bootstrap_ci(gains)
    gap = float(np.nanmean(held._gain_base / held._gap.replace(0, np.nan)))   # prompts with a zero oracle gap are left out
    sgn = lambda r: '+' if r.sign > 0 else '-'
    chosen = f"{bestA.feature}({sgn(bestA)}) | {bestB.feature}({sgn(bestB)})"
    top = sw.sort_values("gain_all", ascending=False).head(10)
    (out_dir / f"sweep_{family}_{tag}_{metric}_top10.md").write_text(
        "| feature | sign | gain (all prompts, in-sample) |\n|---|---|---|\n" +
        "\n".join(f"| {r.feature} | {sgn(r)} | {r.gain_all:+.4f} |" for _, r in top.iterrows()) + "\n")
    return dict(rule=f"{family}_sweep2cv", metric=metric, family=family, n_prompts=len(gains), cv_gain=float(gains.mean()),
                cv_ci_lo=lo, cv_ci_hi=hi, oracle_gap_closed=gap, win_rate_vs_random=float((gains > 0).mean()),
                chosen=chosen, n_features=len(feats))


# ----------------------------------------------------------------------------- tables with one column per (statistic, step)
def reading_wide(res: pd.DataFrame, max_step: int, prefixes=("sa_mean_", "ae_mean_")) -> pd.DataFrame:
    """Every hook statistic at every logged step as a column 'stat@sK', one row per (image_id, seed)."""
    stat_cols = [c for c in res.columns if c.startswith(prefixes)]
    if not stat_cols:
        return pd.DataFrame(columns=["image_id", "seed"])
    sub = res[res.step <= max_step][["image_id", "seed", "step"] + stat_cols]
    wide = sub.pivot_table(index=["image_id", "seed"], columns="step", values=stat_cols)
    wide.columns = [f"{c}@s{int(st)}" for c, st in wide.columns]
    return wide.reset_index()


def npz_wide(nz: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Per-step npz statistics as columns 'name@sK' (cols maps source column -> short name)."""
    if nz.empty:
        return pd.DataFrame(columns=["image_id", "seed"])
    wide = nz.pivot_table(index=["image_id", "seed"], columns="step", values=list(cols))
    wide.columns = [f"{cols[c]}@s{int(st)}" for c, st in wide.columns]
    return wide.reset_index()


# ----------------------------------------------------------------------------- evaluation helpers
def bootstrap_ci(x, n=2000, seed=0):
    """95% percentile bootstrap interval of the mean of x (NaNs dropped; n resamples, fixed seed): (lo, hi)."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def evaluate_rule(pool: pd.DataFrame, score_col: str, higher: bool, k: int, metrics):
    """Applies one rule to one prompt's pool.

    pool: rows for one prompt (one per seed) with score_col and metric columns. Returns, per metric m, m_top1 (the
    metric of the rule's first choice), m_topk (mean over its first k), m_random (pool mean) and m_oracle (pool
    maximum); None if the rule's score is missing on every row.
    """
    out = {}
    if pool[score_col].isna().all():
        return None
    order = pool.sort_values(score_col, ascending=not higher)
    for m in metrics:
        if m not in pool:
            continue
        out[f"{m}_top1"] = float(order[m].iloc[0])
        out[f"{m}_topk"] = float(order[m].iloc[:k].mean())
        out[f"{m}_random"] = float(pool[m].mean())
        out[f"{m}_oracle"] = float(pool[m].max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pool folder written by seed_harness.py (results*.csv, stats/)")
    ap.add_argument("--scores", required=True, help="score_images.py output for the FINAL images")
    ap.add_argument("--preview-scores", default=None, help="score_images.py output for previews (optional)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--abss-step", type=int, default=6,
                    help="step of the ABSS-style statistic: 6 of 30, the counterpart of ABSS's step 10 of 50")
    ap.add_argument("--first-n-seeds", type=int, default=0,
                    help="restrict each prompt's pool to its first n seeds in the order they were logged (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # a (prompt, seed, step) logged twice (a requeued job repeats the pair it was interrupted in) keeps its last row
    res = _read_results(run).drop_duplicates(subset=["image_id", "seed", "step"], keep="last")
    sc = pd.read_csv(args.scores)
    sc["image_id"] = sc["image_id"].astype(str)
    sc["seed"] = sc["seed"].astype(int)
    res["image_id"] = res["image_id"].astype(str)
    metrics = [m for m in METRICS if m in sc.columns]
    if args.first_n_seeds:
        # step 1 exists for every candidate; damaged lines can leave a spurious step 0
        order = res[res.step == 1].drop_duplicates(["image_id", "seed"])[["image_id", "seed"]]
        keep = order.groupby("image_id").head(args.first_n_seeds)
        sc = sc.merge(keep, on=["image_id", "seed"], how="inner")
        res = res.merge(keep, on=["image_id", "seed"], how="inner")
        print(f"[eval] restricted to the first {args.first_n_seeds} seeds per prompt: {len(keep)} candidates")

    # ---- candidate table: one row per (prompt, seed) with final-image scores
    sc = sc[sc[metrics].notna().any(axis=1)] if metrics else sc          # drop unscored rows
    finals = sc[[c for c in ["image_id", "seed", "prompt", "dataset", "category"] if c in sc] + metrics].drop_duplicates(
        ["image_id", "seed"], keep="last")

    # ---- intrinsic features per (prompt, seed): hotspot stats at chosen steps + trajectory variants
    feat = finals[["image_id", "seed"]].copy()
    hs = HOTSPOTS.get(args.model, {})
    stat_cols = [c for c in res.columns if c.startswith(("sa_mean_", "ae_mean_"))]
    rules = []  # (name, column, higher_better)

    def add_step_feature(col, step, name):
        """Adds column `name` to feat: statistic `col` at `step`."""
        sub = res[res.step == step][["image_id", "seed", col]].rename(columns={col: name})
        nonlocal feat
        feat = feat.merge(sub, on=["image_id", "seed"], how="left")

    for target, (col, step, sign) in hs.items():
        if col not in res.columns:
            print(f"[eval] hotspot column missing: {col}")
            continue
        add_step_feature(col, step, f"reading_{target}_s{step}")
        rules.append((f"reading_{target}_s{step}", f"reading_{target}_s{step}", sign > 0))
        # trajectory variants on the same layer
        traj = res[res.step <= max(5, step)][["image_id", "seed", "step", col]].pivot_table(
            index=["image_id", "seed"], columns="step", values=col)
        traj = traj.sort_index(axis=1)
        mean_name, slope_name = f"reading_{target}_traj_mean1-{traj.columns.max()}", f"reading_{target}_traj_slope"
        feat = feat.merge(traj.mean(axis=1).rename(mean_name).reset_index(), on=["image_id", "seed"], how="left")
        xs = np.array(traj.columns, float)
        slope = traj.apply(lambda r: np.polyfit(xs[~r.isna()], r.dropna().values, 1)[0] if r.notna().sum() >= 2 else np.nan, axis=1)
        feat = feat.merge(slope.rename(slope_name).reset_index(), on=["image_id", "seed"], how="left")
        rules += [(mean_name, mean_name, sign > 0), (slope_name, slope_name, sign > 0)]
        # two-tailed band distance: |z| of the reading against its mean and spread over the whole pool (lower is better)
        band = f"reading_{target}_banddist"
        v = feat[f"reading_{target}_s{step}"]
        feat[band] = ((v - v.mean()) / (v.std() + 1e-8)).abs()
        rules.append((band, band, False))

    # The rules above are fixed in advance. Rules chosen on this pool (the best block, step and sign) are evaluated
    # with cross-validation further down, which protects against optimistic selection.

    # ---- ABSS-style subject-token ("body") mass, with/without BOS+EOS, plus the cross-attention (CA) entropy masking ablation
    ids_seeds = list(zip(feat.image_id, feat.seed))
    nz_all = {}
    for inc in (False, True):
        nz = load_npz_feats(run / "stats", ids_seeds, [args.abss_step], include_special=inc)
        nz_all[inc] = nz
        if nz.empty:
            continue
        s = nz[nz.step == args.abss_step].drop(columns=["step"])
        suffix = "_withBOSEOS" if inc else ""
        s = s.rename(columns={"abss_mass": f"abss_s{args.abss_step}{suffix}",
                              "ca_ent_masked": f"ca_ent_masked_s{args.abss_step}",
                              "ca_ent_full": f"ca_ent_full_s{args.abss_step}",
                              "bos_mass": f"bos_mass_s{args.abss_step}"})
        keep = ["image_id", "seed", f"abss_s{args.abss_step}{suffix}"] + (
            [f"ca_ent_masked_s{args.abss_step}", f"ca_ent_full_s{args.abss_step}", f"bos_mass_s{args.abss_step}"] if not inc else [])
        feat = feat.merge(s[keep], on=["image_id", "seed"], how="left")
        rules.append((f"abss_s{args.abss_step}{suffix}", f"abss_s{args.abss_step}{suffix}", True))
        if not inc:
            rules += [(f"ca_ent_masked_s{args.abss_step}", f"ca_ent_masked_s{args.abss_step}", False),
                      (f"ca_ent_full_s{args.abss_step}", f"ca_ent_full_s{args.abss_step}", False)]

    # ---- PSP-style: evaluator on the x0 preview
    if args.preview_scores:
        pv = pd.read_csv(args.preview_scores, on_bad_lines="skip")
        for c in ("seed", "step"):
            pv[c] = pd.to_numeric(pv[c], errors="coerce")
        pv = pv[pv.step.notna() & (pv.step == pv.step.round()) & pv.seed.notna()].copy()   # damaged lines from concurrent writers
        pv["seed"] = pv["seed"].astype(int)
        pv["step"] = pv["step"].astype(int)
        num = pd.to_numeric(pv["image_id"], errors="coerce")                              # '349.0' and '349' must match the finals
        pv["image_id"] = num.astype("Int64").astype(str).where(num.notna(), pv["image_id"].astype(str))
        pv = pv.drop_duplicates(subset=["image_id", "seed", "step"], keep="last")   # several jobs may have scored one pool folder
        for step in sorted(pv.step.unique()):
            for m in metrics:
                if m in pv:
                    nm = f"preview_{m}_s{int(step)}"
                    sub = pv[pv.step == step][["image_id", "seed", m]].rename(columns={m: nm})
                    feat = feat.merge(sub, on=["image_id", "seed"], how="left")
                    rules.append((nm, nm, True))

    table = finals.merge(feat, on=["image_id", "seed"], how="left")
    table.to_csv(out / "candidates.csv", index=False)

    # ---- evaluate every rule per prompt
    records = []
    for name, col, higher in [("random", None, True)] + rules:
        per_prompt = []
        for pid, pool in table.groupby("image_id"):
            if len(pool) < 2:
                continue
            if col is None:     # random: every quantity is the pool mean, the expected value of a random draw
                r = {f"{m}_top1": float(pool[m].mean()) for m in metrics}
                r.update({f"{m}_topk": float(pool[m].mean()) for m in metrics})
                r.update({f"{m}_random": float(pool[m].mean()) for m in metrics})
                r.update({f"{m}_oracle": float(pool[m].max()) for m in metrics})
            else:
                r = evaluate_rule(pool, col, higher, args.k, metrics)
                if r is None:
                    continue
            r["image_id"] = pid
            per_prompt.append(r)
        if not per_prompt:
            continue
        pp = pd.DataFrame(per_prompt)
        for m in metrics:
            gain = pp[f"{m}_top1"] - pp[f"{m}_random"]
            # fraction of the oracle gap closed, per prompt; prompts with a zero gap are left out of its mean
            gap = (pp[f"{m}_top1"] - pp[f"{m}_random"]) / (pp[f"{m}_oracle"] - pp[f"{m}_random"]).replace(0, np.nan)
            lo, hi = bootstrap_ci(gain)
            records.append(dict(rule=name, metric=m, n_prompts=len(pp),
                                top1_mean=pp[f"{m}_top1"].mean(), topk_mean=pp[f"{m}_topk"].mean(),
                                random_mean=pp[f"{m}_random"].mean(), oracle_mean=pp[f"{m}_oracle"].mean(),
                                gain_top1=gain.mean(), gain_ci_lo=lo, gain_ci_hi=hi,
                                oracle_gap_closed=float(np.nanmean(gap)),
                                win_rate_vs_random=float((gain > 0).mean())))
    summary = pd.DataFrame(records)

    # ---- cross-validated rule families (2-fold over prompts), per metric:
    #      reading = every self-attention entropy x step (both signs); caent = every per-layer cross-attention
    #      entropy x step; abss = body-token mass x step x token set;
    #      camap = CA entropy (masked / with special tokens) and BOS mass x step.
    sweep_rows = []
    # On the transformers (SD3.5, FLUX, Hunyuan-DiT, Qwen-Image) the attention-entropy reading is logged under the 'ae_mean_'
    # prefix: joint attention on the MMDiTs (multimodal diffusion transformers), self-attention on Hunyuan-DiT.
    # `is_sd3` is true for all of them.
    is_sd3 = args.model.startswith(("sd3", "flux", "hunyuan", "qwen"))
    fam = {"reading": reading_wide(res, max_step=12, prefixes=("ae_mean_",) if is_sd3 else ("sa_mean_",))}
    if not is_sd3:
        fam["caent"] = reading_wide(res, max_step=12, prefixes=("ae_mean_",))   # on the U-Nets ae_mean_ is the per-layer cross-attention entropy
    if not nz_all.get(False, pd.DataFrame()).empty:
        w = npz_wide(nz_all[False], {"abss_mass": "abss", "ca_ent_masked": "caentM", "ca_ent_full": "caentF", "bos_mass": "bos"})
        if not nz_all.get(True, pd.DataFrame()).empty:
            w = w.merge(npz_wide(nz_all[True], {"abss_mass": "abssBOS"}), on=["image_id", "seed"], how="outer")
        fam["abss"] = w[["image_id", "seed"] + [c for c in w.columns if c.startswith(("abss@", "abssBOS@"))]]
        fam["camap"] = w[["image_id", "seed"] + [c for c in w.columns if c.startswith(("caentM@", "caentF@", "bos@"))]]
    for family, wide in fam.items():
        feats_f = [c for c in wide.columns if "@s" in c]
        if not feats_f:
            continue
        for m in metrics:
            tbl_f = finals[["image_id", "seed", m]].merge(wide, on=["image_id", "seed"], how="inner")
            r = cv_sweep(tbl_f, feats_f, m, out_dir=out, tag=args.model, family=family)
            if r:
                sweep_rows.append(r)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(out / "sweep_summary.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    # ---- Markdown report
    lines = [f"# Seed selection: {args.model} ({run.name}); k={args.k}; N per prompt = {table.groupby('image_id').size().median():.0f}", ""]
    for m in metrics:
        sub = summary[summary.metric == m].sort_values("gain_top1", ascending=False)
        lines += [f"## {m}", "", "| rule | top-1 | top-k | random | oracle | gain (95% CI) | gap closed | win-rate |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r.rule} | {r.top1_mean:.4f} | {r.topk_mean:.4f} | {r.random_mean:.4f} | {r.oracle_mean:.4f} | "
                         f"{r.gain_top1:+.4f} [{r.gain_ci_lo:+.4f}, {r.gain_ci_hi:+.4f}] | {r.oracle_gap_closed:+.2f} | {r.win_rate_vs_random:.2f} |")
        lines.append("")
    if sweep_rows:
        lines += ["## Cross-validated rules (2-fold over prompts; rule chosen on one half, every prompt scored held-out)", "",
                  "| family | metric | held-out gain (95% CI) | gap closed | win-rate | n | chosen per fold |", "|---|---|---|---|---|---|---|"]
        for r in sweep_rows:
            lines.append(f"| {r['family']} | {r['metric']} | {r['cv_gain']:+.4f} [{r['cv_ci_lo']:+.4f}, {r['cv_ci_hi']:+.4f}] | "
                         f"{r['oracle_gap_closed']:+.2f} | {r['win_rate_vs_random']:.2f} | {r['n_prompts']} | {r['chosen']} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
