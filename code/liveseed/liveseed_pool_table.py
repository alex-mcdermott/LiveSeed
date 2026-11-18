#!/usr/bin/env python
"""liveseed_pool_table.py: exports one pool as the single table that simulate_schedules.py reads (App. C).

Input:  --run, a pool folder of seed_harness.py holding results*.csv (the logged readings), scores_finals.csv
        (score_images.py on the finished images) and, if present, scores_previews.csv (score_images.py on the decoded
        previews) and scores_finals_vqa.csv / scores_vqa.csv / scores_hpsv3.csv (evaluators scored into separate files).
Output: --out (.parquet, or CSV for any other extension), one row per (prompt, seed):
  prompt_id, seed
  draw              the seed's rank in the harness's draw order for that prompt (-1 for a seed outside it)
  E__<block>__<t>   attention-entropy reading of <block> at step <t> (self-attention on the U-Nets, joint attention on the
                    transformers), steps 1..--max-step, z-scored within the pool unless --raw
                    (the simulator z-scores again over the survivors of each exit, which is unaffected by this)
  P__<eval>__<t>    preview score: evaluator <eval> on the predicted clean image decoded at step <t>
  S__<eval>         evaluator <eval> on the finished image
Evaluator names follow the simulator: hpsv21, imagereward, pickscore, clip, vqascore, hpsv3.
Only prompts with the full number of seeds are exported.

Usage:
  python liveseed_pool_table.py --run runs/sd14_pool --model sd14 --out runs/pool_sd14.parquet
"""
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_trajectory_metric import _load_selection_table, _norm_id   # noqa: E402

# column of score_images.py -> evaluator name in the simulator's schema
EV = {"hpsv2": "hpsv21", "imagereward": "imagereward", "pickscore": "pickscore", "clip_l14": "clip", "vqascore": "vqascore"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)                 # the pool folder
    ap.add_argument("--model", required=True)               # decides the default reading: sa_mean_ on U-Nets, ae_mean_ on transformers
    ap.add_argument("--max-step", type=int, default=12)     # readings of steps 1..max-step are exported
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="export the raw readings (not z-scored within the pool)")
    ap.add_argument("--prefix", default="sa_mean_|ae_mean_",
                    help="reading columns to export as E__ (regex alternatives). Other choices: sa_entfirst_|ae_avgfirst_ "
                         "(the other aggregation order), sa_cond_|ae_cond_ (conditional branch only)")
    a = ap.parse_args()
    run = Path(a.run)
    # one row per (prompt, seed): evaluator scores and one column per reading "<statistic>_<block>@s<step>"
    tbl, feats, metrics = _load_selection_table(str(run), str(run / "scores_finals.csv"), a.model, a.max_step,
                                                prefixes=tuple(a.prefix.split("|")) if a.prefix != "sa_mean_|ae_mean_" else None,
                                                zscore=not a.raw)
    tbl["image_id"] = tbl["image_id"].astype(str)
    tbl["seed"] = tbl["seed"].astype("int64")
    out = tbl[["image_id", "seed"]].rename(columns={"image_id": "prompt_id"}).copy()

    # Draw order: the rank of each seed in the sequence the harness derives from the prompt (seed_harness.seeds_for), so
    # that the simulator's sub-pools of N are the first N seeds drawn, i.e. the pool a run with N seeds would have made.
    # Seeds outside the sequence (for example a fixed seed added to a pool) get -1 and so come first.
    from seed_harness import seeds_for
    sc_ = pd.read_csv(run / "scores_finals.csv", usecols=lambda c: c in ("dataset", "image_id", "prompt", "seed")).drop_duplicates(["image_id", "seed"])
    sc_["prompt_id"] = _norm_id(sc_["image_id"]).astype(str)
    key = {r.prompt_id: f"{r.dataset}:{r.prompt_id}:{str(r.prompt).strip()}" for r in sc_.itertuples()}    # the harness's seed key
    nmax = int(out.groupby("prompt_id").size().max())
    rank = {}
    for pid, k in key.items():
        for i, s_ in enumerate(seeds_for(k, nmax)):
            rank[(pid, int(s_))] = i
    out["draw"] = [rank.get((p, int(s_)), -1) for p, s_ in zip(out.prompt_id, out.seed)]
    n_unranked = int((out["draw"] < 0).sum())
    if n_unranked:
        print(f"[pool] {n_unranked} candidates outside the derived seed sequence get draw = -1 (placed first)")

    # readings: 'sa_mean_<block>@s<step>' or 'ae_mean_<block>@s<step>' -> 'E__<block>__<step>'
    # (a double underscore inside a block name is collapsed, because '__' separates the fields of the new name)
    E = {}
    for f in feats:
        m = re.match(rf"(?:{a.prefix})(.+)@s(\d+)$", f)
        if not m:
            continue
        E[f"E__{m.group(1).replace('__', '_')}__{int(m.group(2))}"] = np.nan_to_num(tbl[f].to_numpy(float))
    out = pd.concat([out, pd.DataFrame(E, index=out.index)], axis=1)

    # finished-image scores
    for m in metrics:
        if m in EV:
            out[f"S__{EV[m]}"] = tbl[m].to_numpy(float)
    # held-out evaluators scored into separate files: VQAScore (scores_finals_vqa.csv or scores_vqa.csv) and HPSv3 (scores_hpsv3.csv)
    for names, colname, key in ((("scores_finals_vqa.csv", "scores_vqa.csv"), "vqa", "S__vqascore"), (("scores_hpsv3.csv",), "hpsv3", "S__hpsv3")):
        f = next((run / n for n in names if (run / n).exists()), None)
        if f is None:
            continue
        v = pd.read_csv(f, low_memory=False)
        v = v[pd.to_numeric(v["seed"], errors="coerce").notna()].drop_duplicates(subset=["image_id", "seed"], keep="last")
        col = next((c for c in v.columns if colname in c.lower()), None)
        if col:
            v["prompt_id"] = _norm_id(v["image_id"]).astype(str)
            v["seed"] = v["seed"].astype(float).astype("int64")
            out = out.merge(v[["prompt_id", "seed", col]].rename(columns={col: key}), on=["prompt_id", "seed"], how="left")

    # preview scores, one column per (evaluator, step)
    pv = run / "scores_previews.csv"
    if pv.exists():
        p = pd.read_csv(pv)
        p = p[pd.to_numeric(p["step"], errors="coerce").notna() & pd.to_numeric(p["seed"], errors="coerce").notna()].copy()   # damaged lines
        p["prompt_id"] = _norm_id(p["image_id"]).astype(str)
        p["seed"] = p["seed"].astype(float).astype("int64")
        p["step"] = p["step"].astype(float).astype(int)
        p = p.drop_duplicates(subset=["prompt_id", "seed", "step"], keep="last")
        for m, name in EV.items():
            if m not in p.columns:
                continue
            w = p.pivot_table(index=["prompt_id", "seed"], columns="step", values=m)
            w.columns = [f"P__{name}__{int(s)}" for s in w.columns]
            out = out.merge(w.reset_index(), on=["prompt_id", "seed"], how="left")

    sizes = out.groupby("prompt_id").size()
    out = out[out.prompt_id.isin(sizes[sizes == sizes.max()].index)].reset_index(drop=True)      # complete pools only
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out) if a.out.endswith(".parquet") else out.to_csv(a.out, index=False)
    nE = sum(c.startswith("E__") for c in out.columns)
    print(f"{a.out}: {out.prompt_id.nunique()} prompts x {int(sizes.max())} seeds, {nE} readings, "
          f"finished-image scores {[c[3:] for c in out.columns if c.startswith('S__')]}, "
          f"preview scores {sorted({c for c in out.columns if c.startswith('P__')})[:12]}")


if __name__ == "__main__":
    main()
