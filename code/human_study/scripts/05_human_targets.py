#!/usr/bin/env python
"""05_human_targets.py: turns the clean crowd answers into one human target per image.

Place in the pipeline: after 03_qc_and_payment.py, whose clean data it reads. The targets written here are what
LiveSeed is calibrated to and evaluated against in the crowd study (paper Sec. 4.3, App. E).

Targets are formed within a prompt, because the task is ranking seeds of the same prompt (paper Sec. 3.3): each
rater's ratings are z-scored within the grid they rated, which removes that rater's leniency and use of the scale, and
then averaged over the grid's raters. Check cards and images reported as not loading are dropped. Image quality (IQ)
and prompt alignment (PA) are kept as separate targets. The top-three choices give a third target, preference.

Reads   <results>/qc/clean_calibration.csv, <results>/qc/clean_selection.csv, data/manifests/image_blinding_map.csv
Writes  <out>/human_targets.csv   model, prompt_id, seed, z_iq, z_pa (z-scored within rater and grid), zs_iq, zs_pa
                                  (z-scored within a rater's session), iq, pa (raw 1-5), each the mean over the raters
        <out>/human_picks.csv     model, prompt_id, seed, pref_points (3, 2, 1 points for a rater's first, second and
                                  third choice, mean over raters), pref_first (share of raters who chose it first)
Usage:  python scripts/05_human_targets.py [--results data/results] [--out data/results/analysis]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import MAN, RESULTS


def zs(x):
    """Return x z-scored (population standard deviation); all zeros if x is constant, so a flat group carries no signal."""
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def human_targets(cal):
    """Return one row per rated image (model, prompt_id, seed) with the mean over its raters of six columns.

    cal: the rows of clean_calibration.csv.
      z_iq, z_pa    ratings z-scored within a rater and grid: differences between the seeds of one prompt only.
      zs_iq, zs_pa  ratings z-scored within a rater's whole session: differences between prompts are kept.
      iq, pa        the raw 1-5 ratings.
    """
    I = pd.read_csv(MAN / "image_blinding_map.csv")
    I = I[I.kind == "pool"][["img", "model", "prompt_id", "seed"]]
    c = cal[(cal.flagged.astype(int) == 0)].merge(I, on="img")                        # check cards are not 'pool' images: dropped here
    for v in ("pa", "iq"):
        c[v] = pd.to_numeric(c[v], errors="coerce")
        c["z_" + v] = c.groupby(["pid", "grid_id"])[v].transform(zs)                  # within a rater and prompt: seed differences only
        c["zs_" + v] = c.groupby(["pid", "batch_id"])[v].transform(zs)                # within a rater's session: keeps differences between prompts
    return c.groupby(["model", "prompt_id", "seed"], as_index=False)[["z_iq", "z_pa", "zs_iq", "zs_pa", "iq", "pa"]].mean()


def human_picks(sel):
    """Return the top-three preference of every rated image (model, prompt_id, seed, pref_points, pref_first).

    sel: the rows of clean_selection.csv. A rater's first, second and third choice score 3, 2 and 1 points and every other
    image 0; pref_points is the mean over the grid's raters, and pref_first the share of raters who chose the image first.
    """
    I = pd.read_csv(MAN / "image_blinding_map.csv")
    I = I[I.kind == "pool"][["img", "model", "prompt_id", "seed"]]
    # Long format: one row per (rater, choice page, rank).
    L = sel.melt(id_vars=["pid", "grid_id"], value_vars=["pick1", "pick2", "pick3"], var_name="rank", value_name="img").dropna(subset=["img"])
    L["points"] = L["rank"].map(dict(pick1=3, pick2=2, pick3=1))
    L["first"] = (L["rank"] == "pick1").astype(float)
    # The number of raters who chose on each grid, found through their first pick.
    n = sel.merge(I, left_on="pick1", right_on="img").groupby(["model", "prompt_id"]).pid.nunique().rename("raters").reset_index()
    P = L.merge(I, on="img").groupby(["model", "prompt_id", "seed"], as_index=False)[["points", "first"]].sum().merge(n, on=["model", "prompt_id"])
    P["pref_points"] = P.points / P.raters
    P["pref_first"] = P["first"] / P.raters
    # Images that nobody chose get zeros through the left merge from the image list.
    return I[["model", "prompt_id", "seed"]].merge(P[["model", "prompt_id", "seed", "pref_points", "pref_first"]], how="left", on=["model", "prompt_id", "seed"]).fillna(dict(pref_points=0.0, pref_first=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS), help="the results folder holding qc/clean_calibration.csv and qc/clean_selection.csv")
    ap.add_argument("--out", default=None, help="where to write the two tables (default: <results>/analysis)")
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    ids = {"pid": str, "grid_id": str, "img": str}
    H = human_targets(pd.read_csv(res / "qc" / "clean_calibration.csv", dtype=ids))
    H.to_csv(out / "human_targets.csv", index=False)
    print(f"{out / 'human_targets.csv'}: {len(H)} images, {H.groupby(['model', 'prompt_id']).ngroups} grids")
    P = human_picks(pd.read_csv(res / "qc" / "clean_selection.csv", dtype={"pid": str, "grid_id": str, "pick1": str, "pick2": str, "pick3": str}))
    P.to_csv(out / "human_picks.csv", index=False)
    print(f"{out / 'human_picks.csv'}: {len(P)} images, {int((P.pref_points > 0).sum())} chosen at least once")


if __name__ == "__main__":
    main()
