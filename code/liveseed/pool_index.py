"""pool_index.py: the index of a pool's finished images, which score_images.py reads.

seed_harness.py writes one results*.csv per run (or per slice of a run). Each has one row per (prompt, seed, logged step);
the row of the final step carries the image's path. This script joins the slices, keeps those rows and writes
index_finals.csv (dataset, image_id, prompt, seed, image_path) into the pool folder.

Usage:  python pool_index.py --pool runs/sd14_pool
"""
import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="the --out folder of seed_harness.py")
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.pool) / "results*.csv")))
    if not files:
        raise SystemExit(f"no results*.csv in {args.pool}")
    r = pd.concat([pd.read_csv(x, on_bad_lines="skip", low_memory=False) for x in files], ignore_index=True)
    r = r.drop_duplicates(subset=["image_id", "seed", "step"], keep="last")
    p = r.image_path.astype(str)
    f = r[r.image_path.notna() & (p.str.lower() != "nan") & (p != "")].copy()
    f["image_id"] = f["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    f["seed"] = f["seed"].astype(int)
    f = f.sort_values(["dataset", "image_id", "seed"])
    out = Path(args.pool) / "index_finals.csv"
    f[["dataset", "image_id", "prompt", "seed", "image_path"]].to_csv(out, index=False)
    print(f"{out}: {len(f)} finals, {f.image_id.nunique()} prompts, {len(f) / max(1, f.image_id.nunique()):.1f} seeds per prompt")


if __name__ == "__main__":
    main()
