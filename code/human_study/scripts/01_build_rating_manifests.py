#!/usr/bin/env python
"""01_build_rating_manifests.py: chooses the images to be rated, defines the check images and assigns the blinded file names.

Place in the pipeline: the first step. 02_generate_batches.py deals the cells written here into participant sessions.

Design (paper App. E): 120 prompts per model (40 T2I-CompBench, 40 TIFA and 40 DrawBench prompts with ids 1-100) x 10 seeds on
SDXL, SD3.5-Large, FLUX.1-dev and SD3.5-Large Turbo = 480 cells (prompt x model) and 4,800 images.

Reads, below config `paths.root`. Every source is a pool generated with the attention hooks, so that each rated image has
its logged readings.
  outputs/seedsel/<model>/human_rated80/index_finals.csv      the T2I-CompBench and TIFA prompts. Seed 123456 comes first;
                                                              on SDXL that image is the one rated in the expert study.
  outputs/seedsel/<model>/human_drawbench40/index_finals.csv  the DrawBench prompts, where this pool exists (SD3.5-Large).
  outputs/<pool table>.parquet                                otherwise: the first ten draws of the model's larger
                                                              DrawBench pool, listed in the schedule simulator's pool table.
  prompts/HumanStudy_DrawBench40.csv                          the forty DrawBench prompts (ID, prompt).
Writes data/manifests/, which is never served:
  calibration_cells.csv   cell_id, model, dataset, image_id, prompt_id, prompt, seeds, imgs (both '+'-joined, in draw order)
  check_cards.csv         cell_id, model, prompt_id, seed, img
  image_blinding_map.csv  img, kind ('pool' or 'check_corrupt'), model, dataset, image_id, prompt_id, seed, draw, source, served
and, with --export-images, the blinded JPEGs in <images_out>/img/.
Usage: python scripts/01_build_rating_manifests.py [--export-images]

prompt_id is <dataset>_<image_id>. The ids of the three datasets overlap, so the bare id is never used as a key.

Check cards. Each cell gets one: a copy of one of its ten images, chosen at random, then scrambled and blurred by corrupt().
The pools have no spare seeds, so the check is made from a rated image; with its centre destroyed, the copy tells the rater
nothing about the intact image. The choices are drawn from one random generator in the order of config `study.models`, so
a model appended to that list leaves the check cards of the earlier models unchanged.

This script serves the rating stage (config `study.stage: rating`): it builds no pairs and no gold pairs.
"""
import argparse, shutil
import numpy as np, pandas as pd
from common import CFG, MAN, IMG_OUT, ROOT, blind, corrupt

S = CFG["study"]
N = S["pool_size"]                      # seeds per prompt (ten in the study)
SEL = ROOT / "outputs" / "seedsel"      # the generated pools: <model>/<pool>/images_jpg/<dataset>_<id>_seed<seed>.jpg


def rated_pool(m):
    """Return the rows (dataset, image_id, prompt, seed, pool) of model m's T2I-CompBench and TIFA prompts, in generation order."""
    I = pd.read_csv(SEL / m / "human_rated80" / "index_finals.csv", dtype={"image_id": str})
    return I.assign(pool="human_rated80")[["dataset", "image_id", "prompt", "seed", "pool"]]


# Models whose larger DrawBench pool is not the default one: model -> (pool table below outputs/, pool folder below outputs/seedsel/<model>/).
TABLES = {"sd35l_turbo": ("liveseed_sim_e7/pool_sd35l_turbo_drawbench_e7.parquet", "drawbench_e7")}


def drawbench_pool(m):
    """Return the rows, in the format of rated_pool(), of model m's forty DrawBench prompts.

    They come from the model's own forty-prompt pool if it exists, and otherwise from the first ten draws of its larger
    DrawBench pool.
    """
    D = pd.read_csv(ROOT / "prompts" / "HumanStudy_DrawBench40.csv")
    text = dict(zip(D.ID.astype(str), D.prompt.astype(str)))
    if (SEL / m / "human_drawbench40" / "index_finals.csv").exists():
        I = pd.read_csv(SEL / m / "human_drawbench40" / "index_finals.csv", dtype={"image_id": str})
        return I.assign(pool="human_drawbench40")[["dataset", "image_id", "prompt", "seed", "pool"]]
    table, pool = TABLES.get(m, (f"liveseed_sim_e3/pool_{m}_drawbench50.parquet", "drawbench50"))
    P = pd.read_parquet(ROOT / "outputs" / table, columns=["prompt_id", "seed", "draw"])
    P = P[P.prompt_id.astype(str).isin(text) & (P.draw < N)].sort_values(["prompt_id", "draw"], kind="stable")
    return pd.DataFrame(dict(dataset="DrawBench", image_id=P.prompt_id.astype(str), prompt=P.prompt_id.astype(str).map(text), seed=P.seed.values, pool=pool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-images", action="store_true", help="copy the rated images and write the corrupted check images into <images_out>/img/ under their blinded names")
    args = ap.parse_args()
    assert S["stage"] == "rating", "config study.stage is not 'rating'"
    MAN.mkdir(parents=True, exist_ok=True)
    (IMG_OUT / "img").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(S["rng_seed"])
    cells, checks, imgs, seeds_db = [], [], [], {}
    for m in S["models"]:
        T = pd.concat([rated_pool(m), drawbench_pool(m)], ignore_index=True)
        T["prompt_id"] = T.dataset + "_" + T.image_id
        n = T.groupby("prompt_id", sort=False).seed.nunique()
        # Every model must have the configured number of prompts, each with exactly N distinct seeds.
        assert len(n) == S["calibration_prompts_per_model"] and (n == N).all(), (m, len(n), n[n != N].head())
        for key, g in T.groupby("prompt_id", sort=False):
            seeds = [int(s) for s in g.seed]
            pt = g.prompt.iloc[0]
            r = g.iloc[0]
            if r.dataset == "DrawBench":                                                   # every model must show the same seeds of a DrawBench prompt
                assert seeds_db.setdefault(key, set(seeds)) == set(seeds), (m, key)
            src = lambda s: SEL / m / r.pool / "images_jpg" / f"{r.dataset}_{r.image_id}_seed{s}.jpg"
            names = [blind(m, key, s) for s in seeds]
            cells.append(dict(cell_id=f"{m}_{key}", model=m, dataset=r.dataset, image_id=r.image_id, prompt_id=key, prompt=pt,
                              seeds="+".join(map(str, seeds)), imgs="+".join(names)))
            for d, (s, x) in enumerate(zip(seeds, names)):
                imgs.append(dict(img=x, kind="pool", model=m, dataset=r.dataset, image_id=r.image_id, prompt_id=key, seed=s, draw=d, source=str(src(s)), served=1))
            # The cell's check card: one of its seeds at random. "check" enters the hash, so the card's blinded name differs
            # from that of the intact image of the same seed.
            cs = seeds[int(rng.integers(N))]
            x = blind("check", m, key, cs)
            checks.append(dict(cell_id=f"{m}_{key}", model=m, prompt_id=key, seed=cs, img=x))
            imgs.append(dict(img=x, kind="check_corrupt", model=m, dataset=r.dataset, image_id=r.image_id, prompt_id=key, seed=cs, draw=-1, source=str(src(cs)), served=1))
    C, K, I = pd.DataFrame(cells), pd.DataFrame(checks), pd.DataFrame(imgs)
    assert I.img.is_unique and C.groupby("model").prompt_id.apply(set).map(len).eq(S["calibration_prompts_per_model"]).all()
    C.to_csv(MAN / "calibration_cells.csv", index=False)
    K.to_csv(MAN / "check_cards.csv", index=False)
    I.to_csv(MAN / "image_blinding_map.csv", index=False)
    missing = [p for p in I.source.unique() if not pd.io.common.file_exists(p)]
    print(f"grids {len(C)}, study images {(I.kind == 'pool').sum()}, check cards {len(K)}; sources missing {len(missing)}" + (f", first {missing[0]}" if missing else ""))
    print(C.groupby(["model", "dataset"]).size().unstack())

    if args.export_images:
        assert not missing, "some source images are missing: generate or copy the pools' JPEGs first"
        crng = np.random.default_rng(S["rng_seed"] + 1)        # a separate generator for the corruption
        n_copy = n_make = 0
        for r in I.itertuples():
            dst = IMG_OUT / "img" / r.img
            if dst.exists():                                    # a rerun keeps the files already exported
                continue
            if r.kind == "pool":
                shutil.copyfile(r.source, dst)
                n_copy += 1
            else:
                corrupt(r.source, dst, crng)
                n_make += 1
        print(f"-> {IMG_OUT / 'img'}: copied {n_copy}, corrupted {n_make}")


if __name__ == "__main__":
    main()
