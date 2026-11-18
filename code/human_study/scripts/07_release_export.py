#!/usr/bin/env python
"""07_release_export.py: writes the anonymised public dataset from the clean data.

Place in the pipeline: the last step, after 03_qc_and_payment.py. For the rating study, the four CSV files it writes are
those in human_studies/crowd/ of this supplementary material (described in human_studies/README.md).

Reads   <results>/qc/clean_calibration.csv and clean_selection.csv (only the raters used for analysis);
        data/manifests/image_blinding_map.csv and calibration_cells.csv; release_salt from the secrets file.
Writes  <results>/release/ (or --out).
  Rating design (config `study.stage: rating`, the study that was run):
    calibration_ratings.csv  rater, model, dataset, prompt_id, prompt, seed, file, display_position, prompt_alignment,
                             image_quality (1-5), image_failed
    top3_selections.csv      rater, model, dataset, prompt_id, prompt, shown_seeds (display order, '+'-joined), pick1_seed,
                             pick2_seed, pick3_seed, image_failed
    prompts.csv              model, dataset, image_id, prompt_id, prompt (every rated grid)
    images.csv               model, dataset, prompt_id, seed, draw, file (the image's name in the study; the images are
                             released alongside)
  Pairwise design (`study.stage: pairwise`; not run):
    pairwise_judgements.csv  rater, model, prompt_id, prompt, comparison, left_method, left_seed, right_method, right_seed,
                             overall, fewer_errors, follows_prompt (each 'left' or 'right'), image_failed
    self_ties.csv            the pairs not shown because both methods picked the same seed
    calibration_ratings.csv, images.csv as above, without dataset and display_position
Usage:  python scripts/07_release_export.py [--results data/results] [--out DIR]

Anonymisation. Prolific IDs are replaced by salted hashes; the salt is never published or printed, so a rater code cannot
be linked back to a Prolific account. Timestamps, response times, study and session ids, screen sizes and browser strings
are dropped. Pilot, excluded and rejected participants, spare raters beyond the required number per session, and gold or
check items are not released.
"""
import argparse, hashlib
import pandas as pd, yaml
from pathlib import Path
from common import CFG, MAN, RESULTS, HERE


def rating_release(res, out, rid):
    """Write the release of the rating design. rid maps a Prolific ID to its rater code.

    Check cards are not 'pool' images, so the merge with the blinding map drops their ratings.
    """
    I = pd.read_csv(MAN / "image_blinding_map.csv", dtype={"image_id": str})
    I = I[I.kind == "pool"]
    cells = pd.read_csv(MAN / "calibration_cells.csv", dtype={"image_id": str})[["model", "dataset", "image_id", "prompt_id", "prompt"]]

    # Ratings: the blinded file name gives model, prompt and seed; the cell gives the dataset and the prompt text.
    C = pd.read_csv(res / "qc" / "clean_calibration.csv", dtype={"pid": str, "img": str, "grid_id": str, "batch_id": str})
    C = C.merge(I[["img", "model", "prompt_id", "seed"]], on="img").merge(cells[["model", "prompt_id", "dataset", "prompt"]], on=["model", "prompt_id"])
    C = C.assign(rater=C.pid.map(rid)).rename(columns={"img": "file", "display_pos": "display_position", "pa": "prompt_alignment", "iq": "image_quality", "flagged": "image_failed"})
    cols = ["rater", "model", "dataset", "prompt_id", "prompt", "seed", "file", "display_position", "prompt_alignment", "image_quality", "image_failed"]
    C[cols].sort_values(["model", "prompt_id", "seed", "rater"]).to_csv(out / "calibration_ratings.csv", index=False)

    # Top-three choices: file names become seeds.
    S = pd.read_csv(res / "qc" / "clean_selection.csv", dtype={"pid": str, "grid_id": str, "batch_id": str})
    seed_of = dict(zip(I.img, I.seed))
    grid_of = dict(zip(I.img, zip(I.model, I.prompt_id)))
    rows = []
    for r in S.itertuples():
        shown = str(r.shown).split("+")
        m, p = grid_of[shown[0]]
        assert all(grid_of[x] == (m, p) for x in shown), f"a top-three set mixes prompts: {r.grid_id}"
        pk = [seed_of.get(x) if isinstance(x, str) and x else None for x in (r.pick1, r.pick2, r.pick3)]   # an empty pick is read as NaN, not a string
        rows.append(dict(rater=rid(r.pid), model=m, prompt_id=p, shown_seeds="+".join(str(seed_of[x]) for x in shown),
                         pick1_seed=pk[0], pick2_seed=pk[1], pick3_seed=pk[2], image_failed=int(r.flagged)))
    T = pd.DataFrame(rows).merge(cells[["model", "prompt_id", "dataset", "prompt"]], on=["model", "prompt_id"])
    T[["rater", "model", "dataset", "prompt_id", "prompt", "shown_seeds", "pick1_seed", "pick2_seed", "pick3_seed", "image_failed"]] \
        .sort_values(["model", "prompt_id", "rater"]).to_csv(out / "top3_selections.csv", index=False)

    cells.sort_values(["model", "prompt_id"]).to_csv(out / "prompts.csv", index=False)
    I[I.served == 1][["model", "dataset", "prompt_id", "seed", "draw", "img"]].rename(columns={"img": "file"}).to_csv(out / "images.csv", index=False)
    print(f"release (rating design): {len(C)} image ratings, {len(T)} top-three sets, {C.rater.nunique()} raters, {len(cells)} prompts x models -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "release"
    out.mkdir(parents=True, exist_ok=True)
    salt = yaml.safe_load(open(HERE / CFG["secrets_file"]))["release_salt"]
    rid = lambda pid: "R" + hashlib.sha256((salt + str(pid)).encode()).hexdigest()[:10]       # the rater code: 'R' + 10 hex characters
    if CFG["study"].get("stage") == "rating":
        rating_release(res, out, rid)
        return

    # ---------------- pairwise design
    M = pd.read_csv(MAN / "pairs_master.csv")
    PI = pd.read_csv(MAN / "pairwise_batches_index.csv", dtype={"item_id": str})
    R = pd.read_csv(res / "qc" / "clean_pairwise.csv", dtype={"pid": str, "item_id": str})
    # Gold pairs are dropped (kind == "real"), and each answer gets its pair's methods and seeds.
    R = R.merge(PI[PI.kind == "real"][["item_id", "ref"]].drop_duplicates("item_id"), on="item_id").merge(M, left_on="ref", right_on="pair_id")
    side = lambda r, img: (r.method_a, r.seed_a) if img == r.img_a else (r.method_b, r.seed_b)   # which method and seed an image is
    rows = []
    for r in R.itertuples():
        lm, ls = side(r, r.left_img)
        rm, rs = side(r, r.right_img)
        rows.append(dict(rater=rid(r.pid), model=r.model, prompt_id=r.prompt_id, prompt=r.prompt, comparison=r.comparison, left_method=lm, left_seed=ls,
                         right_method=rm, right_seed=rs, overall=r.q_overall, fewer_errors=r.q_quality, follows_prompt=r.q_align, image_failed=int(r.flagged)))
    pd.DataFrame(rows).sort_values(["model", "comparison", "prompt_id", "rater"]).to_csv(out / "pairwise_judgements.csv", index=False)
    M[M.is_same_seed == 1][["model", "prompt_id", "prompt", "comparison", "method_a", "seed_a", "method_b", "seed_b"]].to_csv(out / "self_ties.csv", index=False)
    I = pd.read_csv(MAN / "image_blinding_map.csv")
    I = I[I.kind == "pool"]
    C = pd.read_csv(res / "qc" / "clean_calibration.csv", dtype={"pid": str, "img": str}).merge(I[["img", "model", "prompt_id", "seed"]], on="img")
    P = pd.read_csv(MAN / "calibration_cells.csv")[["model", "prompt_id", "prompt"]]
    C = C.merge(P, on=["model", "prompt_id"])
    C = C.assign(rater=C.pid.map(rid)).rename(columns={"pa": "prompt_alignment", "iq": "image_quality", "flagged": "image_failed"})
    C[["rater", "model", "prompt_id", "prompt", "seed", "prompt_alignment", "image_quality", "image_failed"]].sort_values(["model", "prompt_id", "seed", "rater"]).to_csv(out / "calibration_ratings.csv", index=False)
    I[I.served == 1][["model", "prompt_id", "seed", "draw", "img"]].rename(columns={"img": "file"}).to_csv(out / "images.csv", index=False)
    print(f"release: {len(rows)} pairwise judgements, {len(C)} calibration ratings, {R.pid.nunique() + C.pid.nunique()} rater sessions -> {out}")


if __name__ == "__main__":
    main()
