#!/usr/bin/env python
"""00_simulate_participants.py: synthetic participants in the exact format of the Sheet export, to test the pipeline end to end.

Place in the pipeline: after 01 and 02 (it needs their manifests) and in place of a real study; scripts 03, 05 and 07 then
run on its output with `--results <out>`. Because the answers carry a planted signal and known bad participants, the run shows
whether quality control excludes whom it should and keeps exactly the required number of clean raters per session.

Reads   data/manifests/calibration_batches_index.csv, selection_batches_index.csv and image_blinding_map.csv. The pairwise
        design (config `study.stage` other than 'rating') reads pairwise_batches_index.csv, pairs_master.csv and
        gold_pairs.csv instead of the selection index, and the schedule simulator's tables of logged readings.
Writes  <out>/{responses_pairwise,responses_calibration,responses_selection,submissions,assignments}.csv with the columns
        listed in common.py, and <out>/truth_pairwise.csv, the planted probability of each pair.
Usage:  python scripts/00_simulate_participants.py [--out data/dryrun/results] [--effect 0.55]

The simulated participants
  Every session gets its clean raters and, with probability 0.3, one extra bad participant (about one participant in eleven):
  a random clicker, a rater who gives the same answer throughout, or a speeder, who answers like a clean rater but finishes
  faster than quality control allows.
  Ratings (PA = prompt alignment, IQ = image quality). Each image carries two planted signals, z-scored within its grid. A
  clean rater's IQ is 3 + 0.9 x (first signal) + noise and the PA is 3 + 0.6 x (second signal) + noise, each plus a leniency
  offset of that rater, rounded and clipped to 1-5. Check cards get an IQ of 1-2 and a PA of 1-3 from them. In the rating stage the
  signals are drawn per image from a generator seeded by the cell; in the pairwise design they are two logged step-2 readings.
  Top three. A rater chooses the three images with the highest sum of the two planted signals plus noise; the random clicker
  and the flat rater choose at random.
  Pairwise. Each distinct pair has a latent probability that a rater chooses LiveSeed's image: mean `--effect` on every
  question, spread 0.15 across pairs, plus a bias of 0.04 towards the left image. Clean raters choose the clean image of a
  gold pair 97% of the time.
"""
import hashlib, argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from common import CFG, MAN, HERE, ROOT, PAIR_COLS, CAL_COLS, SEL_COLS, SUB_COLS, ASSIGN_COLS


def hash_of(text):
    """Return a non-negative integer that depends only on the text; unlike Python's hash(), it is the same in every run."""
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:12], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "dryrun" / "results"))
    ap.add_argument("--effect", type=float, default=0.55)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # The rating stage has the rating task only, so the pairwise manifests are replaced by empty frames and the pairwise
    # loop below does nothing.
    rating = CFG["study"].get("stage") == "rating"
    PI = pd.DataFrame(columns=["batch_id"]) if rating else pd.read_csv(MAN / "pairwise_batches_index.csv", dtype={"batch_id": str, "item_id": str})
    M = pd.DataFrame(columns=["pair_id", "model", "comparison", "is_same_seed"]).set_index("pair_id") if rating else pd.read_csv(MAN / "pairs_master.csv").set_index("pair_id")
    G = None if rating else pd.read_csv(MAN / "gold_pairs.csv").set_index("gold_id")
    CI = pd.read_csv(MAN / "calibration_batches_index.csv", dtype={"batch_id": str, "grid_id": str}).fillna({"check_img": ""})
    SI = pd.read_csv(MAN / "selection_batches_index.csv", dtype={"batch_id": str, "grid_id": str}) if rating else pd.DataFrame(columns=["batch_id"])
    Imap = pd.read_csv(MAN / "image_blinding_map.csv")
    Imap = Imap[Imap.kind == "pool"].set_index("img")                         # blinded file name -> seed, for the rated images
    pref = {pid: float(np.clip(rng.normal(a.effect, 0.15), 0.02, 0.98)) for pid in M.index}   # per pair: P(a rater chooses LiveSeed's image)
    RP, RC, RS, SB, AS = [], [], [], [], []       # rows of responses_pairwise, responses_calibration, responses_selection, submissions, assignments
    t0 = pd.Timestamp("2026-11-02T14:00:00Z")
    k = 0

    def stamp():
        """Return the next synthetic timestamp; they are 37 s apart and strictly increasing."""
        nonlocal k
        k += 1
        return (t0 + pd.Timedelta(seconds=37 * k)).isoformat()

    # ---------------- pairwise
    for bid, g in PI.groupby("batch_id"):
        kinds = ["good"] * CFG["pairwise"]["raters_per_pair"] + ([["random", "straight", "speeder"][rng.integers(3)]] if rng.random() < 0.3 else [])
        for j, kind in enumerate(kinds):
            pid = f"SIM{bid}{j}{kind[:2]}"
            AS.append(dict(timestamp=stamp(), pid=pid, study_id="SIM", session_id=pid, task="pairwise", batch_id=bid, status="complete", pilot=0))
            for it in g.sort_values("item_idx").itertuples():
                swap = rng.random() < 0.5                                     # which image is shown on the left
                left, right = (it.img_2, it.img_1) if swap else (it.img_1, it.img_2)
                ans = {}
                for q in ("q_overall", "q_quality", "q_align"):
                    if kind == "straight":
                        ans[q] = "left"
                        continue
                    if kind == "random":
                        ans[q] = "left" if rng.random() < 0.5 else "right"
                        continue
                    if it.kind == "gold":
                        good_img = G.loc[it.ref, "img_clean"]
                        pick_good = rng.random() < 0.97
                        ans[q] = ("left" if left == good_img else "right") if pick_good else ("right" if left == good_img else "left")
                        continue
                    ls_img = M.loc[it.ref, "img_a"]                           # img_a is LiveSeed's image of the pair
                    p_ls = pref[it.ref]
                    p_left = (p_ls if left == ls_img else 1 - p_ls) + 0.04    # + the bias towards the left image
                    ans[q] = "left" if rng.random() < p_left else "right"
                RP.append(dict(timestamp=stamp(), pid=pid, study_id="SIM", session_id=pid, batch_id=bid, item_idx=it.item_idx, item_id=it.item_id,
                               left_img=left, right_img=right, rt_ms=int(rng.integers(5000, 25000)), flagged=0, **ans))
            dur = 90 if kind == "speeder" else int(rng.integers(260, 480))    # 90 s is below qc.pairwise_min_total_seconds
            SB.append(dict(timestamp=stamp(), pid=pid, study_id="SIM", session_id=pid, task="pairwise", batch_id=bid, n_items=len(g), duration_sec=dur,
                           screen_w=1920, screen_h=1080, user_agent="sim", pilot=0))

    # ---------------- calibration (the rating task) and the top-three choices
    # Pairwise design only: the planted signals are two logged step-2 readings of each image, taken from the schedule
    # simulator's pool tables (columns E__<block>__<step>). In the rating stage this loop is empty.
    readers = {}
    for m in ([] if rating else CFG["study"]["models"]):
        P = pd.read_parquet(ROOT / "outputs" / "liveseed_sim_e3" / f"pool_{m}_drawbench50.parquet")
        P["prompt_id"] = P.prompt_id.astype(int)
        c2 = [c for c in P.columns if c.startswith("E__") and c.endswith("__2")]
        sig_iq, sig_pa = c2[len(c2) // 3], c2[2 * len(c2) // 3]               # two arbitrary, different blocks
        P = P.set_index(["prompt_id", "seed"])
        readers[m] = (P[sig_iq], P[sig_pa])
    for bid, g in CI.groupby("batch_id"):
        kinds = ["good"] * CFG["calibration"]["raters_per_grid"] + ([["random", "flat", "speeder"][rng.integers(3)]] if rng.random() < 0.3 else [])
        for j, kind in enumerate(kinds):
            pid = f"SIMC{bid}{j}{kind[:2]}"
            AS.append(dict(timestamp=stamp(), pid=pid, study_id="SIMC", session_id=pid, task="calibration", batch_id=bid, status="complete", pilot=0))
            len_iq, len_pa = rng.normal(0, 0.5), rng.normal(0, 0.5)           # this rater's leniency on each scale

            # Part 1: one row per rated image, the grid's check card included.
            for c in g.sort_values("grid_idx").itertuples():
                imgs = c.imgs.split("+") + ([c.check_img] if c.check_img else [])
                seeds = [int(Imap.loc[x, "seed"]) for x in c.imgs.split("+")]
                if rating:
                    # Seeded by the cell, so every rater of a cell sees the same planted signals.
                    grng = np.random.default_rng(abs(hash_of(c.cell_id)))
                    vi = grng.normal(size=len(seeds))
                    vp = grng.normal(size=len(seeds))
                else:
                    ri, rp = readers[c.model]
                    vi = np.array([ri.loc[(c.prompt_id, s)] for s in seeds])
                    vp = np.array([rp.loc[(c.prompt_id, s)] for s in seeds])
                zi = (vi - vi.mean()) / (vi.std() + 1e-9)
                zp = (vp - vp.mean()) / (vp.std() + 1e-9)
                order = rng.permutation(len(imgs))                            # the display order
                for pos, ix in enumerate(order):
                    img = imgs[ix]
                    if kind == "flat":
                        iq = pa = 4
                    elif kind == "random":
                        iq, pa = int(rng.integers(1, 6)), int(rng.integers(1, 6))
                    elif img == c.check_img:
                        iq, pa = int(rng.integers(1, 3)), int(rng.integers(1, 4))
                    else:
                        iq = int(np.clip(np.rint(3 + len_iq + 0.9 * zi[ix] + rng.normal(0, 0.7)), 1, 5))
                        pa = int(np.clip(np.rint(3 + len_pa + 0.6 * zp[ix] + rng.normal(0, 0.7)), 1, 5))
                    RC.append(dict(timestamp=stamp(), pid=pid, study_id="SIMC", session_id=pid, batch_id=bid, grid_idx=c.grid_idx, grid_id=c.grid_id, img=img,
                                   display_pos=pos, pa=pa, iq=iq, rt_ms=int(rng.integers(3000, 15000)), flagged=0))

            # Part 2: the top three of each choice page.
            for c in SI[SI.batch_id == bid].sort_values("grid_idx").itertuples():
                imgs = c.imgs.split("+")
                # The same seed as in Part 1, so the two draws from grng repeat the cell's planted IQ and PA signals; the
                # third term is this rater's noise.
                grng = np.random.default_rng(abs(hash_of(c.cell_id)))
                like = grng.normal(size=len(imgs)) + grng.normal(size=len(imgs)) + rng.normal(0, 1, len(imgs))
                if kind in ("random", "flat"):
                    like = rng.normal(size=len(imgs))
                top = [imgs[i] for i in np.argsort(-like)[:3]]
                RS.append(dict(timestamp=stamp(), pid=pid, study_id="SIMC", session_id=pid, batch_id=bid, grid_idx=c.grid_idx, grid_id=c.grid_id, pick1=top[0], pick2=top[1], pick3=top[2],
                               shown="+".join(imgs), rt_ms=int(rng.integers(15000, 60000)), flagged=0))
            dur = 120 if kind == "speeder" else int(rng.integers(520, 900))   # 120 s is below qc.calibration_min_total_seconds
            SB.append(dict(timestamp=stamp(), pid=pid, study_id="SIMC", session_id=pid, task="calibration", batch_id=bid, n_items=len(g), duration_sec=dur,
                           screen_w=1920, screen_h=1080, user_agent="sim", pilot=0))

    # ---------------- write the exports
    Mt = M.reset_index()[["pair_id", "model", "comparison", "is_same_seed"]].assign(latent=lambda d: d.pair_id.map(pref))
    Mt.to_csv(out / "truth_pairwise.csv", index=False)                         # the planted per-pair probability of choosing LiveSeed
    for name, rows, cols in (("responses_pairwise", RP or [{c: None for c in PAIR_COLS}][:0], PAIR_COLS), ("responses_calibration", RC, CAL_COLS), ("responses_selection", RS, SEL_COLS), ("submissions", SB, SUB_COLS), ("assignments", AS, ASSIGN_COLS)):
        pd.DataFrame(rows, columns=cols).to_csv(out / f"{name}.csv", index=False)
    print(f"simulated: {len(set(r['pid'] for r in RP))} pairwise and {len(set(r['pid'] for r in RC))} calibration participants; "
          f"{len(RP)} pairwise answers, {len(RC)} calibration ratings -> {out}")


if __name__ == "__main__":
    main()
