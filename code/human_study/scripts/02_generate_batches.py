#!/usr/bin/env python
"""02_generate_batches.py: deals the cells into participant sessions (batches) and writes what the task page and the backend need.

Place in the pipeline: after 01_build_rating_manifests.py, whose manifests it reads. Its served output is uploaded to the
image host beside web/, and setup() in apps_script/Code.gs loads the session lists from there.

Reads   data/manifests/calibration_cells.csv and check_cards.csv; config.yaml. The pairwise design also reads
        distinct_pairs.csv, gold_pairs.csv and pairs_master.csv.
Writes  served to participants: only opaque ids, prompts and blinded image paths
          data/batches_calibration/<batch>.json   one session: `grids` (Part 1, rated) and `pick_grids` (Part 2, top three)
          data/batches_*/index.json               the batch ids and the raters needed per session, for Code.gs's setup()
          web/config.js                           window.STUDY_CFG, the page's settings, taken from config.yaml
        private, never uploaded: which model, prompt and seeds a grid holds
          data/manifests/calibration_batches_index.csv   batch_id, grid_idx, grid_id, cell_id, model, prompt_id, imgs, check_img
          data/manifests/selection_batches_index.csv     batch_id, grid_idx, grid_id, cell_id, model, prompt_id, imgs
          secrets.yaml                                   created on the first run (see ensure_secrets)
        budget.md, the cost of the study (also printed)
Usage:  python scripts/02_generate_batches.py

Rating task ("calibration" in the code). The cells (prompt x model) are dealt into sessions of `grids_per_session` grids with
no prompt twice in a session and the models spread evenly. The grids listed in `check_cards_in_grids` (the second and fourth)
also carry their cell's check card, a corrupted image whose image quality should be rated 1-2. The page shows a session's
images one at a time in random order (web/calibration.html). `raters_per_grid` participants rate every session.
Part 2, the top three, shows the grids of ANOTHER session that has no prompt in common with this one (`pick_grids`), so the
choices cannot lean on the scores the rater has just given, and every grid still gets `raters_per_grid` sets of choices.
Check cards appear in Part 1 only.

Pairwise task (not part of the rating stage). The distinct pairs, without those whose two methods chose the same seed, are
dealt into sessions of `real_pairs_per_session` with no prompt twice in a session (a prompt recurs across models and
comparisons); two gold pairs go to positions 6 and 16.
`raters_per_pair` participants answer every session, so every distinct pair gets exactly that many judgements.
"""
import json, math, hashlib, secrets
import numpy as np, pandas as pd, yaml
from common import CFG, MAN, BATCH_P, BATCH_C, HERE

S, PW, CA, PAY, H = CFG["study"], CFG["pairwise"], CFG["calibration"], CFG["pay"], CFG["hosting"]
RATING = S.get("stage") == "rating"   # the rating stage has no pairwise task


def opaque(*parts):
    """Return a 12-character salted hash of the parts: an id that can be served without revealing what it stands for."""
    return hashlib.sha256(("::".join(map(str, parts)) + CFG["blinding"]["salt"]).encode()).hexdigest()[:12]


def deal(units, key, size, rng, spread=None):
    """Deal units (a list of dicts) into batches of at most `size`, with no value of `key` twice in a batch.

    Units whose key is most frequent are placed first, as they are the hardest to place without a repeat. Each unit goes
    to the emptiest batch that does not hold its key yet; among equally full batches, to the one with the fewest units of
    the same `spread` value (a second key that is kept as even as possible within a batch). Returns a list of lists.
    """
    units = [units[i] for i in rng.permutation(len(units))]
    freq = pd.Series([u[key] for u in units]).value_counts()
    units.sort(key=lambda u: -freq[u[key]])                    # a stable sort: the random order survives among equal frequencies
    n_b = math.ceil(len(units) / size)
    batches = [[] for _ in range(n_b)]
    keys = [set() for _ in range(n_b)]
    for u in units:
        cand = [b for b in range(n_b) if len(batches[b]) < size and u[key] not in keys[b]]
        assert cand, "cannot place a unit without repeating its prompt; enlarge the batches"
        same = lambda b: sum(x[spread] == u[spread] for x in batches[b]) if spread else 0
        b = min(cand, key=lambda b: (len(batches[b]), same(b), b))
        batches[b].append(u)
        keys[b].add(u[key])
    return [[x for x in bt] for bt in batches]


def part2_of(batches, rng):
    """Return a permutation p: Part 2 of session b shows the grids of session p[b], which shares no prompt with session b.

    Nobody therefore chooses among images they have just rated, and because p is a permutation every grid gets exactly
    raters_per_grid sets of choices. Starts from a random permutation and swaps entries until no session clashes.
    """
    prompts = [{c["prompt_id"] for c in bt} for bt in batches]
    n = len(batches)
    p = rng.permutation(n)
    clash = lambda i: bool(prompts[i] & prompts[p[i]])         # also true when p[i] == i
    for _ in range(100000):
        bad = [i for i in range(n) if clash(i)]
        if not bad:
            return [int(x) for x in p]
        i, j = bad[0], int(rng.integers(n))
        p[i], p[j] = p[j], p[i]
    raise RuntimeError("no Part 2 assignment without shared prompts")


def ensure_secrets():
    """Create the secrets file (config `secrets_file`, untracked) on the first run; its contents are never printed.

    export_key authorises the backend's export and exclude requests; release_salt salts the rater codes of the public dataset.
    """
    f = HERE / CFG["secrets_file"]
    if not f.exists():
        f.write_text(yaml.safe_dump(dict(export_key=secrets.token_urlsafe(24), release_salt=secrets.token_hex(16))))
        f.chmod(0o600)
        print(f"created {f} (export_key, release_salt); paste export_key into the Apps Script's Script Properties as EXPORT_KEY")


def main():
    ensure_secrets()
    rng = np.random.default_rng(S["rng_seed"] + 2)
    D, G = (pd.DataFrame(), pd.DataFrame()) if RATING else (pd.read_csv(MAN / "distinct_pairs.csv"), pd.read_csv(MAN / "gold_pairs.csv"))
    C = pd.read_csv(MAN / "calibration_cells.csv")
    K = pd.read_csv(MAN / "check_cards.csv").set_index("cell_id")
    BATCH_P.mkdir(parents=True, exist_ok=True)
    BATCH_C.mkdir(parents=True, exist_ok=True)
    for f in list(BATCH_P.glob("*.json")) + list(BATCH_C.glob("*.json")):        # sessions of an earlier run must not be served
        f.unlink()

    # ---------------- pairwise (in the rating stage D is empty: no sessions, and an index.json with an empty list)
    units = D.to_dict("records")
    batches = deal(units, "prompt_id", PW["real_pairs_per_session"], rng) if units else []
    idx_rows, ids = [], []
    for bi, bt in enumerate(batches):
        bid = f"p{bi:03d}"
        ids.append(bid)
        bt = [bt[i] for i in rng.permutation(len(bt))]
        gold = G.iloc[rng.choice(len(G), PW["gold_pairs_per_session"], replace=False)].to_dict("records")
        items = [dict(kind="real", **u) for u in bt]
        for pos, gd in zip((5, 15), gold):                      # 0-based: the gold pairs are the 6th and 16th items
            items.insert(min(pos, len(items)), dict(kind="gold", **gd))
        served = []
        for k, it in enumerate(items):
            if it["kind"] == "real":
                pair = [it["img_a"], it["img_b"]]
                iid = opaque("item", it["pair_id"])
                ref = it["pair_id"]
            else:
                pair = [it["img_clean"], it["img_corrupt"]]
                iid = opaque("gold", it["gold_id"], bid)
                ref = it["gold_id"]
            if rng.random() < 0.5:                              # random order within the served pair
                pair = pair[::-1]
            served.append(dict(item_id=iid, prompt=it["prompt"], img_1="img/" + pair[0], img_2="img/" + pair[1]))
            idx_rows.append(dict(batch_id=bid, item_idx=k, item_id=iid, kind=it["kind"], ref=ref, img_1=pair[0], img_2=pair[1]))
        json.dump(dict(batch_id=bid, task="pairwise", items=served), open(BATCH_P / f"{bid}.json", "w"), indent=0)
    if not RATING:
        pd.DataFrame(idx_rows).to_csv(MAN / "pairwise_batches_index.csv", index=False)
    json.dump(dict(task="pairwise", raters=PW["raters_per_pair"], batches=ids), open(BATCH_P / "index.json", "w"), indent=0)

    # ---------------- calibration (the rating task) and its Part 2
    cells = C.to_dict("records")
    cb = deal(cells, "prompt_id", CA["grids_per_session"], rng, spread="model")
    part2 = part2_of(cb, rng)
    cidx, cids, sidx = [], [], []            # private index of Part 1, the batch ids, private index of Part 2
    for bi, bt in enumerate(cb):
        bid = f"c{bi:03d}"
        cids.append(bid)
        grids = []
        for gi, cell in enumerate(bt):
            imgs = cell["imgs"].split("+")
            check = None
            if (gi + 1) in CA["check_cards_in_grids"]:          # the config counts grids from 1
                check = K.loc[cell["cell_id"], "img"]
            gid = opaque("grid", cell["cell_id"], bid)
            grids.append(dict(grid_id=gid, prompt=cell["prompt"], images=["img/" + x for x in imgs], check="img/" + check if check else None))
            cidx.append(dict(batch_id=bid, grid_idx=gi, grid_id=gid, cell_id=cell["cell_id"], model=cell["model"], prompt_id=cell["prompt_id"],
                             imgs=cell["imgs"], check_img=check or ""))
        picks = []
        for gi, cell in enumerate(cb[part2[bi]]):               # the grids of the partner session, without check cards
            gid = opaque("pick", cell["cell_id"], bid)
            picks.append(dict(grid_id=gid, prompt=cell["prompt"], images=["img/" + x for x in cell["imgs"].split("+")]))
            sidx.append(dict(batch_id=bid, grid_idx=gi, grid_id=gid, cell_id=cell["cell_id"], model=cell["model"], prompt_id=cell["prompt_id"], imgs=cell["imgs"]))
        json.dump(dict(batch_id=bid, task="calibration", grids=grids, pick_grids=picks), open(BATCH_C / f"{bid}.json", "w"), indent=0)
    pd.DataFrame(cidx).to_csv(MAN / "calibration_batches_index.csv", index=False)
    pd.DataFrame(sidx).to_csv(MAN / "selection_batches_index.csv", index=False)
    json.dump(dict(task="calibration", raters=CA["raters_per_grid"], batches=cids), open(BATCH_C / "index.json", "w"), indent=0)

    # ---------------- pages' config
    # Pay per session = hourly rate x the session's estimated minutes.
    wc = dict(appsScriptUrl=H["apps_script_url"], completion=dict(pairwise=H["completion_code_pairwise"], calibration=H["completion_code_calibration"]),
              fallback=dict(pairwise=H["fallback_code_pairwise"], calibration=H["fallback_code_calibration"]),
              contactEmail=H["contact_email"], retention=H["data_retention"], outcomes=S["outcomes"],
              minutes=dict(pairwise=PW["minutes_per_session"], calibration=CA["minutes_per_session"]),
              pay=dict(pairwise=round(PAY["gbp_per_hour"] * PW["minutes_per_session"] / 60, 2), calibration=round(PAY["gbp_per_hour"] * CA["minutes_per_session"] / 60, 2)),
              minSecondsPerPair=PW["min_seconds_per_pair"])
    (HERE / "web" / "config.js").write_text("// generated by scripts/02_generate_batches.py from config.yaml; do not edit by hand\nwindow.STUDY_CFG = " + json.dumps(wc, indent=1) + ";\n")

    # ---------------- budget
    # Cost = participants x pay x the platform's multiplier (fee and VAT), plus the pilot and a buffer for replacing
    # excluded sessions.
    n_pw, n_ca = len(ids) * PW["raters_per_pair"], len(cids) * CA["raters_per_grid"]
    pay_pw, pay_ca = wc["pay"]["pairwise"], wc["pay"]["calibration"]
    mult = PAY["prolific_multiplier"]
    pil_pw, pil_ca = (0 if RATING else PAY["pilot_pairwise_participants"]), PAY["pilot_calibration_participants"]
    main_pw, main_ca = n_pw * pay_pw * mult, n_ca * pay_ca * mult
    pilot = (pil_pw * pay_pw + pil_ca * pay_ca) * mult
    buf = PAY["buffer_fraction"] * (main_pw + main_ca)
    total = main_pw + main_ca + pilot + buf
    tie = pd.DataFrame(dict(size=[0], sum=[0])) if RATING else pd.read_csv(MAN / "pairs_master.csv").groupby(["model", "comparison"]).is_same_seed.agg(["size", "sum"])
    lines = ["# Budget (generated by scripts/02_generate_batches.py)", "",
             f"Pay £{PAY['gbp_per_hour']:.2f}/h; Prolific total = pay × {mult} (33.33% fee + 20% VAT on the fee). Session times are estimates: reset `minutes_per_session` from the pilot's median and rerun.", "",
             "| | sessions (participants) | minutes | pay each | cost incl. fees |", "|---|---|---|---|---|",
             f"| Pairwise: {len(D)} distinct pairs × {PW['raters_per_pair']} raters = {len(D) * PW['raters_per_pair']} judgements per question, {len(ids)} sessions of ≤ {PW['real_pairs_per_session']} pairs + {PW['gold_pairs_per_session']} gold | {n_pw} | {PW['minutes_per_session']} | £{pay_pw:.2f} | £{main_pw:.2f} |",
             f"| Calibration: {len(C)} grids ({S['calibration_prompts_per_model']} prompts × {len(S['models'])} models) of {S['pool_size']} images × {CA['raters_per_grid']} raters = {S['pool_size'] * len(C) * CA['raters_per_grid']} image ratings, one image at a time, then the top three of {CA['grids_per_session']} other grids, {len(cids)} sessions of {CA['grids_per_session']} grids | {n_ca} | {CA['minutes_per_session']} | £{pay_ca:.2f} | £{main_ca:.2f} |",
             f"| Pilot: {pil_pw} pairwise + {pil_ca} calibration participants | {pil_pw + pil_ca} | | | £{pilot:.2f} |",
             f"| Buffer ({int(PAY['buffer_fraction'] * 100)}% of the main runs, for replacing excluded sessions) | | | | £{buf:.2f} |",
             f"| **Total** | **{n_pw + n_ca + pil_pw + pil_ca}** + replacements | | | **£{total:.2f}** |", "",
             f"Self-ties (both methods picked the same seed; never shown, scored 0.5): {int(tie['sum'].sum())} of {int(tie['size'].sum())} pairs.",
             f"Cap: £{PAY['budget_cap_gbp']} . Total {'within' if total <= PAY['budget_cap_gbp'] else 'OVER'} the cap"
             + ("." if total <= PAY['budget_cap_gbp'] else f": reduce `calibration_prompts_per_model` (each prompt costs about £{3 * CA['raters_per_grid'] / CA['grids_per_session'] * pay_ca * mult * (1 + PAY['buffer_fraction']):.2f}) and rerun 01 and 02."), ""]
    if RATING:                                                  # the pairwise lines do not apply
        lines = [l for l in lines if not l.startswith(("| Pairwise", "Self-ties"))]
    (HERE / "budget.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"pairwise batches {len(ids)} (sizes {sorted(set(len(b) for b in batches))}), calibration batches {len(cids)}; web/config.js written")


if __name__ == "__main__":
    main()
