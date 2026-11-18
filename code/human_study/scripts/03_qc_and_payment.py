#!/usr/bin/env python
"""03_qc_and_payment.py: quality control. Decides who is paid and whose answers are kept, writes the clean data, and lists
the sessions that still need raters.

Place in the pipeline: after the study (or after 00_simulate_participants.py), on the Sheet tabs exported as CSV.
08_effort_screen.py, 05_refit_calibration.py and 07_release_export.py read its output. With --push, the backend offers the
slots of excluded and rejected participants to new ones, which is how excluded sessions are replaced (paper App. E).

Reads   <results>/responses_pairwise.csv, responses_calibration.csv, responses_selection.csv, submissions.csv and
        assignments.csv (columns in common.py); data/manifests/calibration_batches_index.csv, which names the check
        cards; data/batches_*/index.json, the list of sessions.
Writes  <results>/qc/
          participants.csv              pid, task, batch_id, status, reasons, duration_sec, submitted, pilot, used
                                        (used = 1: this rater's answers are in the clean data)
          approve_pids.txt, reject_pids.txt, exclude_pids.txt   Prolific IDs by decision; approve includes the excluded
          clean_pairwise.csv, clean_calibration.csv, clean_selection.csv   the answers of the used raters only
          sessions_needing_raters.csv   task, batch_id, clean, needed
Usage:  python scripts/03_qc_and_payment.py [--results data/results] [--push]
        --push sends the excluded and rejected participants to the Apps Script (key from the secrets file).

Rules. They follow Prolific's attention-check policy: a submission is rejected only for a clear failure of fair checks;
otherwise the participant is paid and the data are excluded. Check that policy as it currently stands before rejecting
anyone; when in doubt, approve and exclude.
  calibration, the rating task (two check cards per session: a scrambled image of the same prompt, whose image quality, IQ,
  must be rated 1-2)
    REJECT             IQ >= calibration_check_fail_iq (3) on BOTH check cards. Prompt alignment (PA) is not checked,
                       because the scrambling leaves the edges of the image intact.
    APPROVE_EXCLUDE    IQ >= calibration_check_fail_iq on one check card; or a session shorter than
                       calibration_min_total_seconds; or >= calibration_flat_exclude of the session's ratings equal to one
                       value (on either scale); or > 30% of the images reported as not loading; or the top three of a set
                       missing; or no confirmed submission
    APPROVE_KEEP       otherwise
  pairwise (two gold pairs per session: a clean image against a scrambled copy of itself)
    REJECT             the scrambled image chosen on BOTH the quality and the overall question in BOTH gold pairs
    APPROVE_EXCLUDE    the scrambled image chosen on the quality question in either gold pair; or a session shorter than
                       pairwise_min_total_seconds; or >= straightline_exclude of the real pairs answered on the same side
                       (overall question); or > 30% of the images reported as not loading; or no confirmed submission
    APPROVE_KEEP       otherwise

Answers are de-duplicated first, keeping the last one per item: the page may send an answer more than once, and a corrected
answer follows the original. Pilot participants (pilot = 1) are reported separately and never enter the clean data.
Every session contributes exactly `raters_per_grid` (or `raters_per_pair`) clean raters: if more completed, the earliest
submissions are kept.
"""
import argparse, json, urllib.request
import numpy as np, pandas as pd, yaml
from common import CFG, MAN, RESULTS, HERE, PAIR_COLS, CAL_COLS, SEL_COLS, SUB_COLS, ASSIGN_COLS
from pathlib import Path

Q = CFG["qc"]        # the thresholds


def load(res, name, cols):
    """Read <res>/<name>.csv with the id columns as strings; return an empty frame with columns `cols` if it is absent."""
    f = res / f"{name}.csv"
    if not f.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(f, dtype={"pid": str, "batch_id": str, "item_id": str, "grid_id": str, "img": str})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    res = Path(a.results)
    out = res / "qc"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- load, keeping the last answer per item
    RP, RC = load(res, "responses_pairwise", PAIR_COLS), load(res, "responses_calibration", CAL_COLS)
    RS = load(res, "responses_selection", SEL_COLS).drop_duplicates(["pid", "batch_id", "grid_id"], keep="last")
    SB, AS = load(res, "submissions", SUB_COLS), load(res, "assignments", ASSIGN_COLS)
    RP = RP.drop_duplicates(["pid", "batch_id", "item_id"], keep="last")
    RC = RC.drop_duplicates(["pid", "batch_id", "grid_id", "img"], keep="last")
    SB = SB.drop_duplicates(["pid", "task", "batch_id"], keep="last")
    # The pilot flag may be exported as 1, 1.0 or True.
    pilot = set(AS[AS.pilot.astype(str).isin(["1", "1.0", "True"])].pid) | set(SB[SB.pilot.astype(str).isin(["1", "1.0", "True"])].pid)
    rating = CFG["study"].get("stage") == "rating"   # the rating stage has no pairwise task, so no pairwise manifests
    PI = pd.DataFrame(columns=["kind", "item_id"]) if rating else pd.read_csv(MAN / "pairwise_batches_index.csv", dtype={"batch_id": str, "item_id": str})
    G = pd.DataFrame(columns=["img_corrupt"]) if rating else pd.read_csv(MAN / "gold_pairs.csv")
    CI = pd.read_csv(MAN / "calibration_batches_index.csv", dtype={"batch_id": str, "grid_id": str}).fillna({"check_img": ""})
    corrupt = set(G.img_corrupt)               # the scrambled images of the gold pairs
    checks = set(CI.check_img) - {""}          # the check cards of the rating task
    rows = []                                  # one row per participant and session

    # ---------------- pairwise
    gold_items = set(PI[PI.kind == "gold"].item_id)
    for (pid, bid), g in RP.groupby(["pid", "batch_id"]):
        sub = SB[(SB.pid == pid) & (SB.task == "pairwise") & (SB.batch_id == bid)]
        dur = float(sub.duration_sec.iloc[0]) if len(sub) else np.nan
        gg = g[g.item_id.isin(gold_items)]
        real = g[~g.item_id.isin(gold_items)]
        chose = lambda r, q: r.left_img if r[q] == "left" else (r.right_img if r[q] == "right" else None)   # the image chosen on question q
        fail_q = sum(chose(r, "q_quality") in corrupt for _, r in gg.iterrows())
        fail_both = sum((chose(r, "q_quality") in corrupt) and (chose(r, "q_overall") in corrupt) for _, r in gg.iterrows())
        ans = real[real.flagged.astype(int) == 0].q_overall
        sl = max((ans == "left").mean(), (ans == "right").mean()) if len(ans) else np.nan      # share of answers on the more frequent side
        flag = real.flagged.astype(int).mean() if len(real) else 0
        reasons = []
        if len(gg) >= 2 and fail_both >= 2:
            status = "REJECT"
            reasons.append("scrambled image chosen on quality and overall in both gold pairs")
        else:
            if fail_q:
                reasons.append(f"{fail_q} gold pair(s) failed on quality")
            if dur == dur and dur < Q["pairwise_min_total_seconds"]:          # dur == dur is False for NaN (no submission)
                reasons.append(f"{dur:.0f} s < {Q['pairwise_min_total_seconds']} s")
            if sl == sl and sl >= Q["straightline_exclude"]:
                reasons.append(f"{sl:.0%} same side")
            if flag > 0.3:
                reasons.append(f"{flag:.0%} flagged images")
            if not len(sub):
                reasons.append("no confirmed submission")
            status = "APPROVE_EXCLUDE" if reasons else "APPROVE_KEEP"
        rows.append(dict(pid=pid, task="pairwise", batch_id=bid, status=status, reasons="; ".join(reasons), duration_sec=dur,
                         submitted=sub.timestamp.iloc[0] if len(sub) else "", pilot=int(pid in pilot)))

    # ---------------- calibration (the rating task)
    for (pid, bid), g in RC.groupby(["pid", "batch_id"]):
        sub = SB[(SB.pid == pid) & (SB.task == "calibration") & (SB.batch_id == bid)]
        dur = float(sub.duration_sec.iloc[0]) if len(sub) else np.nan
        ok = g[g.flagged.astype(int) == 0]                                    # images that loaded
        ck = ok[ok.img.isin(checks)]
        n_fail = int((pd.to_numeric(ck.iq) >= Q["calibration_check_fail_iq"]).sum())
        real = ok[~ok.img.isin(checks)]
        # Flat ratings: the share of the session's rated images that received the rater's most frequent score, on the
        # scale where that share is larger.
        flat = max([pd.to_numeric(real[c]).value_counts(normalize=True).max() for c in ("pa", "iq")] or [0]) if len(real) else 0
        flag = g.flagged.astype(int).mean()
        reasons = []
        if len(ck) >= 2 and n_fail >= 2:
            status = "REJECT"
            reasons.append(f"scrambled check image rated quality >= {Q['calibration_check_fail_iq']} twice")
        else:
            if n_fail:
                reasons.append(f"{n_fail} check image(s) rated quality >= {Q['calibration_check_fail_iq']}")
            if dur == dur and dur < Q["calibration_min_total_seconds"]:
                reasons.append(f"{dur:.0f} s < {Q['calibration_min_total_seconds']} s")
            if flat >= Q["calibration_flat_exclude"]:
                reasons.append(f"{flat:.0%} identical ratings")
            if flag > 0.3:
                reasons.append(f"{flag:.0%} flagged images")
            if not len(sub):
                reasons.append("no confirmed submission")
            n_sel = len(RS[(RS.pid == pid) & (RS.batch_id == bid)])           # choice pages answered in Part 2
            if n_sel < CFG["calibration"]["grids_per_session"]:
                reasons.append(f"top three of {n_sel} sets only")
            status = "APPROVE_EXCLUDE" if reasons else "APPROVE_KEEP"
        rows.append(dict(pid=pid, task="calibration", batch_id=bid, status=status, reasons="; ".join(reasons), duration_sec=dur,
                         submitted=sub.timestamp.iloc[0] if len(sub) else "", pilot=int(pid in pilot)))

    # ---------------- decisions, payment lists and clean data
    P = pd.DataFrame(rows, columns=["pid", "task", "batch_id", "status", "reasons", "duration_sec", "submitted", "pilot"])
    # Exactly k clean raters per session: keep the earliest submissions, so every grid has the same number of ratings.
    need = {"pairwise": CFG["pairwise"]["raters_per_pair"], "calibration": CFG["calibration"]["raters_per_grid"]}
    P["used"] = 0
    keep = P[(P.status == "APPROVE_KEEP") & (P.pilot == 0)].sort_values("submitted")
    for (task, bid), g in keep.groupby(["task", "batch_id"]):
        P.loc[g.index[:need[task]], "used"] = 1
    P.to_csv(out / "participants.csv", index=False)
    # The approve list holds both APPROVE_KEEP and APPROVE_EXCLUDE: excluded participants are paid.
    for st, fn in (("APPROVE", "approve_pids.txt"), ("REJECT", "reject_pids.txt"), ("APPROVE_EXCLUDE", "exclude_pids.txt")):
        sel = P[P.status.str.startswith(st)] if st == "APPROVE" else P[P.status == st]
        (out / fn).write_text("\n".join(sorted(set(sel.pid))) + "\n")
    used = P[P.used == 1]
    cp = RP.merge(used[used.task == "pairwise"][["pid", "batch_id"]], on=["pid", "batch_id"])
    cp.to_csv(out / "clean_pairwise.csv", index=False)
    cc = RC.merge(used[used.task == "calibration"][["pid", "batch_id"]], on=["pid", "batch_id"])
    cc.to_csv(out / "clean_calibration.csv", index=False)
    RS.merge(used[used.task == "calibration"][["pid", "batch_id"]], on=["pid", "batch_id"]).to_csv(out / "clean_selection.csv", index=False)

    # ---------------- sessions still short of clean raters
    allb = [("pairwise", b) for b in json.load(open(HERE / "data" / "batches_pairwise" / "index.json"))["batches"]] + \
           [("calibration", b) for b in json.load(open(HERE / "data" / "batches_calibration" / "index.json"))["batches"]]
    cnt = used.groupby(["task", "batch_id"]).size()
    short = [dict(task=t, batch_id=b, clean=int(cnt.get((t, b), 0)), needed=need[t]) for t, b in allb if cnt.get((t, b), 0) < need[t]]
    pd.DataFrame(short, columns=["task", "batch_id", "clean", "needed"]).to_csv(out / "sessions_needing_raters.csv", index=False)
    summ = P[P.pilot == 0].groupby(["task", "status"]).size().unstack(fill_value=0)
    print(summ.to_string())
    print(f"pilot participants: {int(P.pilot.sum())}")
    print(f"sessions still needing clean raters: {len(short)} (pairwise {sum(s['task'] == 'pairwise' for s in short)}, calibration {sum(s['task'] == 'calibration' for s in short)})")

    # ---------------- tell the backend whom to replace
    if a.push:
        sec = yaml.safe_load(open(HERE / CFG["secrets_file"]))
        bad = P[P.status.isin(["APPROVE_EXCLUDE", "REJECT"]) & (P.pilot == 0)]
        body = json.dumps(dict(action="exclude", key=sec["export_key"], pids=[dict(pid=r.pid, task=r.task) for r in bad.itertuples()])).encode()
        # The same content type as the page's requests; Code.gs parses the body as JSON.
        req = urllib.request.Request(CFG["hosting"]["apps_script_url"], data=body, headers={"Content-Type": "text/plain;charset=utf-8"})
        print("push:", json.loads(urllib.request.urlopen(req, timeout=60).read().decode()))


if __name__ == "__main__":
    main()
