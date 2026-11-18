"""08_effort_screen.py: effort statistics of every rater of the rating task, for inspecting sessions by hand.

Place in the pipeline: after 03_qc_and_payment.py, whose participants.csv it reads. It takes no decision itself: the table
shows how varied a rater's scores are, how fast they answered and how well they agree with the session's clean raters, for
reviewing sessions beyond the automatic rules of 03 (quality control of the crowd study, paper App. E).

Reads   data/results/responses_calibration.csv, data/results/qc/participants.csv,
        data/manifests/calibration_batches_index.csv (to tell check cards from rated images).
Writes  data/results/qc/effort_screen.csv, one row per non-pilot rater and session, over the session's 40 rated images:
          checks                 the image quality (IQ) given to the check cards, ascending, '+'-joined
          n                      number of rated images
          same_pair, top_pair    share of images given the rater's most frequent (PA, IQ) pair, and that pair
                                 (PA = prompt alignment)
          pa_values, iq_values   number of distinct scores used on each scale
          pa_sd, iq_sd           standard deviation of the scores
          med_s, fast_share      median response time in seconds, and the share of images answered in under 2 s
          r_pa, r_iq             Spearman correlation with the mean score of the session's other clean raters; random
                                 answers give about 0 +- 0.16 on 40 images
          status, reasons, used, duration_sec   copied from participants.csv
Usage:  python scripts/08_effort_screen.py
"""
import pandas as pd, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "data" / "results"

# Every answer, the last one per image (a corrected answer follows the original in the Sheet).
R = pd.read_csv(RES / "responses_calibration.csv").drop_duplicates(["pid", "batch_id", "img"], keep="last")
CI = pd.read_csv(HERE / "data" / "manifests" / "calibration_batches_index.csv", dtype=str).fillna({"check_img": ""})
checks = set(CI.check_img) - {""}
P = pd.read_csv(RES / "qc" / "participants.csv")
P = P[P.pilot == 0]
for c in ("pa", "iq", "rt_ms"):
    R[c] = pd.to_numeric(R[c], errors="coerce")
real = R[~R.img.isin(checks)].merge(P[["pid", "batch_id"]], on=["pid", "batch_id"])     # the rated images of every non-pilot rater
used = real.merge(P[P.used == 1][["pid", "batch_id"]], on=["pid", "batch_id"])            # those of the raters kept by quality control

rows = []
for (pid, bid), me in real.groupby(["pid", "batch_id"]):
    me = me.set_index("img")
    ck = sorted(R[(R.pid == pid) & (R.batch_id == bid) & R.img.isin(checks)].iq.dropna().astype(int).tolist())
    # The reference: the mean score per image of the session's kept raters, without this rater.
    oth = used[(used.batch_id == bid) & (used.pid != pid)].groupby("img")[["pa", "iq"]].mean()
    j = me[["pa", "iq"]].join(oth, rsuffix="_o", how="inner")
    # The longest run of identical consecutive (PA, IQ) answers in display order. It is computed but not written to the table.
    longest = max((len(list(g)) for _, g in __import__("itertools").groupby(zip(me.sort_values("display_pos").pa, me.sort_values("display_pos").iq))), default=0)
    rows.append(dict(pid=pid, batch_id=bid, checks="+".join(map(str, ck)), n=len(me),
                     same_pair=round(me.groupby(["pa", "iq"]).size().max() / len(me), 2), top_pair=str(me.groupby(["pa", "iq"]).size().idxmax()),
                     pa_values=me.pa.nunique(), iq_values=me.iq.nunique(), pa_sd=round(me.pa.std(), 2), iq_sd=round(me.iq.std(), 2),
                     med_s=round(me.rt_ms.median() / 1000, 1), fast_share=round((me.rt_ms < 2000).mean(), 2),
                     r_pa=round(j.pa.corr(j.pa_o, method="spearman"), 2), r_iq=round(j.iq.corr(j.iq_o, method="spearman"), 2)))
S = pd.DataFrame(rows).merge(P[["pid", "batch_id", "status", "reasons", "used", "duration_sec"]], on=["pid", "batch_id"])
S.to_csv(RES / "qc" / "effort_screen.csv", index=False)
print(len(S), "raters screened")
