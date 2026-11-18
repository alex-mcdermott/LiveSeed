"""common.py: configuration, paths, image-name blinding and the column schemas shared by the crowd study's scripts.

The crowd study (paper Sec. 4.3 and App. E): each participant rates 42 images one at a time for prompt alignment (PA) and
image quality (IQ) on 1-5 scales, then chooses a first, second and third favourite among the ten images of four further prompts.

Pipeline. The numbered scripts run in this order; the numbering has gaps because steps that are not needed to reproduce the
study are not shipped.
  01_build_rating_manifests.py   which images are rated, the check images and the blinded file names (data/manifests/)
  02_generate_batches.py         the sessions served to participants (data/batches_*/), web/config.js and the budget
  (the study runs: web/ is the task page, apps_script/Code.gs the backend that stores answers in a Google Sheet;
   the Sheet's tabs are then exported as CSV files into data/results/)
  03_qc_and_payment.py           quality control: who is paid, who is excluded, the clean data (data/results/qc/)
  08_effort_screen.py            per-rater effort statistics, for inspecting borderline sessions by hand
  05_human_targets.py            one human target per image from the clean ratings and top-three choices (App. E)
  07_release_export.py           the anonymised public dataset
  00_simulate_participants.py    synthetic participants in the Sheet's export format, to test 03-07 without a study

Vocabulary used in the code.
  calibration   the rating task. The name refers to the use of its ratings as calibration targets for LiveSeed (App. E).
  pairwise      a second task (two images side by side, three questions per pair) that the code supports and the study did
                not run: config `study.stage: rating` switches it off, and every script then skips its pairwise branch.
  cell          one prompt on one model with its ten seeds.
  grid          a cell as it appears in one session, under an opaque id of its own.
  batch         one session: four grids rated image by image (Part 1), then the grids of another session for the
                top-three choice (Part 2).
  check card    a check image: a scrambled copy of one image of the grid, whose image quality should be rated 1 or 2.
  pid           the participant's Prolific ID. `pilot` marks participants of the pilot run, who never enter the clean data.

Configuration: config.yaml in the human-study folder, a copy of config.example.yaml with the blank fields filled in.

The Sheet tabs written by apps_script/Code.gs, one CSV each in data/results/ (the *_COLS lists below):
  assignments.csv            timestamp, pid, study_id, session_id, task, batch_id, status, pilot
  responses_pairwise.csv     timestamp, pid, study_id, session_id, batch_id, item_idx, item_id, left_img, right_img,
                             q_overall, q_quality, q_align, rt_ms, flagged
  responses_calibration.csv  timestamp, pid, study_id, session_id, batch_id, grid_idx, grid_id, img, display_pos, pa, iq,
                             rt_ms, flagged
  responses_selection.csv    timestamp, pid, study_id, session_id, batch_id, grid_idx, grid_id, pick1, pick2, pick3, shown,
                             rt_ms, flagged
  submissions.csv            timestamp, pid, study_id, session_id, task, batch_id, n_items, duration_sec, screen_w, screen_h,
                             user_agent, pilot
`rt_ms` is the response time in milliseconds, `display_pos` the image's position in the session's random order, `shown` the
'+'-joined images of a choice page in display order, and `flagged` marks an image that the participant reported as not loading.

Blinding: answers name the images a participant saw by their blinded file names only. The mapping from file names to model,
prompt and seed lives in data/manifests/, which is never served, so a participant who reads the page's network traffic
learns nothing about the images' origin.
"""
import hashlib, os, sys
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent.parent          # the human-study folder
CFG = yaml.safe_load(open(HERE / "config.yaml"))
ROOT = Path(CFG["paths"]["root"])                      # where the generated pools and prompt lists live
MAN = HERE / "data" / "manifests"                      # private: links blinded names to model, prompt and seed
BATCH_P = HERE / "data" / "batches_pairwise"           # served: one JSON file per session
BATCH_C = HERE / "data" / "batches_calibration"
RESULTS = HERE / "data" / "results"                    # the Sheet tabs exported as CSV
IMG_OUT = Path(CFG["paths"]["images_out"])             # the blinded JPEGs, synced to the image host
MODEL_KEY = {"sd14": "SD14", "sdxl": "SDXL", "sd35": "SD35", "flux": "FLUX"}        # suffixes of the schedule simulator's picks_<KEY>.csv files
sys.path.insert(0, str(HERE.parent / "liveseed"))   # makes simulate_schedules importable (folds, calibration, schedule)

# The columns of each Sheet tab, as in TABS in apps_script/Code.gs.
PAIR_COLS = ["timestamp", "pid", "study_id", "session_id", "batch_id", "item_idx", "item_id", "left_img", "right_img",
             "q_overall", "q_quality", "q_align", "rt_ms", "flagged"]
CAL_COLS = ["timestamp", "pid", "study_id", "session_id", "batch_id", "grid_idx", "grid_id", "img", "display_pos", "pa", "iq", "rt_ms", "flagged"]
# One row per choice page: the top three of a grid, and the ten images in the order shown.
SEL_COLS = ["timestamp", "pid", "study_id", "session_id", "batch_id", "grid_idx", "grid_id", "pick1", "pick2", "pick3", "shown", "rt_ms", "flagged"]
SUB_COLS = ["timestamp", "pid", "study_id", "session_id", "task", "batch_id", "n_items", "duration_sec", "screen_w", "screen_h", "user_agent", "pilot"]
ASSIGN_COLS = ["timestamp", "pid", "study_id", "session_id", "task", "batch_id", "status", "pilot"]


def liveseed_rule():
    """Return the label under which the schedule simulator's picks file lists LiveSeed's pick, e.g. 'LiveSeed ridge exit3, N=10'."""
    return f"{CFG['study']['liveseed_rule']}, N={CFG['study']['pool_size']}"


def blind(*parts) -> str:
    """Return a deterministic blinded file name: SHA-256 of the parts and the secret salt, 16 hex characters, '.jpg'.

    Without the salt, a file name cannot be traced back to its model, prompt or seed.
    """
    token = "::".join(str(p) for p in parts) + "::" + CFG["blinding"]["salt"]
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16] + ".jpg"


def corrupt(src, dst, rng):
    """Write the check image of `src` to `dst`: the central 70% of the image cut into an 8 x 8 grid of tiles, the tiles
    shuffled with the generator `rng` and blurred, the border left intact.

    A check image is recognisably the same picture but plainly broken, so an attentive rater gives it an image-quality
    rating of 1 or 2 (the rule that 03_qc_and_payment.py tests).
    """
    from PIL import Image, ImageFilter
    im = Image.open(src).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = int(0.15 * w), int(0.15 * h), int(0.85 * w), int(0.85 * h)
    box = im.crop((x0, y0, x1, y1))
    bw, bh = box.size
    t = 8
    tw, th = bw // t, bh // t
    tiles = [box.crop((i * tw, j * th, (i + 1) * tw, (j + 1) * th)) for j in range(t) for i in range(t)]
    order = rng.permutation(len(tiles))
    out = Image.new("RGB", (tw * t, th * t))
    for k, o in enumerate(order):
        out.paste(tiles[o], ((k % t) * tw, (k // t) * th))
    out = out.filter(ImageFilter.GaussianBlur(3))
    im.paste(out, (x0, y0))
    im.save(dst, quality=90)


def pool_dir(m):
    """Return the folder of model m's fifty-seed DrawBench pool."""
    return ROOT / "outputs" / "seedsel" / m / "drawbench50"


def jpg_path(m, prompt_id, seed):
    """Return the path of one finished image of that pool."""
    return pool_dir(m) / "images_jpg" / f"DrawBench_{prompt_id}_seed{seed}.jpg"


def read_export(name, cols):
    """Read a Sheet tab exported as CSV (data/results/<name>.csv); return an empty frame with the schema if the file is absent.

    The id columns are read as strings, because a hexadecimal id made only of digits would otherwise be parsed as a number.
    """
    import pandas as pd
    f = RESULTS / f"{name}.csv"
    if not f.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(f, dtype={"pid": str, "batch_id": str, "item_id": str, "grid_id": str})
