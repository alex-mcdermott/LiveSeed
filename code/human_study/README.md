# The crowd study's code

This folder is the code of the paper's crowd study (Sec. 4.3, App. E). Each participant rated 42 images one at a time for prompt alignment and image quality on 1 to 5 scales, then chose a first, second and third favourite among the ten images of four further prompts. The data it produced is in `human_studies/crowd/` at the top level, with a README on its collection and anonymisation.

## Layout

| Path | What it is |
|---|---|
| `web/calibration.html`, `web/common.js`, `web/style.css` | The task page as participants saw it: consent, instructions with the rubric and nine worked examples (`web/examples/`), the rating screens and the top-three pages. Identifying details in the consent text are replaced by placeholders in brackets. |
| `apps_script/Code.gs` | The backend, a Google Apps Script web app. It assigns each participant a session, stores answers in a Google Sheet as they are given, and confirms the submission before the page returns the participant to the recruitment platform. |
| `scripts/` | The pipeline before and after the study (below). `common.py` documents the vocabulary and the Sheet's columns. |
| `config.example.yaml` | The study's settings: models, prompts per set, raters per grid, pay and quality-control thresholds. Secrets and hosting details are blank. Copy it to `config.yaml` and fill them in. |

## Pipeline

The numbering has gaps because steps that are not needed to reproduce the study are not included.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_build_rating_manifests.py` | Chooses the rated images, makes one check image per grid (a scrambled copy of one of its images), and gives every image a blinded file name. |
| 2 | `02_generate_batches.py` | Deals the grids into sessions, writes one JSON file per session and the page's `config.js`, and prints the budget. |
| | *the study runs* | Participants open the page; the backend stores their answers; the Sheet's tabs are exported as CSV files. |
| 3 | `03_qc_and_payment.py` | Quality control. A session is excluded when it fails a check image, gives 90% or more identical ratings, lasts under five minutes, reports more than 30% of its images as not loading, or is incomplete. A session that fails both check images is rejected. The script lists who is paid and writes the clean data. |
| 4 | `08_effort_screen.py` | Per-rater effort statistics, used to inspect borderline sessions by hand. |
| 5 | `05_human_targets.py` | Reduces the clean ratings and top-three choices to one human target per image: ratings are z-scored within each rater and grid, then averaged over the grid's raters. |
| 6 | `07_release_export.py` | Writes the anonymised release in `human_studies/crowd/`: rater IDs become salted hashes, and timestamps, response times and browser details are removed. |
| test | `00_simulate_participants.py` | Writes synthetic participants in the Sheet's export format, so that steps 3 to 6 can be tested without running a study. |

## Things to know

- **"calibration" means the rating task.** The code, the Sheet's tabs and the file names call the rating task "calibration", because its ratings are the targets LiveSeed is calibrated to. A second, pairwise task is supported by the code and was not run (`study.stage: rating` switches it off).
- **Blinding.** Answers name images only by blinded file names. The mapping to model, prompt and seed lives in `data/manifests/`, which is never served, so the page's network traffic reveals nothing about an image's origin.
- **Inputs that are not included.** Steps 1 and 2 read the generated pools at full size (`outputs/seedsel/<model>/...` under `paths.root`), which `code/liveseed/seed_harness.py` regenerates from the prompt and seed lists. Steps 3 to 6 read the Sheet's raw export, which holds participant IDs and is not released; `00_simulate_participants.py` produces a stand-in.
- **Check images.** Each session has two. A check image is the same picture with its centre cut into tiles, shuffled and blurred, so an attentive rater gives it an image quality of 1 or 2.
