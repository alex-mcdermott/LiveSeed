# Human studies

Two sets of human judgements of generated images, used in Sec. 4.3 and 4.4 of the paper and Appendices E and F.

| Folder | Study | What it holds |
|---|---|---|
| `crowd/` | Crowd study (Prolific), App. E | 14,400 image ratings and 1,440 top-three choices on ten-seed pools of four models, the 4,800 rated images, the rubric shown to participants |
| `expert/` | Expert study, Sec. 4.4 and App. F | Ratings by three annotators of 800 images (400 SDXL, 400 SD3.5-Medium), the images, the rubric |

Images are included at 320 × 320 px (JPEG) to fit the size limit. The full-size images can be regenerated from each row's model, prompt and seed with `code/liveseed/seed_harness.py`, using the settings below.

## Crowd study (`crowd/`)

Judgements of images generated from the same prompt with different random seeds, so that seed-selection methods can be calibrated and evaluated against people. Every image of a ten-seed pool was rated for **prompt alignment** and **image quality** (1–5, one image at a time). For four other prompts, each participant then chose their **top three** of the prompt's ten images.

**Composition**
- **Prompts:** 120 per model: 40 DrawBench, 40 T2I-CompBench and 40 TIFA. DrawBench keeps its deliberately misspelled and rare-word prompts.
- **Models:** SDXL, Stable Diffusion 3.5-Large, FLUX.1-dev and Stable Diffusion 3.5-Large-Turbo, at their diffusers defaults, 1024 × 1024. Seed 123456 is always the first of the ten.
- **Images:** ten seeds per prompt and model: 4,800 images.

**Files**

| File | Rows | Columns |
|---|---|---|
| `calibration_ratings.csv` | 14,400 (480 prompt × model grids × 10 seeds × 3 raters) | `rater`, `model`, `dataset`, `prompt_id`, `prompt`, `seed`, `file`, `display_position` (position in the session), `prompt_alignment`, `image_quality` (1–5), `image_failed` (the participant reported that the image did not load) |
| `top3_selections.csv` | 1,440 (480 grids × 3 raters) | `rater`, `model`, `dataset`, `prompt_id`, `prompt`, `shown_seeds` (the ten seeds in display order, `+`-joined), `pick1_seed`, `pick2_seed`, `pick3_seed`, `image_failed` |
| `prompts.csv` | 480 | `model`, `dataset`, `image_id`, `prompt_id`, `prompt` (every rated grid) |
| `images.csv` | 4,800 | `model`, `dataset`, `prompt_id`, `seed`, `draw` (position in the seed draw), `file` (the image's name in `images/`) |
| `images/` | 4,800 | the rated images, 320 px, named as in the study (blinded names) |
| `rubric.md`, `rubric_examples/` | | the instructions, the two rating scales and the nine worked examples every participant saw |

**Collection**
- **Participants:** adult Prolific participants on desktop computers, prescreened for an approval rate of 98–100% and fluent English.
- **Pay:** £2.40 for a session advertised at 10 minutes. The median session took 14.0 minutes, about £10.30 per hour.
- **Task:** 42 images one at a time (four prompts × ten seeds, plus two check images), each shown with its prompt and both rubrics on screen. Then four choice pages, each showing the ten images of a prompt not rated in the first part.
- **Quality control:** two check images per session, each a scrambled copy of an image of the same prompt, whose image quality must be rated 1–2.
  - One failed check, or 90% or more identical ratings, excluded the session; it was still paid.
  - Two failed checks, three low-effort sessions with identical ratings and two sessions with no data were rejected.
  - The quality-control script (`code/human_study/scripts/03_qc_and_payment.py`) also excludes a session shorter than 300 s or with more than 30% of its images reported as not loading; no session met either rule.
  - Excluded sessions were replaced, so every grid has three raters who passed. That gives 360 raters in all; 34 sessions were paid and excluded, and 15 were rejected.
- **Consent:** participants gave informed consent on the study page, including to the public release of their anonymised ratings. The task page and its consent text are in `code/human_study/web/`.

**Anonymisation**
- **Rater codes:** Prolific IDs are replaced by salted hashes (`R` + 10 hex characters). The salt is not published, so a code cannot be linked to a Prolific account.
- **Fields removed:** timestamps, response times, study and session IDs, screen sizes and browser strings.
- **Sessions not released:** excluded, rejected and pilot sessions, and the check images.
- **Other personal data:** none was collected.

**Agreement** (App. E). Agreement is measured within a prompt, since the question is whether raters agree about seeds.
- **Krippendorff's α:** 0.31 for alignment and 0.22 for quality, after removing each rater's leniency.
- **Reliability of the three-rater mean (ICC(3,3)):** 0.55 for alignment and 0.42 for quality.
- **Repeatability:** on the 80 SDXL images also rated in the expert study, alignment repeats at its reliability ceiling (r = 0.75, ceiling 0.75). Image quality does not (0.35 against a ceiling of 0.66).
- **Consequence:** single ratings are noisy and should be aggregated.

## Expert study (`expert/`)

Three annotators rated each of 800 images on the same two 1–5 scales, one image at a time with its prompt (`rubric.md`). The images are 400 prompts (200 T2I-CompBench, 200 TIFA), each generated once by SDXL (base 1.0) and once by SD3.5-Medium, with seed 123456, 30 steps, guidance 7.0 and 1024 × 1024.

| File | Rows | Columns |
|---|---|---|
| `expert_ratings.csv` | 2,400 (800 images × 3 annotators) | `model`, `dataset`, `prompt_id`, `prompt`, `category`, `seed`, `guidance`, `steps`, `file`, `annotator` (A1–A3), `image_quality`, `prompt_alignment` |
| `expert_means_as_analysed.csv` | 400 prompts | the per-image means the paper's analyses read, one row per prompt with both models' columns (the layout `code/liveseed/fit_trajectory_metric.py` reads) |
| `images/sdxl/`, `images/sd35m/` | 400 each | the rated images, 320 px, named `<dataset>_<prompt_id>.jpg` |

The two files agree on all 1,600 image × rubric means.

## Licence

The ratings are released under CC BY 4.0. The images inherit their models' licences: SDXL under the CreativeML Open RAIL++-M licence, SD3.5-Large, SD3.5-Large-Turbo and SD3.5-Medium under the Stability AI Community License, and FLUX.1-dev under the FLUX.1 [dev] Non-Commercial License.
