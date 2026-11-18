# LiveSeed: Attention Entropy Knows Which Seeds to Keep

Alex J. McDermott<sup>1,2</sup>, Silpa Vadakkeeveetil Sreelatha<sup>2</sup>, Anjan Dutta<sup>2</sup>, Yi-Zhe Song<sup>1,2</sup>

<sup>1</sup>SketchX &nbsp; <sup>2</sup>Surrey Institute for People-Centred AI, University of Surrey

**Paper:** arXiv link to be added.

Generating several seeds and keeping the best is the simplest way to get a better image from a text-to-image diffusion model, but every candidate is finished and then judged. LiveSeed reads a signal the generator already computes: the entropy of its self-attention (U-Nets, Hunyuan-DiT) or joint attention (MMDiTs) during the first denoising steps. The N seeds of a prompt run as one batch, are scored from those readings at an early exit, and only the best are completed. No images are decoded, no auxiliary network runs and the prompt needs no annotation.

This repository contains the code of the method, the data and code of the paper's two human studies, and the prompt and seed lists of the experiments.

| Path | Contents |
|---|---|
| `human_studies/` | The crowd study (14,400 ratings, 1,440 top-three choices, 4,800 images) and the expert study (2,400 ratings of 800 images), with the rubrics. See `human_studies/README.md`. |
| `code/liveseed/` | The method: attention-entropy hooks, the pool harness, calibration of LiveSeed-Fast and LiveSeed-Max, Algorithm A run live, and offline selection and schedule simulation. Its `README.md` explains how the files fit together and where to start reading. |
| `code/human_study/` | The crowd study's task page, backend, batch generation, quality control and anonymised export. Its `README.md` walks through the pipeline. |
| `code/environment/` | Container definition and pinned Python packages. |
| `prompts/`, `seeds/` | The prompt sets and seed lists of the experiments. |

## Code (`code/liveseed/`)

| File | Role |
|---|---|
| `seed_harness.py` | Generates a pool (prompts × N seeds) and logs the attention-entropy readings of every hooked block at steps 1–K, decoded previews and the final images. Models: `sd14`, `sd15`, `sd21`, `sdxl`, `sd35` (SD3.5-Medium), `sd35l`, `sd35l_turbo`, `flux`, `hunyuan`, `qwen`. |
| `ae_hook.py`, `ae_hook_SD35.py`, `ae_hook_FLUX.py`, `ae_hook_HUNYUAN.py`, `ae_hook_QWEN.py` | Attention processors that compute the entropy of self-attention (U-Nets, Hunyuan-DiT) or joint attention (MMDiTs) per block and step (Sec. 3.1). |
| `fast_hooks.py` | The deployed reader: only the chosen block–step readings, gated to the steps before the exit. |
| `fused_hooks.py` | The exact reading from one memory-efficient attention call with the keys as values, without materialising the attention map (Sec. 3.1, App. D). |
| `liveseed_live.py` | `calibrate`: fits LiveSeed-Fast (five maps) and LiveSeed-Max (ridge) on a scored pool. `run`: Algorithm A live, with time and memory, and the ABSS, best-of-N and single-seed baselines on the same card. |
| `fit_trajectory_metric.py`, `seed_select_eval.py`, `token_layout.py`, `posthoc_rule_eval.py` | Offline calibration and selection on logged pools with two-fold cross-validation over prompts and bootstrap intervals. Also the correlation of readings with human ratings (Sec. 4.4). |
| `abss_statistic.py`, `abss_core_tokens_gpt.py` | Our implementation of the ABSS statistic (cross-attention mass on the prompt's core words), and the GPT-5.1 query for those words with ABSS's instruction and examples (needs an OpenAI key; spaCy is the offline alternative). |
| `liveseed_pool_table.py`, `simulate_schedules.py` | Exports a pool's readings and scores, then simulates exit schedules, best-of-k and preview baselines offline with bootstrap intervals (App. C). |
| `pool_index.py` | Writes a pool's index of finished images (`index_finals.csv`), which `score_images.py` reads. |
| `score_images.py` | HPSv2.1, ImageReward, PickScore and CLIP scores of finished images or previews (`--resume` continues an interrupted run). |

### Environment

The experiments ran in an Apptainer container built from `code/environment/liveseed.def`, with the packages in `requirements.txt` (PyTorch 2.8, CUDA 12.6, diffusers 0.32.2). SciPy, Matplotlib, the scorers and spaCy are installed on top into the user site by `install_extras.sh` (listed in `requirements-extra.txt`). The scorers go in without their dependencies, so that the container's PyTorch is kept. Qwen-Image alone needs diffusers 0.35.2, in a separate user site.

```bash
cd code/environment && apptainer build liveseed.sif liveseed.def
apptainer exec liveseed.sif bash install_extras.sh      # set PYTHONUSERBASE to choose the user site
```

Model weights download from the Hugging Face Hub on first use. SD3.5 and FLUX.1 are gated: accept their licences on the Hub and set `HF_TOKEN`. The paper's runs used one A100-80GB or one H200 per job. SD1.x also runs on a 16 GB card, which is how the example below was tested.

### Example: calibrate on a pool, then select live

Run from `code/liveseed/`. With SD1.4, a pool of six prompts × six seeds and four test prompts, this runs in about ten minutes on a 16 GB GPU once the weights have downloaded; the paper's pools are far larger (Sec. 4).

```bash
# 1. a pool: readings logged at steps 1-12 for every seed, finished images and previews
python seed_harness.py --model sd14 --csv ../../prompts/DrawBench.csv --slice 0 6 --n-seeds 6 --steps 30 --log-steps 12 --out runs/sd14_pool
# 2. index and score the finished images (HPSv2.1 is the calibration target; --metrics hps scores it alone)
python pool_index.py --pool runs/sd14_pool
python score_images.py --index runs/sd14_pool/index_finals.csv --out runs/sd14_pool/scores_finals.csv
# 3. calibrate both forms on the pool; --cap is the last step whose readings are used, here the exit step
python liveseed_live.py calibrate --model sd14 --pool runs/sd14_pool --cap 3 --out runs/maps_sd14.json
# 4. Algorithm A on prompts outside the pool: N seeds in one batch, exit at step 3, keep the best
python liveseed_live.py run --model sd14 --prompts ../../prompts/HPD_photo.csv --maps runs/maps_sd14.json --mode liveseed --form ridge --schedule 3:1 --N 10 --limit 4 --out runs/live
```

## Human study code (`code/human_study/`)

- `web/` is the task page that participants saw, including the consent text; identifying details are replaced by placeholders.
- `apps_script/Code.gs` is the backend that assigned sessions and stored answers in a Google Sheet.
- `scripts/` holds the pipeline:
  - `01`–`02` build the image manifests and session batches (blinded file names, check images, the four-grids-per-session layout);
  - `00` simulates participants to test the pipeline end to end;
  - `03` and `08` apply quality control and payment;
  - `05` reduces the clean ratings and top-three choices to one human target per image;
  - `07` writes the anonymised release in `human_studies/crowd/`.
- `config.example.yaml` holds the study's settings: models, prompts per set, raters per grid, pay and quality-control thresholds. Its secrets and hosting details are left blank.

## Prompt and seed lists

- `prompts/` contains the benchmarks' prompts as used:
  - DrawBench (200), Attend-and-Excite (276), HPD v2 photo (250), PartiPrompts stratified (240), Pick-a-Pic test (500), GenAI-Bench (400) and GenEval 2 (800);
  - ABSS's protocol subsets (`ABSS_*.csv`);
  - the crowd study's DrawBench and rated subsets (`HumanStudy_*.csv`);
  - the unseen prompts of the frozen-reading test (`Frozen-Test-Prompts.csv`).
- `seeds/` contains three lists:
  - ABSS's released seeds (`abss_global_pool.csv`);
  - the seeds of the fifty-seed pools (`e3_drawbench_50seeds.csv`);
  - the prompts of the live timing runs (`e2_live_prompts.csv`).

The prompts belong to their benchmarks and keep those benchmarks' licences.

## Citation

```bibtex
@article{mcdermott2026liveseed,
  title   = {LiveSeed: Attention Entropy Knows Which Seeds to Keep},
  author  = {McDermott, Alex J. and Vadakkeeveetil Sreelatha, Silpa and Dutta, Anjan and Song, Yi-Zhe},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Licence

The code is released under the MIT licence (`LICENSE`). The human-study ratings are released under CC BY 4.0. The rated images inherit their models' licences, and the prompt sets keep their benchmarks' licences (see `human_studies/README.md`).
