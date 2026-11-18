# LiveSeed: the method's code

LiveSeed runs the N seeds of a prompt as one batch, reads the entropy of the model's attention maps during the first denoising steps, scores every seed from those readings and finishes only the best. This folder holds everything needed to do that: the hooks that take the readings, the harness that builds a calibration pool, the calibration of the two scoring forms, the live algorithm, and the offline analyses.

The folder's parent `README.md` (two levels up) has the environment set-up and a four-command example that runs on a 16 GB GPU.

## Terms

| Term | Meaning |
|---|---|
| reading | One normalised attention entropy, in [0, 1], for one (block, step). Sec. 3.1, Eq. 2. |
| pool | Prompts × seeds generated to completion with every reading logged, and the finished images scored. Calibration and all offline evaluation run on pools. |
| LiveSeed-Fast | Five readings with signs and equal weights, chosen greedily. `subset5` or `greedy` in the code. Sec. 3.2. |
| LiveSeed-Max | A ridge regression over every reading up to the exit step. `ridge` in the code. Sec. 3.2. |
| exit, schedule | An exit `s:k` scores the batch at step `s` and keeps the `k` best seeds. A schedule is one or two exits, for example `3:1`. Sec. 3.4. |
| ABSS | The baseline: cross-attention mass on the prompt's core words at step 10 of 50. |

## How the files fit together

```
prompts ──► seed_harness.py ──► pool (results.csv, images/, previews/, stats/)
                 │  uses ae_hook*.py                │
                 │                                  ├─► pool_index.py ──► score_images.py ──► scores_finals.csv
                 │                                  │
                 │                                  ├─► liveseed_live.py calibrate ──► maps.json
                 │                                  │
                 │                                  └─► offline: fit_trajectory_metric.py, seed_select_eval.py,
                 │                                      liveseed_pool_table.py ──► simulate_schedules.py
                 ▼
new prompts + maps.json ──► liveseed_live.py run ──► the kept image, time and memory
                                 uses fused_hooks.py / fast_hooks.py
```

## Where to start reading

1. **`liveseed_live.py`** is the method. `calibrate` fits both forms on a pool; `run` is Algorithm A, and its main loop is numbered (1) to (6): encode, denoise all N seeds to the exit while reading, score, keep the best, finish, decode.
2. **`ae_hook.py`** is the reference for how a reading is computed (U-Nets). The other hook files follow its design on their backbones.
3. **`seed_harness.py`** builds a pool. Its docstring documents every output file and column.

## Files

**Taking the readings (Sec. 3.1)**

| File | Backbones | Notes |
|---|---|---|
| `ae_hook.py` | SD1.x, SD2.x, SDXL | Self-attention entropy, both aggregation orders, and the cross-attention statistics that the ABSS baseline needs. Reference design. |
| `ae_hook_SD35.py` | SD3.5-Medium, SD3.5-Large | Joint attention of the MMDiT blocks. |
| `ae_hook_FLUX.py` | FLUX.1-dev, FLUX.1-schnell | Double-stream and single-stream blocks. |
| `ae_hook_HUNYUAN.py` | Hunyuan-DiT | Self-attention. |
| `ae_hook_QWEN.py` | Qwen-Image | Joint attention; needs diffusers 0.35 or later. |
| `fused_hooks.py` | U-Nets, SD3.5, FLUX.1, Hunyuan-DiT | The same reading from one memory-efficient attention call, without building the attention matrix (App. D). Used by the live loop. |
| `fast_hooks.py` | U-Nets, SD3.5, Hunyuan-DiT | Step-gated readers that compute only the wanted (block, step) readings. |

The logging hooks (`ae_hook*.py`) record every block at every logged step and are what pools are built with. The fused and fast hooks are what a deployment uses: they read only what the calibrated form needs and leave the model's own attention computation untouched.

**Building and scoring a pool (App. C)**

| File | Role |
|---|---|
| `seed_harness.py` | Generates prompts × seeds and logs readings, text-token attention, previews and finished images. `--log-mode deployed` reads only the listed (block, step) pairs. |
| `pool_index.py` | Writes `index_finals.csv`, the list of finished images that the scorer reads. |
| `score_images.py` | HPSv2.1, ImageReward, PickScore and CLIP scores (optionally VQAScore and HPSv3). HPSv2.1 is the calibration target. |
| `token_layout.py` | Where the special tokens sit in the stored text-token vectors, per text encoder. |

**Calibration and selection (Sec. 3.2 to 3.4)**

| File | Role |
|---|---|
| `liveseed_live.py` | `calibrate` and `run` (above). `run` also times the ABSS, best-of-N and single-seed baselines on the same card. |
| `fit_trajectory_metric.py` | The greedy five-map selection and the ridge, with two-fold cross-validation over prompts. Also the correlation of readings with human ratings (Sec. 4.4). |
| `seed_select_eval.py` | Evaluates selection rules offline on one pool: every rule sees the same finished images, so the comparison is exact. |
| `posthoc_rule_eval.py` | The same evaluation for a rule given as a per-image score file. |
| `liveseed_pool_table.py` | Exports a pool as one table for the simulator. |
| `simulate_schedules.py` | Simulates exit schedules, best-of-k and preview baselines offline, with bootstrap intervals over prompts. No GPU needed. |

**The ABSS baseline**

| File | Role |
|---|---|
| `abss_statistic.py` | ABSS's seed score from the stored attention maps and vectors. |
| `abss_core_tokens_gpt.py` | ABSS's core words from GPT-5.1 with its published instruction and examples. Needs `OPENAI_API_KEY`. spaCy nouns are the offline alternative. |

## Things to know

- **Two aggregation orders.** The per-row order of Eq. 2 takes the entropy of every attention row and then the mean. The pooled order averages the rows first and takes one entropy. Column prefixes: on U-Nets `sa_entfirst_` is per-row and `sa_mean_` is pooled; on the transformers `ae_mean_` is per-row and `ae_avgfirst_` is pooled. App. C and the calibration-settings table in App. G state which order each experiment uses.
- **Set `--cap` to the last exit step** when calibrating for a live run, as the example does. `calibrate` fits on readings up to `--cap`, and `run` can only use readings taken before its last exit.
- **The ridge penalty grid** in the code is 1 to 10^4. The rating-prediction results of Sec. 4.4 were computed with 10^-1 to 10^3, as the calibration-settings table states.
- **Scoring long lists.** `score_images.py` writes every 200 images; `--resume` continues an interrupted run.
- **Gated weights.** SD3.5 and FLUX.1 need their licences accepted on the Hugging Face Hub and `HF_TOKEN` set.
