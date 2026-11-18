#!/usr/bin/env python
"""liveseed_live.py: calibrates LiveSeed on a scored pool and runs it live on new prompts, with time and memory.

It implements Sec. 3.2-3.4 of the paper: the two scoring forms, their calibration on a pool, and the live algorithm
(Algorithm A, App. A). There are two sub-commands.

calibrate
    Fits both forms on a pool (prompts x seeds generated to completion by seed_harness.py with the readings logged, the
    finished images scored by score_images.py). A reading is the normalised attention entropy of one (block, step).
      LiveSeed-Fast  five readings with signs, equal weights, chosen greedily (`subset5` in the code);
      LiveSeed-Max   a ridge regression over every reading at steps <= --cap (`ridge`).
    Input:   <pool>/results*.csv (the readings) and <pool>/scores_finals.csv (the `hpsv2` column, HPSv2.1, is the target).
    Output:  one JSON file (--out), which `run` reads through --maps:
               subset5   [[reading, sign], ...]   reading = "<statistic>_<block>@s<step>", sign = +1 or -1
               single    the best single reading, in the same format
               ridge     {"cols": [reading, ...], "w": [weight, ...], "alpha": the chosen penalty}
             and the model, the pool, the cap, the number of prompts and of candidate readings.

run
    Algorithm A for every prompt of --prompts (a CSV with the columns ID and prompt, and optionally Dataset). The N seeds
    of a prompt are denoised as ONE batch, step by step. At every exit (s_j, k_j) the readings taken so far by the gated
    attention hooks are z-scored over the rows still alive and combined into one score per seed, and the batch is cut
    to its k_j best rows: the latents, the prompt embeddings and the other per-sample tensors are sliced. The survivor is
    finished and decoded. Nothing is decoded early and no evaluator is loaded in the LiveSeed mode.

    --mode liveseed  Exits come from --schedule: "3:1" keeps one seed at step 3; "3:half,8:1" keeps ceil(N/2) at step 3
                     and one at step 8. --form is subset5 (LiveSeed-Fast) or ridge (LiveSeed-Max), read from --maps.
    --mode abss      The baseline ABSS (Zhang, Liu & Hong, NeurIPS 2026) as a batch pruner under its own settings
                     (50 steps, classifier-free guidance (CFG) 7.5): its statistic at step 10, on the cross-attention maps of
                     the layers with (size/32)^2 queries (16x16 at 512 px; U-Nets) or on the image-to-text attention of
                     block 12 (diffusion transformers, DiTs); keep three, finish them, report the one ABSS scores highest. The core words come
                     from --tokens (the format of abss_core_tokens_gpt.py). ABSS obtains them with one GPT call per
                     prompt; its measured duration is passed as --gpt-seconds and recorded beside the timings.
    --mode bestofN   Finish all N, score every image with HPSv2.1 and keep the best. The evaluator is loaded once, before
                     the timed runs, so t_evaluator holds only its forward passes. With --evaluator none all N are
                     finished and decoded without scoring.
    --mode single    One seed to completion: the reference cost.

    Hunyuan-DiT reads the entropy of its self-attention (attn1), takes ABSS's statistic from the cross-attention of block
    12 (attn2), and keeps the pipeline's DDPM scheduler, so each seed's noise generator is carried through pruning.

    Output (--out):
      live_<model>_<mode>[_<form>].csv   One row per (prompt, N, repeat):
          model, mode, form, schedule, prompt_id, N, rep, steps, guidance, size, dtype, hook
          pick                 the seed whose image is kept
          survivors            JSON {exit step: [seeds kept, best first]}
          score_at_<s>         JSON {seed: score} over the rows alive at exit s
          nbar                 effective number of generations: candidate-steps spent / steps (App. D)
          t_<phase>            CUDA-synchronised seconds per phase: encode, denoise_to_exit1, exit<s> (scoring and
                               pruning), denoise_to_exit2, denoise_rest, denoise (no exit), decode, evaluator
          t_total              the sum of the phases
          t_gpt_added          --gpt-seconds in the abss mode, else 0; it is not included in t_total
          torch_max_alloc_GB   torch.cuda.max_memory_allocated over the run of this row
          image                path of the kept image (saved for the first repeat only)
          abss_core_tokens (abss); hps_all or evaluator (bestofN); error (a pool that did not fit on the card)
          offline_pick, pick_equals_offline, t_single_image, clip_cosine_vs_single (with --validate)
        The file is rewritten after every prompt, and rows of an earlier run with other (prompt, N, repeat) are kept.
      summary_<model>_<mode>[_<form>].json   load_s (pipeline load time), device_peak_GB (peak memory in use on the device,
                                             sampled every 50 ms), gpu.
      images/                                the kept images.

    --validate (LiveSeed mode) also (a) recomputes the pick from the readings of the same run with the offline simulator's
    rule and (b) generates the picked seed alone and reports the CLIP image cosine between it and the batched survivor.

Environment: LIVESEED_FUSED=0 (DiTs) or LIVESEED_FUSED_UNET=0 (U-Nets) selects the standard hook (fast_hooks.py), which
forms the attention map, instead of the default fused hook (fused_hooks.py; Sec. 3.1, App. D), which does not.
LIVESEED_HY_PRECISION is read by fused_hooks.py for Hunyuan-DiT and is recorded in the `hook` column.

Usage (from code/liveseed/, one GPU):
  python liveseed_live.py calibrate --model sd14 --pool runs/sd14_pool --out runs/maps_sd14.json
  python liveseed_live.py run --model sd14 --prompts ../../prompts/HPD_photo.csv --maps runs/maps_sd14.json \\
         --mode liveseed --form ridge --schedule 3:1 --N 10 --limit 4 --out runs/live
  python liveseed_live.py calibrate --model sdxl --pool runs/sdxl_pool --cap 8 --first-exit 3 --out runs/maps_sdxl.json
  python liveseed_live.py run --model sdxl --prompts ../../seeds/e2_live_prompts.csv --maps runs/maps_sdxl.json \\
         --mode liveseed --form subset5 --schedule 3:half,8:1 --N 10,20,50 --repeats 3 --out runs/live_sdxl --validate
  python liveseed_live.py run --model sdxl --prompts ../../seeds/e2_live_prompts.csv --mode abss \\
         --tokens runs/abss_tokens.json --N 10 --out runs/live_sdxl
  python liveseed_live.py run --model sdxl --prompts ../../seeds/e2_live_prompts.csv --mode bestofN --N 10 --out runs/live_sdxl
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seed_harness import MODEL_CFG, seeds_for   # noqa: E402

# Model name -> architecture family: "unet" (SD1.x, SD2.1, SDXL), "sd3" (SD3.5), "flux" (FLUX.1), "hunyuan" (Hunyuan-DiT).
FAMILY = {m: c["family"] for m, c in MODEL_CFG.items()}


# =============================================================================================== timing and memory
class Clock:
    """Phase timer that synchronises CUDA at both ends of a phase, so GPU work is charged to the phase that launched it.

    `t` maps a phase name to its accumulated seconds; starting a name again adds to the same entry.
    """

    def __init__(self):
        self.t = {}
        self.cur = None
        self.t0 = None

    def start(self, name):
        torch.cuda.synchronize()
        self.cur, self.t0 = name, time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        self.t[self.cur] = self.t.get(self.cur, 0.0) + time.perf_counter() - self.t0
        self.cur = None

    def total(self):
        return float(sum(self.t.values()))


class DevicePeak:
    """Background thread that samples the memory in use on the device (total - free) every 50 ms and keeps the peak.

    Unlike torch.cuda.max_memory_allocated, this includes what the allocator has reserved and any other process on the card.
    """

    def __init__(self):
        import threading
        self.peak = 0
        self.stop = False
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        while not self.stop:
            free, total = torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)
            time.sleep(0.05)

    def gb(self):
        """Stops the sampling and returns the peak in GiB."""
        self.stop = True
        return self.peak / 2**30


# =============================================================================================== hooks
def install_gated(pipe, family, pairs, order="pooled"):
    """Installs attention hooks that compute only the wanted readings, and returns the object that collects them.

    pairs: the wanted {(attention processor name, step)}. A gated hook leaves the layer's output untouched and computes the
    entropy only when its (name, current step) is wanted, so the readings cost nothing at the other layers and steps.
    order: the U-Net reading. "pooled" (columns `sa_mean_`) averages the attention rows over heads and queries and takes
    one entropy; "entfirst" (`sa_entfirst_`) is the per-row order of Sec. 3.1: the entropy of each row, averaged afterwards.

    The returned object has `.step` (set it to the 1-based step before each denoising step), `.reset()` and
    `.values_per_sample()` -> {(name, step): np.array over the samples of the batch at that step}.

    The fused hooks are the default on every family: they give the same readings within 2e-4 and identical picks, faster.
    """
    import fast_hooks as fh
    # cfg tells the sink whether the batch is [uncond(n), cond(n)], whose two halves are combined into one reading per
    # sample. FLUX.1-dev has a single branch (its guidance is embedded).
    if os.environ.get("LIVESEED_FUSED", "1") == "1" and family in ("sd3", "flux"):
        import fused_hooks as fu
        sink = fh.Sink(set(pairs), cfg=(family == "sd3"))
        fu.install(pipe, family, sink)
        return sink
    if family == "hunyuan":   # self-attention (attn1) of Hunyuan-DiT, fused by default
        import fused_hooks as fu
        sink = fh.Sink(set(pairs), cfg=True)
        (fu.install if os.environ.get("LIVESEED_FUSED", "1") == "1" else fh.install)(pipe, "hunyuan", sink)
        return sink
    if os.environ.get("LIVESEED_FUSED_UNET", "1") == "1" and family == "unet":   # the U-Nets have their own switch
        import fused_hooks as fu
        sink = fh.Sink(set(pairs), cfg=True)
        fu.install(pipe, family, sink, order=order)
        return sink
    if family == "flux":   # the standard (unfused) FLUX hook: the pool harness's collector, restricted to the wanted pairs
        import ae_hook_FLUX as hfx
        col = hfx.AttnEntropyCollector(device="cuda", extra_orders=False, wanted=set(pairs))
        # The single-stream blocks learn the text length from a hooked double-stream block. With only the wanted blocks
        # hooked it must be set by hand: FLUX's T5 sequence is padded to 512 (LiveRun.prepare).
        col.n_txt = 512
        procs = dict(pipe.transformer.attn_processors)
        want_names = {n for n, _ in pairs}
        for name in procs:
            if name in want_names:
                procs[name] = col.get_processor(name)
        pipe.transformer.set_attn_processor(procs)

        class FluxShim:
            """Gives the collector the interface of fast_hooks.Sink (step, reset, values_per_sample)."""

            @property
            def step(self_):
                return col.step

            @step.setter
            def step(self_, v):
                col.step = v

            def reset(self_):
                col.new_image("live")

            def values_per_sample(self_):
                return col.values_per_sample()

        return FluxShim()
    # the standard hook of the U-Nets and SD3.5
    sink = fh.Sink(set(pairs), cfg=True)
    fh.install(pipe, family, sink, order=order)
    return sink


# =============================================================================================== calibration (Sec. 3.3)
def calibrate(args):
    """Fits LiveSeed-Fast and LiveSeed-Max on ALL prompts of a pool and writes the JSON file that `run` reads.

    The candidates are the readings of every logged block at steps <= --cap, z-scored within each prompt. Both forms
    are calibrated for the same objective: the mean over prompts of the HPSv2.1 gain of the top-scored seed over its
    prompt's mean (the within-prompt gain of Sec. 3.3). Nothing is held out here, because the result is applied to other prompts; the held-out
    evaluation of the same procedure is in fit_trajectory_metric.py.
    """
    from fit_trajectory_metric import _load_selection_table, greedy, ridge, sel_objective
    # One row per (prompt, seed): HPSv2.1 of the finished image and one column per reading ("<statistic>_<block>@s<step>").
    # The default reading is sa_mean_ on U-Nets and ae_mean_ on DiTs; --order entfirst asks for the per-row order on U-Nets.
    tbl, feats, metrics = _load_selection_table(
        args.pool, str(Path(args.pool) / "scores_finals.csv"), args.model, args.cap,
        prefixes=("sa_entfirst_",) if getattr(args, "order", "pooled") == "entfirst" else None)
    tbl["image_id"] = tbl["image_id"].astype(str)
    obj = lambda d, s: sel_objective(d, s, "hpsv2")
    # The ridge target is centred within the prompt: the task is to rank seeds of one prompt, not to predict which
    # prompts score high.
    tbl["_t"] = tbl["hpsv2"] - tbl.groupby("image_id")["hpsv2"].transform("mean")

    # LiveSeed-Fast: greedy forward selection of up to five (reading, sign) pairs; `single` is the best one alone.
    ch5, v5 = greedy(tbl, feats, obj, 5)
    ch1, v1 = greedy(tbl, feats, obj, 1)
    # A two-exit schedule scores the batch at its first exit from the readings taken by then, so the subset needs at
    # least one of them. If the greedy choice has none, the best single early reading is put first and the fifth member dropped.
    if args.first_exit and not any(feat_to_pair(f)[1] <= args.first_exit for f, _ in ch5):
        early = [f for f in feats if feat_to_pair(f)[1] <= args.first_exit]
        ce, ve = greedy(tbl, early, obj, 1)
        ch5 = list(ce) + list(ch5)[:4]
        print(f"[calibrate] no reading at step <= {args.first_exit} in the greedy subset: added {ce} (in-sample single gain {ve:+.4f})")

    # LiveSeed-Max: ridge weights over all candidates; the penalty alpha is chosen by an inner cross-validation over prompts.
    w, a = ridge(tbl, feats, "_t", obj)

    out = dict(model=args.model, pool=args.pool, cap=args.cap, n_prompts=int(tbl.image_id.nunique()), n_candidates=len(feats),
               subset5=[(f, int(s)) for f, s in ch5], subset5_insample_gain=float(v5), single=[(f, int(s)) for f, s in ch1],
               ridge=dict(cols=feats, w=[float(x) for x in w], alpha=float(a)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[calibrate] {args.model} cap {args.cap}: subset {out['subset5']} (in-sample gain {v5:+.4f}); ridge alpha {a}, {len(feats)} readings -> {args.out}")


def feat_to_pair(f):
    """Splits a reading's column name into its block and step: 'sa_mean_<block>@s<k>' -> ('<block>', k).

    <block> is the attention processor's name with underscores for dots (resolve_names maps it back).
    """
    col, st = f.split("@s")
    lay = col.split("_entfirst_", 1)[1] if col.startswith("sa_entfirst_") else col.split("_mean_", 1)[1]
    return lay, int(st)


def resolve_names(pipe, family, lays):
    """Maps block names as written in the column names to the model's attention processor names: {block: processor name}.

    The pool's columns replace the dots of a processor name with underscores and may drop the '_processor' suffix, so
    both sides are compared in that reduced spelling.
    """
    module = pipe.transformer if family in ("sd3", "flux", "hunyuan") else pipe.unet
    names = list(module.attn_processors)
    key = {n.replace(".", "_").replace("_processor", ""): n for n in names}
    out = {}
    for lay in lays:
        k = lay.replace(".", "_").replace("_processor", "")
        if k not in key:
            raise SystemExit(f"reading {lay} names no attention processor of {family}")
        out[lay] = key[k]
    return out


# =============================================================================================== the live batch
class LiveRun:
    """One prompt and N seeds denoised as one batch that can be pruned between steps.

    It re-implements the denoising loop of each diffusers pipeline (encode, step, decode) so that rows can be dropped
    from the batch mid-trajectory, which the pipelines' own __call__ does not allow. State:
      latents    the rows still alive, in batch order
      cond       the per-sample conditioning tensors of those rows
      alive      for each batch row, its index into `seeds`
    """

    def __init__(self, pipe, model, prompt, seeds, steps, guidance, size, dtype):
        self.pipe, self.model, self.family = pipe, model, FAMILY[model]
        self.prompt, self.seeds, self.steps, self.guidance, self.size, self.dtype = prompt, list(seeds), steps, guidance, size, dtype
        self.device = torch.device("cuda")        # diffusers' prepare_latents reads device.type
        self.alive = list(range(len(seeds)))      # indices into seeds, in batch order
        self.readings = {}                        # not used: the runner keeps the readings itself (see run)
        self.events = []                          # not used

    # ---- encode: the prompt's embeddings repeated for N samples, one initial latent per seed, the timesteps
    def prepare(self):
        """Encodes the prompt for N samples, draws each seed's initial latent and sets the timesteps, per model family.

        Each seed has its own generator, so a row's latent is the one that seed gives when generated alone. Under
        CFG every conditioning tensor is stored as [unconditional(N), conditional(N)].
        """
        p, N = self.pipe, len(self.seeds)
        gens = [torch.Generator(device=self.device).manual_seed(int(s)) for s in self.seeds]
        H = W = self.size
        if self.family == "unet" and self.model != "sdxl":   # SD1.x, SD2.1: one text encoder
            pe, ne = p.encode_prompt(self.prompt, self.device, N, True, None)
            self.cond = dict(prompt_embeds=torch.cat([ne, pe]))
            self.latents = p.prepare_latents(N, p.unet.config.in_channels, H, W, pe.dtype, self.device, gens)
            p.scheduler.set_timesteps(self.steps, device=self.device)
            self.timesteps = p.scheduler.timesteps
        elif self.family == "unet":   # SDXL: also the pooled text embedding and the size/crop conditioning (time ids)
            pe, ne, pooled, npooled = p.encode_prompt(prompt=self.prompt, device=self.device, num_images_per_prompt=N,
                                                      do_classifier_free_guidance=True)
            add_time_ids = p._get_add_time_ids((H, W), (0, 0), (H, W), dtype=pe.dtype,
                                               text_encoder_projection_dim=p.text_encoder_2.config.projection_dim)
            add_time_ids = add_time_ids.to(self.device).repeat(N, 1)
            self.cond = dict(prompt_embeds=torch.cat([ne, pe]), add_text_embeds=torch.cat([npooled, pooled]),
                             add_time_ids=torch.cat([add_time_ids, add_time_ids]))
            self.latents = p.prepare_latents(N, p.unet.config.in_channels, H, W, pe.dtype, self.device, gens)
            p.scheduler.set_timesteps(self.steps, device=self.device)
            self.timesteps = p.scheduler.timesteps
        elif self.family == "sd3":   # SD3.5: text sequence (CLIP, then T5 padded to 256 tokens) and pooled embedding
            pe, ne, pooled, npooled = p.encode_prompt(prompt=self.prompt, prompt_2=None, prompt_3=None, device=self.device,
                                                      num_images_per_prompt=N, do_classifier_free_guidance=True,
                                                      max_sequence_length=256)
            self.cond = dict(prompt_embeds=torch.cat([ne, pe]), pooled=torch.cat([npooled, pooled]))
            self.latents = p.prepare_latents(N, p.transformer.config.in_channels, H, W, pe.dtype, self.device, gens)
            p.scheduler.set_timesteps(self.steps, device=self.device)
            self.timesteps = p.scheduler.timesteps
        elif self.family == "hunyuan":   # follows HunyuanDiTPipeline.__call__ (diffusers 0.32.2) with N copies of the prompt
            from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import get_resize_crop_region_for_grid, get_2d_rotary_pos_embed
            # two text encoders: BERT (77 tokens) and mT5 (256 tokens), each with its attention mask
            pe, ne, pm, nm = p.encode_prompt(prompt=self.prompt, device=self.device, dtype=p.transformer.dtype,
                                             num_images_per_prompt=N, do_classifier_free_guidance=True,
                                             max_sequence_length=77, text_encoder_index=0)
            pe2, ne2, pm2, nm2 = p.encode_prompt(prompt=self.prompt, device=self.device, dtype=p.transformer.dtype,
                                                 num_images_per_prompt=N, do_classifier_free_guidance=True,
                                                 max_sequence_length=256, text_encoder_index=1)
            p.scheduler.set_timesteps(self.steps, device=self.device)
            self.timesteps = p.scheduler.timesteps
            self.latents = p.prepare_latents(N, p.transformer.config.in_channels, H, W, pe.dtype, self.device, gens)
            self.gens = gens                     # DDPM draws step noise from each seed's own generator: kept per alive row
            # rotary position embedding of the latent grid and the size conditioning, as the pipeline builds them
            ps = p.transformer.config.patch_size
            gh, gw = H // 8 // ps, W // 8 // ps
            crops = get_resize_crop_region_for_grid((gh, gw), 512 // 8 // ps)
            self.rope = get_2d_rotary_pos_embed(p.transformer.inner_dim // p.transformer.num_heads, crops, (gh, gw),
                                                device=self.device, output_type="pt")
            add_time_ids = torch.tensor([list((1024, 1024) + (H, W) + (0, 0))], dtype=pe.dtype)
            self.cond = dict(prompt_embeds=torch.cat([ne, pe]).to(self.device), mask=torch.cat([nm, pm]).to(self.device),
                             prompt_embeds_2=torch.cat([ne2, pe2]).to(self.device), mask_2=torch.cat([nm2, pm2]).to(self.device),
                             add_time_ids=torch.cat([add_time_ids] * 2).to(dtype=pe.dtype, device=self.device).repeat(N, 1),
                             style=torch.tensor([0, 0], device=self.device).repeat(N))
        else:   # flux: no CFG batch (the guidance scale is an input of the transformer); T5 sequence padded to 512 tokens
            from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
            pe, pooled, text_ids = p.encode_prompt(prompt=self.prompt, prompt_2=None, device=self.device,
                                                   num_images_per_prompt=N, max_sequence_length=512)
            self.cond = dict(prompt_embeds=pe, pooled=pooled, text_ids=text_ids)
            self.latents, self.latent_image_ids = p.prepare_latents(N, p.transformer.config.in_channels // 4, H, W, pe.dtype,
                                                                    self.device, gens)
            # the resolution-dependent timestep shift of FluxPipeline.__call__
            sigmas = np.linspace(1.0, 1 / self.steps, self.steps)
            mu = calculate_shift(self.latents.shape[1], p.scheduler.config.get("base_image_seq_len", 256),
                                 p.scheduler.config.get("max_image_seq_len", 4096),
                                 p.scheduler.config.get("base_shift", 0.5), p.scheduler.config.get("max_shift", 1.16))
            self.timesteps, _ = retrieve_timesteps(p.scheduler, self.steps, self.device, None, sigmas, mu=mu)
            self.guidance_t = (torch.full([1], self.guidance, device=self.device, dtype=torch.float32).expand(N)
                               if p.transformer.config.guidance_embeds else None)

    # ---- denoise: one step on the rows alive (index i, 0-based)
    @torch.no_grad()
    def step(self, i):
        """Advances every alive row by denoising step i (0-based). The installed hooks take their readings during this call.

        With CFG the denoiser sees [latents, latents] against [unconditional, conditional] and the two predictions are
        combined as u + guidance * (c - u).
        """
        p, t = self.pipe, self.timesteps[i]
        n = self.latents.shape[0]
        if self.family == "unet":
            lmi = p.scheduler.scale_model_input(torch.cat([self.latents] * 2), t)
            kw = dict(encoder_hidden_states=self.cond["prompt_embeds"], return_dict=False)
            if self.model == "sdxl":
                kw["added_cond_kwargs"] = {"text_embeds": self.cond["add_text_embeds"], "time_ids": self.cond["add_time_ids"]}
            noise = p.unet(lmi, t, **kw)[0]
            u, c = noise.chunk(2)
            noise = u + self.guidance * (c - u)
            self.latents = p.scheduler.step(noise, t, self.latents, return_dict=False)[0]
        elif self.family == "sd3":
            lmi = torch.cat([self.latents] * 2)
            ts = t.expand(lmi.shape[0])
            noise = p.transformer(hidden_states=lmi, timestep=ts, encoder_hidden_states=self.cond["prompt_embeds"],
                                  pooled_projections=self.cond["pooled"], return_dict=False)[0]
            u, c = noise.chunk(2)
            noise = u + self.guidance * (c - u)
            self.latents = p.scheduler.step(noise, t, self.latents, return_dict=False)[0]
        elif self.family == "hunyuan":
            lmi = p.scheduler.scale_model_input(torch.cat([self.latents] * 2), t)
            t_expand = torch.tensor([t] * lmi.shape[0], device=self.device).to(dtype=lmi.dtype)
            noise = p.transformer(lmi, t_expand, encoder_hidden_states=self.cond["prompt_embeds"], text_embedding_mask=self.cond["mask"],
                                  encoder_hidden_states_t5=self.cond["prompt_embeds_2"], text_embedding_mask_t5=self.cond["mask_2"],
                                  image_meta_size=self.cond["add_time_ids"], style=self.cond["style"], image_rotary_emb=self.rope,
                                  return_dict=False)[0]
            noise, _ = noise.chunk(2, dim=1)                                   # the learned-sigma half is dropped, as the pipeline
            u, c = noise.chunk(2)
            noise = u + self.guidance * (c - u)
            # the DDPM step adds noise: each row draws it from its own seed's generator
            self.latents = p.scheduler.step(noise, t, self.latents, generator=self.gens, return_dict=False)[0]
        else:   # flux: one branch; the pipeline passes the timestep divided by 1000
            ts = t.expand(n).to(self.latents.dtype)
            noise = p.transformer(hidden_states=self.latents, timestep=ts / 1000, guidance=self.guidance_t,
                                  pooled_projections=self.cond["pooled"], encoder_hidden_states=self.cond["prompt_embeds"],
                                  txt_ids=self.cond["text_ids"], img_ids=self.latent_image_ids, return_dict=False)[0]
            self.latents = p.scheduler.step(noise, t, self.latents, return_dict=False)[0]

    # ---- prune: keep the given batch rows
    def keep(self, rows):
        """Cuts the batch to the given rows (indices into the current batch) and updates `alive`.

        Only tensors with one entry per sample are sliced. The Euler and flow-matching Euler schedulers keep no
        per-sample state, so the survivors continue on the trajectories they were on. Hunyuan-DiT's DDPM scheduler
        draws noise at every step, so its per-seed generators are sliced as well.
        """
        rows = [int(r) for r in rows]
        n = self.latents.shape[0]
        idx = torch.tensor(rows, device=self.device)
        self.latents = self.latents[idx]
        if self.family in ("unet", "sd3", "hunyuan"):   # CFG batches are [uncond(n), cond(n)]
            both = torch.cat([idx, idx + n])            # the same rows in both halves
            for k in list(self.cond):
                if k in ("prompt_embeds", "add_text_embeds", "add_time_ids", "pooled", "mask", "prompt_embeds_2", "mask_2", "style"):
                    self.cond[k] = self.cond[k][both]
            if self.family == "hunyuan":
                self.gens = [self.gens[r] for r in rows]
        else:   # flux: text_ids and latent_image_ids have no batch axis and are left as they are
            for k in ("prompt_embeds", "pooled"):
                self.cond[k] = self.cond[k][idx]
            if self.guidance_t is not None:
                self.guidance_t = self.guidance_t[: len(rows)]
        self.alive = [self.alive[r] for r in rows]

    @torch.no_grad()
    def decode(self):
        """Decodes the alive rows with the VAE and returns a list of PIL images in batch order."""
        p = self.pipe
        lat = self.latents
        if self.family == "unet":
            # fp16 overflows in the SDXL VAE, so it is upcast to fp32 for the decode and cast back afterwards
            if self.model == "sdxl" and p.vae.dtype == torch.float16 and p.vae.config.force_upcast:
                p.upcast_vae()
                lat = lat.to(next(iter(p.vae.post_quant_conv.parameters())).dtype)
            img = p.vae.decode(lat / p.vae.config.scaling_factor, return_dict=False)[0]
            if self.model == "sdxl" and p.vae.dtype == torch.float32:
                p.vae.to(dtype=torch.float16)
        elif self.family == "sd3":
            img = p.vae.decode((lat / p.vae.config.scaling_factor) + p.vae.config.shift_factor, return_dict=False)[0]
        elif self.family == "hunyuan":   # as the pipeline: postprocess on the VAE's own dtype
            img = p.vae.decode(lat / p.vae.config.scaling_factor, return_dict=False)[0]
            return p.image_processor.postprocess(img, output_type="pil", do_denormalize=[True] * img.shape[0])
        else:   # flux: the latents are a sequence of 2x2 patches and are unpacked to a grid first
            lat = p._unpack_latents(lat, self.size, self.size, p.vae_scale_factor)
            img = p.vae.decode((lat / p.vae.config.scaling_factor) + p.vae.config.shift_factor, return_dict=False)[0]
        return p.image_processor.postprocess(img.float(), output_type="pil")


# =============================================================================================== the score at an exit (Sec. 3.2)
def liveseed_score(readings_by_pair, alive_seeds, pairs_feats, form, maps):
    """Returns one score per alive seed, in the order of `alive_seeds`; the highest-scoring seeds are kept.

    readings_by_pair  {(processor name, step): {seed: reading}}, holding every seed that was alive when that reading was
                      taken. The readings are looked up by seed and not by batch row, because the survivors of an
                      earlier exit are an arbitrary subset of the rows the reading was taken on.
    alive_seeds       the seeds of the rows alive now, in batch order.
    pairs_feats       [(reading's column name, processor name, step)] for the readings available at this exit.
    form, maps        "subset5" or "ridge", and the calibration JSON written by `calibrate`.
    """
    # the matrix of readings: one row per alive seed, one column per reading
    Z = []
    for f, name, st in pairs_feats:
        d = readings_by_pair[(name, st)]
        Z.append(np.asarray([d[sd] for sd in alive_seeds], float))
    Z = np.stack(Z, 1)                                        # [alive, R]

    # Each reading is z-scored over the seeds alive NOW (mean and population standard deviation of its column). The
    # entropy of different blocks and steps lives on different scales, and only differences between seeds of this prompt
    # matter. Calibration z-scores within each prompt's pool in the same way (with the sample standard deviation, a
    # common factor that leaves the ranking unchanged).
    Zs = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-8)

    # LiveSeed-Max: the ridge weights of the readings available at this exit (the weights of later steps are left out)
    if form == "ridge":
        w = np.array([maps["ridge"]["w"][maps["ridge"]["cols"].index(f)] for f, _, _ in pairs_feats])
        return Zs @ w

    # LiveSeed-Fast: the signed sum of the chosen readings; a sign of -1 scores lower entropy higher. The paper writes
    # the mean, which ranks the seeds identically.
    sc = np.zeros(len(alive_seeds))
    sign = {f: s for f, s in maps["subset5"]}
    for j, (f, _, _) in enumerate(pairs_feats):
        if f in sign:
            sc += sign[f] * Zs[:, j]
    return sc


# =============================================================================================== the runner (Sec. 3.4, Algorithm A)
def run(args):
    """Runs one mode on every prompt of --prompts, for every pool size of --N and every repeat, and records time and memory.

    For each (prompt, N, repeat) the steps of Algorithm A are marked (1)-(6) below: encode, denoise all rows to the
    exit step while the hooks read, score, keep the best, finish the survivors, decode. The baselines share the loop:
    `abss` has one exit that keeps three, `bestofN` and `single` have none.
    """
    import seed_harness as sh
    cfg = MODEL_CFG[args.model]
    family = cfg["family"]
    # Settings: the model's own unless given; ABSS runs under its paper's (50 steps, guidance 7.5). FLUX runs in bf16.
    dtype = torch.bfloat16 if (args.dtype == "bf16" or (args.dtype == "auto" and family == "flux")) else torch.float16
    steps = args.steps or (50 if args.mode == "abss" else (28 if args.model == "flux" else 30))
    guidance = args.guidance if args.guidance is not None else (7.5 if args.mode == "abss" else cfg["guidance"])
    size = args.size or cfg["size"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    prompts = pd.read_csv(args.prompts, encoding="utf-8-sig")
    prompts.columns = [c.strip() for c in prompts.columns]
    if args.limit:
        prompts = prompts.head(args.limit)
    Ns = [int(x) for x in args.N.split(",")]

    # ---- load the pipeline, and for ABSS its cross-attention hook
    torch.cuda.init()
    dev = DevicePeak()
    torch.cuda.reset_peak_memory_stats()
    clock = Clock()
    clock.start("load")
    pipe = sh_load_plain(args.model, dtype)
    col = None
    if args.mode == "abss":
        # ABSS reads the layers whose maps have (size/32)^2 queries: 16x16 at 512 px
        col = ABSSHook(pipe, family, step=args.abss_step, queries=(size // 32) ** 2, block=args.abss_block)
    clock.stop()
    t_load = clock.t["load"]

    # ---- LiveSeed: which readings the chosen form needs, and gated hooks for exactly those
    sched = parse_schedule(args.schedule)             # [(exit step, number kept or "half"), ...]
    pairs_feats, sink = [], None                      # pairs_feats: [(reading's column name, processor name, step)]
    if args.mode == "liveseed":
        maps = json.load(open(args.maps))
        cap = sched[-1][0]                            # no reading is needed after the last exit
        feats = [f for f, _ in maps["subset5"]] if args.form == "subset5" else [f for f in maps["ridge"]["cols"]]
        feats = [f for f in feats if feat_to_pair(f)[1] <= cap]
        names = resolve_names(pipe, family, [feat_to_pair(f)[0] for f in feats])
        pairs_feats = [(f, names[feat_to_pair(f)[0]], feat_to_pair(f)[1]) for f in feats]
        # the entropy order the calibration used is read off the column names (U-Nets only)
        order = "entfirst" if any(f.startswith("sa_entfirst_") for f in feats) else "pooled"
        sink = install_gated(pipe, family, {(n, s) for _, n, s in pairs_feats}, order=order)
        print(f"[live] reading {'ae_mean_ (row entropies)' if family == 'hunyuan' else 'order ' + order}, hook {hook_kind(family)}", flush=True)
        print(f"[live] {args.mode}/{args.form}: {len(pairs_feats)} readings on {len({n for _, n, _ in pairs_feats})} blocks, schedule {sched}", flush=True)

    # ---- what the baselines need: ABSS's core words per prompt, the evaluator of best-of-N, CLIP for --validate
    tokens = json.load(open(args.tokens)) if args.mode == "abss" else {}
    scorer = None
    if args.mode == "bestofN" and args.evaluator != "none":
        clock.start("evaluator_load")
        scorer = load_hps()
        clock.stop()
    clip_img = load_clip_image() if args.validate else None

    records = []
    for _, r in prompts.iterrows():
        pid, prompt = str(r["ID"]), str(r["prompt"]).strip()
        ds = str(r.get("Dataset", "DrawBench"))
        for N in Ns:
            # the seeds are a deterministic function of the prompt (the pool harness's rule) unless --seedlist gives them
            seeds = seeds_for(f"{ds}:{pid}:{prompt}", N) if not args.seedlist else [int(x) for x in args.seedlist.split(",")][:N]
            if args.mode == "single":
                seeds = seeds[:1]              # the reference: one seed to completion, no selection
            for rep in range(args.repeats):
              try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                ck = Clock()
                rec = dict(model=args.model, mode=args.mode, form=args.form if args.mode == "liveseed" else "",
                           schedule=args.schedule if args.mode == "liveseed" else "",
                           prompt_id=pid, N=N, rep=rep, steps=steps, guidance=guidance, size=size, dtype=str(dtype).split(".")[-1],
                           hook=hook_kind(family))

                # (1) encode the prompt and draw the N initial latents
                lr = LiveRun(pipe, args.model, prompt, seeds, steps, guidance, size, dtype)
                ck.start("encode")
                lr.prepare()
                ck.stop()

                # exits: {step: number kept}. LiveSeed takes its schedule; ABSS keeps three at its read step; else none.
                exits = dict(sched) if args.mode == "liveseed" else ({args.abss_step: 3} if args.mode == "abss" else {})
                if sink is not None:
                    sink.reset()
                if col is not None:
                    col.new_image("live", n_active_tokens=None)
                survivors = {}                 # exit step -> the seeds kept there, best first
                readings = {}                  # (processor name, step) -> {seed: reading}, for every seed alive at that step
                phase = "denoise_to_exit1" if exits else "denoise"
                ck.start(phase)
                for i in range(steps):
                    # (2) one denoising step on every alive row; the hooks are told the 1-based step and read during it
                    if sink is not None:
                        sink.step = i + 1
                    if col is not None:
                        col.step = i + 1
                    lr.step(i)
                    s = i + 1
                    if s in exits:
                        ck.stop()
                        ck.start(f"exit{s}")
                        k = exits[s]
                        k = math.ceil(len(lr.alive) / 2) if k == "half" else int(k)
                        if args.mode == "liveseed":
                            # (3) score: collect the readings up to step s, keyed by seed, then z-score and combine them
                            R = per_row_readings(lr, pairs_feats, s, sink, family)        # {(name, step): array over the rows alive when read}
                            for key_, v in R.items():
                                if key_ in readings:
                                    continue                                              # read at an earlier exit, over the batch alive then
                                assert len(v) == len(lr.alive), f"reading {key_}: {len(v)} values for {len(lr.alive)} alive rows"
                                readings[key_] = {lr.seeds[lr.alive[i]]: float(x) for i, x in enumerate(v)}
                            score = liveseed_score(readings, [lr.seeds[i] for i in lr.alive], [pf for pf in pairs_feats if pf[2] <= s],
                                                   args.form, maps)
                        else:   # abss
                            score = abss_batch_score(lr, col, tokens.get(pid, {}), prompt, args, family, size)
                            rec["abss_core_tokens"] = col.last_core
                        # (4) keep the k best rows; the stable sort breaks ties in favour of the earlier row
                        order = np.argsort(-score, kind="stable")[:k]
                        survivors[s] = [lr.seeds[lr.alive[j]] for j in order]
                        rec[f"score_at_{s}"] = json.dumps({str(lr.seeds[lr.alive[j]]): round(float(score[j]), 6) for j in range(len(lr.alive))})
                        lr.keep(sorted(order.tolist()))                  # keep batch order of the survivors
                        # (5) the following steps of this loop finish the survivors (or take them to the next exit)
                        ck.stop()
                        ck.start("denoise_rest" if len(lr.alive) == 1 or s == max(exits) else "denoise_to_exit2")
                ck.stop()

                # (6) decode what is left: the survivor(s), or all N for best-of-N
                ck.start("decode")
                imgs = lr.decode()
                ck.stop()
                pick = lr.seeds[lr.alive[0]]
                if args.mode == "abss":   # ABSS's own top-1 among the three it kept: its highest score
                    pick = survivors[args.abss_step][0]
                    imgs = [imgs[lr.alive.index(lr.seeds.index(pick))]]
                if args.mode == "bestofN" and scorer is not None:
                    ck.start("evaluator")
                    hs = [scorer(im, prompt) for im in imgs]
                    ck.stop()
                    j = int(np.argmax(hs))
                    pick = lr.seeds[lr.alive[j]]
                    imgs = [imgs[j]]
                    rec["hps_all"] = json.dumps([round(float(h), 5) for h in hs])
                elif args.mode == "bestofN":   # --evaluator none: all N finished and decoded, none scored (the cost of generation alone)
                    rec["evaluator"] = "none"
                    imgs = imgs[:1]

                # ---- record: the pick, the cost in candidate-steps (nbar), the time of every phase, the peak memory
                img_path = out / "images" / f"{ds}_{pid}_N{N}_{args.mode}{('_' + args.form) if args.form and args.mode == 'liveseed' else ''}_seed{pick}.png"
                if rep == 0:
                    imgs[0].save(img_path)
                nb = nbar_of(N, [(s, len(v)) for s, v in sorted(survivors.items())], steps) if survivors else (N if args.mode == "bestofN" else 1)
                rec.update(dict(pick=pick, survivors=json.dumps({str(k): v for k, v in survivors.items()}), nbar=round(nb, 3),
                                **{f"t_{k}": round(v, 4) for k, v in ck.t.items()},
                                t_total=round(ck.total(), 4), t_gpt_added=args.gpt_seconds if args.mode == "abss" else 0.0,
                                torch_max_alloc_GB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                                image=str(img_path) if rep == 0 else ""))

                if args.validate and rep == 0 and args.mode == "liveseed":
                    # (a) offline pick from the same readings (the simulator's rule applied to the full-batch readings of the exit steps)
                    rec["offline_pick"] = offline_pick(readings, lr, pairs_feats, sched, args.form, maps, seeds)
                    rec["pick_equals_offline"] = int(rec["offline_pick"] == pick)
                    # (b) the survivor vs the same seed generated alone
                    ck2 = Clock()
                    ck2.start("single")
                    single = LiveRun(pipe, args.model, prompt, [pick], steps, guidance, size, dtype)
                    single.prepare()
                    if sink is not None:
                        sink.reset()
                    for i in range(steps):
                        if sink is not None:
                            sink.step = i + 1
                        single.step(i)
                    simg = single.decode()[0]
                    ck2.stop()
                    rec["t_single_image"] = round(ck2.t["single"], 4)
                    rec["clip_cosine_vs_single"] = round(float(clip_img(imgs[0], simg)), 5)
                    simg.save(out / "images" / f"{ds}_{pid}_single_seed{pick}.png")
                records.append(rec)
                print(f"[live] {pid} N={N} rep={rep} {args.mode}: pick {pick} total {ck.total():.2f}s {dict((k, round(v, 2)) for k, v in ck.t.items())} nbar {nb:.2f}" +
                      (f" offline={rec.get('pick_equals_offline')} cos={rec.get('clip_cosine_vs_single')}" if args.validate and args.mode == 'liveseed' else ""), flush=True)
              except (torch.OutOfMemoryError, RuntimeError) as e:        # a pool too large for the card: record it and go on with the next size
                # The standard FLUX hook (ae_hook_FLUX) catches its own out-of-memory error and logs "stats failed", so the
                # reading is missing and per_row_readings raises "the gate did not fire"; that is the same condition and is
                # skipped the same way. Any other RuntimeError is raised again.
                if isinstance(e, RuntimeError) and not isinstance(e, torch.OutOfMemoryError) and "gate did not fire" not in str(e):
                    raise
                torch.cuda.empty_cache()
                err = "OOM" if isinstance(e, torch.OutOfMemoryError) else "hook OOM (gate did not fire)"
                records.append(dict(model=args.model, mode=args.mode, form=args.form if args.mode == "liveseed" else "",
                                    schedule=args.schedule if args.mode == "liveseed" else "", prompt_id=pid, N=N, rep=rep, error=err))
                print(f"[live] {pid} N={N} rep={rep} {args.mode}: {err}, skipped", flush=True)
                break

        # ---- write the table after every prompt, so an interrupted run keeps its rows
        _csv = out / f"live_{args.model}_{args.mode}{('_' + args.form) if args.mode == 'liveseed' else ''}.csv"
        _new = pd.DataFrame(records)
        # Pool sizes may be run as separate jobs that write the same file, so an existing file is merged instead of
        # overwritten: its rows are kept unless this run has the same (prompt, N, repeat), which replaces them.
        if _csv.exists() and len(_new):
            _old = pd.read_csv(_csv)
            _keys = [c for c in ("prompt_id", "N", "rep") if c in _old.columns and c in _new.columns]
            for _k in _keys:   # the csv may hold the ids as text
                _old[_k] = _old[_k].astype(str)
                _new[_k] = _new[_k].astype(str)
            _old = _old.merge(_new[_keys].drop_duplicates(), on=_keys, how="left", indicator=True)
            _old = _old[_old._merge == "left_only"].drop(columns="_merge")
            _new = pd.concat([_old, _new], ignore_index=True)
        _new.to_csv(_csv, index=False)

    rec_summary = dict(model=args.model, mode=args.mode, load_s=round(t_load, 2), device_peak_GB=round(dev.gb(), 2),
                       gpu=torch.cuda.get_device_name(0))
    json.dump(rec_summary, open(out / f"summary_{args.model}_{args.mode}{('_' + args.form) if args.mode == 'liveseed' else ''}.json", "w"), indent=1)
    print("[live] done", rec_summary)


def sh_load_plain(model, dtype):
    """Loads the diffusers pipeline of `model` on the GPU without any hook (the gated ones are installed afterwards).

    SD1.x and SD2.1 get the Euler scheduler; the other models keep their pipeline's own (Hunyuan-DiT: DDPM).
    """
    import seed_harness as sh
    from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, EulerDiscreteScheduler
    cfg = sh.MODEL_CFG[model]
    kw = dict(torch_dtype=dtype, use_safetensors=True)
    if cfg["variant"]:
        kw["variant"] = cfg["variant"]
    if cfg["family"] == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(cfg["repo"], **kw)
    elif cfg["family"] == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(cfg["repo"], torch_dtype=dtype)
    elif cfg["family"] == "hunyuan":
        from diffusers import HunyuanDiTPipeline
        pipe = HunyuanDiTPipeline.from_pretrained(cfg["repo"], torch_dtype=dtype)
    else:
        pipe = DiffusionPipeline.from_pretrained(cfg["repo"], **kw)
        if model in ("sd14", "sd15", "sd21"):
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def parse_schedule(s):
    """Parses --schedule: "3:half,8:1" -> [(3, "half"), (8, 1)], a list of (exit step, number kept) in the order given."""
    out = []
    for part in (s or "").split(","):
        if not part.strip():
            continue
        st, k = part.split(":")
        out.append((int(st), "half" if k.strip() == "half" else int(k)))
    return out


def nbar_of(N, exits, T):
    """Effective number of generations of a schedule: the candidate-steps spent, divided by the T steps of one image.

    exits: [(step, number kept)] in increasing step order. N rows run to the first exit, the kept rows to the next,
    and the last survivors to step T (the cost C of Sec. 3.4, over T).
    """
    cost, alive, prev = 0, N, 0
    for s, k in exits:
        cost += alive * (s - prev)
        alive, prev = k, s
    return (cost + alive * (T - prev)) / T


# ----------------------------------------------------------------------------------------------- per-row readings
def per_row_readings(lr, pairs_feats, s, sink, family):
    """Returns {(processor name, step): array} for every wanted reading at steps <= s.

    Each array holds one value per batch row, in the batch order of the rows that were alive when the reading was taken;
    a reading taken before an earlier exit therefore has more entries than there are rows now. A missing reading means
    that the hook's gate did not fire, which is raised as an error.
    """
    vals = sink.values_per_sample()
    out = {}
    for f, name, st in pairs_feats:
        if st <= s:
            if (name, st) not in vals:
                raise RuntimeError(f"no reading for {(name, st)}: the gate did not fire (check the processor name and the step counter)")
            out[(name, st)] = vals[(name, st)]
    return out


def offline_pick(readings, lr, pairs_feats, sched, form, maps, seeds):
    """Recomputes the pick from the logged readings alone, with the rule of the offline simulator (--validate).

    readings: {(processor name, step): {seed: reading}} as the live run stored them. Rows that were pruned before a step
    have no reading there; the rule only ever scores the survivors, exactly as Algorithm A does. Returns the picked seed.
    """
    alive = list(seeds)
    for s, k in sched:
        k = math.ceil(len(alive) / 2) if k == "half" else int(k)
        feats = [pf for pf in pairs_feats if pf[2] <= s]
        sc = liveseed_score(readings, alive, feats, form, maps)
        order = np.argsort(-sc, kind="stable")[:k]
        alive = [alive[j] for j in sorted(order.tolist())]
        if k == 1:
            break
    return alive[0]


# =============================================================================================== the ABSS baseline in a batch
# ABSS (Zhang, Liu & Hong, NeurIPS 2026) ranks seeds by the cross-attention mass on the prompt's core words at one early
# step. The classes below wrap the model's own attention processors: the wrapped processor (the attribute `fused`,
# PyTorch's scaled-dot-product attention) still produces the layer's output, and at the read step the wrapper recomputes
# the attention probabilities it needs from the same queries and keys.
class ABSSHook:
    """Collects, at ONE step, the attention ABSS's statistic reads, for every sample of the batch.

    U-Nets: the conditional-branch cross-attention maps, averaged over heads, of the layers with `queries` queries
    (`maps`). DiTs: the image-to-text attention of block `block`, averaged over heads and image tokens (`vecs`).
    Only those layers are hooked and only that step is computed, so ABSS is charged its own read and no more; the pool
    harness's logging hooks, which read every layer at every step, would overstate its cost.
    """

    def __init__(self, pipe, family, step, queries, block=12):
        self.family, self.read_step, self.queries, self.block = family, step, queries, block
        self.step = 1                 # the current 1-based denoising step, set by the runner
        self.maps = {}                # U-Nets: {layer name: (n, N_q, N_k) tensor}
        self.vecs = {}                # DiTs: {layer name: (n, N_txt) array}
        self.last_core = ""           # the core words of the last prompt scored, for the record
        module = pipe.transformer if family in ("sd3", "flux", "hunyuan") else pipe.unet
        procs = dict(module.attn_processors)
        n = 0
        for name, proc in procs.items():
            if family == "unet" and name.endswith("attn2.processor"):   # every cross-attention layer; the query count is checked at call time
                procs[name] = _ABSSUnetCA(self, name, proc)
                n += 1
            elif family == "sd3" and name == f"transformer_blocks.{block}.attn.processor":
                procs[name] = _ABSSSD3(self, name, proc)
                n += 1
            elif family == "flux" and name == f"transformer_blocks.{block}.attn.processor":
                procs[name] = _ABSSFlux(self, name, proc)
                n += 1
            elif family == "hunyuan" and name == f"blocks.{block}.attn2.processor":
                procs[name] = _ABSSHunyuan(self, name, proc)
                n += 1
        module.set_attn_processor(procs)
        print(f"[live] ABSS hook on {n} layers, read at step {step}", flush=True)

    def new_image(self, key, n_active_tokens=None):
        self.maps, self.vecs = {}, {}
        self.step = 1

    def flush_ca_maps(self):
        """Returns the stored U-Net maps and clears them."""
        m, self.maps = self.maps, {}
        return m

    def flush_vecs(self):
        """Returns the stored DiT text vectors and clears them."""
        v, self.vecs = self.vecs, {}
        return v


class _ABSSUnetCA:
    """U-Net cross-attention layer. At the read step, for the layers with the wanted query count, it stores the
    conditional-branch attention map of each sample, averaged over heads: (n, N_q, N_k)."""

    def __init__(self, parent, name, fused):
        self.parent, self.name, self.fused = parent, name, fused

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if self.parent.step == self.parent.read_step and encoder_hidden_states is not None:
            with torch.no_grad():
                hs = hidden_states
                if hs.ndim == 4:      # a (B, C, H, W) feature map: flatten to (B, H*W, C) queries
                    B, C, Hh, Ww = hs.shape
                    hs = hs.view(B, C, Hh * Ww).transpose(1, 2)
                B, N_q, _ = hs.shape
                if N_q == self.parent.queries:
                    q = attn.head_to_batch_dim(attn.to_q(hs))                        # (B*H, N_q, d)
                    k = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))     # (B*H, N_k, d)
                    H = q.shape[0] // B
                    # attention probabilities in fp32: softmax(scale * q k^T) over the text tokens
                    p = torch.baddbmm(torch.empty(q.shape[0], N_q, k.shape[1], dtype=torch.float32, device=q.device),
                                      q.float(), k.float().transpose(-1, -2), beta=0, alpha=attn.scale).softmax(-1)
                    n = B // 2 if B % 2 == 0 and B >= 2 else B                  # samples: half the batch under CFG
                    cond = p[(B - n) * H:].view(n, H, N_q, -1).mean(1)          # (n, N_q, N_k), the conditional branch
                    self.parent.maps[self.name] = cond.to(torch.float16).cpu()
        return out


class _ABSSSD3:
    """SD3.5 joint-attention block. At the read step it stores, per sample, the image-to-text attention averaged over
    heads and image queries, conditional branch: (n, N_txt). The values are the raw probabilities, not renormalised
    over the text tokens, which is the quantity ABSS's DiT statistic uses."""

    def __init__(self, parent, name, fused):
        self.parent, self.name, self.fused = parent, name, fused

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if self.parent.step == self.parent.read_step and encoder_hidden_states is not None:
            with torch.no_grad():
                B, N_img, _ = hidden_states.shape
                H = attn.heads
                Dh = attn.to_q.out_features // H
                N_txt = encoder_hidden_states.shape[1]
                # image queries and the joint key set [text, image], with the block's own query/key normalisation
                iq = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (H, Dh)))
                ik = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (H, Dh)))
                tk = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (H, Dh)))
                k = torch.cat([tk, ik], 1).permute(0, 2, 1, 3).reshape(B * H, -1, Dh)
                q = iq.permute(0, 2, 1, 3).reshape(B * H, N_img, Dh)
                kt = k.transpose(1, 2)
                acc = torch.zeros(B * H, N_txt, device=q.device, dtype=torch.float32)
                chunk = 512           # image queries per chunk, so the full attention matrix is never held at once
                for s0 in range(0, N_img, chunk):
                    p = torch.baddbmm(torch.empty(B * H, min(chunk, N_img - s0), k.shape[1], dtype=q.dtype, device=q.device),
                                      q[:, s0:s0 + chunk], kt, beta=0, alpha=Dh ** -0.5).softmax(-1).float()
                    acc += p[..., :N_txt].sum(1)                                # the text columns, summed over this chunk's queries
                acc = acc.view(B, H, N_txt).mean(1) / N_img                     # (B, N_txt): mean over heads and image queries
                n = B // 2 if B % 2 == 0 and B >= 2 else B
                self.parent.vecs[self.name] = acc[B - n:].cpu().numpy()          # conditional branch
        return out


class _ABSSFlux:
    """FLUX double-stream block. At the read step it stores, per sample, the image-to-text attention averaged over heads
    and image queries: (B, n_txt). FLUX has no unconditional branch."""

    def __init__(self, parent, name, fused):
        self.parent, self.name, self.fused = parent, name, fused

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kw):
        out = (self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kw)
               if image_rotary_emb is not None else self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw))
        if self.parent.step == self.parent.read_step and encoder_hidden_states is not None:
            with torch.no_grad():
                # queries and keys as the block's processor forms them: [text, image] tokens, normalised, rotary-embedded
                B = encoder_hidden_states.shape[0]
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                head_dim = key.shape[-1] // attn.heads
                query = query.view(B, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(B, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                eq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, head_dim).transpose(1, 2)
                ek = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_added_q is not None:
                    eq = attn.norm_added_q(eq)
                if attn.norm_added_k is not None:
                    ek = attn.norm_added_k(ek)
                query = torch.cat([eq, query], dim=2)
                key = torch.cat([ek, key], dim=2)
                if image_rotary_emb is not None:
                    from diffusers.models.embeddings import apply_rotary_emb
                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)
                n_txt = encoder_hidden_states.shape[1]
                q_img = query[:, :, n_txt:]                 # the image queries only
                kT = key.transpose(-1, -2)
                scale = head_dim ** -0.5
                acc = torch.zeros(B, n_txt, device=query.device, dtype=torch.float32)
                N_img = q_img.shape[2]
                chunk = 512
                for r0 in range(0, N_img, chunk):
                    p = torch.softmax((torch.matmul(q_img[:, :, r0:r0 + chunk], kT) * scale).float(), dim=-1)
                    acc += p[..., :n_txt].sum(dim=(1, 2))   # the text columns, summed over heads and this chunk's queries
                self.parent.vecs[self.name] = (acc / (attn.heads * N_img)).cpu().numpy()     # (B, n_txt): mean over heads and image queries
        return out


class _ABSSHunyuan:
    """Hunyuan-DiT cross-attention (attn2) of one block. At the read step it stores, per sample, the conditional-branch
    attention over the 333 text keys (77 BERT, then 256 mT5), averaged over heads and image queries: (n, 333).

    The queries and keys are formed as the model's processor forms them (query/key normalisation, rotary embedding on
    the queries only) and the scores are computed in fp32. This is the text vector that ae_hook_HUNYUAN.py logs.
    """

    def __init__(self, parent, name, fused):
        self.parent, self.name, self.fused = parent, name, fused

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, image_rotary_emb=None):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, temb, image_rotary_emb)
        if self.parent.step == self.parent.read_step and encoder_hidden_states is not None and attention_mask is None:
            with torch.no_grad():
                from diffusers.models.embeddings import apply_rotary_emb
                hs = hidden_states
                B = hs.shape[0]
                ehs = attn.norm_encoder_hidden_states(encoder_hidden_states) if attn.norm_cross else encoder_hidden_states
                q = attn.to_q(hs)
                k = attn.to_k(ehs)
                D = k.shape[-1] // attn.heads
                q = q.view(B, -1, attn.heads, D).transpose(1, 2)
                k = k.view(B, -1, attn.heads, D).transpose(1, 2)
                if attn.norm_q is not None:
                    q = attn.norm_q(q)
                if attn.norm_k is not None:
                    k = attn.norm_k(k)
                if image_rotary_emb is not None:
                    q = apply_rotary_emb(q, image_rotary_emb)
                n = B // 2 if B % 2 == 0 and B >= 2 else B
                q, k = q[B - n:], k[B - n:]                                      # the conditional branch
                Nq, T = q.shape[2], k.shape[2]
                kT = k.transpose(-1, -2).float()
                acc = torch.zeros(n, T, device=q.device, dtype=torch.float32)
                for r0 in range(0, Nq, 512):                                     # image queries in chunks of 512
                    acc += torch.softmax(torch.matmul(q[:, :, r0:r0 + 512].float(), kT) * D ** -0.5, dim=-1).sum(dim=(1, 2))
                self.parent.vecs[self.name] = (acc / (attn.heads * Nq)).cpu().numpy()   # (n, 333)
        return out


def hook_kind(family):
    """Names the hook a live run uses, for the record: "fused" (the default on every family) or "standard"."""
    if family == "unet":
        return "fused" if os.environ.get("LIVESEED_FUSED_UNET", "1") == "1" else "standard"
    if family in ("sd3", "flux", "hunyuan"):
        if os.environ.get("LIVESEED_FUSED", "1") != "1":
            return "standard"
        return "fused-" + os.environ.get("LIVESEED_HY_PRECISION", "center") if family == "hunyuan" else "fused"
    return "standard"


def abss_batch_score(lr, col, tok, prompt, args, family, size):
    """ABSS's statistic at its read step on the live batch, as its paper defines it: one score per alive row.

    tok: the prompt's entry of the --tokens file; its "entity" list holds the 1-based positions of the core words among
    the prompt's whitespace-split words. A prompt without a core word scores 0 on every row, as in the authors' code,
    and the first k rows then survive.
    """
    from abss_statistic import smooth2d_reflect, smooth1d_reflect, BETA, clip_indices, t5_indices, words_from_positions
    words = words_from_positions(prompt, tok.get("entity", []))
    col.last_core = " ".join(words)
    n_alive = len(lr.alive)
    if family == "unet":
        maps = col.flush_ca_maps()
        if not maps:
            raise RuntimeError("ABSS hook stored no map at the read step (query count / step mismatch)")
        A = torch.stack([maps[k] for k in sorted(maps)]).float().mean(0).numpy()        # (n, N_q, N_k): mean over the read layers
        core = clip_indices(lr.pipe.tokenizer, prompt, words)       # token positions of the core words, <BOS> at 0
        res = size // 32
        # The statistic drops <BOS> and the last position before its softmax over tokens, so the indices shift by one.
        core_ae = [i - 1 for i in core if 1 <= i <= A.shape[-1] - 2]
        scores = []
        for row in range(A.shape[0]):
            # sharpen: at every position, a softmax over the remaining tokens of BETA x the attention
            S = A[row][:, 1:-1] * BETA
            S = S - S.max(1, keepdims=True)
            P = np.exp(S)
            P = P / P.sum(1, keepdims=True)
            # per core token: its res x res map, Gaussian-smoothed, then the spatial mean; the score is the mean over core tokens
            scores.append(float(np.mean([smooth2d_reflect(P[:, i].reshape(res, res), args.k, args.sigma).mean() for i in core_ae]))
                          if core_ae else 0.0)
        return np.asarray(scores[:n_alive])
    vecs = col.flush_vecs()
    if not vecs:
        raise RuntimeError("ABSS hook stored no text vector at the read step")
    V = next(iter(vecs.values()))                                                        # (n, N_txt) raw image->text attention per token
    # the core words' positions among the text keys of each model
    if family == "flux":
        core = t5_indices(lr.pipe.tokenizer_2, prompt, words)
    elif family == "hunyuan":   # the mT5 positions, offset by the 77 BERT keys that come first
        core = [77 + i for i in t5_indices(lr.pipe.tokenizer_2, prompt, words, max_len=256) if 77 + i < V.shape[1]]
    else:   # SD3.5: the CLIP positions and the T5 positions, which follow the 77 CLIP keys
        cc = clip_indices(lr.pipe.tokenizer, prompt, words)
        ct = [77 + i for i in t5_indices(lr.pipe.tokenizer_3, prompt, words, max_len=256) if 77 + i < V.shape[1]]
        core = sorted(set(cc) | set(ct))
    # per row: the vector smoothed along the tokens by a 1-D Gaussian, then the mean over the core tokens
    scores = [float(np.mean(smooth1d_reflect(V[row].astype(np.float64), args.k, args.sigma)[core])) if core else 0.0 for row in range(V.shape[0])]
    return np.asarray(scores[:n_alive])


# =============================================================================================== evaluators (best-of-N, --validate)
def load_hps():
    """Returns score(image, prompt) -> HPSv2.1 as a float."""
    import hpsv2
    return lambda img, prompt: float(np.asarray(hpsv2.score(img, prompt, hps_version="v2.1")).flatten()[0])


def load_clip_image():
    """Returns cos(image_a, image_b): the cosine of their CLIP ViT-L/14 image embeddings."""
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.eval().to("cuda")

    @torch.no_grad()
    def cos(a, b):
        x = model.encode_image(torch.stack([pre(a), pre(b)]).to("cuda")).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return float((x[0] @ x[1]).item())

    return cos


# =============================================================================================== command line
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate")
    c.add_argument("--model", required=True)
    c.add_argument("--pool", required=True)             # the --out folder of seed_harness.py, with scores_finals.csv in it
    c.add_argument("--cap", type=int, default=8)        # readings at steps <= cap are the candidates of both forms
    c.add_argument("--out", required=True)              # the JSON file that `run --maps` reads
    c.add_argument("--first-exit", type=int, default=None,
                   help="guarantee a reading at a step <= this in the five-map subset (the first exit of a two-exit schedule)")
    c.add_argument("--order", choices=["pooled", "entfirst"], default="pooled",
                   help="U-Net reading: sa_mean_ (entropy of the row pooled over heads and queries) or "
                        "sa_entfirst_ (entropy of each row, then the mean)")

    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODEL_CFG))
    r.add_argument("--prompts", required=True)          # CSV with the columns ID, prompt and optionally Dataset
    r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=["liveseed", "abss", "bestofN", "single"], default="liveseed")
    r.add_argument("--form", choices=["subset5", "ridge"], default="subset5")    # LiveSeed-Fast / LiveSeed-Max
    r.add_argument("--schedule", default="3:1")         # exits as step:kept, comma-separated; kept may be "half"
    r.add_argument("--maps")                            # the JSON written by `calibrate`
    r.add_argument("--N", default="10")                 # pool sizes, comma-separated
    r.add_argument("--repeats", type=int, default=1)    # timing repeats of each (prompt, N)
    r.add_argument("--limit", type=int, default=0)      # only the first `limit` prompts (0 = all)
    # 0 or None = the default of the model and mode (see run)
    r.add_argument("--steps", type=int, default=0)
    r.add_argument("--guidance", type=float, default=None)
    r.add_argument("--size", type=int, default=0)
    r.add_argument("--dtype", default="auto")           # "bf16", or "auto": bf16 on FLUX and fp16 elsewhere
    # ABSS: the core-word file, the read step, the DiT block, the Gaussian's half-width and sigma, the GPT call's seconds
    r.add_argument("--tokens")
    r.add_argument("--abss-step", type=int, default=10)
    r.add_argument("--abss-block", type=int, default=12)
    r.add_argument("--k", type=int, default=1)
    r.add_argument("--sigma", type=float, default=0.5)
    r.add_argument("--gpt-seconds", type=float, default=0.0)
    r.add_argument("--seedlist", default=None)          # comma-separated seeds to use instead of the prompt-derived ones
    r.add_argument("--validate", action="store_true")   # LiveSeed mode: offline pick and single-seed comparison
    r.add_argument("--evaluator", choices=["hpsv21", "none"], default="hpsv21", help="bestofN: none = finish and decode all N without scoring")
    a = ap.parse_args()
    calibrate(a) if a.cmd == "calibrate" else run(a)


if __name__ == "__main__":
    main()
