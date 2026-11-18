#!/usr/bin/env python
"""seed_harness.py: generates a pool (prompts x seeds) and logs the attention-entropy readings of every hooked block.

A pool is what LiveSeed is calibrated and evaluated on (Sec. 3.3 of the paper, App. C). For every prompt the script draws
N seeds, runs each to completion with the attention hooks attached, and logs everything a selection rule could score, so
that every rule (LiveSeed, ABSS, the preview baselines, random, best-of-N) is evaluated offline from one run:

  * the readings: one normalised attention entropy per hooked block and step (Sec. 3.1, Eq. 2), for steps 1..--log-steps;
  * the attention mass on every text token, per block and logged step (what the ABSS baseline's statistic needs);
  * a decoded preview of the predicted clean image at --preview-steps (for baselines that score an early preview);
  * the finished image.

Usage (from code/liveseed/):
  python seed_harness.py --model sd14 --csv ../../prompts/DrawBench.csv --slice 0 6 --n-seeds 6 --steps 30 --log-steps 12 --out runs/sd14_pool

Input
  --csv   a prompt list with the columns Dataset, ID, prompt and, optionally, category. --slice START COUNT keeps COUNT rows
          from row START.

Output layout (everything under --out; <stem> = <Dataset>_<ID>_seed<seed>)
  run_config.json                 the settings of the run: model key and repository, steps, guidance, size, scheduler,
                                  dtype, seed options, logged and preview steps, logging mode, GPU, host name, PyTorch.
  results.csv                     one row per (prompt, seed, logged step); the logged steps are 1..--log-steps and the
                                  final step. Columns below. It is named results.<suffix>.csv when the environment
                                  variable SEEDSEL_RESULTS_SUFFIX is set, so that concurrent jobs writing slices of one
                                  pool never share a file; every reader globs results*.csv.
  images/<stem>.png               the finished image.
  previews/<stem>_step<kk>.jpg    the predicted clean image (x0) at each step of --preview-steps, decoded by the VAE.
  stats/<stem>.npz                the text-token attention vectors of the image (arrays below).
  camaps/<stem>_step<kk>.npz      only with --store-ca-maps, on U-Nets: cross-attention maps for the ABSS statistic.

Columns of results.csv
  dataset, image_id, category, prompt   copied from the prompt list (image_id is its ID column).
  seed          the seed of the image.
  step          the denoising step the row describes, counted from 1.
  image_path    the path of the finished image, on the row of the final step only; empty on every other row.
  gen_seconds   on the row of the final step only: wall-clock seconds for the whole image, hooks and previews included.
  readings      one column per statistic and block, named <prefix>_<block>, where <block> is the attention processor's
                name with dots replaced by underscores. All entropies are divided by ln(number of keys), so they lie
                in [0, 1]. The row of the final step has no readings unless --log-steps reaches it (the hooks are
                switched off after the last logged step). The main prefixes:
                  U-Nets (ae_hook.py), self-attention layers
                    sa_mean_      entropy of the attention row pooled over heads, queries and both branches of
                                  classifier-free guidance (CFG): pool first, then one entropy.
                    sa_entfirst_  the order of Eq. 2: the entropy of every row, then the mean over heads and queries.
                    sa_cond_      sa_mean_ on the conditional CFG branch alone.
                  MMDiTs with joint attention (SD3.5, FLUX, Qwen-Image) and Hunyuan-DiT (self-attention), ae_hook_*.py
                    ae_mean_      the order of Eq. 2 on the image-query rows.
                    ae_avgfirst_  the pooled order (pool first, then one entropy); not logged on Qwen-Image.
                    ae_cond_      ae_mean_ on the conditional CFG branch alone (FLUX has one branch and no such column).
                    txtmass_      the attention mass of the image queries on the text keys (joint attention only).
                  Other columns (cross-attention entropy and token masses on U-Nets, per-head spread, the CFG difference,
                  ae_img_ on FLUX and Qwen-Image) are diagnostics; the hook files define them.

Arrays of stats/<stem>.npz
  steps (S,), layers (L,)   the logged steps and the block names that have a vector.
  vec (S, L, T)             per step and block, the attention on each of the T text keys, averaged over heads and image
                            queries and normalised to sum to one over the text keys (conditional CFG branch). The blocks
                            are the cross-attention layers of a U-Net or of Hunyuan-DiT and the joint-attention blocks of
                            an MMDiT. token_layout.py says where the special tokens sit in T.
  n_active                  the number of prompt tokens, special tokens and padding excluded.
  body_words, body_method   the prompt's nouns and how they were found ("spacy" or "heuristic").
  clip_body_idx             the positions of those nouns' tokens in the indexing of `vec` (despite its name it holds the
                            T5 positions on FLUX and the Qwen2.5-VL positions on Qwen-Image).
  t5_body_idx               their positions in the T5 sequence where the model has one (empty otherwise).
  text_encoder              "clip", "t5" (FLUX), "bert77+mt5_256" (Hunyuan-DiT) or "qwen2.5-vl".

Arrays of camaps/<stem>_step<kk>.npz
  layers (L,), maps (L, N_q, N_k)   for every cross-attention layer with N_q = --ca-map-res squared queries, the attention
                            map of the conditional branch averaged over heads (float16); step, n_active, clip_body_idx,
                            size and res accompany it. abss_statistic.py reads these files.

Model keys (--model). Size and guidance are the defaults and can be overridden with --size and --guidance. --steps defaults
to 30 and --dtype to fp16 for every model, so pass the values noted below where a model needs others.
  sd14, sd15     Stable Diffusion 1.4 / 1.5, 512 px, CFG 7.5, Euler sampler.
  sd21           Stable Diffusion 2.1, 768 px, CFG 7.5, Euler sampler.
  sdxl           SDXL base 1.0, 1024 px, CFG 7.0, the pipeline's default sampler (Euler).
  sd35, sd35l    SD3.5-Medium / SD3.5-Large, 1024 px, CFG 7.0, flow-matching Euler. Gated repositories.
  sd35l_turbo    SD3.5-Large Turbo, 1024 px, no CFG; use --steps 4. Gated repository.
  flux           FLUX.1-dev, 1024 px, embedded guidance 3.5; use --steps 28 --dtype bf16. Gated repository.
  flux_schnell   FLUX.1-schnell, 1024 px, no guidance, T5 length 256.
  hunyuan        Hunyuan-DiT v1.2, 1024 px, CFG 5.0, the pipeline's own scheduler.
  qwen           Qwen-Image, 1328 px, true CFG 4.0 with the negative prompt " "; use --steps 50 --dtype bf16. Needs
                 diffusers >= 0.35, hence a separate Python environment.

How the seeds are drawn
  By default each prompt gets its own --n-seeds distinct seeds, a deterministic function of the prompt (seeds_for): the
  SHA-1 hash of "<Dataset>:<ID>:<prompt>" seeds a NumPy RandomState, which draws integers in [1, 2^31 - 2] until N are
  distinct. A rerun, or a run with a larger N, therefore reproduces the same seeds in the same order. Alternatives:
  --seed-list a,b,c uses the same seeds for every prompt, --fixed-seed S a single seed for every prompt, and
  --include-seeds a,b places the given seeds first and fills the rest of the N from the draw. Each image is generated
  alone (batch of one) from torch.Generator(device="cuda").manual_seed(seed).

Logging modes
  --log-mode full      (default) every hooked block at every logged step, both aggregation orders, the conditional-branch
                       entropy, the token vectors and, with --store-ca-maps, the cross-attention maps.
  --log-mode deployed  only the (block, step) readings listed in --deployed-maps, a json {"<model>": ["<reading
                       column>@s<step>", ...]} or a flat list of those, through the step-gated hooks (fast_hooks.py on
                       U-Nets, SD3.5 and Hunyuan-DiT; the `wanted` filter of ae_hook_FLUX.py on FLUX). Nothing else is
                       computed: this is the cost of the method as deployed (Sec. 4.2, App. D).

Resuming: with --resume, a (prompt, seed) pair is skipped when its image and its .npz exist and results*.csv already holds
its final-step row, so a job that was interrupted and restarted continues where it stopped.

Environment variables: SEEDSEL_RESULTS_SUFFIX (above); SEEDSEL_SA_LAYERS (the default of --sa-layers);
SEEDSEL_FUSED_READINGS=1 (SD3.5: readings from the fused hook, which does not materialise the attention matrix);
SEEDSEL_NO_HOOKS=1 (images only, no readings); SEEDSEL_CPU_OFFLOAD=1 (model CPU offload for cards with less memory).
"""
from __future__ import annotations

import argparse, os, csv, json, math, re, sys, time, gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # the hook modules are imported from this folder

# --------------------------------------------------------------------------- models
# One entry per --model key: the Hugging Face repository, the architecture family (which selects the pipeline class and the
# hook module), the default image side in pixels, the default guidance scale and the weight-file variant to download.
MODEL_CFG = {
    "sd14": dict(repo="CompVis/stable-diffusion-v1-4", family="unet", size=512, guidance=7.5, variant=None),
    "sd15": dict(repo="stable-diffusion-v1-5/stable-diffusion-v1-5", family="unet", size=512, guidance=7.5, variant=None),
    # An open mirror of the gated stabilityai repository.
    "sd21": dict(repo="sd2-community/stable-diffusion-2-1", family="unet", size=768, guidance=7.5, variant="fp16"),
    "sdxl": dict(repo="stabilityai/stable-diffusion-xl-base-1.0", family="unet", size=1024, guidance=7.0, variant="fp16"),
    "sd35": dict(repo="stabilityai/stable-diffusion-3.5-medium", family="sd3", size=1024, guidance=7.0, variant=None),
    # The SD3.5 model of the ABSS paper (38 MMDiT blocks, of which ABSS reads block 18). Gated repository.
    "sd35l": dict(repo="stabilityai/stable-diffusion-3.5-large", family="sd3", size=1024, guidance=7.0, variant=None),
    # Embedded (distilled) guidance; bf16 and 28 steps are the recommended settings.
    "flux": dict(repo="black-forest-labs/FLUX.1-dev", family="flux", size=1024, guidance=3.5, variant=None),
    # Hunyuan-DiT with the defaults of the ABSS authors' backend: fp16, the pipeline's own scheduler, 1024 px.
    # The pipeline's default guidance is 5.0; ABSS runs it at 7.5 (pass --guidance 7.5 to match).
    "hunyuan": dict(repo="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", family="hunyuan", size=1024, guidance=5.0, variant=None),
    # Qwen-Image needs diffusers >= 0.35 (a separate Python environment). `guidance` is passed as true_cfg_scale with the
    # negative prompt " " (the model card's setting); the model card also recommends 50 steps and bf16.
    "qwen": dict(repo="Qwen/Qwen-Image", family="qwen", size=1328, guidance=4.0, variant=None),
    # Few-step distilled models without CFG: SD3.5-Large Turbo (4 steps, gated repository) and FLUX.1-schnell.
    "sd35l_turbo": dict(repo="stabilityai/stable-diffusion-3.5-large-turbo", family="sd3", size=1024, guidance=0.0, variant=None),
    "flux_schnell": dict(repo="black-forest-labs/FLUX.1-schnell", family="flux", size=1024, guidance=0.0, variant=None),
}


# --------------------------------------------------------------------------- hooks
def build_unet_collector(ca_map_steps=None, ca_map_queries=None, extra_orders=True):
    """Returns the U-Net collector of ae_hook.py, extended to keep the text-token mass vector of every cross-attention layer.

    The vector is the conditional branch's attention averaged over heads and queries, normalised to sum to one: shape (N_k,),
    N_k = 77 CLIP positions. It is what stats/<stem>.npz stores.
    """
    import ae_hook

    class VecCAProcessor(ae_hook._AEAttnProcessor):
        def _record_entropy(self, attn_probs: torch.Tensor):
            super()._record_entropy(attn_probs)
            try:
                BH = attn_probs.shape[0]
                # CFG batch layout is [uncond, cond]; keep the conditional half.
                probs = attn_probs[BH // 2:] if BH >= 2 else attn_probs
                dist = probs.float().mean(dim=0).mean(dim=0)          # (N_k,)
                dist = dist / dist.sum().clamp_min_(1e-8)
                self.parent._push_vec(self.layer_name, dist.detach().cpu().numpy().astype(np.float32))
            except Exception:
                # A failed statistic must not stop the image.
                pass

    class VecCollector(ae_hook.AttnEntropyCollector):
        def __init__(self, device="cuda", **kw):
            super().__init__(device, **kw)
            self.vecs: Dict[str, List[np.ndarray]] = {}

        def get_processor(self, layer_name):
            return VecCAProcessor(self, layer_name=layer_name)

        def _push_vec(self, layer_name, vec):
            self.vecs.setdefault(layer_name, []).append(vec)

        def new_image(self, key, n_active_tokens=None):
            super().new_image(key, n_active_tokens=n_active_tokens)
            self.vecs = {}

        def flush_vecs(self) -> Dict[str, np.ndarray]:
            """The vectors gathered since the last call, {layer name: (N_k,)}, averaged over the layer's calls in the step."""
            out = {k: np.mean(np.stack(v), axis=0) for k, v in self.vecs.items() if v}
            self.vecs = {}
            return out

    return VecCollector(device="cuda", ca_map_steps=ca_map_steps, ca_map_queries=ca_map_queries, extra_orders=extra_orders)


def build_sd3_collector(extra_orders=True):
    """Returns the SD3.5 collector of ae_hook_SD35.py, extended to keep the text-token mass vector of every joint-attention block.

    The vector is the attention of the image queries on the N_txt text keys (conditional branch, averaged over heads and
    queries), normalised to sum to one over the text keys: shape (N_txt,).
    """
    import ae_hook_SD35 as h35

    class VecJAProcessor(h35._SD3JointAEAttnProcessor):
        def _log_entropy(self, img_probs, N_txt, N_img, B=1):
            super()._log_entropy(img_probs, N_txt, N_img, B=B)
            try:
                BH = img_probs.shape[0]
                # CFG batch layout is [uncond, cond]; keep the conditional half when there is a CFG batch.
                probs = img_probs[BH // 2:] if (B >= 2 and BH >= 2) else img_probs
                text = probs[:, :, :N_txt].float().mean(dim=0).mean(dim=0)  # (N_txt,)
                text = text / text.sum().clamp_min_(1e-8)
                self.parent._push_vec(self.layer_name, text.detach().cpu().numpy().astype(np.float32))
            except Exception:
                # A failed statistic must not stop the image.
                pass

    class VecCollector(h35.AttnEntropyCollector):
        def __init__(self, device="cuda", **kw):
            super().__init__(device, **kw)
            self.vecs: Dict[str, List[np.ndarray]] = {}

        def get_processor(self, layer_name, orig=None):
            return VecJAProcessor(self, layer_name=layer_name, orig=orig)

        def _push_vec(self, layer_name, vec):
            self.vecs.setdefault(layer_name, []).append(vec)

        def new_image(self, key, n_active_tokens=None):
            super().new_image(key)
            self.vecs = {}

        def flush_vecs(self) -> Dict[str, np.ndarray]:
            """The vectors gathered since the last call, {block name: (N_txt,)}, averaged over the block's calls in the step."""
            out = {k: np.mean(np.stack(v), axis=0) for k, v in self.vecs.items() if v}
            self.vecs = {}
            return out

    col = VecCollector(device="cuda", extra_orders=extra_orders)
    # Fused readings (ae_hook_SD35.py) avoid materialising the joint attention matrix, which matters on SD3.5-Large.
    col.fused = os.environ.get("SEEDSEL_FUSED_READINGS", "0") == "1"
    if col.fused:
        print("[harness] SD3 hook: fused readings (no ja_head_std / ja_avgfirst)", flush=True)
    return col


class DeployedCollector:
    """Deployed mode on U-Nets, SD3.5 and Hunyuan-DiT: a fast_hooks.Sink behind the interface of the full collectors, so
    that main() needs no second code path. Only the wanted (layer, step) readings are computed."""

    def __init__(self, sink):
        self.sink = sink
        self.vecs = {}

    @property
    def step(self):           # the harness sets col.step before every denoising step; the sink's gate reads it
        return self.sink.step

    @step.setter
    def step(self, v):
        self.sink.step = v

    def new_image(self, key, n_active_tokens=None):
        self.sink.reset()

    def flush_and_get_layer_means(self):
        """The readings of the step just completed, {results.csv column: value}; called before the step counter advances."""
        vals = self.sink.values()
        out = {}
        for (name, st), v in vals.items():
            if st == self.sink.step:
                # the column names of full mode: sa_mean_ for U-Net self-attention, ae_mean_ for the transformers
                unet = name.startswith(("down_blocks", "mid_block", "up_blocks"))
                out[("sa_mean_" if unet else "ae_mean_") + name.replace(".", "_")] = v
        return out

    def flush_vecs(self):
        return {}

    def flush_ca_maps(self):
        return {}


def wanted_pairs(path: str, model: str):
    """Reads --deployed-maps and returns the set of (layer processor name, step) pairs to read.

    The json is {"<model>": ["sa_mean_<layer>@s<k>", ...]} or a flat list of such entries.
    """
    spec = json.load(open(path))
    feats = spec[model] if isinstance(spec, dict) else spec
    out = set()
    for f in feats:
        col, st = f.split("@s")
        lay = col.split("_mean_", 1)[1] if "_mean_" in col else col
        out.add((lay, int(st)))     # still in the csv spelling (underscores); build_pipe maps it to the processor name
    return out


# --------------------------------------------------------------------------- pipeline
def build_pipe(model: str, dtype: torch.dtype, sa_layers: Optional[str] = None, ca_map_steps=None, ca_map_queries=None,
               extra_orders: bool = True, log_mode: str = "full", wanted=None):
    """Loads the pipeline of `model` on the GPU and installs the attention hooks.

    Returns (pipe, collector, switch): the collector receives the readings, and the switch turns the hooks on and off
    (HookSwitch or ModuleProcSwitch).

    sa_layers: a regex; only the self-attention layers (on FLUX and Qwen-Image, the joint-attention blocks) whose processor
        name matches are hooked. None hooks all of them. Cross-attention layers are always hooked, they are cheap. SD3.5
        ignores it: every joint-attention block is hooked.
    ca_map_steps, ca_map_queries: the steps and the query count of the cross-attention maps to store (U-Nets).
    extra_orders: also log the second aggregation order (the full logging mode).
    log_mode: "deployed" installs the step-gated hooks for the `wanted` (layer, step) pairs only.
    """
    from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, EulerDiscreteScheduler
    cfg = MODEL_CFG[model]
    kw = dict(torch_dtype=dtype, use_safetensors=True)
    if cfg["variant"] and dtype != torch.float32:   # fp32 loads the full-precision weights
        kw["variant"] = cfg["variant"]
    if cfg["family"] == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(cfg["repo"], **kw)
    elif cfg["family"] == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(cfg["repo"], torch_dtype=dtype)
    elif cfg["family"] == "hunyuan":
        from diffusers import HunyuanDiTPipeline
        # The pipeline keeps its own scheduler, as in the ABSS authors' backend.
        pipe = HunyuanDiTPipeline.from_pretrained(cfg["repo"], torch_dtype=dtype)
    elif cfg["family"] == "qwen":
        from diffusers import QwenImagePipeline                                       # diffusers >= 0.35
        pipe = QwenImagePipeline.from_pretrained(cfg["repo"], torch_dtype=dtype)
    else:
        try:
            pipe = DiffusionPipeline.from_pretrained(cfg["repo"], **kw)
        except Exception as e:   # fp32 without the full-precision files in the cache: upcast the fp16 files
            if not (cfg["variant"] and dtype == torch.float32):
                raise
            print(f"[harness] full-precision weights unavailable ({type(e).__name__}); loading the {cfg['variant']} files in fp32", flush=True)
            pipe = DiffusionPipeline.from_pretrained(cfg["repo"], variant=cfg["variant"], **kw)
        if model in ("sd14", "sd15", "sd21"):
            # One fixed, stated sampler for every method compared (ABSS leaves its sampler unstated).
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    if os.environ.get("SEEDSEL_CPU_OFFLOAD", "0") == "1":
        # Model CPU offload, for cards with less memory. WARNING: an offloaded run without hooks did not reproduce the
        # images of a hooked run of the same seeds on a larger card, so never pair its images with readings logged elsewhere.
        pipe.enable_model_cpu_offload()
        print("[harness] model CPU offload on", flush=True)
    else:
        pipe = pipe.to("cuda")

    # NB: diffusers' set_attn_processor() POPS entries out of the dict it is given, so
    # every install below passes a copy and HookSwitch keeps the originals.

    # ---- deployed mode: step-gated hooks on the wanted layers only
    if log_mode == "deployed" and cfg["family"] == "qwen":
        raise SystemExit("[harness] deployed mode is not implemented for Qwen-Image (full logging only)")
    if log_mode == "deployed":
        import fast_hooks as fh
        names_all = list((pipe.transformer if cfg["family"] in ("sd3", "flux", "hunyuan") else pipe.unet).attn_processors)
        # The csv columns spell a processor name with underscores for its dots and may drop the trailing "processor",
        # so both sides are compared in that reduced spelling (as liveseed_live.resolve_names does).
        reduced = lambda n: n.replace(".", "_").replace("_processor", "")
        key = {reduced(n): n for n in names_all}
        pairs = {(key.get(reduced(n), n), st) for n, st in (wanted or set())}
        missing = [n for n, _ in pairs if n not in names_all]
        if missing:
            raise SystemExit(f"[harness] deployed maps name layers the model does not have: {missing[:5]}")
        if cfg["family"] == "flux":
            import ae_hook_FLUX as hfx
            col = hfx.AttnEntropyCollector(device="cuda", extra_orders=False, wanted=pairs)
            # The T5 sequence length of the FluxPipeline default (max_sequence_length=512). The single-stream blocks see
            # text and image concatenated, so the hook needs it when no double-stream block is hooked to measure it.
            col.n_txt = 512
            fast = dict(pipe.transformer.attn_processors)
            procs = {name: (col.get_processor(name) if name in {n for n, _ in pairs} else proc) for name, proc in fast.items()}
            pipe.transformer.set_attn_processor(dict(procs))
            print(f"[harness] FLUX deployed mode: {len(pairs)} (block, step) readings on {len({n for n, _ in pairs})} blocks", flush=True)
            return pipe, col, HookSwitch(pipe.transformer, procs, fast)
        sink = fh.Sink(pairs)
        n = fh.install(pipe, cfg["family"], sink)
        print(f"[harness] deployed mode: gated hooks on {n} layers, {len(pairs)} (layer, step) readings", flush=True)
        module = pipe.transformer if cfg["family"] in ("sd3", "hunyuan") else pipe.unet   # Hunyuan-DiT: fast_hooks.FastHY
        procs = dict(module.attn_processors)
        return pipe, DeployedCollector(sink), HookSwitch(module, procs, procs)   # nothing to switch off: the gate is per step

    # ---- full mode: the logging hooks of each family. `fast` holds the model's own processors, `procs` the hooked set.
    if cfg["family"] == "sd3":
        col = build_sd3_collector(extra_orders=extra_orders)
        fast = {name: proc for name, proc in pipe.transformer.attn_processors.items()}
        procs = {}
        for name, proc in fast.items():
            # The hook wraps the model's own processor, which still produces the block's output, so the images are
            # identical to an unhooked run. attn2 (the extra self-attention of SD3.5-Medium's dual blocks) is not read.
            procs[name] = col.get_processor(name, proc) if "attn2" not in name else proc
        pipe.transformer.set_attn_processor(dict(procs))
        hooks = HookSwitch(pipe.transformer, procs, fast)
    elif cfg["family"] == "flux":
        import ae_hook_FLUX as hfx
        col = hfx.AttnEntropyCollector(device="cuda", extra_orders=extra_orders)
        fast = {name: proc for name, proc in pipe.transformer.attn_processors.items()}
        sa_re = re.compile(sa_layers) if sa_layers else None
        procs = {name: (col.get_processor(name) if (sa_re is None or sa_re.search(name)) else proc) for name, proc in fast.items()}
        n_hook = sum(1 for q in procs.values() if isinstance(q, hfx._FluxAEAttnProcessor))
        print(f"[harness] FLUX: hooking {n_hook}/{len(procs)} attention blocks" + (f" matching /{sa_layers}/" if sa_layers else ""), flush=True)
        pipe.transformer.set_attn_processor(dict(procs))
        hooks = HookSwitch(pipe.transformer, procs, fast)
    elif cfg["family"] == "hunyuan":
        import ae_hook_HUNYUAN as hhy
        col = hhy.AttnEntropyCollector(device="cuda", extra_orders=extra_orders)
        fast = {name: proc for name, proc in pipe.transformer.attn_processors.items()}
        sa_re = re.compile(sa_layers) if sa_layers else None
        # attn1 = self-attention (the reading), attn2 = cross-attention (always hooked, for the token vectors).
        procs = {name: (col.get_processor(name) if ("attn2" in name or sa_re is None or sa_re.search(name)) else proc) for name, proc in fast.items()}
        n_hook = sum(1 for q in procs.values() if isinstance(q, hhy._HunyuanAEAttnProcessor))
        print(f"[harness] Hunyuan-DiT: hooking {n_hook}/{len(procs)} attention layers (every attn2; attn1" + (f" matching /{sa_layers}/)" if sa_layers else " all)"), flush=True)
        pipe.transformer.set_attn_processor(dict(procs))
        hooks = HookSwitch(pipe.transformer, procs, fast)
    elif cfg["family"] == "qwen":
        import ae_hook_QWEN as hqw
        col = hqw.AttnEntropyCollector(device="cuda", extra_orders=extra_orders)
        # This transformer has no attn_processors API, so the Attention modules are found by name.
        mods = {n: m for n, m in pipe.transformer.named_modules() if n.endswith(".attn") and hasattr(m, "set_processor")}
        fast = {n: m.processor for n, m in mods.items()}
        sa_re = re.compile(sa_layers) if sa_layers else None
        procs = {n: (col.get_processor(n, p) if (sa_re is None or sa_re.search(n)) else p) for n, p in fast.items()}
        col.install_pass_counter(pipe.transformer)
        print(f"[harness] Qwen-Image: hooking {sum(isinstance(q, hqw._QwenAEAttnProcessor) for q in procs.values())}/{len(procs)} joint-attention blocks", flush=True)
        hooks = ModuleProcSwitch(mods, procs, fast)
        hooks.on = False
        hooks.enable()
    else:
        from diffusers.models.attention_processor import AttnProcessor2_0
        pipe.unet.set_attn_processor(AttnProcessor2_0())
        names = list(pipe.unet.attn_processors.keys())
        fast = {name: AttnProcessor2_0() for name in names}
        col = build_unet_collector(ca_map_steps=ca_map_steps, ca_map_queries=ca_map_queries, extra_orders=extra_orders)
        procs = {}
        sa_re = re.compile(sa_layers) if sa_layers else None
        n_sa = 0
        for name in names:
            if "attn2" in name:                              # cross-attention: always hooked
                procs[name] = col.get_processor(name)
            elif sa_re is None or sa_re.search(name):        # self-attention: the reading
                procs[name] = col.get_sa_processor(name)
                n_sa += 1
            else:
                procs[name] = fast[name]
        print(f"[harness] hooking all cross-attention layers and {n_sa} self-attention layer(s)" + (f" matching /{sa_layers}/" if sa_layers else ""), flush=True)
        pipe.unet.set_attn_processor(dict(procs))
        hooks = HookSwitch(pipe.unet, procs, fast)
        # NB: do not call enable_attention_slicing() here: it replaces every attention
        # processor with a SlicedAttnProcessor and silently removes the hooks.
    return pipe, col, hooks


class ModuleProcSwitch:
    """HookSwitch for transformers without the attn_processors API (Qwen-Image, diffusers 0.35): swaps each Attention module's
    processor directly. `.module.attn_processors` gives the current processors, for the harness's install check."""

    def __init__(self, mods: dict, hooked: dict, fast: dict):
        self.mods, self.hooked, self.fast, self.on = mods, hooked, fast, False
        sw = self

        class _View:
            @property
            def attn_processors(self):
                return {n: m.processor for n, m in sw.mods.items()}

        self.module = _View()

    def enable(self):
        if not self.on:
            for n, m in self.mods.items():
                m.set_processor(self.hooked[n])
            self.on = True

    def disable(self):
        if self.on:
            for n, m in self.mods.items():
                m.set_processor(self.fast[n])
            self.on = False


class HookSwitch:
    """Hooks cost a manual fp32 attention and per-layer host syncs on every step, but
    statistics are only needed for the first --log-steps steps. Swap to the fast default
    processors after the last logged step, and back before the next image."""

    def __init__(self, module, hooked: dict, fast: dict):
        self.module, self.hooked, self.fast, self.on = module, hooked, fast, True

    def enable(self):
        if not self.on:
            assert len(self.hooked) > 0, "hooked processor dict is empty (consumed?)"
            self.module.set_attn_processor(dict(self.hooked))
            self.on = True

    def disable(self):
        if self.on:
            assert len(self.fast) > 0, "fast processor dict is empty (consumed?)"
            self.module.set_attn_processor(dict(self.fast))
            self.on = False


class X0Tap:
    """Wraps scheduler.step so that the predicted clean latent (x0) of every step is available, in `.x0`, for the previews.

    `generator` is an explicit parameter of the wrapper because a pipeline passes its generator to scheduler.step only when
    the step's signature names one (prepare_extra_step_kwargs). A wrapper taking only (*args, **kwargs) would hide it, and a
    stochastic sampler (Hunyuan-DiT's DDPMScheduler) would then draw every step's noise from the unseeded global CUDA
    stream, so its images would not be reproducible from the seed. The generator is forwarded only when the wrapped step
    accepts one; deterministic samplers (Euler with s_churn 0, flow matching) ignore it, so their images do not depend on it.
    """

    def __init__(self, pipe, family: str):
        import inspect
        self.pipe, self.family, self.x0 = pipe, family, None
        self._orig = pipe.scheduler.step
        self._takes_generator = "generator" in inspect.signature(self._orig).parameters
        pipe.scheduler.step = self._step

    def _step(self, model_output, timestep, sample, *args, generator=None, **kwargs):
        if generator is not None and self._takes_generator:
            kwargs["generator"] = generator
        sched = self.pipe.scheduler
        x0 = None
        # x0 is computed from the values before the step: `sample` is the noisy latent at noise level sigma.
        try:
            if self.family in ("sd3", "flux", "qwen"):
                idx = sched.step_index if sched.step_index is not None else 0
                sigma = sched.sigmas[idx]
                x0 = sample - sigma * model_output                      # flow matching: the output is the velocity
            else:
                sigma = sched.sigmas[sched.step_index] if getattr(sched, "step_index", None) is not None else None
                if sigma is not None:
                    if getattr(sched.config, "prediction_type", "epsilon") == "epsilon":
                        x0 = sample - sigma * model_output              # Euler, eps-pred (input is pre-scaled)
                    elif sched.config.prediction_type == "v_prediction":
                        x0 = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        except Exception:
            x0 = None
        out = self._orig(model_output, timestep, sample, *args, **kwargs)
        # Schedulers without a sigma schedule (DDPM): use the scheduler's own estimate.
        if x0 is None and hasattr(out, "pred_original_sample") and out.pred_original_sample is not None:
            x0 = out.pred_original_sample
        self.x0 = x0
        return out


@torch.no_grad()
def decode(pipe, latents, family: str):
    """Decode a predicted-x0 latent to a PIL preview (the first image of the batch).

    fp16 VAEs (SDXL's in particular) overflow to NaN on decode, which the pipelines avoid by
    upcasting for the final image (`force_upcast`). Previews bypass the pipeline, so decode
    with a persistent fp32 copy of the VAE whenever the pipeline's VAE is not fp32.
    """
    vae = pipe.vae
    if vae.dtype != torch.float32:
        if getattr(decode, "_vae32", None) is None:
            import copy
            decode._vae32 = copy.deepcopy(vae).to(torch.float32)
        vae = decode._vae32
    lat = latents.to(device=vae.device, dtype=vae.dtype)
    # Undo each family's latent packing and scaling, as its pipeline does before decoding the final image.
    if family == "qwen":
        side = int(round(math.sqrt(lat.shape[1]))) * 2 * pipe.vae_scale_factor     # packed (B, N, 64) -> pixels per side
        lat = pipe._unpack_latents(lat, side, side, pipe.vae_scale_factor)
        zm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        zs = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        img = vae.decode(lat * zs + zm, return_dict=False)[0][:, :, 0]             # a video VAE: keep its single frame
        return pipe.image_processor.postprocess(img, output_type="pil")[0]
    if family == "flux":
        side = int(round(math.sqrt(lat.shape[1]))) * 2 * pipe.vae_scale_factor     # packed (B, N, 64) -> pixels per side
        lat = pipe._unpack_latents(lat, side, side, pipe.vae_scale_factor)
        lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
    elif family == "sd3":
        lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
    else:
        lat = lat / vae.config.scaling_factor
    img = vae.decode(lat).sample
    if not torch.isfinite(img).all():
        if not getattr(decode, "_warned", False):
            decode._warned = True
            print("[harness] WARNING: non-finite values in a decoded preview (fp32 VAE); previews may be black", flush=True)
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


# --------------------------------------------------------------------------- tokens
# Function words that the fallback of body_tokens() never counts as a subject.
_STOP = set("""a an the and or of on in at to with under over above below beside near
behind front next between among into onto from by for is are was were be being been
it its this that these those his her their there here who which what where when
""".split())


def body_tokens(prompt: str) -> List[str]:
    """The prompt's nouns and proper nouns, lower-cased: an offline stand-in for the core words of the ABSS baseline.

    Uses spaCy (en_core_web_sm) when it is installed, else a stop-word heuristic; `body_tokens.method` records which.
    """
    try:
        import spacy
        if not hasattr(body_tokens, "_nlp"):
            body_tokens._nlp = spacy.load("en_core_web_sm")
        doc = body_tokens._nlp(prompt)
        nouns = [t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]
        body_tokens.method = "spacy"
        if nouns:
            return nouns
    except Exception:
        pass
    body_tokens.method = "heuristic"
    words = [w.strip(".,;:!?\"'()").lower() for w in prompt.split()]
    return [w for w in words if w and w not in _STOP and not w.endswith("ing") and not w.endswith("ly")]


def token_indices(tokenizer, prompt: str, words: List[str], offset: int = 1, max_len: int = 77) -> List[int]:
    """Sorted positions, in the encoder's token sequence, of the sub-word tokens belonging to `words`.

    Every occurrence of a word's token sequence in the prompt's token sequence is taken. `offset` is the number of special
    tokens before the prompt: 1 for encoders that start with a beginning-of-sequence token (CLIP: BOS at position 0), 0 for
    T5. `max_len` is the encoder's sequence length, special tokens included.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=max_len - 2)
    out = []
    for w in dict.fromkeys(words):          # each distinct word once, in order
        wid = tokenizer.encode(w, add_special_tokens=False)
        if not wid:
            continue
        for s in range(len(ids) - len(wid) + 1):
            if ids[s:s + len(wid)] == wid:
                out.extend(range(s + offset, s + offset + len(wid)))
    return sorted(set(out))


# --------------------------------------------------------------------------- main
def seeds_for(prompt_key: str, n: int) -> List[int]:
    """n distinct seeds in [1, 2^31 - 2], a deterministic function of the prompt key (stable across restarts of a job).

    The first eight hex digits of the key's SHA-1 hash seed a NumPy RandomState, which is drawn from until n distinct
    values are found. The first n seeds of a longer list are therefore the same seeds.

    NB: do not use RandomState.choice(2**31 - 1, replace=False) here: the legacy implementation
    materialises and shuffles the whole population (16 GB, ~2 min) on every call.
    """
    import hashlib  # stable across processes (Python's hash() is salted per run)
    h = int(hashlib.sha1(prompt_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(h % (2**32 - 1))
    out: List[int] = []
    while len(out) < n:
        s = int(rng.randint(1, 2**31 - 1))
        if s not in out:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CFG))
    ap.add_argument("--csv", required=True, help="prompt list with the columns Dataset,ID,prompt[,category]")
    ap.add_argument("--out", required=True)      # the pool folder (see the module docstring for its layout)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--fixed-seed", type=int, default=None, help="use this single seed for every prompt; overrides --n-seeds")
    ap.add_argument("--seed-list", type=str, default=None,
                    help="comma-separated seeds used for EVERY prompt (a global pool, as in the ABSS authors' code); overrides --n-seeds")
    ap.add_argument("--include-seeds", type=str, default=None,
                    help="comma-separated seeds placed first in every prompt's pool; the rest are drawn as usual")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=None)      # default: the model's entry in MODEL_CFG
    ap.add_argument("--size", type=int, default=None)            # image side in pixels; default: the model's entry in MODEL_CFG
    ap.add_argument("--log-steps", type=int, default=12, help="log the readings for steps 1..K and the final step")
    ap.add_argument("--preview-steps", type=str, default="2,5", help="comma list of steps at which the decoded x0 preview is saved")
    ap.add_argument("--slice", nargs=2, type=int, metavar=("START", "COUNT"))      # rows of --csv to run
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")   # fp32: ABSS's precision
    ap.add_argument("--resume", action="store_true")             # skip the (prompt, seed) pairs already complete
    ap.add_argument("--jpeg-previews", action="store_true", default=True)
    ap.add_argument("--store-ca-maps", type=str, default="",
                    help="comma list of steps at which the heads-averaged conditional cross-attention maps of the --ca-map-res layers "
                         "are saved (U-Nets; camaps/<stem>_step<k>.npz)")
    ap.add_argument("--ca-map-res", type=int, default=16,
                    help="side of the attention grid whose layers are stored (16 = the 16x16 layers at 512 px; 24 at 768 px). "
                         "Layers are matched by query count = res*res")
    ap.add_argument("--log-mode", choices=["full", "deployed"], default="full",
                    help="full: every hooked block, both aggregation orders, conditional-branch entropy, token vectors; "
                         "deployed: only --deployed-maps through the gated hooks")
    ap.add_argument("--deployed-maps", type=str, default=None, help="json listing the readings of --log-mode deployed")
    ap.add_argument("--sa-layers", type=str, default=os.environ.get("SEEDSEL_SA_LAYERS") or None,
                    help="regex over processor names; only the matching self-attention layers are hooked (on FLUX and Qwen-Image the "
                         "matching joint-attention blocks; SD3.5 ignores it). Default: all. Example, two SDXL layers: "
                         "'down_blocks\\.1\\.attentions\\.0\\.transformer_blocks\\.0\\.attn1|up_blocks\\.1\\.attentions\\.1\\.transformer_blocks\\.1\\.attn1'")
    args = ap.parse_args()

    # ---- settings
    cfg = MODEL_CFG[args.model]
    guidance = args.guidance if args.guidance is not None else cfg["guidance"]
    size = args.size or cfg["size"]
    preview_steps = sorted({int(s) for s in args.preview_steps.split(",") if s.strip()})
    ca_steps = sorted({int(s) for s in args.store_ca_maps.split(",") if s.strip()})
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if args.log_mode == "deployed" and not args.deployed_maps:
        raise SystemExit("--log-mode deployed needs --deployed-maps")
    wanted = wanted_pairs(args.deployed_maps, args.model) if args.deployed_maps else None
    last_hooked = max([args.log_steps] + ca_steps)        # hooks stay on through the last stored map
    # Images only, no statistics. --log-steps 0 is not enough for that: it leaves the hooks on for the whole image.
    no_hooks = os.environ.get("SEEDSEL_NO_HOOKS", "0") == "1"

    # ---- output folders
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    (out / "stats").mkdir(exist_ok=True)
    if ca_steps:
        (out / "camaps").mkdir(exist_ok=True)
    suffix = os.environ.get("SEEDSEL_RESULTS_SUFFIX", "")      # one results file per job: concurrent slices must not share a file
    results_path = out / (f"results.{suffix}.csv" if suffix else "results.csv")

    # ---- prompts
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if args.slice:
        df = df.iloc[args.slice[0]: args.slice[0] + args.slice[1]]
    print(f"[harness] {args.model} | {len(df)} prompts x {args.n_seeds} seeds | {args.steps} steps, CFG {guidance}, {size}px")

    # ---- model, hooks and tokenizers
    pipe, col, hooks = build_pipe(args.model, dtype, sa_layers=args.sa_layers, ca_map_steps=ca_steps, ca_map_queries=args.ca_map_res ** 2,
                                  extra_orders=(args.log_mode == "full"), log_mode=args.log_mode, wanted=wanted)
    tap = X0Tap(pipe, cfg["family"])
    tok = pipe.tokenizer
    tok3 = getattr(pipe, "tokenizer_3", None)     # SD3.5's T5 tokenizer
    if cfg["family"] in ("flux", "hunyuan"):
        # T5 (mT5 on Hunyuan-DiT, whose cross-attention keys are the 77 BERT tokens then the 256 mT5 tokens)
        tok3 = pipe.tokenizer_2

    json.dump(dict(model=args.model, repo=cfg["repo"], steps=args.steps, guidance=guidance, size=size, fixed_seed=args.fixed_seed,
                   include_seeds=args.include_seeds, seed_list=args.seed_list,
                   scheduler=type(pipe.scheduler).__name__, dtype=args.dtype, n_seeds=args.n_seeds,
                   log_steps=args.log_steps, preview_steps=preview_steps, sa_layers=args.sa_layers,
                   store_ca_maps=ca_steps, ca_map_res=args.ca_map_res, log_mode=args.log_mode, deployed_maps=args.deployed_maps,
                   hook_version="2026-10-12 (both aggregation orders, conditional-branch entropy, txtmass on SD3.5, stored CA maps)",
                   gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), host=os.uname().nodename,
                   torch=torch.__version__), open(out / "run_config.json", "w"), indent=2)

    # ---- resume: the (image_id, seed, step) rows already written by earlier runs into any results*.csv of the pool
    done = set()
    if args.resume:
        prev_files = sorted(out.glob("results*.csv"))
        prev = pd.concat([pd.read_csv(f, on_bad_lines="skip") for f in prev_files], ignore_index=True) if prev_files else pd.DataFrame()
    if args.resume and len(prev):
        done = set(zip(prev.image_id.astype(str), prev.seed.astype(int), prev.step.astype(int)))

    # ---- generation: one image per (prompt, seed)
    rows: List[dict] = []
    for _, r in df.iterrows():
        ds, pid, prompt = str(r["Dataset"]), str(r["ID"]), str(r["prompt"]).strip()
        cat = str(r.get("category", ""))
        t_prompt = time.time()

        # The prompt's seeds (see "How the seeds are drawn" in the module docstring).
        if args.fixed_seed is not None:
            seeds = [args.fixed_seed]
        elif args.seed_list:
            seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
        else:
            seeds = seeds_for(f"{ds}:{pid}:{prompt}", args.n_seeds)
            if args.include_seeds:
                inc = [int(s) for s in args.include_seeds.split(",") if s.strip()]
                seeds = inc + [s for s in seeds if s not in inc][: max(0, args.n_seeds - len(inc))]

        # The prompt's nouns and their token positions, stored with the token vectors for the ABSS-style statistic.
        # n_active = the number of prompt tokens (no special tokens, no padding) in the encoder the attention runs over.
        words = body_tokens(prompt)
        clip_idx = token_indices(tok, prompt, words)
        t5_idx = token_indices(tok3, prompt, words, offset=0, max_len=(512 if cfg["family"] == "flux" else 256)) if tok3 is not None else []
        n_active = len(tok.encode(prompt, add_special_tokens=False, truncation=True, max_length=75))
        if cfg["family"] == "qwen":      # Qwen2.5-VL tokens; the pipeline drops the 34-token system template, so the prompt starts at 0
            # Byte-level BPE: a word in mid-sentence carries its leading space, so both spellings are searched.
            clip_idx = token_indices(tok, prompt, list(words) + [" " + w for w in words], offset=0, max_len=1024)
            n_active = len(tok.encode(prompt, add_special_tokens=False))
        if cfg["family"] == "flux":      # statistics are over the T5 sequence: content tokens 0..n_active-1, EOS at n_active
            clip_idx = t5_idx
            n_active = len(tok3.encode(prompt, add_special_tokens=False, truncation=True, max_length=511))
        t_prep = time.time() - t_prompt

        for seed in seeds:
            stem = f"{ds}_{pid}_seed{seed}"
            final_png = out / "images" / f"{stem}.png"
            npz_path = out / "stats" / f"{stem}.npz"
            if args.resume and final_png.exists() and npz_path.exists() and (pid, seed, args.steps) in done:
                continue

            hooks.disable() if no_hooks else hooks.enable()
            col.new_image(stem, n_active_tokens=n_active)
            col.step = 1                      # the hooks read this: the 1-based index of the step being computed
            step_rows, vec_steps, vec_layers, vec_data, t0 = [], [], None, [], time.time()

            def cb(p, i, t, kw):
                """The pipeline's callback after denoising step i (0-based): collects the step's readings, vectors and preview."""
                step = i + 1
                nonlocal vec_layers
                scal = col.flush_and_get_layer_means()      # {results.csv column: value} for this step
                vecs = col.flush_vecs()                     # {layer: token vector} for this step
                maps = col.flush_ca_maps() if hasattr(col, "flush_ca_maps") else {}
                if maps and step in ca_steps:
                    lay = sorted(maps)
                    # maps[k] is (batch, N_q, N_k); the batch holds one image, so [0] gives that image's map.
                    np.savez_compressed(out / "camaps" / f"{stem}_step{step:02d}.npz", layers=np.array(lay),
                                        maps=np.stack([maps[k][0].numpy() for k in lay]),
                                        step=step, n_active=n_active, clip_body_idx=np.array(clip_idx), size=size, res=args.ca_map_res)
                col.step = step + 1
                if step <= args.log_steps or step == args.steps:
                    if vec_layers is None and vecs:
                        vec_layers = sorted(vecs.keys())
                    # Token vectors exist only while the hooks are installed (steps <= log_steps);
                    # the final step is logged for image_path/gen_seconds without statistics.
                    if vec_layers and vecs and all(k in vecs for k in vec_layers):
                        vec_data.append(np.stack([vecs[k] for k in vec_layers]))
                        vec_steps.append(step)
                    step_rows.append(dict(dataset=ds, image_id=pid, category=cat, prompt=prompt, seed=seed,
                                          step=step, image_path="", **scal))
                if step == last_hooked and last_hooked < args.steps:
                    hooks.disable()   # remaining steps run at full speed; final step is logged without stats
                if step in preview_steps and tap.x0 is not None:
                    img = decode(p, tap.x0, cfg["family"])
                    pp = out / "previews" / f"{stem}_step{step:02d}.jpg"
                    img.save(pp, quality=92)
                return kw

            try:
                # Once per run: check that the hooks really are installed, by the class names of the processors.
                if not hasattr(main, "_hook_checked"):
                    main._hook_checked = True
                    names = list(hooks.module.attn_processors.values())
                    n_hook = sum(1 for q in names if type(q).__name__.startswith(("Vec", "Fast")) or "AE" in type(q).__name__ or "SA" in type(q).__name__)
                    print(f"[harness] attention processors installed: {n_hook}/{len(names)} are hooks", flush=True)
                    if n_hook == 0 and not no_hooks:
                        raise RuntimeError("no attention hooks installed; refusing to run a campaign without statistics")
                # Qwen-Image: true CFG needs a negative prompt.
                gkw = dict(true_cfg_scale=guidance, negative_prompt=" ") if cfg["family"] == "qwen" else dict(guidance_scale=guidance)
                if args.model == "flux_schnell":
                    gkw["max_sequence_length"] = 256   # schnell's T5 length (its model card)
                res = pipe([prompt], num_inference_steps=args.steps, **gkw,
                           width=size, height=size,
                           generator=torch.Generator(device="cuda").manual_seed(seed),
                           callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
                res.images[0].save(final_png)
                # The last logged row is the final step's: it carries the image path and the generation time.
                if step_rows:
                    step_rows[-1]["image_path"] = str(final_png)
                    step_rows[-1]["gen_seconds"] = round(time.time() - t0, 2)
                np.savez_compressed(npz_path, steps=np.array(vec_steps), layers=np.array(vec_layers or []),
                                    vec=np.stack(vec_data) if vec_data else np.zeros((0,)),
                                    clip_body_idx=np.array(clip_idx), t5_body_idx=np.array(t5_idx),
                                    n_active=n_active, body_words=np.array(words),
                                    body_method=getattr(body_tokens, "method", "?"),
                                    text_encoder={"flux": "t5", "hunyuan": "bert77+mt5_256", "qwen": "qwen2.5-vl"}.get(cfg["family"], "clip"))
                rows.extend(step_rows)
            except Exception as e:
                # One failed image must not end the run: report it and go on. Its rows are not written, so --resume retries it.
                import traceback
                print(f"[harness] ERROR {stem}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()   # gc.collect() per image cost seconds; Python's own GC suffices

        # Write the prompt's rows as soon as its seeds are done, so that an interrupted job loses one prompt at most.
        if rows:
            _append(results_path, rows)
            rows = []
        print(f"[harness] done prompt {ds}/{pid} ({len(seeds)} seeds) prep={t_prep:.2f}s total={time.time() - t_prompt:.1f}s", flush=True)

    if rows:
        _append(results_path, rows)
    print("[harness] finished.")


def _append(path: Path, rows: List[dict]):
    """Appends rows to the results file, writing the header once.

    Duplicates left by a restarted job are dropped when the file is read (pool_index.py, seed_select_eval.py); rewriting the
    whole file after every prompt would make the run quadratic in the number of prompts.
    """
    df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        extra = [c for c in df.columns if c not in cols]
        if extra:  # schema grew (new layer column): rewrite once with the union of columns
            prev = pd.read_csv(path)
            df = pd.concat([prev, df], ignore_index=True)
            df.to_csv(path, index=False)
            return
        df = df[cols]
        df.to_csv(path, index=False, header=False, mode="a")
    else:
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
