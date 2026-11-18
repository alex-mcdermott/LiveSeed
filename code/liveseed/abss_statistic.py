#!/usr/bin/env python
"""abss_statistic.py: the seed score of the ABSS baseline, computed from the maps and vectors that seed_harness.py stores.

ABSS (Zhang, Liu & Hong, NeurIPS 2026, sec. 3.2 and Appendix B) ranks the seeds of a prompt by the cross-attention mass on
the prompt's core tokens at one early step (step 10 of 50). This file is our implementation of that statistic as the ABSS
paper describes it (our paper: App. B, "Our implementation of ABSS").

Inputs
  --run      a pool folder of seed_harness.py: camaps/ for the U-Nets (generate with --store-ca-maps 10), stats/*.npz for
             the DiTs, and results*.csv for the logged text mass (txtmass_<block>).
  --model    sd14, sd15, sd21, sdxl, sd35, sd35l, flux or hunyuan.
  --tokens   the core words of every prompt, a json in the ABSS authors' format (abss_core_tokens_gpt.py writes it).
  --prompts  the prompt list (Dataset, ID, prompt) the pool was generated from.
  --step     the step the statistic is read at (default 10); --block the DiT block (default 12); --k and --sigma the
             Gaussian's half-width and standard deviation (defaults 1 and 0.5).
Output
  --out      a CSV with one row per (image_id, seed): dataset, image_id, seed, step, n_core_tokens, core_words and one
             column abss_<variant> per form of the statistic listed below (on the DiTs also block and txtmass).
             `abss_core` is the paper's form. The per-token smoothed maps are not kept.
Usage (from code/liveseed/; the first two lines make the inputs of the third, and the second needs OPENAI_API_KEY)
  python seed_harness.py --model sd14 --csv ../../prompts/ABSS_DrawBench.csv --steps 50 --log-steps 10 --store-ca-maps 10 --out runs/sd14_abss
  python abss_core_tokens_gpt.py --prompts ../../prompts/ABSS_DrawBench.csv --out runs/tokens/drawbench_gpt.json
  python abss_statistic.py --run runs/sd14_abss --model sd14 --tokens runs/tokens/drawbench_gpt.json --prompts ../../prompts/ABSS_DrawBench.csv --out runs/sd14_abss/abss_scores.csv

The statistic on U-Nets (SD1.x at 16x16, SD2.x at 24x24, SDXL at 32x32)
  1. Aggregate: stack every cross-attention layer and head at that resolution and average, giving one map A[h, w, token].
     The harness stores the heads-averaged conditional-branch map of each such layer; the mean over layers of those is the
     same average, because every layer has the same number of heads.
  2. Sharpen: at every position, a softmax over the tokens of beta * A, with beta = 100.
  3. For every core token: its spatial slice, smoothed by a normalised (2k+1)x(2k+1) Gaussian with reflection padding,
     then the spatial mean. The score is the mean over the core tokens.
  Steps 2-3 are, word for word, Attend-and-Excite's `_compute_max_attention_per_index` (diffusers,
  pipeline_stable_diffusion_attend_and_excite.py; InitNO's code, on which ABSS's Stable Diffusion code is built, has the same
  function), with the final max over positions replaced by the mean that the ABSS paper states. That function first drops
  the beginning-of-sequence token (BOS, index 0) and the last position (index 76) from the token axis, takes the softmax of
  100 x the remaining 75 probabilities, and smooths with a 3x3 Gaussian of sigma 0.5 (k = 1) under reflection padding of 1.
  The slicing matters: without it the softmax collapses onto BOS, which holds about 75% of SD1.x's cross-attention at every
  position, and every core token's score is about 1e-35. The sliced form is therefore the only reading under which the
  statistic is not degenerate, and it is the default here. The ABSS paper's remark "including BOS/EOS (index 0 for BOS, n-1
  for EOS)" describes the original indexing, which the shift by one undoes.
  Columns:
    abss_core           the default: slicing as above, the Gaussian of --k and --sigma (3x3, sigma 0.5), mean over positions.
    abss_core_k2s1      the same with a 5x5 Gaussian (k = 2, sigma = 1).
    abss_core_nosmooth  the same without the Gaussian.
    abss_core_max       Attend-and-Excite's own statistic: the max over positions instead of the mean.
    abss_core_special   the literal reading: the softmax over all 77 positions, with BOS and the end-of-sequence token
                        (EOS) added to the core set. Degenerate, as explained above.
    abss_core_raw       no temperature: the plain mean probability on the core tokens.

The statistic on DiTs (SD3.5, FLUX, Hunyuan-DiT)
  One block (--block; ABSS reads block 12 on FLUX, the default here, and block 18 on SD3.5-Large). The attention of the
  image queries on the text keys, averaged over image queries and heads, gives one number per text token; a normalised
  1-D Gaussian along the tokens with reflection padding smooths it; the score is the mean over the core tokens. The
  harness's token vector is that distribution normalised over the text tokens, so multiplying by the logged text mass
  (txtmass_<block>) restores the raw mean attention.
  Columns:
    abss_core             the paper's form: smoothed, mean over the core tokens.
    abss_code             the form of ABSS's released code: the plain mean over the core tokens, no smoothing.
    abss_core_clip        SD3.5 only: abss_core over the CLIP positions of the core words alone.
    abss_core_normalised  abss_core on the normalised vector, without the text-mass factor.
  Token positions: SD3.5's text keys are 77 CLIP tokens followed by 256 T5 tokens (positions 77 and up); a core word is
  matched in both tokenizers and both sets of positions are used. FLUX reads T5 only: no BOS, EOS at n_active and padding
  after it; the word-to-T5-span matching is ported from the authors' abss.py. Hunyuan-DiT's keys are 77 BERT tokens
  followed by 256 mT5 tokens; as in the authors' backend, only the mT5 positions are used.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BETA = 100.0     # the temperature of the softmax over tokens (step 2 on U-Nets)
# The processor name of block b whose attention the DiT statistic reads, per model.
DIT_BLOCK = {"sd35": "transformer_blocks.{b}.attn.processor", "sd35l": "transformer_blocks.{b}.attn.processor", "flux": "transformer_blocks.{b}.attn.processor",
             "hunyuan": "blocks.{b}.attn2.processor"}   # Hunyuan-DiT: the cross-attention of block b, as in the authors' hunyuan backend (which probes block 12)


# ----------------------------------------------------------------------------- Gaussian smoothing
def gauss1d(k, sigma):
    """A normalised 1-D Gaussian kernel with 2k+1 taps."""
    x = np.arange(-k, k + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def smooth2d_reflect(M, k=2, sigma=1.0):
    """M: a 2-D map. Returns it smoothed by a normalised (2k+1)x(2k+1) Gaussian with reflection padding (same shape).

    The Gaussian is separable, so it is applied along the rows and then along the columns.
    """
    g = gauss1d(k, sigma)
    P = np.pad(M, k, mode="reflect")
    P = np.apply_along_axis(lambda r: np.convolve(r, g, mode="valid"), 1, P)
    P = np.apply_along_axis(lambda c: np.convolve(c, g, mode="valid"), 0, P)
    return P


def smooth1d_reflect(v, k=2, sigma=1.0):
    """v: a 1-D vector. Returns it smoothed by a normalised Gaussian of 2k+1 taps with reflection padding (same length)."""
    g = gauss1d(k, sigma)
    P = np.pad(v, k, mode="reflect")
    return np.convolve(P, g, mode="valid")


# ----------------------------------------------------------------------------- core words -> token indices
def words_from_positions(prompt, positions):
    """The words at the given 1-based positions of the whitespace-split prompt (the tokens file stores positions)."""
    from abss_core_tokens_gpt import split_words
    w = split_words(prompt)
    return [w[p - 1] for p in positions if 1 <= p <= len(w)]


def clip_indices(tokenizer, prompt, words):
    """Positions of the words' tokens in the 77-token CLIP sequence, with BOS at position 0."""
    from seed_harness import token_indices
    return token_indices(tokenizer, prompt, words, offset=1, max_len=77)


def t5_indices(tokenizer_t5, prompt, words, max_len=512):
    """Positions of the words' tokens in the padded T5 sequence, as a sorted list.

    A port of the authors' word_positions_to_t5_indices: the words are searched in order, each from where the previous
    match ended, so a repeated word maps to its successive occurrences. A word is first looked for as a single token; if
    there is none, as a run of up to eight consecutive pieces that spell it.
    """
    enc = tokenizer_t5([prompt], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
    toks = tokenizer_t5.convert_ids_to_tokens(enc.input_ids[0].tolist())
    out, cursor = [], 0
    stripped = [t.lstrip("▁").lower() for t in toks]      # "▁" marks the start of a word in SentencePiece
    for w in words:
        w = w.lower()
        span = []
        for i in range(cursor, len(toks)):
            if toks[i].lower() == ("▁" + w) or stripped[i] == w:
                span = [i]
                break
        if not span:
            for i in range(cursor, len(toks)):
                acc = ""
                s = []
                for j in range(i, min(i + 8, len(toks))):
                    acc += stripped[j]
                    s.append(j)
                    if acc == w:
                        span = s
                        break
                    if len(acc) > len(w):
                        break
                if span:
                    break
        if span:
            cursor = span[-1] + 1
            out.extend(span)
    return sorted(dict.fromkeys(out))


def load_tokenizers(model):
    """Returns (first tokenizer, T5 tokenizer or None) of the model: CLIP and T5 in general, BERT and mT5 for Hunyuan-DiT."""
    from transformers import CLIPTokenizer, T5TokenizerFast
    if model == "hunyuan":   # BERT (77 tokens) and mT5 (256); the ABSS score uses the mT5 positions only, offset by the 77 BERT keys
        from transformers import AutoTokenizer
        rep = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
        return AutoTokenizer.from_pretrained(rep, subfolder="tokenizer"), AutoTokenizer.from_pretrained(rep, subfolder="tokenizer_2")
    rep = {"sd14": "CompVis/stable-diffusion-v1-4", "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5", "sd21": "sd2-community/stable-diffusion-2-1",
           "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
           "sd35": "stabilityai/stable-diffusion-3.5-medium", "sd35l": "stabilityai/stable-diffusion-3.5-large", "flux": "black-forest-labs/FLUX.1-dev"}[model]
    clip = CLIPTokenizer.from_pretrained(rep, subfolder="tokenizer")
    t5 = T5TokenizerFast.from_pretrained(rep, subfolder="tokenizer_3" if model in ("sd35", "sd35l") else "tokenizer_2") if model in ("sd35", "sd35l", "flux") else None
    return clip, t5


# ----------------------------------------------------------------------------- scores
def unet_scores(run, step, tokens, prompts, clip_tok, k, sigma, special_eos_fn):
    """The U-Net statistic for every stored map of `step`: one row per (image_id, seed) with the abss_* columns.

    run: the pool folder; tokens: {prompt id: {"entity": [1-based word positions], ...}}; prompts: {prompt id: prompt};
    k, sigma: the Gaussian of the default column. Maps of prompts that are not in `prompts` are skipped.
    """
    recs = []
    files = sorted(glob.glob(str(run / "camaps" / f"*_step{step:02d}.npz")))
    for f in files:
        # File names are <Dataset>_<ID>_seed<seed>_step<kk>.npz; the dataset name is taken up to the first underscore.
        m = re.match(r"([^_]+)_(.+)_seed(\d+)_step\d+\.npz$", os.path.basename(f))
        if not m:
            continue
        ds, pid, seed = m.group(1), m.group(2), int(m.group(3))
        if pid not in prompts:
            continue
        z = np.load(f, allow_pickle=True)
        A = z["maps"].astype(np.float32).mean(0)          # [N_q, N_k]: mean over the stored layers (= over layers and heads)
        n_active = int(z["n_active"])
        res = int(z["res"]) if "res" in z else int(round(math.sqrt(A.shape[0])))
        if A.shape[0] != res * res:
            continue
        words = words_from_positions(prompts[pid], tokens.get(pid, {}).get("entity", []))
        core = clip_indices(clip_tok, prompts[pid], words)                         # BOS-inclusive indexing (BOS 0, tokens from 1)
        eos = min(n_active + 1, A.shape[1] - 1)
        rec = dict(dataset=ds, image_id=pid, seed=seed, step=step, n_core_tokens=len(core), core_words=" ".join(words))

        def softmax_tokens(M):                                                     # temperature-scaled softmax over the token axis at every position
            S = M * BETA
            S = S - S.max(1, keepdims=True)          # subtract the row maximum so that exp() cannot overflow
            E = np.exp(S)
            return E / E.sum(1, keepdims=True)

        # Attend-and-Excite / InitNO form: drop BOS (0) and the last position, softmax over the 75 remaining, indices shift by one
        P_ae = softmax_tokens(A[:, 1:-1])
        core_ae = [i - 1 for i in core if 1 <= i <= A.shape[1] - 2]

        def score(P, idx, kk, ss):
            """Mean over the tokens `idx` of the spatial mean of the token's smoothed map; NaN without tokens."""
            if not idx:
                return np.nan
            return float(np.mean([smooth2d_reflect(P[:, i].reshape(res, res), kk, ss).mean() for i in idx]))

        rec["abss_core"] = score(P_ae, core_ae, k, sigma)                          # the default: A&E slicing, k = 1, sigma = 0.5, mean over positions
        rec["abss_core_k2s1"] = score(P_ae, core_ae, 2, 1.0)
        rec["abss_core_nosmooth"] = float(np.mean([P_ae[:, i].mean() for i in core_ae])) if core_ae else np.nan
        # A&E's own max statistic
        rec["abss_core_max"] = float(np.mean([smooth2d_reflect(P_ae[:, i].reshape(res, res), k, sigma).max() for i in core_ae])) if core_ae else np.nan
        P_all = softmax_tokens(A)                                                   # the literal reading: every position in the softmax, BOS and EOS in the set
        rec["abss_core_special"] = score(P_all, sorted(set(core) | {0, eos}), k, sigma)
        # no temperature: the plain attention mass on the core tokens
        rec["abss_core_raw"] = float(np.mean([A[:, i].mean() for i in core])) if core else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def dit_scores(run, step, block, tokens, prompts, clip_tok, t5_tok, model, k, sigma):
    """The DiT statistic at (`step`, `block`) for every image of the pool: one row per (image_id, seed) with the abss_* columns.

    Arguments as in unet_scores; clip_tok and t5_tok are the model's two tokenizers (load_tokenizers).
    """
    import fit_trajectory_metric as F
    res = F._read_results(run).drop_duplicates(subset=["image_id", "seed", "step"], keep="last")
    res["image_id"] = res["image_id"].astype(str)
    layer = DIT_BLOCK[model].format(b=block)
    col = "txtmass_" + layer.replace(".", "_")
    # The text mass of the block at this step, per (image_id, seed): the factor that turns the stored, normalised vector
    # back into the raw mean attention.
    tm = res[res.step == step].set_index(["image_id", "seed"])[col] if col in res.columns else None
    if tm is None and model != "hunyuan":
        print(f"[abss] WARNING: {col} not in results.csv; the DiT score is on the normalised vector (ranking unchanged if the text mass is constant across seeds)", flush=True)
    recs = []
    for f in sorted(glob.glob(str(run / "stats" / "*.npz"))):
        # File names are <Dataset>_<ID>_seed<seed>.npz; the dataset name is taken up to the first underscore.
        m = re.match(r"([^_]+)_(.+)_seed(\d+)\.npz$", os.path.basename(f))
        if not m:
            continue
        ds, pid, seed = m.group(1), m.group(2), int(m.group(3))
        if pid not in prompts:
            continue
        z = np.load(f, allow_pickle=True)
        steps = [int(s) for s in z["steps"]]
        if step not in steps or z["vec"].ndim != 3:
            continue
        layers = [str(l) for l in z["layers"]]
        if layer not in layers:
            continue
        v = z["vec"][steps.index(step), layers.index(layer)].astype(np.float64)       # normalised over the text tokens
        n_active = int(z["n_active"])
        T = len(v)
        scale = float(tm.get((pid, seed), np.nan)) if tm is not None else 1.0
        raw = v * (scale if np.isfinite(scale) else 1.0)
        words = words_from_positions(prompts[pid], tokens.get(pid, {}).get("entity", []))
        if model == "flux":
            core = t5_indices(t5_tok, prompts[pid], words)
            core_clip = []
            content_len = n_active + 1                                                    # content tokens then EOS; padding after
        elif model == "hunyuan":
            # the authors' hunyuan backend: map_entity_tokens on tokenizer_2 (mT5, 256), indices + 77 (the BERT keys come first), plain
            # mean of the block's cross-attention probabilities at those keys. Our vec is already the attention probability (mean over
            # heads and image queries of the softmax over the 333 text keys), so no text-mass factor (scale stays 1). Difference: the
            # authors average both CFG branches (their maps keep the batch); the harness's vec is the conditional branch only.
            core = [77 + i for i in t5_indices(t5_tok, prompts[pid], words, max_len=256) if 77 + i < T]
            core_clip = []
            content_len = T
        else:
            # SD3.5: the CLIP positions and the T5 positions (offset by the 77 CLIP keys) of the core words, together.
            cc = clip_indices(clip_tok, prompts[pid], words)
            ct = [77 + i for i in t5_indices(t5_tok, prompts[pid], words, max_len=256) if 77 + i < T]
            core = sorted(set(cc) | set(ct))
            core_clip = cc
            content_len = T
        rec = dict(dataset=ds, image_id=pid, seed=seed, step=step, block=block, n_core_tokens=len(core), core_words=" ".join(words), txtmass=scale)
        sm = smooth1d_reflect(raw, k, sigma)
        rec["abss_core"] = float(np.mean(sm[core])) if core else np.nan                  # paper: smoothed, mean over the core tokens
        rec["abss_code"] = float(np.mean(raw[core])) if core else np.nan                 # released code: plain mean
        rec["abss_core_clip"] = float(np.mean(sm[core_clip])) if core_clip else np.nan
        rec["abss_core_normalised"] = float(np.mean(smooth1d_reflect(v, k, sigma)[core])) if core else np.nan   # without the text-mass factor
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True, choices=["sd14", "sd15", "sd21", "sdxl", "sd35", "sd35l", "flux", "hunyuan"])
    ap.add_argument("--tokens", required=True, help="core-token json (the ABSS authors' format; abss_core_tokens_gpt.py writes it)")
    ap.add_argument("--prompts", required=True, help="prompt csv (Dataset,ID,prompt)")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    # Prompt ids are matched as strings. The keys of the tokens file are normalised through int(), so ids must be integers.
    tokens = {str(int(k)): v for k, v in json.load(open(a.tokens, encoding="utf-8")).items()}
    prompts = {str(r["ID"]).strip(): str(r["prompt"]).strip() for r in pd.read_csv(a.prompts, encoding="utf-8-sig").to_dict("records")}
    clip_tok, t5_tok = load_tokenizers(a.model)
    # ABSS reports no SDXL results. On SDXL the U-Net statistic is read at its coarsest cross-attention layers, 32x32 at
    # 1024 px (generate the pool with --ca-map-res 32).
    if a.model in ("sd14", "sd15", "sd21", "sdxl"):
        df = unet_scores(run, a.step, tokens, prompts, clip_tok, a.k, a.sigma, None)
    else:
        df = dit_scores(run, a.step, a.block, tokens, prompts, clip_tok, t5_tok, a.model, a.k, a.sigma)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[abss] wrote {a.out}: {len(df)} candidates, {df.image_id.nunique() if len(df) else 0} prompts, mean core tokens {df.n_core_tokens.mean() if len(df) else float('nan'):.2f}, "
          f"{int((df.n_core_tokens == 0).sum()) if len(df) else 0} with none; columns {[c for c in df.columns if c.startswith('abss_')]}")


if __name__ == "__main__":
    main()
