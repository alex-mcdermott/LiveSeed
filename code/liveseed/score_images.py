#!/usr/bin/env python
"""score_images.py: scores images against their prompts with the automatic evaluators (HPSv2.1, ImageReward, PickScore, CLIP).

HPSv2.1 on the finished images is the default calibration target of LiveSeed (Sec. 3.3 of the paper); the other evaluators
are held out and used to check the selection (Sec. 4.1). The same script scores previews for the preview baselines.

Input   --index  a CSV with at least the columns `image_path` and `prompt`; every other column is carried through.
                 Typically index_finals.csv from pool_index.py, or an index of preview images.
Output  --out    the same rows plus one column per evaluator:
                   imagereward   ImageReward v1.0.
                   hpsv2         HPS v2.1.
                   pickscore     PickScore, scaled by the model's logit scale (about 20-23, as in the literature).
                   clip_l14      the raw cosine similarity of OpenAI CLIP ViT-L/14.
                   vqascore      VQAScore with CLIP-FlanT5-XXL (only with --metrics vqa).
                   hpsv3         HPSv3 (only with --metrics hpsv3).
                 An image that cannot be opened keeps its row, with the reason in `score_error` and no scores. A score
                 that fails on one image is written as NaN.

Usage:  python score_images.py --index runs/sd14_pool/index_finals.csv --out runs/sd14_pool/scores_finals.csv
        --metrics hps scores HPSv2.1 alone (the default is ir,hps,pick,clip).
        --resume keeps the rows already in --out and scores only the images that are missing from it.

Weights download on first use into HF_HOME. A requested evaluator that cannot be loaded stops the run before any image
is scored, so that a scores file never lacks a column that was asked for.
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- evaluators
def load_ir():
    import ImageReward as RM
    return RM.load("ImageReward-v1.0", device=DEV)


def load_pick():
    """PickScore: a fine-tuned CLIP ViT-H/14, used with that CLIP model's processor. Returns (processor, model)."""
    from transformers import AutoProcessor, AutoModel
    proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(DEV)
    return proc, model


@torch.no_grad()
def pick_score(proc, model, image, prompt):
    """PickScore of one image: logit scale x cosine similarity of the normalised text and image embeddings."""
    im = proc(images=image, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEV)
    tx = proc(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEV)
    ie = model.get_image_features(**im)
    ie = ie / ie.norm(dim=-1, keepdim=True)
    te = model.get_text_features(**tx)
    te = te / te.norm(dim=-1, keepdim=True)
    return float((model.logit_scale.exp() * (te @ ie.T)).item())


def load_clip():
    """OpenAI CLIP ViT-L/14 through open_clip. Returns (model, image preprocessing, tokenizer)."""
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-L-14")
    return model.eval().to(DEV), pre, tok


@torch.no_grad()
def clip_cos(model, pre, tok, image, prompt):
    """Raw cosine similarity between the CLIP embeddings of the image and of the prompt (no logit scale)."""
    im = pre(image).unsqueeze(0).to(DEV)
    tx = tok([prompt]).to(DEV)
    ie = model.encode_image(im)
    ie = ie / ie.norm(dim=-1, keepdim=True)
    te = model.encode_text(tx)
    te = te / te.norm(dim=-1, keepdim=True)
    return float((ie @ te.T).item())


def load_hpsv3():
    """HPSv3 (Ma et al. 2025, MizzenAI/HPSv3): a Qwen2-VL-7B-based preference model, used as a held-out evaluator.

    Returns score(path, prompt). The model's reward() returns, per image, a (mu, sigma) pair; the score is mu.
    """
    from hpsv3 import HPSv3RewardInferencer
    inf = HPSv3RewardInferencer(device=DEV)

    def score(path, prompt):
        r = inf.reward([path], [prompt])[0]
        return float(r[0]) if hasattr(r, "__len__") else float(r)

    return score


def load_vqa(model_name="clip-flant5-xxl"):
    """VQAScore (Lin et al., 2024) through the t2v_metrics package; the CLIP-FlanT5-XXL judge."""
    import t2v_metrics
    return t2v_metrics.VQAScore(model=model_name)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)       # the CSV of images to score
    ap.add_argument("--out", required=True)         # the CSV of scores to write
    ap.add_argument("--metrics", default="ir,hps,pick,clip",
                    help="comma list of ir,hps,pick,clip,vqa,hpsv3 (vqa = VQAScore, CLIP-FlanT5-XXL, needs t2v_metrics; hpsv3 needs the hpsv3 package)")
    ap.add_argument("--image-col", default="image_path")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    want = set(args.metrics.split(","))

    # Keep the rows that name an image. In a harness results file only the final-step rows do.
    df = pd.read_csv(args.index)
    df = df[df[args.image_col].notna() & (df[args.image_col].astype(str).str.lower() != "nan") & (df[args.image_col].astype(str).str.len() > 0)].copy()
    out = Path(args.out)
    if args.resume and out.exists():
        prev = pd.read_csv(out, on_bad_lines="skip")
        have = set(prev[args.image_col])
        df = df[~df[args.image_col].isin(have)]
        print(f"[score] resume: {len(have)} scored, {len(df)} to do")
    if df.empty:
        print("[score] nothing to do")
        return

    # ---- load the requested evaluators; a failure is reported here and made fatal just below
    ir = pick = clip = None
    hps = False
    if "ir" in want:
        try:
            ir = load_ir()
            print("[score] ImageReward ready")
        except Exception as e:
            warnings.warn(f"ImageReward unavailable: {e}")
    if "pick" in want:
        try:
            pick = load_pick()
            print("[score] PickScore ready")
        except Exception as e:
            warnings.warn(f"PickScore unavailable: {e}")
    if "clip" in want:
        try:
            clip = load_clip()
            print("[score] CLIP ViT-L/14 ready")
        except Exception as e:
            warnings.warn(f"open_clip unavailable: {e}")
    if "hps" in want:
        try:
            import hpsv2
            hps = True
            print("[score] HPSv2 ready")
        except Exception as e:
            warnings.warn(f"hpsv2 unavailable: {e}")
    hps3 = None
    if "hpsv3" in want:
        try:
            hps3 = load_hpsv3()
            print("[score] HPSv3 ready")
        except Exception as e:
            warnings.warn(f"HPSv3 unavailable: {e}")
    vqa = None
    if "vqa" in want:
        try:
            vqa = load_vqa()
            print("[score] VQAScore (clip-flant5-xxl) ready")
        except Exception as e:
            warnings.warn(f"VQAScore unavailable: {e}")

    # A requested metric that failed to load stops the run: otherwise the scores file would silently lack its column.
    missing = [m for m, ok in (("ir", ir is not None), ("pick", pick is not None), ("clip", clip is not None), ("hps", hps),
                               ("hpsv3", hps3 is not None), ("vqa", vqa is not None)) if m in want and not ok]
    if missing:
        raise SystemExit(f"[score] FATAL: requested metric(s) unavailable: {missing} (see the warnings above)")

    # ---- score image by image; a metric that fails on one image is NaN for that image and the run goes on
    rows = []
    flushed = False      # set once this run has written to --out, so that later batches are added, not overwritten
    for _, r in tqdm(df.iterrows(), total=len(df), desc="scoring"):
        p = str(r[args.image_col])
        prompt = str(r["prompt"])
        rec = dict(r)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            rec["score_error"] = str(e)
            rows.append(rec)
            continue
        if ir is not None:
            try:
                rec["imagereward"] = float(ir.score(prompt, p))
            except Exception as e:
                rec["imagereward"] = np.nan
        if hps:
            try:
                import hpsv2
                s = hpsv2.score(img, prompt, hps_version="v2.1")
                rec["hpsv2"] = float(s[0] if isinstance(s, (list, tuple, np.ndarray)) else s)
            except Exception as e:
                rec["hpsv2"] = np.nan
                if not globals().get("_hps_warned"):       # report the reason once, not for every image
                    warnings.warn(f"hpsv2 scoring failed (first failure): {type(e).__name__}: {e}")
                    globals()["_hps_warned"] = True
        if pick is not None:
            try:
                rec["pickscore"] = pick_score(*pick, img, prompt)
            except Exception as e:
                rec["pickscore"] = np.nan
        if clip is not None:
            try:
                rec["clip_l14"] = clip_cos(*clip, img, prompt)
            except Exception as e:
                rec["clip_l14"] = np.nan
        if vqa is not None:
            try:
                rec["vqascore"] = float(vqa(images=[p], texts=[prompt]).flatten()[0])
            except Exception as e:
                rec["vqascore"] = np.nan
        if hps3 is not None:
            try:
                rec["hpsv3"] = hps3(p, prompt)
            except Exception as e:
                rec["hpsv3"] = np.nan
        rows.append(rec)
        if len(rows) % 200 == 0:      # write every 200 images, so that an interrupted run can be continued with --resume
            _flush(out, rows, args.resume or flushed)
            flushed = True
            rows = []
    _flush(out, rows, args.resume or flushed)
    print("[score] done ->", out)


def _flush(out: Path, rows, resume: bool):
    """Writes `rows` to `out`. With `resume`, the rows already in the file are kept and the new ones are added after them;
    without it, the file is replaced by `rows`."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if resume and out.exists():
        new = pd.concat([pd.read_csv(out, on_bad_lines="skip"), new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
