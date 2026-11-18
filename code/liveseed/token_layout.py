#!/usr/bin/env python
"""token_layout.py: where the prompt's tokens and the special tokens sit in a stored text-token attention vector.

seed_harness.py stores, per image, the attention mass on every text key (stats/<stem>.npz: vec[step, layer, token]). The
readers of those vectors (the ABSS-style statistic in seed_select_eval.py and fit_trajectory_metric.py) need to know which
positions hold the prompt and which hold special tokens. That depends on the text encoder the attention runs over:

  CLIP  (SD1.x, SD2.x, SDXL, and the first 77 positions of SD3.5): the beginning-of-sequence token <BOS> at 0, the
        prompt's tokens at 1..n_active, the end-of-sequence token <EOS> at n_active + 1 and padding after it. SD3.5
        appends its 256 T5 positions from position 77 on.
  T5    (FLUX.1): no <BOS>; the prompt's tokens at 0..n_active-1, </s> (EOS) at n_active and padding after it. FLUX
        attends to the padding (it is not masked), which is why the vector carries most of its mass beyond the prompt.

Assuming the CLIP layout on FLUX would drop the first prompt token from the content range, include </s> in it, and report
the first word's mass as the BOS mass; hence this helper. Only vectors whose `text_encoder` field is "t5" (FLUX) get the T5
layout; every other vector is read with the CLIP layout.

Usage:  z = numpy.load("runs/sd14_pool/stats/<stem>.npz");  L = layout(z);  z["vec"][..., L["content"]]
"""
import numpy as np


def layout(z):
    """z: an opened stats/<stem>.npz of the harness. Returns a dict of positions along the token axis of z["vec"]:

    content   the positions of the prompt's tokens (a list);
    bos, eos  the positions of the beginning and end tokens (bos is None for T5);
    special   [bos, eos] without the None;
    T         the length of the token axis (0 when the file holds no vector);
    t5        True when the vector is over a T5 sequence;
    n_active  the number of prompt tokens the harness counted.
    The ranges are clipped to T, so a prompt truncated by the encoder stays in bounds.
    """
    T = int(z["vec"].shape[-1]) if z["vec"].ndim == 3 else 0
    n_active = int(z["n_active"])
    t5 = ("text_encoder" in z.files) and str(z["text_encoder"]) == "t5"
    if t5:
        content = list(range(0, min(n_active, T)))
        eos = min(n_active, T - 1)
        bos = None
    else:
        content = list(range(1, min(n_active + 1, T)))
        eos = min(n_active + 1, T - 1)
        bos = 0
    special = [i for i in (bos, eos) if i is not None]
    return dict(content=content, bos=bos, eos=eos, special=special, T=T, t5=t5, n_active=n_active)


def body_indices(z):
    """The positions of the prompt's subject words (its nouns) that the harness stored, as a sorted list.

    They are in the vector's own indexing: the field is called clip_body_idx, but on FLUX it holds T5 positions.
    Positions outside the token axis are dropped.
    """
    T = int(z["vec"].shape[-1]) if z["vec"].ndim == 3 else 0
    return sorted(set(int(i) for i in list(z["clip_body_idx"]) if 0 <= int(i) < T))
