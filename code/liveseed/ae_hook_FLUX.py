"""
ae_hook_FLUX.py - attention-entropy hooks for FLUX.1 (diffusers FluxTransformer2DModel).

This file computes the reading of Sec. 3.1 (Eq. 2) on FLUX's joint attention. The design (one collector, one processor per
layer, `col.step`, a flush per denoising step, entropies divided by ln of the number of keys, statistics that never
interrupt generation) is that of ae_hook.py, the reference; only the differences are described here.

What is hooked
  FLUX has 19 double-stream blocks (`transformer_blocks.N.attn`: joint attention over the concatenated [text, image]
  sequence, text first, separate projections for the two streams) and 38 single-stream blocks
  (`single_transformer_blocks.N.attn`: one set of projections over the already-concatenated sequence). Both use
  FluxAttnProcessor2_0. This module replaces it with a processor that reproduces it, keeping the fused attention kernel for
  the block's OUTPUT, and in addition materialises the attention probabilities of the IMAGE queries in row chunks (never the
  whole matrix at once) to log the statistics.

Readings returned by flush_and_get_layer_means() (<block> is the processor name with dots replaced by underscores, e.g.
ae_mean_single_transformer_blocks_0_attn_processor; S = N_txt + N_img keys)
  ae_mean_<block>      The LiveSeed reading, in the PER-ROW order of Eq. 2: the entropy of every image-query row over the
                       joint key set, then the mean over heads, image queries and the images of the batch, / ln S (the
                       SD3.5 quantity).
  ae_img_<block>       The same over the image keys only (rows renormalised), / ln N_img: the self-attention part.
  ae_avgfirst_<block>  The POOLED order: rows averaged over heads and image queries, then one entropy, / ln S (only with
                       extra_orders=True; App. G ablates the two orders).
  txtmass_<block>      Mean attention mass the image queries place on the text keys.
Other outputs
  flush_vecs()         {block: (N_txt,) vector}: how that text mass is shared among the T5 token positions (sums to one),
                       for the subject-token statistic of the ABSS baseline.
  values_per_sample()  {(block, step): (B,) array}: ae_mean separately for each image of the batch, which the live
                       algorithm needs to rank the seeds of one batch.

FLUX.1-dev is guidance-distilled: it runs one conditional forward per step, so there are no classifier-free-guidance (CFG)
branches to average and every batch row is an image.

Install on a diffusers pipeline:
    col = AttnEntropyCollector()                                   # or wanted={(name, step), ...} to read only those
    procs = {name: col.get_processor(name) for name in pipe.transformer.attn_processors}
    pipe.transformer.set_attn_processor(procs)

    col.new_image("key")
    # in the pipeline's step callback:
    scalars = col.flush_and_get_layer_means()
    vectors = col.flush_vecs()
    col.step = step + 1
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FluxAEAttnProcessor(nn.Module):
    """FluxAttnProcessor2_0 with the statistics added. `chunk` is the number of image queries whose rows are materialised
    at a time."""
    def __init__(self, parent: "AttnEntropyCollector", layer_name: str, chunk: int = 512):
        super().__init__()
        self.parent, self.layer_name, self.chunk = parent, layer_name, chunk

    @torch.no_grad()
    def _log_stats(self, q_img: torch.Tensor, k_all: torch.Tensor, n_txt: int):
        """Pushes this call's statistics to the collector.

        q_img: (B, H, N_img, D) image queries; k_all: (B, H, S, D) all keys, text first. Both are the tensors the model
        attends with (after normalisation and rotary embedding).
        """
        B, H, N_img, D = q_img.shape
        S = k_all.shape[2]
        scale = D ** -0.5
        kT = k_all.transpose(-1, -2)

        # Sums accumulated over the row chunks; they stay on the GPU so that the loop does not wait on the host.
        ent_sum = torch.zeros((), device=q_img.device, dtype=torch.float32)        # row entropies, whole batch
        ent_rows = torch.zeros(B, device=q_img.device, dtype=torch.float32)       # per-sample sum of the row entropies (the live batch)
        ent_img_sum = torch.zeros_like(ent_sum)                                    # row entropies over the image keys only
        txt_mass = torch.zeros_like(ent_sum)                                       # mass on the text keys
        txt_vec = torch.zeros(n_txt, device=q_img.device, dtype=torch.float32)     # that mass per text position
        row_acc = torch.zeros(S, device=q_img.device, dtype=torch.float32)       # pooled order: the row summed over heads and queries
        rows = 0
        for r0 in range(0, N_img, self.chunk):
            s = torch.matmul(q_img[:, :, r0:r0 + self.chunk], kT) * scale          # (B, H, c, S), model dtype
            p = torch.softmax(s.float(), dim=-1)                                   # softmax in fp32
            e = -(p * p.clamp_min(1e-12).log()).sum(-1)                               # (B, H, c)
            ent_sum += e.sum()
            ent_rows += e.sum((1, 2))
            if self.parent.extra_orders:
                row_acc += p.sum(dim=(0, 1, 2))
            pt = p[..., :n_txt]                                                    # text keys come first
            txt_mass += pt.sum(-1).sum()
            txt_vec += pt.sum(dim=(0, 1, 2))
            pi = p[..., n_txt:]                                                    # image keys, renormalised to a distribution
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
            ent_img_sum += (-(pi * pi.clamp_min(1e-12).log()).sum(-1)).sum()
            rows += B * H * p.shape[2]

        # Means over the rows (batch x heads x image queries), each divided by the entropy of a uniform row.
        self.parent._push(self.layer_name, float(ent_sum.item() / rows / math.log(max(S, 2))))
        self.parent._push_rows(self.layer_name, ent_rows / (rows / B) / math.log(max(S, 2)))     # (B,) per-sample readings, GPU
        self.parent._push_img(self.layer_name, float(ent_img_sum.item() / rows / math.log(max(N_img, 2))))
        if self.parent.extra_orders:
            a = (row_acc / row_acc.sum().clamp_min(1e-12)).clamp_min(1e-12)
            self.parent._push_avgfirst(self.layer_name, float((-(a * a.log()).sum()).item() / math.log(max(S, 2))))
        self.parent._push_txtmass(self.layer_name, float(txt_mass.item() / rows))
        v = txt_vec / txt_vec.sum().clamp_min(1e-12)
        self.parent._push_vec(self.layer_name, v.detach().cpu().numpy().astype(np.float32))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        """The attention of FluxAttnProcessor2_0, with the statistics read from its queries and keys. Double-stream blocks
        pass encoder_hidden_states and get (image_out, text_out) back; single-stream blocks get one tensor."""
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            # Double-stream block: project the text stream separately and put it in front of the image stream.
            eq = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            query = torch.cat([eq, query], dim=2)
            key = torch.cat([ek, key], dim=2)
            value = torch.cat([ev, value], dim=2)
            n_txt = encoder_hidden_states.shape[1]
            self.parent.n_txt = n_txt        # remembered for the single-stream blocks, which are not told the text length
        else:
            n_txt = self.parent.n_txt        # single-stream blocks see the concatenated sequence, text first
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # The block's output: the fused kernel, as in the model's own processor.
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        # The statistics: image-query rows only, and only at the (block, step) pairs the collector wants.
        if self.parent.active and n_txt is not None and self.parent.wants(self.layer_name):
            try:
                self._log_stats(query[:, :, n_txt:], key, n_txt)
            except Exception as e:   # never break generation for a statistic
                self.parent.errors += 1
                if self.parent.errors <= 3:
                    print(f"[ae_hook_FLUX] stats failed at {self.layer_name}: {type(e).__name__}: {e}", flush=True)

        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_out, out = out[:, :encoder_hidden_states.shape[1]], out[:, encoder_hidden_states.shape[1]:]
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            encoder_out = attn.to_add_out(encoder_out)
            return out, encoder_out
        return out


class AttnEntropyCollector:
    """Same interface as ae_hook_SD35.AttnEntropyCollector (new_image / flush_and_get_layer_means), plus flush_vecs and
    values_per_sample.

    Arguments and attributes:
        extra_orders  also log the pooled order (ae_avgfirst_).
        wanted        {(layer_name, step)}: read only these (block, step) pairs; None reads every hooked block at every step.
        step          1-based index of the denoising step being computed; the caller sets it before each step.
        n_txt         text length, learned from the double-stream blocks. If only single-stream blocks are hooked the caller
                      must set it (512, the T5 length of FluxPipeline's default).
        active        set to False to switch all statistics off without removing the processors.
        errors        number of failed statistics calls (the first three are printed).
        device        stored but not used.
    """
    def __init__(self, device: str = "cuda", extra_orders: bool = True, wanted=None):
        self.device = device
        self.n_txt: Optional[int] = None
        self.active = True
        self.errors = 0
        # Accumulators, {layer_name: [value per call]}; `rows` is keyed by (layer_name, step) and holds GPU tensors.
        self.entropies: Dict[str, List[float]] = {}
        self.img_ent: Dict[str, List[float]] = {}
        self.txtmass: Dict[str, List[float]] = {}
        self.vecs: Dict[str, List[np.ndarray]] = {}
        self.avgfirst: Dict[str, List[float]] = {}
        self.rows: Dict[tuple, torch.Tensor] = {}
        self.extra_orders = extra_orders
        # Gated mode: `wanted` = {(layer_name, step)}; the caller sets `step` before each denoising step
        # and only the listed (block, step) pairs are read; None = read every hooked block at every active step.
        self.wanted = set(wanted) if wanted is not None else None
        self.step = 1

    def wants(self, layer_name: str) -> bool:
        """Whether this layer is to be read at the current step."""
        return self.wanted is None or (layer_name, self.step) in self.wanted

    def get_processor(self, layer_name: str):
        """The processor for a double- or single-stream block's `attn`."""
        return _FluxAEAttnProcessor(self, layer_name)

    def _push_rows(self, layer_name, t):
        self.rows[(layer_name, self.step)] = t                     # per-sample reading at this (block, step), no sync

    def values_per_sample(self):
        """{(layer_name, step): np.ndarray (B,)}: the reading of each image of the batch, for every pair read since
        new_image(). The tensors are moved to the host only here."""
        return {k: v.detach().float().cpu().numpy().reshape(-1) for k, v in self.rows.items()}

    def _push_avgfirst(self, layer_name, v):
        if math.isfinite(v):
            self.avgfirst.setdefault(layer_name, []).append(float(v))

    def _push(self, layer_name, v):
        if math.isfinite(v):
            self.entropies.setdefault(layer_name, []).append(float(v))

    def _push_img(self, layer_name, v):
        if math.isfinite(v):
            self.img_ent.setdefault(layer_name, []).append(float(v))

    def _push_txtmass(self, layer_name, v):
        if math.isfinite(v):
            self.txtmass.setdefault(layer_name, []).append(float(v))

    def _push_vec(self, layer_name, vec):
        self.vecs.setdefault(layer_name, []).append(vec)

    def new_image(self, key: str, n_active_tokens=None):
        """Resets all state before an image (or a batch). The arguments are accepted for interface compatibility only."""
        self.entropies, self.img_ent, self.txtmass, self.vecs, self.avgfirst, self.rows = {}, {}, {}, {}, {}, {}
        self.step = 1

    def flush_and_get_layer_means(self) -> dict:
        """Returns {"<prefix>_<block>": float} for the step just computed (prefixes ae_mean, ae_img, txtmass, ae_avgfirst)
        and clears those accumulators. The vectors and the per-sample readings are kept until flush_vecs() / new_image()."""
        out = {}
        for prefix, d in (("ae_mean", self.entropies), ("ae_img", self.img_ent), ("txtmass", self.txtmass), ("ae_avgfirst", self.avgfirst)):
            for layer_name, vals in d.items():
                if vals:
                    out[f"{prefix}_{layer_name.replace('.', '_')}"] = float(np.nanmean(np.array(vals, dtype=np.float32)))
        self.entropies, self.img_ent, self.txtmass, self.avgfirst = {}, {}, {}, {}
        return out

    def flush_vecs(self) -> Dict[str, np.ndarray]:
        """Returns {layer_name: (N_txt,) text-key mass vector}, averaged over the layer's calls since the last flush."""
        out = {k: np.mean(np.stack(v), axis=0) for k, v in self.vecs.items() if v}
        self.vecs = {}
        return out
