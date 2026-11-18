"""
ae_hook_HUNYUAN.py - attention-entropy hooks for Hunyuan-DiT v1.2 (diffusers HunyuanDiT2DModel).

This file computes the reading of Sec. 3.1 (Eq. 2) on Hunyuan-DiT's self-attention. The design (one collector, one processor
per layer, `col.step`, a flush per denoising step, entropies divided by ln of the number of keys, statistics that never
interrupt generation) is that of ae_hook.py, the reference; the collector's interface is that of ae_hook_FLUX.py. Only the
differences are described here.

What is hooked
  Hunyuan-DiT is a diffusion transformer with U-Net-style blocks: each of its 40 `blocks.N` has a self-attention `attn1` over
  the image tokens (rotary position embedding, RoPE, on queries and keys; query/key normalisation) and a cross-attention
  `attn2` from the image tokens to the text sequence, which is the 77 BERT tokens followed by the 256 mT5 tokens (the block
  passes no attention mask, so padding is attended). Both use HunyuanAttnProcessor2_0. This module replaces it with a
  processor that reproduces it exactly for the block's OUTPUT and, in addition, materialises the attention probabilities in
  row chunks (never the whole matrix at once) to log the statistics.

Readings returned by flush_and_get_layer_means() (<block> is the processor name with dots replaced by underscores, e.g.
ae_mean_blocks_0_attn1_processor)
  attn1 (self-attention, keys = the S image tokens):
    ae_mean_<block>      The LiveSeed reading, in the PER-ROW order of Eq. 2: the entropy of every row, then the mean over
                         heads, queries and both branches of classifier-free guidance (CFG), / ln S.
    ae_cond_<block>      The same, conditional branch only.
    ae_avgfirst_<block>  The POOLED order: the row averaged over heads, queries and branches first, then one entropy, / ln S
                         (only with extra_orders=True; App. G ablates the two orders).
  attn2 (cross-attention, keys = the 333 text tokens):
    ca_mean_<block>      Mean normalised row entropy over the text keys, conditional branch.
Other outputs
  flush_vecs()           {block: (333,) vector} for the attn2 layers: the mass on each text key (conditional branch, mean over
                         heads and image queries). The ABSS baseline's statistic is its mean over the core tokens' positions
                         (T5 positions + 77, as in the ABSS authors' Hunyuan-DiT code).
  values_per_sample()    {(block, step): (n,) array} for the attn1 layers: ae_mean separately for each image of the batch
                         (its two CFG branches averaged), which the live algorithm needs to rank the seeds of one batch.

The diffusers pipeline stacks the batch as [unconditional, conditional] under CFG.

Install on a diffusers pipeline (the processor tells attn1 from attn2 by whether it is given encoder_hidden_states):
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


class _HunyuanAEAttnProcessor(nn.Module):
    """HunyuanAttnProcessor2_0 with the statistics added. `chunk` is the number of queries whose rows are materialised at
    a time."""
    def __init__(self, parent: "AttnEntropyCollector", layer_name: str, chunk: int = 512):
        super().__init__()
        self.parent, self.layer_name, self.chunk = parent, layer_name, chunk

    @torch.no_grad()
    def _self_stats(self, q: torch.Tensor, k: torch.Tensor):
        """Self-attention statistics. q, k: (B, H, S, D) after norm and RoPE; B = 2 x images under CFG ([uncond, cond])."""
        B, H, S, D = q.shape
        cfg = self.parent.cfg and B % 2 == 0
        kT = k.transpose(-1, -2)
        scale = D ** -0.5
        ent_b = torch.zeros(B, device=q.device, dtype=torch.float32)               # per batch row: sum of row entropies
        row_acc = torch.zeros(S, device=q.device, dtype=torch.float32)             # pooled order: the row summed over everything
        for r0 in range(0, S, self.chunk):
            # (B, H, c, S). The scores are computed in fp32: in fp16 they overflow on some blocks (4 of the 40).
            p = torch.softmax(torch.matmul(q[:, :, r0:r0 + self.chunk].float(), kT.float()) * scale, dim=-1)
            ent_b += (-(p * p.clamp_min(1e-12).log()).sum(-1)).sum((1, 2))
            if self.parent.extra_orders:
                row_acc += p.sum(dim=(0, 1, 2))
        norm = H * S * math.log(max(S, 2))                                           # rows per batch row x ln S
        per = ent_b / norm                                                           # (B,) normalised mean row entropy
        self.parent._push("ae_mean", self.layer_name, float(per.mean().item()))
        if cfg:
            self.parent._push("ae_cond", self.layer_name, float(per[B // 2:].mean().item()))
            self.parent._push_rows(self.layer_name, (per[:B // 2] + per[B // 2:]) / 2)   # per image, both branches pooled
        else:
            self.parent._push_rows(self.layer_name, per)
        if self.parent.extra_orders:
            a = (row_acc / row_acc.sum().clamp_min(1e-12)).clamp_min(1e-12)
            self.parent._push("ae_avgfirst", self.layer_name, float((-(a * a.log()).sum()).item() / math.log(max(S, 2))))

    @torch.no_grad()
    def _cross_stats(self, q: torch.Tensor, k: torch.Tensor):
        """Cross-attention statistics. q: (B, H, N_img, D), k: (B, H, T, D) text keys; statistics on the conditional branch."""
        B = q.shape[0]
        if self.parent.cfg and B % 2 == 0:
            q, k = q[B // 2:], k[B // 2:]
        Bc, H, N, D = q.shape
        T = k.shape[2]
        kT = k.transpose(-1, -2)
        scale = D ** -0.5
        ent = torch.zeros((), device=q.device, dtype=torch.float32)
        vec = torch.zeros(T, device=q.device, dtype=torch.float32)
        for r0 in range(0, N, self.chunk):
            p = torch.softmax(torch.matmul(q[:, :, r0:r0 + self.chunk].float(), kT.float()) * scale, dim=-1)   # (Bc, H, c, T)
            ent += (-(p * p.clamp_min(1e-12).log()).sum(-1)).sum()
            vec += p.sum(dim=(0, 1, 2))
        self.parent._push("ca_mean", self.layer_name, float(ent.item() / (Bc * H * N) / math.log(max(T, 2))))
        self.parent._push_vec(self.layer_name, (vec / (Bc * H * N)).cpu().numpy().astype(np.float32))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, image_rotary_emb=None):
        """The attention of HunyuanAttnProcessor2_0, with the statistics read from its queries and keys before the fused
        kernel produces the output."""
        from diffusers.models.embeddings import apply_rotary_emb
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        is_cross = encoder_hidden_states is not None
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # RoPE on the queries always, and on the keys only in self-attention (text keys carry no image position).
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # The statistics, at the (block, step) pairs the collector wants. They assume unmasked attention, so a call with
        # an attention mask is not read.
        if self.parent.active and self.parent.wants(self.layer_name) and attention_mask is None:
            try:
                (self._cross_stats if is_cross else self._self_stats)(query, key)
            except Exception as e:   # never break generation for a statistic
                self.parent.errors += 1
                if self.parent.errors <= 3:
                    print(f"[ae_hook_HUNYUAN] stats failed at {self.layer_name}: {type(e).__name__}: {e}", flush=True)

        # The block's output: the fused kernel, as in the model's own processor.
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class AttnEntropyCollector:
    """Same interface as ae_hook_FLUX.AttnEntropyCollector (new_image / flush_and_get_layer_means / flush_vecs / values_per_sample).

    Arguments and attributes (those not listed are as in ae_hook_FLUX.py):
        cfg     True if the batch is stacked [unconditional, conditional]; False treats every batch row as an image.
        n_txt   length of the text sequence (77 BERT + 256 mT5 tokens); informational, not read in this file.
        scal    accumulator {prefix: {layer_name: [value per call]}} behind flush_and_get_layer_means().
    """
    def __init__(self, device: str = "cuda", extra_orders: bool = True, wanted=None, cfg: bool = True):
        self.device, self.extra_orders, self.cfg = device, extra_orders, cfg
        self.active = True
        self.errors = 0
        self.step = 1
        self.n_txt = 77 + 256
        self.wanted = set(wanted) if wanted is not None else None
        self.scal: Dict[str, Dict[str, List[float]]] = {}
        self.vecs: Dict[str, List[np.ndarray]] = {}
        self.rows: Dict[tuple, torch.Tensor] = {}

    def wants(self, layer_name: str) -> bool:
        """Whether this layer is to be read at the current step."""
        return self.wanted is None or (layer_name, self.step) in self.wanted

    def get_processor(self, layer_name: str):
        """The processor for an `attn1` or `attn2` layer."""
        return _HunyuanAEAttnProcessor(self, layer_name)

    def _push(self, prefix, layer_name, v):
        if math.isfinite(v):
            self.scal.setdefault(prefix, {}).setdefault(layer_name, []).append(float(v))

    def _push_rows(self, layer_name, t):
        self.rows[(layer_name, self.step)] = t      # (n,) GPU tensor: no host sync until values_per_sample()

    def _push_vec(self, layer_name, vec):
        self.vecs.setdefault(layer_name, []).append(vec)

    def values_per_sample(self):
        """{(layer_name, step): np.ndarray (n,)}: the attn1 reading of each image, for every pair read since new_image()."""
        return {k: v.detach().float().cpu().numpy().reshape(-1) for k, v in self.rows.items()}

    def new_image(self, key: str, n_active_tokens=None):
        """Resets all state before an image (or a batch). The arguments are accepted for interface compatibility only."""
        self.scal, self.vecs, self.rows, self.step = {}, {}, {}, 1

    def flush_and_get_layer_means(self) -> dict:
        """Returns {"<prefix>_<block>": float} for the step just computed and clears the scalar accumulators."""
        out = {}
        for prefix, d in self.scal.items():
            for layer_name, vals in d.items():
                if vals:
                    out[f"{prefix}_{layer_name.replace('.', '_')}"] = float(np.nanmean(np.array(vals, dtype=np.float32)))
        self.scal = {}
        return out

    def flush_vecs(self) -> Dict[str, np.ndarray]:
        """Returns {layer_name: (T,) text-key mass vector}, averaged over the layer's calls since the last flush."""
        out = {k: np.mean(np.stack(v), axis=0) for k, v in self.vecs.items() if v}
        self.vecs = {}
        return out
