#!/usr/bin/env python
"""
fast_hooks.py - the deployed reader: the attention-entropy reading of Sec. 3.1 at the chosen (block, step) pairs only.

The logging hooks (ae_hook.py and its siblings; the paper's "campaign hook") read every block at every logged step and
several diagnostics. A calibrated LiveSeed form needs a few readings before the exit step, so here the model's own attention
processor produces the block's output, untouched, and the entropy is a side computation that runs only at the wanted
(block, step) pairs. The attention rows are formed in chunks of queries, no attention matrix is kept, and nothing is copied
to the host until the caller asks for the values (App. D).

The quantities are those of the logging hooks; ae_hook.py is the reference for the definitions. Every entropy is divided by
ln of the number of keys, and under classifier-free guidance (CFG), where diffusers stacks the batch as
[unconditional, conditional], a sample's two branches are averaged.
  FastSA          U-Net self-attention, pooled order (sa_mean_): the entropy of the attention row averaged over heads,
                  queries and both CFG branches, / ln N.
  FastSAEntFirst  U-Net self-attention, per-row order of Eq. 2 (sa_entfirst_): the mean over heads, queries and both CFG
                  branches of each row's entropy, / ln N.
  FastJA          SD3.5 joint attention (ae_mean_): the mean over heads, image queries and both CFG branches of each
                  image-query row's entropy over the joint (text + image) keys, / ln S.
  FastHY          Hunyuan-DiT self-attention (ae_mean_): as FastSAEntFirst, on the queries and keys that model attends with.
FLUX is not handled here: ae_hook_FLUX.py has the same gate (its `wanted` argument) and fused_hooks.py reads it faster.
Unlike the logging hooks, readings are kept per sample, so a batch of seeds can be ranked.

Usage:  sink = Sink(wanted={(layer_name, step), ...});  install(pipe, family, sink);  sink.step = 1 before each image and
        sink.step = i + 2 in callback_on_step_end (i is the 0-based index of the step just finished, and `step` is the
        1-based index of the step about to be computed);  sink.values() -> {(layer_name, step): float}
"""
import math, torch


class Sink:
    """Holds the gate and the readings. wanted = {(processor name, step)}.

    Readings are kept PER SAMPLE: with a batch of n prompts-or-seeds under CFG the processor sees 2n rows
    [uncond(n), cond(n)] x heads, and each sample's reading pools its own heads and its own two branches, exactly the
    logging hook's quantity for that sample alone. values() returns the first sample's reading (the single-image case);
    values_per_sample() the array over samples, for the live batch. cfg=False treats every batch row as a sample.
    """
    def __init__(self, wanted, cfg=True):
        self.wanted = set(wanted)
        self.layers = {n for n, _ in self.wanted}       # the layers install() must hook
        self.step = 1                                   # 1-based step being computed; the caller keeps it up to date
        self.buf = {}
        self.cfg = cfg

    def want(self, name):
        return (name, self.step) in self.wanted

    def push(self, name, t):
        self.buf[(name, self.step)] = t          # GPU tensor (n_samples,); no sync here

    def values(self):
        return {k: float(v.reshape(-1)[0]) for k, v in self.buf.items()}

    def values_per_sample(self):
        return {k: v.detach().float().cpu().numpy().reshape(-1) for k, v in self.buf.items()}

    def reset(self):
        self.buf = {}
        self.step = 1


# In every class below, `fused` is the processor being replaced: the model's own, which uses the fused attention kernel.
# It is called first and its result is returned unchanged, so the hook cannot alter the image.

class FastSA:
    """U-Net self-attention, pooled order (the logging hook's sa_mean_), per sample: the attention rows are summed over
    queries, heads and the sample's two CFG branches, normalised to one distribution over the N keys, and its entropy is
    divided by ln N. Rows are formed `chunk` queries at a time in the model's dtype and accumulated in fp32."""
    def __init__(self, sink, name, fused, chunk=1024):
        self.sink, self.name, self.fused, self.chunk = sink, name, fused, chunk

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if self.sink.want(self.name):
            with torch.no_grad():
                B = hidden_states.shape[0]
                q = attn.head_to_batch_dim(attn.to_q(hidden_states))          # (B*H, N, D), batch-major
                k = attn.head_to_batch_dim(attn.to_k(hidden_states))
                BH, N, _ = q.shape
                H = BH // B
                acc = torch.zeros(BH, N, device=q.device, dtype=torch.float32)
                kt = k.transpose(1, 2)
                for s in range(0, N, self.chunk):
                    p = torch.baddbmm(
                        torch.empty(BH, min(self.chunk, N - s), N, dtype=q.dtype, device=q.device),
                        q[:, s:s + self.chunk], kt, beta=0, alpha=attn.scale,
                    ).softmax(-1)
                    acc += p.float().sum(1)                                   # (B*H, N): the row summed over this chunk's queries
                acc = acc.view(B, H, N).sum(1)                                # (B, N): pooled over heads
                n = B // 2 if (self.sink.cfg and B % 2 == 0 and B >= 2) else B
                acc = acc[:n] + acc[n:] if n != B else acc                    # (n, N): the sample's two CFG branches pooled
                a = (acc / acc.sum(-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8)
                self.sink.push(self.name, -(a * a.log()).sum(-1) / math.log(max(N, 2)))   # (n,)
        return out


class FastSAEntFirst:
    """The U-Net self-attention reading in the per-row order of Eq. 2 (sa_entfirst_, the U-Net reading under ABSS's
    protocol, Table 1): the mean over heads, queries and both CFG branches of the per-row entropy, / ln N; chunked over
    queries like FastSA (the unfused reference for fused_hooks.FusedSA)."""
    def __init__(self, sink, name, fused, chunk=1024):
        self.sink, self.name, self.fused, self.chunk = sink, name, fused, chunk

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if self.sink.want(self.name):
            with torch.no_grad():
                B = hidden_states.shape[0]
                q = attn.head_to_batch_dim(attn.to_q(hidden_states))          # (B*H, N, D), batch-major
                k = attn.head_to_batch_dim(attn.to_k(hidden_states))
                BH, N, _ = q.shape
                H = BH // B
                tot = torch.zeros(BH, device=q.device, dtype=torch.float32)
                kt = k.transpose(1, 2)
                for s in range(0, N, self.chunk):
                    p = torch.baddbmm(
                        torch.empty(BH, min(self.chunk, N - s), N, dtype=q.dtype, device=q.device),
                        q[:, s:s + self.chunk], kt, beta=0, alpha=attn.scale,
                    ).softmax(-1).float().clamp_min(1e-8)
                    tot += -(p * p.log()).sum((1, 2))                           # (B*H,): per-row entropies summed over this chunk's queries
                tot = tot.view(B, H).sum(1)                                      # (B,): pooled over heads
                n = B // 2 if (self.sink.cfg and B % 2 == 0 and B >= 2) else B
                # Sum -> mean over the sample's rows: H heads x N queries, times two under CFG.
                tot = (tot[:n] + tot[n:]) / (2 * H * N) if n != B else tot / (H * N)
                self.sink.push(self.name, tot / math.log(max(N, 2)))            # (n,)
        return out


class FastJA:
    """SD3.5 joint attention (the logging hook's ae_mean_, ae_hook_SD35.py), per sample: the mean over heads, image
    queries and both CFG branches of each image-query row's entropy over the joint keys [text, image], / ln S."""
    def __init__(self, sink, name, fused, chunk=512):
        self.sink, self.name, self.fused, self.chunk = sink, name, fused, chunk

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if self.sink.want(self.name) and encoder_hidden_states is not None:
            with torch.no_grad():
                B, N_img, _ = hidden_states.shape
                H = attn.heads
                Dh = attn.to_q.out_features // H
                # Image queries, image keys and text keys with the model's query/key normalisation. The text queries and
                # the values are not needed: only the image-query rows are read.
                iq = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (H, Dh)))
                ik = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (H, Dh)))
                tk = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (H, Dh)))
                k = torch.cat([tk, ik], 1).permute(0, 2, 1, 3).reshape(B * H, -1, Dh)      # (B*H, S, Dh), text first
                q = iq.permute(0, 2, 1, 3).reshape(B * H, N_img, Dh)
                S = k.shape[1]
                kt = k.transpose(1, 2)
                tot = torch.zeros(B * H, device=q.device, dtype=torch.float32)
                for s in range(0, N_img, self.chunk):
                    p = torch.baddbmm(
                        torch.empty(B * H, min(self.chunk, N_img - s), S, dtype=q.dtype, device=q.device),
                        q[:, s:s + self.chunk], kt, beta=0, alpha=Dh ** -0.5,
                    ).softmax(-1).float().clamp_min(1e-8)
                    tot += -(p * p.log()).sum((1, 2))                           # (B*H,): per-row entropies summed over this chunk's queries
                tot = tot.view(B, H).sum(1)                                      # (B,): pooled over heads
                n = B // 2 if (self.sink.cfg and B % 2 == 0 and B >= 2) else B
                tot = (tot[:n] + tot[n:]) / (2 * H * N_img) if n != B else tot / (H * N_img)
                self.sink.push(self.name, tot / math.log(max(S, 2)))            # (n,)
        return out


def hy_qk(attn, hidden_states, image_rotary_emb):
    """Hunyuan-DiT attn1: returns the queries and keys HunyuanAttnProcessor2_0 attends with, each (B, H, S, D): query/key
    normalisation, then the rotary position embedding (RoPE) on both (self-attention)."""
    from diffusers.models.embeddings import apply_rotary_emb
    hs = hidden_states
    if hs.ndim == 4:
        hs = hs.flatten(2).transpose(1, 2)
    if attn.group_norm is not None:
        hs = attn.group_norm(hs.transpose(1, 2)).transpose(1, 2)
    B = hs.shape[0]
    q = attn.to_q(hs)
    k = attn.to_k(hs)
    D = k.shape[-1] // attn.heads
    q = q.view(B, -1, attn.heads, D).transpose(1, 2)
    k = k.view(B, -1, attn.heads, D).transpose(1, 2)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    if image_rotary_emb is not None:
        q = apply_rotary_emb(q, image_rotary_emb)
        k = apply_rotary_emb(k, image_rotary_emb)
    return q, k


class FastHY:
    """Hunyuan-DiT self-attention (attn1): the logging hook's ae_mean_ quantity (ae_hook_HUNYUAN._self_stats) per sample:
    the mean over heads, image queries and both CFG branches of the normalised row entropy, chunked over queries. The
    scores are computed in fp32 because in fp16 they overflow on some blocks (4 of the 40). The model's own processor makes
    the output; the wrapper declares temb and image_rotary_emb because diffusers' Attention.forward drops any keyword the
    processor's signature does not name."""
    def __init__(self, sink, name, fused, chunk=256):
        self.sink, self.name, self.fused, self.chunk = sink, name, fused, chunk

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, image_rotary_emb=None):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, temb, image_rotary_emb)
        # Self-attention only (no encoder states), and only unmasked calls.
        if self.sink.want(self.name) and encoder_hidden_states is None and attention_mask is None:
            with torch.no_grad():
                q, k = hy_qk(attn, hidden_states, image_rotary_emb)
                B, H, S, D = q.shape
                kT = k.transpose(-1, -2).float()
                scale = D ** -0.5
                ent = torch.zeros(B, device=q.device, dtype=torch.float32)
                for r0 in range(0, S, self.chunk):
                    p = torch.softmax(torch.matmul(q[:, :, r0:r0 + self.chunk].float(), kT) * scale, dim=-1)
                    ent += (-(p * p.clamp_min(1e-12).log()).sum(-1)).sum((1, 2))
                per = ent / (H * S * math.log(max(S, 2)))                       # (B,): normalised mean row entropy per batch row
                n = B // 2 if (self.sink.cfg and B % 2 == 0 and B >= 2) else B
                self.sink.push(self.name, (per[:n] + per[n:]) / 2 if n != B else per)   # (n,): the sample's two branches pooled
        return out


def install(pipe, family, sink, order="pooled"):
    """Replaces the processor of every layer the sink ever reads; all other layers keep their own processor. Returns the
    number of layers hooked.

    family: "sd3" (FastJA on pipe.transformer), "hunyuan" (FastHY on the attn1 layers of pipe.transformer), anything else
    is a U-Net (pipe.unet), where order selects "pooled" (default, FastSA, sa_mean_) or "entfirst" (FastSAEntFirst,
    sa_entfirst_). The names in sink.wanted must be diffusers processor names (keys of attn_processors).
    """
    module = pipe.transformer if family in ("sd3", "hunyuan") else pipe.unet
    procs = dict(module.attn_processors)
    n = 0
    for name, proc in procs.items():
        if name in sink.layers:
            cls = FastJA if family == "sd3" else (FastHY if family == "hunyuan" else (FastSAEntFirst if order == "entfirst" else FastSA))
            procs[name] = cls(sink, name, proc)
            n += 1
    module.set_attn_processor(procs)
    return n
