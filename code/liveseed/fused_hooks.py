#!/usr/bin/env python
"""
fused_hooks.py - the attention-entropy reading from one memory-efficient attention call, without the attention matrix
(Sec. 3.1, "Efficient implementation"; App. D).

Identity: for a row with scores s_j = scale * q.k_j and p = softmax(s),
    H(p) = logsumexp(s) - sum_j p_j s_j,   sum_j p_j s_j = scale * q . (sum_j p_j k_j),
and sum_j p_j k_j is the output of an ordinary attention call with the KEYS as values. torch's memory-efficient kernel returns
the row log-sum-exp beside the output, so one call (V = K) gives the exact reading: no N x S matrix, no fp32 copy, no
elementwise log. On random tensors it agrees with the explicit computation (max difference 4e-6 of the normalised reading in
fp16, 4e-5 in bf16) and is 15x faster on SD3.5-sized blocks, 7x on FLUX-sized. The model's own output is untouched.

The readings are those of the logging hooks; ae_hook.py is the reference for the definitions. Every entropy is divided by ln
of the number of keys, readings are kept per sample, and under classifier-free guidance (CFG), where diffusers stacks the
batch as [unconditional, conditional], a sample's two branches are averaged.
  FusedSA    U-Net self-attention (attn1): the per-row order of Eq. 2 (sa_entfirst_) or the pooled order (sa_mean_).
  FusedJA    SD3.5 joint attention: image-query rows over the joint [text, image] keys, per-row order (ae_mean_).
  FusedFlux  FLUX double- and single-stream blocks: the same quantity; FLUX.1-dev has a single branch.
  FusedHY    Hunyuan-DiT self-attention (attn1), per-row order (ae_mean_).

Interface as fast_hooks.py: sink = Sink(wanted={(layer_name, step), ...}); install(pipe, family, sink) for family "sd3",
"flux", "unet" or "hunyuan"; the sink's push / values_per_sample / step / reset are unchanged. liveseed_live.install_gated
uses these hooks by default; LIVESEED_FUSED=0 (transformers) or LIVESEED_FUSED_UNET=0 (U-Nets) selects the unfused ones.
"""
import math, torch
from fast_hooks import Sink   # noqa: F401  (re-exported for callers)


def row_entropy_sum(q, k, scale):
    """q (B,H,N,D), k (B,H,S,D) -> (B,) sum over heads and queries of the row entropies (nats)."""
    q = q.contiguous()
    k = k.contiguous()
    # Arguments: (query, key, value = key, no bias, return the log-sum-exp, no dropout, not causal).
    out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, k, None, True, 0.0, False, scale=scale)[:2]
    es = scale * torch.einsum("bhnd,bhnd->bhn", q.float(), out.float())        # (B,H,N): expected score of each row
    return (lse[..., :q.shape[2]].float() - es).sum((1, 2))                    # lse is cut back to the N queries


def pooled_row_log(q, k, scale):
    """q (B,H,N,D), k (B,H,S,D) -> (B,H,S) fp32: log of the attention COLUMN sums, sum_q p_qj, without the N x S matrix.
    This is what the pooled order needs: the row averaged over queries is the column sums divided by N.

    Two memory-efficient attention calls. The first (V = K, as row_entropy_sum) returns each row's log-sum-exp lse_q. The
    second takes the keys as queries, the queries as keys and the values w_q = exp(m - lse_q) (m = the head's smallest
    lse, so 0 < w <= 1), and returns out'_j = sum_q exp(s_qj) w_q / sum_q exp(s_qj) and its own
    lse'_j = log sum_q exp(s_qj); hence sum_q p_qj = sum_q exp(s_qj - lse_q) = out'_j exp(lse'_j - m).

    The second call runs in bf16, which has fp32's range, so w cannot underflow as it can in fp16. On random tensors the
    normalised reading is within 5e-6 of the fp32 call and 3x faster than the chunked fast_hooks.FastSA (42 against 127 ms
    on an RTX A4000 for 20 x 8 heads x 4096 tokens); fp32 is no faster.
    """
    q = q.contiguous()
    k = k.contiguous()
    B, H, N, D = q.shape
    lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, k, None, True, 0.0, False, scale=scale)[1][..., :N].float()
    m = lse.amin(-1, keepdim=True)                                                   # (B,H,1)
    w = torch.exp(m - lse).unsqueeze(-1).expand(B, H, N, 8).contiguous()             # (B,H,N,8): the kernel wants a vector value
    o2, lse2 = torch.ops.aten._scaled_dot_product_efficient_attention(
        k.to(torch.bfloat16), q.to(torch.bfloat16), w.to(torch.bfloat16), None, True, 0.0, False, scale=scale)[:2]
    S = k.shape[2]
    return o2[..., 0].float().clamp_min(1e-30).log() + lse2[..., :S].float() - m      # (B,H,S)


# In the classes that take it, `fused` is the processor being replaced: the model's own, which uses the fused attention
# kernel. It is called first and its result is returned unchanged, so the hook cannot alter the image.

class FusedSA:
    """U-Net self-attention (attn1), both readings of the logging hook (ae_hook.py) per sample, both CFG branches pooled:
      order "entfirst": the mean over heads, queries and branches of the per-row entropy, / ln N
                        (sa_entfirst_, Eq. 2; the U-Net reading under ABSS's protocol, Table 1)
      order "pooled"  : the entropy of the row averaged over heads, queries and branches, / ln N
                        (sa_mean_, the U-Net reading of the paper's 30-step pools, App. C)
    The model's own processor produces the block's output, untouched."""
    def __init__(self, sink, name, fused, order="entfirst"):
        self.sink, self.name, self.fused, self.order = sink, name, fused, order

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if self.sink.want(self.name) and encoder_hidden_states is None:      # self-attention calls only
            with torch.no_grad():
                hs = hidden_states
                if hs.ndim == 4:
                    hs = hs.flatten(2).transpose(1, 2)
                B, N, _ = hs.shape
                H = attn.heads
                q = attn.to_q(hs)
                k = attn.to_k(hs)
                D = q.shape[-1] // H
                q = q.view(B, N, H, D).transpose(1, 2)                                        # (B, H, N, D)
                k = k.view(B, N, H, D).transpose(1, 2)
                if getattr(attn, "norm_q", None) is not None:
                    q = attn.norm_q(q)
                if getattr(attn, "norm_k", None) is not None:
                    k = attn.norm_k(k)
                n = B // 2 if (self.sink.cfg and B % 2 == 0 and B >= 2) else B                # samples; B = 2n under CFG
                if self.order == "entfirst":
                    tot = row_entropy_sum(q, k, attn.scale)                                   # (B,)
                    # Sum -> mean over the sample's rows: H heads x N queries, times two under CFG.
                    tot = (tot[:n] + tot[n:]) / (2 * H * N) if n != B else tot / (H * N)
                    self.sink.push(self.name, tot / math.log(max(N, 2)))
                else:
                    lc = pooled_row_log(q, k, attn.scale)                                     # (B,H,N)
                    lc = torch.cat([lc[:n], lc[n:]], 1) if n != B else lc                     # (n, 2H or H, N): the sample's heads and branches
                    la = torch.logsumexp(lc, 1)                                               # (n, N): log of the pooled row (unnormalised)
                    a = torch.softmax(la, -1).clamp_min(1e-8)                                 # normalised over the keys
                    self.sink.push(self.name, -(a * a.log()).sum(-1) / math.log(max(N, 2)))
        return out


class FusedJA:
    """SD3.5 joint attention: image queries over [text keys, image keys] (the fast_hooks.FastJA quantity)."""
    def __init__(self, sink, name, fused):
        self.sink, self.name, self.fused = sink, name, fused

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if self.sink.want(self.name) and encoder_hidden_states is not None:
            with torch.no_grad():
                B, N_img, _ = hidden_states.shape
                H = attn.heads
                Dh = attn.to_q.out_features // H
                # Image queries, image keys and text keys with the model's query/key normalisation.
                iq = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (H, Dh)))
                ik = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (H, Dh)))
                tk = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (H, Dh)))
                q = iq.transpose(1, 2)                                                    # (B, H, N_img, Dh)
                k = torch.cat([tk, ik], 1).transpose(1, 2)                                # (B, H, S, Dh), text first
                S = k.shape[2]
                tot = row_entropy_sum(q, k, Dh ** -0.5)                                   # (B,)
                n = B // 2 if (self.sink.cfg and B % 2 == 0 and B >= 2) else B
                tot = (tot[:n] + tot[n:]) / (2 * H * N_img) if n != B else tot / (H * N_img)
                self.sink.push(self.name, tot / math.log(max(S, 2)))
        return out


class FusedFlux:
    """FLUX double- and single-stream blocks: image queries over the joint sequence (text first), after the rotary position
    embedding (the ae_hook_FLUX ae_mean quantity). This class does not wrap the model's processor: it repeats its steps
    and produces the output with the same scaled_dot_product_attention call, as ae_hook_FLUX.py does. n_txt_ref is a
    one-element list shared by all blocks: a double-stream block writes the text length into it and the single-stream
    blocks, which see only the concatenated sequence, read it."""
    def __init__(self, sink, name, n_txt_ref):
        self.sink, self.name, self.n_txt_ref = sink, name, n_txt_ref

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        import torch.nn.functional as F
        bs = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        hd = key.shape[-1] // attn.heads
        query = query.view(bs, -1, attn.heads, hd).transpose(1, 2)
        key = key.view(bs, -1, attn.heads, hd).transpose(1, 2)
        value = value.view(bs, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if encoder_hidden_states is not None:
            # Double-stream block: project the text stream separately and put it in front of the image stream.
            eq = attn.add_q_proj(encoder_hidden_states).view(bs, -1, attn.heads, hd).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(bs, -1, attn.heads, hd).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(bs, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            query = torch.cat([eq, query], 2)
            key = torch.cat([ek, key], 2)
            value = torch.cat([ev, value], 2)
            n_txt = encoder_hidden_states.shape[1]
            self.n_txt_ref[0] = n_txt
        else:
            n_txt = self.n_txt_ref[0]
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # The block's output.
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        # The reading: image-query rows over all keys.
        if self.sink.want(self.name) and n_txt is not None:
            with torch.no_grad():
                q_img = query[:, :, n_txt:]
                S = key.shape[2]
                N_img = q_img.shape[2]
                tot = row_entropy_sum(q_img, key, hd ** -0.5) / (attn.heads * N_img)    # FLUX is guidance-distilled: one branch
                self.sink.push(self.name, tot / math.log(max(S, 2)))

        out = out.transpose(1, 2).reshape(bs, -1, attn.heads * hd).to(query.dtype)
        if encoder_hidden_states is not None:
            enc, out = out[:, :encoder_hidden_states.shape[1]], out[:, encoder_hidden_states.shape[1]:]
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            enc = attn.to_add_out(enc)
            return out, enc
        return out


class FusedHY:
    """Hunyuan-DiT self-attention (attn1): fast_hooks.FastHY's quantity (the logging hook's ae_mean_) from one
    memory-efficient attention call, on the queries and keys the model attends with (query/key normalisation, rotary
    position embedding). precision:
      "center": keys centred per head and sample before the call. Subtracting the mean key shifts every score of a row by the same
                constant, so the softmax and the entropy are unchanged, but the large common component of the scores (the raw
                q.k overflows fp16 on 4 of the 40 blocks) no longer has to cancel between lse and the expected score.
      "fp16"  : the call as on the other families.
      "fp32"  : q and k in fp32 (the exact reference for the fused identity; slower)."""
    def __init__(self, sink, name, fused, precision="center"):
        self.sink, self.name, self.fused, self.precision = sink, name, fused, precision

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, image_rotary_emb=None):
        out = self.fused(attn, hidden_states, encoder_hidden_states, attention_mask, temb, image_rotary_emb)
        # Self-attention only (no encoder states), and only unmasked calls.
        if self.sink.want(self.name) and encoder_hidden_states is None and attention_mask is None:
            with torch.no_grad():
                from fast_hooks import hy_qk
                q, k = hy_qk(attn, hidden_states, image_rotary_emb)
                B, H, S, D = q.shape
                if self.precision == "center":
                    k = k - k.mean(2, keepdim=True)
                elif self.precision == "fp32":
                    q, k = q.float(), k.float()
                tot = row_entropy_sum(q, k, D ** -0.5)                                       # (B,)
                n = B // 2 if (self.sink.cfg and B % 2 == 0 and B >= 2) else B
                tot = (tot[:n] + tot[n:]) / (2 * H * S) if n != B else tot / (H * S)
                self.sink.push(self.name, tot / math.log(max(S, 2)))
        return out


def install(pipe, family, sink, n_txt=512, order="entfirst", precision=None):
    """Replaces the processor of every block the sink reads and returns the number of blocks hooked.

    family     "sd3" (FusedJA), "flux" (FusedFlux), "unet" (FusedSA, with the reading `order`) or "hunyuan" (FusedHY).
    n_txt      FLUX only. The single-stream blocks learn the text length from a double-stream block, or from n_txt
               (FLUX's T5 sequence is padded to 512) when no double-stream block is hooked.
    precision  Hunyuan-DiT only: see FusedHY; None reads the environment variable LIVESEED_HY_PRECISION (default "center").
    """
    if family == "hunyuan":
        import os
        prec = precision or os.environ.get("LIVESEED_HY_PRECISION", "center")
        module = pipe.transformer
        procs = dict(module.attn_processors)
        n = 0
        for name, proc in procs.items():
            if name in sink.layers:
                procs[name] = FusedHY(sink, name, proc, prec)
                n += 1
        module.set_attn_processor(procs)
        return n
    if family == "unet":
        module = pipe.unet
        procs = dict(module.attn_processors)
        n = 0
        for name, proc in procs.items():
            if name in sink.layers:
                procs[name] = FusedSA(sink, name, proc, order)
                n += 1
        module.set_attn_processor(procs)
        return n
    # SD3.5 and FLUX
    module = pipe.transformer
    procs = dict(module.attn_processors)
    n = 0
    ref = [n_txt]          # the text length shared by all FusedFlux blocks
    for name, proc in procs.items():
        if name in sink.layers:
            procs[name] = FusedJA(sink, name, proc) if family == "sd3" else FusedFlux(sink, name, ref)
            n += 1
    module.set_attn_processor(procs)
    return n
