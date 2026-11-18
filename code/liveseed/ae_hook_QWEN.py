"""
ae_hook_QWEN.py - attention-entropy hooks for Qwen-Image (diffusers >= 0.35 QwenImageTransformer2DModel).

This file computes the reading of Sec. 3.1 (Eq. 2) on Qwen-Image's joint attention, with the fused computation of Sec. 3.1
and App. D. The design (one collector, one processor per layer, `col.step`, a flush per denoising step, entropies divided by
ln of the number of keys, statistics that never interrupt generation) is that of ae_hook.py, the reference; the collector's
interface is that of ae_hook_FLUX.py. Only the differences are described here.

What is hooked
  Qwen-Image has 60 double-stream blocks (`transformer_blocks.N.attn`, QwenDoubleStreamAttnProcessor2_0): joint attention
  over the concatenated [text, image] sequence (text first), 24 heads of 128, query/key RMS-norm and complex rotary position
  embedding (RoPE) on both streams, no attention mask (the text sequence is the prompt's own length; at batch 1 there is no
  padding). At 1328 px the image stream is 83 x 83 = 6,889 tokens, so a materialised joint map is 24 x 6,889 x ~6,900 per
  block; this module never forms it. It keeps the model's own processor for the block's OUTPUT (images are bit-identical
  with and without the hook) and recomputes the image queries and the joint keys to log the statistics with the exact
  log-sum-exp identity of fused_hooks.py:
      H(p) = logsumexp(s) - sum_j p_j s_j,    sum_j p_j s_j = scale * q . (attention output with the KEYS as values).

Readings returned by flush_and_get_layer_means() (<block> is the layer name with dots replaced by underscores, e.g.
ae_mean_transformer_blocks_0_attn; S = N_txt + N_img keys)
  ae_mean_<block>   The LiveSeed reading, in the PER-ROW order of Eq. 2: the entropy of every image-query row over the joint
                    key set, then the mean over heads and image queries, / ln S, averaged over the two passes of
                    classifier-free guidance (CFG) (the SD3.5 / FLUX quantity).
  ae_cond_<block>   The same, conditional pass only.
  With extra_orders=True, on the conditional pass:
  ae_img_<block>    Normalised entropy (/ ln N_img) over the image keys alone (one more efficient-attention call).
  txtmass_<block>   Attention mass the image queries place on the text keys.
Other outputs
  flush_vecs()         {block: (N_txt,) vector}: how that text mass is shared among the text positions (sums to one;
                       conditional pass, extra_orders=True). It is exp(s_text - lse), cheap because the text is short.
  values_per_sample()  {(block, step): (B,) array}: the conditional pass's reading for each image of the batch.

True CFG runs the conditional and the unconditional pass as two separate transformer calls per step (conditional first:
QwenImagePipeline.__call__, diffusers 0.35.2), so the collector counts transformer calls within a step through a forward
pre-hook (install_pass_counter): pass 0 = conditional, pass 1 = unconditional.

Install on a diffusers pipeline (module by module, as seed_harness.py does; each processor wraps the module's own):
    col = AttnEntropyCollector()
    for name, module in pipe.transformer.named_modules():
        if name.endswith(".attn") and hasattr(module, "set_processor"):
            module.set_processor(col.get_processor(name, module.processor))
    col.install_pass_counter(pipe.transformer)

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


def _lse_and_expect(q: torch.Tensor, k: torch.Tensor, scale: float):
    """q (B,H,N,D), k (B,H,S,D) -> (lse, E_p[s]) each (B,H,N) in fp32, from one memory-efficient attention call with V = K.

    lse is the log-sum-exp of each row's scores and E_p[s] the expected score under the row's attention; the row entropy in
    nats is lse - E_p[s].
    """
    q = q.contiguous()
    k = k.contiguous()
    # Arguments: (query, key, value = key, no bias, return the log-sum-exp, no dropout, not causal).
    out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, k, None, True, 0.0, False, scale=scale)[:2]
    lse = lse[..., :q.shape[2]].float()         # the kernel may pad this axis beyond the N queries
    es = scale * (q.float() * out.float()).sum(-1)
    return lse, es


class _QwenAEAttnProcessor(nn.Module):
    """Wraps the block's own processor (`original`), which produces the output; the statistics are a side computation."""
    def __init__(self, parent: "AttnEntropyCollector", layer_name: str, original):
        super().__init__()
        self.parent, self.layer_name, self.original = parent, layer_name, original

    @torch.no_grad()
    def _stats(self, attn, hidden_states, encoder_hidden_states, image_rotary_emb):
        """Recomputes the image queries and the joint keys the model attends with, and pushes this pass's statistics."""
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

        # Image queries, image keys and text keys, as the model's processor builds them: projection, norm, RoPE.
        # (The text queries and all values are not needed: only image-query rows are read.)
        iq = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        ik = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        tk = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            iq = attn.norm_q(iq)
        if attn.norm_k is not None:
            ik = attn.norm_k(ik)
        if attn.norm_added_k is not None:
            tk = attn.norm_added_k(tk)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            iq = apply_rotary_emb_qwen(iq, img_freqs, use_real=False)
            ik = apply_rotary_emb_qwen(ik, img_freqs, use_real=False)
            tk = apply_rotary_emb_qwen(tk, txt_freqs, use_real=False)
        q = iq.transpose(1, 2)                                                                  # (B, H, N, D)
        k_img = ik.transpose(1, 2)
        k_txt = tk.transpose(1, 2)
        B, H, N, D = q.shape
        T = k_txt.shape[2]
        S = T + N
        scale = D ** -0.5
        k_all = torch.cat([k_txt, k_img], dim=2)

        # The reading: row entropy = lse - E_p[s], mean over heads and image queries, / ln S.
        lse, es = _lse_and_expect(q, k_all, scale)                                               # joint key set
        per = (lse - es).mean((1, 2)) / math.log(max(S, 2))                                       # (B,) normalised mean row entropy
        cond = self.parent.pass_index == 0
        self.parent._push("ae_mean", self.layer_name, float(per.mean().item()))     # pushed on both passes: the flush averages them

        # Optional self-check: the explicit (chunked softmax) entropy of block 0 at step 1, stored beside the fused one.
        if cond and self.parent.debug_explicit and self.step_is_first() and self.layer_name.startswith("transformer_blocks.0."):
            ent = torch.zeros((), device=q.device, dtype=torch.float32)
            kT = k_all.float().transpose(-1, -2)
            for r0 in range(0, N, 256):                                                           # explicit check of the fused identity
                pr = torch.softmax(torch.matmul(q[:, :, r0:r0 + 256].float(), kT) * scale, dim=-1)
                ent += (-(pr * pr.clamp_min(1e-30).log()).sum(-1)).sum()
            self.parent.debug.append(dict(block=self.layer_name, fused=float(per.mean().item()),
                                          explicit=float(ent.item() / (B * H * N) / math.log(max(S, 2)))))

        if cond:
            self.parent._push("ae_cond", self.layer_name, float(per.mean().item()))
            self.parent._push_rows(self.layer_name, per)
            if self.parent.extra_orders:
                lse_i, es_i = _lse_and_expect(q, k_img, scale)                                    # image keys only
                self.parent._push("ae_img", self.layer_name, float(((lse_i - es_i).mean() / math.log(max(N, 2))).item()))
                # p(text key | image query) = exp(score - logsumexp over all keys)                      (B, H, N, T)
                p_txt = torch.exp(torch.einsum("bhnd,bhtd->bhnt", q.float(), k_txt.float()) * scale - lse[..., None])
                self.parent._push("txtmass", self.layer_name, float(p_txt.sum(-1).mean().item()))
                v = p_txt.mean((0, 1, 2))
                self.parent._push_vec(self.layer_name, (v / v.sum().clamp_min(1e-12)).cpu().numpy().astype(np.float32))

    def step_is_first(self):
        return self.parent.step == 1

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, attention_mask=None, image_rotary_emb=None):
        """Runs the model's own processor for the output, then reads the statistics if this (block, step) is wanted."""
        out = self.original(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, encoder_hidden_states_mask=encoder_hidden_states_mask,
                            attention_mask=attention_mask, image_rotary_emb=image_rotary_emb)
        # The statistics assume unmasked attention, so a call with an attention mask is not read.
        if self.parent.active and self.parent.wants(self.layer_name) and attention_mask is None and encoder_hidden_states is not None:
            try:
                self._stats(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
            except Exception as e:   # never break generation for a statistic
                self.parent.errors += 1
                if self.parent.errors <= 3:
                    print(f"[ae_hook_QWEN] stats failed at {self.layer_name}: {type(e).__name__}: {e}", flush=True)
        return out


class AttnEntropyCollector:
    """Same interface as ae_hook_FLUX.AttnEntropyCollector.

    Attributes not found there:
        pass_index      0 during the conditional transformer call of a step, 1 during the unconditional one (kept up to
                        date by install_pass_counter).
        debug_explicit  set to True to store, in `debug`, the explicit entropy of block 0 at step 1 beside the fused one.
        scal            accumulator {prefix: {layer_name: [value per call]}} behind flush_and_get_layer_means().
    Here extra_orders switches ae_img_, txtmass_ and the text vector on or off.
    """
    def __init__(self, device: str = "cuda", extra_orders: bool = True, wanted=None):
        self.device, self.extra_orders = device, extra_orders
        self.active = True
        self.errors = 0
        self.step = 1
        self.n_txt = None
        self.wanted = set(wanted) if wanted is not None else None
        self.pass_index, self._pass_step = 0, None
        self.debug_explicit, self.debug = False, []      # self-check of the fused identity (see _QwenAEAttnProcessor._stats)
        self.scal: Dict[str, Dict[str, List[float]]] = {}
        self.vecs: Dict[str, List[np.ndarray]] = {}
        self.rows: Dict[tuple, torch.Tensor] = {}

    def install_pass_counter(self, transformer):
        """Registers a forward pre-hook that counts the transformer's calls within a denoising step: pass_index 0 =
        conditional, 1 = unconditional (true CFG). It relies on the caller setting self.step. Returns the hook's handle."""
        def pre(module, args, kwargs):
            if self._pass_step != self.step:        # first call of a new step
                self._pass_step, self.pass_index = self.step, 0
            else:
                self.pass_index += 1
        return transformer.register_forward_pre_hook(pre, with_kwargs=True)

    def wants(self, layer_name: str) -> bool:
        """Whether this layer is to be read at the current step."""
        return self.wanted is None or (layer_name, self.step) in self.wanted

    def get_processor(self, layer_name: str, original):
        """The processor for a block's `attn`; `original` is the processor it replaces, which keeps producing the output."""
        return _QwenAEAttnProcessor(self, layer_name, original)

    def _push(self, prefix, layer_name, v):
        if math.isfinite(v):
            self.scal.setdefault(prefix, {}).setdefault(layer_name, []).append(float(v))

    def _push_rows(self, layer_name, t):
        self.rows[(layer_name, self.step)] = t      # (B,) GPU tensor: no host sync until values_per_sample()

    def _push_vec(self, layer_name, vec):
        self.vecs.setdefault(layer_name, []).append(vec)

    def values_per_sample(self):
        """{(layer_name, step): np.ndarray (B,)}: the conditional pass's reading of each image, since new_image()."""
        return {k: v.detach().float().cpu().numpy().reshape(-1) for k, v in self.rows.items()}

    def new_image(self, key: str, n_active_tokens=None):
        """Resets all state before an image (or a batch). The arguments are accepted for interface compatibility only."""
        self.scal, self.vecs, self.rows, self.step, self._pass_step, self.pass_index = {}, {}, {}, 1, None, 0

    def flush_and_get_layer_means(self) -> dict:
        """Returns {"<prefix>_<block>": float} for the step just computed and clears the scalar accumulators. ae_mean has
        one value per CFG pass, so its mean here is the average of the two passes."""
        out = {}
        for prefix, d in self.scal.items():
            for layer_name, vals in d.items():
                if vals:
                    out[f"{prefix}_{layer_name.replace('.', '_')}"] = float(np.nanmean(np.array(vals, dtype=np.float32)))
        self.scal = {}
        return out

    def flush_vecs(self) -> Dict[str, np.ndarray]:
        """Returns {layer_name: (N_txt,) text-key mass vector}, averaged over the layer's calls since the last flush."""
        out = {k: np.mean(np.stack(v), axis=0) for k, v in self.vecs.items() if v}
        self.vecs = {}
        return out
