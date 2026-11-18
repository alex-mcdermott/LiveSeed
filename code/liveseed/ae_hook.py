"""
ae_hook.py - attention-entropy hooks for the U-Net backbones (SD1.x, SD2.x, SDXL), used when a pool is logged.

This file computes the reading of Sec. 3.1 (Eq. 2) on a U-Net and is the reference design for the other hook files
(ae_hook_SD35.py, ae_hook_FLUX.py, ae_hook_HUNYUAN.py, ae_hook_QWEN.py); fast_hooks.py and fused_hooks.py compute the same
readings at lower cost for the live algorithm.

What is hooked
  attn1 (self-attention) of every transformer block: the LiveSeed reading. The keys are the block's N spatial tokens.
  attn2 (cross-attention to the prompt): diagnostics, and the stored maps that the ABSS baseline's statistic needs. The
  cross-attention quantities are not part of a LiveSeed score.

Design shared by all hook files
  * One collector object owns the state; it hands out one attention processor per layer. A processor computes its statistics
    while the layer runs and pushes plain floats to the collector.
  * The caller sets `col.step` (1-based index of the denoising step being computed) before each step, and calls
    `col.flush_and_get_layer_means()` after it. The flush returns {key: float} for that step and clears the state.
  * Every entropy is divided by ln K (K = number of keys), so a reading lies in [0, 1] whatever the resolution.
  * Under classifier-free guidance (CFG) diffusers stacks the batch as [unconditional, conditional]; the default readings
    average the two branches.
  * A failure inside a statistic is caught and never interrupts generation.

Readings returned by flush_and_get_layer_means() (<layer> is the diffusers processor name with dots replaced by
underscores, e.g. sa_mean_down_blocks_0_attentions_0_transformer_blocks_0_attn1_processor)
  sa_mean_<layer>       Self-attention entropy in the POOLED order: the attention distributions are averaged over heads,
                        queries and both CFG branches, then one entropy is taken, / ln N. This is the U-Net reading of the
                        paper's 30-step pools (App. C).
  sa_entfirst_<layer>   Self-attention entropy in the PER-ROW order of Eq. 2: the entropy of every row (one head, one query),
                        then the mean over heads, queries and both CFG branches, / ln N. This is the U-Net reading under
                        ABSS's protocol (Table 1). App. G ablates the two orders.
  sa_cond_<layer>       The pooled-order entropy of the conditional CFG branch alone.
  sa_cfg_delta_<layer>  The pooled-order entropy of the conditional branch minus that of the unconditional branch.
  sa_head_std_<layer>   Standard deviation, over heads and batch rows, of the entropy of each head's query-averaged row.
  ae_mean_<module>_pooled, ae_mean_<module>_t<i>_unpooled
                        Cross-attention entropy over the prompt's real tokens (pooled order, / ln of their number), per
                        transformer block i and averaged over the blocks of one attention module.
  ca_top_mass_*, ca_pad_mass_*
                        Cross-attention mass on the single most attended real token, and on BOS, EOS and padding together
                        (same `_pooled` / `_t<i>_unpooled` suffixes).
  sa_entfirst_ and (without a CFG batch) sa_cond_ are logged only when the collector is built with extra_orders=True.

The processors here form the full attention matrix and also compute the layer's output from it (scores and softmax in fp32,
probabilities cast back to the model's dtype), so the output does not come from the fused scaled_dot_product_attention kernel.

Install on a diffusers pipeline:
    col = AttnEntropyCollector()
    procs = {}
    for name in pipe.unet.attn_processors:
        procs[name] = col.get_processor(name) if "attn2" in name else col.get_sa_processor(name)
    pipe.unet.set_attn_processor(procs)

    col.new_image("key", n_active_tokens=N)      # N = prompt tokens without BOS, EOS and padding
    # in the pipeline's step callback:
    data = col.flush_and_get_layer_means()
    col.step = step + 1
"""
import math, re, torch
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F

# Base class of the cross-attention processor. Its __call__ is fully overridden below, so any of diffusers' processor
# classes serves; the fallbacks cover diffusers versions that name it differently.
try:
    from diffusers.models.attention_processor import AttnProcessor
except Exception:
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor
    except Exception:
        from diffusers.models.attention_processor import AttnProcessor as AttnProcessor


@dataclass
class _AEState:
    """Values pushed by the processors since the last flush: {layer_name: [float, ...]}, one entry per call of the layer."""
    key: str | None = None
    entropies: dict | None = None       # cross-attention (CA) entropy over the real prompt tokens
    sa_entropies: dict | None = None    # self-attention (SA) entropy, pooled order
    ca_top_masses: dict | None = None   # CA mass on the most attended real token
    ca_pad_masses: dict | None = None   # CA mass on BOS, EOS and PAD
    sa_head_stds: dict | None = None    # SA: standard deviation of the per-head entropies
    sa_cfg_deltas: dict | None = None   # SA: entropy of the conditional branch minus the unconditional branch
    sa_entfirst: dict | None = None     # SA entropy, per-row order (Eq. 2): the mean over rows of each row's entropy
    sa_cond: dict | None = None         # SA entropy, pooled order, conditional CFG branch only
    n_active_tokens: int | None = None  # real prompt tokens (excl. BOS, EOS, PAD)


class _AEAttnProcessor(AttnProcessor):
    """
    Installed on cross-attention (attn2) layers.

    Uses the pooled order: the attention distributions are averaged over all heads, spatial queries and both CFG branches
    into a single distribution over the active text tokens (BOS, EOS and PAD excluded), and its Shannon entropy is taken.
    """
    def __init__(self, parent, layer_name: str = "unknown"):
        super().__init__()
        self.parent = parent
        self.layer_name = layer_name

    def _record_entropy(self, attn_probs: torch.Tensor):
        """Pushes this call's cross-attention statistics to the collector.

        attn_probs: (B*H, N_q, N_k) attention probabilities, batch-major over (batch, head); N_k is the padded context
        length (77 for CLIP).
        """
        n_active = self.parent.state.n_active_tokens
        if n_active is not None and n_active >= 2:
            # Layout: BOS at 0, real tokens at 1..n_active, EOS at n_active+1, PAD after
            active = attn_probs[:, :, 1 : n_active + 1].float()  # (B*H, N_q, N_real)
        else:
            active = attn_probs.float()

        # Pooled order: collapse heads and spatial queries into a single distribution, renormalised over the kept tokens.
        avg_dist = active.mean(dim=0).mean(dim=0)               # (N_real,)
        avg_dist = avg_dist / avg_dist.sum().clamp_min_(1e-8)

        p = avg_dist.clamp_min_(1e-8)
        ent = -(p * p.log()).sum().item()
        # Longer prompts admit a higher entropy; dividing by ln(number of tokens) maps it to [0, 1].
        norm_factor = math.log(max(p.shape[0], 2))
        self.parent._push(self.layer_name, ent / norm_factor)

        # Mass on the most attended real token.
        top_mass = avg_dist.max().item()
        self.parent._push_ca_top(self.layer_name, top_mass)

        # Mass that escapes to tokens without content: BOS, and everything after the last real token (EOS and PAD).
        if n_active is not None and n_active >= 2:
            full_dist = attn_probs.float().mean(dim=0).mean(dim=0)  # (N_k,)
            full_dist = full_dist / full_dist.sum().clamp_min_(1e-8)
            bos_mass = full_dist[0].item()
            pad_tail = full_dist[n_active + 1:].sum().item() if full_dist.shape[0] > n_active + 1 else 0.0
            pad_mass = bos_mass + pad_tail
            self.parent._push_ca_pad(self.layer_name, pad_mass)

        # Stored cross-attention maps, for the ABSS baseline's statistic: at the listed steps, for the layers whose query
        # count matches the wanted resolution (16x16 = 256 queries at 512 px), keep the heads-averaged map of the
        # CONDITIONAL branch, (N_q, N_k) in fp16 on the host. The caller collects them with flush_ca_maps().
        steps = self.parent.ca_map_steps
        if steps and self.parent.step in steps and (self.parent.ca_map_queries is None or attn_probs.shape[1] == self.parent.ca_map_queries):
            BH, N_q, N_k = attn_probs.shape
            B = getattr(self, "_batch", 2)
            H = max(BH // max(B, 1), 1)
            n = B // 2 if (B % 2 == 0 and B >= 2) else B                      # CFG: rows [uncond(n), cond(n)] x heads
            cond = attn_probs[(B - n) * H:].view(n, H, N_q, N_k)             # the conditional branch, per sample
            self.parent._push_ca_map(self.layer_name, cond.float().mean(dim=1).to(torch.float16).cpu())   # (n, N_q, N_k)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """Attention with the probability matrix formed explicitly, so that it can be read; returns the layer's output."""
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        context_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = attn.to_q(hidden_states)
        key   = attn.to_k(context_states)
        value = attn.to_v(context_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # The scores and the softmax are computed in fp32 whatever the model's dtype; the probabilities are then cast back
        # to the model's dtype, and the statistics read these cast probabilities.
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1],
                        dtype=torch.float32, device=query.device),
            query.float(), key.float().transpose(-1, -2),
            beta=0, alpha=attn.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = attention_scores.softmax(dim=-1).to(dtype=value.dtype)

        # A failed statistic must not stop the image.
        try:
            self._batch = batch_size       # read by _record_entropy to split the (batch x head) rows into CFG branches
            self._record_entropy(attention_probs)
        except Exception:
            pass

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states) if len(attn.to_out) > 1 else hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class _SAAttnProcessor:
    """
    Installed on self-attention (attn1) layers: the LiveSeed reading on a U-Net.

    The main value (sa_mean_) uses the pooled order: the attention distributions are averaged over all heads, spatial
    queries and both CFG branches, and the Shannon entropy of that one distribution over the N spatial keys is taken
    (there is no padding to exclude). The per-row order of Eq. 2 (sa_entfirst_) is logged beside it.
    """
    def __init__(self, parent, layer_name: str = "unknown"):
        self.parent = parent
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """Self-attention with the probability matrix formed explicitly; logs the readings and returns the layer's output."""
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Scores and softmax in fp32, probabilities cast back to the model's dtype (as in the cross-attention processor).
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1],
                        dtype=torch.float32, device=query.device),
            query.float(), key.float().transpose(-1, -2),
            beta=0, alpha=attn.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        probs = attention_scores.softmax(dim=-1).to(dtype=value.dtype)      # (B*H, N, N), batch-major over (batch, head)

        # A failed statistic must not stop the image.
        try:
            # Pooled order (sa_mean_): one distribution over the N keys, averaged over batch rows (the CFG branches),
            # heads and queries, then its entropy / ln N.
            avg_dist = probs.float().mean(dim=0).mean(dim=0)  # (N,)
            avg_dist = avg_dist / avg_dist.sum().clamp_min_(1e-8)
            p = avg_dist.clamp_min_(1e-8)
            ent = -(p * p.log()).sum().item()
            norm_factor = math.log(max(p.shape[0], 2))
            self.parent._push_sa(self.layer_name, ent / norm_factor)

            # Spread across heads: the entropy of each (batch row, head)'s query-averaged distribution, then the
            # standard deviation of those entropies.
            per_bh = probs.float().mean(dim=1)  # (B*H, N_k)
            per_bh = per_bh / per_bh.sum(dim=-1, keepdim=True).clamp_min_(1e-8)
            per_bh_p = per_bh.clamp_min_(1e-8)
            per_bh_ent = -(per_bh_p * per_bh_p.log()).sum(dim=-1) / norm_factor  # (B*H,)
            head_std = per_bh_ent.std().item()
            self.parent._push_sa_head_std(self.layer_name, head_std)

            # The two CFG branches separately (batch layout [uncond, cond], so the first half of the B*H rows is the
            # unconditional branch): the conditional entropy, and conditional minus unconditional.
            BH = probs.shape[0]
            if batch_size >= 2 and BH >= 2:
                half = BH // 2

                def _half_ent(p_slice: torch.Tensor) -> float:
                    """Pooled-order normalised entropy of one branch's rows."""
                    d = p_slice.float().mean(dim=0).mean(dim=0)
                    d = d / d.sum().clamp_min_(1e-8)
                    pp = d.clamp_min_(1e-8)
                    return -(pp * pp.log()).sum().item() / norm_factor

                uncond_ent = _half_ent(probs[:half])
                cond_ent = _half_ent(probs[half : 2 * half])
                cfg_delta = cond_ent - uncond_ent
                self.parent._push_sa_cfg_delta(self.layer_name, cfg_delta)
                self.parent._push_sa_cond(self.layer_name, cond_ent)        # the conditional branch alone
            elif self.parent.extra_orders:
                self.parent._push_sa_cond(self.layer_name, ent / norm_factor)   # no CFG batch: the one branch is the conditional one

            # Per-row order (sa_entfirst_, Eq. 2): the entropy of every row (head x query), then the mean, / ln N.
            if self.parent.extra_orders:
                pf = probs.float().clamp_min(1e-8)
                row_ent = -(pf * pf.log()).sum(dim=-1)                      # (B*H, N_q)
                self.parent._push_sa_entfirst(self.layer_name, row_ent.mean().item() / norm_factor)
        except Exception:
            pass

        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states) if len(attn.to_out) > 1 else hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class AttnEntropyCollector:
    """
    Hands out the cross-attention and self-attention processors and collects what they push, per denoising step.

    Public interface:
        col = AttnEntropyCollector()
        col.new_image("key", n_active_tokens=N)
        # in step callback:
        data = col.flush_and_get_layer_means()
        # data contains "ae_mean_*" (CA, block-pooled) and "sa_mean_*" (SA) keys; the module docstring lists them all

    Arguments:
        ca_map_steps    steps (1-based) at which cross-attention maps are kept for flush_ca_maps(); none by default.
        ca_map_queries  keep maps only from layers with this many queries (e.g. 256 for 16x16); None keeps every layer.
        extra_orders    also log the per-row entropy (sa_entfirst_); switch off for cheaper runs.
        device          stored but not used.
    """
    def __init__(self, device="cuda", ca_map_steps=None, ca_map_queries=None, extra_orders=True):
        self.device = device
        self.state = _AEState(key=None, entropies={}, sa_entropies={},
                              ca_top_masses={}, ca_pad_masses={},
                              sa_head_stds={}, sa_cfg_deltas={}, sa_entfirst={}, sa_cond={})
        # The caller sets `step` (1-based, the step being computed) before each denoising step; the processors read it to
        # decide whether a cross-attention map is wanted at this step.
        self.step = 1
        self.ca_map_steps = set(ca_map_steps or [])
        self.ca_map_queries = ca_map_queries
        self.extra_orders = extra_orders
        self.ca_maps: dict = {}

    def get_processor(self, layer_name: str) -> _AEAttnProcessor:
        """The processor for a cross-attention (attn2) layer."""
        return _AEAttnProcessor(self, layer_name=layer_name)

    def get_sa_processor(self, layer_name: str) -> _SAAttnProcessor:
        """The processor for a self-attention (attn1) layer."""
        return _SAAttnProcessor(self, layer_name=layer_name)

    # ------------------------------------------------------------------ stored cross-attention maps
    def _push_ca_map(self, layer_name: str, m):
        self.ca_maps[layer_name] = m

    def flush_ca_maps(self) -> dict:
        """Returns {layer_name: (n, N_q, N_k) fp16 tensor} for the maps kept since the last call, and clears them."""
        out, self.ca_maps = self.ca_maps, {}
        return out

    # ------------------------------------------------------------------ scalar accumulators
    # One per statistic; each appends a finite value to the list of its layer. Non-finite values are dropped so that a
    # single bad call cannot poison the step's mean.
    def _push_sa_entfirst(self, layer_name: str, val: float):
        if self.state.sa_entfirst is not None and math.isfinite(val):
            self.state.sa_entfirst.setdefault(layer_name, []).append(float(val))

    def _push_sa_cond(self, layer_name: str, val: float):
        if self.state.sa_cond is not None and math.isfinite(val):
            self.state.sa_cond.setdefault(layer_name, []).append(float(val))

    def _push(self, layer_name: str, ent_mean: float):
        if self.state.entropies is not None and math.isfinite(ent_mean):
            self.state.entropies.setdefault(layer_name, []).append(float(ent_mean))

    def _push_sa(self, layer_name: str, ent_mean: float):
        if self.state.sa_entropies is not None and math.isfinite(ent_mean):
            self.state.sa_entropies.setdefault(layer_name, []).append(float(ent_mean))

    def _push_ca_top(self, layer_name: str, val: float):
        if self.state.ca_top_masses is not None and math.isfinite(val):
            self.state.ca_top_masses.setdefault(layer_name, []).append(float(val))

    def _push_ca_pad(self, layer_name: str, val: float):
        if self.state.ca_pad_masses is not None and math.isfinite(val):
            self.state.ca_pad_masses.setdefault(layer_name, []).append(float(val))

    def _push_sa_head_std(self, layer_name: str, val: float):
        if self.state.sa_head_stds is not None and math.isfinite(val):
            self.state.sa_head_stds.setdefault(layer_name, []).append(float(val))

    def _push_sa_cfg_delta(self, layer_name: str, val: float):
        if self.state.sa_cfg_deltas is not None and math.isfinite(val):
            self.state.sa_cfg_deltas.setdefault(layer_name, []).append(float(val))

    # ------------------------------------------------------------------ per image and per step
    def new_image(self, key: str, n_active_tokens: int | None = None):
        """Resets all state before an image. n_active_tokens is the prompt's token count without BOS, EOS and padding;
        the cross-attention statistics use it to find the real tokens."""
        self.state = _AEState(key=key, entropies={}, sa_entropies={},
                              ca_top_masses={}, ca_pad_masses={},
                              sa_head_stds={}, sa_cfg_deltas={}, sa_entfirst={}, sa_cond={},
                              n_active_tokens=n_active_tokens)
        self.step = 1
        self.ca_maps = {}

    def flush_and_get_layer_means(self) -> dict:
        """
        Returns the statistics of the step just computed and clears the accumulators for the next step. A layer called
        several times within the step contributes the mean of its calls.

        Keys (names with dots replaced by underscores):
          - "ae_mean_<module>_pooled"        — CA entropy, averaged over the transformer_blocks
                                               within the same attentions module
          - "ae_mean_<module>_t<N>_unpooled" — CA entropy at one transformer block
          - "ca_top_mass_*", "ca_pad_mass_*" — the CA masses, with the same two suffixes
          - "sa_mean_<layer>"                — SA entropy (pooled order) per individual layer
          - "sa_entfirst_<layer>", "sa_cond_<layer>", "sa_cfg_delta_<layer>", "sa_head_std_<layer>"
        """
        out = {}

        # --- Cross-attention: emit both pooled (per module) and unpooled (per block) ---
        def _pool_and_emit(src_dict: dict, prefix: str):
            """Emit pooled-per-module and per-block-unpooled entries for a CA-side scalar dict."""
            raw = {}
            for layer_name, vals in (src_dict or {}).items():
                arr = np.array(vals, dtype=np.float32)
                if np.any(np.isfinite(arr)):
                    raw[layer_name] = float(np.nanmean(arr))

            groups: dict[str, list] = {}
            for layer_name, val in raw.items():
                # "<module>.transformer_blocks.<i>.attn2..." -> module and block index (SDXL has several blocks per module)
                m = re.match(r'^(.*?)(?:\.transformer_blocks\.(\d+))?\.attn2', layer_name)
                module_key = m.group(1) if m else layer_name
                block_idx  = m.group(2) if m else None
                groups.setdefault(module_key, []).append(val)
                if block_idx is not None:
                    safe_module = module_key.replace(".", "_")
                    out[f"{prefix}_{safe_module}_t{block_idx}_unpooled"] = val
            for module_key, group_vals in groups.items():
                safe = module_key.replace(".", "_")
                out[f"{prefix}_{safe}_pooled"] = float(np.nanmean(group_vals))

        _pool_and_emit(self.state.entropies,    "ae_mean")      # CA entropy
        _pool_and_emit(self.state.ca_top_masses, "ca_top_mass")  # mass on the top real token
        _pool_and_emit(self.state.ca_pad_masses, "ca_pad_mass")  # mass on BOS, EOS and padding

        # --- Self-attention (per-block; no module-level pooling) ---
        def _emit_sa(src_dict: dict, prefix: str):
            for layer_name, vals in (src_dict or {}).items():
                arr = np.array(vals, dtype=np.float32)
                if np.any(np.isfinite(arr)):
                    safe = layer_name.replace(".", "_")
                    out[f"{prefix}_{safe}"] = float(np.nanmean(arr))

        _emit_sa(self.state.sa_entropies,  "sa_mean")         # SA entropy, pooled order
        _emit_sa(self.state.sa_head_stds,  "sa_head_std")     # standard deviation of the per-head entropies
        _emit_sa(self.state.sa_cfg_deltas, "sa_cfg_delta")    # conditional minus unconditional
        _emit_sa(self.state.sa_entfirst,   "sa_entfirst")     # SA entropy, per-row order (Eq. 2)
        _emit_sa(self.state.sa_cond,       "sa_cond")         # conditional branch only

        self.state.entropies    = {}
        self.state.sa_entropies = {}
        self.state.ca_top_masses = {}
        self.state.ca_pad_masses = {}
        self.state.sa_head_stds  = {}
        self.state.sa_cfg_deltas = {}
        self.state.sa_entfirst   = {}
        self.state.sa_cond       = {}
        return out

    def flush_and_get_mean(self) -> float:
        """Flushes the step and returns the mean of all its `*_pooled` (cross-attention) values, or NaN if there are none."""
        layer_means = self.flush_and_get_layer_means()
        ca_vals = [v for k, v in layer_means.items() if k.endswith("_pooled")]
        if not ca_vals:
            return float("nan")
        return float(np.nanmean(ca_vals))
