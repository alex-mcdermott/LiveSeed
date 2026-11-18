"""
ae_hook_SD35.py - attention-entropy hooks for SD3.5 (diffusers SD3Transformer2DModel), used when a pool is logged.

SD3.5 is a multimodal diffusion transformer (MMDiT): every JointTransformerBlock has a joint attention `attn` over the
concatenated text and image tokens. This file computes the reading of Sec. 3.1 (Eq. 2) on that joint attention. The design
(one collector, one processor per layer, `col.step`, a flush per denoising step, entropies divided by ln of the number of
keys, statistics that never interrupt generation) is that of ae_hook.py, the reference; only the differences are described
here.

What is hooked
  `transformer_blocks.N.attn` (joint attention). Only the IMAGE-query rows are read; their keys are all S = N_txt + N_img
  tokens. The image-only self-attention `attn2` that some SD3.5-Medium blocks also have keeps its default processor.

Readings returned by flush_and_get_layer_means() (<layer> is the processor name with dots replaced by underscores, e.g.
ae_mean_transformer_blocks_0_attn_processor)
  ae_mean_<layer>       The LiveSeed reading, in the PER-ROW order of Eq. 2: the entropy of every image-query row over the
                        joint keys, then the mean over heads, image queries and both branches of classifier-free guidance
                        (CFG; batch layout [unconditional, conditional]), / ln S.
  ae_cond_<layer>       The same for the conditional branch alone.
  ja_cfg_delta_<layer>  Conditional minus unconditional per-row entropy.
  ae_avgfirst_<layer>   The POOLED order: rows averaged over heads, image queries and branches, then one entropy, / ln S
                        (only with extra_orders=True; App. G ablates the two orders).
  ja_head_std_<layer>   Standard deviation, over heads and batch rows, of the entropy of each head's query-averaged row.
  ja_top_mass_<layer>   Largest share of the text-key mass held by a single text token.
  txtmass_<layer>       Mean attention mass that the image queries of the conditional branch place on the text keys.

Two ways of computing them
  explicit  The S x S attention matrix is formed in the model's dtype and every statistic above is read from it.
  fused     With `orig` given and the collector attribute `fused` set to True, the row entropies come from one
            memory-efficient attention call with the keys as values (Sec. 3.1, App. D; fused_hooks.py), and the matrix is
            never formed. ja_head_std_ and ae_avgfirst_ are not available in this mode.
  In both, when the model's own processor is passed as `orig`, the block's output comes from it, so the images are
  identical to an unhooked run. With orig=None the output is computed here from the explicit matrix.

Install on a diffusers pipeline: see AttnEntropyCollector.
"""
import math
import torch
import numpy as np
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class _AEState:
    """Values pushed by the processors since the last flush: {layer_name: [float, ...]}. JA = joint attention."""
    key: str | None = None
    entropies: dict | None = None       # JA entropy, per-row order (Eq. 2): the reading
    ja_top_masses: dict | None = None   # largest single-token share of the text-key mass
    ja_head_stds: dict | None = None    # standard deviation of the per-head entropies
    ja_cfg_deltas: dict | None = None   # entropy of the conditional branch minus the unconditional branch
    ja_avgfirst: dict | None = None     # JA entropy, pooled order: rows averaged first, then one entropy
    ja_cond: dict | None = None         # JA entropy (per-row order) of the conditional CFG branch only
    txtmass: dict | None = None         # mean mass the image queries put on the text keys, conditional branch


class _SD3JointAEAttnProcessor(nn.Module):
    """
    Replaces JointAttnProcessor2_0 on a JointTransformerBlock's `attn`.

    Recomputes the joint text+image attention so that the entropy of the image-query rows can be recorded per layer. The
    return value is (image_hidden_states, encoder_hidden_states), as JointTransformerBlock.forward() expects.
    """

    def __init__(self, parent: "AttnEntropyCollector", layer_name: str, orig=None):
        super().__init__()
        self.parent = parent
        self.layer_name = layer_name
        # When the model's own processor is given, the block's OUTPUT comes from it and the probabilities computed below
        # are used for the readings only, so images are identical to an unhooked run. (The explicit softmax in the model's
        # dtype is not bit-identical to the fused kernel; on the four-step SD3.5-Large Turbo it changes the images, with
        # maximum pixel differences of 127-239.) With orig=None the output is computed from the explicit matrix.
        self.orig = orig

    def _log_entropy(self, img_probs: torch.Tensor, N_txt: int, N_img: int, B: int = 1):
        """Pushes every statistic of the explicit path to the collector.

        img_probs: (B*H, N_img, N_txt+N_img), the image-query rows of the attention matrix over all keys (text first),
        batch-major over (batch, head). B is the batch size (2 x images under CFG).
        """
        try:
            norm_factor = math.log(max(N_txt + N_img, 2))       # ln S: the entropy of a uniform row
            p = img_probs.float().clamp_min_(1e-8)

            # The reading (Eq. 2): entropy of each row, then the mean over batch rows, heads and image queries.
            ent = -(p * p.log()).sum(dim=-1)  # (B*H, N_img)
            ent_mean = ent.mean().item() / norm_factor
            self.parent._push(self.layer_name, ent_mean)

            # Top-1 text-token mass: the image queries' attention on the text keys, averaged over heads and queries and
            # normalised to sum to one over the text tokens; its largest entry.
            text_slice = img_probs[:, :, :N_txt].float()  # (B*H, N_img, N_txt)
            avg_text = text_slice.mean(dim=0).mean(dim=0)  # (N_txt,)
            avg_text = avg_text / avg_text.sum().clamp_min_(1e-8)
            top_mass = avg_text.max().item()
            self.parent._push_ja_top(self.layer_name, top_mass)

            # Spread across heads: the entropy of each (batch row, head)'s query-averaged row, then their standard deviation.
            per_bh = p.mean(dim=1)  # (B*H, N_txt+N_img) averaged over query positions
            per_bh = per_bh / per_bh.sum(dim=-1, keepdim=True).clamp_min_(1e-8)
            per_bh_p = per_bh.clamp_min_(1e-8)
            per_bh_ent = -(per_bh_p * per_bh_p.log()).sum(dim=-1) / norm_factor  # (B*H,)
            head_std = per_bh_ent.std().item()
            self.parent._push_ja_head_std(self.layer_name, head_std)

            # The two CFG branches separately (the first half of the B*H rows is the unconditional branch).
            BH = img_probs.shape[0]
            if B >= 2 and BH >= 2:
                half = BH // 2

                def _half_ent(p_slice: torch.Tensor) -> float:
                    """Per-row normalised entropy of one branch's rows."""
                    return (-(p_slice.float().clamp_min_(1e-8) *
                              p_slice.float().clamp_min_(1e-8).log()).sum(dim=-1)
                            ).mean().item() / norm_factor

                uncond_ent = _half_ent(img_probs[:half])
                cond_ent   = _half_ent(img_probs[half : 2 * half])
                cfg_delta = cond_ent - uncond_ent
                self.parent._push_ja_cfg_delta(self.layer_name, cfg_delta)
                self.parent._push_ja_cond(self.layer_name, cond_ent)                 # the conditional branch alone
                cond_probs = img_probs[half: 2 * half]
            else:
                self.parent._push_ja_cond(self.layer_name, ent_mean)                 # no CFG batch: the one branch is the conditional one
                cond_probs = img_probs

            # Text mass of the conditional branch: mean over heads and image queries of the attention on the text keys
            # (the DiT quantity ABSS reads before smoothing; the harness's token vector is this distribution normalised).
            self.parent._push_txtmass(self.layer_name, cond_probs[:, :, :N_txt].float().sum(dim=-1).mean().item())

            # Pooled order (ae_avgfirst_): average the rows over heads and image queries, then one entropy / ln S.
            if self.parent.extra_orders:
                avg = img_probs.float().mean(dim=0).mean(dim=0)
                avg = (avg / avg.sum().clamp_min_(1e-8)).clamp_min_(1e-8)
                self.parent._push_ja_avgfirst(self.layer_name, -(avg * avg.log()).sum().item() / norm_factor)
        except Exception:
            pass       # a failed statistic must not stop the image

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask=None,
        **kwargs,
    ):
        """Logs the readings of this joint-attention call and returns (image_out, text_out)."""
        B, N_img, _ = hidden_states.shape
        N_txt = encoder_hidden_states.shape[1]
        H = attn.heads
        Dh = attn.to_q.out_features // H

        # --- Image stream projections (with the model's query/key normalisation) ---
        image_q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (H, Dh)))
        image_k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (H, Dh)))
        image_v = attn.to_v(hidden_states).unflatten(-1, (H, Dh))

        # --- Text stream projections ---
        text_q = attn.norm_added_q(attn.add_q_proj(encoder_hidden_states).unflatten(-1, (H, Dh)))
        text_k = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (H, Dh)))
        text_v = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (H, Dh))

        # Concatenate text first, then image → (B, N_txt+N_img, H, Dh), so the text keys are columns 0..N_txt-1 of every row.
        q_all = torch.cat([text_q, image_q], dim=1)
        k_all = torch.cat([text_k, image_k], dim=1)
        v_all = torch.cat([text_v, image_v], dim=1)

        S = q_all.shape[1]

        # ------------------------------------------------------------------ fused path
        if self.orig is not None and getattr(self.parent, "fused", False) and attention_mask is None:
            # Fused readings, for models where the materialised S x S matrix is too costly (on SD3.5-Large it makes
            # logging cost 4.8 times an image).
            # Row entropy H = logsumexp(s) - sum_j p_j s_j, and the second term = scale * q . (attention output with the KEYS
            # as values), so one memory-efficient attention call gives every image row's entropy without the matrix
            # (fused_hooks.py). The text-key probabilities come from the small image x text score block and the same
            # log-sum-exp.
            # Logged: ae_mean, ae_cond, ja_cfg_delta, txtmass, ja_top_mass and the text vector; NOT ja_head_std or
            # ae_avgfirst, which need the pooled rows of the full matrix.
            try:
                with torch.no_grad():
                    scale = Dh ** -0.5
                    q = image_q.transpose(1, 2).contiguous()                                  # (B, H, N_img, Dh)
                    k = k_all.transpose(1, 2).contiguous()                                    # (B, H, S, Dh), text first
                    # Arguments: (query, key, value = key, no bias, return the log-sum-exp, no dropout, not causal).
                    o, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, k, None, True, 0.0, False, scale=scale)[:2]
                    lse = lse[..., :N_img].float()                                            # (B, H, N_img); the kernel may pad this axis
                    ent = lse - scale * torch.einsum("bhnd,bhnd->bhn", q.float(), o.float())  # (B, H, N_img) row entropies (nats)
                    norm_factor = math.log(max(S, 2))
                    self.parent._push(self.layer_name, ent.mean().item() / norm_factor)

                    # p(text key | image query) = exp(score - logsumexp over all keys)                 (B, H, N_img, N_txt)
                    pt = torch.exp(scale * torch.einsum("bhnd,bhtd->bhnt", q.float(), k[:, :, :N_txt].float()) - lse[..., None])
                    avg_text = pt.mean(dim=(0, 1, 2))
                    avg_text = avg_text / avg_text.sum().clamp_min(1e-8)
                    self.parent._push_ja_top(self.layer_name, avg_text.max().item())

                    if B >= 2:      # CFG batch [uncond, cond]
                        h = B // 2
                        cond_ent = ent[h:2 * h].mean().item() / norm_factor
                        self.parent._push_ja_cfg_delta(self.layer_name, cond_ent - ent[:h].mean().item() / norm_factor)
                        self.parent._push_ja_cond(self.layer_name, cond_ent)
                        pc = pt[h:2 * h]
                    else:
                        self.parent._push_ja_cond(self.layer_name, ent.mean().item() / norm_factor)
                        pc = pt
                    self.parent._push_txtmass(self.layer_name, pc.sum(dim=-1).mean().item())

                    # The text-key mass vector of the conditional branch, for collectors that keep it (seed_harness.py's).
                    if hasattr(self.parent, "_push_vec"):
                        v = pc.mean(dim=(0, 1, 2))
                        v = v / v.sum().clamp_min(1e-8)
                        self.parent._push_vec(self.layer_name, v.detach().cpu().numpy().astype(np.float32))
            except Exception as e:      # never break generation for a statistic; report the first three failures
                self.parent.errors = getattr(self.parent, "errors", 0) + 1
                if self.parent.errors <= 3:
                    print(f"[ae_hook_SD35] fused readings failed at {self.layer_name}: {type(e).__name__}: {e}", flush=True)
            return self.orig(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        # ------------------------------------------------------------------ explicit path
        # Reshape to (B*H, S, Dh) for batched matmul
        q2d = q_all.permute(0, 2, 1, 3).reshape(B * H, S, Dh)
        k2d = k_all.permute(0, 2, 1, 3).reshape(B * H, S, Dh)
        v2d = v_all.permute(0, 2, 1, 3).reshape(B * H, S, Dh)

        scale = Dh ** -0.5
        attn_scores = torch.baddbmm(
            torch.empty(B * H, S, S, dtype=q2d.dtype, device=q2d.device),
            q2d,
            k2d.transpose(-1, -2),
            beta=0,
            alpha=scale,
        )

        if attention_mask is not None:
            mask = attention_mask.to(attn_scores.dtype)
            if mask.dim() == 4:
                mask = mask.repeat_interleave(H, 0)
            attn_scores = attn_scores + mask

        probs = attn_scores.softmax(dim=-1)

        # The readings: image query rows attending to all (text + image) keys
        self._log_entropy(probs[:, N_txt:, :], N_txt, N_img, B=B)

        if self.orig is not None:
            return self.orig(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        # No original processor: finish the attention from the explicit probabilities.
        out2d = torch.bmm(probs.to(v2d.dtype), v2d)
        out = out2d.reshape(B, H, S, Dh).permute(0, 2, 1, 3).flatten(2, 3)

        text_out = out[:, :N_txt]
        image_out = out[:, N_txt:]

        image_out = attn.to_out[0](image_out)
        image_out = attn.to_out[1](image_out)  # Dropout (no-op at eval)
        if not attn.context_pre_only:
            text_out = attn.to_add_out(text_out)

        return image_out, text_out


class AttnEntropyCollector:
    """
    Hands out the SD3.5 joint-attention processors and collects the per-step, per-layer readings. The public interface is
    that of ae_hook.AttnEntropyCollector, so seed_harness.py can use the same callback logic.

    Usage:
        col = AttnEntropyCollector()
        procs = {}
        for name, proc in pipe.transformer.attn_processors.items():
            if "attn2" not in name:              # joint attention: hooked, the model's processor makes the output
                procs[name] = col.get_processor(name, orig=proc)
            else:                                # image-only self-attention: left as it is
                procs[name] = proc
        pipe.transformer.set_attn_processor(procs)

        col.new_image("key")
        # inside step callback:
        layer_means = col.flush_and_get_layer_means()

    extra_orders=True also logs the pooled order (ae_avgfirst_). Setting the attribute `col.fused = True` after
    construction selects the fused path of the processors (see the module docstring); `device` is stored but not used.
    """

    def __init__(self, device: str = "cuda", extra_orders: bool = True):
        self.device = device
        self.extra_orders = extra_orders
        self.step = 1
        self.state = _AEState(key=None, entropies={},
                              ja_top_masses={}, ja_head_stds={}, ja_cfg_deltas={}, ja_avgfirst={}, ja_cond={}, txtmass={})

    # ------------------------------------------------------------------ scalar accumulators
    # One per statistic; each appends a finite value to the list of its layer (non-finite values are dropped).
    def _push_ja_avgfirst(self, layer_name: str, val: float):
        if self.state.ja_avgfirst is not None and math.isfinite(val):
            self.state.ja_avgfirst.setdefault(layer_name, []).append(float(val))

    def _push_ja_cond(self, layer_name: str, val: float):
        if self.state.ja_cond is not None and math.isfinite(val):
            self.state.ja_cond.setdefault(layer_name, []).append(float(val))

    def _push_txtmass(self, layer_name: str, val: float):
        if self.state.txtmass is not None and math.isfinite(val):
            self.state.txtmass.setdefault(layer_name, []).append(float(val))

    def get_processor(self, layer_name: str, orig=None) -> _SD3JointAEAttnProcessor:
        """The processor for a joint-attention layer; pass the layer's current processor as `orig` to keep its output."""
        return _SD3JointAEAttnProcessor(self, layer_name=layer_name, orig=orig)

    def _push(self, layer_name: str, ent_mean: float):
        if self.state.entropies is not None and math.isfinite(ent_mean):
            self.state.entropies.setdefault(layer_name, []).append(float(ent_mean))

    def _push_ja_top(self, layer_name: str, val: float):
        if self.state.ja_top_masses is not None and math.isfinite(val):
            self.state.ja_top_masses.setdefault(layer_name, []).append(float(val))

    def _push_ja_head_std(self, layer_name: str, val: float):
        if self.state.ja_head_stds is not None and math.isfinite(val):
            self.state.ja_head_stds.setdefault(layer_name, []).append(float(val))

    def _push_ja_cfg_delta(self, layer_name: str, val: float):
        if self.state.ja_cfg_deltas is not None and math.isfinite(val):
            self.state.ja_cfg_deltas.setdefault(layer_name, []).append(float(val))

    # ------------------------------------------------------------------ per image and per step
    def new_image(self, key: str):
        """Resets all state before an image."""
        self.step = 1
        self.state = _AEState(key=key, entropies={},
                              ja_top_masses={}, ja_head_stds={}, ja_cfg_deltas={}, ja_avgfirst={}, ja_cond={}, txtmass={})

    def flush_and_get_layer_means(self) -> dict:
        """Returns {"<prefix>_<layer>": float} for the step just computed (the prefixes are listed in the module docstring)
        and clears the accumulators. A layer called several times within the step contributes the mean of its calls."""
        out = {}

        def _emit(src_dict: dict, prefix: str):
            for layer_name, vals in (src_dict or {}).items():
                if vals:
                    arr = np.array(vals, dtype=np.float32)
                    if np.any(np.isfinite(arr)):
                        safe = layer_name.replace(".", "_")
                        out[f"{prefix}_{safe}"] = float(np.nanmean(arr))

        _emit(self.state.entropies,    "ae_mean")       # JA entropy, per-row order: the reading
        _emit(self.state.ja_top_masses, "ja_top_mass")  # top-1 text-token mass
        _emit(self.state.ja_head_stds,  "ja_head_std")  # standard deviation of the per-head entropies
        _emit(self.state.ja_cfg_deltas, "ja_cfg_delta") # conditional minus unconditional
        _emit(self.state.ja_avgfirst,  "ae_avgfirst")   # JA entropy, pooled order
        _emit(self.state.ja_cond,      "ae_cond")       # conditional branch only
        _emit(self.state.txtmass,      "txtmass")       # text mass, conditional branch (ABSS DiT statistic)

        self.state.entropies     = {}
        self.state.ja_top_masses = {}
        self.state.ja_head_stds  = {}
        self.state.ja_cfg_deltas = {}
        self.state.ja_avgfirst   = {}
        self.state.ja_cond       = {}
        self.state.txtmass       = {}
        return out

    def flush_and_get_mean(self) -> float:
        """Flushes the step and returns the mean of all its values, over every layer and every statistic (NaN if none)."""
        layer_means = self.flush_and_get_layer_means()
        vals = list(layer_means.values())
        if not vals:
            return float("nan")
        return float(np.nanmean(vals))
