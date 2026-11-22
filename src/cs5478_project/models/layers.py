"""Attention and positional-encoding primitives used by SingleFrameNet."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mha_sdpa(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key_value: torch.Tensor,
    num_heads: int,
    key_padding_mask: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run nn.MultiheadAttention via F.scaled_dot_product_attention.

    Reuses the parameters of an existing ``nn.MultiheadAttention`` (so old
    checkpoints load without remapping) but routes through SDPA directly,
    skipping the dispatcher overhead in ``MultiheadAttention.forward``.

    Assumes K and V share the same input tensor (the only shape used in this
    repo). Masks are bool with True=ignore, matching the rest of the codebase.

    Args:
        mha: existing nn.MultiheadAttention whose in/out projections are reused.
        query: [B, Lq, D].
        key_value: [B, Lk, D] used as both K and V.
        num_heads: number of attention heads (same as ``mha.num_heads``).
        key_padding_mask: optional [B, Lk] bool, True=ignore.
        attn_mask: optional [B*H, Lq, Lk] bool, True=ignore (same convention as
            ``nn.MultiheadAttention``).

    Returns:
        [B, Lq, D] attention output. Numerically equivalent to
        ``mha(query, key_value, key_value, ..., need_weights=False)[0]``.
    """
    B, Lq, D = query.shape
    Lk = key_value.shape[1]
    H = num_heads
    Dh = D // H

    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
    if mha.in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
    q = F.linear(query, w_q, b_q)
    # Fused K and V projection: K and V share input so one matmul suffices.
    kv_w = torch.cat([w_k, w_v], dim=0)
    kv_b = None if b_k is None else torch.cat([b_k, b_v], dim=0)
    kv = F.linear(key_value, kv_w, kv_b)
    k, v = kv.chunk(2, dim=-1)

    q = q.view(B, Lq, H, Dh).transpose(1, 2)
    k = k.view(B, Lk, H, Dh).transpose(1, 2)
    v = v.view(B, Lk, H, Dh).transpose(1, 2)

    # Merge masks. Inputs use True=ignore; SDPA bool mask uses True=keep, so
    # invert at the end.
    merged: torch.Tensor | None = None
    if attn_mask is not None:
        merged = attn_mask.view(B, H, Lq, Lk)
    if key_padding_mask is not None:
        kpm = key_padding_mask.view(B, 1, 1, Lk)
        merged = kpm if merged is None else (merged | kpm)
    sdpa_mask = (~merged) if merged is not None else None

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
    out = out.transpose(1, 2).contiguous().view(B, Lq, D)
    return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)


def gen_sineembed_for_position(pos_tensor: torch.Tensor, hidden_dim: int = 256) -> torch.Tensor:
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/"""
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)

    # Use ... to handle arbitrary tensor ranks
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    # Add a new dimension for broadcasting with dim_t
    pos_x = x_embed.unsqueeze(-1) / dim_t
    pos_y = y_embed.unsqueeze(-1) / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[..., 2] * scale
        pos_w = w_embed.unsqueeze(-1) / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)

        h_embed = pos_tensor[..., 3] * scale
        pos_h = h_embed.unsqueeze(-1) / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos
