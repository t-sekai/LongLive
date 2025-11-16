# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

USE_FLASH_ATTN_3 = FLASH_ATTN_3_AVAILABLE and os.environ.get("LONGLIVE_DISABLE_FA3", "0") != "1"

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # original shapes
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        # In your current runs, context_lens is None so this is fine.
        if q_lens is not None or k_lens is not None:
            raise RuntimeError(
                "SDPA fallback in flash_attention does not support q_lens/k_lens; "
                f"got q_lens={q_lens is not None}, k_lens={k_lens is not None}"
            )

        # SDPA expects [B, H, L, D]
        q_sdpa = q.to(dtype).permute(0, 2, 1, 3)  # [B, H, Lq, D]
        k_sdpa = k.to(dtype).permute(0, 2, 1, 3)  # [B, H, Lk, D]
        v_sdpa = v.to(dtype).permute(0, 2, 1, 3)  # [B, H, Lk, D]

        if q_scale is not None:
            q_sdpa = q_sdpa * q_scale

        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
        )
        # back to [B, Lq, H, D]
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.type(out_dtype)

    total_queries = None

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

     # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        num_heads = q.size(1)
        q_lens = torch.full(
            (b,),
            lq,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
        num_heads = q.size(1)

    # Avoid .item() on CUDA during graph capture: just read the batch dimension
    total_queries = q.size(0)

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    global USE_FLASH_ATTN_3
    x = None
    if (version is None or version == 3) and USE_FLASH_ATTN_3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        try:
            fa3_out = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)[0]
            if fa3_out.size(0) == total_queries:
                x = fa3_out
            elif fa3_out.dim() == 3 and fa3_out.size(0) == num_heads and fa3_out.size(1) == total_queries:
                # FlashAttention-3 may return [num_heads, total_queries, head_dim]
                x = fa3_out.permute(1, 0, 2).contiguous()
            else:
                raise RuntimeError(
                    f"Unexpected FlashAttention-3 output shape {tuple(fa3_out.shape)}; "
                    f"expected ({total_queries}, {num_heads}, head_dim) or "
                    f"({num_heads}, {total_queries}, head_dim)."
                )
            x = x.unflatten(0, (b, lq))
        except RuntimeError as err:
            warnings.warn(
                f"FlashAttention-3 failed ({err}); falling back to FlashAttention-2.",
                stacklevel=2,
            )
            USE_FLASH_ATTN_3 = False
            x = None
    if x is None:
        assert FLASH_ATTN_2_AVAILABLE, "FlashAttention-2 is not available"
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
