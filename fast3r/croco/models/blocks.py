# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References:
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py


import collections.abc
from itertools import repeat
import math

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0,
        attn_mask=None, is_causal=False, attn_implementation="pytorch_naive",
        attn_bias_for_inference_enabled=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        # use attention biasing to accommodate for longer sequences than during training
        self.attn_bias_for_inference_enabled = attn_bias_for_inference_enabled
        gamma = 1.0
        train_seqlen = 20
        inference_seqlen = 137
        self.attn_bias_scale = head_dim**-0.5 * (gamma * math.log(inference_seqlen) / math.log(train_seqlen))**0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.attn_mask = attn_mask
        self.is_causal = is_causal
        self.attn_implementation = attn_implementation

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        if self.rope is not None:
            device_type = next(self.parameters()).device.type
            if device_type == 'mps':
                # Skip autocast for MPS devices
                # The q, k, v are already computed above, so we don't need to recompute them
                pass
            else:
                # Use autocast for supported devices
                with torch.autocast(device_type=device_type, dtype=torch.float32):
                    pass  # q, k, v are already computed above
            q = self.rope(q, xpos) if xpos is not None else q
            k = self.rope(k, xpos) if xpos is not None else k

        if not self.training and self.attn_bias_for_inference_enabled:
            scale = self.attn_bias_scale
        else:
            scale = self.scale

        # Important: For the fusion Transformer, we forward through the attention with bfloat16 precision
        # If you are not using this block for the fusion Transformer, you should double check the precision of the input and output
        if self.attn_implementation == "pytorch_naive":
            assert self.attn_mask is None, "attn_mask not supported for pytorch_naive implementation of scaled dot product attention"
            assert self.is_causal is False, "is_causal not supported for pytorch_naive implementation of scaled dot product attention"
            dtype = k.dtype
            device_type = next(self.parameters()).device.type
            if device_type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = (q @ k.transpose(-2, -1)) * scale
                    x = x.softmax(dim=-1)
                    x = self.attn_drop(x)
                    x = (x @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
            else:
                # For CPU or MPS, don't use autocast
                x = (q @ k.transpose(-2, -1)) * scale
                x = x.softmax(dim=-1)
                x = self.attn_drop(x)
                if dtype == torch.float32:  # if input was FP32, cast back to FP32
                    x = x.to(torch.float32)
                x = (x @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
        elif self.attn_implementation == "flash_attention":
            device_type = next(self.parameters()).device.type
            if device_type == "cuda":
                with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    # cast to BF16 to use flash_attention
                    q = q.to(torch.bfloat16)
                    k = k.to(torch.bfloat16)
                    v = v.to(torch.bfloat16)
                    x = scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal, scale=self.scale)
                    # cast back to original dtype
                    x = x.to(k.dtype)
                    x = x.transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
            else:
                # Fall back to naive implementation for non-CUDA devices
                x = (q @ k.transpose(-2, -1)) * scale
                x = x.softmax(dim=-1)
                x = self.attn_drop(x)
                if k.dtype == torch.float32:  # if input was FP32, cast back to FP32
                    x = x.to(torch.float32)
                x = (x @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
        else:
            raise ValueError(f"Unknown attn_implementation: {self.attn_implementation}")

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rope=None,
        attn_implementation="pytorch_naive",
        attn_bias_for_inference_enabled=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_implementation=attn_implementation,
            attn_bias_for_inference_enabled=attn_bias_for_inference_enabled,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, attn_mask=None, is_causal=False, attn_implementation="pytorch_naive"
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.dropout_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

        self.attn_mask = attn_mask
        self.is_causal = is_causal
        self.attn_implementation = attn_implementation

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = (
            self.projq(query)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.projk(key)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.projv(value)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.rope is not None:
            with torch.autocast(device_type=next(self.parameters()).device.type, dtype=torch.float32):  # FIXME: for some reason Lightning didn't pick up torch.cuda.amp.custom_fwd when using bf16-true
                q = self.rope(q, qpos) if qpos is not None else q
                k = self.rope(k, kpos) if kpos is not None else k

        if self.attn_implementation == "pytorch_naive":
            assert self.attn_mask is None, "attn_mask not supported for pytorch_naive implementation of scaled dot product attention"
            assert self.is_causal is False, "is_causal not supported for pytorch_naive implementation of scaled dot product attention"
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif self.attn_implementation == "flash_attention":
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                # cast to BF16 to use flash_attention
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
                x = scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal, scale=self.scale)
                # cast back to FP32
                x = x.to(torch.float32)
                x = x.transpose(1, 2).reshape(B, Nq, C)
                x = self.proj(x)
                x = self.proj_drop(x)
        else:
            raise ValueError(f"Unknown attn_implementation: {self.attn_implementation}")

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
        attn_implementation="pytorch_naive",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_implementation=attn_implementation,
        )
        self.cross_attn = CrossAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_implementation=attn_implementation,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


# patch embedding
class PositionGetter(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class PatchEmbed(nn.Module):
    """just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        torch._assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
