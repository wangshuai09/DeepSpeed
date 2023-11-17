# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.op_builder import InferenceBuilder

from deepspeed.ops.transformer.inference.op_binding.softmax_context import SoftmaxContextOp


def test1():
    
    BATCH= 1
    H = 12  # heads
    N_CTX = 16  # sequence length
    D_HEAD = 64
    heads = 64
    norm_factor = 0.1
    num_kv = 64
    no_masking = True
    layer_id = 1
    num_layers = 10
    alibi = None

  
    class Config:
        dtype = torch.float16
        rotary_dim = 64 
        rotate_half = True
        rotate_every_two = True
        triangular_masking = True
        local_attention = True
        window_size = 1
        rope_theta =  1000
    
    config = Config()

    softmax_context_op = SoftmaxContextOp(config)

    q = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=config.dtype, device="cuda").normal_(mean=0, std=.5)
    k = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=config.dtype, device="cuda").normal_(mean=0, std=.5)
    v = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=config.dtype, device="cuda").normal_(mean=0, std=.5)

    # adjust it to expected tensor format and run test
    qkv = torch.randn((BATCH, N_CTX, 3 * H * D_HEAD), dtype=config.dtype, device='cuda:0', requires_grad=False)
    qkv[:, :, :H * D_HEAD] = q.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))
    qkv[:, :, 1 * H * D_HEAD:2 * H * D_HEAD] = k.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))
    qkv[:, :, 2 * H * D_HEAD:] = v.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))

    attn_mask = torch.zeros((BATCH, H, N_CTX, N_CTX), dtype=config.dtype, device="cuda:0")

    out = softmax_context_op(query_key_value=qkv, attn_mask=attn_mask, heads=heads, num_kv=num_kv,
                norm_factor=norm_factor, no_masking=no_masking, layer_id=layer_id, num_layers=num_layers, alibi=alibi)

test1()