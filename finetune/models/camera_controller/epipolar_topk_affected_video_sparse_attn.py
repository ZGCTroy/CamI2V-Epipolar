# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import torch
import random

from typing import Any
from typing import Tuple, Optional  
from einops import rearrange
block_sparse_attn=None
major, minor = torch.cuda.get_device_capability(0)
if major >= 9 and minor == 0:# check if H100
    # from vsa_cuda import block_sparse_fwd, block_sparse_bwd
    # from vsa.block_sparse_wrapper import block_sparse_attn_SM90
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    # block_sparse_fwd, block_sparse_bwd = _get_sm90_ops()
else:
    from vsa.block_sparse_wrapper import block_sparse_attn_triton
    block_sparse_fwd = None
    block_sparse_bwd = None
    block_sparse_attn = block_sparse_attn_triton

import flash_attn_interface

def get_attention_score(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        qkv: BHLD
        epipolar_mask_weight: B,L,L
    '''
    scale_factor = q.new_tensor(q.size(-1)**-0.5)  # 保持与q相同的类型
    QK = torch.matmul(q, k.transpose(-2, -1)) * scale_factor # BHLD x BHDL -> BHLL

    attention_output = flash_attn_interface.flash_attn_func(
        q.bfloat16().transpose(1,2).contiguous(),  # qkv: BLHD
        k.bfloat16().transpose(1,2).contiguous(), 
        v.bfloat16().transpose(1,2).contiguous()
    ).transpose(1,2) # BHLD
    
    return attention_output, QK

# @torch.jit.script
def epipolar_topk_affected_video_sparse_attn(
        q, k, v, epipolar_mask_weight: torch.Tensor,
        block_size: Tuple[int, int, int] = (1, 8, 8),
        topk: int = 64,
        coarse_gate: Optional[torch.Tensor] = None, fine_gate: Optional[torch.Tensor] = None,
        gate_type='tanh_gate',
    ):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    NOTE: We assume q, k, v is zero padded!!
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """
    f, h, w = 11, 48, 80
    rearrange_op = lambda x: rearrange(
        x,
        'b head (num_repeat f_blocks block_f h_blocks block_h w_blocks block_w) d -> b head (num_repeat f_blocks h_blocks w_blocks block_f block_h block_w) d',
        num_repeat=1,
        f_blocks=f // block_size[0],
        block_f=block_size[0],
        h_blocks=h // block_size[1],
        block_h=block_size[1],
        w_blocks=w // block_size[2],
        block_w=block_size[2],
    ).contiguous()
    q, k, v, coarse_gate, fine_gate = map(rearrange_op, (q, k, v, coarse_gate, fine_gate))  # BHLD
    
    dtype = q.dtype
    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements == 64
    assert q.shape[2] % block_elements == 0
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    
    k_seq_len = k.shape[2]
    num_kv_blocks = k_seq_len // block_elements
    q_compress = q.view(batch_size, num_heads, q_seq_len // block_elements, block_elements, head_dim).bfloat16().mean(dim=3).to(dtype) # Q = [new_chunk_Q]
    k_compress = k.view(batch_size, num_heads, k_seq_len // block_elements, block_elements, head_dim).bfloat16().mean(dim=3).to(dtype) # K = [history_chunk_K, new_chunk_K]
    v_compress = v.view(batch_size, num_heads, k_seq_len // block_elements, block_elements, head_dim).bfloat16().mean(dim=3).to(dtype)

    output_compress, block_attn_score = get_attention_score(q_compress, k_compress, v_compress)
    output_compress = output_compress.view(batch_size, num_heads, q_seq_len // block_elements, 1, head_dim)
    output_compress = output_compress.repeat(1, 1, 1, block_elements, 1).view(batch_size, num_heads, q_seq_len, head_dim)
    
    topk = min(topk, num_kv_blocks)

    block_attn_score = block_attn_score.float().softmax(dim=-1) * epipolar_mask_weight.unsqueeze(1) # B,H,L,L
    topK_indices = torch.topk(block_attn_score, topk, dim=-1, sorted=False).indices
    block_mask = torch.zeros_like(block_attn_score, dtype=torch.bool).scatter_(-1, topK_indices, True) # BHLL
    variable_block_sizes = torch.full((block_mask.shape[3],), block_elements, device=block_mask.device, dtype=torch.int32)
    output_select = []
    for batch_idx in range(batch_size):
        output_select.append(
            block_sparse_attn(
                q[batch_idx:batch_idx+1], k[batch_idx:batch_idx+1], v[batch_idx:batch_idx+1], 
                block_mask[batch_idx:batch_idx+1], variable_block_sizes
            )[0]
        )
    output_select = torch.cat(output_select, dim=0)
    # output_select = block_sparse_attn(
    #     q.contiguous(), k.contiguous(), v.contiguous(),
    #     block_mask, variable_block_sizes
    # )[0]
    # flash3_output_select = flash_attn_interface.flash_attn_func(
    #     q.transpose(1,2).contiguous(),  # qkv: BLHD
    #     k.transpose(1,2).contiguous(), 
    #     v.transpose(1,2).contiguous()
    # ).transpose(1,2) # BHLD

    if gate_type in ['tanh_gate']:
        # final_output = output_compress.float() * coarse_gate.float() + output_select.float() * fine_gate.float() # BHLD
        # if torch.distributed.get_rank() == 0:
        #     print(f'q.shape={q.shape} k.shape={k.shape} v.shape={v.shape} coarse_gate.shape={coarse_gate.shape} fine_gate.shape={fine_gate.shape}')
        #     print(f'q.min={q.min()} q.mean={q.mean()} q.max={q.max()}')
        #     print(f'k.min={k.min()} k.mean={k.mean()} k.max={k.max()}')
        #     print(f'v.min={v.min()} v.mean={v.mean()} v.max={v.max()}')
        #     print(f'coarse_gate.min={coarse_gate.min()} coarse_gate.mean={coarse_gate.mean()} coarse_gate.max={coarse_gate.max()}')
        #     print(f'fine_gate.min={fine_gate.min()} fine_gate.mean={fine_gate.mean()} fine_gate.max={fine_gate.max()}')
        #     print(f'dtype q={q.dtype} k={k.dtype} v={v.dtype} coarse_gate={coarse_gate.dtype} fine_gate={fine_gate.dtype}')
        #     print(f'topk={topk} block_mask.shape={block_mask.shape} block_mask.sum={block_mask.sum()} block_mask.max={block_mask.max()} output_compress.max={output_compress.max()} coarse_gate.max={coarse_gate.max()} fine_gate.max={fine_gate.max()}')
        #     flash3_output_select = flash_attn_interface.flash_attn_func(
        #         q.transpose(1,2).contiguous(),  # qkv: BHLD -> BLHD
        #         k.transpose(1,2).contiguous(), 
        #         v.transpose(1,2).contiguous()
        #     ).transpose(1,2) # BHLD
        #     diff = torch.abs(flash3_output_select - output_select)
        #     print(f'output_select.min={output_select.min()} output_select.mean={output_select.mean()} output_select.max={output_select.max()}')
        #     print(f'flash3_output_select.min={flash3_output_select.min()} flash3_output_select.mean={flash3_output_select.mean()} flash3_output_select.max={flash3_output_select.max()}')
        #     print(f'f3 vs. vsa max diff: max={diff.max()} mean={diff.mean()} min={diff.min()}')
        #     print(f'************')

        final_output = output_compress.float() * coarse_gate.float() + output_select.float() * fine_gate.float() # BHLD
    else:
        raise ValueError(f"gate_type {gate_type} not supported")

    final_output = rearrange(
        final_output,
        'b head (num_repeat f_blocks h_blocks w_blocks block_f block_h block_w) d -> b head (num_repeat f_blocks block_f h_blocks block_h w_blocks block_w) d',
        num_repeat=1,
        f_blocks=f // block_size[0],
        block_f=block_size[0],
        h_blocks=h // block_size[1],
        block_h=block_size[1],
        w_blocks=w // block_size[2],
        block_w=block_size[2],
    )
    return final_output.to(dtype) #BHLD
    