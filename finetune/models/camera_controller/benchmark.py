import torch
# from sparse_linear_attention import SparseLinearAttention
# import flash_attn
import flash_attn_interface
from triton.testing import do_bench
# from vsa import block_sparse_fwd, block_sparse_bwd
from fastvideo_kernel.block_sparse_attn import _get_sm90_ops
block_sparse_fwd, block_sparse_bwd = _get_sm90_ops()
from vsa import BLOCK_M, BLOCK_N
import math

def generate_block_sparse_pattern(bs, h, num_q_blocks, num_kv_blocks, k, device="cuda"):
    """
    Generate a block sparse pattern where each q block attends to exactly k kv blocks.
    
    Args:
        bs: batch size
        h: number of heads
        num_q_blocks: number of query blocks
        num_kv_blocks: number of key-value blocks
        k: number of kv blocks each q block attends to
        device: device to create tensors on
        
    Returns:
        q2k_block_sparse_index: [bs, h, num_q_blocks, k]
            Contains the indices of kv blocks that each q block attends to.
        q2k_block_sparse_num: [bs, h, num_q_blocks]
            Contains the number of kv blocks that each q block attends to (all equal to k).
        k2q_block_sparse_index: [bs, h, num_kv_blocks, num_q_blocks]
            Contains the indices of q blocks that attend to each kv block.
        k2q_block_sparse_num: [bs, h, num_kv_blocks]
            Contains the number of q blocks that attend to each kv block.
        block_sparse_mask: [bs, h, num_q_blocks, num_kv_blocks]
            Binary mask where 1 indicates attention connection.
    """
    # Ensure k is not larger than num_kv_blocks
    k = min(k, num_kv_blocks)
    
    # Create random scores for sampling
    scores = torch.rand(bs, h, num_q_blocks, num_kv_blocks, device=device)
    
    # Get top-k indices for each q block
    _, q2k_block_sparse_index = torch.topk(scores, k, dim=-1)
    q2k_block_sparse_index = q2k_block_sparse_index.to(torch.int32)
    
    # sort q2k_block_sparse_index
    q2k_block_sparse_index, _ = torch.sort(q2k_block_sparse_index, dim=-1)

    # All q blocks attend to exactly k kv blocks
    q2k_block_sparse_num = torch.full((bs, h, num_q_blocks), k, dtype=torch.int32, device=device)
    
    # Create the corresponding mask
    block_sparse_mask = torch.zeros(bs, h, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
    
    # Fill in the mask based on the indices
    for b in range(bs):
        for head in range(h):
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx]
                block_sparse_mask[b, head, q_idx, kv_indices] = True
    
    # Create the reverse mapping (k2q)
    # First, initialize lists to collect q indices for each kv block
    k2q_indices_list = [[[] for _ in range(num_kv_blocks)] for _ in range(bs * h)]
    
    # Populate the lists based on q2k mapping
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx].tolist()
                for kv_idx in kv_indices:
                    k2q_indices_list[flat_idx][kv_idx].append(q_idx)
    
    # Find the maximum number of q blocks that attend to any kv block
    max_q_per_kv = 0
    for flat_idx in range(bs * h):
        for kv_idx in range(num_kv_blocks):
            max_q_per_kv = max(max_q_per_kv, len(k2q_indices_list[flat_idx][kv_idx]))
    
    # Create tensors for k2q mapping
    k2q_block_sparse_index = torch.full((bs, h, num_kv_blocks, max_q_per_kv), -1, 
                                        dtype=torch.int32, device=device)
    k2q_block_sparse_num = torch.zeros((bs, h, num_kv_blocks), 
                                       dtype=torch.int32, device=device)
    
    # Fill the tensors
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for kv_idx in range(num_kv_blocks):
                q_indices = k2q_indices_list[flat_idx][kv_idx]
                num_q = len(q_indices)
                k2q_block_sparse_num[b, head, kv_idx] = num_q
                if num_q > 0:
                    k2q_block_sparse_index[b, head, kv_idx, :num_q] = torch.tensor(
                        q_indices, dtype=torch.int32, device=device)
                
    return q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, block_sparse_mask

def main():
    sparsity=0.0
    # attn = SparseLinearAttention(
    #     head_dim=128,
    #     topk=1-sparsity,                 # = 1 - sparsity
    #     feature_map="softmax",    # options: elu, relu, softmax
    #     BLKQ=64,
    #     BLKK=64,
    # ).cuda()

    B, H, L, D = 1, 12, 40960, 16
    k_seq_ratio = 1
    q = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((B, H, L * k_seq_ratio, D), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((B, H, L * k_seq_ratio, D), dtype=torch.bfloat16, device='cuda')

    print(f'********************* Test On H20 *********************')
    print(f'********************* (B,H,L,D), Q.shape=({B},{H},{L},{D}), K.shape=({B},{H},{L*k_seq_ratio},{D})*********************')
    print(f'********************* sparsity={sparsity} *********************')

    # # SLA
    # torch.cuda.synchronize()
    # fwd_time = do_bench(
    #     lambda: attn(q, k, v), # BHLD
    #     warmup=5,
    #     rep=20,
    #     quantiles=None
    # )
    # print(f"SLA - Time: {fwd_time:.2f}")

    # VSA forward
    num_q_blocks = q.shape[2] // 64
    num_kv_blocks = k.shape[2] // 64
    topk = math.ceil((1 - sparsity) * num_kv_blocks)
    q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num, _ = generate_block_sparse_pattern(q.shape[0], q.shape[1], num_q_blocks, num_kv_blocks, topk, device="cuda")
    variable_block_sizes = torch.ones(num_kv_blocks, device=q.device).int() * BLOCK_M
    torch.cuda.synchronize()
    fwd_time = do_bench(
        lambda: block_sparse_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, variable_block_sizes),
        warmup=5,
        rep=20,
        quantiles=None
    )
    print(f"VSA Forward - Time: {fwd_time:.2f}")

    # VSA backward
    o, l_vec = block_sparse_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, variable_block_sizes)
    vsa_fwd_output = o.clone()
    grad_output = torch.randn_like(o)
    torch.cuda.synchronize()
    bwd_time = do_bench(
        lambda: block_sparse_bwd(q, k, v, o, l_vec, grad_output, k2q_block_sparse_index, k2q_block_sparse_num, variable_block_sizes),
        warmup=5,
        rep=20,
        quantiles=None
    )
    print(f"VSA Backward - Time: {bwd_time:.2f}")


    q, k, v = q.permute(0,2,1,3).contiguous(), k.permute(0,2,1,3).contiguous(), v.permute(0,2,1,3).contiguous() # B,L,H,D
    # flash2_output = flash_attn.flash_attn_func(q, k, v).clone()
    # # flash2
    # torch.cuda.synchronize()
    # fwd_time = do_bench(
    #     lambda: flash_attn.flash_attn_func(q, k, v), # BLHD
    #     warmup=5,
    #     rep=20,
    #     quantiles=None
    # )
    # print(f"Flash Attention 2 - Time: {fwd_time:.2f}")

    # flash3
    flash3_output = flash_attn_interface.flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    fwd_time = do_bench(
        lambda: flash_attn_interface.flash_attn_func(q, k, v), # BLHD
        warmup=5,
        rep=20,
        quantiles=None
    )
    print(f"Flash Attention 3 - Time: {fwd_time:.2f}")

    print(f'flash3_output.shape={flash3_output.shape}')
    # print(f'flash2_output.shape={flash2_output.shape}')
    print(f'vsa_fwd_output.shape={vsa_fwd_output.transpose(1,2).shape}')
    print(f'flash3 vs. vsa_fwd_output diff.abs.min={(flash3_output - vsa_fwd_output.transpose(1,2)).abs().min()}')
    print(f'flash3 vs. vsa_fwd_output diff.abs.max={(flash3_output - vsa_fwd_output.transpose(1,2)).abs().max()}')
    # print(f'flash3 vs. flash2 diff.abs.min={(flash3_output - flash2_output).abs().min()}')
    # print(f'flash3 vs. flash2 diff.abs.max={(flash3_output - flash2_output).abs().max()}')