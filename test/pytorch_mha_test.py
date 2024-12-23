import torch
import safetensors.torch
from modeling_esm_plusplus import MultiHeadAttention
import math

def verify_tensor(name: str, tensor: torch.Tensor):
    """Print tensor stats"""
    print(f"\nStats for {name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")

def clone_and_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Make tensor both cloned and contiguous"""
    return tensor.clone().contiguous()

def test_multihead_attention():
    """Test MultiHeadAttention by following original implementation exactly"""
    torch.manual_seed(42)
    
    # Model dimensions
    batch_size = 2
    seq_len = 3
    n_heads = 4
    d_head = 4
    d_model = n_heads * d_head
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    verify_tensor("input", x)
    
    # Create mask exactly as in ESM++
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    attention_mask = attention_mask.triu_(1).bool()
    
    # Initialize model
    mha = MultiHeadAttention(d_model, n_heads)
    tensors_to_save = {}
    
    with torch.no_grad():
        # 1. Get output directly from MHA for reference
        output_ref, attn_weights_ref = mha(x, attention_mask=attention_mask, output_attentions=True)
        verify_tensor("output_ref", output_ref)
        tensors_to_save["output_ref"] = clone_and_contiguous(output_ref)
        
        # 2. Now follow original implementation exactly
        qkv_BLD3 = mha.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        verify_tensor("qkv_BLD3", qkv_BLD3)
        tensors_to_save["qkv_BLD3"] = clone_and_contiguous(qkv_BLD3)
        
        query_BLD, key_BLD = (
            mha.q_ln(query_BLD).to(query_BLD.dtype),
            mha.k_ln(key_BLD).to(query_BLD.dtype),
        )
        verify_tensor("query_BLD after norm", query_BLD)
        verify_tensor("key_BLD after norm", key_BLD)
        tensors_to_save["query_BLD_norm"] = clone_and_contiguous(query_BLD)
        tensors_to_save["key_BLD_norm"] = clone_and_contiguous(key_BLD)
        
        # Apply rotary
        query_BLD, key_BLD = mha._apply_rotary(query_BLD, key_BLD)
        verify_tensor("query_BLD after rotary", query_BLD)
        verify_tensor("key_BLD after rotary", key_BLD)
        tensors_to_save["query_BLD_rotary"] = clone_and_contiguous(query_BLD)
        tensors_to_save["key_BLD_rotary"] = clone_and_contiguous(key_BLD)
        
        # Reshape
        query_BHLD, key_BHLD, value_BHLD = map(mha.reshaper, (query_BLD, key_BLD, value_BLD))
        verify_tensor("query_BHLD", query_BHLD)
        verify_tensor("key_BHLD", key_BHLD)
        verify_tensor("value_BHLD", value_BHLD)
        tensors_to_save["query_BHLD"] = clone_and_contiguous(query_BHLD)
        tensors_to_save["key_BHLD"] = clone_and_contiguous(key_BHLD)
        tensors_to_save["value_BHLD"] = clone_and_contiguous(value_BHLD)
        
        # Manual attention computation - exactly as in original
        L, S = query_BLD.size(-2), key_BLD.size(-2)
        scale = 1 / math.sqrt(query_BLD.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query_BLD.dtype, device=query_BLD.device)
        verify_tensor("attn_bias", attn_bias)
        tensors_to_save["attn_bias"] = clone_and_contiguous(attn_bias)
            
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_mask.masked_fill_(attention_mask.logical_not(), float('-inf'))
            else:
                attn_bias += attention_mask
        
        # Clone intermediate attention weights to avoid memory sharing
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * scale
        verify_tensor("raw attn_weights", attn_weights)
        tensors_to_save["raw_attn_weights"] = clone_and_contiguous(attn_weights)
        
        attn_weights = attn_weights + attn_bias  # Create new tensor instead of inplace
        verify_tensor("biased attn_weights", attn_weights)
        tensors_to_save["biased_attn_weights"] = clone_and_contiguous(attn_weights)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        verify_tensor("final attn_weights", attn_weights)
        tensors_to_save["final_attn_weights"] = clone_and_contiguous(attn_weights)
        
        context_BHLD = torch.matmul(attn_weights, value_BHLD)
        verify_tensor("context_BHLD", context_BHLD)
        tensors_to_save["context_BHLD"] = clone_and_contiguous(context_BHLD)
        
        context_BLD = mha.reshaper(context_BHLD, pattern="b h s d -> b s (h d)")
        verify_tensor("context_BLD", context_BLD)
        tensors_to_save["context_BLD"] = clone_and_contiguous(context_BLD)
        
        output = mha.out_proj(context_BLD)
        verify_tensor("output", output)
        tensors_to_save["output"] = clone_and_contiguous(output)
        
        # Compare with reference
        diff = (output - output_ref).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nImplementation comparison:")
        print(f"Max difference: {max_diff:.8f}")
        print(f"Mean difference: {mean_diff:.8f}")
        print(f"Positions with diff > 1e-5: {(diff > 1e-5).sum().item()}")
        
        if max_diff >= 1e-5:
            print("\nLARGE DIFFERENCE DETECTED - SAVING TENSORS FOR ANALYSIS")
        else:
            print("\nOutputs match!")
            
        # Save model weights too
        tensors_to_save.update({
            "input": clone_and_contiguous(x),
            "attention_mask": clone_and_contiguous(attention_mask),
            "layernorm_qkv.0.weight": clone_and_contiguous(mha.layernorm_qkv[0].weight),
            "layernorm_qkv.0.bias": clone_and_contiguous(mha.layernorm_qkv[0].bias),
            "layernorm_qkv.1.weight": clone_and_contiguous(mha.layernorm_qkv[1].weight),
            "q_ln.weight": clone_and_contiguous(mha.q_ln.weight),
            "k_ln.weight": clone_and_contiguous(mha.k_ln.weight),
            "out_proj.weight": clone_and_contiguous(mha.out_proj.weight)
        })
        
        safetensors.torch.save_file(tensors_to_save, "mha_test.safetensors")

if __name__ == "__main__":
    test_multihead_attention()

