using ESM
using Flux
using SafeTensors
using Test
using Statistics
using NeuralAttentionlib

function test_swiglu_layer()
    # Load test data
    test_data = load_safetensors("swiglu_test.safetensors")

    # Extract tensors
    x = permutedims(Array(test_data["input"]), (3,2,1))  # Convert to Julia's layout
    expected_output = permutedims(Array(test_data["output"]), (3,2,1))

    # Create layer with loaded weights
    d_model = size(x, 1)
    ffn = SwiGLUFFN(d_model, 2.0)

    # Load weights
    ffn.norm.diag.scale .= Array(test_data["ln.weight"])
    ffn.norm.diag.bias .= Array(test_data["ln.bias"])
    ffn.dense1.weight .= permutedims(Array(test_data["dense1.weight"]), (1,2))
    ffn.dense2.weight .= permutedims(Array(test_data["dense2.weight"]), (1,2))

    # Get output
    output = ffn(x)

    # Compare outputs
    rtol = 1e-4  # Relative tolerance
    atol = 1e-4  # Absolute tolerance

    @testset "SwiGLU Layer Tests" begin
        @test size(output) == size(expected_output)
        @test all(isapprox.(output, expected_output, rtol=rtol, atol=atol))

        # Print mean absolute error for debugging
        mae = mean(abs.(output - expected_output))
        println("Mean Absolute Error: $mae")
    end
end

"""
Convert from PyTorch BHSD format to Julia DHSB format.
B: batch, H: heads, S: sequence, D: dimension
"""
function torch_to_julia_heads(x::AbstractArray)
    permutedims(x, (4, 2, 3, 1))
end

"""
Convert from PyTorch BSD format to Julia DSB format.
B: batch, S: sequence, D: dimension
"""
function torch_to_julia_order(x::AbstractArray)
    permutedims(x, (3, 2, 1))
end


# Helper for dimension checking
function print_tensor_stats(name::String, tensor::AbstractArray)
    println("\nStats for $name:")
    println("  Shape: $(size(tensor))")
    println("  Mean: $(mean(tensor))")
    println("  Std: $(std(tensor))")
    println("  Min: $(minimum(tensor))")
    println("  Max: $(maximum(tensor))")
end

function compare_tensors(julia_tensor::AbstractArray, torch_tensor::AbstractArray, name::String; rtol=1f-4, atol=1f-4)
    print_tensor_stats("Julia $name", julia_tensor)
    print_tensor_stats("Torch $name", torch_tensor)

    diff = abs.(julia_tensor - torch_tensor)
    println("\nDifference stats for $name:")
    println("Max difference: $(maximum(diff))")
    println("Mean difference: $(mean(diff))")
    println("Std of difference: $(std(diff))")

    @test isapprox(julia_tensor, torch_tensor; rtol=rtol, atol=atol)
end

# Load test data
test_data = load_safetensors("mha_test.safetensors")

function test_multihead_attention()
    # Load and prepare input
    x_torch = Array(test_data["input"])  # [batch, seq_len, d_model]
    println("Input tensor shape (PyTorch): ", size(x_torch))

    x = torch_to_julia_order(x_torch)  # [d_model, seq_len, batch]
    println("Input tensor shape (Julia): ", size(x))

    d_model = size(x, 1)
    n_heads = 4

    mha = ESM.MultiHeadAttention(d_model, n_heads)

    # Load weights from PyTorch - keeping original shapes
    mha.layernorm_qkv[1].diag.scale .= Array(test_data["layernorm_qkv.0.weight"])
    mha.layernorm_qkv[1].diag.bias .= Array(test_data["layernorm_qkv.0.bias"])
    mha.layernorm_qkv[2].weight .= permutedims(Array(test_data["layernorm_qkv.1.weight"]), (1,2))
    mha.q_ln.diag.scale .= Array(test_data["q_ln.weight"])
    mha.k_ln.diag.scale .= Array(test_data["k_ln.weight"])
    mha.out_proj.weight .= permutedims(Array(test_data["out_proj.weight"]), (1,2))

    @testset "MHA Step-by-Step Validation" begin
        # 1. First LayerNorm
        x_ln = mha.layernorm_qkv[1](x)  # LayerNorm operates on d_model dimension
        println("\nAfter first layernorm shape: ", size(x_ln))

        # 2. QKV Projection
        qkv = mha.layernorm_qkv[2](x_ln)  # [3*d_model, seq_len, batch]
        println("After QKV projection shape: ", size(qkv))

        qkv_torch = Array(test_data["qkv_BLD3"])  # [batch, seq_len, 3*d_model]
        qkv_julia = torch_to_julia_order(qkv_torch)  # [3*d_model, seq_len, batch]

        println("\nQKV shapes comparison:")
        println("Julia computed: ", size(qkv))
        println("Torch expected: ", size(qkv_julia))

        compare_tensors(qkv, qkv_julia, "QKV projection")

        # 3. Split QKV
        q, k, v = Flux.chunk(qkv, 3, dims=1)  # Split along d_model dimension
        println("\nQuery/Key/Value shapes:")
        println("q: ", size(q))
        println("k: ", size(k))
        println("v: ", size(v))

        # 4. Q/K Layer Norms
        q = mha.q_ln(q)
        k = mha.k_ln(k)

        q_torch = Array(test_data["query_BLD_norm"])  # [batch, seq_len, d_model]
        k_torch = Array(test_data["key_BLD_norm"])  # [batch, seq_len, d_model]

        compare_tensors(q, torch_to_julia_order(q_torch), "Q after norm")
        compare_tensors(k, torch_to_julia_order(k_torch), "K after norm")

        # 5. Reshape for RoPE
        d_head = d_model ÷ n_heads
        println("\nReshaping for RoPE:")
        println("d_head: ", d_head)
        println("n_heads: ", n_heads)

        q_rotary = reshape(q, d_head, n_heads, size(q, 2), size(q, 3))  # [d_head, n_heads, seq_len, batch]
        k_rotary = reshape(k, d_head, n_heads, size(k, 2), size(k, 3))
        println("q_rotary shape: ", size(q_rotary))
        println("k_rotary shape: ", size(k_rotary))

        # 6. Apply RoPE
        q_rotary, k_rotary = mha.rotary(q_rotary, k_rotary)

        # Reshape back and compare with PyTorch
        q_rope = reshape(q_rotary, d_model, size(q, 2), size(q, 3))  # [d_model, seq_len, batch]
        k_rope = reshape(k_rotary, d_model, size(k, 2), size(k, 3))

        q_rope_torch = Array(test_data["query_BLD_rotary"])  # [batch, seq_len, d_model]
        k_rope_torch = Array(test_data["key_BLD_rotary"])  # [batch, seq_len, d_model]

        compare_tensors(q_rope, torch_to_julia_order(q_rope_torch), "Q after RoPE")
        compare_tensors(k_rope, torch_to_julia_order(k_rope_torch), "K after RoPE")

        # For now, stop here to check intermediate results
        return
    end
end

@testset "ESM.jl" begin
    @testset "SwiGLUFFN" begin
        test_swiglu_layer()
    end
end

@testset "MultiHeadAttention" begin
    test_multihead_attention()
end
