using ESM
using Flux
using SafeTensors
using Test
using Statistics
using NeuralAttentionlib
using NeuralAttentionlib.Masks
using Zygote: gradient

function test_swiglu_layer()
    @testset "SwiGLU Layer" begin
        # Forward pass test with loaded weights
        test_data = load_safetensors("swiglu_test.safetensors")
        x = permutedims(Array(test_data["input"]), (3,2,1))  # Convert to Julia's layout
        expected_output = permutedims(Array(test_data["output"]), (3,2,1))

        d_model = size(x, 1)
        ffn = SwiGLUFFN(d_model, 2.0)

        # Load weights
        ffn.norm.diag.scale .= Array(test_data["ln.weight"])
        ffn.norm.diag.bias .= Array(test_data["ln.bias"])
        ffn.dense1.weight .= permutedims(Array(test_data["dense1.weight"]), (1,2))
        ffn.dense2.weight .= permutedims(Array(test_data["dense2.weight"]), (1,2))

        output = ffn(x)

        # Test forward pass
        @test size(output) == size(expected_output)
        @test all(isapprox.(output, expected_output, rtol=1f-4, atol=1f-4))

        # Test differentiability
        loss(m, x) = sum(abs2, m(x))
        gs = gradient(m -> loss(m, x), ffn)[1]

        @test !isnothing(gs.norm.diag.scale)
        @test !isnothing(gs.norm.diag.bias)
        @test !isnothing(gs.dense1.weight)
        @test !isnothing(gs.dense2.weight)
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

function test_rotate_half()
    @testset "rotate_half Tests" begin
        # Test with small array
        x = reshape(Float32.(1:16), (4, 2, 2, 1))  # [d_head, n_heads, seq_len, batch]
        rotated = ESM.rotate_half(x)

        # Check dimensions haven't changed
        @test size(rotated) == size(x)

        # Check that the rotation happened correctly
        # First half should be negated second half
        # Second half should be original first half
        d_2 = size(x, 1) ÷ 2
        @test rotated[1:d_2, :, :, :] == -x[(d_2+1):end, :, :, :]
        @test rotated[(d_2+1):end, :, :, :] == x[1:d_2, :, :, :]
    end
end

function test_rotary_embedding_trainable()
    @testset "RotaryEmbedding Trainable Parameters" begin
        re = RotaryEmbedding(64)
        # Should return empty NamedTuple
        @test isempty(Flux.trainable(re))
    end
end

function test_mha_with_mask_and_attention()
    @testset "MultiHeadAttention with Mask and Attention Return" begin
        # Setup dimensions
        d_model = 64
        n_heads = 4
        seq_len = 8
        batch_size = 2

        # Create input tensor [d_model, seq_len, batch]
        x = randn(Float32, d_model, seq_len, batch_size)

        # Create proper attention mask using NeuralAttentionlib
        mask = CausalMask()

        # Initialize MHA
        mha = ESM.MultiHeadAttention(d_model, n_heads)

        # Test without mask or attention return
        output1 = mha(x)
        @test size(output1) == (d_model, seq_len, batch_size)

        # Test with mask
        output2 = mha(x; mask=mask)
        @test size(output2) == (d_model, seq_len, batch_size)

        # Test with attention scores return
        output3, attention = mha(x; return_attention=true)
        @test size(output3) == (d_model, seq_len, batch_size)
        @test size(attention) == (seq_len, seq_len, n_heads, batch_size)

        # Test with both mask and attention return
        output4, attention_masked = mha(x; mask=mask, return_attention=true)
        @test size(output4) == (d_model, seq_len, batch_size)
        @test size(attention_masked) == (seq_len, seq_len, n_heads, batch_size)

        # Check that masked attention scores follow causal pattern
        mask_pattern = trues(seq_len, seq_len) .* mask  # Get the actual mask pattern

        # Test that all values where mask is false are approximately zero
        # This creates a single test that checks all masked positions at once
        @test all(abs.(attention_masked[.!mask_pattern, :, :]) .< 1e-6)
    end
end

function test_regression_head()
    @testset "RegressionHead" begin
        # Test dimensions
        d_model = 64
        output_dim = 10
        hidden_dim = 32
        batch_size = 2
        seq_len = 3

        # Create input tensor [d_model, seq_len, batch]
        x = randn(Float32, d_model, seq_len, batch_size)

        # Test with default hidden_dim
        head1 = ESM.RegressionHead(d_model, output_dim)
        out1 = head1(x)
        @test size(out1) == (output_dim, seq_len, batch_size)

        # Test with custom hidden_dim
        head2 = ESM.RegressionHead(d_model, output_dim, hidden_dim=hidden_dim)
        out2 = head2(x)
        @test size(out2) == (output_dim, seq_len, batch_size)
    end
end

function test_unified_transformer_block()
    @testset "UnifiedTransformerBlock" begin
        # Setup dimensions
        d_model = 64
        n_heads = 4
        seq_len = 8
        batch_size = 2

        # Create input tensor [d_model, seq_len, batch]
        x = randn(Float32, d_model, seq_len, batch_size)

        # Create block
        block = ESM.UnifiedTransformerBlock(d_model, n_heads)

        # Test without mask or attention return
        output1 = block(x)
        @test size(output1) == (d_model, seq_len, batch_size)

        # Test with mask
        mask = CausalMask()
        output2 = block(x, mask=mask)
        @test size(output2) == (d_model, seq_len, batch_size)

        # Test with attention return
        output3, attention = block(x, return_attention=true)
        @test size(output3) == (d_model, seq_len, batch_size)
        @test size(attention) == (seq_len, seq_len, n_heads, batch_size)
    end
end

function test_transformer_stack()
    @testset "TransformerStack" begin
        # Setup dimensions
        d_model = 64
        n_heads = 4
        n_layers = 3
        seq_len = 8
        batch_size = 2

        # Create input tensor [d_model, seq_len, batch]
        x = randn(Float32, d_model, seq_len, batch_size)

        # Create stack
        stack = ESM.TransformerStack(d_model, n_heads, n_layers)

        # Test basic forward pass
        output = stack(x)
        @test size(output.last_hidden_state) == (d_model, seq_len, batch_size)
        @test isnothing(output.hidden_states)
        @test isnothing(output.attentions)

        # Test with hidden states collection
        output = stack(x, output_hidden_states=true)
        @test size(output.last_hidden_state) == (d_model, seq_len, batch_size)
        @test length(output.hidden_states) == n_layers
        @test all(size(hs) == (d_model, seq_len, batch_size) for hs in output.hidden_states)

        # Test with attention weights collection
        output = stack(x, output_attentions=true)
        @test size(output.last_hidden_state) == (d_model, seq_len, batch_size)
        @test length(output.attentions) == n_layers
        @test all(size(attn) == (seq_len, seq_len, n_heads, batch_size) for attn in output.attentions)
    end
end

@testset "SwiGLUFFN" begin
    test_swiglu_layer()
end

@testset "RotaryEmbedding" begin
    test_rotate_half()
    test_rotary_embedding_trainable()
end

@testset "MultiHeadAttention" begin
    test_multihead_attention()
    test_mha_with_mask_and_attention()
end

@testset "RegressionHead" begin
    test_regression_head()
end

@testset "UnifiedTransformerBlock" begin
    test_unified_transformer_block()
end

@testset "TransformerStack" begin
    test_transformer_stack()
end
