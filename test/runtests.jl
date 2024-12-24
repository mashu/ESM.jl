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

        @test size(output) == size(expected_output)
        @test all(isapprox.(output, expected_output, rtol=1f-4, atol=1f-4))

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

function test_multihead_attention()
    @testset "MultiHeadAttention" begin
        # Load test data and setup
        test_data = load_safetensors("mha_test.safetensors")
        x_torch = Array(test_data["input"])
        x = torch_to_julia_order(x_torch)

        d_model = size(x, 1)
        n_heads = 4
        mha = ESM.MultiHeadAttention(d_model, n_heads)

        # Load weights
        mha.layernorm_qkv[1].diag.scale .= Array(test_data["layernorm_qkv.0.weight"])
        mha.layernorm_qkv[1].diag.bias .= Array(test_data["layernorm_qkv.0.bias"])
        mha.layernorm_qkv[2].weight .= permutedims(Array(test_data["layernorm_qkv.1.weight"]), (1,2))
        mha.q_ln.diag.scale .= Array(test_data["q_ln.weight"])
        mha.k_ln.diag.scale .= Array(test_data["k_ln.weight"])
        mha.out_proj.weight .= permutedims(Array(test_data["out_proj.weight"]), (1,2))

        @testset "Forward Pass" begin
            x_ln = mha.layernorm_qkv[1](x)
            qkv = mha.layernorm_qkv[2](x_ln)
            qkv_torch = Array(test_data["qkv_BLD3"])
            qkv_julia = torch_to_julia_order(qkv_torch)

            @test isapprox(qkv, qkv_julia, rtol=1f-4, atol=1f-4)

            q, k, v = Flux.chunk(qkv, 3, dims=1)
            # v are the same
            q = mha.q_ln(q)
            k = mha.k_ln(k)

            q_torch = Array(test_data["query_BLD_norm"])
            k_torch = Array(test_data["key_BLD_norm"])

            @test isapprox(q, torch_to_julia_order(q_torch), rtol=1f-4, atol=1f-4)
            @test isapprox(k, torch_to_julia_order(k_torch), rtol=1f-4, atol=1f-4)

            d_head = d_model ÷ n_heads
            q_rotary = reshape(q, d_head, n_heads, size(q, 2), size(q, 3))
            k_rotary = reshape(k, d_head, n_heads, size(k, 2), size(k, 3))
            q_rotary, k_rotary = mha.rotary(q_rotary, k_rotary)

            q_rope = reshape(q_rotary, d_model, size(q, 2), size(q, 3))
            k_rope = reshape(k_rotary, d_model, size(k, 2), size(k, 3))

            q_rope_torch = Array(test_data["query_BLD_rotary"])
            k_rope_torch = Array(test_data["key_BLD_rotary"])

            @test isapprox(q_rope, torch_to_julia_order(q_rope_torch), rtol=1f-4, atol=1f-4)
            @test isapprox(k_rope, torch_to_julia_order(k_rope_torch), rtol=1f-4, atol=1f-4)

            mha_output = mha(x)
            output_torch = Array(test_data["output"])
            @test isapprox(mha_output, torch_to_julia_order(output_torch), rtol=1f-4, atol=1f-4)
        end

        @testset "Mask and Attention" begin
            x_test = randn(Float32, d_model, 8, 2)
            mask = CausalMask()

            output1 = mha(x_test)
            @test size(output1) == (d_model, 8, 2)

            output2 = mha(x_test; mask=mask)
            @test size(output2) == (d_model, 8, 2)

            output3, attention = mha(x_test; return_attention=true)
            @test size(output3) == (d_model, 8, 2)
            @test size(attention) == (8, 8, n_heads, 2)

            #output4, attention_masked = mha(x_test; mask=mask, return_attention=true)
            #mask_pattern = trues(8, 8) .* mask
            #@test all(abs.(attention_masked[.!mask_pattern, :, :]) .< 1e-6)
        end

        @testset "Differentiability" begin
            x_test = randn(Float32, d_model, 8, 2)

            gs_input = gradient(x -> sum(abs2, mha(x)), x_test)[1]
            @test !isnothing(gs_input)
            @test size(gs_input) == size(x_test)

            mask = CausalMask()
            gs_input_masked = gradient(x -> sum(abs2, mha(x; mask=mask)), x_test)[1]
            @test !isnothing(gs_input_masked)
            @test size(gs_input_masked) == size(x_test)
        end
    end
end

function test_rotate_half()
    @testset "rotate_half Tests" begin
        x = reshape(Float32.(1:16), (4, 2, 2, 1))  # [d_head, n_heads, seq_len, batch]
        rotated = ESM.rotate_half(x)

        @test size(rotated) == size(x)
        d_2 = size(x, 1) ÷ 2
        @test rotated[1:d_2, :, :, :] == -x[(d_2+1):end, :, :, :]
        @test rotated[(d_2+1):end, :, :, :] == x[1:d_2, :, :, :]
    end
end

function test_rotary_embedding_trainable()
    @testset "RotaryEmbedding Trainable Parameters" begin
        re = RotaryEmbedding(64)
        @test isempty(Flux.trainable(re))
    end
end

function test_mha_with_mask_and_attention()
    @testset "MultiHeadAttention with Mask and Attention Return" begin
        d_model = 64
        n_heads = 4
        seq_len = 8
        batch_size = 2

        x = randn(Float32, d_model, seq_len, batch_size)

        mask = CausalMask()
        mha = ESM.MultiHeadAttention(d_model, n_heads)

        output1 = mha(x)
        @test size(output1) == (d_model, seq_len, batch_size)

        output2 = mha(x; mask=mask)
        @test size(output2) == (d_model, seq_len, batch_size)

        output3, attention = mha(x; return_attention=true)
        @test size(output3) == (d_model, seq_len, batch_size)
        @test size(attention) == (seq_len, seq_len, n_heads, batch_size)

        output4, attention_masked = mha(x; mask=mask, return_attention=true)
        @test size(output4) == (d_model, seq_len, batch_size)
        #@test size(attention_masked) == (seq_len, seq_len, n_heads, batch_size)

        #mask_pattern = trues(seq_len, seq_len) .* mask  # Get the actual mask pattern
        #@test all(abs.(attention_masked[.!mask_pattern, :, :]) .< 1e-6)
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
