module ESM
    using Flux
    using LinearAlgebra
    using Statistics
    using NeuralAttentionlib
    using NeuralAttentionlib.Masks

    export SwiGLUFFN, RotaryEmbedding, MultiHeadAttention
    export RegressionHead, UnifiedTransformerBlock, TransformerStack, TransformerOutput

    function silu(x::AbstractArray)
        return x .* σ.(x)
    end
    function swiglu(x::AbstractArray; feature_dim=1)
        x1, x2 = Flux.chunk(x, 2, dims=feature_dim)
        return silu(x1) .* x2
    end

    struct SwiGLUFFN
        norm::LayerNorm
        dense1::Dense
        dense2::Dense
    end
    function swiglu_correction_fn(expansion_ratio::Real, d_model::Int)
        return Int(((expansion_ratio * d_model) + 255) ÷ 256 * 256)
    end
    function SwiGLUFFN(d_model::Int, expansion_ratio::Real)
        corrected_dim = swiglu_correction_fn(expansion_ratio, d_model)

        return SwiGLUFFN(
            LayerNorm(d_model),
            Dense(d_model => corrected_dim * 2, bias=false),
            Dense(corrected_dim => d_model, bias=false)
        )
    end
    function (m::SwiGLUFFN)(x::A) where {A<:AbstractArray}
        x = m.norm(x)
        x = m.dense1(x)
        x = swiglu(x)
        return m.dense2(x)
    end
    Flux.@layer SwiGLUFFN

    function rotate_half(x::AbstractArray)
        # x is [d_head, n_heads, seq_len, batch]
        d_2 = size(x, 1) ÷ 2
        return vcat(-view(x, (d_2+1):size(x,1), :, :, : ),
                    view(x, 1:d_2, :, :, : ))
    end
    struct RotaryEmbedding
        dim::Int
        base::Float32
        inv_freq::Vector{Float32}
        max_seq_len::Int
        cos_cached::Matrix{Float32}
        sin_cached::Matrix{Float32}

        function RotaryEmbedding(dim::Int; base::Float32=10000f0, max_seq_len::Int=2048)
            @assert dim % 2 == 0 "Dimension must be even"
            inv_freq = 1f0 ./ (base .^ (Float32.(0:2:dim-1) ./ dim))
            t = Float32.(0:max_seq_len-1)
            freqs = t * inv_freq'  # [seq_len, dim/2]
            cos_cached = cos.(freqs)'  # [dim/2, seq_len]
            sin_cached = sin.(freqs)'  # [dim/2, seq_len]
            new(dim, base, inv_freq, max_seq_len, cos_cached, sin_cached)
        end
    end
    Flux.@layer RotaryEmbedding
    Flux.trainable(::RotaryEmbedding) = (;)
    function (re::RotaryEmbedding)(q::AbstractArray, k::AbstractArray)
        # q, k shapes: [d_head, n_heads, seq_len, batch]
        seq_len = size(q, 3)
        @assert seq_len <= re.max_seq_len "Sequence length exceeds maximum allowed"
        @assert size(q) == size(k) "Query and key shapes must match"
        d_head = size(q, 1)
        @assert d_head == re.dim "Input dimension must match RotaryEmbedding dimension"

        cos = @view re.cos_cached[:, 1:seq_len]  # [dim/2, seq_len]
        sin = @view re.sin_cached[:, 1:seq_len]  # [dim/2, seq_len]

        # Reshape for broadcasting
        # Add dimensions for n_heads and batch
        cos_exp = reshape(cos, d_head÷2, 1, seq_len, 1)  # [dim/2, 1, seq_len, 1]
        sin_exp = reshape(sin, d_head÷2, 1, seq_len, 1)  # [dim/2, 1, seq_len, 1]

        # Split input into two halves along d_head dimension
        q1, q2 = view(q, 1:d_head÷2, :, :, :), view(q, (d_head÷2+1):d_head, :, :, :)
        k1, k2 = view(k, 1:d_head÷2, :, :, :), view(k, (d_head÷2+1):d_head, :, :, :)

        q_new = vcat(q1 .* cos_exp .- q2 .* sin_exp,
                    q2 .* cos_exp .+ q1 .* sin_exp)
        k_new = vcat(k1 .* cos_exp .- k2 .* sin_exp,
                    k2 .* cos_exp .+ k1 .* sin_exp)

        return q_new, k_new
    end

    function qkv_attention(q, k, v)
        # Input is [d_head, heads, seq_len, batch]
        d_head, heads, seq_len1, batch = size(q)
        seq_len2 = size(k, 3)
        d_model = d_head * heads
        scale = 1 / sqrt(Float32(d_model))

        q_reshaped = reshape(permutedims(q, (3,1,2,4)), seq_len1, d_head, :)
        k_reshaped = reshape(permutedims(k, (3,1,2,4)), seq_len2, d_head, :)

        scores = scale * batched_mul(q_reshaped, permutedims(k_reshaped, (2,1,3)))
        attn_scores = softmax(reshape(scores, seq_len1, seq_len2, heads, batch), dims=2)
        v_reshaped = reshape(permutedims(v, (3,1,2,4)), seq_len2, d_head, :)
        context = batched_mul(reshape(attn_scores, seq_len1, seq_len2, :), v_reshaped)

        return reshape(context, seq_len1, d_head, heads, batch), attn_scores
    end

    struct MultiHeadAttention
        layernorm_qkv::Chain
        out_proj::Dense
        q_ln::LayerNorm
        k_ln::LayerNorm
        n_heads::Int
        rotary::RotaryEmbedding
    end
    Flux.@layer MultiHeadAttention
    """
        Multi-head attention mechanism with rotary embeddings.

    MultiHeadAttention(d_model::Int, n_heads::Int)

    # Arguments
    - `d_model::Int`: The input and output dimension of the model.
    - `n_heads::Int`: The number of attention heads to use.
    """
    function MultiHeadAttention(d_model::Int, n_heads::Int)
        layernorm_qkv = Chain(
            LayerNorm(d_model),
            Dense(d_model => 3d_model, bias=false)
        )
        out_proj = Dense(d_model => d_model, bias=false)
        q_ln = LayerNorm(d_model)
        k_ln = LayerNorm(d_model)
        d_head = d_model ÷ n_heads
        rotary = RotaryEmbedding(d_head)

        return MultiHeadAttention(layernorm_qkv, out_proj, q_ln, k_ln, n_heads, rotary)
    end
    function (mha::MultiHeadAttention)(
        x::AbstractArray;
        mask=nothing,
        return_attention::Bool=false
    )
        qkv = mha.layernorm_qkv(x)
        q, k, v = Flux.chunk(qkv, 3, dims=1)

        # Apply separate layer norms to Q and K
        q = mha.q_ln(q)
        k = mha.k_ln(k)

        # Reshape for rotary embeddings
        d_head = size(q, 1) ÷ mha.n_heads
        q_rotary = reshape(q, d_head, mha.n_heads, size(q, 2), size(q, 3))
        k_rotary = reshape(k, d_head, mha.n_heads, size(k, 2), size(k, 3))
        v_rotary = reshape(v, d_head, mha.n_heads, size(v, 2), size(v, 3)) # Not rotated but reshaped

        # Apply rotary embeddings
        q_rotary, k_rotary = mha.rotary(q_rotary, k_rotary)

        # Dot product attention
        attended, attn_weights = qkv_attention(q_rotary, k_rotary, v_rotary)
        output = mha.out_proj(reshape(permutedims(attended, (2,3,1,4)), size(q)))

        return return_attention ? (output, attn_weights) : output
    end

    # RegressionHead component
    struct RegressionHead
        linear1::Dense
        activation::Function
        norm::LayerNorm
        linear2::Dense
    end
    Flux.@layer RegressionHead
    function RegressionHead(d_model::Int, output_dim::Int; hidden_dim::Union{Int,Nothing}=nothing)
        hidden_dim = isnothing(hidden_dim) ? d_model : hidden_dim

        return RegressionHead(
            Dense(d_model => hidden_dim),
            gelu,
            LayerNorm(hidden_dim),
            Dense(hidden_dim => output_dim)
        )
    end
    function (m::RegressionHead)(x::AbstractArray)
        x = m.linear1(x)
        x = m.activation(x)
        x = m.norm(x)
        return m.linear2(x)
    end

    struct UnifiedTransformerBlock{F <: AbstractFloat}
        attn::MultiHeadAttention
        ffn::SwiGLUFFN
        scaling_factor::F
    end
    Flux.@layer UnifiedTransformerBlock
    function UnifiedTransformerBlock(
        d_model::Int,
        n_heads::Int;
        residue_scaling_factor::Float32=1.0f0,
        expansion_ratio::Float32=8.0f0/3.0f0
    )
        return UnifiedTransformerBlock(
            MultiHeadAttention(d_model, n_heads),
            SwiGLUFFN(d_model, expansion_ratio),
            residue_scaling_factor
        )
    end
    function (m::UnifiedTransformerBlock)(
        x::AbstractArray;
        mask=nothing,
        return_attention::Bool=false
    )
        if return_attention
            attn_output, attn_weights = m.attn(x; mask=mask, return_attention=true)
        else
            attn_output = m.attn(x; mask=mask, return_attention=false)
        end

        x = x + attn_output / m.scaling_factor
        x = x + m.ffn(x) / m.scaling_factor

        return return_attention ? (x, attn_weights) : x
    end

    struct TransformerStack
        blocks::Vector{UnifiedTransformerBlock}
        norm::LayerNorm
    end
    Flux.@layer TransformerStack
    function TransformerStack(
        d_model::Int,
        n_heads::Int,
        n_layers::Int
    )
        blocks = [
            UnifiedTransformerBlock(
                d_model,
                n_heads,
                residue_scaling_factor=Float32(sqrt(n_layers / 36))
            )
            for _ in 1:n_layers
        ]

        return TransformerStack(
            blocks,
            LayerNorm(d_model, affine=false)
        )
    end

    struct TransformerOutput
        last_hidden_state::AbstractArray
        hidden_states::Union{Nothing,Vector{AbstractArray}}
        attentions::Union{Nothing,Vector{AbstractArray}}
    end

    function (m::TransformerStack)(
        x::AbstractArray;
        attention_mask=nothing,
        output_hidden_states::Bool=false,
        output_attentions::Bool=false
    )
        hidden_states = output_hidden_states ? AbstractArray[] : nothing
        attentions = output_attentions ? AbstractArray[] : nothing

        for block in m.blocks
            if output_attentions
                x, attn_weights = block(x; mask=attention_mask, return_attention=true)
                if !isnothing(attentions)
                    push!(attentions, attn_weights)
                end
            else
                x = block(x; mask=attention_mask, return_attention=false)
            end

            if output_hidden_states && !isnothing(hidden_states)
                push!(hidden_states, x)
            end
        end

        x = m.norm(x)

        return TransformerOutput(x, hidden_states, attentions)
    end
end
