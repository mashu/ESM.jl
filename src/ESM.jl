module ESM
    using Flux
    using LinearAlgebra
    using Statistics
    using NeuralAttentionlib
    using NeuralAttentionlib.Masks

    export SwiGLUFFN, RotaryEmbedding, MultiHeadAttention

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

    struct MultiHeadAttention
        layernorm_qkv::Chain
        out_proj::Dense
        q_ln::LayerNorm
        k_ln::LayerNorm
        n_heads::Int
        rotary::RotaryEmbedding
    end
    Flux.@layer MultiHeadAttention
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

        # Apply rotary embeddings
        q_rotary, k_rotary = mha.rotary(q_rotary, k_rotary)

        # Reshape back
        q = reshape(q_rotary, :, size(q, 2), size(q, 3))
        k = reshape(k_rotary, :, size(k, 2), size(k, 3))

        if return_attention
            context, score = NeuralAttentionlib.multihead_qkv_attention(
                NeuralAttentionlib.score_returning,
                mha.n_heads,
                q, k, v,
                mask,
                nothing  # dropout
            )
        else
            context = NeuralAttentionlib.multihead_qkv_attention(
                mha.n_heads,
                q, k, v,
                mask,
                nothing  # dropout
            )
        end

        output = mha.out_proj(context)

        return return_attention ? (output, score) : output
    end
end
