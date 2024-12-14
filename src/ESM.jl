module ESM
    using Flux
    using NeuralAttentionlib
    using SafeTensors

    export SwiGLUFFN

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

end

