using ESM
using SafeTensors
using Test
using Statistics

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

@testset "ESM.jl" begin
    @testset "SwiGLUFFN" begin
        test_swiglu_layer()
    end
end
