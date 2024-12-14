import torch
import safetensors.torch
from modeling_esm_plusplus import swiglu_ln_ffn

def test_swiglu_layer():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create test input
    batch_size = 2
    seq_len = 3
    d_model = 4
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize layer
    ffn = swiglu_ln_ffn(d_model, expansion_ratio=2.0)
    
    # Get output
    with torch.no_grad():
        output = ffn(x)
    
    # Save everything to safetensors
    tensors = {
        "input": x,
        "output": output,
        "ln.weight": ffn[0].weight,
        "ln.bias": ffn[0].bias,
        "dense1.weight": ffn[1].weight,
        "dense2.weight": ffn[3].weight
    }
    
    safetensors.torch.save_file(tensors, "swiglu_test.safetensors")

if __name__ == "__main__":
    test_swiglu_layer()

