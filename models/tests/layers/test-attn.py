#!/usr/bin/env python
import pytest
import torch
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.utils import load_layer_kernels
import sys

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device():
    """Get available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def precision():
    #return torch.bfloat16 # triton attn fails bf16
    return torch.float16
    #return torch.float32

@pytest.fixture
def attention_params():
    """Basic attention parameters for single-head attention"""
    return {
        "num_heads": 1,
        "embed_dim": 64,
        "qkv_bias": False,
        "qk_norm": False,
        "is_causal": False,
        "window_size": None,
        "dropout_p": 0.0,
        "softcap": None,
        "use_alibi_slopes": False,
        "use_rotary_embeddings": False,
    }


@pytest.fixture
def layer_kernels():
    """Create minimal layer_kernels DotDict"""
    return load_layer_kernels()

class TestAttentionEquivalence:
    """Test equivalence between two Attention implementations"""
    
    @pytest.mark.parametrize("embed_dim", [64])
    @pytest.mark.parametrize("seq_len", [2048])
    @pytest.mark.parametrize("window_size", [None, 512])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash attention requires CUDA")
    def test_forward(
        self, random_seed, device, precision, layer_kernels, embed_dim, seq_len, window_size
    ):
        """Test equivalence across different embedding dimensions and sequence lengths"""
        batch_size = 2

        if precision == torch.bfloat16:
            rtol=5e-2
            atol=5e-3
        elif precision == torch.float16:
            rtol=0.0
            atol=1e-2
        elif precision == torch.float32:
            rtol=0.0
            atol=1e-2
        else:
            raise ValueError(f"Unknown precision: {precision}")
        
        attention_params = {
            "num_heads": 1,
            "embed_dim": embed_dim,
            "qkv_bias": False,
            "qk_norm": False,
            "is_causal": False,
            "window_size": window_size,
            "dropout_p": 0.0,
            "softcap": None,
            "use_alibi_slopes": False,
            "use_rotary_embeddings": False,
        }
        
        x = torch.randn(batch_size * seq_len, embed_dim, device=device, dtype=precision)
        print(f"{x.shape=}")
        
        ref_attn = MultiHeadSelfAttention(
            **attention_params,
            layer_kernels=layer_kernels,
            attention_implementation="flex_attention",
            _compile=False,
        ).to(device).to(precision)
        
        test_attn = MultiHeadSelfAttention(
            **attention_params,
            layer_kernels=layer_kernels,
            attention_implementation="triton_attention",
        ).to(device).to(precision)
        
        test_attn.load_state_dict(ref_attn.state_dict())
        
        ref_attn.eval()
        test_attn.eval()
        
        with torch.no_grad():
            output_ref = ref_attn(x, [seq_len], batch_size)
            output_test = test_attn(x, [seq_len], batch_size)
       
        torch.testing.assert_close(
            output_ref,
            output_test,
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.parametrize("embed_dim", [64])
    @pytest.mark.parametrize("seq_len", [2048])
    @pytest.mark.parametrize("window_size", [None, 512])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash attention requires CUDA")
    def test_backward(
        self, random_seed, device, precision, layer_kernels, embed_dim, seq_len, window_size
    ):
        """Test equivalence across different embedding dimensions and sequence lengths"""
        batch_size = 2

        attention_params = {
            "num_heads": 1,
            "embed_dim": embed_dim,
            "qkv_bias": False,
            "qk_norm": False,
            "is_causal": False,
            "window_size": window_size,
            "dropout_p": 0.0,
            "softcap": None,
            "use_alibi_slopes": False,
            "use_rotary_embeddings": False,
        }

        if precision == torch.bfloat16:
            rtol=5e-2
            atol=5e-3
        elif precision == torch.float16:
            rtol=0.0
            atol=1e-2
        elif precision == torch.float32:
            rtol=0.0
            atol=1e-2
        else:
            raise ValueError(f"Unknown precision: {precision}")

        x = torch.randn(batch_size * seq_len, embed_dim, device=device, dtype=precision)
        x_clone = x.clone().detach().requires_grad_(True)

        ref_attn = MultiHeadSelfAttention(
            **attention_params,
            layer_kernels=layer_kernels,
            attention_implementation="flex_attention",
            _compile=False,
        ).to(device).to(precision)

        test_attn = MultiHeadSelfAttention(
            **attention_params,
            layer_kernels=layer_kernels,
            attention_implementation="triton_attention",
        ).to(device).to(precision)

        test_attn.load_state_dict(ref_attn.state_dict())

        ref_attn.train()
        test_attn.train()

        # Forward pass
        output_ref = ref_attn(x, [seq_len], batch_size)
        output_test = test_attn(x_clone, [seq_len], batch_size)
        
        # Loss
        torch.manual_seed(42)
        target = torch.randn_like(output_ref)
        loss_ref = torch.nn.functional.mse_loss(output_ref, target)
        loss_test = torch.nn.functional.mse_loss(output_test, target)
        
        # Backward pass
        loss_ref.backward()
        loss_test.backward()

        torch.testing.assert_close(
            output_ref,
            output_test,
            rtol=rtol,
            atol=atol,
        )

        # Compare parameter gradients
        ref_params = dict(ref_attn.named_parameters())
        test_params = dict(test_attn.named_parameters())

        for name in ref_params.keys():
            ref_grad = ref_params[name].grad
            test_grad = test_params[name].grad
            
            assert ref_grad is not None and test_grad is not None, f"Gradient not computed for {name}"
            try:
                torch.testing.assert_close(
                    ref_grad,
                    test_grad,
                    rtol=rtol,
                    atol=atol,
                    msg=f"Parameter gradient for {name} differs beyond tolerance (dtype={precision})"
                )
            except AssertionError as e:
                # Print diagnostic info for debugging
                diff = torch.abs(ref_grad - test_grad)
                rel_diff = diff / (torch.abs(ref_grad) + 1e-8)
                print(f"\nParameter gradient comparison for {name} (dtype={precision}):")
                print(f"  Max abs diff: {diff.max().item():.6e}")
                print(f"  Mean abs diff: {diff.mean().item():.6e}")
                print(f"  Max rel diff: {rel_diff.max().item():.6e}")
                print(f"  Mean rel diff: {rel_diff.mean().item():.6e}")
                print(f"  Gradient shape: {ref_grad.shape}")
                raise

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s",])
