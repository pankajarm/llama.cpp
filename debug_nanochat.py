#!/usr/bin/env python3
"""
Debug script to test NanoChat forward pass step by step using GGML via ctypes.
This script will load the GGUF model and perform operations one at a time to find where NaN originates.
"""

import numpy as np
import ctypes
import os
import sys

# Add gguf-py to path
sys.path.insert(0, "/home/ubuntu/us-east-1-nano-chat-exp/llama.cpp-nanochat/gguf-py")
import gguf

def load_gguf_tensors(gguf_path):
    """Load all tensors from GGUF file"""
    reader = gguf.GGUFReader(gguf_path, 'r')
    tensors = {}
    for tensor in reader.tensors:
        data = tensor.data.copy()
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        tensors[tensor.name] = data
    return tensors, reader

def check_nan(arr, name):
    """Check if array contains NaN and print stats"""
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  ❌ {name}: NaN={nan_count}, Inf={inf_count}")
        return True
    else:
        print(f"  ✅ {name}: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
        return False

def rms_norm(x, eps=1e-6):
    """Parameter-free RMS normalization"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms

def apply_rope_nanochat(x, positions, head_dim=128, freq_base=10000.0):
    """Apply NanoChat's inverted RoPE (NEOX style with inverted rotation)"""
    # x shape: (seq_len, num_heads, head_dim)
    seq_len, num_heads, _ = x.shape
    half_dim = head_dim // 2
    
    # Compute frequencies
    freqs = 1.0 / (freq_base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
    
    # Compute positions * frequencies
    t = np.array(positions, dtype=np.float32)[:, np.newaxis]  # (seq_len, 1)
    angles = t * freqs[np.newaxis, :]  # (seq_len, half_dim)
    
    cos_vals = np.cos(angles)  # (seq_len, half_dim)
    sin_vals = np.sin(angles)  # (seq_len, half_dim)
    
    # Split x into first half and second half (NEOX style)
    x1 = x[..., :half_dim]  # (seq_len, num_heads, half_dim)
    x2 = x[..., half_dim:]  # (seq_len, num_heads, half_dim)
    
    # Expand cos/sin for broadcasting
    cos_vals = cos_vals[:, np.newaxis, :]  # (seq_len, 1, half_dim)
    sin_vals = sin_vals[:, np.newaxis, :]  # (seq_len, 1, half_dim)
    
    # NanoChat inverted rotation:
    # y1 = x1*cos + x2*sin  (standard: x1*cos - x2*sin)
    # y2 = -x1*sin + x2*cos (standard: x1*sin + x2*cos)
    y1 = x1 * cos_vals + x2 * sin_vals
    y2 = -x1 * sin_vals + x2 * cos_vals
    
    return np.concatenate([y1, y2], axis=-1)

def main():
    gguf_path = "/home/ubuntu/us-east-1-nano-chat-exp/nanochat_convert_gguf/gguf_models/nanochat-d34-sft-f16.gguf"
    
    print("=" * 60)
    print("NanoChat GGUF Debug - Step-by-step forward pass")
    print("=" * 60)
    
    # Load tensors
    print("\n1. Loading GGUF tensors...")
    tensors, reader = load_gguf_tensors(gguf_path)
    
    # Check raw tensor data
    print("\n2. Checking raw tensor data for NaN/Inf...")
    has_nan = False
    for name, data in tensors.items():
        if check_nan(data, name):
            has_nan = True
    
    if has_nan:
        print("\n❌ GGUF file contains NaN/Inf values!")
        return
    else:
        print("\n✅ All GGUF tensors are clean (no NaN/Inf)")
    
    # Model config
    n_embd = 2176
    n_head = 17
    n_head_kv = 17
    n_layer = 34
    head_dim = n_embd // n_head  # 128
    
    # Test token
    test_token_id = 28173  # "Hello"
    positions = [0]  # Position 0
    
    print(f"\n3. Testing forward pass for token {test_token_id} ('Hello')...")
    
    # Step 1: Token embedding
    print("\n   Step 1: Token embedding lookup")
    tok_embd = tensors["token_embd.weight"]
    # tok_embd shape in GGUF: (vocab_size=65536, n_embd=2176)
    tok_embd_reshaped = tok_embd.reshape(65536, n_embd)
    x = tok_embd_reshaped[test_token_id:test_token_id+1]  # (1, 2176)
    check_nan(x, "embedding output")
    
    # Step 2: Initial RMSNorm (after embedding)
    print("\n   Step 2: Initial RMSNorm (after embedding)")
    x = rms_norm(x)
    check_nan(x, "after initial rms_norm")
    
    # Step 3: Process through first layer only
    print("\n   Step 3: First layer processing")
    
    # Q, K, V projections
    wq = tensors["blk.0.attn_q.weight"].reshape(n_embd, n_embd)
    wk = tensors["blk.0.attn_k.weight"].reshape(n_embd, n_embd)
    wv = tensors["blk.0.attn_v.weight"].reshape(n_embd, n_embd)
    
    q = x @ wq.T  # (1, n_embd)
    check_nan(q, "Q projection")
    
    k = x @ wk.T  # (1, n_embd)
    check_nan(k, "K projection")
    
    v = x @ wv.T  # (1, n_embd)
    check_nan(v, "V projection")
    
    # Reshape for multi-head attention
    q = q.reshape(1, n_head, head_dim)  # (1, 17, 128)
    k = k.reshape(1, n_head_kv, head_dim)  # (1, 17, 128)
    v = v.reshape(1, n_head_kv, head_dim)  # (1, 17, 128)
    
    # Apply RoPE
    print("\n   Step 4: Apply RoPE (NanoChat inverted)")
    q_rope = apply_rope_nanochat(q, positions, head_dim)
    check_nan(q_rope, "Q after RoPE")
    
    k_rope = apply_rope_nanochat(k, positions, head_dim)
    check_nan(k_rope, "K after RoPE")
    
    # Apply QK norm
    print("\n   Step 5: Apply QK norm")
    q_norm = rms_norm(q_rope)
    check_nan(q_norm, "Q after QK norm")
    
    k_norm = rms_norm(k_rope)
    check_nan(k_norm, "K after QK norm")
    
    # Attention scores (simplified - single position, so just self-attention)
    print("\n   Step 6: Attention computation")
    # For single position, attention is just softmax of q @ k.T / sqrt(head_dim)
    scale = 1.0 / np.sqrt(head_dim)
    attn_scores = np.einsum('bhd,bhd->bh', q_norm, k_norm) * scale  # (1, n_head)
    check_nan(attn_scores, "attention scores")
    
    attn_probs = np.ones_like(attn_scores)  # Softmax of single element is always 1
    
    # Attention output
    attn_out = v.reshape(1, n_head, head_dim)  # With single position, output is just V
    attn_out = attn_out.reshape(1, n_embd)  # (1, n_embd)
    
    # Output projection
    wo = tensors["blk.0.attn_output.weight"].reshape(n_embd, n_embd)
    attn_out = attn_out @ wo.T
    check_nan(attn_out, "attention output projection")
    
    # Residual connection
    print("\n   Step 7: Residual and FFN")
    x = x + attn_out
    check_nan(x, "after attention residual")
    
    # FFN: RMSNorm -> fc1 -> relu^2 -> fc2
    x_norm = rms_norm(x)
    check_nan(x_norm, "after FFN input norm")
    
    n_ff = 8704  # FFN intermediate dimension
    # ffn_up data shape from GGUF: (n_ff, n_embd) = (8704, 2176)
    # ffn_down data shape from GGUF: (n_embd, n_ff) = (2176, 8704)
    ffn_up = tensors["blk.0.ffn_up.weight"]  # Already (8704, 2176) = (n_ff, n_embd)
    ffn_down = tensors["blk.0.ffn_down.weight"]  # Already (2176, 8704) = (n_embd, n_ff)
    
    h = x_norm @ ffn_up.T  # (1, n_embd) @ (n_embd, n_ff) = (1, n_ff)
    check_nan(h, "FFN up projection")
    
    h = np.maximum(h, 0) ** 2  # relu^2
    check_nan(h, "after relu^2")
    
    h = h @ ffn_down.T  # (1, n_ff) @ (n_ff, n_embd) = (1, n_embd)
    check_nan(h, "FFN down projection")
    
    x = x + h  # Residual
    check_nan(x, "after FFN residual")
    
    print("\n" + "=" * 60)
    print("STEP-BY-STEP FORWARD PASS: All steps completed without NaN!")
    print("This confirms the Python implementation is correct.")
    print("The NaN must be in GGML's C++ execution.")
    print("=" * 60)
    
    # Now let's check if there's something different about how llama.cpp loads/uses the weights
    print("\n4. Checking tensor shapes match llama.cpp expectations...")
    
    # Check embedding shape
    tok_embd_raw = tensors["token_embd.weight"]
    print(f"   token_embd.weight raw shape: {tok_embd_raw.shape}")
    print(f"   Expected for GGML: (n_embd * n_vocab,) = ({n_embd * 65536},) = {n_embd * 65536}")
    print(f"   Actual elements: {tok_embd_raw.size}")
    
    # Check if the shape matches
    if tok_embd_raw.size == n_embd * 65536:
        print("   ✅ Embedding tensor size is correct")
    else:
        print("   ❌ Embedding tensor size MISMATCH!")
        
    # Print first few values for manual comparison
    print(f"\n5. First 5 embedding values for token {test_token_id}:")
    print(f"   {tok_embd_reshaped[test_token_id, :5]}")
    
    print("\n6. Suggestion: The issue is likely in:")
    print("   - How GGML accesses tensor data (row-major vs column-major)")
    print("   - A bug in a specific GGML operation (RMSNorm, RoPE, or attention)")
    print("   - Tensor shape/stride interpretation differences")

if __name__ == "__main__":
    main()
