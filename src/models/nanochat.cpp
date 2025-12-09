#include "models.h"

// NanoChat architecture implementation
// Key features:
// - Parameter-free RMSNorm (no learnable norm weights)
// - 2-layer MLP with relu2 activation: relu(x).square()
// - QK norm AFTER RoPE application
// - Final logit softcapping (15.0)
// - Token embedding norm
// - INVERTED RoPE rotation (clockwise instead of counterclockwise)
//   NanoChat uses: y1 = x1*cos + x2*sin, y2 = -x1*sin + x2*cos
//   This is equivalent to standard RoPE with negated positions

llm_build_nanochat::llm_build_nanochat(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Token embeddings
    inpL = build_inp_embd(model.tok_embd);

    // NanoChat: RMSNorm immediately after token embeddings (parameter-free)
    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    cb(inpL, "tok_norm", -1);

    // Position input for RoPE
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Pre-attention RMSNorm (parameter-free)
        cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
        cb(cur, "attn_norm", il);

        // Self-attention
        {
            // Q, K, V projections
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // Apply RoPE FIRST (before QK norm)
            // NanoChat uses:
            // 1. NEOX-style rotation: dims split into first/second half (not adjacent pairs)
            //    d = head_dim // 2
            //    x1, x2 = x[..., :d], x[..., d:]  # first half, second half
            // 2. INVERTED rotation direction:
            //    y1 = x1*cos + x2*sin   (standard uses: x1*cos - x2*sin)
            //    y2 = -x1*sin + x2*cos  (standard uses: x1*sin + x2*cos)
            // 
            // Using NEOX mode with negated freq_scale achieves both:
            // - NEOX splits dims correctly
            // - Negative freq_scale negates the angle, inverting the rotation
            const float nanochat_freq_scale = -freq_scale;  // Negate to invert rotation
            const int nanochat_rope_type = GGML_ROPE_TYPE_NEOX;  // Use NEOX-style dim splitting
            
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, nanochat_rope_type, n_ctx_orig, freq_base, nanochat_freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, nanochat_rope_type, n_ctx_orig, freq_base, nanochat_freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            // QK norm AFTER RoPE (parameter-free RMSNorm)
            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            cb(Qcur, "Qcur_normed", il);

            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            cb(Kcur, "Kcur_normed", il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr,  // no bias
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Pre-FFN RMSNorm (parameter-free)
        cur = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);
        cb(cur, "ffn_norm", il);

        // FFN: 2-layer MLP with relu2 (LLM_FFN_RELU_SQR)
        // out = ffn_down(relu2(ffn_up(x))) where relu2(x) = relu(x).square()
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,  // up projection
                nullptr,                   nullptr, nullptr,  // no gate (not used for 2-layer MLP)
                model.layers[il].ffn_down, nullptr, nullptr,  // down projection
                nullptr,
                LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    cur = inpL;

    // Final RMSNorm (parameter-free)
    cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    // Final logit softcapping: scale * tanh(logits / scale)
    // NanoChat uses scale = 15.0
    if (hparams.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
