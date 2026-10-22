/* V4.1's CPU reference follows the scalar GPU graph one token at a time.
 * In particular, compression ownership, delayed HC mixing and activation
 * rounding are model semantics, rather than optional backend optimizations. */
#include "ds4_deepseek41_cpu_math.inc"

#if defined(__clang__)
#pragma float_control(precise, on, push)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#endif

#include "ds4_deepseek41_cpu_x86.inc"

/* DS4_CPU_PROFILE splits each token into phases. ds41_prof(next) charges the
 * main thread's wall time since the previous switch to the running phase, so
 * pool work is counted where it is waited for. Bytes are the weights read. */
enum {
    DS41_PROF_GLUE, DS41_PROF_EREAD, DS41_PROF_ENGRAM, DS41_PROF_HC,
    DS41_PROF_Q, DS41_PROF_KV, DS41_PROF_COMP, DS41_PROF_INDEX, DS41_PROF_STAGE,
    DS41_PROF_ATTN, DS41_PROF_OUT, DS41_PROF_ROUTER, DS41_PROF_SHARED,
    DS41_PROF_ROUTED, DS41_PROF_HEAD, DS41_PROF_N
};
static const char *const ds41_prof_name[DS41_PROF_N] = {
    "glue", "engram_read", "engram", "hc", "attn_q", "attn_kv", "compress",
    "indexer", "stage", "attention", "attn_out", "router", "shared", "routed",
    "head"
};
static bool ds41_prof_on;
static int ds41_prof_phase;
static double ds41_prof_t, ds41_prof_sec[DS41_PROF_N];
static uint64_t ds41_prof_bytes[DS41_PROF_N];

static void ds41_prof(int phase) {
    if (!ds41_prof_on) return;
    const double t = now_sec();
    ds41_prof_sec[ds41_prof_phase] += t - ds41_prof_t;
    ds41_prof_t = t;
    ds41_prof_phase = phase;
}

static bool ds41_cpu_matvec(float *out, const ds4_model *m,
                           const ds4_tensor *weight, const float *in, bool round) {
    if (!out || !m || !weight || !in || weight->ndim < 2 ||
        !weight->dim[0] || !weight->dim[1] || weight->dim[1] > UINT32_MAX)
        return false;
    /* A group of the output projection is a matrix slice of a 3D tensor.
     * The F32 Q8 reference derives its row count from elements, not dim[1]. */
    ds4_tensor matrix = *weight;
    matrix.ndim = 2;
    matrix.elements = matrix.dim[0] * matrix.dim[1];
    switch (matrix.type) {
    case DS4_TENSOR_F32: {
        /* The 384-row router is below ds4_parallel_for's 512-row floor. */
        matvec_f32_ctx ctx = {.out = out, .data = tensor_data(m, &matrix),
                              .x = in, .in_dim = matrix.dim[0]};
        ds4_parallel_for_min_rows(matrix.dim[1], matvec_f32_worker, &ctx, 2);
        break;
    }
    case DS4_TENSOR_F16: matvec_f16(out, m, &matrix, in); break;
    case DS4_TENSOR_Q8_0: ds41_cpu_matvec_q8(out, m, &matrix, in); break;
    default: return false;
    }
    if (ds41_prof_on) ds41_prof_bytes[ds41_prof_phase] += matrix.bytes;
    if (round) ds41_cpu_bf16_rows(out, (size_t)matrix.dim[1]);
    return true;
}

static bool ds41_cpu_norm(float *out, const float *in, const ds4_model *m,
                         const ds4_tensor *weight) {
    if (!out || !in || !m || !weight || weight->type != DS4_TENSOR_F32 ||
        !weight->dim[0]) return false;
    rms_norm_weight(out, in, tensor_data(m, weight), weight->dim[0], DS4_RMS_EPS);
    ds41_cpu_bf16_rows(out, (size_t)weight->dim[0]);
    return true;
}

#include "ds4_deepseek41_cpu_attention.inc"

typedef struct ds41_cpu_state {
    const ds4_model *model;
    const ds4_weights *weights;
    uint32_t ctx, pos;
    bool valid;
    ds41_cpu_attention_state *attention;
    ds4_engram_layout engram;
    ds4_engram_history history;
    ds4_engram_table table[2];
    ds4_engram_reader *reader;
    uint32_t *token_map;
    float rows[2][DS4_ENGRAM_COLS * DS4_ENGRAM_DIM];
    float engram_kv[5 * 5120];
    float residual[4 * 5120], after_attn[4 * 5120], flat_norm[4 * 5120];
    float pre[4], mix[24], attn_split[24], ffn_split[24];
    float x[5120], norm[5120], block[5120], shared[5120];
    float shared_gate[2304], shared_up[2304], shared_mid[2304];
    float expert_mid[6 * 2304];
} ds41_cpu_state;

static ds4_context_memory ds41_cpu_memory(uint32_t ctx) {
    ds4_context_memory memory = {.prefill_cap = 1, .raw_cap = 128,
                                 .comp_cap = ctx + 1u};
    memory.raw_bytes = 40u * 128u * 512u * sizeof(float);
    for (uint32_t i = 0; i < 4; i++) {
        const uint64_t cap = (uint64_t)ctx / (i < 3 ? 2u : 1u) + 1u;
        memory.compressed_bytes += cap * (512u + 128u) * sizeof(float);
    }
    const uint64_t blocks = ((uint64_t)ctx + 7u) / 8u;
    memory.scratch_bytes = sizeof(ds41_cpu_state) + sizeof(ds41_cpu_attention_state) +
        (uint64_t)DS4_N_VOCAB * (sizeof(uint32_t) + 2u * sizeof(float)) +
        ((uint64_t)ctx + 1u) * sizeof(float) + blocks * (sizeof(float) + sizeof(uint8_t));
    memory.total_bytes = memory.raw_bytes + memory.compressed_bytes + memory.scratch_bytes;
    return memory;
}

static bool ds41_cpu_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
    return false;
}

static bool ds41_cpu_read_array(const ds4_model *m, const char *key,
                               uint32_t type, uint64_t count,
                               void *out, size_t bytes) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) || arr.type != type || arr.len != count)
        return false;
    ds4_cursor c = cursor_at(m, arr.data_pos);
    return cursor_read(&c, out, bytes);
}

static void ds41_cpu_free(ds41_cpu_state *s) {
    if (!s) return;
    ds4_engram_reader_free(s->reader);
    ds4_engram_table_close(&s->table[0]);
    ds4_engram_table_close(&s->table[1]);
    ds41_cpu_attention_free(s->attention);
    free(s->token_map);
    free(s);
}

static void ds41_cpu_reset(ds41_cpu_state *s) {
    if (!s) return;
    s->pos = 0;
    s->valid = true;
    ds4_engram_history_reset(&s->history);
    ds41_cpu_attention_reset(s->attention);
    memset(s->residual, 0, sizeof(s->residual));
    memset(s->pre, 0, sizeof(s->pre));
}

static ds41_cpu_state *ds41_cpu_create(const ds4_model *m, const ds4_weights *w,
                                     const char *path, uint32_t ctx,
                                     char *error, size_t cap) {
    if (!m || !w || !path || !ctx || ctx > 1048576u ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK41 ||
        DS4_N_EMBD != 5120 || DS4_N_LAYER != 40 || DS4_N_HC != 4 ||
        DS4_N_EXPERT_USED != 6 || DS4_N_FF_EXP != 2304) {
        ds41_cpu_error(error, cap, "unsupported V4.1 CPU model shape or context");
        return NULL;
    }
    ds41_cpu_state *s = calloc(1, sizeof(*s));
    if (!s) {
        ds41_cpu_error(error, cap, "cannot allocate V4.1 CPU state");
        return NULL;
    }
    s->table[0].fd = s->table[1].fd = -1;
    s->model = m;
    s->weights = w;
    s->ctx = ctx;
    s->token_map = malloc((size_t)DS4_N_VOCAB * sizeof(*s->token_map));
    s->attention = ds41_cpu_attention_create(ctx);
    if (!s->token_map || !s->attention) {
        ds41_cpu_error(error, cap, "cannot allocate V4.1 CPU caches");
        goto fail;
    }
    s->engram.token_map = s->token_map;
    s->engram.vocab_size = DS4_N_VOCAB;
    if (!model_get_u32(m, "deepseek41.engram.compressed_vocab_size", &s->engram.compressed_vocab_size) ||
        !model_get_u32(m, "deepseek41.engram.pad_id", &s->engram.pad_id) ||
        !ds41_cpu_read_array(m, "deepseek41.engram.token_map", GGUF_VALUE_UINT32,
                            DS4_N_VOCAB, s->token_map, (size_t)DS4_N_VOCAB * 4) ||
        !ds41_cpu_read_array(m, "deepseek41.engram.rows", GGUF_VALUE_UINT32,
                            2, s->engram.rows, sizeof(s->engram.rows)) ||
        !ds41_cpu_read_array(m, "deepseek41.engram.primes", GGUF_VALUE_UINT32,
                            48, s->engram.primes, sizeof(s->engram.primes)) ||
        !ds41_cpu_read_array(m, "deepseek41.engram.multipliers", GGUF_VALUE_UINT64,
                            8, s->engram.multipliers, sizeof(s->engram.multipliers)) ||
        !ds4_engram_layout_valid(&s->engram)) {
        ds41_cpu_error(error, cap, "invalid V4.1 Engram metadata");
        goto fail;
    }
    for (uint32_t i = 0; i < 2; i++) {
        const uint32_t il = i ? 14u : 1u;
        const ds4_tensor *table = required_tensorf(m, "blk.%u.engram_embd.weight", il);
        if (!ds4_engram_table_open(&s->table[i], path, table->abs_offset, s->engram.rows[i])) {
            ds41_cpu_error(error, cap, "cannot open V4.1 Engram table");
            goto fail;
        }
        /* Offsets are valid only for the file used by the weight mapping. */
        struct stat weights_stat, rows_stat;
        if (fstat(m->fd, &weights_stat) || fstat(s->table[i].fd, &rows_stat) ||
            weights_stat.st_dev != rows_stat.st_dev || weights_stat.st_ino != rows_stat.st_ino) {
            ds41_cpu_error(error, cap, "V4.1 Engram file differs from model mapping");
            goto fail;
        }
    }
    /* Fresh text reads cold rows, 0.3-0.8 ms each here (ZFS reads a 64 KiB
     * record per 264-byte row). Twelve threads, two columns each, have layer
     * 1's table ready in ~1.4 ms, under layer 0, and both tables in ~2 ms
     * (four threads: 2.0-2.6 and 3.7-5 ms; serial: 7-11 and 16-22 ms).
     * Creation failure falls back to synchronous reads. */
    s->reader = ds4_engram_reader_create(12);
    ds41_cpu_reset(s);
    return s;
fail:
    ds41_cpu_free(s);
    return NULL;
}

static bool ds41_cpu_hc_mix(ds41_cpu_state *s, const ds4_layer_weights *l, bool ffn) {
    const ds4_tensor *fn = ffn ? l->hc_ffn_fn : l->hc_attn_fn;
    const ds4_tensor *scale = ffn ? l->hc_ffn_scale : l->hc_attn_scale;
    const ds4_tensor *base = ffn ? l->hc_ffn_base : l->hc_attn_base;
    if (!scale || !base || scale->type != DS4_TENSOR_F32 || base->type != DS4_TENSOR_F32)
        return false;
    rms_norm_no_weight(s->flat_norm, ffn ? s->after_attn : s->residual,
                       4u * 5120u, DS4_RMS_EPS);
    if (!ds41_cpu_matvec(s->mix, s->model, fn, s->flat_norm, false)) return false;
    hc_split_sinkhorn_one(ffn ? s->ffn_split : s->attn_split, s->mix,
                         tensor_data(s->model, scale), tensor_data(s->model, base),
                         4, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS);
    return true;
}

static void ds41_cpu_hc_sum(float *out, const float *residual, const float *pre) {
    hc_weighted_sum_one(out, residual, pre, 5120, 4);
    ds41_cpu_bf16_rows(out, 5120);
}

static void ds41_cpu_hc_expand(float *out, const float *block,
                              const float *residual, const float *split) {
    hc_post_one(out, block, residual, split + 4, split + 8, 5120, 4);
    ds41_cpu_bf16_rows(out, 4u * 5120u);
}

static bool ds41_cpu_moe(ds41_cpu_state *s, const ds4_layer_weights *l) {
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED], probs[DS4_MAX_EXPERT];
    if (!l->ffn_exp_probs_b || !l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps ||
        !glm_graph_gate_pair_type_supported(l->ffn_gate_exps->type, l->ffn_up_exps->type) ||
        !glm_graph_down_type_supported(l->ffn_down_exps->type)) return false;
    ds41_prof(DS41_PROF_ROUTER);
    if (!ds41_cpu_matvec(probs, s->model, l->ffn_gate_inp, s->norm, false)) return false;
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++)
        probs[i] = sqrtf(softplus_stable(probs[i]));
    layer_topk_selected_experts_from_probs(selected, expert_weight, s->model, l, probs);
    ds41_prof(DS41_PROF_SHARED);
    if (l->ffn_gate_shexp->elements == l->ffn_gate_shexp->dim[0] * 2304u &&
        ds41_cpu_matvec_q8_pair(s->shared_gate, s->shared_up, s->model,
                                l->ffn_gate_shexp, l->ffn_up_shexp, s->norm)) {
        if (ds41_prof_on)
            ds41_prof_bytes[DS41_PROF_SHARED] += l->ffn_gate_shexp->bytes + l->ffn_up_shexp->bytes;
        ds41_cpu_bf16_rows(s->shared_gate, 2304);
        ds41_cpu_bf16_rows(s->shared_up, 2304);
    } else if (!ds41_cpu_matvec(s->shared_gate, s->model, l->ffn_gate_shexp, s->norm, true) ||
               !ds41_cpu_matvec(s->shared_up, s->model, l->ffn_up_shexp, s->norm, true)) {
        return false;
    }
    swiglu(s->shared_mid, s->shared_gate, s->shared_up, 2304, DS4_SWIGLU_CLAMP_EXP);
    ds41_cpu_bf16_rows(s->shared_mid, 2304);
    if (!ds41_cpu_matvec(s->shared, s->model, l->ffn_down_shexp, s->shared_mid, true)) return false;
    /* The quantized routed-expert kernels retain F32 intermediates and apply
     * routing weights before down projection. Shared experts round to BF16. */
    ds41_prof(DS41_PROF_ROUTED);
    if (ds41_prof_on)
        ds41_prof_bytes[DS41_PROF_ROUTED] += DS4_N_EXPERT_USED *
            ((l->ffn_gate_exps->bytes + l->ffn_up_exps->bytes +
              l->ffn_down_exps->bytes) / DS4_N_EXPERT);
    ds41_cpu_routed_moe(s->block, s->expert_mid, s->model, l,
                       s->norm, selected, expert_weight);
    for (uint32_t i = 0; i < 5120; i++) s->block[i] = ds41_cpu_bf16(s->block[i] + s->shared[i]);
    ds41_prof(DS41_PROF_HC);
    return true;
}

static bool ds41_cpu_logits(ds41_cpu_state *s, float *logits) {
    if (!s || !s->valid || !logits) return false;
    ds41_cpu_hc_sum(s->x, s->residual, s->pre);
    return ds41_cpu_norm(s->norm, s->x, s->model, s->weights->output_norm) &&
        ds41_cpu_matvec(logits, s->model, s->weights->output, s->norm, false);
}

static bool ds41_cpu_forward(ds41_cpu_state *s, int token, float *logits,
                            char *error, size_t cap) {
    if (!s || !s->valid) return ds41_cpu_error(error, cap, "invalid V4.1 CPU state; reset required");
    if (s->pos >= s->ctx) return ds41_cpu_error(error, cap, "V4.1 CPU context is full");
    if (token < 0 || (uint32_t)token >= DS4_N_VOCAB)
        return ds41_cpu_error(error, cap, "V4.1 CPU token is outside vocabulary");
    const bool profile = getenv("DS4_CPU_PROFILE") != NULL;
    const double started = profile ? now_sec() : 0;
    const uint64_t dispatches = g_pool_dispatches;
    ds41_prof_on = profile;
    ds41_prof_t = started;
    ds41_prof_phase = DS41_PROF_EREAD;
    memset(ds41_prof_sec, 0, sizeof(ds41_prof_sec));
    memset(ds41_prof_bytes, 0, sizeof(ds41_prof_bytes));
    double engram_seconds = 0, attention_seconds = 0, moe_seconds = 0;
    uint32_t ids[2][DS4_ENGRAM_COLS];
    ds4_engram_history next_history = s->history;
    if (!ds4_engram_hash(&s->engram, &next_history, &token, NULL, 1, &ids[0][0]))
        return ds41_cpu_error(error, cap, "V4.1 Engram hash failed");
    /* The rows are read while layer 0 runs; layers 1 and 14 wait for their
     * own table. The bytes and every use of them are unchanged. */
    if (s->reader) {
        if (!ds4_engram_reader_start(s->reader, s->table, ids, &s->rows[0][0]))
            return ds41_cpu_error(error, cap, "V4.1 Engram row read failed");
    } else {
        for (uint32_t i = 0; i < 2; i++) {
            if (!ds4_engram_read(&s->table[i], ids[i], DS4_ENGRAM_COLS, s->rows[i]))
                return ds41_cpu_error(error, cap, "V4.1 Engram row read failed");
        }
    }
    double read_seconds = profile ? now_sec() - started : 0;
    ds41_prof(DS41_PROF_GLUE);
    embed_token_any(s->model, s->weights, token, s->x);
    hc_from_plain_embedding(s->residual, s->x, 5120, 4);
    s->pre[0] = 1.0f;
    s->pre[1] = s->pre[2] = s->pre[3] = 0.0f;
    for (uint32_t il = 0; il < 40; il++) {
        const ds4_layer_weights *l = &s->weights->layer[il];
        if (il == 1 || il == 14) {
            const uint32_t i = il == 1 ? 0u : 1u;
            if (s->reader) {
                ds41_prof(DS41_PROF_EREAD);
                const double wait = profile ? now_sec() : 0;
                if (!ds4_engram_reader_wait(s->reader, i)) {
                    (void)ds4_engram_reader_wait(s->reader, 1);
                    s->valid = false;
                    return ds41_cpu_error(error, cap, "V4.1 Engram row read failed");
                }
                if (profile) read_seconds += now_sec() - wait;
            }
            const double phase = profile ? now_sec() : 0;
            ds41_prof(DS41_PROF_ENGRAM);
            if (!ds41_cpu_matvec(s->engram_kv, s->model, l->engram_kv, s->rows[i], true))
                goto layer_fail;
            ds41_cpu_engram_add(s->residual, s->engram_kv,
                               tensor_data(s->model, l->engram_q_norm),
                               tensor_data(s->model, l->engram_k_norm), 5120, DS4_RMS_EPS);
            if (profile) engram_seconds += now_sec() - phase;
        }
        ds41_prof(DS41_PROF_HC);
        if (!ds41_cpu_hc_mix(s, l, false)) goto layer_fail;
        /* The newly computed attention mixer is consumed by the next FFN.
         * Attention itself consumes the preceding layer's FFN preweights. */
        ds41_cpu_hc_sum(s->x, s->residual, s->pre);
        if (!ds41_cpu_norm(s->norm, s->x, s->model, l->attn_norm)) goto layer_fail;
        double phase = profile ? now_sec() : 0;
        if (!ds41_cpu_attention_forward(s->attention, s->model, l, il, s->pos, s->norm, s->block))
            goto layer_fail;
        if (profile) attention_seconds += now_sec() - phase;
        ds41_prof(DS41_PROF_HC);
        ds41_cpu_hc_expand(s->after_attn, s->block, s->residual, s->attn_split);
        if (!ds41_cpu_hc_mix(s, l, true)) goto layer_fail;
        ds41_cpu_hc_sum(s->x, s->after_attn, s->attn_split);
        if (!ds41_cpu_norm(s->norm, s->x, s->model, l->ffn_norm)) goto layer_fail;
        phase = profile ? now_sec() : 0;
        if (!ds41_cpu_moe(s, l)) goto layer_fail;
        if (profile) moe_seconds += now_sec() - phase;
        ds41_cpu_hc_expand(s->residual, s->block, s->after_attn, s->ffn_split);
        memcpy(s->pre, s->ffn_split, sizeof(s->pre));
        continue;
layer_fail:
        s->valid = false;
        /* The readers write s->rows until the second table is complete. */
        if (s->reader) (void)ds4_engram_reader_wait(s->reader, 1);
        if (error && cap) snprintf(error, cap, "V4.1 CPU layer %u failed at token position %u", il, s->pos);
        return false;
    }
    ds41_prof(DS41_PROF_HEAD);
    const double head_started = profile ? now_sec() : 0;
    if (logits && !ds41_cpu_logits(s, logits)) {
        s->valid = false;
        return ds41_cpu_error(error, cap, "V4.1 CPU output projection failed");
    }
    s->history = next_history;
    s->pos++;
    if (profile) {
        const double finished = now_sec(), total = finished - started;
        const double head_seconds = finished - head_started;
        fprintf(stderr, "ds4: CPU profile pos=%u total=%.3f read=%.3f engram=%.3f "
                "attention=%.3f moe=%.3f head=%.3f other=%.3f seconds "
                "dispatches=%" PRIu64 "\n",
                s->pos - 1u, total, read_seconds, engram_seconds, attention_seconds,
                moe_seconds, head_seconds, total - read_seconds - engram_seconds -
                attention_seconds - moe_seconds - head_seconds,
                g_pool_dispatches - dispatches);
        ds41_prof(DS41_PROF_GLUE);
        fprintf(stderr, "ds4: CPU phases pos=%u", s->pos - 1u);
        for (int i = 0; i < DS41_PROF_N; i++)
            fprintf(stderr, " %s=%.2f/%.3f", ds41_prof_name[i],
                    ds41_prof_sec[i] * 1e3, ds41_prof_bytes[i] / 1e9);
        fprintf(stderr, " ms/GB\n");
        ds41_prof_on = false;
    }
    return true;
}

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
