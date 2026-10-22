/* V4.1's CPU reference follows the scalar GPU graph one token at a time.
 * In particular, compression ownership, delayed HC mixing and activation
 * rounding are model semantics, rather than optional backend optimizations. */
#include "ds4_deepseek41_cpu_math.inc"

#if defined(__clang__)
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
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
    DS41_PROF_ROUTED, DS41_PROF_PROJ, DS41_PROF_FFN, DS41_PROF_HEAD, DS41_PROF_N
};
static const char *const ds41_prof_name[DS41_PROF_N] = {
    "glue", "engram_read", "engram", "hc", "attn_q", "attn_kv", "compress",
    "indexer", "stage", "attention", "attn_out", "router", "shared", "routed",
    "attn_proj", "ffn", "head"
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
    /* Inside a staged dispatch the caller counts the bytes. */
    if (ds41_prof_on && !g_parallel_depth) ds41_prof_bytes[ds41_prof_phase] += matrix.bytes;
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

/* One HC mixer: the RMS norm of the four streams in state, the F16
 * projection into mix and the Sinkhorn split.  Only the HC post after the
 * sublayer and the next sublayer's input read the split, so the staged
 * dispatches run the mixer beside the sublayer's projections: the
 * sequential norm as a one-row stage, then the projection rows. */
typedef struct {
    const ds4_model *m;
    const ds4_tensor *fn, *scale, *base;
    const float *state;
    float *flat, *mix, *split;
    matvec_f16_ctx rows;
} ds41_cpu_hc_job;

static bool ds41_cpu_hc_run(const ds41_cpu_hc_job *j) {
    if (!j->scale || !j->base || j->scale->type != DS4_TENSOR_F32 ||
        j->base->type != DS4_TENSOR_F32) return false;
    rms_norm_no_weight(j->flat, j->state, 4u * 5120u, DS4_RMS_EPS);
    if (!ds41_cpu_matvec(j->mix, j->m, j->fn, j->flat, false)) return false;
    hc_split_sinkhorn_one(j->split, j->mix, tensor_data(j->m, j->scale),
                         tensor_data(j->m, j->base), 4, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS);
    return true;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
static bool ds41_cpu_hc_staged(const ds41_cpu_hc_job *j) {
    return j->fn && j->fn->type == DS4_TENSOR_F16 && j->fn->ndim == 2 &&
        j->fn->dim[0] == 4u * 5120u && j->fn->dim[1] == 24 &&
        j->scale && j->scale->type == DS4_TENSOR_F32 &&
        j->base && j->base->type == DS4_TENSOR_F32;
}

static void ds41_cpu_hc_norm_worker(void *vctx, uint64_t first, uint64_t end) {
    const ds41_cpu_hc_job *j = vctx;
    if (first < end) rms_norm_no_weight(j->flat, j->state, 4u * 5120u, DS4_RMS_EPS);
}

static void ds41_cpu_hc_split(void *vctx) {
    const ds41_cpu_hc_job *j = vctx;
    hc_split_sinkhorn_one(j->split, j->mix, tensor_data(j->m, j->scale),
                         tensor_data(j->m, j->base), 4, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS);
}

static ds4_pool_stage ds41_cpu_hc_norm_stage(ds41_cpu_hc_job *j) {
    return (ds4_pool_stage){.fn = ds41_cpu_hc_norm_worker, .ctx = j,
                            .n_rows = 1, .group_rows = 1};
}

/* The 24 projection rows as three F16 row groups, after stage norm. */
static ds4_pool_stage ds41_cpu_hc_mix_stage(ds41_cpu_hc_job *j, uint32_t norm) {
    j->rows = (matvec_f16_ctx){.out = j->mix, .data = tensor_data(j->m, j->fn),
                               .x = j->flat, .in_dim = 4u * 5120u, .out_dim = 24};
    return (ds4_pool_stage){.fn = matvec_f16_group_worker, .ctx = &j->rows,
                            .n_rows = 24 / DS4_F16_ROWS, .group_rows = 24 / DS4_F16_ROWS,
                            .deps = 1u << norm, .finish = ds41_cpu_hc_split, .finish_ctx = j};
}

static bool ds41_cpu_q8_shape(const ds4_tensor *t, uint64_t in_dim, uint64_t rows) {
    return t && t->type == DS4_TENSOR_Q8_0 && t->ndim == 2 &&
        t->dim[0] == in_dim && t->dim[1] == rows;
}

static bool ds41_cpu_f16_shape(const ds4_tensor *t, uint64_t in_dim, uint64_t rows) {
    return t && t->type == DS4_TENSOR_F16 && t->ndim == 2 &&
        t->dim[0] == in_dim && t->dim[1] == rows;
}
#endif

static void ds41_cpu_hc_sum(float *out, const float *residual, const float *pre) {
    hc_weighted_sum_one(out, residual, pre, 5120, 4);
    ds41_cpu_bf16_rows(out, 5120);
}

static void ds41_cpu_hc_expand(float *out, const float *block,
                              const float *residual, const float *split) {
    hc_post_one(out, block, residual, split + 4, split + 8, 5120, 4);
    ds41_cpu_bf16_rows(out, 4u * 5120u);
}

/* The HC post after attention and the FFN input: after = the expanded HC
 * state, x = its BF16 sum with the attention mixer's pre weights, norm = the
 * FFN norm of x.  Staged, the first two run per column. */
typedef struct {
    const ds4_model *m;
    const ds4_tensor *norm_weight;
    const float *block, *residual, *split;
    float *after, *x, *norm;
} ds41_cpu_hc_post_job;

static bool ds41_cpu_hc_post_run(const ds41_cpu_hc_post_job *j) {
    ds41_cpu_hc_expand(j->after, j->block, j->residual, j->split);
    ds41_cpu_hc_sum(j->x, j->after, j->split);
    return ds41_cpu_norm(j->norm, j->x, j->m, j->norm_weight);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* ds41_cpu_hc_expand() for the columns [d0, d1): per element the products
 * and sums of hc_post_one() in its order, then the BF16 rounding. */
static void ds41_cpu_hc_expand_cols(float *out, const float *block,
                                   const float *residual, const float *split,
                                   uint64_t d0, uint64_t d1) {
    const float *post = split + 4, *comb = split + 8;
    for (uint32_t dst = 0; dst < 4; dst++) {
        float *o = out + (size_t)dst * 5120u;
        for (uint64_t d = d0; d < d1; d++) {
            float v = block[d] * post[dst];
            for (uint32_t src = 0; src < 4; src++)
                v += comb[dst + src * 4u] * residual[(size_t)src * 5120u + d];
            o[d] = ds41_cpu_bf16(v);
        }
    }
}

/* Per element hc_weighted_sum_one(): the streams in order from 0.0f. */
static void ds41_cpu_hc_post_worker(void *vctx, uint64_t d0, uint64_t d1) {
    const ds41_cpu_hc_post_job *j = vctx;
    ds41_cpu_hc_expand_cols(j->after, j->block, j->residual, j->split, d0, d1);
    for (uint64_t d = d0; d < d1; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) acc += j->after[(size_t)h * 5120u + d] * j->split[h];
        j->x[d] = ds41_cpu_bf16(acc);
    }
}

static void ds41_cpu_hc_post_norm(void *vctx) {
    const ds41_cpu_hc_post_job *j = vctx;
    ds41_cpu_norm(j->norm, j->x, j->m, j->norm_weight);
}
#endif

#include "ds4_deepseek41_cpu_attention.inc"

/* Per-token activations of a layer-major prefill chunk (see ds41_cpu_prefill),
 * allocated with the first chunk: about 1 MB per token. */
#define DS41_CPU_BATCH 512u
#define DS41_CPU_BATCH_TOKEN_BYTES (4u * (4u * 4u * 5120u + 3u * 5120u + 52u + 1280u + \
    2u * 64u * 512u + 512u + 8192u + 2u * 24u * 256u + 5u * 5120u + 3u * 2304u + \
    6u * (2304u + 5120u + 1u) + 6u + 48u + 6u + 512u))
typedef struct ds41_cpu_batch {
    float *residual, *after_attn, *flat, *x, *norm, *block, *shared;
    float *pre, *attn_split, *ffn_split, *qr, *q, *kv, *heads, *low;
    float *rows, *engram_kv, *shared_gate, *shared_up, *shared_mid;
    float *mid, *partial, *weight;
    int *selected;
    uint32_t *ids, *entry, *sel;
} ds41_cpu_batch;

typedef struct ds41_cpu_state {
    const ds4_model *model;
    const ds4_weights *weights;
    uint32_t ctx, pos;
    bool valid;
    ds41_cpu_attention_state *attention;
    ds41_cpu_batch *batch;
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
        (uint64_t)DS41_CPU_BATCH * DS41_CPU_BATCH_TOKEN_BYTES +
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

static void ds41_cpu_batch_free(ds41_cpu_batch *b);

static void ds41_cpu_free(ds41_cpu_state *s) {
    if (!s) return;
    ds41_cpu_batch_free(s->batch);
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

/* The attention mixer reads the HC state, the FFN mixer the state after
 * attention. */
static ds41_cpu_hc_job ds41_cpu_hc_mixer(ds41_cpu_state *s, const ds4_layer_weights *l,
                                        bool ffn) {
    return (ds41_cpu_hc_job){
        .m = s->model,
        .fn = ffn ? l->hc_ffn_fn : l->hc_attn_fn,
        .scale = ffn ? l->hc_ffn_scale : l->hc_attn_scale,
        .base = ffn ? l->hc_ffn_base : l->hc_attn_base,
        .state = ffn ? s->after_attn : s->residual,
        .flat = s->flat_norm, .mix = s->mix,
        .split = ffn ? s->ffn_split : s->attn_split,
    };
}

/* The HC mixer of in (the attention mixer of the HC state, the FFN mixer of
 * the state after attention) into split; flat and mix are scratch. */
static bool ds41_cpu_hc_mix_into(const ds4_model *m, const ds4_layer_weights *l,
                                 bool ffn, const float *in, float *flat, float *mix,
                                 float *split) {
    const ds41_cpu_hc_job j = {
        .m = m, .fn = ffn ? l->hc_ffn_fn : l->hc_attn_fn,
        .scale = ffn ? l->hc_ffn_scale : l->hc_attn_scale,
        .base = ffn ? l->hc_ffn_base : l->hc_attn_base,
        .state = in, .flat = flat, .mix = mix, .split = split,
    };
    return ds41_cpu_hc_run(&j);
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

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* The V4.1 FFN as one staged dispatch, like V4's layer_ffn_graph(). The
 * FFN input comes from the attention mixer, so this layer's FFN mixer only
 * feeds the HC post and the next layer: its norm, projection and Sinkhorn
 * split overlap the expert weights.  Every stage runs the row kernels of
 * the plain path (ds41_cpu_hc_run, ds41_cpu_moe, ds41_cpu_hc_expand) on the
 * same inputs, the finish steps are its serial code, and the last stage
 * applies its BF16 roundings, shared + routed sum and HC post per column,
 * so every value is bitwise the plain one. */
typedef struct {
    ds41_cpu_state *s;
    const ds4_layer_weights *l;
    float logits[DS4_MAX_EXPERT];
    ds41_cpu_hc_job hc;
    matvec_f32_ctx router;
    ds41_cpu_q8_pair_ctx shared_gate_up;
    ds41_cpu_q8_ctx shared_down;
    glm_routed_moe_f32_mid_ctx mid;
    glm_routed_moe_f32_down_ctx down;
} ds41_cpu_ffn_graph;

static void ds41_cpu_ffn_route(void *vctx) {
    ds41_cpu_ffn_graph *g = vctx;
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED], probs[DS4_MAX_EXPERT];
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++)
        probs[i] = sqrtf(softplus_stable(g->logits[i]));
    layer_topk_selected_experts_from_probs(selected, expert_weight, g->s->model, g->l, probs);
    ds41_cpu_routed_bind(&g->mid, &g->down, g->s->model, g->l, selected, expert_weight);
}

static void ds41_cpu_ffn_shared_mid(void *vctx) {
    ds41_cpu_state *s = vctx;
    ds41_cpu_bf16_rows(s->shared_gate, 2304);
    ds41_cpu_bf16_rows(s->shared_up, 2304);
    swiglu(s->shared_mid, s->shared_gate, s->shared_up, 2304, DS4_SWIGLU_CLAMP_EXP);
    ds41_cpu_bf16_rows(s->shared_mid, 2304);
}

static void ds41_cpu_ffn_post_worker(void *vctx, uint64_t d0, uint64_t d1) {
    ds41_cpu_state *s = vctx;
    for (uint64_t d = d0; d < d1; d++)
        s->block[d] = ds41_cpu_bf16(s->block[d] + ds41_cpu_bf16(s->shared[d]));
    ds41_cpu_hc_expand_cols(s->residual, s->block, s->after_attn, s->ffn_split, d0, d1);
}

static bool ds41_cpu_ffn_graph_supported(const ds4_layer_weights *l,
                                         const ds41_cpu_hc_job *hc) {
    const ds4_tensor *r = l->ffn_gate_inp;
    return ds41_cpu_hc_staged(hc) && r && r->type == DS4_TENSOR_F32 && r->ndim == 2 &&
        r->dim[0] == 5120 && r->dim[1] == DS4_N_EXPERT &&
        l->ffn_exp_probs_b && ds41_cpu_routed_rows(l) &&
        ds41_cpu_q8_shape(l->ffn_gate_shexp, 5120, 2304) &&
        ds41_cpu_q8_shape(l->ffn_up_shexp, 5120, 2304) &&
        ds41_cpu_q8_shape(l->ffn_down_shexp, 2304, 5120);
}

/* From the FFN norm in s->norm and the HC state in s->after_attn to
 * s->ffn_split and the new HC state in s->residual. */
static void ds41_cpu_ffn_graph_run(ds41_cpu_state *s, const ds4_layer_weights *l,
                                   const ds41_cpu_hc_job *hc) {
    const ds4_model *m = s->model;
    const uint64_t mid_tiles = 2304 / 16, tiles = 5120 / 16;
    ds41_cpu_ffn_graph g = {.s = s, .l = l, .hc = *hc};
    g.router = (matvec_f32_ctx){.out = g.logits, .data = tensor_data(m, l->ffn_gate_inp),
                                .x = s->norm, .in_dim = 5120};
    g.shared_gate_up.units = mid_tiles;
    for (unsigned k = 0; k < 2; k++) {
        g.shared_gate_up.m[k] = (ds41_cpu_q8_ctx){
            .ref = {.out = k ? s->shared_up : s->shared_gate,
                    .data = tensor_data(m, k ? l->ffn_up_shexp : l->ffn_gate_shexp),
                    .x = s->norm, .in_dim = 5120, .blocks = 5120 / 32},
            .rows = 2304, .group_rows = 2304,
        };
    }
    g.shared_down = (ds41_cpu_q8_ctx){
        .ref = {.out = s->shared, .data = tensor_data(m, l->ffn_down_shexp),
                .x = s->shared_mid, .in_dim = 2304, .blocks = 2304 / 32},
        .rows = 5120, .group_rows = 5120,
    };
    ds41_cpu_routed_init(&g.mid, &g.down, l, s->norm, s->expert_mid, s->block);
    /* Independent stages first: a thread waits at the first stage whose
     * inputs are not ready.  The gate/up rows of each expert are their own
     * NUMA group, so each node reads the half of every expert placed on it. */
    ds4_pool_stage stages[8] = {
        ds41_cpu_hc_norm_stage(&g.hc),
        { .fn = matvec_f32_worker, .ctx = &g.router, .n_rows = DS4_N_EXPERT,
          .group_rows = DS4_N_EXPERT, .finish = ds41_cpu_ffn_route, .finish_ctx = &g },
        { .fn = ds41_cpu_q8_pair_worker, .ctx = &g.shared_gate_up, .n_rows = 2 * mid_tiles,
          .group_rows = mid_tiles, .finish = ds41_cpu_ffn_shared_mid, .finish_ctx = s },
        { .fn = ds41_cpu_mid_worker, .ctx = &g.mid, .n_rows = DS4_N_EXPERT_USED * mid_tiles,
          .group_rows = mid_tiles, .deps = 1u << 1 },
        ds41_cpu_hc_mix_stage(&g.hc, 0),
        { .fn = ds41_cpu_q8_worker, .ctx = &g.shared_down, .n_rows = tiles,
          .group_rows = tiles, .deps = 1u << 2 },
        { .fn = ds41_cpu_down_worker, .ctx = &g.down, .n_rows = tiles,
          .group_rows = tiles, .deps = 1u << 3 },
        /* Light elementwise work: claims of 256 columns. */
        { .fn = ds41_cpu_ffn_post_worker, .ctx = s, .n_rows = 5120, .group_rows = 5120,
          .min_chunk = 256, .deps = (1u << 4) | (1u << 5) | (1u << 6) },
    };
    ds4_pool_run_stages(stages, 8);
    if (ds41_prof_on)
        ds41_prof_bytes[DS41_PROF_FFN] += l->hc_ffn_fn->bytes + l->ffn_gate_inp->bytes +
            l->ffn_gate_shexp->bytes + l->ffn_up_shexp->bytes + l->ffn_down_shexp->bytes +
            DS4_N_EXPERT_USED * ((l->ffn_gate_exps->bytes + l->ffn_up_exps->bytes +
                                  l->ffn_down_exps->bytes) / DS4_N_EXPERT);
}
#endif

/* The FFN from s->after_attn (with the norm of its attention-mixed input in
 * s->norm) to s->ffn_split and the next HC state in s->residual. */
static bool ds41_cpu_ffn(ds41_cpu_state *s, const ds4_layer_weights *l) {
    const ds41_cpu_hc_job hc = ds41_cpu_hc_mixer(s, l, true);
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (ds41_cpu_ffn_graph_supported(l, &hc)) {
        ds41_prof(DS41_PROF_FFN);
        ds41_cpu_ffn_graph_run(s, l, &hc);
        ds41_prof(DS41_PROF_HC);
        return true;
    }
#endif
    if (!ds41_cpu_hc_run(&hc) || !ds41_cpu_moe(s, l)) return false;
    ds41_cpu_hc_expand(s->residual, s->block, s->after_attn, s->ffn_split);
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
        /* The newly computed attention mixer is consumed by the next FFN.
         * Attention itself consumes the preceding layer's FFN preweights,
         * so the mixer runs inside the attention projections. */
        const ds41_cpu_hc_job hc = ds41_cpu_hc_mixer(s, l, false);
        /* After attention: the HC post, then the FFN input in s->x and its
         * norm in s->norm, which attention no longer reads by then. */
        const ds41_cpu_hc_post_job post = {
            .m = s->model, .norm_weight = l->ffn_norm, .block = s->block,
            .residual = s->residual, .split = s->attn_split,
            .after = s->after_attn, .x = s->x, .norm = s->norm,
        };
        ds41_cpu_hc_sum(s->x, s->residual, s->pre);
        if (!ds41_cpu_norm(s->norm, s->x, s->model, l->attn_norm)) goto layer_fail;
        double phase = profile ? now_sec() : 0;
        if (!ds41_cpu_attention_forward(s->attention, s->model, l, il, s->pos, s->norm,
                                        s->block, &hc, &post))
            goto layer_fail;
        if (profile) attention_seconds += now_sec() - phase;
        ds41_prof(DS41_PROF_HC);
        phase = profile ? now_sec() : 0;
        if (!ds41_cpu_ffn(s, l)) goto layer_fail;
        if (profile) moe_seconds += now_sec() - phase;
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

/* Layer-major prefill.  Each layer runs over a chunk of prompt tokens before
 * the next one, so a dense tile or a routed expert is decoded once for all
 * the chunk's tokens that use it.  Attention, compression and indexing run
 * per token in position order on the shared caches; HC, norms, the router
 * and the other glue run per token in parallel.  Every value is computed by
 * the same operations as in ds41_cpu_forward, so the logits are bitwise
 * those of the token loop. */
typedef struct {
    ds41_cpu_state *s;
    ds41_cpu_batch *b;
    const ds4_layer_weights *l;
    uint32_t il;
    bool routed;    /* partial holds the routed experts' down projections */
    int failed;
} ds41_cpu_batch_ctx;

static void ds41_cpu_batch_fail(ds41_cpu_batch_ctx *c) {
    __atomic_store_n(&c->failed, 1, __ATOMIC_RELAXED);
}

static void ds41_cpu_batch_free(ds41_cpu_batch *b) {
    if (!b) return;
    free(b->residual); free(b->after_attn); free(b->flat); free(b->x);
    free(b->norm); free(b->block); free(b->shared); free(b->pre);
    free(b->attn_split); free(b->ffn_split); free(b->qr); free(b->q);
    free(b->kv); free(b->heads); free(b->low); free(b->rows);
    free(b->engram_kv); free(b->shared_gate); free(b->shared_up);
    free(b->shared_mid); free(b->mid); free(b->partial); free(b->weight);
    free(b->selected); free(b->ids); free(b->entry); free(b->sel);
    free(b);
}

static ds41_cpu_batch *ds41_cpu_batch_create(void) {
    ds41_cpu_batch *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    const size_t n = DS41_CPU_BATCH, f = sizeof(float);
    const size_t pairs = n * DS4_N_EXPERT_USED;
    b->residual = malloc(n * 4u * 5120u * f);
    b->after_attn = malloc(n * 4u * 5120u * f);
    b->flat = malloc(n * 4u * 5120u * f);
    b->x = malloc(n * 5120u * f);
    b->norm = malloc(n * 5120u * f);
    b->block = malloc(n * 5120u * f);
    b->shared = malloc(n * 5120u * f);
    b->pre = malloc(n * 4u * f);
    b->attn_split = malloc(n * 24u * f);
    b->ffn_split = malloc(n * 24u * f);
    b->qr = malloc(n * 1280u * f);
    b->q = malloc(n * 64u * 512u * f);
    b->kv = malloc(n * 512u * f);
    b->heads = malloc(n * 64u * 512u * f);
    b->low = malloc(n * 8192u * f);
    b->rows = malloc(n * 2u * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * f);
    b->engram_kv = malloc(n * 5u * 5120u * f);
    b->shared_gate = malloc(n * 2304u * f);
    b->shared_up = malloc(n * 2304u * f);
    b->shared_mid = malloc(n * 2304u * f);
    b->mid = malloc(pairs * 2304u * f);
    b->partial = malloc(pairs * 5120u * f);
    b->weight = malloc(pairs * f);
    b->selected = malloc(pairs * sizeof(int));
    b->ids = malloc(n * 2u * DS4_ENGRAM_COLS * sizeof(uint32_t));
    b->entry = malloc(pairs * sizeof(uint32_t));
    b->sel = malloc(n * 512u * sizeof(uint32_t));
    if (!b->residual || !b->after_attn || !b->flat || !b->x || !b->norm ||
        !b->block || !b->shared || !b->pre || !b->attn_split || !b->ffn_split ||
        !b->qr || !b->q || !b->kv || !b->heads || !b->low || !b->rows ||
        !b->engram_kv || !b->shared_gate || !b->shared_up || !b->shared_mid ||
        !b->mid || !b->partial || !b->weight || !b->selected || !b->ids || !b->entry ||
        !b->sel) {
        ds41_cpu_batch_free(b);
        return NULL;
    }
    return b;
}

/* n tokens through one dense matrix, as ds41_cpu_matvec does each of them. */
static bool ds41_cpu_matmul(float *out, uint64_t out_stride, const ds4_model *m,
                            const ds4_tensor *w, const float *x, uint64_t x_stride,
                            uint32_t n, bool round) {
    if (!w || w->ndim < 2 || !w->dim[0] || !w->dim[1]) return false;
    ds4_tensor matrix = *w;
    matrix.ndim = 2;
    matrix.elements = matrix.dim[0] * matrix.dim[1];
    if (ds41_cpu_matmul_q8(out, out_stride, m, &matrix, x, x_stride, n,
                           matrix.dim[1], 0, round)) return true;
    for (uint32_t t = 0; t < n; t++)
        if (!ds41_cpu_matvec(out + (size_t)t * out_stride, m, w,
                             x + (size_t)t * x_stride, round)) return false;
    return true;
}

static void ds41_cpu_batch_read_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    const size_t cols = DS4_ENGRAM_COLS;
    for (uint64_t t = t0; t < t1; t++)
        for (uint32_t i = 0; i < 2; i++)
            if (!ds4_engram_read(&c->s->table[i], c->b->ids + (t * 2u + i) * cols, cols,
                                 c->b->rows + (t * 2u + i) * cols * DS4_ENGRAM_DIM))
                ds41_cpu_batch_fail(c);
}

/* Engram, the attention HC mixer and the attention norm of each token. */
static void ds41_cpu_batch_attn_in_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    const ds4_model *m = c->s->model;
    const ds4_layer_weights *l = c->l;
    float mix[24];
    for (uint64_t t = t0; t < t1; t++) {
        float *residual = b->residual + t * 4u * 5120u;
        if (c->il == 1 || c->il == 14)
            ds41_cpu_engram_add(residual, b->engram_kv + t * 5u * 5120u,
                               tensor_data(m, l->engram_q_norm),
                               tensor_data(m, l->engram_k_norm), 5120, DS4_RMS_EPS);
        if (!ds41_cpu_hc_mix_into(m, l, false, residual, b->flat + t * 4u * 5120u,
                                  mix, b->attn_split + t * 24u)) {
            ds41_cpu_batch_fail(c);
            continue;
        }
        ds41_cpu_hc_sum(b->x + t * 5120u, residual, b->pre + t * 4u);
        if (!ds41_cpu_norm(b->norm + t * 5120u, b->x + t * 5120u, m, l->attn_norm))
            ds41_cpu_batch_fail(c);
    }
}

static void ds41_cpu_batch_qkv_norm_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    const ds4_model *m = c->s->model;
    for (uint64_t t = t0; t < t1; t++) {
        float *qr = c->b->qr + t * 1280u, *kv = c->b->kv + t * 512u;
        if (!ds41_cpu_norm(qr, qr, m, c->l->attn_q_a_norm) ||
            !ds41_cpu_norm(kv, kv, m, c->l->attn_kv_a_norm)) ds41_cpu_batch_fail(c);
    }
}

/* HC after attention, the FFN mixer and norm, and the router of each token. */
static void ds41_cpu_batch_ffn_in_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    const ds4_model *m = c->s->model;
    const ds4_layer_weights *l = c->l;
    float mix[24], probs[DS4_MAX_EXPERT];
    for (uint64_t t = t0; t < t1; t++) {
        float *after = b->after_attn + t * 4u * 5120u, *norm = b->norm + t * 5120u;
        const float *split = b->attn_split + t * 24u;
        ds41_cpu_hc_expand(after, b->block + t * 5120u, b->residual + t * 4u * 5120u, split);
        if (!ds41_cpu_hc_mix_into(m, l, true, after, b->flat + t * 4u * 5120u,
                                  mix, b->ffn_split + t * 24u)) {
            ds41_cpu_batch_fail(c);
            continue;
        }
        ds41_cpu_hc_sum(b->x + t * 5120u, after, split);
        if (!ds41_cpu_norm(norm, b->x + t * 5120u, m, l->ffn_norm) ||
            !ds41_cpu_matvec(probs, m, l->ffn_gate_inp, norm, false)) {
            ds41_cpu_batch_fail(c);
            continue;
        }
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++)
            probs[i] = sqrtf(softplus_stable(probs[i]));
        layer_topk_selected_experts_from_probs(b->selected + t * DS4_N_EXPERT_USED,
                                               b->weight + t * DS4_N_EXPERT_USED, m, l, probs);
    }
}

static void ds41_cpu_batch_swiglu_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        float *mid = c->b->shared_mid + t * 2304u;
        swiglu(mid, c->b->shared_gate + t * 2304u, c->b->shared_up + t * 2304u,
               2304, DS4_SWIGLU_CLAMP_EXP);
        ds41_cpu_bf16_rows(mid, 2304);
    }
}

/* Routed slots added in slot order, as ds41_cpu_down_worker, then the
 * shared expert and the FFN HC expansion. */
static void ds41_cpu_batch_ffn_out_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    for (uint64_t t = t0; t < t1; t++) {
        float *block = b->block + t * 5120u;
        const float *shared = b->shared + t * 5120u;
        if (c->routed) {
            const float *p = b->partial + t * DS4_N_EXPERT_USED * 5120u;
            for (uint32_t i = 0; i < 5120; i++) {
                float sum = 0.0f;
                for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++)
                    sum = sum + p[(size_t)slot * 5120u + i];
                block[i] = sum;
            }
        }
        for (uint32_t i = 0; i < 5120; i++) block[i] = ds41_cpu_bf16(block[i] + shared[i]);
        ds41_cpu_hc_expand(b->residual + t * 4u * 5120u, block,
                          b->after_attn + t * 4u * 5120u, b->ffn_split + t * 24u);
        memcpy(b->pre + t * 4u, b->ffn_split + t * 24u, 4u * sizeof(float));
    }
}

static bool ds41_cpu_prefill_layer(ds41_cpu_state *s, uint32_t il, uint32_t n,
                                   double *phase) {
    ds41_cpu_batch *b = s->batch;
    const ds4_model *m = s->model;
    const ds4_layer_weights *l = &s->weights->layer[il];
    ds41_cpu_attention_state *as = s->attention;
    ds41_cpu_batch_ctx c = {.s = s, .b = b, .l = l, .il = il};
    if (!l->ffn_exp_probs_b || !l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps ||
        !glm_graph_gate_pair_type_supported(l->ffn_gate_exps->type, l->ffn_up_exps->type) ||
        !glm_graph_down_type_supported(l->ffn_down_exps->type)) return false;
    double t = phase ? now_sec() : 0;
#define DS41_PHASE(i) do { if (phase) { const double now = now_sec(); \
    phase[i] += now - t; t = now; } } while (0)
    if (il == 1 || il == 14) {
        const size_t cols = (size_t)DS4_ENGRAM_COLS * DS4_ENGRAM_DIM;
        if (!ds41_cpu_matmul(b->engram_kv, 5u * 5120u, m, l->engram_kv,
                             b->rows + (il == 1 ? 0 : cols), 2u * cols, n, true)) return false;
    }
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_attn_in_worker, &c, 1);
    if (c.failed) return false;
    DS41_PHASE(0);
    if (!ds41_cpu_matmul(b->qr, 1280, m, l->attn_q_a, b->norm, 5120, n, true) ||
        !ds41_cpu_matmul(b->kv, 512, m, l->attn_kv, b->norm, 5120, n, true)) return false;
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_qkv_norm_worker, &c, 1);
    if (c.failed ||
        !ds41_cpu_matmul(b->q, 64u * 512u, m, l->attn_q_b, b->qr, 1280, n, true)) return false;
    DS41_PHASE(1);
    /* A compressed layer that is no index source attends the rows its
     * token's last index source selected: keep that selection per token. */
    const bool index = ds41_index_source(il);
    const bool reuse = !index && ds4_layer_compress_ratio(il) != 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t *sel = b->sel + (size_t)i * 512u;
        memcpy(as->qr, b->qr + (size_t)i * 1280u, sizeof(as->qr));
        memcpy(as->q, b->q + (size_t)i * 64u * 512u, sizeof(as->q));
        memcpy(as->kv, b->kv + (size_t)i * 512u, sizeof(as->kv));
        if (reuse) memcpy(as->selected_comp, sel, sizeof(as->selected_comp));
        if (!ds41_cpu_attention_core(as, m, l, il, s->pos + i, b->norm + (size_t)i * 5120u))
            return false;
        if (index) memcpy(sel, as->selected_comp, sizeof(as->selected_comp));
        memcpy(b->heads + (size_t)i * 64u * 512u, as->heads, sizeof(as->heads));
    }
    DS41_PHASE(2);
    const ds4_tensor *oa = l->attn_output_a;
    if (oa->type == DS4_TENSOR_Q8_0 && oa->dim[0] == 4096u && oa->elements == 4096u * 8192u &&
        ds41_cpu_matmul_q8(b->low, 8192, m, oa, b->heads, 64u * 512u, n, 1024u, 4096u, true)) {
        if (!ds41_cpu_matmul(b->block, 5120, m, l->attn_output_b, b->low, 8192, n, true))
            return false;
    } else {
        for (uint32_t i = 0; i < n; i++) {
            memcpy(as->heads, b->heads + (size_t)i * 64u * 512u, sizeof(as->heads));
            if (!ds41_cpu_attention_output(as, m, l, b->block + (size_t)i * 5120u)) return false;
        }
    }
    DS41_PHASE(3);
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_ffn_in_worker, &c, 1);
    if (c.failed) return false;
    DS41_PHASE(4);
    if (!ds41_cpu_matmul(b->shared_gate, 2304, m, l->ffn_gate_shexp, b->norm, 5120, n, true) ||
        !ds41_cpu_matmul(b->shared_up, 2304, m, l->ffn_up_shexp, b->norm, 5120, n, true))
        return false;
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_swiglu_worker, &c, 1);
    if (!ds41_cpu_matmul(b->shared, 5120, m, l->ffn_down_shexp, b->shared_mid, 2304, n, true))
        return false;
    DS41_PHASE(5);
    c.routed = ds41_cpu_routed_moe_batch(b->partial, b->mid, b->entry, m, l, b->norm, 5120, n,
                                         b->selected, b->weight);
    if (!c.routed) {
        for (uint32_t i = 0; i < n; i++)
            ds41_cpu_routed_moe(b->block + (size_t)i * 5120u, s->expert_mid, m, l,
                                b->norm + (size_t)i * 5120u,
                                b->selected + (size_t)i * DS4_N_EXPERT_USED,
                                b->weight + (size_t)i * DS4_N_EXPERT_USED);
    }
    DS41_PHASE(6);
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_ffn_out_worker, &c, 1);
    DS41_PHASE(4);
#undef DS41_PHASE
    return !c.failed;
}

/* Up to DS41_CPU_BATCH prompt tokens at the current position; logits, when
 * given, receive the last token's. */
static bool ds41_cpu_prefill(ds41_cpu_state *s, const int *tokens, uint32_t n,
                             float *logits, char *error, size_t cap) {
    if (!s || !s->valid) return ds41_cpu_error(error, cap, "invalid V4.1 CPU state; reset required");
    if (n > DS41_CPU_BATCH) return ds41_cpu_error(error, cap, "V4.1 CPU prefill chunk too large");
    if (n > s->ctx - s->pos) return ds41_cpu_error(error, cap, "V4.1 CPU context is full");
    /* Past 2048 blocks layer 20 masks blocks for the later index sources of
     * its own token only; such chunks keep the token loop. */
    const uint32_t ratio20 = ds4_layer_compress_ratio(20);
    const uint64_t last_comp = ratio20 ? ((uint64_t)s->pos + n) / ratio20 : 0;
    if (n <= 1 || (last_comp + 7u) / 8u > 2048u) {
        for (uint32_t i = 0; i < n; i++)
            if (!ds41_cpu_forward(s, tokens[i], i + 1u == n ? logits : NULL, error, cap))
                return false;
        return true;
    }
    for (uint32_t i = 0; i < n; i++)
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB)
            return ds41_cpu_error(error, cap, "V4.1 CPU token is outside vocabulary");
    if (!s->batch && !(s->batch = ds41_cpu_batch_create()))
        return ds41_cpu_error(error, cap, "cannot allocate V4.1 CPU prefill batch");
    ds41_cpu_batch *b = s->batch;
    const bool profile = getenv("DS4_CPU_PROFILE") != NULL;
    const double started = profile ? now_sec() : 0;
    double phase[8] = {0};
    ds4_engram_history history = s->history;
    for (uint32_t i = 0; i < n; i++)
        if (!ds4_engram_hash(&s->engram, &history, &tokens[i], NULL, 1,
                             b->ids + (size_t)i * 2u * DS4_ENGRAM_COLS))
            return ds41_cpu_error(error, cap, "V4.1 Engram hash failed");
    ds41_cpu_batch_ctx c = {.s = s, .b = b};
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_read_worker, &c, 1);
    if (c.failed) return ds41_cpu_error(error, cap, "V4.1 Engram row read failed");
    const double read_seconds = profile ? now_sec() - started : 0;
    for (uint32_t i = 0; i < n; i++) {
        float *x = b->x + (size_t)i * 5120u, *pre = b->pre + (size_t)i * 4u;
        embed_token_any(s->model, s->weights, tokens[i], x);
        hc_from_plain_embedding(b->residual + (size_t)i * 4u * 5120u, x, 5120, 4);
        pre[0] = 1.0f;
        pre[1] = pre[2] = pre[3] = 0.0f;
    }
    for (uint32_t il = 0; il < 40; il++) {
        if (!ds41_cpu_prefill_layer(s, il, n, profile ? phase : NULL)) {
            s->valid = false;
            if (error && cap) snprintf(error, cap, "V4.1 CPU prefill layer %u failed at token position %u", il, s->pos);
            return false;
        }
    }
    memcpy(s->residual, b->residual + (size_t)(n - 1u) * 4u * 5120u, sizeof(s->residual));
    memcpy(s->pre, b->pre + (size_t)(n - 1u) * 4u, sizeof(s->pre));
    const double head_started = profile ? now_sec() : 0;
    if (logits && !ds41_cpu_logits(s, logits)) {
        s->valid = false;
        return ds41_cpu_error(error, cap, "V4.1 CPU output projection failed");
    }
    s->history = history;
    s->pos += n;
    if (profile) {
        const double finished = now_sec();
        fprintf(stderr, "ds4: CPU prefill pos=%u n=%u total=%.3f s (%.2f tok/s) read=%.3f "
                "hc_in=%.3f qkv=%.3f attention=%.3f attn_out=%.3f ffn_in=%.3f shared=%.3f "
                "routed=%.3f head=%.3f\n", s->pos - n, n, finished - started,
                n / (finished - started), read_seconds, phase[0], phase[1], phase[2],
                phase[3], phase[4], phase[5], phase[6], finished - head_started);
    }
    return true;
}

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
