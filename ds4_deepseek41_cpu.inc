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

/* The bytes the decode reads for one routed expert of a layer. */
static uint64_t ds41_cpu_routed_expert_bytes(const ds4_layer_weights *l) {
    return (l->ffn_gate_exps->bytes + l->ffn_up_exps->bytes + l->ffn_down_exps->bytes) /
        DS4_N_EXPERT;
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
/* The positions a layer-major chunk can end at: past 2048 blocks of 8 on
 * layer 20 the prompt keeps the token loop, so no chunk attends or scores
 * more compressed rows than this. */
#define DS41_CPU_BATCH_COMP 16384u
#define DS41_CPU_BATCH_TOKEN_BYTES (4u * (3u * 4u * 5120u + 4u * 5120u + 52u + 1280u + \
    2u * 64u * 512u + 512u + 8192u + 2u * 24u * 256u + 5u * 5120u + 3u * 2304u + \
    6u * (2304u + 5120u + 1u) + 6u + 48u + 6u + 512u + 24u + 384u + 2u * 4u * 5120u + \
    2u * 512u + 32u * 128u + 32u + 2u * 24u * 256u + 48u))
/* The Q2 experts' chunk workspace, moe_grouped_ffn's pf_buf slots, which
 * ds41_cpu_prefill frees on return as V4's prefill does: per token the Q8_K
 * and IQ2_XXS-order input, the six mids, their Q8_K rows and the six down
 * rows; per pool thread the tile scratch. */
#define DS41_CPU_Q8K_TOKEN_BYTES (20u * (sizeof(block_q8_K) + sizeof(iq2xxs_q8)) + \
    6u * (2304u * 4u + 9u * sizeof(block_q8_K) + 5120u * 4u))
#define DS41_CPU_Q8K_THREAD_BYTES (2u * DS4_MOE_ROW_BLOCK * 20u * (QK_K + QK_K / 32u * 4u))
/* The query side's scratch of one pool thread (see
 * ds41_cpu_batch_attention_worker): the attended row list, the transposed
 * query, the head scores, the index scores and the top-k heap. */
typedef struct {
    const float *rows[128 + 512];
    float *qt, *score, *index_scores;
    float inv[64];
    ds41_cpu_candidate heap[512];
} ds41_cpu_batch_attention_scratch;

/* The chunk's Engram rows, read by helper threads while the layers before
 * 1 and 14 run: every token's row of layer 1's table, then of layer 14's.
 * The rows are the bytes ds4_engram_read() returns and land where the
 * synchronous read put them; only the time they are read changes.  A row
 * not in the page cache is a small random read, and the device serves
 * them faster the more are in flight: 64 readers keep a cold chunk's
 * 24576 rows from stalling layer 1.  A reader takes one token's rows of
 * a table at a time, so a chunk of n tokens starts at most n readers. */
#define DS41_CPU_ENGRAM_READERS 64u
typedef struct {
    const ds4_engram_table *table;
    const uint32_t *ids;
    float *rows;
    uint32_t n, next[2], done[2];
    int failed, stop;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t threads[DS41_CPU_ENGRAM_READERS];
    uint32_t started;
} ds41_cpu_engram_batch;

typedef struct ds41_cpu_batch {
    float *residual, *after_attn, *flat, *x, *norm, *block, *shared;
    float *pre, *attn_split, *ffn_split, *qr, *q, *kv, *heads, *low;
    float *rows, *engram_kv, *shared_gate, *shared_up, *shared_mid;
    float *mid, *partial, *weight, *mix, *probs, *ckv, *cscore, *iq, *iw;
    float *prev_window;
    /* The rows the chunk's queries attend, rounded to F16 once per layer:
     * the raw window before the chunk, then the chunk's own KV rows, and
     * room for rounded_rows compressed rows. */
    float *rounded_raw, *rounded_comp;
    uint64_t rounded_rows;
    /* One ds41_cpu_batch_attention_scratch and its buffers per pool
     * thread (DS4_MAX_THREADS slots, whatever the pool's size, so that
     * g_pool_thread always has one), attn_stride bytes apart, allocated
     * with the batch; a slot's pages are touched only by its thread. */
    uint8_t *attn;
    size_t attn_stride;
    /* The next chunk's Engram row ids and rows (next_n tokens), which
     * ahead reads while ahead_live (see ds41_cpu_prefill). */
    uint32_t *next_ids, next_n;
    float *next_rows;
    ds41_cpu_engram_batch ahead;
    bool ahead_live;
    double *xt;
    int *selected;
    uint32_t *ids, *entry, *sel;
} ds41_cpu_batch;

/* The query side's scratch of one pool thread: the scratch struct, the
 * transposed query (64 heads x 512), the scores (64 heads x 640 rows) and
 * the index scores of up to min(ctx, DS41_CPU_BATCH_COMP) + 1 keys, each
 * on whole cache lines. */
static size_t ds41_cpu_batch_attention_stride(uint32_t ctx) {
    const uint64_t keys = (ctx < DS41_CPU_BATCH_COMP ? ctx : DS41_CPU_BATCH_COMP) + 1u;
    const size_t lines[4] = {
        sizeof(ds41_cpu_batch_attention_scratch), 64u * 512u * sizeof(float),
        64u * (128u + 512u) * sizeof(float), (size_t)keys * sizeof(float)
    };
    size_t stride = 0;
    for (uint32_t i = 0; i < 4; i++) stride += (lines[i] + 63u) & ~(size_t)63u;
    return stride;
}

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
    /* The staged FFN's Q8_K experts: input, mids and the six down rows. */
    block_q8_K expert_xq[5120 / QK_K], expert_midq[6 * 2304 / QK_K];
    float expert_part[6 * 5120];
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
    /* The rows a chunk's queries attend, rounded to F16 once per layer:
     * the raw window, the chunk's rows and at most the compressed rows of
     * DS41_CPU_BATCH_COMP positions. */
    const uint64_t rounded = (uint64_t)DS41_CPU_BATCH + 128u +
        (ctx < DS41_CPU_BATCH_COMP ? ctx : DS41_CPU_BATCH_COMP);
    memory.scratch_bytes = sizeof(ds41_cpu_state) + sizeof(ds41_cpu_attention_state) +
        (uint64_t)DS41_CPU_BATCH * (DS41_CPU_BATCH_TOKEN_BYTES + DS41_CPU_Q8K_TOKEN_BYTES) +
        (uint64_t)DS4_MAX_THREADS * DS41_CPU_Q8K_THREAD_BYTES +
        rounded * 512u * sizeof(float) +
        (uint64_t)DS4_N_VOCAB * (sizeof(uint32_t) + 2u * sizeof(float)) +
        ((uint64_t)ctx + 1u) * sizeof(float) + blocks * (sizeof(float) + sizeof(uint8_t)) +
        (uint64_t)DS4_MAX_THREADS * ds41_cpu_batch_attention_stride(ctx);
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

/* Q2 routed experts in the arithmetic of the CUDA V4.1 path and of the V4
 * CPU decode: the expert input and every SwiGLU mid are quantized to Q8_K
 * (ds4_quantize_row_q8_K is the formula of q8_K_quantize_kernel), gate/up
 * and down are integer IQ2_XXS and Q2_K dots, the router weight scales the
 * mid before down and the six down rows are summed in F32, as in
 * moe_gate_up_mid_aligned_q8k and moe_down_sum_qwarp32.  The result is not
 * the F32-activation reference of ds41_cpu_routed_moe, which remains the
 * path of the other expert types and the oracle of the unit tests.
 * True when the experts are IQ2_XXS/IQ2_XXS/Q2_K, which take this path. */
static bool ds41_cpu_q8k_experts(const ds4_layer_weights *l) {
    const ds4_tensor *gate = l->ffn_gate_exps, *up = l->ffn_up_exps;
    const ds4_tensor *down = l->ffn_down_exps;
    if (!gate || !up || !down || gate->type != DS4_TENSOR_IQ2_XXS ||
        up->type != DS4_TENSOR_IQ2_XXS || down->type != DS4_TENSOR_Q2_K) return false;
    if (DS4_N_EMBD % QK_K != 0 || DS4_N_EMBD > DS4_MAX_EMBD ||
        DS4_N_FF_EXP % QK_K != 0 || DS4_N_FF_EXP > DS4_MAX_FF_EXP ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED)
        ds4_die("V4.1 CPU routed-MoE shape does not fit the Q8_K buffers");
    /* The kernels index by the tensor dims, the buffers by the shape. */
    if (gate->dim[0] != DS4_N_EMBD || gate->dim[1] != DS4_N_FF_EXP ||
        up->dim[0] != DS4_N_EMBD || up->dim[1] != DS4_N_FF_EXP ||
        down->dim[0] != DS4_N_FF_EXP || down->dim[1] != DS4_N_EMBD)
        ds4_die("V4.1 CPU routed-MoE found unexpected expert dimensions");
    return true;
}

/* One token.  Returns false, having done nothing, for other expert types. */
static bool ds41_cpu_routed_moe_q8k(float *out, float *mid, const ds4_model *m,
                                   const ds4_layer_weights *l, const float *x,
                                   const int *selected, const float *weights) {
    if (!ds41_cpu_q8k_experts(l)) return false;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++)
        if (selected[i] < 0 || (uint32_t)selected[i] >= DS4_N_EXPERT)
            ds4_die("V4.1 CPU routed-MoE selected expert is outside range");
    block_q8_K xq[DS4_MAX_EMBD / QK_K];
    block_q8_K midq[DS4_MAX_EXPERT_USED * (DS4_MAX_FF_EXP / QK_K)];
    const uint64_t mid_blocks = DS4_N_FF_EXP / QK_K;
    ds4_quantize_row_q8_K(x, xq, DS4_N_EMBD);
    matvec_experts_mid_prequant(mid, m, l->ffn_gate_exps, l->ffn_up_exps, xq, selected,
                                weights, (int)DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP);
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++)
        ds4_quantize_row_q8_K(mid + (uint64_t)i * DS4_N_FF_EXP,
                              midq + i * mid_blocks, DS4_N_FF_EXP);
    matvec_experts_down_accum_prequant(out, m, l->ffn_down_exps, midq, selected,
                                       (int)DS4_N_EXPERT_USED);
    return true;
}

/* n tokens, x and out n rows of DS4_N_EMBD: moe_grouped_ffn, the V4 prefill
 * kernels grouped by expert, whose every output is the one-token kernels'
 * (the same integer dots, mids and slot-order sums).  Returns false, having
 * done nothing, for other expert types. */
static bool ds41_cpu_routed_moe_q8k_batch(float *out, const ds4_model *m,
                                         const ds4_layer_weights *l, const float *x,
                                         uint32_t n, const int *selected,
                                         const float *weights) {
    if (!ds41_cpu_q8k_experts(l)) return false;
    uint64_t in, mid, out_dim, gate_rb, up_rb, down_rb;
    const uint8_t *gate = tensor_expert_bytes(m, l->ffn_gate_exps, 0, &in, &mid, &gate_rb);
    const uint8_t *up = tensor_expert_bytes(m, l->ffn_up_exps, 0, &in, &mid, &up_rb);
    const uint8_t *down = tensor_expert_bytes(m, l->ffn_down_exps, 0, &mid, &out_dim, &down_rb);
    for (uint32_t t0 = 0; t0 < n; t0 += DS4_MOE_CHUNK_TOKENS) {
        const uint32_t c = n - t0 < DS4_MOE_CHUNK_TOKENS ? n - t0 : DS4_MOE_CHUNK_TOKENS;
        moe_grouped_ffn(out + (uint64_t)t0 * DS4_N_EMBD, gate, up, down, gate_rb, up_rb,
                        down_rb, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
                        x + (uint64_t)t0 * DS4_N_EMBD, selected + (uint64_t)t0 * DS4_N_EXPERT_USED,
                        weights + (uint64_t)t0 * DS4_N_EXPERT_USED, c, DS4_SWIGLU_CLAMP_EXP);
    }
    return true;
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
    /* Routing weights scale the expert mids before the down projection.
     * Q2 experts quantize their input and mids to Q8_K as CUDA does, the
     * other expert types keep F32 intermediates. Shared experts round to BF16. */
    ds41_prof(DS41_PROF_ROUTED);
    if (ds41_prof_on)
        ds41_prof_bytes[DS41_PROF_ROUTED] += DS4_N_EXPERT_USED * ds41_cpu_routed_expert_bytes(l);
    if (!ds41_cpu_routed_moe_q8k(s->block, s->expert_mid, s->model, l,
                                 s->norm, selected, expert_weight))
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
    /* Q2 experts: ds41_cpu_routed_moe_q8k's kernels and Q8_K activations. */
    bool q8k;
    matvec_iq2_xxs_mid_ctx q8k_mid;
    matvec_q2_k_accum_ctx q8k_down;
    iq2xxs_q8 yq[5120 / QK_K];
} ds41_cpu_ffn_graph;

static void ds41_cpu_ffn_route(void *vctx) {
    ds41_cpu_ffn_graph *g = vctx;
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED], probs[DS4_MAX_EXPERT];
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++)
        probs[i] = sqrtf(softplus_stable(g->logits[i]));
    layer_topk_selected_experts_from_probs(selected, expert_weight, g->s->model, g->l, probs);
    if (!g->q8k) {
        ds41_cpu_routed_bind(&g->mid, &g->down, g->s->model, g->l, selected, expert_weight);
        return;
    }
    /* matvec_experts_mid_prequant and matvec_experts_down_accum_prequant's
     * contexts, as V4's ffn_graph_route() fills them. */
    ds41_cpu_state *s = g->s;
    for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
        uint64_t in, out;
        if (selected[slot] < 0 || (uint32_t)selected[slot] >= DS4_N_EXPERT)
            ds4_die("V4.1 CPU routed-MoE selected expert is outside range");
        const uint32_t e = (uint32_t)selected[slot];
        g->q8k_mid.gate_base[slot] = tensor_expert_bytes(s->model, g->l->ffn_gate_exps, e,
                                                         &in, &out, &g->q8k_mid.gate_row_bytes[slot]);
        g->q8k_mid.up_base[slot] = tensor_expert_bytes(s->model, g->l->ffn_up_exps, e,
                                                       &in, &out, &g->q8k_mid.up_row_bytes[slot]);
        g->q8k_mid.expert_weight[slot] = expert_weight[slot];
        g->q8k_down.base[slot] = tensor_expert_bytes(s->model, g->l->ffn_down_exps, e,
                                                     &in, &out, &g->q8k_down.row_bytes[slot]);
        g->q8k_down.xq[slot] = s->expert_midq + slot * (2304 / QK_K);
    }
    ds4_quantize_row_q8_K(s->norm, s->expert_xq, 5120);
    iq2xxs_q8_prepare(g->yq, s->expert_xq, 5120 / QK_K);
}

static void ds41_cpu_ffn_routed_midq(void *vctx) {
    ds41_cpu_state *s = vctx;
    for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++)
        ds4_quantize_row_q8_K(s->expert_mid + slot * 2304, s->expert_midq + slot * (2304 / QK_K),
                              2304);
}

static void ds41_cpu_ffn_shared_mid(void *vctx) {
    ds41_cpu_state *s = vctx;
    ds41_cpu_bf16_rows(s->shared_gate, 2304);
    ds41_cpu_bf16_rows(s->shared_up, 2304);
    swiglu(s->shared_mid, s->shared_gate, s->shared_up, 2304, DS4_SWIGLU_CLAMP_EXP);
    ds41_cpu_bf16_rows(s->shared_mid, 2304);
}

static void ds41_cpu_ffn_post_worker(void *vctx, uint64_t d0, uint64_t d1) {
    ds41_cpu_ffn_graph *g = vctx;
    ds41_cpu_state *s = g->s;
    /* The Q8_K down rows in slot order from 0.0f, as the plain path sums them. */
    if (g->q8k) {
        for (uint64_t d = d0; d < d1; d++) {
            float sum = 0.0f;
            for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++)
                sum += s->expert_part[slot * 5120u + d];
            s->block[d] = sum;
        }
    }
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

/* The stages from the FFN norm in s->norm and the HC state in s->after_attn
 * to s->ffn_split and the new HC state in s->residual, with their contexts
 * in g.  They read their inputs only when they run. */
static void ds41_cpu_ffn_graph_stages(ds41_cpu_ffn_graph *g, ds41_cpu_state *s,
                                      const ds4_layer_weights *l, const ds41_cpu_hc_job *hc,
                                      ds4_pool_stage stages[8]) {
    const ds4_model *m = s->model;
    const uint64_t mid_tiles = 2304 / 16, tiles = 5120 / 16;
    *g = (ds41_cpu_ffn_graph){.s = s, .l = l, .hc = *hc};
    g->router = (matvec_f32_ctx){.out = g->logits, .data = tensor_data(m, l->ffn_gate_inp),
                                 .x = s->norm, .in_dim = 5120};
    g->shared_gate_up.units = mid_tiles;
    for (unsigned k = 0; k < 2; k++) {
        g->shared_gate_up.m[k] = (ds41_cpu_q8_ctx){
            .ref = {.out = k ? s->shared_up : s->shared_gate,
                    .data = tensor_data(m, k ? l->ffn_up_shexp : l->ffn_gate_shexp),
                    .x = s->norm, .in_dim = 5120, .blocks = 5120 / 32},
            .rows = 2304, .group_rows = 2304,
        };
    }
    g->shared_down = (ds41_cpu_q8_ctx){
        .ref = {.out = s->shared, .data = tensor_data(m, l->ffn_down_shexp),
                .x = s->shared_mid, .in_dim = 2304, .blocks = 2304 / 32},
        .rows = 5120, .group_rows = 5120,
    };
    ds41_cpu_routed_init(&g->mid, &g->down, l, s->norm, s->expert_mid, s->block);
    g->q8k = ds41_cpu_q8k_experts(l);
    g->q8k_mid = (matvec_iq2_xxs_mid_ctx){
        .mid = s->expert_mid, .xq = g->yq, .clamp = DS4_SWIGLU_CLAMP_EXP,
        .in_dim = 5120, .out_dim = 2304, .n_expert = (int)DS4_N_EXPERT_USED,
    };
    g->q8k_down = (matvec_q2_k_accum_ctx){.part = s->expert_part, .in_dim = 2304, .out_dim = 5120};
    /* Independent stages first: a thread waits at the first stage whose
     * inputs are not ready.  The gate/up rows of each expert are their own
     * NUMA group, so each node reads the half of every expert placed on it.
     * No stage declares its weights for claims ahead: the router rows are
     * serial double sums that the kernel, not the stream, paces, and the
     * experts are known only after it. */
    const ds4_pool_stage st[8] = {
        ds41_cpu_hc_norm_stage(&g->hc),
        { .fn = matvec_f32_worker, .ctx = &g->router, .n_rows = DS4_N_EXPERT,
          .group_rows = DS4_N_EXPERT, .finish = ds41_cpu_ffn_route, .finish_ctx = g },
        { .fn = ds41_cpu_q8_pair_worker, .ctx = &g->shared_gate_up, .n_rows = 2 * mid_tiles,
          .group_rows = mid_tiles, .finish = ds41_cpu_ffn_shared_mid, .finish_ctx = s },
        { .fn = ds41_cpu_mid_worker, .ctx = &g->mid, .n_rows = DS4_N_EXPERT_USED * mid_tiles,
          .group_rows = mid_tiles, .deps = 1u << 1 },
        ds41_cpu_hc_mix_stage(&g->hc, 0),
        { .fn = ds41_cpu_q8_worker, .ctx = &g->shared_down, .n_rows = tiles,
          .group_rows = tiles, .deps = 1u << 2 },
        { .fn = ds41_cpu_down_worker, .ctx = &g->down, .n_rows = tiles,
          .group_rows = tiles, .deps = 1u << 3 },
        /* Light elementwise work: claims of 256 columns. */
        { .fn = ds41_cpu_ffn_post_worker, .ctx = g, .n_rows = 5120, .group_rows = 5120,
          .min_chunk = 256, .deps = (1u << 4) | (1u << 5) | (1u << 6) },
    };
    memcpy(stages, st, sizeof(st));
    /* Q8_K experts: the mids in rows (each expert's a NUMA group), quantized
     * by the stage's finish step, then the down rows expert-major, as in
     * V4's layer_ffn_graph(). */
    if (g->q8k) {
        stages[3] = (ds4_pool_stage){ .fn = matvec_iq2_xxs_mid_worker, .ctx = &g->q8k_mid,
            .n_rows = DS4_N_EXPERT_USED * 2304u, .group_rows = 2304, .deps = 1u << 1,
            .finish = ds41_cpu_ffn_routed_midq, .finish_ctx = s };
        stages[6] = (ds4_pool_stage){ .fn = matvec_q2_k_accum_worker, .ctx = &g->q8k_down,
            .n_rows = DS4_N_EXPERT_USED * 5120u, .group_rows = 5120, .deps = 1u << 3 };
    }
}

static void ds41_cpu_ffn_graph_bytes(const ds4_layer_weights *l, int phase) {
    if (ds41_prof_on)
        ds41_prof_bytes[phase] += l->hc_ffn_fn->bytes + l->ffn_gate_inp->bytes +
            l->ffn_gate_shexp->bytes + l->ffn_up_shexp->bytes + l->ffn_down_shexp->bytes +
            DS4_N_EXPERT_USED * ds41_cpu_routed_expert_bytes(l);
}

static void ds41_cpu_ffn_graph_run(ds41_cpu_state *s, const ds4_layer_weights *l,
                                   const ds41_cpu_hc_job *hc) {
    ds41_cpu_ffn_graph g;
    ds4_pool_stage stages[8];
    ds41_cpu_ffn_graph_stages(&g, s, l, hc, stages);
    ds4_pool_run_stages(stages, 8);
    ds41_cpu_ffn_graph_bytes(l, DS41_PROF_FFN);
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
        /* The staged FFN runs in the attention output's dispatch when both
         * are staged: its stages read their inputs only when they run. */
        ds4_pool_stage ffn_stages[8];
        uint32_t n_ffn = 0;
        bool ffn_done = false;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        ds41_cpu_ffn_graph ffn;
        const ds41_cpu_hc_job ffn_hc = ds41_cpu_hc_mixer(s, l, true);
        if (ds41_cpu_ffn_graph_supported(l, &ffn_hc)) {
            ds41_cpu_ffn_graph_stages(&ffn, s, l, &ffn_hc, ffn_stages);
            n_ffn = 8;
        }
#endif
        ds41_cpu_hc_sum(s->x, s->residual, s->pre);
        if (!ds41_cpu_norm(s->norm, s->x, s->model, l->attn_norm)) goto layer_fail;
        double phase = profile ? now_sec() : 0;
        if (!ds41_cpu_attention_forward(s->attention, s->model, l, il, s->pos, s->norm,
                                        s->block, &hc, &post, ffn_stages, n_ffn, &ffn_done))
            goto layer_fail;
        if (profile) attention_seconds += now_sec() - phase;
        ds41_prof(DS41_PROF_HC);
        phase = profile ? now_sec() : 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        if (ffn_done) ds41_cpu_ffn_graph_bytes(l, DS41_PROF_OUT);
#endif
        if (!ffn_done && !ds41_cpu_ffn(s, l)) goto layer_fail;
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
 * the chunk's tokens that use it.  The raw window rows are copied in
 * position order and the compressed rows published in parallel, each from
 * its own tokens; the query side (indexing, staging, the heads), HC, norms,
 * the router and the other glue run per token in parallel.  Every value is
 * computed by the same operations as in ds41_cpu_forward, so the logits
 * are bitwise those of the token loop. */
typedef struct {
    ds41_cpu_state *s;
    ds41_cpu_batch *b;
    const ds4_layer_weights *l;
    uint32_t il, n;
    bool mixed;     /* mix holds the HC mixer projections */
    bool routed;    /* probs holds the router, partial the routed experts */
    int failed;
} ds41_cpu_batch_ctx;

static void ds41_cpu_batch_fail(ds41_cpu_batch_ctx *c) {
    __atomic_store_n(&c->failed, 1, __ATOMIC_RELAXED);
}

static void ds41_cpu_engram_batch_finish(ds41_cpu_engram_batch *e, bool stop);

static void ds41_cpu_batch_free(ds41_cpu_batch *b) {
    if (!b) return;
    if (b->ahead_live) ds41_cpu_engram_batch_finish(&b->ahead, true);
    free(b->residual); free(b->after_attn); free(b->flat); free(b->x);
    free(b->norm); free(b->block); free(b->shared); free(b->pre);
    free(b->attn_split); free(b->ffn_split); free(b->qr); free(b->q);
    free(b->kv); free(b->heads); free(b->low); free(b->rows);
    free(b->engram_kv); free(b->shared_gate); free(b->shared_up);
    free(b->shared_mid); free(b->mid); free(b->partial); free(b->weight);
    free(b->selected); free(b->ids); free(b->entry); free(b->sel);
    free(b->mix); free(b->probs); free(b->xt);
    free(b->ckv); free(b->cscore); free(b->iq); free(b->iw);
    free(b->prev_window); free(b->rounded_raw); free(b->rounded_comp);
    free(b->attn); free(b->next_ids); free(b->next_rows);
    free(b);
}

static ds41_cpu_batch_attention_scratch *ds41_cpu_batch_attention_thread(ds41_cpu_batch *b) {
    uint8_t *p = b->attn + (size_t)g_pool_thread * b->attn_stride;
    ds41_cpu_batch_attention_scratch *sc = (ds41_cpu_batch_attention_scratch *)p;
    p += (sizeof(*sc) + 63u) & ~(size_t)63u;
    sc->qt = (float *)p;
    p += 64u * 512u * sizeof(float);
    sc->score = (float *)p;
    p += 64u * (128u + 512u) * sizeof(float);
    sc->index_scores = (float *)p;
    return sc;
}

static ds41_cpu_batch *ds41_cpu_batch_create(uint32_t ctx) {
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
    b->next_rows = malloc(n * 2u * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * f);
    b->next_ids = malloc(n * 2u * DS4_ENGRAM_COLS * sizeof(uint32_t));
    b->entry = malloc(pairs * sizeof(uint32_t));
    b->sel = malloc(n * 512u * sizeof(uint32_t));
    b->mix = malloc(n * 24u * f);
    b->probs = malloc(n * DS4_N_EXPERT * f);
    b->xt = malloc(n * 4u * 5120u * sizeof(double));
    b->ckv = malloc(n * 512u * f);
    b->cscore = malloc(n * 512u * f);
    b->iq = malloc(n * 32u * 128u * f);
    b->iw = malloc(n * 32u * f);
    b->prev_window = malloc(128u * 512u * f);
    b->rounded_raw = malloc((128u + n) * 512u * f);
    b->attn_stride = ds41_cpu_batch_attention_stride(ctx);
    void *attn = NULL;
    if (posix_memalign(&attn, 64, (size_t)DS4_MAX_THREADS * b->attn_stride) == 0)
        b->attn = attn;
    if (!b->residual || !b->after_attn || !b->flat || !b->x || !b->norm ||
        !b->block || !b->shared || !b->pre || !b->attn_split || !b->ffn_split ||
        !b->qr || !b->q || !b->kv || !b->heads || !b->low || !b->rows ||
        !b->engram_kv || !b->shared_gate || !b->shared_up || !b->shared_mid ||
        !b->mid || !b->partial || !b->weight || !b->selected || !b->ids || !b->entry ||
        !b->sel || !b->mix || !b->probs || !b->xt || !b->ckv || !b->cscore ||
        !b->iq || !b->iw || !b->prev_window || !b->rounded_raw || !b->attn ||
        !b->next_rows || !b->next_ids) {
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
                           matrix.dim[1], 0, round) ||
        ds41_cpu_matmul_f16(out, out_stride, m, &matrix, x, x_stride, n, round)) return true;
    for (uint32_t t = 0; t < n; t++)
        if (!ds41_cpu_matvec(out + (size_t)t * out_stride, m, w,
                             x + (size_t)t * x_stride, round)) return false;
    return true;
}

static void *ds41_cpu_engram_batch_thread(void *vctx) {
    ds41_cpu_engram_batch *e = vctx;
    const size_t cols = DS4_ENGRAM_COLS;
    for (uint32_t i = 0; i < 2; i++) {
        for (;;) {
            const uint32_t t = __atomic_fetch_add(&e->next[i], 1, __ATOMIC_RELAXED);
            if (t >= e->n) break;
            if (!__atomic_load_n(&e->stop, __ATOMIC_RELAXED) &&
                !ds4_engram_read(&e->table[i], e->ids + ((size_t)t * 2u + i) * cols, cols,
                                 e->rows + ((size_t)t * 2u + i) * cols * DS4_ENGRAM_DIM))
                __atomic_store_n(&e->failed, 1, __ATOMIC_RELAXED);
            if (__atomic_add_fetch(&e->done[i], 1, __ATOMIC_ACQ_REL) == e->n) {
                pthread_mutex_lock(&e->mutex);
                pthread_cond_broadcast(&e->cond);
                pthread_mutex_unlock(&e->mutex);
            }
        }
    }
    return NULL;
}

/* Start the reads; without helper threads they run here before returning. */
static void ds41_cpu_engram_batch_start(ds41_cpu_engram_batch *e, const ds4_engram_table table[2],
                                        const uint32_t *ids, float *rows, uint32_t n) {
    *e = (ds41_cpu_engram_batch){.table = table, .ids = ids, .rows = rows, .n = n};
    pthread_mutex_init(&e->mutex, NULL);
    pthread_cond_init(&e->cond, NULL);
    const uint32_t readers = n < DS41_CPU_ENGRAM_READERS ? n : DS41_CPU_ENGRAM_READERS;
    while (e->started < readers &&
           !pthread_create(&e->threads[e->started], NULL, ds41_cpu_engram_batch_thread, e))
        e->started++;
    if (!e->started) ds41_cpu_engram_batch_thread(e);
}

/* Wait for the rows of table i; false when a read failed. */
static bool ds41_cpu_engram_batch_wait(ds41_cpu_engram_batch *e, uint32_t i) {
    pthread_mutex_lock(&e->mutex);
    while (__atomic_load_n(&e->done[i], __ATOMIC_ACQUIRE) < e->n)
        pthread_cond_wait(&e->cond, &e->mutex);
    pthread_mutex_unlock(&e->mutex);
    return !__atomic_load_n(&e->failed, __ATOMIC_RELAXED);
}

/* Join the readers, early (skipping the reads left) when stop is set. */
static void ds41_cpu_engram_batch_finish(ds41_cpu_engram_batch *e, bool stop) {
    if (stop) __atomic_store_n(&e->stop, 1, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < e->started; i++) pthread_join(e->threads[i], NULL);
    pthread_cond_destroy(&e->cond);
    pthread_mutex_destroy(&e->mutex);
}

/* The readers of a chunk's rows, b->ids of n tokens: the read ahead when it
 * read these very ids, whose rows become b->rows (they are the bytes reading
 * the ids now returns), else own started on them once any other read ahead
 * is stopped. */
static ds41_cpu_engram_batch *ds41_cpu_engram_batch_take(ds41_cpu_batch *b,
                                                         const ds4_engram_table table[2],
                                                         uint32_t n, ds41_cpu_engram_batch *own) {
    if (b->ahead_live) {
        b->ahead_live = false;
        if (b->next_n == n &&
            !memcmp(b->next_ids, b->ids, (size_t)n * 2u * DS4_ENGRAM_COLS * sizeof(*b->ids))) {
            float *rows = b->rows;
            b->rows = b->next_rows;
            b->next_rows = rows;
            return &b->ahead;
        }
        ds41_cpu_engram_batch_finish(&b->ahead, true);
    }
    ds41_cpu_engram_batch_start(own, table, b->ids, b->rows, n);
    return own;
}

/* Start reading ahead the rows of the next chunk's next_n tokens into
 * b->next_rows, hashed after history, the hash state its own chunk starts
 * from; nothing when a token is outside the vocabulary. */
static void ds41_cpu_engram_batch_ahead(ds41_cpu_batch *b, const ds4_engram_layout *layout,
                                        ds4_engram_history history,
                                        const ds4_engram_table table[2],
                                        const int *next, uint32_t next_n) {
    for (uint32_t i = 0; i < next_n; i++)
        if (!ds4_engram_hash(layout, &history, &next[i], NULL, 1,
                             b->next_ids + (size_t)i * 2u * DS4_ENGRAM_COLS)) return;
    ds41_cpu_engram_batch_start(&b->ahead, table, b->next_ids, b->next_rows, next_n);
    b->next_n = next_n;
    b->ahead_live = true;
}

/* Engram and the flat HC input of each token before attention. */
static void ds41_cpu_batch_attn_flat_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    const ds4_model *m = c->s->model;
    const ds4_layer_weights *l = c->l;
    for (uint64_t t = t0; t < t1; t++) {
        float *residual = b->residual + t * 4u * 5120u;
        if (c->il == 1 || c->il == 14)
            ds41_cpu_engram_add(residual, b->engram_kv + t * 5u * 5120u,
                               tensor_data(m, l->engram_q_norm),
                               tensor_data(m, l->engram_k_norm), 5120, DS4_RMS_EPS);
        rms_norm_no_weight(b->flat + t * 4u * 5120u, residual, 4u * 5120u, DS4_RMS_EPS);
    }
}

/* The HC split of the mixer projection, computed here unless batched. */
static bool ds41_cpu_batch_split(ds41_cpu_batch_ctx *c, uint64_t t, bool ffn, float *split) {
    const ds4_model *m = c->s->model;
    const ds4_layer_weights *l = c->l;
    float *mix = c->b->mix + t * 24u;
    if (!c->mixed && !ds41_cpu_matvec(mix, m, ffn ? l->hc_ffn_fn : l->hc_attn_fn,
                                      c->b->flat + t * 4u * 5120u, false)) return false;
    hc_split_sinkhorn_one(split, mix, tensor_data(m, ffn ? l->hc_ffn_scale : l->hc_attn_scale),
                         tensor_data(m, ffn ? l->hc_ffn_base : l->hc_attn_base),
                         4, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS);
    return true;
}

/* The attention HC split, input and norm of each token. */
static void ds41_cpu_batch_attn_in_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    for (uint64_t t = t0; t < t1; t++) {
        if (!ds41_cpu_batch_split(c, t, false, b->attn_split + t * 24u)) {
            ds41_cpu_batch_fail(c);
            continue;
        }
        ds41_cpu_hc_sum(b->x + t * 5120u, b->residual + t * 4u * 5120u, b->pre + t * 4u);
        if (!ds41_cpu_norm(b->norm + t * 5120u, b->x + t * 5120u, c->s->model, c->l->attn_norm))
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

/* HC after attention and the flat input of the FFN mixer of each token. */
static void ds41_cpu_batch_ffn_flat_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    for (uint64_t t = t0; t < t1; t++) {
        float *after = b->after_attn + t * 4u * 5120u;
        ds41_cpu_hc_expand(after, b->block + t * 5120u, b->residual + t * 4u * 5120u,
                          b->attn_split + t * 24u);
        rms_norm_no_weight(b->flat + t * 4u * 5120u, after, 4u * 5120u, DS4_RMS_EPS);
    }
}

/* The FFN HC split, input and norm of each token. */
static void ds41_cpu_batch_ffn_in_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    for (uint64_t t = t0; t < t1; t++) {
        if (!ds41_cpu_batch_split(c, t, true, b->ffn_split + t * 24u)) {
            ds41_cpu_batch_fail(c);
            continue;
        }
        ds41_cpu_hc_sum(b->x + t * 5120u, b->after_attn + t * 4u * 5120u, b->attn_split + t * 24u);
        if (!ds41_cpu_norm(b->norm + t * 5120u, b->x + t * 5120u, c->s->model, c->l->ffn_norm))
            ds41_cpu_batch_fail(c);
    }
}

/* Expert selection of each token from the router, computed here unless batched. */
static void ds41_cpu_batch_router_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    const ds4_model *m = c->s->model;
    for (uint64_t t = t0; t < t1; t++) {
        float *probs = b->probs + t * DS4_N_EXPERT;
        if (!c->routed && !ds41_cpu_matvec(probs, m, c->l->ffn_gate_inp,
                                           b->norm + t * 5120u, false)) {
            ds41_cpu_batch_fail(c);
            continue;
        }
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++)
            probs[i] = sqrtf(softplus_stable(probs[i]));
        layer_topk_selected_experts_from_probs(b->selected + t * DS4_N_EXPERT_USED,
                                               b->weight + t * DS4_N_EXPERT_USED, m, c->l, probs);
    }
}

static void ds41_cpu_batch_swiglu_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        float *mid = c->b->shared_mid + t * 2304u;
        const float *gate = c->b->shared_gate + t * 2304u, *up = c->b->shared_up + t * 2304u;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        /* silu(g) * u * 1 is silu(g) * u exactly. */
        for (uint32_t i = 0; i < 2304; i += 16)
            _mm512_storeu_ps(mid + i, ds41_cpu_swiglu16(_mm512_loadu_ps(gate + i),
                                                        _mm512_loadu_ps(up + i),
                                                        DS4_SWIGLU_CLAMP_EXP, 1.0f));
#else
        swiglu(mid, gate, up, 2304, DS4_SWIGLU_CLAMP_EXP);
#endif
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

/* RoPE and quantization of each token's KV row, which depend on the token
 * and its position only. */
static void ds41_cpu_batch_kv_rope_worker(void *vctx, uint64_t t0, uint64_t t1) {
    ds41_cpu_batch_ctx *c = vctx;
    const uint32_t ratio = ds4_layer_compress_ratio(c->il);
    for (uint64_t t = t0; t < t1; t++) {
        float *kv = c->b->kv + t * 512u;
        ds41_cpu_rope(kv, 512, 1, c->s->pos + (uint32_t)t, ratio != 0, false);
        ds41_cpu_quantize(kv, 512, 1, DS41_CPU_FP8_E8M0);
    }
}

/* The compressed rows a KV source layer completes in the chunk, in
 * parallel: row r is completed by the token at pos = first + r * ratio.
 * Pooling and publishing a row read only its tokens' compressor
 * projections, so each row is the one ds41_cpu_attention_publish computes
 * at pos.  A ratio 2 row completed by the chunk's first token pools it
 * with the previous chunk's last token, kept in previous_kv. */
static void ds41_cpu_batch_publish_worker(void *vctx, uint64_t r0, uint64_t r1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_state *s = c->s;
    ds41_cpu_batch *b = c->b;
    ds41_cpu_attention_state *as = s->attention;
    const uint32_t il = c->il, ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    const uint32_t first = ratio == 2u ? s->pos | 1u : s->pos;
    float latent[512], index_k[128];
    for (uint64_t r = r0; r < r1; r++) {
        const uint32_t pos = first + (uint32_t)r * ratio, i = pos - s->pos;
        if (ratio == 2u) {
            float *kv = b->ckv + (size_t)i * 512u, *score = b->cscore + (size_t)i * 512u;
            ds41_cpu_attention_pool2(latent, i ? kv - 512 : as->previous_kv[owner],
                                     i ? score - 512 : as->previous_score[owner],
                                     kv, score, pos);
        } else memcpy(latent, b->ckv + (size_t)i * 512u, sizeof(latent));
        if (!ds41_cpu_attention_publish_row(as, s->model, c->l, owner, ratio, pos,
                                            latent, index_k))
            ds41_cpu_batch_fail(c);
    }
}

/* The KV side of the chunk's attention: RoPE and quantization of each
 * token's KV (in parallel), then in position order its copy into the
 * layer's raw window and, on the KV source layers, the compressed rows and
 * their indexer keys, as ds41_cpu_attention_cache does them token by
 * token.  A ratio 2 layer then keeps the chunk's last even position's
 * projections as the first half of the next pair, as the token loop's
 * pooling leaves them. */
static bool ds41_cpu_batch_attention_kv(ds41_cpu_batch_ctx *c, uint32_t n) {
    ds41_cpu_state *s = c->s;
    ds41_cpu_batch *b = c->b;
    ds41_cpu_attention_state *as = s->attention;
    const uint32_t il = c->il, ratio = ds4_layer_compress_ratio(il);
    /* The raw rows before the chunk, which the chunk's rows overwrite. */
    memcpy(b->prev_window, as->window[il], 128u * 512u * sizeof(float));
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_kv_rope_worker, c, 1);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t pos = s->pos + i;
        const float *kv = b->kv + (size_t)i * 512u;
        memcpy(as->window[il] + (size_t)(pos % 128u) * 512u, kv, 512u * sizeof(float));
    }
    if (!ds41_kv_source(il) || !n) return true;
    if (ratio != 1u && ratio != 2u) return false;
    const uint32_t rows = (s->pos + n) / ratio - s->pos / ratio;
    ds4_parallel_for_min_rows(rows, ds41_cpu_batch_publish_worker, c, 2);
    const uint32_t last_even = (s->pos + n - 1u) & ~1u;
    if (ratio == 2u && last_even >= s->pos) {
        const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : 2u;
        const size_t i = last_even - s->pos;
        memcpy(as->previous_kv[owner], b->ckv + i * 512u, sizeof(as->previous_kv[owner]));
        memcpy(as->previous_score[owner], b->cscore + i * 512u,
               sizeof(as->previous_score[owner]));
    }
    return !c->failed;
}

/* Rows r of the chunk's attended rows rounded to F16: the 128 raw window
 * rows before the chunk, the chunk's n KV rows, then the compressed rows
 * of the layer's owner.  The rounding is a function of each row, which no
 * query of the chunk changes, so a token reads the values the copy it
 * would round for itself holds. */
static void ds41_cpu_batch_round_worker(void *vctx, uint64_t r0, uint64_t r1) {
    ds41_cpu_batch_ctx *c = vctx;
    ds41_cpu_batch *b = c->b;
    const uint32_t il = c->il, n = c->n;
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    for (uint64_t r = r0; r < r1; r++) {
        if (r < 128u + n) {
            const float *src = r < 128u ? b->prev_window + r * 512u : b->kv + (r - 128u) * 512u;
            ds41_cpu_f16_round(b->rounded_raw + r * 512u, src, 512);
        } else {
            const uint64_t row = r - 128u - n;
            ds41_cpu_f16_round(b->rounded_comp + row * 512u,
                               c->s->attention->compressed[owner] + row * 512u, 512);
        }
    }
}

/* The query side of one token's attention after the KV pass: index scores
 * and selection on an index source, staging and the sink softmax heads.
 * The token reads only cache rows up to its own position, which the KV
 * pass left as the token loop would have at that position, and every step
 * is the one of ds41_cpu_attention_cache and ds41_cpu_attention_attend
 * on the same inputs: the lanes workers compute each of their items the
 * same way whatever the split. */
static void ds41_cpu_batch_attention_token(ds41_cpu_batch_ctx *c,
                                           ds41_cpu_batch_attention_scratch *sc,
                                           uint32_t i) {
    ds41_cpu_state *s = c->s;
    ds41_cpu_batch *b = c->b;
    ds41_cpu_attention_state *as = s->attention;
    const uint32_t il = c->il, pos = s->pos + i;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0u;
    uint32_t *sel = b->sel + (size_t)i * 512u;
    float *q = b->q + (size_t)i * 64u * 512u, *heads = b->heads + (size_t)i * 64u * 512u;
    ds41_cpu_rope(q, 512, 64, pos, ratio != 0, false);
    if (n_comp && ds41_index_source(il)) {
        float *iq = b->iq + (size_t)i * 32u * 128u;
        ds41_cpu_rope(iq, 128, 32, pos, true, false);
        ds41_cpu_quantize(iq, 128, 32, DS41_CPU_FP4_E8M0);
        ds41_cpu_index_score_context score = {
            iq, as->index_cache[owner], b->iw + (size_t)i * 32u, sc->index_scores
        };
        ds41_cpu_index_scores(&score, n_comp, sc->qt);
        /* Within 2048 blocks ds41_cpu_attention_candidates only clears the
         * block mask on layer 20, so the later sources mask nothing. */
        ds41_cpu_attention_topk(sel, sc->index_scores, n_comp,
                                n_comp < 512u ? n_comp : 512u, sc->heap);
    }
    const uint32_t attended = n_comp < 512u ? n_comp : 512u;
    const uint32_t n_raw = pos < 127u ? pos + 1u : 128u, n_rows = n_raw + attended;
    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t p = pos + 1u - n_raw + r;
        const uint32_t row = p >= s->pos ? 128u + (p - s->pos) : p % 128u;
        sc->rows[r] = b->rounded_raw + (size_t)row * 512u;
    }
    for (uint32_t r = 0; r < attended; r++)
        sc->rows[n_raw + r] = b->rounded_comp + (size_t)sel[r] * 512u;
    ds41_cpu_f16_round(q, q, 64u * 512u);
    attn_lanes_transpose_q(sc->qt, q, 64, 512);
    layer_attention_lanes_ctx ctx = {
        .out_heads = heads,
        .sinks = tensor_data(s->model, c->l->attn_sinks),
        .q = q,
        .qt = sc->qt,
        .rows = sc->rows,
        .score = sc->score,
        .inv = sc->inv,
        .n_head = 64,
        .score_stride = 128 + 512,
        .n_rows = n_rows,
        .n_chunks = (n_rows + ATTN_LANES_ROWS_PER_ITEM - 1) / ATTN_LANES_ROWS_PER_ITEM,
        .n_dim_chunks = 512 / 16,
    };
    ds41_cpu_attention_scores(&ctx);
    ds41_cpu_attention_weight_worker(&ctx, 0, 64);
    attn_lanes_sum_worker(&ctx, 0, 4u * ctx.n_dim_chunks);
    ds41_cpu_bf16_rows(heads, 64u * 512u);
    ds41_cpu_rope(heads, 512, 64, pos, ratio != 0, true);
}

/* Item p of (n + 1) / 2 pairs the short early token p with the long late
 * token n - 1 - p, as V4's prefill does, to balance the claims.  Every
 * value a token reads from its thread's scratch it has written first. */
static void ds41_cpu_batch_attention_worker(void *vctx, uint64_t p0, uint64_t p1) {
    ds41_cpu_batch_ctx *c = vctx;
    const uint32_t n = c->n;
    ds41_cpu_batch_attention_scratch *sc = ds41_cpu_batch_attention_thread(c->b);
    for (uint64_t p = p0; p < p1; p++) {
        ds41_cpu_batch_attention_token(c, sc, (uint32_t)p);
        if (n - 1u - (uint32_t)p != (uint32_t)p)
            ds41_cpu_batch_attention_token(c, sc, n - 1u - (uint32_t)p);
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
    for (uint32_t ffn = 0; ffn < 2; ffn++) {
        const ds4_tensor *scale = ffn ? l->hc_ffn_scale : l->hc_attn_scale;
        const ds4_tensor *base = ffn ? l->hc_ffn_base : l->hc_attn_base;
        if (!scale || !base || scale->type != DS4_TENSOR_F32 || base->type != DS4_TENSOR_F32)
            return false;
    }
    double t = phase ? now_sec() : 0;
#define DS41_PHASE(i) do { if (phase) { const double now = now_sec(); \
    phase[i] += now - t; t = now; } } while (0)
    if (il == 1 || il == 14) {
        const size_t cols = (size_t)DS4_ENGRAM_COLS * DS4_ENGRAM_DIM;
        if (!ds41_cpu_matmul(b->engram_kv, 5u * 5120u, m, l->engram_kv,
                             b->rows + (il == 1 ? 0 : cols), 2u * cols, n, true)) return false;
    }
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_attn_flat_worker, &c, 1);
    c.mixed = ds41_cpu_matmul_f32(b->mix, 24, m, l->hc_attn_fn, b->flat, 4u * 5120u, n, b->xt) ||
        ds41_cpu_matmul_f16(b->mix, 24, m, l->hc_attn_fn, b->flat, 4u * 5120u, n, false);
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_attn_in_worker, &c, 1);
    if (c.failed) return false;
    DS41_PHASE(0);
    if (!ds41_cpu_matmul(b->qr, 1280, m, l->attn_q_a, b->norm, 5120, n, true) ||
        !ds41_cpu_matmul(b->kv, 512, m, l->attn_kv, b->norm, 5120, n, true)) return false;
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_qkv_norm_worker, &c, 1);
    if (c.failed ||
        !ds41_cpu_matmul(b->q, 64u * 512u, m, l->attn_q_b, b->qr, 1280, n, true)) return false;
    DS41_PHASE(1);
    /* The compressor and indexer projections of the chunk, which the
     * token loop below would otherwise compute one token at a time. */
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool kv_source = ds41_kv_source(il), index = ds41_index_source(il);
    if (kv_source &&
        (!ds41_cpu_matmul(b->ckv, 512, m, l->attn_compressor_kv, b->norm, 5120, n, ratio == 1u) ||
         (ratio == 2u && !ds41_cpu_matmul(b->cscore, 512, m, l->attn_compressor_gate,
                                          b->norm, 5120, n, false)))) return false;
    if (index &&
        (!ds41_cpu_matmul(b->iq, 32u * 128u, m, l->indexer_attn_q_b, b->qr, 1280, n, true) ||
         !ds41_cpu_matmul(b->iw, 32, m, l->indexer_proj, b->norm, 5120, n, true))) return false;
    /* The KV side in position order, then the query side of the tokens in
     * parallel.  A compressed layer that is no index source attends the
     * rows its token's last index source selected, kept per token in sel;
     * the state keeps the last token's selection, as after the token loop. */
    if (!ds41_cpu_batch_attention_kv(&c, n)) return false;
    c.n = n;
    /* The rows the queries read, rounded once for the chunk rather than
     * copied and rounded by every token. */
    const uint64_t comp_rows = ratio ? ((uint64_t)s->pos + n) / ratio : 0;
    if (comp_rows > b->rounded_rows) {
        float *grown = realloc(b->rounded_comp, comp_rows * 512u * sizeof(float));
        if (!grown) return false;
        b->rounded_comp = grown;
        b->rounded_rows = comp_rows;
    }
    ds4_parallel_for_min_rows(128u + n + comp_rows, ds41_cpu_batch_round_worker, &c, 16);
    ds4_parallel_for_min_rows((n + 1u) / 2u, ds41_cpu_batch_attention_worker, &c, 1);
    if (c.failed) return false;
    if (ratio && (s->pos + n) / ratio)
        memcpy(as->selected_comp, b->sel + (size_t)(n - 1u) * 512u, sizeof(as->selected_comp));
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
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_ffn_flat_worker, &c, 1);
    c.mixed = ds41_cpu_matmul_f32(b->mix, 24, m, l->hc_ffn_fn, b->flat, 4u * 5120u, n, b->xt) ||
        ds41_cpu_matmul_f16(b->mix, 24, m, l->hc_ffn_fn, b->flat, 4u * 5120u, n, false);
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_ffn_in_worker, &c, 1);
    if (c.failed) return false;
    c.routed = ds41_cpu_matmul_f32(b->probs, DS4_N_EXPERT, m, l->ffn_gate_inp, b->norm, 5120,
                                   n, b->xt);
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_router_worker, &c, 1);
    if (c.failed) return false;
    DS41_PHASE(4);
    if (!ds41_cpu_matmul(b->shared_gate, 2304, m, l->ffn_gate_shexp, b->norm, 5120, n, true) ||
        !ds41_cpu_matmul(b->shared_up, 2304, m, l->ffn_up_shexp, b->norm, 5120, n, true))
        return false;
    ds4_parallel_for_min_rows(n, ds41_cpu_batch_swiglu_worker, &c, 1);
    if (!ds41_cpu_matmul(b->shared, 5120, m, l->ffn_down_shexp, b->shared_mid, 2304, n, true))
        return false;
    DS41_PHASE(5);
    /* Q2 experts write block; the F32 batch writes the partials. */
    const bool q8k = ds41_cpu_routed_moe_q8k_batch(b->block, m, l, b->norm, n,
                                                   b->selected, b->weight);
    c.routed = !q8k && ds41_cpu_routed_moe_batch(b->partial, b->mid, b->entry, m, l, b->norm,
                                                 5120, n, b->selected, b->weight);
    if (!q8k && !c.routed) {
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

/* Whether a chunk of n tokens at pos runs layer-major.  Past 2048 blocks
 * layer 20 masks blocks for the later index sources of its own token only;
 * such chunks keep the token loop. */
static bool ds41_cpu_prefill_batched(uint32_t pos, uint32_t n) {
    const uint32_t ratio20 = ds4_layer_compress_ratio(20);
    const uint64_t last_comp = ratio20 ? ((uint64_t)pos + n) / ratio20 : 0;
    return n > 1 && n <= DS41_CPU_BATCH && (last_comp + 7u) / 8u <= 2048u &&
        (uint64_t)pos + n <= DS41_CPU_BATCH_COMP;
}

/* Up to DS41_CPU_BATCH prompt tokens at the current position; logits, when
 * given, receive the last token's.  next (next_n tokens, maybe none) is the
 * chunk the caller prefills after this one: its Engram rows are read while
 * the layers after 14 run, as this chunk's are while the layers before. */
static bool ds41_cpu_prefill(ds41_cpu_state *s, const int *tokens, uint32_t n,
                             const int *next, uint32_t next_n,
                             float *logits, char *error, size_t cap) {
    if (!s || !s->valid) return ds41_cpu_error(error, cap, "invalid V4.1 CPU state; reset required");
    if (n > DS41_CPU_BATCH) return ds41_cpu_error(error, cap, "V4.1 CPU prefill chunk too large");
    if (n > s->ctx - s->pos) return ds41_cpu_error(error, cap, "V4.1 CPU context is full");
    if (!ds41_cpu_prefill_batched(s->pos, n)) {
        for (uint32_t i = 0; i < n; i++)
            if (!ds41_cpu_forward(s, tokens[i], i + 1u == n ? logits : NULL, error, cap))
                return false;
        return true;
    }
    for (uint32_t i = 0; i < n; i++)
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB)
            return ds41_cpu_error(error, cap, "V4.1 CPU token is outside vocabulary");
    if (!s->batch && !(s->batch = ds41_cpu_batch_create(s->ctx)))
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
    ds41_cpu_engram_batch own, *rows = ds41_cpu_engram_batch_take(b, s->table, n, &own);
    double read_seconds = profile ? now_sec() - started : 0;
    for (uint32_t i = 0; i < n; i++) {
        float *x = b->x + (size_t)i * 5120u, *pre = b->pre + (size_t)i * 4u;
        embed_token_any(s->model, s->weights, tokens[i], x);
        hc_from_plain_embedding(b->residual + (size_t)i * 4u * 5120u, x, 5120, 4);
        pre[0] = 1.0f;
        pre[1] = pre[2] = pre[3] = 0.0f;
    }
    /* moe_grouped_ffn's workspace is freed on every return, as V4's
     * prefill frees it. */
    for (uint32_t il = 0; il < 40; il++) {
        if (il == 1 || il == 14) {
            const double wait = profile ? now_sec() : 0;
            const bool ok = ds41_cpu_engram_batch_wait(rows, il == 1 ? 0u : 1u);
            if (il == 14) ds41_cpu_engram_batch_finish(rows, false);
            if (profile) read_seconds += now_sec() - wait;
            if (!ok) {
                if (il == 1) ds41_cpu_engram_batch_finish(rows, true);
                pf_bufs_free();
                s->valid = false;
                return ds41_cpu_error(error, cap, "V4.1 Engram row read failed");
            }
        }
        /* The readers are free: start on the next chunk's rows, hashed
         * after this chunk's tokens as its own call will hash them. */
        if (il == 14 && next && ds41_cpu_prefill_batched(s->pos + n, next_n))
            ds41_cpu_engram_batch_ahead(b, &s->engram, history, s->table, next, next_n);
        if (!ds41_cpu_prefill_layer(s, il, n, profile ? phase : NULL)) {
            if (il < 14) ds41_cpu_engram_batch_finish(rows, true);
            pf_bufs_free();
            s->valid = false;
            if (error && cap) snprintf(error, cap, "V4.1 CPU prefill layer %u failed at token position %u", il, s->pos);
            return false;
        }
    }
    pf_bufs_free();
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
