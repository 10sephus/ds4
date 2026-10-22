/* V4.1 CPU attention. Included after the dense projection and norm helpers.
 * Cache ownership follows the encoder's four publishers; every layer keeps
 * its own 128-token raw window. All scratch is allocated with the session. */
typedef struct {
    float score;
    uint32_t index;
} ds41_cpu_candidate;

typedef struct {
    uint32_t ctx, comp_cap[4];
    float *window[40], *compressed[4], *index_cache[4];
    float previous_kv[3][512], previous_score[3][512];
    float *index_scores, *block_scores;
    uint8_t *block_mask;
    uint32_t selected_comp[512], selected_blocks[2048];
    ds41_cpu_candidate heap[2048];
    float qr[1280], q[64 * 512], kv[512], heads[64 * 512], low[8192];
    float pool_kv[512], pool_score[512], latent[512];
    float index_q[32 * 128], index_k[128], index_weights[32];
    float attended_kv[(128 + 512) * 512];
    /* layer_attention_lanes scratch: transposed query, row list, scores
     * [head group][row][16] and 1 / denominator per head. */
    float qt[64 * 512], score[64 * (128 + 512)], inv[64];
    const float *rows[128 + 512];
} ds41_cpu_attention_state;

static void ds41_cpu_attention_free(ds41_cpu_attention_state *s) {
    if (!s) return;
    for (uint32_t il = 0; il < 40; il++) free(s->window[il]);
    for (uint32_t i = 0; i < 4; i++) {
        free(s->compressed[i]);
        free(s->index_cache[i]);
    }
    free(s->index_scores);
    free(s->block_scores);
    free(s->block_mask);
    free(s);
}

static ds41_cpu_attention_state *ds41_cpu_attention_create(uint32_t ctx) {
    if (!ctx || ctx == UINT32_MAX) return NULL;
    ds41_cpu_attention_state *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->ctx = ctx;
    for (uint32_t il = 0; il < 40; il++) {
        s->window[il] = calloc(128u * 512u, sizeof(float));
        if (!s->window[il]) goto fail;
    }
    for (uint32_t i = 0; i < 4; i++) {
        s->comp_cap[i] = ctx / (i < 3 ? 2u : 1u) + 1u;
        s->compressed[i] = calloc((size_t)s->comp_cap[i] * 512u, sizeof(float));
        s->index_cache[i] = calloc((size_t)s->comp_cap[i] * 128u, sizeof(float));
        if (!s->compressed[i] || !s->index_cache[i]) goto fail;
    }
    const size_t blocks = ((size_t)ctx + 7u) / 8u;
    s->index_scores = calloc((size_t)ctx + 1u, sizeof(float));
    s->block_scores = calloc(blocks, sizeof(float));
    s->block_mask = calloc(blocks, sizeof(uint8_t));
    if (!s->index_scores || !s->block_scores || !s->block_mask) goto fail;
    return s;
fail:
    ds41_cpu_attention_free(s);
    return NULL;
}

static void ds41_cpu_attention_reset(ds41_cpu_attention_state *s) {
    if (!s) return;
    /* Position determines the live cache prefix. Resetting it in the caller
     * makes old windows and compressed rows unreachable until overwritten. */
    memset(s->previous_kv, 0, sizeof(s->previous_kv));
    memset(s->previous_score, 0, sizeof(s->previous_score));
    memset(s->selected_comp, 0, sizeof(s->selected_comp));
    memset(s->block_mask, 0, ((size_t)s->ctx + 7u) / 8u);
}

static bool ds41_cpu_candidate_better(ds41_cpu_candidate a, ds41_cpu_candidate b) {
    return a.score > b.score || (a.score == b.score && a.index < b.index);
}

static int ds41_cpu_candidate_compare(const void *va, const void *vb) {
    const ds41_cpu_candidate a = *(const ds41_cpu_candidate *)va;
    const ds41_cpu_candidate b = *(const ds41_cpu_candidate *)vb;
    return ds41_cpu_candidate_better(a, b) ? -1 :
           ds41_cpu_candidate_better(b, a) ? 1 : 0;
}

/* A worst-first heap bounds selection scratch independently of context.
 * Ties use the lower cache row, as in the CUDA top-k reference. */
static void ds41_cpu_attention_topk(uint32_t *out, const float *scores,
                                    uint32_t n, uint32_t k,
                                    ds41_cpu_candidate *heap) {
    uint32_t used = 0;
    for (uint32_t i = 0; i < n; i++) {
        const ds41_cpu_candidate c = {scores[i], i};
        if (used < k) {
            uint32_t p = used++;
            while (p && ds41_cpu_candidate_better(heap[(p - 1u) / 2u], c)) {
                heap[p] = heap[(p - 1u) / 2u];
                p = (p - 1u) / 2u;
            }
            heap[p] = c;
        } else if (ds41_cpu_candidate_better(c, heap[0])) {
            uint32_t p = 0;
            while (p * 2u + 1u < k) {
                uint32_t child = p * 2u + 1u;
                if (child + 1u < k && ds41_cpu_candidate_better(heap[child], heap[child + 1u]))
                    child++;
                if (!ds41_cpu_candidate_better(c, heap[child])) break;
                heap[p] = heap[child];
                p = child;
            }
            heap[p] = c;
        }
    }
    qsort(heap, used, sizeof(*heap), ds41_cpu_candidate_compare);
    for (uint32_t i = 0; i < used; i++) out[i] = heap[i].index;
}

/* The first token of a pair remains F32; only the pooled result is BF16. */
static bool ds41_cpu_attention_pool2(float *out, float *previous_kv,
                                     float *previous_score, const float *kv,
                                     const float *score, uint32_t pos) {
    if (!(pos & 1u)) {
        memcpy(previous_kv, kv, 512u * sizeof(float));
        memcpy(previous_score, score, 512u * sizeof(float));
        return false;
    }
    for (uint32_t d = 0; d < 512; d++) {
        const float peak = fmaxf(previous_score[d], score[d]);
        const float a = expf(previous_score[d] - peak), b = expf(score[d] - peak);
        out[d] = ds41_cpu_bf16((previous_kv[d] * a + kv[d] * b) / (a + b));
    }
    return true;
}

static bool ds41_cpu_attention_publish(ds41_cpu_attention_state *s,
                                       const ds4_model *m, const ds4_layer_weights *l,
                                       uint32_t il, uint32_t pos, const float *norm) {
    if (!ds41_kv_source(il)) return true;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    if (!ds41_cpu_matvec(s->pool_kv, m, l->attn_compressor_kv, norm, ratio == 1u))
        return false;
    if (ratio == 2u) {
        if (!ds41_cpu_matvec(s->pool_score, m, l->attn_compressor_gate, norm, false))
            return false;
        if (!ds41_cpu_attention_pool2(s->latent, s->previous_kv[owner],
                s->previous_score[owner], s->pool_kv, s->pool_score, pos)) return true;
    } else memcpy(s->latent, s->pool_kv, sizeof(s->latent));
    const uint32_t row = (pos + 1u) / ratio - 1u;
    const uint32_t rope_pos = pos + 1u - ratio;
    if (row >= s->comp_cap[owner] ||
        !ds41_cpu_norm(s->latent, s->latent, m, l->attn_compressor_norm) ||
        !ds41_cpu_matvec(s->index_k, m, l->indexer_attn_k, s->latent, true) ||
        !ds41_cpu_norm(s->index_k, s->index_k, m, l->indexer_k_norm)) return false;
    ds41_cpu_rope(s->index_k, 128, 1, rope_pos, true, false);
    ds41_cpu_quantize(s->index_k, 128, 1, DS41_CPU_FP4_E8M0);
    memcpy(s->index_cache[owner] + (size_t)row * 128u, s->index_k, sizeof(s->index_k));
    ds41_cpu_rope(s->latent, 512, 1, rope_pos, true, false);
    ds41_cpu_quantize(s->latent, 512, 1, DS41_CPU_FP4_E4M3);
    memcpy(s->compressed[owner] + (size_t)row * 512u, s->latent, sizeof(s->latent));
    return true;
}

typedef struct {
    const float *q, *keys, *weights;
    float *scores;
} ds41_cpu_index_score_context;

static void ds41_cpu_index_score_worker(void *vctx, uint64_t first, uint64_t end) {
    ds41_cpu_index_score_context *c = vctx;
    for (uint64_t r = first; r < end; r++) {
        const float *key = c->keys + r * 128u;
        float score = 0;
        for (uint32_t h = 0; h < 32; h++) {
            const float dot = dot_f32(c->q + h * 128u, key, 128);
            score += fmaxf(dot * (1.0f / 64.0f), 0.0f) * c->weights[h];
        }
        c->scores[r] = score;
    }
}

/* Layer 20 fixes a block mask for all later decoder index sources at this
 * token. Pinning the newest block prevents eviction of the current latent. */
static void ds41_cpu_attention_candidates(ds41_cpu_attention_state *s,
                                           uint32_t il, uint32_t n_comp) {
    const uint32_t blocks = (n_comp + 7u) / 8u;
    if (il == 20) {
        memset(s->block_mask, 0, blocks);
        if (blocks <= 2048u) return;
        for (uint32_t b = 0; b < blocks; b++) {
            float best = -INFINITY;
            for (uint32_t i = b * 8u; i < n_comp && i < (b + 1u) * 8u; i++)
                best = fmaxf(best, s->index_scores[i]);
            s->block_scores[b] = b == blocks - 1u ? INFINITY : best;
        }
        ds41_cpu_attention_topk(s->selected_blocks, s->block_scores, blocks, 2048, s->heap);
        memset(s->block_mask, 1, blocks);
        for (uint32_t b = 0; b < 2048; b++) s->block_mask[s->selected_blocks[b]] = 0;
    } else if (il > 20) {
        for (uint32_t i = 0; i < n_comp; i++)
            if (s->block_mask[i / 8u]) s->index_scores[i] = -INFINITY;
    }
}

static bool ds41_cpu_attention_select(ds41_cpu_attention_state *s,
                                      const ds4_model *m, const ds4_layer_weights *l,
                                      uint32_t il, uint32_t pos, const float *norm) {
    ds41_prof(DS41_PROF_COMP);
    if (!ds41_cpu_attention_publish(s, m, l, il, pos, norm)) return false;
    ds41_prof(DS41_PROF_INDEX);
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0;
    if (!n_comp || !ds41_index_source(il)) return true;
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    if (!ds41_cpu_matvec(s->index_q, m, l->indexer_attn_q_b, s->qr, true) ||
        !ds41_cpu_matvec(s->index_weights, m, l->indexer_proj, norm, true)) return false;
    ds41_cpu_rope(s->index_q, 128, 32, pos, true, false);
    ds41_cpu_quantize(s->index_q, 128, 32, DS41_CPU_FP4_E8M0);
    ds41_cpu_index_score_context c = {
        s->index_q, s->index_cache[owner], s->index_weights, s->index_scores
    };
    ds4_parallel_for_min_rows(n_comp, ds41_cpu_index_score_worker, &c, 64);
    ds41_cpu_attention_candidates(s, il, n_comp);
    ds41_cpu_attention_topk(s->selected_comp, s->index_scores, n_comp,
                            n_comp < 512u ? n_comp : 512u, s->heap);
    return true;
}

/* Pass B1 of layer_attention_lanes with this path's running peak: fmaxf
 * from the sink, then the expf chain and the denominator in row order.  The
 * score and weighted-sum passes are the V4 ones: per head and row the same
 * products and sums in the same order as dot_f32 and axpy_f32 over the
 * staged rows, so every head output is bitwise the serial result. */
static void ds41_cpu_attention_weight_worker(void *vctx, uint64_t h0, uint64_t h1) {
    const layer_attention_lanes_ctx *ctx = vctx;
    for (uint64_t h = h0; h < h1; h++) {
        float *score = attn_lanes_score(ctx, (uint32_t)h, 0);
        float peak = ctx->sinks[h];
        for (uint32_t r = 0; r < ctx->n_rows; r++) peak = fmaxf(peak, score[(uint64_t)r * 16]);
        float sum = expf(ctx->sinks[h] - peak);
        for (uint32_t r = 0; r < ctx->n_rows; r++) {
            score[(uint64_t)r * 16] = expf(score[(uint64_t)r * 16] - peak);
            sum += score[(uint64_t)r * 16];
        }
        ctx->inv[h] = 1.0f / sum;
    }
}

/* Sink softmax of the 64 staged query heads over the first n_rows staged
 * rows into s->heads. */
static void ds41_cpu_attention_heads(ds41_cpu_attention_state *s,
                                     const float *sinks, uint32_t n_rows) {
    for (uint32_t r = 0; r < n_rows; r++) s->rows[r] = s->attended_kv + (size_t)r * 512u;
    attn_lanes_transpose_q(s->qt, s->q, 64, 512);
    layer_attention_lanes_ctx ctx = {
        .out_heads = s->heads,
        .sinks = sinks,
        .q = s->q,
        .qt = s->qt,
        .rows = s->rows,
        .score = s->score,
        .inv = s->inv,
        .n_head = 64,
        .score_stride = 128 + 512,
        .n_rows = n_rows,
        .n_chunks = (n_rows + ATTN_LANES_ROWS_PER_ITEM - 1) / ATTN_LANES_ROWS_PER_ITEM,
        .n_dim_chunks = 512 / 16,
    };
    if (n_rows) ds4_parallel_for_min_rows(4u * ctx.n_chunks, attn_lanes_score_worker, &ctx, 1);
    ds4_parallel_for_min_rows(64, ds41_cpu_attention_weight_worker, &ctx, 1);
    ds4_parallel_for_min_rows(4u * ctx.n_dim_chunks, attn_lanes_sum_worker, &ctx, 1);
}

/* f16_to_f32(f32_to_f16(x)) for every non-NaN x: both are round to nearest
 * even with the same overflow to infinity and subnormal grid.  A NaN stays a
 * NaN but may keep its payload. */
static void ds41_cpu_f16_round(float *dst, const float *src, uint32_t n) {
    uint32_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        const __m256i h = _mm512_cvtps_ph(_mm512_loadu_ps(src + i),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm512_storeu_ps(dst + i, _mm512_cvtph_ps(h));
    }
#endif
    for (; i < n; i++) dst[i] = f16_to_f32(f32_to_f16(src[i]));
}

typedef struct {
    ds41_cpu_attention_state *s;
    const float *window, *compressed;
    uint32_t first, n_raw;
} ds41_cpu_stage_context;

static void ds41_cpu_stage_worker(void *vctx, uint64_t first, uint64_t end) {
    const ds41_cpu_stage_context *c = vctx;
    for (uint64_t r = first; r < end; r++) {
        const float *src = r < c->n_raw ?
            c->window + (size_t)((c->first + r) % 128u) * 512u :
            c->compressed + (size_t)c->s->selected_comp[r - c->n_raw] * 512u;
        ds41_cpu_f16_round(c->s->attended_kv + (size_t)r * 512u, src, 512);
    }
}

static void ds41_cpu_attention_stage(ds41_cpu_attention_state *s,
                                      uint32_t il, uint32_t pos, uint32_t owner,
                                      uint32_t n_raw, uint32_t n_comp) {
    /* Metal's decode attention stages both Q and KV as F16. Keep this
     * boundary even though most quantized cache values fit exactly. */
    ds41_cpu_stage_context c = {
        s, s->window[il], s->compressed[owner], pos + 1u - n_raw, n_raw
    };
    ds4_parallel_for_min_rows(n_raw + n_comp, ds41_cpu_stage_worker, &c, 64);
    ds41_cpu_f16_round(s->q, s->q, 64u * 512u);
}

static bool ds41_cpu_attention_forward(ds41_cpu_attention_state *s,
                                       const ds4_model *m, const ds4_layer_weights *l,
                                       uint32_t il, uint32_t pos, const float *norm,
                                       float *block) {
    if (!s || il >= 40 || pos >= s->ctx) return false;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0u;
    ds41_prof(DS41_PROF_Q);
    if (!ds41_cpu_matvec(s->qr, m, l->attn_q_a, norm, true) ||
        !ds41_cpu_norm(s->qr, s->qr, m, l->attn_q_a_norm) ||
        !ds41_cpu_matvec(s->q, m, l->attn_q_b, s->qr, true)) return false;
    ds41_prof(DS41_PROF_KV);
    if (!ds41_cpu_matvec(s->kv, m, l->attn_kv, norm, true) ||
        !ds41_cpu_norm(s->kv, s->kv, m, l->attn_kv_a_norm)) return false;
    ds41_prof(DS41_PROF_Q);
    ds41_cpu_rope(s->q, 512, 64, pos, ratio != 0, false);
    ds41_prof(DS41_PROF_KV);
    ds41_cpu_rope(s->kv, 512, 1, pos, ratio != 0, false);
    ds41_cpu_quantize(s->kv, 512, 1, DS41_CPU_FP8_E8M0);
    memcpy(s->window[il] + (size_t)(pos % 128u) * 512u, s->kv, sizeof(s->kv));
    if (!ds41_cpu_attention_select(s, m, l, il, pos, norm)) return false;
    ds41_prof(DS41_PROF_ATTN);
    const uint32_t attended = n_comp < 512u ? n_comp : 512u;
    const uint32_t n_raw = pos < 127u ? pos + 1u : 128u;
    ds41_cpu_attention_stage(s, il, pos, owner, n_raw, attended);
    ds41_cpu_attention_heads(s, tensor_data(m, l->attn_sinks), n_raw + attended);
    ds41_cpu_bf16_rows(s->heads, 64u * 512u);
    ds41_cpu_rope(s->heads, 512, 64, pos, ratio != 0, true);
    ds41_prof(DS41_PROF_OUT);
    uint64_t row_bytes;
    if (!tensor_nbytes(l->attn_output_a->type, 4096u, &row_bytes)) return false;
    for (uint32_t group = 0; group < 8; group++) {
        ds4_tensor a = *l->attn_output_a;
        a.ndim = 2;
        a.dim[0] = 4096;
        a.dim[1] = 1024;
        a.elements = 4096u * 1024u;
        a.bytes = 1024u * row_bytes;
        a.abs_offset += (uint64_t)group * a.bytes;
        if (!ds41_cpu_matvec(s->low + group * 1024u, m, &a,
                             s->heads + group * 4096u, true)) return false;
    }
    if (!ds41_cpu_matvec(block, m, l->attn_output_b, s->low, false)) return false;
    ds41_cpu_bf16_rows(block, 5120);
    return true;
}
