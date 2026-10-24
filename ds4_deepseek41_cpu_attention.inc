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

/* Publish the pooled latent of a KV source layer completed at pos as a
 * compressed row of owner, with its index key.  latent (512) and index_k
 * (128) are the caller's scratch, so that rows can be published in
 * parallel. */
static bool ds41_cpu_attention_publish_row(ds41_cpu_attention_state *s,
                                           const ds4_model *m, const ds4_layer_weights *l,
                                           uint32_t owner, uint32_t ratio, uint32_t pos,
                                           float *latent, float *index_k) {
    const uint32_t row = (pos + 1u) / ratio - 1u;
    const uint32_t rope_pos = pos + 1u - ratio;
    if (row >= s->comp_cap[owner] ||
        !ds41_cpu_norm(latent, latent, m, l->attn_compressor_norm) ||
        !ds41_cpu_matvec(index_k, m, l->indexer_attn_k, latent, true) ||
        !ds41_cpu_norm(index_k, index_k, m, l->indexer_k_norm)) return false;
    ds41_cpu_rope(index_k, 128, 1, rope_pos, true, false);
    ds41_cpu_quantize(index_k, 128, 1, DS41_CPU_FP4_E8M0);
    memcpy(s->index_cache[owner] + (size_t)row * 128u, index_k, 128u * sizeof(float));
    ds41_cpu_rope(latent, 512, 1, rope_pos, true, false);
    ds41_cpu_quantize(latent, 512, 1, DS41_CPU_FP4_E4M3);
    memcpy(s->compressed[owner] + (size_t)row * 512u, latent, 512u * sizeof(float));
    return true;
}

/* Pool the compressor projections in pool_kv and pool_score of a KV source
 * layer and, when a row completes, publish it with its index key. */
static bool ds41_cpu_attention_publish(ds41_cpu_attention_state *s,
                                       const ds4_model *m, const ds4_layer_weights *l,
                                       uint32_t il, uint32_t pos) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    if (ratio == 2u) {
        if (!ds41_cpu_attention_pool2(s->latent, s->previous_kv[owner],
                s->previous_score[owner], s->pool_kv, s->pool_score, pos)) return true;
    } else memcpy(s->latent, s->pool_kv, sizeof(s->latent));
    return ds41_cpu_attention_publish_row(s, m, l, owner, ratio, pos, s->latent, s->index_k);
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

/* The scores of keys [0, n) of one index query, as
 * ds41_cpu_index_score_worker computes them: per head and key dot_f32's
 * chain in input order, then fmaxf(dot / 64, 0) * weight added over the
 * heads in head order.  With AVX-512 a lane is a head (qt is 16 KiB of
 * scratch for the transposed query) and four keys run at once, eight
 * independent chains instead of one; a zero's sign from fmaxf cannot reach
 * a score that starts at +0. */
static void ds41_cpu_index_scores(ds41_cpu_index_score_context *c, uint32_t n, float *qt) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (uint32_t h = 0; h < 32; h++)
        for (uint32_t i = 0; i < 128; i++) qt[i * 32u + h] = c->q[h * 128u + i];
    const __m512 scale = _mm512_set1_ps(1.0f / 64.0f), zero = _mm512_setzero_ps();
    const __m512 w0 = _mm512_loadu_ps(c->weights), w1 = _mm512_loadu_ps(c->weights + 16);
    for (uint32_t r0 = 0; r0 < n; r0 += 4) {
        const uint32_t m = n - r0 < 4 ? n - r0 : 4;
        const float *key[4];
        __m512 a[4], b[4];
        for (uint32_t j = 0; j < 4; j++) {
            key[j] = c->keys + (uint64_t)(r0 + (j < m ? j : m - 1)) * 128u;
            a[j] = b[j] = zero;
        }
        for (uint32_t i = 0; i < 128; i++) {
            const __m512 qa = _mm512_loadu_ps(qt + i * 32u), qb = _mm512_loadu_ps(qt + i * 32u + 16);
            for (uint32_t j = 0; j < 4; j++) {
                const __m512 kv = _mm512_set1_ps(key[j][i]);
                a[j] = _mm512_add_ps(a[j], _mm512_mul_ps(qa, kv));
                b[j] = _mm512_add_ps(b[j], _mm512_mul_ps(qb, kv));
            }
        }
        for (uint32_t j = 0; j < m; j++) {
            float t[32] __attribute__((aligned(64)));
            _mm512_store_ps(t, _mm512_mul_ps(_mm512_max_ps(_mm512_mul_ps(a[j], scale), zero), w0));
            _mm512_store_ps(t + 16, _mm512_mul_ps(_mm512_max_ps(_mm512_mul_ps(b[j], scale), zero), w1));
            float score = 0;
            for (uint32_t h = 0; h < 32; h++) score += t[h];
            c->scores[r0 + j] = score;
        }
    }
#else
    (void)qt;
    ds41_cpu_index_score_worker(c, 0, n);
#endif
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

/* Score the cached index keys of owner with the quantized index query and
 * the head weights, then select the attended compressed rows. */
static void ds41_cpu_attention_rank(ds41_cpu_attention_state *s, uint32_t il,
                                    uint32_t owner, uint32_t n_comp) {
    ds41_cpu_index_score_context c = {
        s->index_q, s->index_cache[owner], s->index_weights, s->index_scores
    };
    /* A row is 32 serial 128-wide dots, about 8 us, so even a few dozen
     * rows are worth splitting. */
    ds4_parallel_for_min_rows(n_comp, ds41_cpu_index_score_worker, &c, 16);
    ds41_cpu_attention_candidates(s, il, n_comp);
    ds41_cpu_attention_topk(s->selected_comp, s->index_scores, n_comp,
                            n_comp < 512u ? n_comp : 512u, s->heap);
}

static void ds41_cpu_attention_index_q(ds41_cpu_attention_state *s, uint32_t pos) {
    ds41_cpu_rope(s->index_q, 128, 32, pos, true, false);
    ds41_cpu_quantize(s->index_q, 128, 32, DS41_CPU_FP4_E8M0);
}

static bool ds41_cpu_attention_select(ds41_cpu_attention_state *s,
                                      const ds4_model *m, const ds4_layer_weights *l,
                                      uint32_t il, uint32_t pos, const float *norm) {
    ds41_prof(DS41_PROF_COMP);
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ds41_kv_source(il) &&
        (!ds41_cpu_matvec(s->pool_kv, m, l->attn_compressor_kv, norm, ratio == 1u) ||
         (ratio == 2u &&
          !ds41_cpu_matvec(s->pool_score, m, l->attn_compressor_gate, norm, false)) ||
         !ds41_cpu_attention_publish(s, m, l, il, pos))) return false;
    ds41_prof(DS41_PROF_INDEX);
    const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0;
    if (!n_comp || !ds41_index_source(il)) return true;
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    if (!ds41_cpu_matvec(s->index_q, m, l->indexer_attn_q_b, s->qr, true) ||
        !ds41_cpu_matvec(s->index_weights, m, l->indexer_proj, norm, true)) return false;
    ds41_cpu_attention_index_q(s, pos);
    ds41_cpu_attention_rank(s, il, owner, n_comp);
    return true;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* Projection rows with ds41_cpu_matvec()'s BF16 rounding applied per claim:
 * Q8_0 16-row tiles, and F16 row groups. */
static void ds41_cpu_q8_bf16_worker(void *vctx, uint64_t first, uint64_t last) {
    const ds41_cpu_q8_ctx *c = vctx;
    ds41_cpu_q8_worker(vctx, first, last);
    const uint64_t end = last * 16 < c->rows ? last * 16 : c->rows;
    ds41_cpu_bf16_rows(c->ref.out + first * 16, (size_t)(end - first * 16));
}

static void ds41_cpu_f16_bf16_worker(void *vctx, uint64_t first, uint64_t last) {
    const matvec_f16_ctx *c = vctx;
    matvec_f16_group_worker(vctx, first, last);
    const uint64_t end = last * DS4_F16_ROWS < c->out_dim ? last * DS4_F16_ROWS : c->out_dim;
    ds41_cpu_bf16_rows(c->out + first * DS4_F16_ROWS, (size_t)(end - first * DS4_F16_ROWS));
}

static ds4_pool_stage ds41_cpu_q8_stage(ds41_cpu_q8_ctx *c, const ds4_model *m,
                                        const ds4_tensor *t, float *out, const float *x) {
    const uint64_t rows = t->dim[1], tiles = (rows + 15) / 16;
    *c = (ds41_cpu_q8_ctx){
        .ref = {.out = out, .data = tensor_data(m, t), .x = x,
                .in_dim = t->dim[0], .blocks = (t->dim[0] + 31) / 32},
        .rows = rows, .group_rows = rows,
    };
    return (ds4_pool_stage){.fn = ds41_cpu_q8_bf16_worker, .ctx = c,
                            .n_rows = tiles, .group_rows = tiles};
}

static ds4_pool_stage ds41_cpu_f16_stage(matvec_f16_ctx *c, const ds4_model *m,
                                         const ds4_tensor *t, float *out,
                                         const float *x, bool round) {
    const uint64_t groups = (t->dim[1] + DS4_F16_ROWS - 1) / DS4_F16_ROWS;
    *c = (matvec_f16_ctx){.out = out, .data = tensor_data(m, t), .x = x,
                          .in_dim = t->dim[0], .out_dim = t->dim[1]};
    return (ds4_pool_stage){.fn = round ? ds41_cpu_f16_bf16_worker : matvec_f16_group_worker,
                            .ctx = c, .n_rows = groups, .group_rows = groups};
}

/* Every projection of the attention input and of the Q low-rank vector,
 * with the attention HC mixer, as one staged dispatch.  The stages run the
 * row kernels of the plain path on the same inputs, round each row as it
 * does, and its serial steps run as finish steps:
 *   HC norm (one row), q_a -> qr norm, kv -> norm, RoPE, FP8, window,
 *   HC mixer <- HC norm -> Sinkhorn, compressor kv and score (KV source
 *   layers), index head weights, publish (one row) <- compressor,
 *   index q <- q_a -> RoPE, FP4, q_b <- q_a.
 * Independent stages come first: a thread waits at the first stage whose
 * inputs are not ready. */
typedef struct {
    ds41_cpu_attention_state *s;
    const ds4_model *m;
    const ds4_layer_weights *l;
    uint32_t il, pos;
    bool published;
    ds41_cpu_hc_job hc;
    ds41_cpu_q8_ctx q_a, kv, q_b;
    matvec_f16_ctx comp_kv, comp_score, index_q, index_weights;
} ds41_cpu_proj_graph;

static void ds41_cpu_proj_qr(void *vctx) {
    ds41_cpu_proj_graph *g = vctx;
    ds41_cpu_norm(g->s->qr, g->s->qr, g->m, g->l->attn_q_a_norm);
}

static void ds41_cpu_proj_kv(void *vctx) {
    ds41_cpu_proj_graph *g = vctx;
    ds41_cpu_attention_state *s = g->s;
    ds41_cpu_norm(s->kv, s->kv, g->m, g->l->attn_kv_a_norm);
    ds41_cpu_rope(s->kv, 512, 1, g->pos, ds4_layer_compress_ratio(g->il) != 0, false);
    ds41_cpu_quantize(s->kv, 512, 1, DS41_CPU_FP8_E8M0);
    memcpy(s->window[g->il] + (size_t)(g->pos % 128u) * 512u, s->kv, sizeof(s->kv));
}

static void ds41_cpu_proj_publish(void *vctx, uint64_t first, uint64_t end) {
    ds41_cpu_proj_graph *g = vctx;
    if (first < end) g->published = ds41_cpu_attention_publish(g->s, g->m, g->l, g->il, g->pos);
}

static void ds41_cpu_proj_index_q(void *vctx) {
    ds41_cpu_proj_graph *g = vctx;
    ds41_cpu_attention_index_q(g->s, g->pos);
}

static bool ds41_cpu_proj_graph_supported(const ds4_layer_weights *l, uint32_t il,
                                          bool index, const ds41_cpu_hc_job *hc) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const ds4_tensor *qn = l->attn_q_a_norm, *kn = l->attn_kv_a_norm;
    return ds41_cpu_hc_staged(hc) &&
        ds41_cpu_q8_shape(l->attn_q_a, 5120, 1280) &&
        ds41_cpu_q8_shape(l->attn_kv, 5120, 512) &&
        ds41_cpu_q8_shape(l->attn_q_b, 1280, 64u * 512u) &&
        qn && qn->type == DS4_TENSOR_F32 && qn->dim[0] == 1280 &&
        kn && kn->type == DS4_TENSOR_F32 && kn->dim[0] == 512 &&
        (!ds41_kv_source(il) ||
         (ds41_cpu_f16_shape(l->attn_compressor_kv, 5120, 512) &&
          (ratio != 2u || ds41_cpu_f16_shape(l->attn_compressor_gate, 5120, 512)))) &&
        (!index ||
         (ds41_cpu_f16_shape(l->indexer_attn_q_b, 1280, 32u * 128u) &&
          ds41_cpu_f16_shape(l->indexer_proj, 5120, 32)));
}

static bool ds41_cpu_proj_graph_run(ds41_cpu_attention_state *s, const ds4_model *m,
                                    const ds4_layer_weights *l, uint32_t il,
                                    uint32_t pos, const float *norm, bool index,
                                    const ds41_cpu_hc_job *hc) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool source = ds41_kv_source(il);
    ds41_cpu_proj_graph g = {.s = s, .m = m, .l = l, .il = il, .pos = pos,
                             .published = true, .hc = *hc};
    ds4_pool_stage st[10];
    uint32_t n = 0, comp = 0;
    st[n++] = ds41_cpu_hc_norm_stage(&g.hc);                         /* 0 */
    st[n] = ds41_cpu_q8_stage(&g.q_a, m, l->attn_q_a, s->qr, norm);  /* 1 */
    st[n].finish = ds41_cpu_proj_qr;
    st[n++].finish_ctx = &g;
    st[n] = ds41_cpu_q8_stage(&g.kv, m, l->attn_kv, s->kv, norm);
    st[n].finish = ds41_cpu_proj_kv;
    st[n++].finish_ctx = &g;
    st[n++] = ds41_cpu_hc_mix_stage(&g.hc, 0);
    if (source) {
        comp |= 1u << n;
        st[n++] = ds41_cpu_f16_stage(&g.comp_kv, m, l->attn_compressor_kv,
                                     s->pool_kv, norm, ratio == 1u);
        if (ratio == 2u) {
            comp |= 1u << n;
            st[n++] = ds41_cpu_f16_stage(&g.comp_score, m, l->attn_compressor_gate,
                                         s->pool_score, norm, false);
        }
    }
    if (index)
        st[n++] = ds41_cpu_f16_stage(&g.index_weights, m, l->indexer_proj,
                                     s->index_weights, norm, true);
    if (source)
        st[n++] = (ds4_pool_stage){.fn = ds41_cpu_proj_publish, .ctx = &g,
                                   .n_rows = 1, .group_rows = 1, .deps = comp};
    if (index) {
        st[n] = ds41_cpu_f16_stage(&g.index_q, m, l->indexer_attn_q_b, s->index_q, s->qr, true);
        st[n].deps = 1u << 1;
        st[n].finish = ds41_cpu_proj_index_q;
        st[n++].finish_ctx = &g;
    }
    st[n] = ds41_cpu_q8_stage(&g.q_b, m, l->attn_q_b, s->q, s->qr);
    st[n++].deps = 1u << 1;
    ds4_pool_run_stages(st, n);
    if (ds41_prof_on) {
        uint64_t bytes = hc->fn->bytes + l->attn_q_a->bytes + l->attn_kv->bytes +
                         l->attn_q_b->bytes;
        if (source) bytes += l->attn_compressor_kv->bytes +
                             (ratio == 2u ? l->attn_compressor_gate->bytes : 0);
        if (index) bytes += l->indexer_proj->bytes + l->indexer_attn_q_b->bytes;
        ds41_prof_bytes[DS41_PROF_PROJ] += bytes;
    }
    return g.published;
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* The scores of two head groups (heads h0 .. h0 + 31) for n <= 8 staged
 * rows from r0: each row value broadcast feeds both groups, sixteen chains.
 * Per head and row these are the products and sums of
 * attn_lanes_score_block in the same order, then the same scaling. */
static void ds41_cpu_attention_score2(const layer_attention_lanes_ctx *ctx, uint32_t h0,
                                      uint32_t r0, uint32_t n) {
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const float *qa = ctx->qt + (uint64_t)h0 * DS4_N_HEAD_DIM, *qb = qa + 16u * DS4_N_HEAD_DIM;
    const float *k[8];
    __m512 a[8], b[8];
    for (uint32_t j = 0; j < 8; j++) {
        k[j] = ctx->rows[r0 + (j < n ? j : n - 1)];
        a[j] = b[j] = _mm512_setzero_ps();
    }
    for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++) {
        const __m512 qva = _mm512_loadu_ps(qa + (uint64_t)i * 16);
        const __m512 qvb = _mm512_loadu_ps(qb + (uint64_t)i * 16);
        for (uint32_t j = 0; j < 8; j++) {
            const __m512 kv = _mm512_set1_ps(k[j][i]);
            a[j] = _mm512_add_ps(a[j], _mm512_mul_ps(qva, kv));
            b[j] = _mm512_add_ps(b[j], _mm512_mul_ps(qvb, kv));
        }
    }
    const __m512 scale = _mm512_set1_ps(kq_scale);
    for (uint32_t j = 0; j < n; j++) {
        _mm512_storeu_ps(attn_lanes_score(ctx, h0, r0 + j), _mm512_mul_ps(a[j], scale));
        _mm512_storeu_ps(attn_lanes_score(ctx, h0 + 16, r0 + j), _mm512_mul_ps(b[j], scale));
    }
}
#endif

/* Every score of the 64 staged heads over the staged rows, as
 * attn_lanes_score_worker computes them. */
static void ds41_cpu_attention_scores(const layer_attention_lanes_ctx *ctx) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (uint32_t h = 0; h < 64; h += 32)
        for (uint32_t r = 0; r < ctx->n_rows; r += 8)
            ds41_cpu_attention_score2(ctx, h, r, ctx->n_rows - r < 8 ? ctx->n_rows - r : 8);
#else
    if (ctx->n_rows) attn_lanes_score_worker((void *)ctx, 0, 4u * ctx->n_chunks);
#endif
}

/* Pass B1 of layer_attention_lanes with this path's running peak: fmaxf
 * from the sink, then the expf chain and the denominator in row order.  The
 * score and weighted-sum passes are the V4 ones: per head and row the same
 * products and sums in the same order as dot_f32 and axpy_f32 over the
 * staged rows, so every head output is bitwise the serial result.  Whole
 * groups of sixteen heads run a lane per head with the same steps. */
static void ds41_cpu_attention_weight_worker(void *vctx, uint64_t h0, uint64_t h1) {
    const layer_attention_lanes_ctx *ctx = vctx;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (; h0 % 16 == 0 && h0 + 16 <= h1; h0 += 16) {
        float *score = attn_lanes_score(ctx, (uint32_t)h0, 0);
        const __m512 sinks = _mm512_loadu_ps(ctx->sinks + h0);
        __m512 peak = sinks;
        for (uint32_t r = 0; r < ctx->n_rows; r++)
            peak = ds41_cpu_fmaxf16(_mm512_loadu_ps(score + (uint64_t)r * 16), peak);
        __m512 sum = ds41_cpu_expf16(_mm512_sub_ps(sinks, peak));
        for (uint32_t r = 0; r < ctx->n_rows; r++) {
            const __m512 e = ds41_cpu_expf16(_mm512_sub_ps(_mm512_loadu_ps(score + (uint64_t)r * 16),
                                                           peak));
            _mm512_storeu_ps(score + (uint64_t)r * 16, e);
            sum = _mm512_add_ps(sum, e);
        }
        _mm512_storeu_ps(ctx->inv + h0, _mm512_div_ps(_mm512_set1_ps(1.0f), sum));
    }
#endif
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

/* The lane passes of the 64 staged query heads over the first n_rows
 * staged rows into s->heads: the row list, the transposed query. */
static void ds41_cpu_attention_lanes(ds41_cpu_attention_state *s,
                                     layer_attention_lanes_ctx *ctx,
                                     const float *sinks, uint32_t n_rows) {
    for (uint32_t r = 0; r < n_rows; r++) s->rows[r] = s->attended_kv + (size_t)r * 512u;
    attn_lanes_transpose_q(s->qt, s->q, 64, 512);
    *ctx = (layer_attention_lanes_ctx){
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
}

/* Sink softmax of the 64 staged query heads over the first n_rows staged
 * rows into s->heads. */
static void ds41_cpu_attention_heads(ds41_cpu_attention_state *s,
                                     const float *sinks, uint32_t n_rows) {
    layer_attention_lanes_ctx ctx;
    ds41_cpu_attention_lanes(s, &ctx, sinks, n_rows);
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

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* The heads, both output projections and the HC post into the FFN input as
 * one staged dispatch:
 *   0 head scores, 1 softmax weights <- 0, 2 weighted sums <- 1 -> BF16 and
 *   inverse RoPE of the heads, 3 output a (eight groups) <- 2, 4 output b
 *   <- 3, 5 HC post and FFN input sum per column <- 4 -> FFN norm.
 * The lane passes, row kernels and serial steps are the plain path's, and
 * the BF16 roundings of low and block are applied per claimed row. */
typedef struct {
    ds41_cpu_attention_state *s;
    uint32_t pos;
    bool compressed;
    layer_attention_lanes_ctx lanes;
    ds41_cpu_q8_ctx out_a, out_b;
} ds41_cpu_out_graph;

static void ds41_cpu_out_heads(void *vctx) {
    ds41_cpu_out_graph *g = vctx;
    ds41_cpu_bf16_rows(g->s->heads, 64u * 512u);
    ds41_cpu_rope(g->s->heads, 512, 64, g->pos, g->compressed, true);
}

static bool ds41_cpu_out_graph_supported(const ds4_layer_weights *l,
                                         const ds41_cpu_hc_post_job *post) {
    const ds4_tensor *a = l->attn_output_a, *n = post->norm_weight;
    return a && a->type == DS4_TENSOR_Q8_0 && a->dim[0] == 4096u &&
        a->elements == 4096u * 8192u && ds41_cpu_q8_shape(l->attn_output_b, 8192, 5120) &&
        n && n->type == DS4_TENSOR_F32 && n->dim[0] == 5120;
}

static void ds41_cpu_out_graph_run(ds41_cpu_attention_state *s, const ds4_model *m,
                                   const ds4_layer_weights *l, uint32_t pos,
                                   bool compressed, uint32_t n_rows, float *block,
                                   const ds41_cpu_hc_post_job *post) {
    ds41_cpu_out_graph g = {.s = s, .pos = pos, .compressed = compressed};
    ds41_cpu_attention_lanes(s, &g.lanes, tensor_data(m, l->attn_sinks), n_rows);
    g.out_a = (ds41_cpu_q8_ctx){
        .ref = {.out = s->low, .data = tensor_data(m, l->attn_output_a), .x = s->heads,
                .in_dim = 4096, .blocks = 4096 / 32},
        .rows = 8192, .group_rows = 1024, .group_in = 4096,
    };
    const uint64_t scores = 4u * g.lanes.n_chunks, sums = 4u * g.lanes.n_dim_chunks;
    ds4_pool_stage st[6] = {
        { .fn = attn_lanes_score_worker, .ctx = &g.lanes,
          .n_rows = n_rows ? scores : 0, .group_rows = scores ? scores : 1 },
        { .fn = ds41_cpu_attention_weight_worker, .ctx = &g.lanes,
          .n_rows = 64, .group_rows = 64, .deps = 1u << 0 },
        { .fn = attn_lanes_sum_worker, .ctx = &g.lanes, .n_rows = sums, .group_rows = sums,
          .deps = 1u << 1, .finish = ds41_cpu_out_heads, .finish_ctx = &g },
        { .fn = ds41_cpu_q8_bf16_worker, .ctx = &g.out_a, .n_rows = 8192 / 16,
          .group_rows = 8192 / 16, .deps = 1u << 2 },
        ds41_cpu_q8_stage(&g.out_b, m, l->attn_output_b, block, s->low),
        /* Light elementwise work: claims of 256 columns. */
        { .fn = ds41_cpu_hc_post_worker, .ctx = (void *)post, .n_rows = 5120,
          .group_rows = 5120, .min_chunk = 256, .deps = 1u << 4,
          .finish = ds41_cpu_hc_post_norm, .finish_ctx = (void *)post },
    };
    st[4].deps = 1u << 3;
    ds4_pool_run_stages(st, 6);
    if (ds41_prof_on)
        ds41_prof_bytes[DS41_PROF_OUT] += l->attn_output_a->bytes + l->attn_output_b->bytes;
}
#endif

/* RoPE of the token's projected q and kv, its kv into the layer's raw
 * window, then the compressor and index selection. */
static bool ds41_cpu_attention_cache(ds41_cpu_attention_state *s, const ds4_model *m,
                                     const ds4_layer_weights *l, uint32_t il,
                                     uint32_t pos, const float *norm) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    ds41_prof(DS41_PROF_Q);
    ds41_cpu_rope(s->q, 512, 64, pos, ratio != 0, false);
    ds41_prof(DS41_PROF_KV);
    ds41_cpu_rope(s->kv, 512, 1, pos, ratio != 0, false);
    ds41_cpu_quantize(s->kv, 512, 1, DS41_CPU_FP8_E8M0);
    memcpy(s->window[il] + (size_t)(pos % 128u) * 512u, s->kv, sizeof(s->kv));
    return ds41_cpu_attention_select(s, m, l, il, pos, norm);
}

/* The staged rows of the token and the sink softmax of its 64 heads into
 * s->heads, rounded to BF16 and rotated back. */
static void ds41_cpu_attention_attend(ds41_cpu_attention_state *s, const ds4_model *m,
                                      const ds4_layer_weights *l, uint32_t il, uint32_t pos) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0u;
    const uint32_t attended = n_comp < 512u ? n_comp : 512u;
    const uint32_t n_raw = pos < 127u ? pos + 1u : 128u;
    ds41_prof(DS41_PROF_STAGE);
    ds41_cpu_attention_stage(s, il, pos, owner, n_raw, attended);
    ds41_prof(DS41_PROF_ATTN);
    ds41_cpu_attention_heads(s, tensor_data(m, l->attn_sinks), n_raw + attended);
    ds41_cpu_bf16_rows(s->heads, 64u * 512u);
    ds41_cpu_rope(s->heads, 512, 64, pos, ratio != 0, true);
}

/* The plain projections: the HC mixer, then q, kv, the compressor and the
 * index selection, one dispatch per matrix. */
static bool ds41_cpu_attention_project(ds41_cpu_attention_state *s, const ds4_model *m,
                                       const ds4_layer_weights *l, uint32_t il,
                                       uint32_t pos, const float *norm,
                                       const ds41_cpu_hc_job *hc) {
    ds41_prof(DS41_PROF_HC);
    if (!ds41_cpu_hc_run(hc)) return false;
    ds41_prof(DS41_PROF_Q);
    if (!ds41_cpu_matvec(s->qr, m, l->attn_q_a, norm, true) ||
        !ds41_cpu_norm(s->qr, s->qr, m, l->attn_q_a_norm) ||
        !ds41_cpu_matvec(s->q, m, l->attn_q_b, s->qr, true)) return false;
    ds41_prof(DS41_PROF_KV);
    if (!ds41_cpu_matvec(s->kv, m, l->attn_kv, norm, true) ||
        !ds41_cpu_norm(s->kv, s->kv, m, l->attn_kv_a_norm)) return false;
    return ds41_cpu_attention_cache(s, m, l, il, pos, norm);
}

/* The grouped low-rank output projection of s->heads into block. */
static bool ds41_cpu_attention_output(ds41_cpu_attention_state *s,
                                      const ds4_model *m, const ds4_layer_weights *l,
                                      float *block) {
    ds41_prof(DS41_PROF_OUT);
    if (l->attn_output_a->type == DS4_TENSOR_Q8_0 && l->attn_output_a->dim[0] == 4096u &&
        l->attn_output_a->elements == 4096u * 8192u) {
        /* The eight groups in one dispatch: same rows, kernel and inputs. */
        ds41_cpu_matvec_q8_grouped(s->low, m, l->attn_output_a, s->heads, 1024u, 4096u);
        if (ds41_prof_on) ds41_prof_bytes[DS41_PROF_OUT] += l->attn_output_a->bytes;
        ds41_cpu_bf16_rows(s->low, 8192u);
    } else {
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
    }
    if (!ds41_cpu_matvec(block, m, l->attn_output_b, s->low, false)) return false;
    ds41_cpu_bf16_rows(block, 5120);
    return true;
}

/* Attention of the normalized input norm into block.  The HC mixer of the
 * layer runs alongside the projections, the HC post and the FFN input
 * after the output projection. */
static bool ds41_cpu_attention_forward(ds41_cpu_attention_state *s,
                                       const ds4_model *m, const ds4_layer_weights *l,
                                       uint32_t il, uint32_t pos, const float *norm,
                                       float *block, const ds41_cpu_hc_job *hc,
                                       const ds41_cpu_hc_post_job *post) {
    if (!s || il >= 40 || pos >= s->ctx) return false;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0u;
    const bool index = n_comp && ds41_index_source(il);
    if (ds41_cpu_proj_graph_supported(l, il, index, hc)) {
        ds41_prof(DS41_PROF_PROJ);
        if (!ds41_cpu_proj_graph_run(s, m, l, il, pos, norm, index, hc)) return false;
        ds41_prof(DS41_PROF_Q);
        ds41_cpu_rope(s->q, 512, 64, pos, ratio != 0, false);
        ds41_prof(DS41_PROF_INDEX);
        if (index) ds41_cpu_attention_rank(s, il, owner, n_comp);
        if (ds41_cpu_out_graph_supported(l, post)) {
            const uint32_t attended = n_comp < 512u ? n_comp : 512u;
            const uint32_t n_raw = pos < 127u ? pos + 1u : 128u;
            ds41_prof(DS41_PROF_STAGE);
            ds41_cpu_attention_stage(s, il, pos, owner, n_raw, attended);
            ds41_prof(DS41_PROF_OUT);
            ds41_cpu_out_graph_run(s, m, l, pos, ratio != 0, n_raw + attended, block, post);
            return true;
        }
    } else if (!ds41_cpu_attention_project(s, m, l, il, pos, norm, hc)) return false;
#else
    if (!ds41_cpu_attention_project(s, m, l, il, pos, norm, hc)) return false;
#endif
    ds41_cpu_attention_attend(s, m, l, il, pos);
    if (!ds41_cpu_attention_output(s, m, l, block)) return false;
    ds41_prof(DS41_PROF_HC);
    return ds41_cpu_hc_post_run(post);
}
