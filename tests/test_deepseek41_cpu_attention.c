/* Numerical and cache-lifetime checks without model weights or a GPU. */
#include "../ds4.c"
#include <assert.h>

typedef struct { double score; uint32_t index; } oracle_candidate;

static int oracle_compare(const void *va, const void *vb) {
    const oracle_candidate *a = va, *b = vb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return (a->index > b->index) - (a->index < b->index);
}

static void check_topk(ds41_cpu_attention_state *s) {
    const uint32_t widths[] = {1, 7, 511, 512, 513, 6400};
    const uint32_t counts[] = {1, 4, 512, 2048};
    float scores[6400];
    uint32_t selected[2048];
    oracle_candidate sorted[6400];
    for (uint32_t mode = 0; mode < 3; mode++) {
        for (size_t w = 0; w < sizeof(widths) / sizeof(*widths); w++) {
            const uint32_t n = widths[w];
            for (uint32_t i = 0; i < n; i++) {
                scores[i] = mode == 0 ? (float)((i * 7919u) % 104729u) :
                            mode == 1 ? (float)(i % 9u) : -INFINITY;
                if (mode == 2 && i % 33u == 0) scores[i] = INFINITY;
                sorted[i] = (oracle_candidate){scores[i], i};
            }
            qsort(sorted, n, sizeof(*sorted), oracle_compare);
            for (size_t c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
                const uint32_t k = counts[c] < n ? counts[c] : n;
                ds41_cpu_attention_topk(selected, scores, n, k, s->heap);
                for (uint32_t i = 0; i < k; i++) assert(selected[i] == sorted[i].index);
            }
        }
    }
}

static void check_pairs(ds41_cpu_attention_state *s) {
    float first[512], second[512], first_score[512], second_score[512], output[512];
    for (uint32_t i = 0; i < 512; i++) {
        first[i] = (float)((int)(i % 13u) - 6) / 8.0f + 0.0001f;
        second[i] = (float)((int)(i % 17u) - 8) / 16.0f - 0.0001f;
        first_score[i] = (float)((int)(i % 5u) - 2) * 100.0f;
        second_score[i] = -first_score[i];
        output[i] = 12345;
    }
    assert(!ds41_cpu_attention_pool2(output, s->previous_kv[1], s->previous_score[1],
                                      first, first_score, 6));
    assert(!memcmp(s->previous_kv[1], first, sizeof(first)));
    assert(!memcmp(s->previous_score[1], first_score, sizeof(first_score)));
    for (uint32_t i = 0; i < 512; i++) {
        assert(output[i] == 12345);
        assert(s->previous_kv[0][i] == 0 && s->previous_kv[2][i] == 0);
    }
    assert(ds41_cpu_attention_pool2(output, s->previous_kv[1], s->previous_score[1],
                                     second, second_score, 7));
    for (uint32_t i = 0; i < 512; i++) {
        const double p = 1.0 / (1.0 + exp((double)second_score[i] - first_score[i]));
        const float expected = ds41_cpu_bf16((float)(first[i] * p + second[i] * (1.0 - p)));
        assert(output[i] == expected);
    }
    /* A new even token replaces the carry; it does not pool with the old pair. */
    assert(!ds41_cpu_attention_pool2(output, s->previous_kv[1], s->previous_score[1],
                                      second, second_score, 8));
    assert(!memcmp(s->previous_kv[1], second, sizeof(second)));
}

static void check_candidates(ds41_cpu_attention_state *s) {
    const uint32_t n = 16401, blocks = (n + 7u) / 8u;
    for (uint32_t i = 0; i < n; i++) s->index_scores[i] = (float)(n - i);
    ds41_cpu_attention_candidates(s, 20, n);
    uint32_t kept = 0;
    for (uint32_t i = 0; i < blocks; i++) kept += !s->block_mask[i];
    assert(kept == 2048 && !s->block_mask[blocks - 1u]);
    for (uint32_t b = 0; b < blocks; b++)
        assert(s->block_mask[b] == (b >= 2047u && b < blocks - 1u));
    /* Layer 20 constructs the mask, but its own scores remain unfiltered. */
    for (uint32_t i = 0; i < n; i++) assert(s->index_scores[i] == (float)(n - i));
    ds41_cpu_attention_candidates(s, 24, n);
    for (uint32_t i = 0; i < n; i++)
        assert(s->index_scores[i] == (s->block_mask[i / 8u] ? -INFINITY : (float)(n - i)));
    ds41_cpu_attention_candidates(s, 20, 8191);
    for (uint32_t b = 0; b < (8191u + 7u) / 8u; b++) assert(!s->block_mask[b]);
}

static void check_ring(ds41_cpu_attention_state *s) {
    const uint32_t il = 7, owner = 1, pos = 258;
    for (uint32_t r = 0; r < 128; r++)
        for (uint32_t d = 0; d < 512; d++) s->window[il][r * 512u + d] = (float)r;
    for (uint32_t r = 0; r < 3; r++)
        for (uint32_t d = 0; d < 512; d++) s->compressed[owner][r * 512u + d] = 500.0f + r;
    s->selected_comp[0] = 2; s->selected_comp[1] = 0; s->selected_comp[2] = 1;
    for (uint32_t i = 0; i < 64 * 512; i++) s->q[i] = 1.0002f;
    ds41_cpu_attention_stage(s, il, pos, owner, 128, 3);
    for (uint32_t r = 0; r < 131; r++) {
        const float expected = r < 128 ? (float)((pos + 1u - 128u + r) % 128u) :
            500.0f + s->selected_comp[r - 128u];
        for (uint32_t d = 0; d < 512; d++) assert(s->attended_kv[r * 512u + d] == expected);
    }
    for (uint32_t i = 0; i < 64 * 512; i++) assert(s->q[i] == 1.0f);
    ds41_cpu_attention_reset(s);
    assert(s->window[il][127 * 512u] == 127 && s->compressed[owner][0] == 500);
    for (uint32_t i = 0; i < 512; i++) assert(s->previous_kv[1][i] == 0);
}

static void check_index_scores(ds41_cpu_attention_state *s) {
    enum { ROWS = 17 };
    for (uint32_t i = 0; i < 32 * 128; i++)
        s->index_q[i] = (float)((int)((i * 19u) % 131u) - 65) / 64.0f;
    for (uint32_t i = 0; i < ROWS * 128; i++)
        s->index_cache[0][i] = (float)((int)((i * 23u) % 139u) - 69) / 32.0f;
    for (uint32_t i = 0; i < 32; i++) s->index_weights[i] = (float)((int)(i % 7u) - 3) / 8.0f;
    ds41_cpu_index_score_context c = {
        s->index_q, s->index_cache[0], s->index_weights, s->index_scores
    };
    ds41_cpu_index_score_worker(&c, 0, ROWS);
    for (uint32_t r = 0; r < ROWS; r++) {
        double expected = 0;
        for (uint32_t h = 0; h < 32; h++) {
            double dot = 0;
            for (uint32_t d = 0; d < 128; d++)
                dot += (double)s->index_q[h * 128u + d] * s->index_cache[0][r * 128u + d];
            expected += fmax(dot / 64.0, 0) * s->index_weights[h];
        }
        assert(fabs(s->index_scores[r] - expected) < 2e-5 * fmax(1.0, fabs(expected)));
    }
}

/* The serial per-head loop the lane-wise heads replaced. */
static void serial_heads(float *heads, const float *q, const float *kv,
                         const float *sinks, uint32_t rows) {
    const float scale = 1.0f / sqrtf(512.0f);
    for (uint32_t h = 0; h < 64; h++) {
        float scores[128 + 512];
        float *out = heads + h * 512u;
        float peak = sinks[h];
        for (uint32_t r = 0; r < rows; r++) {
            scores[r] = dot_f32(q + h * 512u, kv + (size_t)r * 512u, 512) * scale;
            peak = fmaxf(peak, scores[r]);
        }
        memset(out, 0, 512u * sizeof(float));
        float sum = expf(sinks[h] - peak);
        for (uint32_t r = 0; r < rows; r++) {
            const float p = expf(scores[r] - peak);
            sum += p;
            axpy_f32(out, kv + (size_t)r * 512u, p, 512);
        }
        scale_f32(out, 1.0f / sum, 512);
    }
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static float rand_unit(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 40) & 0xffffffu) / 8388608.0f - 1.0f;
}

static void check_softmax(ds41_cpu_attention_state *s) {
    float sinks[64];
    static float expected[64 * 512];
    const uint32_t counts[] = {1, 7, 33, 129, 511, 640};
    for (uint32_t h = 0; h < 64; h++) sinks[h] = (float)((int)(h % 9u) - 4);
    for (uint32_t i = 0; i < 64 * 512; i++)
        s->q[i] = (float)((int)((i * 19u) % 131u) - 65) / 64.0f;
    for (uint32_t i = 0; i < 640 * 512; i++)
        s->attended_kv[i] = (float)((int)((i * 23u) % 139u) - 69) / 32.0f;
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const uint32_t rows = counts[ci];
        ds41_cpu_attention_heads(s, sinks, rows);
        serial_heads(expected, s->q, s->attended_kv, sinks, rows);
        assert(!memcmp(s->heads, expected, sizeof(expected)));
        for (uint32_t h = 0; h < 3; h++) {
            double scores[640], peak = sinks[h], result[512] = {0};
            for (uint32_t r = 0; r < rows; r++) {
                double dot = 0;
                for (uint32_t d = 0; d < 512; d++)
                    dot += (double)s->q[h * 512u + d] * s->attended_kv[r * 512u + d];
                scores[r] = dot / sqrt(512.0);
                peak = fmax(peak, scores[r]);
            }
            double denominator = exp(sinks[h] - peak);
            for (uint32_t r = 0; r < rows; r++) {
                const double weight = exp(scores[r] - peak);
                denominator += weight;
                for (uint32_t d = 0; d < 512; d++)
                    result[d] += weight * s->attended_kv[r * 512u + d];
            }
            for (uint32_t d = 0; d < 512; d++)
                assert(fabs(s->heads[h * 512u + d] - result[d] / denominator) < 2e-5);
        }
    }
    /* Bitwise against the serial loop on F16-staged random rows with signed
     * zeros, exact zero scores and equal scores, at several thread counts. */
    for (uint32_t trial = 0; trial < 24; trial++) {
        const uint32_t rows = counts[trial % 6u] + (trial / 6u == 3u && counts[trial % 6u] < 640u);
        for (uint32_t h = 0; h < 64; h++) sinks[h] = rand_unit() * (trial & 1 ? 8.0f : 0.5f);
        if (trial % 3u == 0) sinks[trial] = 0.0f;
        if (trial % 3u == 1) sinks[trial] = -0.0f;
        for (uint32_t i = 0; i < 64 * 512; i++) {
            const float v = rand_unit() * (trial & 2 ? 4.0f : 1.0f);
            s->q[i] = (i % 97u) == trial ? -0.0f : (i % 89u) == trial ? 0.0f : v;
        }
        for (uint32_t i = 0; i < rows * 512; i++) {
            const float v = rand_unit() * (trial & 4 ? 16.0f : 1.0f);
            s->attended_kv[i] = (i % 61u) == trial ? -0.0f : v;
        }
        /* Head 0 sees zero scores and rows 1/2 repeat row 0. */
        memset(s->q, trial & 1 ? 0x80 : 0, 512 * sizeof(float));
        for (uint32_t r = 1; r < rows && r < 3; r++)
            memcpy(s->attended_kv + r * 512u, s->attended_kv, 512 * sizeof(float));
        ds41_cpu_f16_round(s->q, s->q, 64u * 512u);
        ds41_cpu_f16_round(s->attended_kv, s->attended_kv, rows * 512u);
        serial_heads(expected, s->q, s->attended_kv, sinks, rows);
        const uint32_t threads[] = {1, 3, 8};
        g_requested_threads = threads[trial % 3u];
        ds4_threads_init();
        ds41_cpu_attention_heads(s, sinks, rows);
        ds4_threads_shutdown();
        assert(!memcmp(s->heads, expected, sizeof(expected)));
    }
    g_requested_threads = 0;
    /* A sink contributes to the denominator, never to the value sum. */
    memset(s->q, 0, sizeof(s->q));
    for (uint32_t i = 0; i < 33 * 512; i++) s->attended_kv[i] = 1;
    sinks[0] = logf(2);
    ds41_cpu_attention_heads(s, sinks, 33);
    for (uint32_t d = 0; d < 512; d++) assert(fabsf(s->heads[d] - 33.0f / 35.0f) < 1e-6f);
    sinks[0] = 10000;
    ds41_cpu_attention_heads(s, sinks, 33);
    for (uint32_t d = 0; d < 512; d++) assert(s->heads[d] == 0);
}

/* The F16C staging round trip against the scalar converters on a sweep of
 * every exponent with many mantissas, plus the subnormal and overflow edges. */
static void check_f16_round(void) {
    static float in[1u << 16], out[1u << 16];
    for (uint64_t base = 0; base < (1ull << 32); base += (1ull << 16) * 61ull) {
        uint32_t n = 0;
        for (uint32_t i = 0; i < (1u << 16); i++) {
            const uint32_t bits = (uint32_t)(base + (uint64_t)i * 61ull);
            float x;
            memcpy(&x, &bits, sizeof(x));
            if (!isnan(x)) in[n++] = x;
        }
        ds41_cpu_f16_round(out, in, n);
        for (uint32_t i = 0; i < n; i++) {
            const float e = f16_to_f32(f32_to_f16(in[i]));
            assert(!memcmp(&out[i], &e, sizeof(e)));
        }
    }
    const float edges[] = {65504.0f, 65519.996f, 65520.0f, 65536.0f, 0x1p-24f, 0x1p-25f,
                           0x1.000002p-25f, 0x1.8p-24f, 0x1.4p-24f, 0x1.fffffep-15f,
                           0x1p-14f, -0x1p-25f, -0.0f, 0.0f, INFINITY, -INFINITY,
                           0x1p-149f, -0x1p-126f, 3.0e38f, 1.00048828125f, 1.000732421875f};
    const uint32_t n = sizeof(edges) / sizeof(*edges);
    ds41_cpu_f16_round(out, edges, n);
    for (uint32_t i = 0; i < n; i++) {
        const float e = f16_to_f32(f32_to_f16(edges[i]));
        assert(!memcmp(&out[i], &e, sizeof(e)));
    }
}

/* assert() of a value whose call must run with NDEBUG too. */
static void ensure(bool ok) {
    assert(ok);
    (void)ok;
}

/* One chunk's Engram rows through ds41_cpu_engram_batch_take: hash the
 * tokens after *h into b->ids, take the read ahead or read them, and
 * compare the rows with a plain ds4_engram_read of every token's ids. */
static void engram_chunk(ds41_cpu_batch *b, const ds4_engram_layout *l, ds4_engram_history *h,
                         const ds4_engram_table tab[2], const int *tokens, uint32_t n,
                         bool ahead) {
    const size_t cols = DS4_ENGRAM_COLS, dim = DS4_ENGRAM_DIM;
    for (uint32_t i = 0; i < n; i++)
        ensure(ds4_engram_hash(l, h, &tokens[i], NULL, 1, b->ids + (size_t)i * 2u * cols));
    ds41_cpu_engram_batch own, *e = ds41_cpu_engram_batch_take(b, tab, n, &own);
    assert((e == &b->ahead) == ahead && !b->ahead_live);
    const bool read0 = ds41_cpu_engram_batch_wait(e, 0);
    const bool read1 = ds41_cpu_engram_batch_wait(e, 1);
    ensure(read0 && read1);
    ds41_cpu_engram_batch_finish(e, false);
    float *ref = malloc((size_t)n * 2u * cols * dim * sizeof(float));
    assert(ref);
    for (uint32_t t = 0; t < n; t++)
        for (uint32_t i = 0; i < 2; i++)
            ensure(ds4_engram_read(&tab[i], b->ids + ((size_t)t * 2u + i) * cols, cols,
                                   ref + ((size_t)t * 2u + i) * cols * dim));
    assert(!memcmp(ref, b->rows, (size_t)n * 2u * cols * dim * sizeof(float)));
    free(ref);
}

/* The next chunk's rows read ahead are the bytes its own ids read, a chunk
 * with other ids or another length stops the read ahead and reads its own,
 * an invalid token starts none, and a batch freed with reads in flight
 * joins them. */
static void check_engram_read_ahead(void) {
    static uint32_t map[256];
    for (uint32_t i = 0; i < 256; i++) map[i] = i / 2u;
    ds4_engram_layout l = {.token_map = map, .vocab_size = 256,
                           .compressed_vocab_size = 128, .pad_id = 1};
    for (uint32_t layer = 0; layer < 2; layer++) {
        for (uint32_t j = 0; j < 4; j++)
            l.multipliers[layer][j] = 35184372088831ull - 2u * (j + 4u * layer);
        for (uint32_t j = 0; j < DS4_ENGRAM_COLS; j++) {
            l.primes[layer][j] = 163u + 2u * j + layer;
            l.rows[layer] += l.primes[layer][j];
        }
    }
    assert(ds4_engram_layout_valid(&l));
    /* Two tables back to back; every row decodes (no NaN code, no 255 scale). */
    const char *dir = getenv("TMPDIR");
    if (!dir || !dir[0]) dir = "/tmp";
    char path[4096];
    ensure(snprintf(path, sizeof(path), "%s/ds41-engram-XXXXXX", dir) < (int)sizeof(path));
    const int fd = mkstemp(path);
    ensure(fd >= 0);
    for (uint32_t t = 0; t < 2; t++) {
        for (uint32_t r = 0; r < l.rows[t]; r++) {
            uint8_t row[DS4_ENGRAM_ROW_BYTES];
            for (uint32_t j = 0; j < DS4_ENGRAM_DIM; j++)
                row[j] = (uint8_t)((r * 7u + j * 13u + t * 5u) % 127u) | (uint8_t)((j & 1u) << 7);
            for (uint32_t k = 0; k < 8; k++)
                row[DS4_ENGRAM_DIM + k] = (uint8_t)(120u + (r + k + t) % 12u);
            ensure(write(fd, row, sizeof(row)) == (ssize_t)sizeof(row));
        }
    }
    ds4_engram_table tab[2];
    ensure(ds4_engram_table_open(&tab[0], path, 0, l.rows[0]));
    ensure(ds4_engram_table_open(&tab[1], path, (uint64_t)l.rows[0] * DS4_ENGRAM_ROW_BYTES,
                                 l.rows[1]));
    /* The tables hold their own descriptors: the file goes now. */
    unlink(path);
    close(fd);
    enum { N = 80 };
    const size_t ids = (size_t)N * 2u * DS4_ENGRAM_COLS;
    ds41_cpu_batch *b = calloc(1, sizeof(*b));
    assert(b);
    b->ids = malloc(ids * sizeof(uint32_t));
    b->next_ids = malloc(ids * sizeof(uint32_t));
    b->rows = malloc(ids * DS4_ENGRAM_DIM * sizeof(float));
    b->next_rows = malloc(ids * DS4_ENGRAM_DIM * sizeof(float));
    assert(b->ids && b->next_ids && b->rows && b->next_rows);
    int a[N], next[N], other[N];
    for (uint32_t i = 0; i < N; i++) {
        rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
        a[i] = (int)((rng_state >> 33) & 255u);
        next[i] = other[i] = (int)((rng_state >> 45) & 255u);
    }
    other[N / 2] = (other[N / 2] + 2) & 255;   /* another compressed id */
    ds4_engram_history h, h2;
    ds4_engram_history_reset(&h);
    engram_chunk(b, &l, &h, tab, a, N, false);
    float *ahead_rows = b->next_rows;
    ds41_cpu_engram_batch_ahead(b, &l, h, tab, next, 70);
    assert(b->ahead_live && b->next_n == 70);
    h2 = h;
    engram_chunk(b, &l, &h2, tab, next, 70, true);
    assert(b->rows == ahead_rows);
    ds41_cpu_engram_batch_ahead(b, &l, h, tab, next, 70);
    h2 = h;
    engram_chunk(b, &l, &h2, tab, other, 70, false);
    ds41_cpu_engram_batch_ahead(b, &l, h, tab, next, 70);
    h2 = h;
    engram_chunk(b, &l, &h2, tab, next, 69, false);
    const int bad[2] = {3, 256};
    ds41_cpu_engram_batch_ahead(b, &l, h, tab, bad, 2);
    assert(!b->ahead_live);
    ds41_cpu_engram_batch_ahead(b, &l, h, tab, next, N);
    ds41_cpu_batch_free(b);
    ds4_engram_table_close(&tab[0]);
    ds4_engram_table_close(&tab[1]);
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    assert(!ds41_cpu_attention_create(0));
    ds41_cpu_attention_state *s = ds41_cpu_attention_create(16401);
    assert(s && s->comp_cap[0] == 8201 && s->comp_cap[3] == 16402);
    check_topk(s);
    check_pairs(s);
    check_candidates(s);
    check_ring(s);
    check_index_scores(s);
    check_softmax(s);
    check_f16_round();
    check_engram_read_ahead();
    ds41_cpu_attention_free(s);
    puts("V4.1 CPU attention: pair carry, top-k, candidate mask, ring, index scores, sink softmax "
         "and Engram read ahead PASS");
    return 0;
}
