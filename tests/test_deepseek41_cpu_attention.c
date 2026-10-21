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

static void check_softmax(ds41_cpu_attention_state *s) {
    enum { HEADS = 3 };
    float sinks[HEADS] = {-4, 0, 3.5f};
    const uint32_t counts[] = {1, 33, 129, 640};
    for (uint32_t i = 0; i < HEADS * 512; i++)
        s->q[i] = (float)((int)((i * 19u) % 131u) - 65) / 64.0f;
    for (uint32_t i = 0; i < 640 * 512; i++)
        s->attended_kv[i] = (float)((int)((i * 23u) % 139u) - 69) / 32.0f;
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        ds41_cpu_attention_context c = {s->q, s->attended_kv, sinks, s->heads, counts[ci]};
        ds41_cpu_attention_worker(&c, 0, HEADS);
        for (uint32_t h = 0; h < HEADS; h++) {
            double scores[640], peak = sinks[h], result[512] = {0};
            for (uint32_t r = 0; r < c.rows; r++) {
                double dot = 0;
                for (uint32_t d = 0; d < 512; d++)
                    dot += (double)s->q[h * 512u + d] * s->attended_kv[r * 512u + d];
                scores[r] = dot / sqrt(512.0);
                peak = fmax(peak, scores[r]);
            }
            double denominator = exp(sinks[h] - peak);
            for (uint32_t r = 0; r < c.rows; r++) {
                const double weight = exp(scores[r] - peak);
                denominator += weight;
                for (uint32_t d = 0; d < 512; d++)
                    result[d] += weight * s->attended_kv[r * 512u + d];
            }
            for (uint32_t d = 0; d < 512; d++)
                assert(fabs(s->heads[h * 512u + d] - result[d] / denominator) < 2e-5);
        }
    }
    /* A sink contributes to the denominator, never to the value sum. */
    memset(s->q, 0, sizeof(s->q));
    for (uint32_t i = 0; i < 33 * 512; i++) s->attended_kv[i] = 1;
    sinks[0] = logf(2);
    ds41_cpu_attention_context c = {s->q, s->attended_kv, sinks, s->heads, 33};
    ds41_cpu_attention_worker(&c, 0, 1);
    for (uint32_t d = 0; d < 512; d++) assert(fabsf(s->heads[d] - 33.0f / 35.0f) < 1e-6f);
    sinks[0] = 10000;
    ds41_cpu_attention_worker(&c, 0, 1);
    for (uint32_t d = 0; d < 512; d++) assert(s->heads[d] == 0);
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
    ds41_cpu_attention_free(s);
    puts("V4.1 CPU attention: pair carry, top-k, candidate mask, ring, index scores and sink softmax PASS");
    return 0;
}
