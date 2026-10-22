/* Integer SIMD must preserve the CPU reference's block-level FP ordering.
 * Compile with CPU_CFLAGS; this also tests the scalar fallback without AVX2. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static uint32_t quant_rng = 0x619ac37du;
static uint32_t quant_random(void) {
    quant_rng ^= quant_rng << 13;
    quant_rng ^= quant_rng >> 17;
    quant_rng ^= quant_rng << 5;
    return quant_rng;
}

typedef struct { void *map; size_t size; uint8_t *data; } quant_buffer;

/* Exact 8-byte grid loads and 16-byte activation loads end at PROT_NONE.
 * Separate offset tests below cover addresses unaligned to those widths. */
static quant_buffer quant_alloc(size_t bytes) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const size_t rounded = ((bytes ? bytes : 1) + page - 1) / page * page;
    void *map = mmap(NULL, rounded + page, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(map != MAP_FAILED);
    CHECK(mprotect((uint8_t *)map + rounded, page, PROT_NONE) == 0);
    return (quant_buffer){map, rounded + page, (uint8_t *)map + rounded - bytes};
}

static void quant_free(quant_buffer b) { CHECK(munmap(b.map, b.size) == 0); }

static int8_t quant_byte(unsigned pattern, unsigned lane) {
    switch (pattern) {
    case 0: return -128;
    case 1: return 127;
    case 2: return lane & 1 ? 127 : -128;
    case 3: return (int8_t)((int)(lane * 37u % 256u) - 128);
    case 4: return lane % 3 ? 0 : (lane & 1 ? -1 : 1);
    default: return (int8_t)((int)(quant_random() & 255u) - 128);
    }
}

static int32_t iq2_pair_oracle(const int8_t *g0, const int8_t *g1, const int8_t *q) {
    int64_t sum = 0;
    for (unsigned i = 0; i < 16; i++) sum += (int64_t)(i < 8 ? g0[i] : g1[i - 8]) * q[i];
    return (int32_t)sum;
}

static int32_t q2_oracle(const uint8_t *q2, const int8_t *q8, unsigned shift) {
    int64_t sum = 0;
    for (unsigned i = 0; i < 16; i++) sum += (int64_t)((q2[i] >> shift) & 3u) * q8[i];
    return (int32_t)sum;
}

static int8_t iq2_value(unsigned grid, unsigned sign, unsigned lane) {
    const int value = (int)((iq2xxs_grid[grid] >> (8 * lane)) & 255u);
    return (int8_t)((ksigns_iq2xs[sign] & (1u << lane)) ? -value : value);
}

static void check_integer_dots(void) {
    quant_buffer ga = quant_alloc(8), gb = quant_alloc(8);
    quant_buffer qa = quant_alloc(16), qb = quant_alloc(16);
    int8_t *g0 = (int8_t *)ga.data, *g1 = (int8_t *)gb.data, *q8 = (int8_t *)qa.data;
    uint8_t *q2 = qb.data;

    /* Exhaust all signed-byte products, including the -128 * -128 case that
     * overflows a signed 16-bit pair sum in saturating multiply-add schemes. */
    for (int g = -128; g <= 127; g++) {
        memset(g0, (unsigned char)g, 8);
        memset(g1, (unsigned char)g, 8);
        for (int q = -128; q <= 127; q++) {
            memset(q8, (unsigned char)q, 16);
            CHECK(dot_iq2_pair_16(g0, g1, q8) == 16 * g * q);
        }
    }

    CHECK(pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init) == 0);
    for (unsigned g = 0; g < 256; g++) {
        for (unsigned s = 0; s < 128; s++) {
            int8_t expected0[8], expected1[8];
            memcpy(g0, iq2xxs_signed_grid[g][s], 8);
            memcpy(g1, iq2xxs_signed_grid[255 - g][127 - s], 8);
            for (unsigned i = 0; i < 8; i++) {
                expected0[i] = iq2_value(g, s, i);
                expected1[i] = iq2_value(255 - g, 127 - s, i);
            }
            for (unsigned p = 0; p < 6; p++) {
                for (unsigned i = 0; i < 16; i++) q8[i] = quant_byte(p, i);
                CHECK(dot_iq2_pair_16(g0, g1, q8) == iq2_pair_oracle(expected0, expected1, q8));
            }
        }
    }

    for (unsigned value = 0; value < 256; value++) {
        for (unsigned p = 0; p < 6; p++) {
            for (unsigned i = 0; i < 16; i++) {
                q2[i] = (uint8_t)(value + (p & 1 ? i * 17 : 0));
                q8[i] = quant_byte(p, i);
            }
            for (unsigned shift = 0; shift <= 6; shift += 2)
                CHECK(dot_q2_16(q2, q8, (int)shift) == q2_oracle(q2, q8, shift));
        }
    }

    /* Move each operand through every alignment modulo 32 independently. */
    uint8_t a[64], b[64], c[64], d[64];
    for (unsigned offset = 0; offset < 32; offset++) {
        int8_t *u = (int8_t *)a + offset, *v = (int8_t *)b + (31 - offset);
        int8_t *q = (int8_t *)c + (offset * 7 % 32);
        uint8_t *packed = d + (offset * 13 % 32);
        for (unsigned repeat = 0; repeat < 64; repeat++) {
            for (unsigned i = 0; i < 8; i++) { u[i] = quant_byte(5, i); v[i] = quant_byte(5, i); }
            for (unsigned i = 0; i < 16; i++) { q[i] = quant_byte(5, i); packed[i] = (uint8_t)quant_random(); }
            CHECK(dot_iq2_pair_16(u, v, q) == iq2_pair_oracle(u, v, q));
            for (unsigned shift = 0; shift <= 6; shift += 2)
                CHECK(dot_q2_16(packed, q, (int)shift) == q2_oracle(packed, q, shift));
        }
    }
    quant_free(qb); quant_free(qa); quant_free(gb); quant_free(ga);
}

/* Decode each logical weight directly; do not reuse signed-grid tables,
 * SIMD dot helpers, or Q8 block sums in the independent block oracles. */
static float iq2_block_oracle(unsigned blocks, const block_iq2_xxs *x, const block_q8_K *y) {
    float sum = 0.0f;
    for (unsigned b = 0; b < blocks; b++) {
        const uint8_t *packed = (const uint8_t *)x[b].qs;
        int64_t total = 0;
        for (unsigned group = 0; group < 8; group++) {
            uint32_t signs;
            memcpy(&signs, packed + group * 8 + 4, sizeof(signs));
            int64_t part = 0;
            for (unsigned i = 0; i < 32; i++) {
                const unsigned grid = packed[group * 8 + i / 8];
                const unsigned sign = (signs >> (7 * (i / 8))) & 127u;
                part += (int64_t)iq2_value(grid, sign, i % 8) * y[b].qs[group * 32 + i];
            }
            total += part * (int64_t)(2 * (signs >> 28) + 1);
        }
        const float scale = f16_to_f32(x[b].d) * y[b].d;
        sum += scale * (float)total;
    }
    return 0.125f * sum;
}

static float q2_block_oracle(unsigned blocks, const block_q2_K *x, const block_q8_K *y) {
    float sum = 0.0f;
    for (unsigned b = 0; b < blocks; b++) {
        int64_t total = 0, minimum = 0;
        for (unsigned i = 0; i < QK_K; i++) {
            const unsigned group = i / 16;
            const unsigned byte = 32 * (group / 8) + 16 * (group & 1) + i % 16;
            const unsigned shift = 2 * ((group / 2) & 3);
            const unsigned value = (x[b].qs[byte] >> shift) & 3u;
            total += (int64_t)(x[b].scales[group] & 15u) * value * y[b].qs[i];
            minimum += (int64_t)(x[b].scales[group] >> 4) * y[b].qs[i];
        }
        const float scale = y[b].d * f16_to_f32(x[b].d);
        const float min_scale = y[b].d * f16_to_f32(x[b].dmin);
        sum += scale * (float)total - min_scale * (float)minimum;
    }
    return sum;
}

static void quant_equal(const char *name, float expected, float actual, unsigned blocks, unsigned pattern) {
    uint32_t a, b;
    memcpy(&a, &expected, sizeof(a)); memcpy(&b, &actual, sizeof(b));
    if (a != b) {
        fprintf(stderr, "%s blocks=%u pattern=%u: %08x != %08x (%g != %g)\n",
                name, blocks, pattern, a, b, expected, actual);
        exit(1);
    }
}

static void check_block_dots(void) {
    const unsigned counts[] = {0, 1, 2, 3, 9, 20, 32, 64};
    const uint16_t half[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x8400,
                            0x1801, 0x9a17, 0x3555, 0xba31, 0x7bff, 0xfbff};
    const float scales[] = {0.0f, -0.0f, 0x1p-20f, -0.00173f, 0.375f, -1.13f, 17.3f, -1024.0f};
    for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        const unsigned n = counts[c];
        quant_buffer ib = quant_alloc(n * sizeof(block_iq2_xxs));
        quant_buffer qb = quant_alloc(n * sizeof(block_q2_K));
        quant_buffer yb = quant_alloc(n * sizeof(block_q8_K));
        block_iq2_xxs *iq = (block_iq2_xxs *)ib.data;
        block_q2_K *q2 = (block_q2_K *)qb.data;
        block_q8_K *q8 = (block_q8_K *)yb.data;
        for (unsigned p = 0; p < 12; p++) {
            for (unsigned b = 0; b < n; b++) {
                for (unsigned j = 0; j < sizeof(iq[b].qs); j++) ((uint8_t *)iq[b].qs)[j] = (uint8_t)quant_random();
                for (unsigned j = 0; j < sizeof(q2[b].qs); j++) q2[b].qs[j] = (uint8_t)quant_random();
                for (unsigned j = 0; j < 16; j++) q2[b].scales[j] = (uint8_t)quant_random();
                iq[b].d = half[(p + b) % 12];
                q2[b].d = half[(p + 3 * b) % 12];
                q2[b].dmin = half[(p + 5 * b + 7) % 12];
                q8[b].d = scales[(p + b) % 8];
                for (unsigned j = 0; j < QK_K; j++) q8[b].qs[j] = quant_byte(p % 6, j + b);
                for (unsigned group = 0; group < 16; group++) {
                    int sum = 0;
                    for (unsigned j = 0; j < 16; j++) sum += q8[b].qs[group * 16 + j];
                    q8[b].bsums[group] = (int16_t)sum;
                }
            }
            float actual;
            ds4_vec_dot_iq2_xxs_q8_K((int)(n * QK_K), &actual, iq, q8);
            quant_equal("IQ2_XXS", iq2_block_oracle(n, iq, q8), actual, n, p);
            ds4_vec_dot_q2_K_q8_K((int)(n * QK_K), &actual, q2, q8);
            quant_equal("Q2_K", q2_block_oracle(n, q2, q8), actual, n, p);
        }
        quant_free(yb); quant_free(qb); quant_free(ib);
    }
}

/* Serial indexer reference: the platform's serial dot_f32 order (scalar
 * ascending i on x86), per-head accumulation in head order, then the argmax
 * rounds of the original loop (strict '>' from DS4_NEG_INF, so NaN and -inf
 * rows are never picked and exhausted rounds fall back to row 0). */
static void indexer_oracle(bool *allowed, float *scores, const float *q, const float *weights,
                           const float *comp, unsigned n_comp, unsigned top_k) {
    const unsigned n_head = DS4_N_INDEXER_HEAD, head_dim = DS4_N_INDEXER_HEAD_DIM;
    for (unsigned c = 0; c < n_comp; c++) {
        float s = 0.0f;
        for (unsigned h = 0; h < n_head; h++) {
#if defined(__ARM_NEON)
            float dot = dot_f32(comp + c * head_dim, q + h * head_dim, head_dim);
#else
            float dot = 0.0f;
            for (unsigned i = 0; i < head_dim; i++) dot += comp[c * head_dim + i] * q[h * head_dim + i];
#endif
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }
    memset(allowed, 0, n_comp * sizeof(allowed[0]));
    for (unsigned k = 0; k < top_k; k++) {
        unsigned best = 0;
        float best_score = DS4_NEG_INF;
        for (unsigned c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) { best = c; best_score = scores[c]; }
        }
        allowed[best] = true;
    }
}

static float indexer_random_float(void) {
    return (float)(int32_t)quant_random() * 0x1p-31f;
}

/* Scores feed comparisons only, so a NaN of any payload is the same score. */
static void indexer_scores_equal(const float *expected, const float *actual, unsigned n) {
    for (unsigned c = 0; c < n; c++) {
        uint32_t a, b;
        memcpy(&a, &expected[c], sizeof(a)); memcpy(&b, &actual[c], sizeof(b));
        if (a != b && !(expected[c] != expected[c] && actual[c] != actual[c])) {
            fprintf(stderr, "indexer score %u: %08x != %08x (%g != %g)\n", c, a, b, expected[c], actual[c]);
            exit(1);
        }
    }
}

static void check_indexer(void) {
    const ds4_shape saved = g_ds4_shape;
    const unsigned counts[] = {513, 600, 1500};
    const unsigned heads[] = {64, 20};
    const float specials[] = {NAN, INFINITY, -INFINITY, 0x1p-140f, -0x1p-140f, 0x1p100f};
    g_ds4_shape.n_indexer_head_dim = 128;
    for (unsigned hi = 0; hi < sizeof(heads) / sizeof(*heads); hi++) {
        g_ds4_shape.n_indexer_head = heads[hi];
        const unsigned n_head = DS4_N_INDEXER_HEAD, dim = DS4_N_INDEXER_HEAD_DIM;
        float *q = xmalloc((size_t)n_head * dim * sizeof(float));
        float *qt = xmalloc((size_t)n_head * dim * sizeof(float));
        float *weights = xmalloc(n_head * sizeof(float));
        for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const unsigned n_comp = counts[ci], top_k = 512;
            float *comp = xmalloc((size_t)n_comp * dim * sizeof(float));
            float *scores = xmalloc(n_comp * sizeof(float));
            float *expected = xmalloc(n_comp * sizeof(float));
            bool *allowed = xmalloc(n_comp * sizeof(bool));
            bool *allowed_expected = xmalloc(n_comp * sizeof(bool));
            for (unsigned repeat = 0; repeat < 4; repeat++) {
                /* Random values round at every step, so any reassociation
                 * or fused multiply-add shows; the coarse pass and the
                 * duplicated rows make exact score ties common; the last pass
                 * also plants NaN, infinities, huge and denormal values. */
                const bool coarse = repeat == 1;
                for (unsigned j = 0; j < n_head * dim; j++) {
                    q[j] = coarse ? (float)(int)(indexer_random_float() * 8.0f) * 0.25f : indexer_random_float();
                }
                for (unsigned h = 0; h < n_head; h++) weights[h] = (float)(int)(indexer_random_float() * 8.0f) * 0.125f;
                for (unsigned j = 0; j < n_comp * dim; j++) {
                    comp[j] = coarse ? (float)(int)(indexer_random_float() * 8.0f) * 0.25f : indexer_random_float();
                }
                for (unsigned c = 0; c < n_comp; c += 7) {
                    memcpy(comp + (size_t)c * dim, comp + (size_t)(c / 2) * dim, dim * sizeof(float));
                }
                if (repeat == 3) {
                    for (unsigned c = 0; c < n_comp; c += 29) {
                        comp[(size_t)c * dim + c % dim] = specials[(c / 29) % 6];
                    }
                }
                indexer_oracle(allowed_expected, expected, q, weights, comp, n_comp, top_k);
                indexer_score_rows(scores, qt, q, weights, comp, n_comp);
                indexer_scores_equal(expected, scores, n_comp);
                indexer_select_top_k(allowed, scores, n_comp, top_k);
                CHECK(memcmp(allowed, allowed_expected, n_comp * sizeof(bool)) == 0);
            }
            free(allowed_expected); free(allowed); free(expected); free(scores); free(comp);
        }
        free(weights); free(qt); free(q);
    }
    g_ds4_shape = saved;
}

int main(void) {
    check_integer_dots();
    check_indexer();
    puts("CPU indexer row-parallel scoring and top-k selection oracle PASS");
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    puts("CPU ARM DOTPROD IQ2_XXS/Q2_K integer oracles PASS");
    puts("Block FP oracle skipped: existing ARM DOTPROD accumulation order differs");
#else
    check_block_dots();
#if defined(DS4_NO_GPU) && defined(__AVX2__)
    puts("CPU AVX2 IQ2_XXS/Q2_K integer and bitwise block oracles PASS");
#elif defined(__ARM_NEON)
    puts("CPU NEON IQ2_XXS/Q2_K integer and bitwise block oracles PASS");
#else
    puts("CPU scalar IQ2_XXS/Q2_K integer and bitwise block oracles PASS");
#endif
#endif
    return 0;
}
