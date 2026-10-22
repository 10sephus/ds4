/* Integer SIMD must preserve the CPU reference's block-level FP ordering.
 * Compile with CPU_CFLAGS; this also tests the scalar fallback without AVX. */
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

/* Buffers end at a PROT_NONE page so that any load past the last block,
 * including 64-byte vector loads of a 66-byte IQ2_XXS block, faults. */
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
    quant_buffer qa = quant_alloc(16), qb = quant_alloc(16);
    int8_t *q8 = (int8_t *)qa.data;
    uint8_t *q2 = qb.data;

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
    uint8_t c[64], d[64];
    for (unsigned offset = 0; offset < 32; offset++) {
        int8_t *q = (int8_t *)c + (offset * 7 % 32);
        uint8_t *packed = d + (offset * 13 % 32);
        for (unsigned repeat = 0; repeat < 64; repeat++) {
            for (unsigned i = 0; i < 16; i++) { q[i] = quant_byte(5, i); packed[i] = (uint8_t)quant_random(); }
            for (unsigned shift = 0; shift <= 6; shift += 2)
                CHECK(dot_q2_16(packed, q, (int)shift) == q2_oracle(packed, q, shift));
        }
    }
    quant_free(qb); quant_free(qa);
}

/* The sign and code tables the SIMD kernels derive instead of looking up. */
static void check_iq2_tables(void) {
    for (unsigned s = 0; s < 128; s++) {
        CHECK(ksigns_iq2xs[s] == (s | (unsigned)(__builtin_popcount(s) & 1) << 7));
    }
    CHECK(pthread_once(&iq2xxs_tables_once, iq2xxs_tables_init) == 0);
#if defined(DS4_NO_GPU) && defined(__AVX512BW__)
    static const unsigned values[4] = {8, 25, 43, 0};
    for (unsigned g = 0; g < 256; g++) {
        for (unsigned j = 0; j < 8; j++) {
            const unsigned code = (iq2xxs_codes[g] >> (2 * (j & 3) + 8 * (j >> 2))) & 3u;
            CHECK(code < 3 && values[code] == ((iq2xxs_grid[g] >> (8 * j)) & 255u));
        }
    }
#endif
}

/* Decode each logical weight directly; do not reuse derived tables, SIMD dot
 * helpers, or Q8 block sums in the independent block oracles. */
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

static void fill_q8_block(block_q8_K *q8, float d, unsigned pattern, unsigned lane0) {
    q8->d = d;
    for (unsigned j = 0; j < QK_K; j++) q8->qs[j] = quant_byte(pattern, lane0 + j);
    for (unsigned group = 0; group < 16; group++) {
        int sum = 0;
        for (unsigned j = 0; j < 16; j++) sum += q8->qs[group * 16 + j];
        q8->bsums[group] = (int16_t)sum;
    }
}

static const uint16_t quant_half[12] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x8400,
                                        0x1801, 0x9a17, 0x3555, 0xba31, 0x7bff, 0xfbff};
static const float quant_scale[8] = {0.0f, -0.0f, 0x1p-20f, -0.00173f, 0.375f, -1.13f, 17.3f, -1024.0f};

static void check_rows(const char *label, unsigned n, const block_iq2_xxs *iq0, const block_iq2_xxs *iq1,
                       const block_q2_K *q2, const block_q8_K *q8, unsigned pattern) {
    float actual0, actual1;
    iq2xxs_q8 yq[64];
    iq2xxs_q8_prepare(yq, q8, n);
    ds4_vec_dot_iq2_xxs_q8_K((int)(n * QK_K), &actual0, iq0, yq);
    quant_equal(label, iq2_block_oracle(n, iq0, q8), actual0, n, pattern);
    ds4_vec_dot_iq2_xxs_pair_q8_K((int)(n * QK_K), &actual0, &actual1, iq0, iq1, yq);
    quant_equal(label, iq2_block_oracle(n, iq0, q8), actual0, n, pattern);
    quant_equal(label, iq2_block_oracle(n, iq1, q8), actual1, n, pattern);
    ds4_vec_dot_q2_K_q8_K((int)(n * QK_K), &actual0, q2, q8);
    quant_equal(label, q2_block_oracle(n, q2, q8), actual0, n, pattern);
}

static void check_block_dots(void) {
    const unsigned counts[] = {0, 1, 2, 3, 9, 20, 32, 64};
    for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        const unsigned n = counts[c];
        quant_buffer ib0 = quant_alloc(n * sizeof(block_iq2_xxs));
        quant_buffer ib1 = quant_alloc(n * sizeof(block_iq2_xxs));
        quant_buffer qb = quant_alloc(n * sizeof(block_q2_K));
        quant_buffer yb = quant_alloc(n * sizeof(block_q8_K));
        block_iq2_xxs *iq0 = (block_iq2_xxs *)ib0.data;
        block_iq2_xxs *iq1 = (block_iq2_xxs *)ib1.data;
        block_q2_K *q2 = (block_q2_K *)qb.data;
        block_q8_K *q8 = (block_q8_K *)yb.data;
        for (unsigned p = 0; p < 12; p++) {
            for (unsigned b = 0; b < n; b++) {
                for (unsigned j = 0; j < sizeof(iq0[b].qs); j++) {
                    ((uint8_t *)iq0[b].qs)[j] = (uint8_t)quant_random();
                    ((uint8_t *)iq1[b].qs)[j] = (uint8_t)quant_random();
                }
                for (unsigned j = 0; j < sizeof(q2[b].qs); j++) q2[b].qs[j] = (uint8_t)quant_random();
                for (unsigned j = 0; j < 16; j++) q2[b].scales[j] = (uint8_t)quant_random();
                iq0[b].d = quant_half[(p + b) % 12];
                iq1[b].d = quant_half[(p + 2 * b + 5) % 12];
                q2[b].d = quant_half[(p + 3 * b) % 12];
                q2[b].dmin = quant_half[(p + 5 * b + 7) % 12];
                fill_q8_block(&q8[b], quant_scale[(p + b) % 8], p % 6, b);
            }
            check_rows("random", n, iq0, iq1, q2, q8, p);
        }
        quant_free(yb); quant_free(qb); quant_free(ib1); quant_free(ib0);
    }
}

/* Every grid index, sign index and scale nibble, in rows long enough to cycle
 * through all of them four times, against each activation pattern; and the
 * largest magnitudes both formats can produce, where the 16-bit intermediate
 * sums of the SIMD kernels come closest to saturating. */
static void check_extremes(void) {
    const unsigned n = 32;
    quant_buffer ib0 = quant_alloc(n * sizeof(block_iq2_xxs));
    quant_buffer ib1 = quant_alloc(n * sizeof(block_iq2_xxs));
    quant_buffer qb = quant_alloc(n * sizeof(block_q2_K));
    quant_buffer yb = quant_alloc(n * sizeof(block_q8_K));
    block_iq2_xxs *iq0 = (block_iq2_xxs *)ib0.data;
    block_iq2_xxs *iq1 = (block_iq2_xxs *)ib1.data;
    block_q2_K *q2 = (block_q2_K *)qb.data;
    block_q8_K *q8 = (block_q8_K *)yb.data;

    unsigned heaviest = 0;
    for (unsigned g = 1; g < 256; g++) {
        unsigned sum = 0, best = 0;
        for (unsigned j = 0; j < 8; j++) {
            sum += (iq2xxs_grid[g] >> (8 * j)) & 255u;
            best += (iq2xxs_grid[heaviest] >> (8 * j)) & 255u;
        }
        if (sum > best) heaviest = g;
    }

    for (unsigned p = 0; p < 8; p++) {
        for (unsigned b = 0; b < n; b++) {
            uint8_t *packed0 = (uint8_t *)iq0[b].qs;
            uint8_t *packed1 = (uint8_t *)iq1[b].qs;
            for (unsigned group = 0; group < 8; group++) {
                uint32_t signs0 = ((b + group) & 15u) << 28, signs1 = 15u << 28;
                for (unsigned l = 0; l < 4; l++) {
                    const unsigned t = b * 32 + group * 4 + l;
                    packed0[group * 8 + l] = (uint8_t)t;
                    signs0 |= ((t >> 1) & 127u) << (7 * l);
                    /* All weights at their largest magnitude with one sign. */
                    packed1[group * 8 + l] = (uint8_t)heaviest;
                    signs1 |= (p & 1 ? 127u : 0u) << (7 * l);
                }
                memcpy(packed0 + group * 8 + 4, &signs0, sizeof(signs0));
                memcpy(packed1 + group * 8 + 4, &signs1, sizeof(signs1));
            }
            memset(q2[b].qs, 0xff, sizeof(q2[b].qs));
            memset(q2[b].scales, 0xff, sizeof(q2[b].scales));
            iq0[b].d = quant_half[(p + b) % 12];
            iq1[b].d = quant_half[(3 * p + b) % 12];
            q2[b].d = quant_half[(p + 2 * b + 1) % 12];
            q2[b].dmin = quant_half[(p + 3 * b + 2) % 12];
            fill_q8_block(&q8[b], quant_scale[(p + b) % 8], p % 6, 3 * b);
        }
        check_rows("extremes", n, iq0, iq1, q2, q8, p);
    }
    quant_free(yb); quant_free(qb); quant_free(ib1); quant_free(ib0);
}


static float indexer_random_float(void) {
    return (float)(int32_t)quant_random() * 0x1p-31f;
}

/* The original selection: argmax rounds with a strict '>' from DS4_NEG_INF,
 * so NaN and -inf rows are never picked and exhausted rounds fall back to
 * row 0. */
static void indexer_select_oracle(bool *allowed, const float *scores, unsigned n_comp, unsigned top_k) {
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

/* Serial indexer reference: the platform's serial dot_f32 order (scalar
 * ascending i on x86), per-head accumulation in head order, then the
 * original selection. */
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
    indexer_select_oracle(allowed, scores, n_comp, top_k);
}

/* Selection alone on crafted scores: ties at every level including the
 * top_k boundary, +-0, +-inf, NaN, DS4_NEG_INF itself and values below it,
 * and fewer candidates than top_k with and without row 0 among them. */
static void check_indexer_selection(void) {
    const float values[] = {-2.0f, -1.0f, -0.0f, 0.0f, 0.5f, 1.0f, 1.0e30f, 0x1p-149f,
                            DS4_NEG_INF, -1.0e31f, INFINITY, -INFINITY, NAN};
    const unsigned n_values = sizeof(values) / sizeof(*values);
    const unsigned counts[] = {513, 600, 1500};
    for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const unsigned n_comp = counts[ci];
        float *scores = xmalloc(n_comp * sizeof(float));
        bool *allowed = xmalloc(n_comp * sizeof(bool));
        bool *expected = xmalloc(n_comp * sizeof(bool));
        indexer_candidate *cand = xmalloc(n_comp * sizeof(indexer_candidate));
        for (unsigned pattern = 0; pattern < 9; pattern++) {
            for (unsigned c = 0; c < n_comp; c++) {
                switch (pattern) {
                case 0: scores[c] = values[quant_random() % n_values]; break;
                case 1: scores[c] = values[quant_random() % 8]; break;
                case 2: scores[c] = indexer_random_float(); break;
                case 3: scores[c] = (float)(quant_random() % 3); break;
                case 4: scores[c] = c % 97 ? (c & 1 ? -0.0f : 0.0f) : 2.0f; break;
                case 5: scores[c] = NAN; break;
                case 6: scores[c] = c % 5 ? NAN : (float)(c % 7); break;
                case 7: scores[c] = c % 5 && c ? (float)(c % 7) : NAN; break;
                default: scores[c] = c < 512 ? 1.0f : -INFINITY; break;
                }
            }
            const unsigned top_ks[] = {512, 1, 100, n_comp - 1};
            for (unsigned ti = 0; ti < 4; ti++) {
                indexer_select_oracle(expected, scores, n_comp, top_ks[ti]);
                indexer_select_top_k(allowed, cand, scores, n_comp, top_ks[ti]);
                CHECK(memcmp(allowed, expected, n_comp * sizeof(bool)) == 0);
            }
        }
        free(cand); free(expected); free(allowed); free(scores);
    }
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
            indexer_candidate *cand = xmalloc(n_comp * sizeof(indexer_candidate));
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
                indexer_select_top_k(allowed, cand, scores, n_comp, top_k);
                CHECK(memcmp(allowed, allowed_expected, n_comp * sizeof(bool)) == 0);
            }
            free(cand); free(allowed_expected); free(allowed); free(expected); free(scores); free(comp);
        }
        free(weights); free(qt); free(q);
    }
    g_ds4_shape = saved;
}


#if defined(DS4_NO_GPU) && defined(__AVX2__) && defined(__F16C__)
/* VCVTPH2PS must reproduce the scalar f16_to_f32 bits for every half pattern:
 * zeros, subnormals, normals, infinities and quiet NaNs.  A signalling NaN is
 * the one exception: hardware sets its quiet bit while the scalar routine
 * keeps the raw payload.  The scalar multiply quiets it the same way before
 * anything is accumulated, which check_f16_rows verifies on the dot output. */
static void check_f16_conversion(void) {
    unsigned signalling = 0;
    for (uint32_t base = 0; base < 65536; base += 8) {
        uint16_t h[8];
        float f[8];
        for (unsigned k = 0; k < 8; k++) h[k] = (uint16_t)(base + k);
        _mm256_storeu_ps(f, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)h)));
        for (unsigned k = 0; k < 8; k++) {
            const float e = f16_to_f32(h[k]);
            uint32_t expected, actual;
            memcpy(&expected, &e, sizeof(expected));
            memcpy(&actual, &f[k], sizeof(actual));
            if ((h[k] & 0x7c00u) == 0x7c00u && (h[k] & 0x03ffu) != 0 && !(h[k] & 0x0200u)) {
                expected |= 0x00400000u;
                signalling++;
            }
            if (expected != actual) {
                fprintf(stderr, "f16 %04x: %08x != %08x\n", h[k], expected, actual);
                exit(1);
            }
        }
    }
    CHECK(signalling == 2 * 511);
}
#endif

/* Normal magnitudes 2^-8..2^7 with some signed zeros and subnormals. */
static uint16_t f16_random_weight(void) {
    const uint32_t r = quant_random();
    switch (r % 16) {
    case 0: return (uint16_t)(r & 0x8000u);
    case 1: return (uint16_t)(r & 0x83ffu);
    default: return (uint16_t)((r & 0x83ffu) | ((7 + (r >> 20) % 16) << 10));
    }
}

static void f16_rows_equal(const char *name, const float *expected, const float *actual, unsigned n) {
    for (unsigned r = 0; r < n; r++) {
        uint32_t a, b;
        memcpy(&a, &expected[r], sizeof(a));
        memcpy(&b, &actual[r], sizeof(b));
        if (a != b) {
            fprintf(stderr, "%s row %u: %08x != %08x (%g != %g)\n", name, r, a, b, expected[r], actual[r]);
            exit(1);
        }
    }
}

/* The F16 matvec worker must reproduce dot_f16_row bitwise for every group
 * shape: odd row counts, in_dim tails, and row ranges starting mid-group.
 * Guard pages end each matrix so the 8-lane loads are proven in bounds. */
static void check_f16_rows(void) {
    const unsigned dims[] = {1, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 1027};
    const unsigned counts[] = {1, 3, 5, 7, 8, 9, 13, 24, 29};
    const uint16_t specials[] = {0x7c00, 0xfc00, 0x7e00, 0xfe55, 0x7c01, 0xfdff, 0x0001, 0x83ff};
    float expected[32], actual[32];
    for (unsigned d = 0; d < sizeof(dims) / sizeof(*dims); d++) {
        const unsigned dim = dims[d];
        quant_buffer xb = quant_alloc(dim * sizeof(float));
        float *x = (float *)xb.data;
        for (unsigned i = 0; i < dim; i++) x[i] = (float)(int32_t)quant_random() / 2147483648.0f;
        for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
            const unsigned count = counts[c];
            quant_buffer wb = quant_alloc((size_t)count * dim * sizeof(uint16_t));
            uint16_t *w = (uint16_t *)wb.data;
            matvec_f16_ctx ctx = {.out = actual, .data = w, .x = x, .in_dim = dim};
            for (unsigned pass = 0; pass < 1 + sizeof(specials) / sizeof(*specials); pass++) {
                for (unsigned i = 0; i < count * dim; i++) w[i] = f16_random_weight();
                /* One special value per row at a varying position: infinities
                 * and NaNs must propagate with identical bits (the signalling
                 * NaN payloads 0x7c01 and 0xfdff are quieted in both paths). */
                if (pass) {
                    for (unsigned r = 0; r < count; r++) w[r * dim + (r * 7 + pass) % dim] = specials[pass - 1];
                }
                for (unsigned r = 0; r < count; r++) expected[r] = dot_f16_row(w + r * dim, x, dim);
                for (unsigned row0 = 0; row0 < count; row0 += 3) {
                    memset(actual, 0xff, sizeof(actual));
                    matvec_f16_worker(&ctx, row0, count);
                    f16_rows_equal("F16 rows", expected + row0, actual + row0, count - row0);
                }
            }
            quant_free(wb);
        }
        quant_free(xb);
    }
}


/* Verbatim copies of the scalar Q8_0/Q8_K activation quantizers and Q8_0 row
 * dot: the SIMD block paths must reproduce them bit for bit. */
static void q8_0_quantize_oracle(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32; i++) xq[i0 + i] = 0;
    }
}

static void q8_K_quantize_oracle(const float *x, block_q8_K *y, int64_t k) {
    for (int64_t b = 0; b < k / QK_K; b++, x += QK_K) {
        float max = 0.0f, amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
    }
}

static float q8_0_dot_oracle(const uint8_t *row, const int8_t *xq, const float *xscale, uint64_t in_dim) {
    float acc = 0.0f;
    for (uint64_t b = 0; b * 32 < in_dim; b++) {
        uint16_t h;
        memcpy(&h, row + b * 34, sizeof(h));
        int64_t dot = 0;
        for (uint64_t i = 0; i < 32 && b * 32 + i < in_dim; i++)
            dot += (int64_t)(int8_t)row[b * 34 + 2 + i] * xq[b * 32 + i];
        acc += f16_to_f32(h) * xscale[b] * (float)dot;
    }
    return acc;
}

static float quant_float_bits(uint32_t bits) { float f; memcpy(&f, &bits, sizeof(f)); return f; }

static float quant_uniform(float scale) {
    return scale * (((float)(int32_t)(quant_random() % 2000001u) - 1000000.0f) * 1e-6f);
}

/* Activations that stress the max search, the 1/d rounding and the clamp:
 * exact +/-amax ties, zero rows, round-half-even products, subnormal or
 * zero d, NaN, infinities and arbitrary bit patterns. */
static void quant_activation(float *x, uint64_t n, unsigned pattern) {
    for (uint64_t i = 0; i < n; i++) x[i] = quant_uniform(4.0f);
    switch (pattern) {
    case 0: break;
    case 1: /* the same |x| at several places, either sign: first one wins */
        for (uint64_t i = 0; i < n; i++) x[i] = quant_uniform(2.0f);
        for (unsigned t = 0; t < 6; t++) {
            const uint64_t base = quant_random() % n;
            x[base] = -3.0f;
            x[(base + 1 + quant_random() % 20) % n] = 3.0f;
            x[(base + 1 + quant_random() % 20) % n] = (quant_random() & 1) ? 3.0f : -3.0f;
        }
        break;
    case 2: for (uint64_t i = 0; i < n; i++) x[i] = (quant_random() & 1) ? 0.0f : -0.0f; break;
    case 3: /* amax 127: products are the inputs, many exact .5 ties */
        for (uint64_t i = 0; i < n; i++) x[i] = (float)((int)(quant_random() % 255u) - 127) + 0.5f * (float)(quant_random() & 1);
        for (uint64_t i = 0; i < n; i += 16) x[i] = (quant_random() & 1) ? 127.0f : -127.0f;
        break;
    case 4: for (uint64_t i = 0; i < n; i++) x[i] = quant_uniform(0x1p-128f); break; /* subnormal amax */
    case 5: for (uint64_t i = 0; i < n; i++) x[i] = quant_uniform(0x1p-120f); break; /* subnormal d, finite 1/d */
    case 6: x[quant_random() % n] = quant_float_bits(0x7fc00000u | (quant_random() & 0x3fffffu)); break;
    case 7: x[quant_random() % n] = (quant_random() & 1) ? INFINITY : -INFINITY; break;
    case 8: for (uint64_t i = 0; i < n; i++) x[i] = quant_uniform(FLT_MAX); break;
    case 9: for (uint64_t i = 0; i < n; i++) x[i] = quant_float_bits(quant_random()); break;
    case 10: /* one large value per 32, the rest tiny or zero */
        for (uint64_t i = 0; i < n; i++) x[i] = (i % 32 == 7) ? quant_uniform(1e6f) : quant_uniform(1e-3f) * (float)(quant_random() & 1);
        break;
    default: for (uint64_t i = 0; i < n; i++) x[i] = quant_uniform(8.0f) * (float)(quant_random() & 3); break;
    }
}

static void check_q8_quantizers(void) {
    enum { MAXN = 4096 + 40, PATTERNS = 12 };
    static float x[MAXN];
    static int8_t q0[MAXN + 32], q1[MAXN + 32];
    static float s0[MAXN / 32 + 1], s1[MAXN / 32 + 1];
    static block_q8_K k0[MAXN / QK_K], k1[MAXN / QK_K];
    const uint64_t lengths[] = {1, 7, 31, 32, 33, 64, 100, 256, 512, 1000, 1024, 4096, 4096 + 33};
    for (unsigned rep = 0; rep < 8; rep++) {
        for (unsigned p = 0; p < PATTERNS; p++) {
            for (unsigned l = 0; l < sizeof(lengths) / sizeof(*lengths); l++) {
                const uint64_t n = lengths[l], blocks = (n + 31) / 32;
                quant_activation(x, n, p);
                memset(q0, 0x5a, sizeof(q0)); memset(q1, 0x5a, sizeof(q1));
                memset(s0, 0x5a, sizeof(s0)); memset(s1, 0x5a, sizeof(s1));
                q8_0_quantize_oracle(x, q0, s0, n);
                quantize_q8_0_activation(x, q1, s1, n);
                if (memcmp(q0, q1, sizeof(q0)) || memcmp(s0, s1, blocks * sizeof(*s0))) {
                    fprintf(stderr, "Q8_0 quantizer mismatch n=%llu pattern=%u\n", (unsigned long long)n, p);
                    exit(1);
                }
                if (n % QK_K) continue;
                memset(k0, 0x5a, sizeof(k0)); memset(k1, 0x5a, sizeof(k1));
                q8_K_quantize_oracle(x, k0, (int64_t)n);
                ds4_quantize_row_q8_K(x, k1, (int64_t)n);
                if (memcmp(k0, k1, n / QK_K * sizeof(*k0))) {
                    fprintf(stderr, "Q8_K quantizer mismatch n=%llu pattern=%u\n", (unsigned long long)n, p);
                    exit(1);
                }
            }
        }
    }
}

static void check_q8_0_dots(void) {
    const uint64_t dims[] = {32, 64, 224, 256, 260, 288, 512, 1000, 1024, 2048 + 96, 4096, 4096 + 17};
    const uint16_t half[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x1801, 0x9a17,
                             0x2c00, 0xae66, 0x3555, 0xba31, 0x7bff, 0xfbff};
    const float scales[] = {0.0f, -0.0f, 0x1p-149f, 0x1p-130f, 0x1p-20f, 0.00173f, 0.375f, 17.3f, 1024.0f};
    const uint16_t special[] = {0x7c00, 0xfc00, 0x7c01, 0xfd55, 0x7e00};
    for (unsigned d = 0; d < sizeof(dims) / sizeof(*dims); d++) {
        const uint64_t in_dim = dims[d], blocks = (in_dim + 31) / 32;
        quant_buffer r0b = quant_alloc(blocks * 34), r1b = quant_alloc(blocks * 34);
        quant_buffer xb = quant_alloc(blocks * 32), sb = quant_alloc(blocks * sizeof(float));
        uint8_t *row0 = r0b.data, *row1 = r1b.data;
        int8_t *xq = (int8_t *)xb.data;
        float *xscale = (float *)sb.data;
        for (unsigned p = 0; p < 36; p++) {
            for (uint64_t b = 0; b < blocks; b++) {
                uint16_t h0 = half[(p + b) % 15], h1 = half[(3 * p + 7 * b) % 15];
                xscale[b] = scales[(p + 5 * b) % 9];
                if (p >= 12) { /* random finite scales: every add rounds */
                    h0 = (uint16_t)quant_random();
                    h1 = (uint16_t)quant_random();
                    if ((h0 & 0x7c00) == 0x7c00) h0 ^= 0x4000;
                    if ((h1 & 0x7c00) == 0x7c00) h1 ^= 0x4000;
                    xscale[b] = fabsf(quant_uniform(0.05f));
                }
                memcpy(row0 + b * 34, &h0, 2);
                memcpy(row1 + b * 34, &h1, 2);
                for (unsigned i = 0; i < 32; i++) {
                    row0[b * 34 + 2 + i] = (uint8_t)quant_byte(p % 6, i + (unsigned)b);
                    row1[b * 34 + 2 + i] = (uint8_t)quant_byte((p + 3) % 6, 3 * i + (unsigned)b);
                    /* Padding past in_dim must be ignored, whatever it holds. */
                    xq[b * 32 + i] = quant_byte((p / 6 + 5) % 6, 7 * i + (unsigned)b);
                }
            }
            /* One infinite or NaN weight scale (several would make the NaN
             * payload depend on add operand order, not on the kernel). */
            if (p >= 30) {
                const uint16_t h = special[p - 30 < 5 ? p - 30 : 0];
                memcpy(row1 + (quant_random() % blocks) * 34, &h, 2);
            }
            const float e0 = q8_0_dot_oracle(row0, xq, xscale, in_dim);
            const float e1 = q8_0_dot_oracle(row1, xq, xscale, in_dim);
            float a0, a1;
            quant_equal("Q8_0", e0, dot_q8_0_row(row0, xq, xscale, in_dim, blocks), (unsigned)blocks, p);
            dot_q8_0_row_pair(row0, row1, xq, xscale, in_dim, blocks, &a0, &a1);
            quant_equal("Q8_0 pair row0", e0, a0, (unsigned)blocks, p);
            quant_equal("Q8_0 pair row1", e1, a1, (unsigned)blocks, p);
            dot_q8_0_row_2(row1, xq, xscale, xq, xscale, in_dim, blocks, &a0, &a1);
            quant_equal("Q8_0 row_2", e1, a0, (unsigned)blocks, p);
            quant_equal("Q8_0 row_2", e1, a1, (unsigned)blocks, p);
        }
        quant_free(sb); quant_free(xb); quant_free(r1b); quant_free(r0b);
    }
}

#if defined(DS4_NO_GPU) && defined(__AVX2__) && defined(__F16C__)
/* The SIMD Q8_0 path converts weight scales with F16C.  Exhaustively: equal
 * bits except that F16C quiets signalling NaNs, and equal products with the
 * activation scales either way. */
static void check_f16c_scales(void) {
    /* volatile: the compiler may fold x * 1.0f to x, the hardware does not */
    static volatile float act[] = {0.0f, -0.0f, 0x1p-149f, 0x1p-130f, 0.00173f, 1.0f, 17.3f, 0x1p100f, -2.5f};
    for (uint32_t h = 0; h < 65536; h++) {
        const float sw = f16_to_f32((uint16_t)h);
        const float hw = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
        uint32_t a, b;
        memcpy(&a, &sw, sizeof(a)); memcpy(&b, &hw, sizeof(b));
        CHECK(sw != sw ? (a | 0x00400000u) == b : a == b);
        for (unsigned i = 0; i < sizeof(act) / sizeof(*act); i++) {
            const float ps = sw * act[i], ph = hw * act[i];
            memcpy(&a, &ps, sizeof(a)); memcpy(&b, &ph, sizeof(b));
            CHECK(a == b);
        }
    }
}
#endif


/* The pool hands rows out in chunks claimed by whichever thread is free, so
 * check that every row of every dispatch is claimed exactly once and never in
 * a chunk above the cap, that a nested dispatch runs inline, that the workers
 * really take part, and that a tight loop of tiny dispatches with pauses long
 * enough for the workers to fall asleep never loses a wake-up. */
typedef struct {
    uint32_t claims[64 * DS4_MAX_THREADS];
    uint64_t n_rows;
    uint64_t nested_rows;
    uint32_t threads_seen;
    uint32_t round;
    bool slow;
} pool_test_ctx;

static __thread uint32_t pool_thread_round;

static void pool_nested_worker(void *vctx, uint64_t row0, uint64_t row1) {
    pool_test_ctx *ctx = vctx;
    __atomic_fetch_add(&ctx->nested_rows, row1 - row0, __ATOMIC_RELAXED);
}

static void pool_claim_worker(void *vctx, uint64_t row0, uint64_t row1) {
    pool_test_ctx *ctx = vctx;
    CHECK(row0 < row1 && row1 <= ctx->n_rows && row1 - row0 <= 64);
    if (pool_thread_round != ctx->round) {
        pool_thread_round = ctx->round;
        __atomic_fetch_add(&ctx->threads_seen, 1, __ATOMIC_RELAXED);
    }
    for (uint64_t r = row0; r < row1; r++) {
        __atomic_fetch_add(&ctx->claims[r], 1, __ATOMIC_RELAXED);
    }
    if (row0 == 0) ds4_parallel_for_min_rows(1000, pool_nested_worker, ctx, 1);
    if (ctx->slow) {
        const double until = now_sec() + 50.0e-6 * (double)(row1 - row0);
        while (now_sec() < until) ds4_cpu_relax();
    }
}

static void pool_dispatch(pool_test_ctx *ctx, uint64_t n_rows, bool slow) {
    ctx->n_rows = n_rows;
    ctx->nested_rows = 0;
    ctx->threads_seen = 0;
    ctx->slow = slow;
    ctx->round++;
    memset(ctx->claims, 0, (size_t)n_rows * sizeof(ctx->claims[0]));
    ds4_parallel_for_min_rows(n_rows, pool_claim_worker, ctx, 1);
    for (uint64_t r = 0; r < n_rows; r++) CHECK(ctx->claims[r] == 1);
    CHECK(ctx->nested_rows == 1000);
}

static void check_thread_pool(void) {
    const uint32_t threads[] = {1, 2, 7, DS4_MAX_THREADS};
    pool_test_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx);
    alarm(600);   /* a lost wake-up would otherwise hang here forever */
    CHECK(ds4_default_threads() >= 1 && ds4_default_threads() <= DS4_MAX_THREADS);
    for (size_t i = 0; i < sizeof(threads) / sizeof(*threads); i++) {
        g_requested_threads = threads[i];
        ds4_threads_init();
        CHECK(g_pool.n_threads == threads[i]);
        pool_dispatch(ctx, 64 * threads[i], true);
        CHECK(ctx->threads_seen >= (threads[i] + 1) / 2);
        const uint32_t rounds = threads[i] == DS4_MAX_THREADS ? 2000 : 100000;
        for (uint32_t k = 0; k < rounds; k++) {
            pool_dispatch(ctx, 1 + k % 37, false);
            if (k % 2000 == 0) usleep(3 * DS4_POOL_SPIN_US);
        }
        ds4_threads_shutdown();
        printf("CPU thread pool: %u threads PASS\n", threads[i]);
    }
    g_requested_threads = 0;
    free(ctx);
}


/* Scalar references for the decode glue as it was written before the loops
 * were restructured: per-head RoPE angles, element-at-a-time HC mixing, one
 * head at a time RMS sums, and the E4M3 table search. */
static void rope_oracle(float *x, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                        uint32_t pos, uint64_t n_ctx_orig, float freq_base, float freq_scale,
                        float ext_factor, float attn_factor, bool inverse) {
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)n_rot, n_ctx_orig, freq_base, DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW, corr_dims);
    }
    for (uint32_t h = 0; h < n_head; h++) {
        float *tail = x + (uint64_t)h * head_dim + n_nope;
        float theta_extrap = (float)pos;
        for (uint32_t i = 0; i < n_rot; i += 2) {
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            const float c = cosf(theta) * mscale;
            const float s = sin_sign * sinf(theta) * mscale;
            const float x0 = tail[i + 0];
            const float x1 = tail[i + 1];
            tail[i + 0] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
            theta_extrap *= theta_scale;
        }
    }
}

static void hc_post_oracle(float *out_hc, const float *block_out, const float *residual_hc,
                           const float *post, const float *comb, uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];
            for (uint32_t src = 0; src < n_hc; src++) {
                acc += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * n_embd + d];
            }
            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}

static void hc_sum_oracle(float *out, const float *x, const float *weights,
                          uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) acc += x[(uint64_t)h * n_embd + d] * weights[h];
        out[d] = acc;
    }
}

static void head_norm_oracle(float *x, uint32_t n_head, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

static float e4m3_value_oracle(int i) {
    const int exp = (i >> 3) & 0x0f, mant = i & 0x07;
    return exp == 0 ? (float)mant * 0.001953125f
                    : (1.0f + (float)mant * 0.125f) * ldexpf(1.0f, exp - 7);
}

static float e4m3_oracle(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3_value_oracle(mid) <= ax) lo = mid; else hi = mid - 1;
    }
    if (lo < 126) {
        const float best = fabsf(ax - e4m3_value_oracle(lo));
        const float next = fabsf(ax - e4m3_value_oracle(lo + 1));
        if (next < best || (next == best && (lo & 1) != 0)) lo++;
    }
    return sign * e4m3_value_oracle(lo);
}

static float glue_random(float scale) {
    return ((float)(quant_random() >> 8) / 16777216.0f * 2.0f - 1.0f) * scale;
}

static void glue_equal(const char *name, const void *expected, const void *actual, size_t bytes, unsigned c) {
    if (memcmp(expected, actual, bytes) != 0) {
        fprintf(stderr, "%s case %u differs from the scalar reference\n", name, c);
        exit(1);
    }
}

static void check_decode_glue(void) {
    enum { HEADS = 64, DIM = 512, ROT = 64, EMBD = 4096, HC = 4 };
    float *x = malloc((size_t)HEADS * DIM * sizeof(float));
    float *a = malloc((size_t)HEADS * DIM * sizeof(float));
    float *b = malloc((size_t)HEADS * DIM * sizeof(float));
    CHECK(x && a && b);

    /* Dense and YaRN-compressed parameters as rope_tail_layer_inplace passes them. */
    const float yarn_scale = 1.0f / 16.0f;
    const float yarn_attn = 1.0f / (1.0f + 0.1f * logf(1.0f / yarn_scale));
    for (unsigned c = 0; c < 96; c++) {
        const bool yarn = c & 1, inverse = c & 2;
        const uint32_t heads = c & 4 ? 1 : HEADS, pos = c * c * 977u + c;
        for (uint32_t i = 0; i < heads * DIM; i++) x[i] = glue_random(4.0f);
        memcpy(a, x, (size_t)heads * DIM * sizeof(float));
        memcpy(b, x, (size_t)heads * DIM * sizeof(float));
        rope_oracle(a, heads, DIM, ROT, pos, yarn ? 65536 : 0, yarn ? 160000.0f : 10000.0f,
                    yarn ? yarn_scale : 1.0f, yarn ? 1.0f : 0.0f, yarn ? yarn_attn : 1.0f, inverse);
        rope_tail_ext_inplace(b, heads, DIM, ROT, pos, yarn ? 65536 : 0, yarn ? 160000.0f : 10000.0f,
                              yarn ? yarn_scale : 1.0f, yarn ? 1.0f : 0.0f, yarn ? yarn_attn : 1.0f,
                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, inverse);
        glue_equal("RoPE", a, b, (size_t)heads * DIM * sizeof(float), c);
    }

    /* Head RMS: full groups of eight, a partial group, and extreme magnitudes. */
    for (unsigned c = 0; c < 12; c++) {
        const uint32_t heads = c % 3 == 2 ? 7 : HEADS, dim = c % 3 == 2 ? 96 : DIM;
        const float mag = c < 4 ? 1.0f : c < 8 ? 1e-30f : 1e30f;
        for (uint32_t i = 0; i < heads * dim; i++) x[i] = glue_random(mag);
        if (c == 1) for (uint32_t i = 0; i < dim; i++) x[3 * dim + i] = 0.0f;
        memcpy(a, x, (size_t)heads * dim * sizeof(float));
        memcpy(b, x, (size_t)heads * dim * sizeof(float));
        head_norm_oracle(a, heads, dim, 1e-6f);
        head_rms_norm_inplace(b, heads, dim, 1e-6f);
        glue_equal("head RMS", a, b, (size_t)heads * dim * sizeof(float), c);
    }

    /* HC pre-sum and post mix with signed zeros, infinities and tiny values. */
    float post[HC], comb[HC * HC], block[EMBD];
    for (unsigned c = 0; c < 16; c++) {
        const uint32_t embd = c & 1 ? 1000 : EMBD;
        for (uint32_t i = 0; i < HC * embd; i++) x[i] = glue_random(c & 2 ? 1e-38f : 3.0f);
        for (uint32_t i = 0; i < embd; i++) block[i] = glue_random(c & 4 ? 1e30f : 1.0f);
        for (uint32_t i = 0; i < HC; i++) post[i] = glue_random(2.0f);
        for (uint32_t i = 0; i < HC * HC; i++) comb[i] = glue_random(1.0f);
        if (c == 8) { x[5] = -0.0f; block[5] = -0.0f; post[1] = -0.0f; x[embd + 9] = INFINITY; }
        hc_post_oracle(a, block, x, post, comb, embd, HC);
        hc_post_one(b, block, x, post, comb, embd, HC);
        glue_equal("HC post", a, b, (size_t)HC * embd * sizeof(float), c);
        hc_sum_oracle(a, x, post, embd, HC);
        hc_weighted_sum_one(b, x, post, embd, HC);
        glue_equal("HC sum", a, b, (size_t)embd * sizeof(float), c);
    }

    /* E4M3: every midpoint between grid values, then a sweep of bit patterns.
     * Signaling NaNs are skipped: the only caller passes quotients. */
    for (int i = 0; i < 126; i++) {
        const float mid = 0.5f * (e4m3_value_oracle(i) + e4m3_value_oracle(i + 1));
        const float probes[] = {mid, nextafterf(mid, 0.0f), nextafterf(mid, 1000.0f), -mid};
        for (unsigned j = 0; j < 4; j++) {
            const float e = e4m3_oracle(probes[j]), g = dsv4_e4m3fn_dequant_cpu(probes[j]);
            glue_equal("E4M3 tie", &e, &g, sizeof(e), (unsigned)i);
        }
    }
    for (uint64_t u = 0; u < (1ull << 32); u += 1031) {
        const uint32_t bits = (uint32_t)u;
        if ((bits & 0x7fc00000u) == 0x7f800000u && (bits & 0x3fffffu)) continue;
        float v;
        memcpy(&v, &bits, sizeof(v));
        const float e = e4m3_oracle(v), g = dsv4_e4m3fn_dequant_cpu(v);
        glue_equal("E4M3 sweep", &e, &g, sizeof(e), bits);
    }

    free(b); free(a); free(x);
}

/* Serial sink-aware attention, one head after the other, as the CPU decode
 * path computed it before the lane-wise kernel. */
static void lanes_oracle(float *out, const float *sinks, const float *q,
                         const float *raw, unsigned n_raw,
                         const float *comp, unsigned n_comp, const bool *allowed) {
    const unsigned d = DS4_N_HEAD_DIM;
    const float kq_scale = 1.0f / sqrtf((float)d);
    float score[1024];
    CHECK(n_raw + n_comp <= 1024);
    for (unsigned h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + h * d;
        float max_score = sinks[h];
        for (unsigned r = 0; r < n_raw; r++) {
            score[r] = dot_f32(qh, raw + r * d, d) * kq_scale;
            if (score[r] > max_score) max_score = score[r];
        }
        for (unsigned c = 0; c < n_comp; c++) {
            if (allowed && !allowed[c]) { score[n_raw + c] = DS4_NEG_INF; continue; }
            score[n_raw + c] = dot_f32(qh, comp + c * d, d) * kq_scale;
            if (score[n_raw + c] > max_score) max_score = score[n_raw + c];
        }
        float *oh = out + h * d;
        memset(oh, 0, d * sizeof(oh[0]));
        float denom = expf(sinks[h] - max_score);
        for (unsigned r = 0; r < n_raw; r++) {
            const float weight = expf(score[r] - max_score);
            denom += weight;
            axpy_f32(oh, raw + r * d, weight, d);
        }
        for (unsigned c = 0; c < n_comp; c++) {
            if (score[n_raw + c] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[n_raw + c] - max_score);
            denom += weight;
            axpy_f32(oh, comp + c * d, weight, d);
        }
        const float inv = 1.0f / denom;
        scale_f32(oh, inv, d);
    }
}

/* Magnitude mag with signed zeros mixed in, so the score range, the running
 * max and the sign of zero products are all exercised. */
static float lanes_value(float mag) {
    const uint32_t r = quant_random();
    if (r % 64 == 0) return -0.0f;
    if (r % 64 == 1) return 0.0f;
    return (float)(int32_t)quant_random() * 0x1p-31f * mag;
}

/* The lane-wise decode attention must stay bitwise the serial oracle for
 * every row count, mask, pool size and score stride, including the raw-only
 * case that layer_attention_rows_one used to serve. */
static void check_attention_lanes(void) {
    const unsigned d = DS4_N_HEAD_DIM, nh = DS4_N_HEAD;
    const unsigned raws[] = {0, 1, 5, 128}, comps[] = {0, 1, 17, 384, 520};
    const uint32_t threads[] = {1, 3, 8};
    quant_buffer qb = quant_alloc(nh * d * sizeof(float)), sb = quant_alloc(nh * sizeof(float));
    quant_buffer rb = quant_alloc(128 * d * sizeof(float)), cb = quant_alloc(520 * d * sizeof(float));
    quant_buffer ab = quant_alloc(520 * sizeof(bool));
    quant_buffer eb = quant_alloc(nh * d * sizeof(float)), ob = quant_alloc(nh * d * sizeof(float));
    quant_buffer qtb = quant_alloc(nh * d * sizeof(float)), ib = quant_alloc(nh * sizeof(float));
    float *q = (float *)qb.data, *sinks = (float *)sb.data;
    float *raw = (float *)rb.data, *comp = (float *)cb.data;
    float *expected = (float *)eb.data, *out = (float *)ob.data;
    bool *allowed = (bool *)ab.data;
    ds4_model m = {.map = sb.data, .size = nh * sizeof(float)};
    ds4_tensor t = {.type = DS4_TENSOR_F32, .ndim = 1, .dim = {nh},
                    .elements = nh, .bytes = nh * sizeof(float)};
    ds4_layer_weights l = {.attn_sinks = &t};

    for (size_t ti = 0; ti < sizeof(threads) / sizeof(*threads); ti++) {
        g_requested_threads = threads[ti];
        for (size_t r = 0; r < sizeof(raws) / sizeof(*raws); r++) {
            for (size_t c = 0; c < sizeof(comps) / sizeof(*comps); c++) {
                for (unsigned mask = 0; mask < (comps[c] ? 2u : 1u); mask++) {
                    const unsigned n_raw = raws[r], n_comp = comps[c];
                    const float mag = (float)(1 << (3 * ((r + c + mask) % 4)));
                    for (unsigned i = 0; i < nh * d; i++) q[i] = lanes_value(mag);
                    for (unsigned i = 0; i < 128 * d; i++) raw[i] = lanes_value(2.0f);
                    f16_round_inplace_cpu(raw, 128 * d);
                    for (unsigned i = 0; i < 520 * d; i++) comp[i] = lanes_value(1.0f);
                    for (unsigned i = 0; i < nh; i++) sinks[i] = lanes_value(8.0f);
                    for (unsigned i = 0; i < 520; i++) allowed[i] = quant_random() % 3 != 0;
                    const bool *allow = mask ? allowed : NULL;
                    lanes_oracle(expected, sinks, q, raw, n_raw, comp, n_comp, allow);
                    for (unsigned pad = 0; pad < 14; pad += 13) {
                        const uint32_t cap = n_raw + n_comp + pad;
                        quant_buffer scb = quant_alloc((size_t)nh * cap * sizeof(float));
                        quant_buffer rwb = quant_alloc((size_t)cap * sizeof(float *));
                        ds4_cpu_decode_scratch s = {
                            .attn_score_cap = cap, .attn_score = (float *)scb.data,
                            .attn_qt = (float *)qtb.data, .attn_rows = (const float **)rwb.data,
                            .attn_inv = (float *)ib.data,
                        };
                        memset(out, 0xff, nh * d * sizeof(out[0]));
                        layer_attention_mixed_one_decode_scratch(out, &m, &l, q, raw, n_raw,
                                                                 comp, n_comp, allow, &s);
                        CHECK(!memcmp(out, expected, nh * d * sizeof(out[0])));
                        quant_free(rwb); quant_free(scb);
                    }
                    if (n_comp == 0) {
                        layer_attention_rows_one(expected, &m, &l, q, raw, n_raw);
                        CHECK(!memcmp(out, expected, nh * d * sizeof(out[0])));
                    }
                }
            }
        }
        ds4_threads_shutdown();
    }
    g_requested_threads = 0;
    quant_free(ib); quant_free(qtb); quant_free(ob); quant_free(eb); quant_free(ab);
    quant_free(cb); quant_free(rb); quant_free(sb); quant_free(qb);
}

static float quant_float(void) {
    return (float)((int)(quant_random() % 2001) - 1000) * 0.00137f;
}

/* Any half except Inf/NaN, subnormals included. */
static uint16_t quant_half_finite(void) {
    uint16_t h = (uint16_t)quant_random();
    if ((h & 0x7c00u) == 0x7c00u) h ^= 0x4000u;
    return h;
}

static void quant_bits_equal(const char *name, float expected, float actual, unsigned where) {
    uint32_t a, b;
    memcpy(&a, &expected, sizeof(a)); memcpy(&b, &actual, sizeof(b));
    if (a != b) {
        fprintf(stderr, "%s at %u: %08x != %08x (%g != %g)\n", name, where, a, b, expected, actual);
        exit(1);
    }
}

/* The raw KV cache stores F16-rounded rows and prefill rounds its batch copy
 * the same way before pushing, so a second round trip must change nothing. */
static void check_f16_round_trip(void) {
    for (uint32_t h = 0; h < 0x10000u; h++) {
        if ((h & 0x7c00u) == 0x7c00u && (h & 0x03ffu) != 0) continue; /* NaN payloads collapse */
        const float f = f16_to_f32((uint16_t)h);
        CHECK(f32_to_f16(f) == (uint16_t)h);
        quant_bits_equal("f16 round trip", f, f16_to_f32(f32_to_f16(f)), h);
    }
    for (unsigned i = 0; i < 200000; i++) {
        uint32_t bits = quant_random();
        float x;
        memcpy(&x, &bits, sizeof(x));
        const float once = f16_to_f32(f32_to_f16(x));
        quant_bits_equal("f16 double round trip", once, f16_to_f32(f32_to_f16(once)), i);
    }
}

/* Prefix prefill attention must match the per-token attention each token
 * would run against its raw window and the compressed rows pushed so far,
 * with and without an indexer mask. */
static void check_prefix_attention(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_shape.n_head = 3;
    g_ds4_shape.n_head_dim = 40;
    const uint32_t n_head = DS4_N_HEAD, head_dim = DS4_N_HEAD_DIM;
    const uint32_t n_tok = 23, raw_cap = 6, n_comp_rows = 9, comp_cap = 16;
    const uint64_t q_dim = (uint64_t)n_head * head_dim;
    const uint64_t allowed_stride = (comp_cap + 7u) / 8u;

    quant_buffer sb = quant_alloc(n_head * sizeof(float));
    float *sinks = (float *)sb.data;
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = quant_float();
    ds4_model m = {.map = sb.data, .size = sb.size};
    ds4_tensor sinks_t = {.type = DS4_TENSOR_F32, .ndim = 1, .dim = {n_head},
                          .elements = n_head, .bytes = n_head * sizeof(float)};
    ds4_layer_weights layer = {.attn_sinks = &sinks_t};

    float *q = xmalloc((size_t)n_tok * q_dim * sizeof(q[0]));
    float *kv = xmalloc((size_t)n_tok * head_dim * sizeof(kv[0]));
    float *comp = xmalloc((size_t)n_comp_rows * head_dim * sizeof(comp[0]));
    float *batch = xmalloc((size_t)n_tok * q_dim * sizeof(batch[0]));
    float *one = xmalloc(q_dim * sizeof(one[0]));
    uint32_t *comp_counts = xmalloc((size_t)n_tok * sizeof(comp_counts[0]));
    uint8_t *mask = xmalloc(n_tok);
    uint8_t *bits = xmalloc((size_t)n_tok * allowed_stride);
    bool *allowed = xmalloc(n_comp_rows * sizeof(allowed[0]));
    for (uint64_t i = 0; i < (uint64_t)n_tok * q_dim; i++) q[i] = quant_float();
    for (uint64_t i = 0; i < (uint64_t)n_tok * head_dim; i++) kv[i] = quant_float();
    for (uint64_t i = 0; i < (uint64_t)n_comp_rows * head_dim; i++) comp[i] = quant_float();
    for (uint32_t t = 0; t < n_tok; t++) {
        comp_counts[t] = (t + 1) / 3 < n_comp_rows ? (t + 1) / 3 : n_comp_rows;
        mask[t] = (uint8_t)(t % 3 == 1);
        for (uint64_t b = 0; b < allowed_stride; b++) bits[t * allowed_stride + b] = (uint8_t)quant_random();
    }

    for (unsigned masked = 0; masked < 2; masked++) {
        layer_attention_prefix_batch(batch, &m, &layer, q, kv, comp, comp_counts,
                                     masked ? mask : NULL, masked ? bits : NULL,
                                     allowed_stride, n_tok, raw_cap);
        for (uint32_t t = 0; t < n_tok; t++) {
            const uint32_t raw_count = t + 1 < raw_cap ? t + 1 : raw_cap;
            const bool use_mask = masked && mask[t];
            for (uint32_t c = 0; c < comp_counts[t]; c++)
                allowed[c] = (bits[t * allowed_stride + (c >> 3)] >> (c & 7u)) & 1u;
            layer_attention_mixed_one(one, &m, &layer, q + t * q_dim,
                                      kv + (uint64_t)(t + 1 - raw_count) * head_dim, raw_count,
                                      comp, comp_counts[t], use_mask ? allowed : NULL);
            for (uint64_t i = 0; i < q_dim; i++)
                quant_bits_equal(masked ? "masked prefix attention" : "prefix attention",
                                 one[i], batch[t * q_dim + i], (unsigned)(t * q_dim + i));
        }
    }

    /* Raw-only layers: no compressed rows, the per-token path is rows_one. */
    memset(comp_counts, 0, (size_t)n_tok * sizeof(comp_counts[0]));
    layer_attention_prefix_batch(batch, &m, &layer, q, kv, NULL, comp_counts, NULL, NULL, 0, n_tok, raw_cap);
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t raw_count = t + 1 < raw_cap ? t + 1 : raw_cap;
        layer_attention_rows_one(one, &m, &layer, q + t * q_dim,
                                 kv + (uint64_t)(t + 1 - raw_count) * head_dim, raw_count);
        for (uint64_t i = 0; i < q_dim; i++)
            quant_bits_equal("raw prefix attention", one[i], batch[t * q_dim + i], (unsigned)(t * q_dim + i));
    }

    free(allowed); free(bits); free(mask); free(comp_counts);
    free(one); free(batch); free(comp); free(kv); free(q);
    quant_free(sb);
    g_ds4_shape = saved;
}

int main(void) {
    check_thread_pool();
    check_f16_round_trip();
    check_prefix_attention();
    puts("CPU prefix attention bitwise oracle PASS");
    check_integer_dots();
    check_iq2_tables();
#if defined(DS4_NO_GPU) && defined(__AVX2__) && defined(__F16C__)
    check_f16c_scales();
#endif
    check_q8_quantizers();
    puts("CPU Q8_0/Q8_K activation quantizers match the scalar code bitwise");
    check_indexer();
    check_indexer_selection();
    puts("CPU indexer row-parallel scoring and top-k selection oracle PASS");
    check_attention_lanes();
    puts("CPU lane-wise decode attention bitwise oracle: 1/3/8 threads PASS");
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    puts("CPU ARM DOTPROD Q2_K integer oracle PASS");
    puts("Block FP oracle skipped: existing ARM DOTPROD accumulation order differs");
#else
    check_block_dots();
    check_extremes();
    check_q8_0_dots();
    puts("CPU Q8_0 row dots match the scalar block order bitwise");
    check_f16_rows();
#if defined(DS4_NO_GPU) && defined(__AVX2__) && defined(__F16C__)
    check_f16_conversion();
    puts("CPU F16C conversion and F16 row-group matvec bitwise oracles PASS");
#endif
#if defined(DS4_NO_GPU) && defined(__AVX512BW__)
    puts("CPU AVX-512 IQ2_XXS/Q2_K integer and bitwise block oracles PASS");
#elif defined(DS4_NO_GPU) && defined(__AVX2__)
    puts("CPU AVX2 IQ2_XXS/Q2_K integer and bitwise block oracles PASS");
#elif defined(__ARM_NEON)
    puts("CPU NEON IQ2_XXS/Q2_K integer and bitwise block oracles PASS");
#else
    puts("CPU scalar IQ2_XXS/Q2_K integer and bitwise block oracles PASS");
#endif
#endif
    check_decode_glue();
    puts("CPU decode glue (RoPE, HC, head RMS, E4M3) bitwise oracles PASS");
    return 0;
}
