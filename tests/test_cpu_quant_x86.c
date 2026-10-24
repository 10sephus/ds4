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
    /* 528, 544 and 545 rows for 512: the one-pass selection's 16 and 32
     * dropped rows, and one past its limit. */
    const unsigned counts[] = {513, 528, 544, 545, 600, 1500};
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


/* Pooled F16 matvecs claim whole 8-row groups, and the fused decode
 * projections mix Q8_0 rows and F16 groups of several matrices in one
 * dispatch: every row must still equal its own row dot bitwise, whatever
 * the thread count, NUMA mode or partial last group. */
static void check_f16_dispatch(void) {
    enum { DIM = 1027, ROWS_MAX = 1024 };
    const unsigned rows[] = {1, 7, 24, 64, 256, 1000, ROWS_MAX};
    const size_t n_w = sizeof(rows) / sizeof(*rows);
    size_t bytes = 0;
    for (size_t k = 0; k < n_w; k++) bytes += (size_t)rows[k] * DIM * 2;
    uint16_t *map = malloc(bytes);
    float *x = malloc(DIM * sizeof(float));
    float *expected = malloc(ROWS_MAX * sizeof(float)), *actual = malloc(4 * ROWS_MAX * sizeof(float));
    CHECK(map && x && expected && actual);
    ds4_model m = {0};
    m.map = (const uint8_t *)map;
    m.size = bytes;
    ds4_tensor w[sizeof(rows) / sizeof(*rows)];
    size_t off = 0;
    for (size_t k = 0; k < n_w; k++) {
        memset(&w[k], 0, sizeof(w[k]));
        w[k].ndim = 2; w[k].dim[0] = DIM; w[k].dim[1] = rows[k]; w[k].type = 1;
        w[k].elements = (uint64_t)DIM * rows[k]; w[k].bytes = w[k].elements * 2; w[k].abs_offset = off;
        for (uint64_t i = 0; i < w[k].elements; i++) map[off / 2 + i] = f16_random_weight();
        off += w[k].bytes;
    }
    for (unsigned i = 0; i < DIM; i++) x[i] = (float)(int32_t)quant_random() / 2147483648.0f;

    ds4_numa_init();
    const bool numa = g_numa.on;
    const uint32_t threads[] = {1, 3, 16};
    for (int mode = numa; mode >= 0; mode--) {
        g_numa.on = mode;
        for (size_t t = 0; t < sizeof(threads) / sizeof(*threads); t++) {
            g_requested_threads = threads[t];
            ds4_threads_init();
            for (size_t k = 0; k < n_w; k++) {
                const uint16_t *data = map + w[k].abs_offset / 2;
                for (unsigned r = 0; r < rows[k]; r++) expected[r] = dot_f16_row(data + (uint64_t)r * DIM, x, DIM);
                memset(actual, 0xff, ROWS_MAX * sizeof(float));
                matvec_f16(actual, &m, &w[k], x);
                f16_rows_equal("F16 grouped matvec", expected, actual, rows[k]);
            }
            for (size_t k0 = 0; k0 + 4 <= n_w; k0++) {
                const ds4_tensor *ws[4] = {&w[k0], &w[k0 + 1], &w[k0 + 2], &w[k0 + 3]};
                float *outs[4] = {actual, actual + ROWS_MAX, actual + 2 * ROWS_MAX, actual + 3 * ROWS_MAX};
                for (uint32_t n = 1; n <= 4; n++) {
                    memset(actual, 0xff, 4 * ROWS_MAX * sizeof(float));
                    matvec_f16_multi(outs, &m, ws, n, x);
                    for (uint32_t j = 0; j < n; j++) {
                        const uint16_t *data = map + ws[j]->abs_offset / 2;
                        for (unsigned r = 0; r < ws[j]->dim[1]; r++) {
                            expected[r] = dot_f16_row(data + (uint64_t)r * DIM, x, DIM);
                        }
                        f16_rows_equal("F16 multi matvec", expected, outs[j], (unsigned)ws[j]->dim[1]);
                    }
                }
            }
            ds4_threads_shutdown();
        }
    }
    g_numa.on = numa;
    g_requested_threads = 0;
    free(actual);
    free(expected);
    free(x);
    free(map);
}

/* The fused decode projections of attn_norm (q_a and kv in Q8_0, compressor
 * rows in F16) against dot_q8_0_row and dot_f16_row per row, for ratio 0, 4
 * and 128 layers. */
static void check_attn_input_projections(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH;
    const uint64_t in = DS4_N_EMBD, blocks = in / 32;
    const uint64_t q8_rows[2] = {DS4_N_LORA_Q, DS4_N_HEAD_DIM};
    const uint64_t f16_rows[6] = {2 * DS4_N_HEAD_DIM, 2 * DS4_N_HEAD_DIM,
                                  2 * DS4_N_INDEXER_HEAD_DIM, 2 * DS4_N_INDEXER_HEAD_DIM,
                                  DS4_N_HEAD_DIM, DS4_N_HEAD_DIM};
    uint64_t bytes = 0;
    for (int k = 0; k < 2; k++) bytes += q8_rows[k] * blocks * 34;
    for (int k = 0; k < 6; k++) bytes += f16_rows[k] * in * 2;
    uint8_t *map = malloc(bytes);
    CHECK(map);
    ds4_model m = {0};
    m.map = map;
    m.size = bytes;
    ds4_tensor t[8];
    uint64_t off = 0;
    for (int k = 0; k < 8; k++) {
        const bool f16 = k >= 2;
        const uint64_t r = f16 ? f16_rows[k - 2] : q8_rows[k];
        memset(&t[k], 0, sizeof(t[k]));
        t[k].ndim = 2; t[k].dim[0] = in; t[k].dim[1] = r; t[k].type = f16 ? 1 : 8;
        t[k].elements = in * r; t[k].bytes = f16 ? in * r * 2 : r * blocks * 34; t[k].abs_offset = off;
        if (f16) {
            uint16_t *p = (uint16_t *)(map + off);
            for (uint64_t i = 0; i < in * r; i++) p[i] = f16_random_weight();
        } else {
            for (uint64_t b = 0; b < r * blocks; b++) {
                uint8_t *blk = map + off + b * 34;
                const uint16_t d = f32_to_f16(0.0005f + (float)(quant_random() & 1023) * 1.0e-5f);
                memcpy(blk, &d, 2);
                for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)quant_random();
            }
        }
        off += t[k].bytes;
    }
    float *x = malloc(in * sizeof(float));
    CHECK(x);
    for (uint64_t i = 0; i < in; i++) x[i] = (float)(int32_t)quant_random() / 1073741824.0f;
    int8_t *xq = malloc(in);
    float *xs = malloc(blocks * sizeof(float)), *expected = malloc(2 * DS4_N_HEAD_DIM * sizeof(float));
    CHECK(xq && xs && expected);
    quantize_q8_0_activation(x, xq, xs, in);

    ds4_cpu_decode_scratch sc;
    memset(&sc, 0, sizeof(sc));
    const uint64_t comp = 2 * (DS4_N_HEAD_DIM + DS4_N_INDEXER_HEAD_DIM);
    sc.qr = malloc(DS4_N_LORA_Q * sizeof(float));
    sc.kv_raw = malloc(DS4_N_HEAD_DIM * sizeof(float));
    sc.comp_kv_cur = malloc(comp * sizeof(float));
    sc.comp_sc_cur = malloc(comp * sizeof(float));
    sc.q8_cap = in;
    sc.q8_xq = malloc(in);
    sc.q8_xscale = malloc(blocks * sizeof(float));
    CHECK(sc.qr && sc.kv_raw && sc.comp_kv_cur && sc.comp_sc_cur && sc.q8_xq && sc.q8_xscale);

    const uint32_t threads[] = {1, 5, 16};
    for (size_t ti = 0; ti < sizeof(threads) / sizeof(*threads); ti++) {
        g_requested_threads = threads[ti];
        ds4_threads_init();
        for (int ri = 0; ri < 3; ri++) {
            const uint32_t ratio = ri == 0 ? 0 : ri == 1 ? 4 : 128;
            ds4_layer_weights L;
            memset(&L, 0, sizeof(L));
            L.attn_q_a = &t[0];
            L.attn_kv = &t[1];
            if (ratio == 4) {
                L.attn_compressor_kv = &t[2]; L.attn_compressor_gate = &t[3];
                L.indexer_compressor_kv = &t[4]; L.indexer_compressor_gate = &t[5];
            } else if (ratio == 128) {
                L.attn_compressor_kv = &t[6]; L.attn_compressor_gate = &t[7];
            }
            memset(sc.qr, 0xff, DS4_N_LORA_Q * sizeof(float));
            memset(sc.kv_raw, 0xff, DS4_N_HEAD_DIM * sizeof(float));
            memset(sc.comp_kv_cur, 0xff, comp * sizeof(float));
            memset(sc.comp_sc_cur, 0xff, comp * sizeof(float));
            bool comp_done = false;
            CHECK(attn_input_projections_decode(&m, &L, ratio, x, &sc, &comp_done));
            CHECK(comp_done == (ratio != 0));
            float *q8_out[2] = {sc.qr, sc.kv_raw};
            for (int k = 0; k < 2; k++) {
                const uint8_t *data = map + t[k].abs_offset;
                for (uint64_t r = 0; r < q8_rows[k]; r++) {
                    expected[r] = dot_q8_0_row(data + r * blocks * 34, xq, xs, in, blocks);
                }
                f16_rows_equal("fused Q8_0 projection", expected, q8_out[k], (unsigned)q8_rows[k]);
            }
            const ds4_tensor *fw[4] = {L.attn_compressor_kv, L.attn_compressor_gate,
                                       L.indexer_compressor_kv, L.indexer_compressor_gate};
            float *f16_out[4] = {sc.comp_kv_cur, sc.comp_sc_cur,
                                 sc.comp_kv_cur + 2 * DS4_N_HEAD_DIM, sc.comp_sc_cur + 2 * DS4_N_HEAD_DIM};
            for (int k = 0; k < (ratio == 4 ? 4 : ratio ? 2 : 0); k++) {
                const uint16_t *data = (const uint16_t *)(map + fw[k]->abs_offset);
                for (uint64_t r = 0; r < fw[k]->dim[1]; r++) expected[r] = dot_f16_row(data + r * in, x, in);
                f16_rows_equal("fused F16 projection", expected, f16_out[k], (unsigned)fw[k]->dim[1]);
            }
        }
        ds4_threads_shutdown();
    }
    g_requested_threads = 0;
    g_ds4_shape = saved;
    free(sc.q8_xscale); free(sc.q8_xq); free(sc.comp_sc_cur); free(sc.comp_kv_cur);
    free(sc.kv_raw); free(sc.qr);
    free(expected); free(xs); free(xq); free(x); free(map);
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

#if defined(DS4_NO_GPU) && defined(__AVX512VNNI__)
/* The tile GEMM converts Q8_0 block scales with F16C; every half pattern,
 * including subnormals, infinities and both NaN kinds, must match the
 * software conversion bit for bit. */
static void check_q8_0_scale_f32(void) {
    for (uint32_t h = 0; h < 65536; h++) {
        const float expected = f16_to_f32((uint16_t)h);
        const float actual = q8_0_scale_f32((uint16_t)h);
        if (memcmp(&expected, &actual, sizeof(expected)) != 0) {
            fprintf(stderr, "q8_0_scale_f32(%04x) differs from f16_to_f32\n", h);
            exit(1);
        }
    }
}

/* Weight rows: random quants, block scales cycling through the special
 * halves so a 4-row group mixes subnormal, huge, inf and NaN scales. */
static void q8_0_fill_rows(uint8_t *data, uint64_t rows, uint64_t blocks, unsigned salt) {
    static const uint16_t half[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x8400, 0x1801, 0x9a17,
                                    0x3555, 0xba31, 0x7bff, 0xfbff, 0x7c00, 0x7c01, 0x7e00};
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *block = data + (r * blocks + b) * 34;
            const uint16_t scale = half[(r + 3 * b + salt) % (sizeof(half) / sizeof(*half))];
            memcpy(block, &scale, sizeof(scale));
            for (unsigned i = 0; i < 32; i++) block[2 + i] = (uint8_t)quant_random();
        }
    }
}

/* Activations: pattern 0 comes from the real quantizer, the others are
 * adversarial quants (all 127, all -128, alternating, lane pattern, random)
 * with cycling block scales, and garbage past in_dim inside the last block
 * because neither path may read it. */
static void q8_0_fill_activations(int8_t *xq, float *xscale, const float *x, unsigned pattern,
                                  uint64_t n_tok, uint64_t in_dim, uint64_t blocks) {
    static const float scales[] = {0.0f, -0.0f, 0x1p-140f, 0x1p-20f, 0.375f, -1.13f, 17.3f, 3.0e38f};
    if (pattern == 0) {
        quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
        return;
    }
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t b = 0; b < blocks; b++) {
            xscale[t * blocks + b] = scales[(t + b + pattern) % 8];
        }
        for (uint64_t i = 0; i < blocks * 32; i++) {
            const int8_t v = i < in_dim ? quant_byte(pattern - 1, (unsigned)(i + 5 * t))
                                        : (int8_t)quant_random();
            xq[t * blocks * 32 + i] = v;
        }
    }
}

static void q8_0_expect_batch(const char *name, const float *out, const uint8_t *data,
                              const int8_t *xq, const float *xscale, uint64_t n_tok,
                              uint64_t out_stride, uint64_t col0, uint64_t rows,
                              uint64_t in_dim, uint64_t blocks, unsigned pattern) {
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t r = 0; r < rows; r++) {
            const float expected = dot_q8_0_row(data + r * blocks * 34, xq + t * blocks * 32,
                                                xscale + t * blocks, in_dim, blocks);
            const float actual = out[t * out_stride + col0 + r];
            /* The adversarial scales make an inf * 0 accumulator meet a quieted
             * sNaN block product; which of the two NaNs survives the add is the
             * operand order the compiler chose for a commutative add, not a
             * property of either kernel (the 8-block row dot and the tile
             * kernel pick differently), so two NaNs count as equal. */
            if (expected != expected && actual != actual) continue;
            quant_equal(name, expected, actual, (unsigned)blocks, pattern);
        }
    }
}

/* Whole batched matmuls (tile packing, row blocks, duplicate tail rows,
 * tail tokens and groups) against one dot_q8_0_row per (row, token). */
static void check_q8_0_batch(void) {
    const uint64_t dims[] = {32, 64, 96, 100, 130, 200, 1024, 4096};
    const uint64_t rows[] = {1, 3, 4, 5, 13, 64, 513};
    const uint64_t toks[] = {1, 2, 4, 5, 8, 15, 16, 17, 20, 32, 37, 48};
    for (size_t d = 0; d < sizeof(dims) / sizeof(*dims); d++) {
        const uint64_t in_dim = dims[d], blocks = (in_dim + 31) / 32;
        for (size_t rr = 0; rr < sizeof(rows) / sizeof(*rows); rr++) {
            const uint64_t n_rows = rows[rr];
            if (n_rows > 64 && in_dim != 1024) continue;   /* one parallel shape is enough */
            const size_t bytes = (size_t)n_rows * blocks * 34;
            quant_buffer w0 = quant_alloc(2 * bytes);
            q8_0_fill_rows(w0.data, 2 * n_rows, blocks, (unsigned)(d + rr));
            const ds4_model m = {.map = w0.data, .size = 2 * bytes};
            const ds4_tensor a = {.type = DS4_TENSOR_Q8_0, .ndim = 2, .dim = {in_dim, n_rows},
                                  .elements = in_dim * n_rows, .bytes = bytes};
            ds4_tensor b = a;
            b.abs_offset = bytes;
            for (size_t tt = 0; tt < sizeof(toks) / sizeof(*toks); tt++) {
                const uint64_t n_tok = toks[tt];
                quant_buffer xb = quant_alloc(n_tok * in_dim * sizeof(float));
                quant_buffer qb = quant_alloc(n_tok * blocks * 32);
                quant_buffer sb = quant_alloc(n_tok * blocks * sizeof(float));
                quant_buffer o0 = quant_alloc(n_tok * n_rows * sizeof(float));
                quant_buffer o1 = quant_alloc(n_tok * n_rows * sizeof(float));
                float *x = (float *)xb.data, *out0 = (float *)o0.data, *out1 = (float *)o1.data;
                int8_t *xq = (int8_t *)qb.data;
                float *xscale = (float *)sb.data;
                for (uint64_t i = 0; i < n_tok * in_dim; i++) {
                    x[i] = (float)((int)(quant_random() % 2001) - 1000) * 0.00117f;
                }
                for (unsigned p = 0; p < 6; p++) {
                    q8_0_fill_activations(xq, xscale, x, p, n_tok, in_dim, blocks);
                    matmul_q8_0_batch_prequant(out0, &m, &a, xq, xscale, n_tok);
                    q8_0_expect_batch("Q8_0 batch", out0, w0.data, xq, xscale, n_tok,
                                      n_rows, 0, n_rows, in_dim, blocks, p);
                    matmul_q8_0_pair_batch_prequant(out0, out1, &m, &a, &b, xq, xscale, n_tok);
                    q8_0_expect_batch("Q8_0 pair 0", out0, w0.data, xq, xscale, n_tok,
                                      n_rows, 0, n_rows, in_dim, blocks, p);
                    q8_0_expect_batch("Q8_0 pair 1", out1, w0.data + bytes, xq, xscale, n_tok,
                                      n_rows, 0, n_rows, in_dim, blocks, p);
                }
                /* Grouped: two groups of n_rows rows, activations [token][group]. */
                quant_buffer gx = quant_alloc(n_tok * 2 * in_dim * sizeof(float));
                quant_buffer gq = quant_alloc(n_tok * 2 * blocks * 32);
                quant_buffer gs = quant_alloc(n_tok * 2 * blocks * sizeof(float));
                quant_buffer go = quant_alloc(n_tok * 2 * n_rows * sizeof(float));
                float *gxf = (float *)gx.data, *gout = (float *)go.data;
                const ds4_tensor g = {.type = DS4_TENSOR_Q8_0, .ndim = 2, .dim = {in_dim, 2 * n_rows},
                                      .elements = in_dim * 2 * n_rows, .bytes = 2 * bytes};
                for (uint64_t i = 0; i < n_tok * 2 * in_dim; i++) {
                    gxf[i] = (float)((int)(quant_random() % 2001) - 1000) * 0.00117f;
                }
                quantize_q8_0_activation_batch(gxf, (int8_t *)gq.data, (float *)gs.data,
                                               n_tok * 2, in_dim);
                matmul_q8_0_grouped_batch(gout, &m, &g, gxf, n_tok, 2, in_dim, n_rows);
                for (unsigned group = 0; group < 2; group++) {
                    for (uint64_t t = 0; t < n_tok; t++) {
                        const uint64_t src = t * 2 + group;
                        q8_0_expect_batch("Q8_0 grouped", gout + t * 2 * n_rows + group * n_rows,
                                          w0.data + group * bytes, (int8_t *)gq.data + src * blocks * 32,
                                          (float *)gs.data + src * blocks, 1, 0, 0, n_rows,
                                          in_dim, blocks, group);
                    }
                }
                quant_free(go); quant_free(gs); quant_free(gq); quant_free(gx);
                quant_free(o1); quant_free(o0); quant_free(sb); quant_free(qb); quant_free(xb);
            }
            quant_free(w0);
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
    uint64_t group_rows;
    uint64_t half_rows;
    uint64_t nested_rows;
    uint32_t threads_seen;
    uint32_t round;
    bool slow;
    bool tasks;   /* dispatched by ds4_parallel_for_tasks(): one task per claim */
} pool_test_ctx;

static __thread uint32_t pool_thread_round;

static void pool_nested_worker(void *vctx, uint64_t row0, uint64_t row1) {
    pool_test_ctx *ctx = vctx;
    __atomic_fetch_add(&ctx->nested_rows, row1 - row0, __ATOMIC_RELAXED);
}

static void pool_claim_worker(void *vctx, uint64_t row0, uint64_t row1) {
    pool_test_ctx *ctx = vctx;
    CHECK(row0 < row1 && row1 <= ctx->n_rows);
    CHECK(g_pool_thread < g_pool.n_threads);
    if (g_pool.n_threads > 1) {   /* one thread runs the whole range inline */
        CHECK(row1 - row0 <= (ctx->tasks ? 1 : 64));
        /* A claim never crosses the half of a group that belongs to one node. */
        const uint64_t g = row0 / ctx->group_rows;
        const uint64_t i = row0 - g * ctx->group_rows;
        CHECK(row1 <= g * ctx->group_rows + (i < ctx->half_rows ? ctx->half_rows : ctx->group_rows));
    }
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

static void pool_dispatch(pool_test_ctx *ctx, uint64_t n_rows, uint64_t group_rows, bool slow) {
    ctx->n_rows = n_rows;
    ctx->group_rows = group_rows;
    ctx->half_rows = g_numa.on ? group_rows / 2 : group_rows;
    ctx->nested_rows = 0;
    ctx->threads_seen = 0;
    ctx->slow = slow;
    ctx->tasks = false;
    ctx->round++;
    memset(ctx->claims, 0, (size_t)n_rows * sizeof(ctx->claims[0]));
    ds4_pool_dispatch(n_rows, pool_claim_worker, ctx, group_rows, 1, 64);
    for (uint64_t r = 0; r < n_rows; r++) CHECK(ctx->claims[r] == 1);
    CHECK(ctx->nested_rows == 1000);
}

/* Every task its own claim, the tasks split between the nodes at the midpoint. */
static void pool_dispatch_tasks(pool_test_ctx *ctx, uint64_t n_tasks) {
    ctx->n_rows = n_tasks;
    ctx->group_rows = n_tasks;
    ctx->half_rows = g_numa.on ? n_tasks / 2 : n_tasks;
    ctx->nested_rows = 0;
    ctx->threads_seen = 0;
    ctx->slow = false;
    ctx->tasks = true;
    ctx->round++;
    memset(ctx->claims, 0, (size_t)n_tasks * sizeof(ctx->claims[0]));
    ds4_parallel_for_tasks(n_tasks, pool_claim_worker, ctx);
    for (uint64_t r = 0; r < n_tasks; r++) CHECK(ctx->claims[r] == 1);
    CHECK(ctx->nested_rows == 1000);
}

static void check_thread_pool(void) {
    const uint32_t threads[] = {1, 2, 7, DS4_MAX_THREADS};
    pool_test_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx);
    alarm(600);   /* a lost wake-up would otherwise hang here forever */
    CHECK(ds4_default_threads() >= 1 && ds4_default_threads() <= DS4_MAX_THREADS);
    ds4_numa_init();
    const bool numa = g_numa.on;
    /* The two-cursor split runs where the host has two NUMA nodes in the
     * affinity mask; the plain pool runs everywhere. */
    for (int mode = numa; mode >= 0; mode--) {
        g_numa.on = mode;
        for (size_t i = 0; i < sizeof(threads) / sizeof(*threads); i++) {
            g_requested_threads = threads[i];
            ds4_threads_init();
            CHECK(g_pool.n_threads == threads[i]);
            pool_dispatch(ctx, 64 * threads[i], 64 * threads[i], true);
            CHECK(ctx->threads_seen >= (threads[i] + 1) / 2);
            pool_dispatch(ctx, 6 * 600, 600, false);        /* six experts' rows */
            pool_dispatch(ctx, 7 * 333 + 100, 333, false);  /* odd halves, partial group */
            pool_dispatch(ctx, 3000, 1, false);             /* nothing for node 0 */
            pool_dispatch_tasks(ctx, 64 * threads[i]);      /* one task per claim */
            pool_dispatch_tasks(ctx, 1 + 7 * threads[i]);   /* odd halves */
            const uint32_t rounds = threads[i] == DS4_MAX_THREADS ? 2000 : 100000;
            for (uint32_t k = 0; k < rounds; k++) {
                pool_dispatch(ctx, 1 + k % 37, 1 + k % 37, false);
                if (k % 2000 == 0) usleep(3 * DS4_POOL_SPIN_US);
            }
            ds4_threads_shutdown();
            printf("CPU thread pool: %u threads%s PASS\n", threads[i], mode ? " (NUMA)" : "");
        }
    }
    g_numa.on = numa;
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
                        quant_buffer cmb = quant_alloc((size_t)nh * ((cap + 15) / 16) * sizeof(float));
                        ds4_cpu_decode_scratch s = {
                            .attn_score_cap = cap, .attn_score = (float *)scb.data,
                            .attn_qt = (float *)qtb.data, .attn_rows = (const float **)rwb.data,
                            .attn_cmax = (float *)cmb.data, .attn_inv = (float *)ib.data,
                        };
                        memset(out, 0xff, nh * d * sizeof(out[0]));
                        layer_attention_mixed_one_decode_scratch(out, &m, &l, q, raw, n_raw,
                                                                 comp, n_comp, allow, &s);
                        CHECK(!memcmp(out, expected, nh * d * sizeof(out[0])));
                        quant_free(cmb); quant_free(rwb); quant_free(scb);
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

/* Batched compressor projections must reproduce the per-token dot bit for bit,
 * for F16 weights (batched kernel) and for other types (per-token fallback). */
/* 9 x 5 takes only the per-token row-group path; 77 rows x 150 tokens takes
 * the token-lane tiles (partial row group, several tile blocks) and a tail. */
static void check_compressor_project_batch_shape(uint32_t width, uint32_t n_tok) {
    const uint64_t in_dim = 70;
    const size_t half_bytes = (size_t)in_dim * width * sizeof(uint16_t);
    const size_t f32_bytes = (size_t)in_dim * width * sizeof(float);
    quant_buffer wb = quant_alloc(2 * half_bytes + 2 * f32_bytes);
    uint16_t *wkv = (uint16_t *)wb.data, *wgate = wkv + in_dim * width;
    float *fkv = (float *)(wb.data + 2 * half_bytes), *fgate = fkv + in_dim * width;
    for (uint64_t i = 0; i < in_dim * width; i++) {
        wkv[i] = quant_half_finite(); wgate[i] = quant_half_finite();
        fkv[i] = quant_float(); fgate[i] = quant_float();
    }
    float *x = xmalloc((size_t)n_tok * in_dim * sizeof(x[0]));
    for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) x[i] = quant_float();
    float *kv_all = xmalloc((size_t)n_tok * width * sizeof(kv_all[0]));
    float *sc_all = xmalloc((size_t)n_tok * width * sizeof(sc_all[0]));
    float *expected = xmalloc((size_t)width * sizeof(expected[0]));

    ds4_model m = {.map = wb.data, .size = wb.size};
    ds4_tensor tkv = {.type = DS4_TENSOR_F16, .ndim = 2, .dim = {in_dim, width},
                      .abs_offset = 0, .elements = in_dim * width, .bytes = half_bytes};
    ds4_tensor tgate = tkv;
    tgate.abs_offset = half_bytes;
    compressor_project_batch(kv_all, sc_all, &m, &tkv, &tgate, x, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < width; o++) {
            quant_bits_equal("F16 compressor kv", dot_f16_row(wkv + o * in_dim, x + t * in_dim, in_dim),
                             kv_all[t * width + o], t * width + o);
            quant_bits_equal("F16 compressor gate", dot_f16_row(wgate + o * in_dim, x + t * in_dim, in_dim),
                             sc_all[t * width + o], t * width + o);
        }
    }

    ds4_tensor fkv_t = {.type = DS4_TENSOR_F32, .ndim = 2, .dim = {in_dim, width},
                        .abs_offset = 2 * half_bytes, .elements = in_dim * width, .bytes = f32_bytes};
    ds4_tensor fgate_t = fkv_t;
    fgate_t.abs_offset = 2 * half_bytes + f32_bytes;
    compressor_project_batch(kv_all, sc_all, &m, &fkv_t, &fgate_t, x, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        matvec_any(expected, &m, &fkv_t, x + t * in_dim);
        for (uint32_t o = 0; o < width; o++)
            quant_bits_equal("F32 compressor kv", expected[o], kv_all[t * width + o], t * width + o);
        matvec_any(expected, &m, &fgate_t, x + t * in_dim);
        for (uint32_t o = 0; o < width; o++)
            quant_bits_equal("F32 compressor gate", expected[o], sc_all[t * width + o], t * width + o);
    }
    free(expected); free(sc_all); free(kv_all); free(x);
    quant_free(wb);
}

/* The deferred prefill compressor (update, snapshot, parallel emit) against
 * the streaming update token by token: emitted rows, scores, state and the
 * emit count bitwise, from a mid-window start and with empty score rows. */
static void check_compressor_deferred_shape(uint32_t head_dim, uint32_t pos0, uint32_t n_tok) {
    const uint32_t ratio = 4, width = 2 * head_dim, rows = 2 * ratio;
    const size_t ape_bytes = (size_t)width * ratio * sizeof(uint16_t);
    const size_t norm_bytes = (size_t)head_dim * sizeof(float);
    quant_buffer wb = quant_alloc(ape_bytes + norm_bytes);
    uint16_t *ape = (uint16_t *)wb.data;
    float *norm = (float *)(wb.data + ape_bytes);
    for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) ape[i] = quant_half_finite();
    for (uint32_t i = 0; i < head_dim; i++) norm[i] = quant_float();
    ds4_model m = {.map = wb.data, .size = wb.size};
    ds4_tensor tape = {.type = DS4_TENSOR_F16, .ndim = 2, .dim = {width, ratio},
                       .abs_offset = 0, .elements = (uint64_t)width * ratio, .bytes = ape_bytes};
    ds4_tensor tnorm = {.type = DS4_TENSOR_F32, .ndim = 1, .dim = {head_dim},
                        .abs_offset = ape_bytes, .elements = head_dim, .bytes = norm_bytes};

    const uint64_t n = (uint64_t)n_tok * width, st = (uint64_t)rows * width;
    float *kv = xmalloc(n * sizeof(float)), *sc_a = xmalloc(n * sizeof(float)), *sc_b = xmalloc(n * sizeof(float));
    float *st_kv_a = xmalloc(st * sizeof(float)), *st_sc_a = xmalloc(st * sizeof(float));
    float *st_kv_b = xmalloc(st * sizeof(float)), *st_sc_b = xmalloc(st * sizeof(float));
    for (uint64_t i = 0; i < n; i++) { kv[i] = quant_float(); sc_a[i] = sc_b[i] = quant_float(); }
    for (uint64_t i = 0; i < st; i++) {
        st_kv_a[i] = st_kv_b[i] = quant_float();
        /* The second half's rows past pos0's window start are empty, as
         * compressor_finish_prefill_state_cpu leaves them. */
        const uint32_t r = (uint32_t)(i / width);
        st_sc_a[i] = st_sc_b[i] = r >= ratio + pos0 % ratio ? DS4_NEG_INF : quant_float();
    }

    const uint32_t cap = n_tok / ratio + 2;
    float *comp_a = xmalloc((size_t)cap * head_dim * sizeof(float)), *comp_b = xmalloc((size_t)cap * head_dim * sizeof(float));
    float *out = xmalloc(head_dim * sizeof(float));
    uint32_t n_a = 0;
    for (uint32_t tk = 0; tk < n_tok; tk++) {
        if (compressor_update_one(out, &m, &tape, &tnorm, kv + (uint64_t)tk * width, sc_a + (uint64_t)tk * width,
                                  st_kv_a, st_sc_a, head_dim, ratio, 3, pos0 + tk)) {
            kv_cache_push_comp(comp_a, &n_a, cap, head_dim, out);
        }
    }

    float *ape_rows = compressor_ape_rows(&m, &tape, width, ratio);
    const uint64_t snap = compressor_snapshot_floats(head_dim, ratio);
    float *snapshots = xmalloc((size_t)(n_tok / ratio + 1) * snap * sizeof(float));
    uint32_t *emit_pos = xmalloc((size_t)(n_tok / ratio + 1) * sizeof(uint32_t));
    uint32_t n_b = 0;
    for (uint32_t tk = 0; tk < n_tok; tk++) {
        if (compressor_update_deferred(ape_rows, kv + (uint64_t)tk * width, sc_b + (uint64_t)tk * width,
                                       st_kv_b, st_sc_b, head_dim, ratio, pos0 + tk, snapshots + n_b * snap)) {
            emit_pos[n_b++] = pos0 + tk;
        }
    }
    CHECK(n_b == n_a);
    float *emitted = xmalloc((size_t)(n_b + 1) * head_dim * sizeof(float));
    compressor_emit_deferred(emitted, snapshots, emit_pos, n_b, &m, &tnorm, head_dim, ratio, 3);
    uint32_t n_pushed = 0;
    for (uint32_t w = 0; w < n_b; w++) kv_cache_push_comp(comp_b, &n_pushed, cap, head_dim, emitted + (uint64_t)w * head_dim);

    for (uint64_t i = 0; i < (uint64_t)n_a * head_dim; i++) quant_bits_equal("deferred compressor row", comp_a[i], comp_b[i], (unsigned)i);
    for (uint64_t i = 0; i < n; i++) quant_bits_equal("deferred compressor score", sc_a[i], sc_b[i], (unsigned)i);
    for (uint64_t i = 0; i < st; i++) {
        quant_bits_equal("deferred compressor state kv", st_kv_a[i], st_kv_b[i], (unsigned)i);
        quant_bits_equal("deferred compressor state score", st_sc_a[i], st_sc_b[i], (unsigned)i);
    }
    free(emitted); free(emit_pos); free(snapshots); free(ape_rows); free(out); free(comp_b); free(comp_a);
    free(st_sc_b); free(st_kv_b); free(st_sc_a); free(st_kv_a); free(sc_b); free(sc_a); free(kv);
    quant_free(wb);
}

static void check_compressor_deferred(void) {
    check_compressor_deferred_shape(512, 0, 23);
    check_compressor_deferred_shape(512, 2, 17);
    check_compressor_deferred_shape(128, 7, 30);
    check_compressor_deferred_shape(128, 1, 3);
    printf("CPU deferred prefill compressor bitwise with the streaming update PASS\n");
}

static void check_compressor_project_batch(void) {
    check_compressor_project_batch_shape(9, 5);
    check_compressor_project_batch_shape(77, 150);
}

static uint16_t ffn_graph_half(void) {
    return f32_to_f16((float)(1 + quant_random() % 200) * 0.00011f);
}

/* The staged decode FFN (HC pre-step, router, shared and routed experts and
 * the HC post in one dispatch) must reproduce the plain sublayer bit for bit:
 * norm, routed and shared outputs, post/comb and the new HC state, for a
 * hash-routed and a top-k layer, at several thread counts, with and without
 * the two-cursor NUMA split.  Eight experts are backed by memory; the hash
 * table and the selection bias keep the router inside them. */
static void check_ffn_graph(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH;
    const uint64_t E = DS4_N_EMBD, F = DS4_N_FF_EXP, NE = DS4_N_EXPERT, U = DS4_N_EXPERT_USED;
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * E, n_mix = (uint64_t)n_hc * (2 + n_hc);
    const uint64_t n_exp = 8, n_tok_ids = 16;
    const uint64_t router_bytes = NE * E * 2, bias_bytes = NE * 4, hash_bytes = U * n_tok_ids * 4;
    const uint64_t iq2_row = E / QK_K * 66, q2k_row = F / QK_K * 84;
    const uint64_t gate_bytes = n_exp * F * iq2_row, down_bytes = n_exp * E * q2k_row;
    const uint64_t sh_gate_bytes = F * (E / 32) * 34, sh_down_bytes = E * (F / 32) * 34;
    const uint64_t hcfn_bytes = n_mix * hc_dim * 2;
    enum { N_T = 13 };
    uint64_t off[N_T];
    const uint64_t sizes[N_T] = {router_bytes, bias_bytes, hash_bytes, gate_bytes, gate_bytes,
                                 down_bytes, sh_gate_bytes, sh_gate_bytes, sh_down_bytes,
                                 hcfn_bytes, 3 * 4, n_mix * 4, E * 4};
    uint64_t total = 0;
    for (int i = 0; i < N_T; i++) { off[i] = total; total += (sizes[i] + 63) / 64 * 64; }
    quant_buffer wb = quant_alloc(total);
    uint8_t *w = wb.data;
    for (uint64_t i = 0; i < total; i++) w[i] = (uint8_t)quant_random();
    uint16_t *router = (uint16_t *)(w + off[0]);
    for (uint64_t i = 0; i < NE * E; i++) router[i] = f32_to_f16(quant_float() * 0.05f);
    float *bias = (float *)(w + off[1]);
    for (uint64_t i = 0; i < NE; i++) bias[i] = i < n_exp ? 100.0f : 0.0f;
    int32_t *hash = (int32_t *)(w + off[2]);
    for (uint64_t i = 0; i < U * n_tok_ids; i++) hash[i] = (int32_t)(quant_random() % n_exp);
    for (int t = 3; t <= 4; t++) {                       /* IQ2_XXS: d first */
        for (uint64_t b = 0; b < sizes[t] / 66; b++) {
            const uint16_t d = ffn_graph_half();
            memcpy(w + off[t] + b * 66, &d, 2);
        }
    }
    for (uint64_t b = 0; b < down_bytes / 84; b++) {    /* Q2_K: d, dmin last */
        const uint16_t d = ffn_graph_half(), dmin = ffn_graph_half();
        memcpy(w + off[5] + b * 84 + 80, &d, 2);
        memcpy(w + off[5] + b * 84 + 82, &dmin, 2);
    }
    for (int t = 6; t <= 8; t++) {                       /* Q8_0: d first */
        for (uint64_t b = 0; b < sizes[t] / 34; b++) {
            const uint16_t d = ffn_graph_half();
            memcpy(w + off[t] + b * 34, &d, 2);
        }
    }
    uint16_t *hcfn = (uint16_t *)(w + off[9]);
    for (uint64_t i = 0; i < n_mix * hc_dim; i++) hcfn[i] = f32_to_f16(quant_float() * 0.01f);
    float *hcscale = (float *)(w + off[10]), *hcbase = (float *)(w + off[11]);
    float *normw = (float *)(w + off[12]);
    for (int i = 0; i < 3; i++) hcscale[i] = 0.5f + 0.25f * (float)i;
    for (uint64_t i = 0; i < n_mix; i++) hcbase[i] = quant_float();
    for (uint64_t i = 0; i < E; i++) normw[i] = quant_float();

    ds4_model m = {.map = w, .size = total};
    ds4_tensor t_router = {.type = DS4_TENSOR_F16, .ndim = 2, .dim = {E, NE},
                           .abs_offset = off[0], .elements = E * NE, .bytes = router_bytes};
    ds4_tensor t_bias = {.type = DS4_TENSOR_F32, .ndim = 1, .dim = {NE},
                         .abs_offset = off[1], .elements = NE, .bytes = bias_bytes};
    ds4_tensor t_hash = {.type = DS4_TENSOR_I32, .ndim = 2, .dim = {U, n_tok_ids},
                         .abs_offset = off[2], .elements = U * n_tok_ids, .bytes = hash_bytes};
    /* dim[2] is the model's expert count; only the first n_exp are mapped. */
    ds4_tensor t_gate = {.type = DS4_TENSOR_IQ2_XXS, .ndim = 3, .dim = {E, F, NE},
                         .abs_offset = off[3], .elements = E * F * NE, .bytes = gate_bytes};
    ds4_tensor t_up = t_gate;
    t_up.abs_offset = off[4];
    ds4_tensor t_down = {.type = DS4_TENSOR_Q2_K, .ndim = 3, .dim = {F, E, NE},
                         .abs_offset = off[5], .elements = F * E * NE, .bytes = down_bytes};
    ds4_tensor t_sh_gate = {.type = DS4_TENSOR_Q8_0, .ndim = 2, .dim = {E, F},
                            .abs_offset = off[6], .elements = E * F, .bytes = sh_gate_bytes};
    ds4_tensor t_sh_up = t_sh_gate;
    t_sh_up.abs_offset = off[7];
    ds4_tensor t_sh_down = {.type = DS4_TENSOR_Q8_0, .ndim = 2, .dim = {F, E},
                            .abs_offset = off[8], .elements = F * E, .bytes = sh_down_bytes};
    ds4_tensor t_hcfn = {.type = DS4_TENSOR_F16, .ndim = 2, .dim = {hc_dim, n_mix},
                         .abs_offset = off[9], .elements = hc_dim * n_mix, .bytes = hcfn_bytes};
    ds4_tensor t_hcscale = {.type = DS4_TENSOR_F32, .ndim = 1, .dim = {3},
                            .abs_offset = off[10], .elements = 3, .bytes = 12};
    ds4_tensor t_hcbase = {.type = DS4_TENSOR_F32, .ndim = 1, .dim = {n_mix},
                           .abs_offset = off[11], .elements = n_mix, .bytes = n_mix * 4};
    ds4_tensor t_norm = {.type = DS4_TENSOR_F32, .ndim = 1, .dim = {E},
                         .abs_offset = off[12], .elements = E, .bytes = E * 4};
    ds4_layer_weights layer;
    memset(&layer, 0, sizeof(layer));
    layer.ffn_gate_inp = &t_router;
    layer.ffn_gate_exps = &t_gate;
    layer.ffn_up_exps = &t_up;
    layer.ffn_down_exps = &t_down;
    layer.ffn_gate_shexp = &t_sh_gate;
    layer.ffn_up_shexp = &t_sh_up;
    layer.ffn_down_shexp = &t_sh_down;
    layer.hc_ffn_fn = &t_hcfn;
    layer.hc_ffn_scale = &t_hcscale;
    layer.hc_ffn_base = &t_hcbase;
    layer.ffn_norm = &t_norm;
    CHECK(ffn_graph_supported(&layer));

    ds4_cpu_decode_scratch scratch;
    cpu_decode_scratch_init(&scratch, 64);
    float *res = xmalloc(hc_dim * sizeof(float)), *flat = xmalloc(hc_dim * sizeof(float));
    float *cur0 = xmalloc(E * sizeof(float)), *norm0 = xmalloc(E * sizeof(float));
    float *moe0 = xmalloc(E * sizeof(float)), *sh0 = xmalloc(E * sizeof(float));
    float *sum = xmalloc(E * sizeof(float));
    float *hc0 = xmalloc(hc_dim * sizeof(float)), *hc1 = xmalloc(hc_dim * sizeof(float));
    float post0[4], comb0[16], post1[4], comb1[16];

    ds4_numa_init();
    const bool numa = g_numa.on;
    const uint32_t threads[] = {1, 3, 16, 40};
    for (int mode = numa; mode >= 0; mode--) {
        g_numa.on = mode;
        for (size_t ti = 0; ti < sizeof(threads) / sizeof(*threads); ti++) {
            g_requested_threads = threads[ti];
            ds4_threads_init();
            for (int iter = 0; iter < 6; iter++) {
                const bool hashed = iter & 1;
                layer.ffn_gate_tid2eid = hashed ? &t_hash : NULL;
                layer.ffn_exp_probs_b = hashed ? NULL : &t_bias;
                const int token = (int)(quant_random() % n_tok_ids);
                for (uint64_t i = 0; i < hc_dim; i++) res[i] = quant_float() * (iter < 2 ? 1.0f : 30.0f);

                hc_pre_from_state_one_scratch(&m, &t_hcfn, &t_hcscale, &t_hcbase, res, cur0,
                                              post0, comb0, flat, false);
                rms_norm_weight(norm0, cur0, normw, E, DS4_RMS_EPS);
                layer_routed_moe_one_prealloc(moe0, &m, &layer, norm0, 0, token, DS4_SWIGLU_CLAMP_EXP,
                                              scratch.routed_mid_all, scratch.routed_xq,
                                              scratch.routed_midq, scratch.routed_q8_xq,
                                              scratch.routed_q8_xscale, scratch.routed_q8_midq,
                                              scratch.routed_q8_midscale);
                layer_shared_ffn_one_decode_scratch(sh0, &m, &layer, norm0, &scratch);
                for (uint64_t i = 0; i < E; i++) sum[i] = moe0[i] + sh0[i];
                hc_post_one(hc0, sum, res, post0, comb0, (uint32_t)E, n_hc);

                memset(scratch.ffn_moe, 0xff, E * sizeof(float));
                memset(scratch.ffn_shared, 0xff, E * sizeof(float));
                memset(scratch.ffn_norm, 0xff, E * sizeof(float));
                memset(hc1, 0xff, hc_dim * sizeof(float));
                /* Two iterations in three also fold the HC post into the dispatch. */
                const bool fold = iter % 3 != 0;
                layer_ffn_graph(fold ? hc1 : NULL, &m, &layer, res, token, &scratch, post1, comb1);
                uint64_t live = 0;
                for (uint64_t i = 0; i < E; i++) {
                    quant_bits_equal("FFN graph norm", norm0[i], scratch.ffn_norm[i], (unsigned)i);
                    quant_bits_equal("FFN graph routed", moe0[i], scratch.ffn_moe[i], (unsigned)i);
                    quant_bits_equal("FFN graph shared", sh0[i], scratch.ffn_shared[i], (unsigned)i);
                    live += isfinite(moe0[i]) && isfinite(sh0[i]) && moe0[i] != 0.0f && sh0[i] != 0.0f;
                }
                CHECK(live > E / 2);   /* the weights exercise real values */
                for (uint32_t i = 0; i < n_hc; i++) quant_bits_equal("FFN graph post", post0[i], post1[i], i);
                for (uint32_t i = 0; i < n_hc * n_hc; i++) quant_bits_equal("FFN graph comb", comb0[i], comb1[i], i);
                if (fold) {
                    for (uint64_t i = 0; i < hc_dim; i++) {
                        quant_bits_equal("FFN graph HC post", hc0[i], hc1[i], (unsigned)i);
                    }
                }
            }
            ds4_threads_shutdown();
            printf("CPU staged decode FFN: %u threads%s PASS\n", threads[ti], mode ? " (NUMA)" : "");
        }
    }
    g_numa.on = numa;
    g_requested_threads = 0;
    free(hc1); free(hc0); free(sum); free(sh0); free(moe0); free(norm0); free(cur0);
    free(flat); free(res);
    cpu_decode_scratch_free(&scratch);
    quant_free(wb);
    g_ds4_shape = saved;
}

/* A layer cache for the staged attention oracle, random rows and state. */
typedef struct {
    ds4_layer_cache c;
    uint32_t state_rows, width, index_width;
} attn_test_cache;

static void attn_test_cache_alloc(attn_test_cache *tc, uint32_t ratio, uint32_t cap_raw, uint32_t comp_cap) {
    ds4_layer_cache *c = &tc->c;
    memset(tc, 0, sizeof(*tc));
    c->cap_raw = cap_raw;
    c->raw_base = xmalloc((size_t)2 * cap_raw * DS4_N_HEAD_DIM * sizeof(float));
    c->raw_kv = c->raw_base;
    c->compress_ratio = ratio;
    if (ratio == 0) return;
    const uint32_t coff = ratio == 4 ? 2 : 1;
    tc->width = coff * DS4_N_HEAD_DIM;
    tc->state_rows = coff * ratio;
    c->comp_cap = comp_cap;
    c->attn_comp_kv = xmalloc((size_t)comp_cap * DS4_N_HEAD_DIM * sizeof(float));
    c->attn_state_kv = xmalloc((size_t)tc->width * tc->state_rows * sizeof(float));
    c->attn_state_score = xmalloc((size_t)tc->width * tc->state_rows * sizeof(float));
    if (ratio != 4) return;
    tc->index_width = 2 * DS4_N_INDEXER_HEAD_DIM;
    c->index_comp_kv = xmalloc((size_t)comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    c->index_state_kv = xmalloc((size_t)tc->index_width * tc->state_rows * sizeof(float));
    c->index_state_score = xmalloc((size_t)tc->index_width * tc->state_rows * sizeof(float));
}

static void attn_test_cache_free(attn_test_cache *tc) {
    ds4_layer_cache *c = &tc->c;
    free(c->index_state_score); free(c->index_state_kv); free(c->index_comp_kv);
    free(c->attn_state_score); free(c->attn_state_kv); free(c->attn_comp_kv); free(c->raw_base);
}

static void attn_test_fill(float *x, uint64_t n, float scale) {
    for (uint64_t i = 0; i < n; i++) x[i] = quant_float() * scale;
}

static void attn_test_cache_fill(attn_test_cache *tc, uint32_t n_raw, uint32_t n_comp) {
    ds4_layer_cache *c = &tc->c;
    attn_test_fill(c->raw_kv, (uint64_t)c->cap_raw * DS4_N_HEAD_DIM, 1.0f);
    c->n_raw = n_raw;
    if (c->compress_ratio == 0) return;
    attn_test_fill(c->attn_comp_kv, (uint64_t)c->comp_cap * DS4_N_HEAD_DIM, 1.0f);
    attn_test_fill(c->attn_state_kv, (uint64_t)tc->width * tc->state_rows, 1.0f);
    attn_test_fill(c->attn_state_score, (uint64_t)tc->width * tc->state_rows, 1.0f);
    c->n_comp = n_comp;
    if (c->compress_ratio != 4) return;
    attn_test_fill(c->index_comp_kv, (uint64_t)c->comp_cap * DS4_N_INDEXER_HEAD_DIM, 1.0f);
    attn_test_fill(c->index_state_kv, (uint64_t)tc->index_width * tc->state_rows, 1.0f);
    attn_test_fill(c->index_state_score, (uint64_t)tc->index_width * tc->state_rows, 1.0f);
    c->n_index_comp = n_comp;
}

static void attn_test_copy(float *dst, const float *src, uint64_t n) {
    if (n) memcpy(dst, src, n * sizeof(float));
}

static void attn_test_cache_copy(attn_test_cache *dst, const attn_test_cache *src) {
    const ds4_layer_cache *a = &src->c;
    ds4_layer_cache *b = &dst->c;
    attn_test_copy(b->raw_kv, a->raw_kv, (uint64_t)a->cap_raw * DS4_N_HEAD_DIM);
    b->n_raw = a->n_raw;
    b->n_comp = a->n_comp;
    b->n_index_comp = a->n_index_comp;
    if (a->compress_ratio == 0) return;
    attn_test_copy(b->attn_comp_kv, a->attn_comp_kv, (uint64_t)a->comp_cap * DS4_N_HEAD_DIM);
    attn_test_copy(b->attn_state_kv, a->attn_state_kv, (uint64_t)src->width * src->state_rows);
    attn_test_copy(b->attn_state_score, a->attn_state_score, (uint64_t)src->width * src->state_rows);
    if (a->compress_ratio != 4) return;
    attn_test_copy(b->index_comp_kv, a->index_comp_kv, (uint64_t)a->comp_cap * DS4_N_INDEXER_HEAD_DIM);
    attn_test_copy(b->index_state_kv, a->index_state_kv, (uint64_t)src->index_width * src->state_rows);
    attn_test_copy(b->index_state_score, a->index_state_score, (uint64_t)src->index_width * src->state_rows);
}

static void attn_test_equal(const char *name, const float *expected, const float *actual, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) quant_bits_equal(name, expected[i], actual[i], (unsigned)i);
}

static void attn_test_cache_equal(const attn_test_cache *ea, const attn_test_cache *ga) {
    const ds4_layer_cache *e = &ea->c, *g = &ga->c;
    CHECK(e->n_raw == g->n_raw && e->n_comp == g->n_comp && e->n_index_comp == g->n_index_comp);
    attn_test_equal("attention graph raw KV", e->raw_kv, g->raw_kv, (uint64_t)e->cap_raw * DS4_N_HEAD_DIM);
    if (e->compress_ratio == 0) return;
    attn_test_equal("attention graph compressed KV", e->attn_comp_kv, g->attn_comp_kv,
                    (uint64_t)e->n_comp * DS4_N_HEAD_DIM);
    attn_test_equal("attention graph compressor state", e->attn_state_kv, g->attn_state_kv,
                    (uint64_t)ea->width * ea->state_rows);
    attn_test_equal("attention graph compressor scores", e->attn_state_score, g->attn_state_score,
                    (uint64_t)ea->width * ea->state_rows);
    if (e->compress_ratio != 4) return;
    attn_test_equal("attention graph indexer KV", e->index_comp_kv, g->index_comp_kv,
                    (uint64_t)e->n_index_comp * DS4_N_INDEXER_HEAD_DIM);
    attn_test_equal("attention graph indexer state", e->index_state_kv, g->index_state_kv,
                    (uint64_t)ea->index_width * ea->state_rows);
    attn_test_equal("attention graph indexer scores", e->index_state_score, g->index_state_score,
                    (uint64_t)ea->index_width * ea->state_rows);
}

/* The raw window slides through twice its rows and moves back once it
 * reaches the end: after every push it must hold the last cap_raw rows
 * pushed, oldest first, as the memmove it replaces left them. */
static void check_raw_window(void) {
    ds4_kv_cache kc;
    kv_cache_init(&kc, 64, 4);
    ds4_layer_cache *c = &kc.layer[0];
    const uint32_t cap = c->cap_raw, n_push = 5 * cap + 3;
    CHECK(cap == 4);
    float *rows = xmalloc((size_t)n_push * DS4_N_HEAD_DIM * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)n_push * DS4_N_HEAD_DIM; i++) {
        rows[i] = f16_to_f32(f32_to_f16(quant_float()));
    }
    for (uint32_t t = 0; t < n_push; t++) {
        kv_cache_push_raw(c, rows + (uint64_t)t * DS4_N_HEAD_DIM);
        const uint32_t n = t + 1 < cap ? t + 1 : cap;
        CHECK(c->n_raw == n);
        CHECK(c->raw_kv >= c->raw_base && c->raw_kv + (uint64_t)cap * DS4_N_HEAD_DIM <=
                                          c->raw_base + 2ull * cap * DS4_N_HEAD_DIM);
        attn_test_equal("raw window", rows + (uint64_t)(t + 1 - n) * DS4_N_HEAD_DIM, c->raw_kv,
                        (uint64_t)n * DS4_N_HEAD_DIM);
    }
    free(rows);
    kv_cache_free(&kc);
}

/* The staged decode attention sublayer must reproduce the plain one bit for
 * bit: q, kv, the heads, the output, the new HC state and every cache row
 * and state the token pushes, on a dense layer, a ratio-128 layer and a
 * ratio-4 layer whose indexer selects (its top-k shrunk so that a handful
 * of compressed rows exercise it), with the raw window filling and full,
 * with and without a compression step, with the HC post folded in and with
 * steering keeping it serial, at several thread counts, with and without
 * the two-cursor NUMA split. */
static void check_attn_graph(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_shape.n_indexer_top_k = 6;
    const uint64_t E = DS4_N_EMBD, H = DS4_N_HEAD, D = DS4_N_HEAD_DIM, LQ = DS4_N_LORA_Q;
    const uint64_t OG = DS4_N_OUT_GROUP, LO = DS4_N_LORA_O;
    const uint64_t IH = DS4_N_INDEXER_HEAD, ID = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * E, n_mix = (uint64_t)n_hc * (2 + n_hc);
    const uint64_t group_dim = H / OG * D, low_dim = OG * LO, q_dim = H * D;
    enum { T_HCFN, T_HCSCALE, T_HCBASE, T_NORM, T_QA, T_QANORM, T_QB, T_KV, T_KVNORM, T_SINKS,
           T_OA, T_OB, T_CKV4, T_CGATE4, T_CAPE4, T_CKV128, T_CGATE128, T_CAPE128, T_CNORM,
           T_IKV, T_IGATE, T_IAPE, T_INORM, T_IPROJ, T_IQB, N_T };
    const struct { uint32_t type; uint64_t d0, d1; } spec[N_T] = {
        [T_HCFN] = {DS4_TENSOR_F16, hc_dim, n_mix},
        [T_HCSCALE] = {DS4_TENSOR_F32, 3, 0},
        [T_HCBASE] = {DS4_TENSOR_F32, n_mix, 0},
        [T_NORM] = {DS4_TENSOR_F32, E, 0},
        [T_QA] = {DS4_TENSOR_Q8_0, E, LQ},
        [T_QANORM] = {DS4_TENSOR_F32, LQ, 0},
        [T_QB] = {DS4_TENSOR_Q8_0, LQ, q_dim},
        [T_KV] = {DS4_TENSOR_Q8_0, E, D},
        [T_KVNORM] = {DS4_TENSOR_F32, D, 0},
        [T_SINKS] = {DS4_TENSOR_F32, H, 0},
        [T_OA] = {DS4_TENSOR_Q8_0, group_dim, low_dim},
        [T_OB] = {DS4_TENSOR_Q8_0, low_dim, E},
        [T_CKV4] = {DS4_TENSOR_F16, E, 2 * D},
        [T_CGATE4] = {DS4_TENSOR_F16, E, 2 * D},
        [T_CAPE4] = {DS4_TENSOR_F16, 2 * D, 4},
        [T_CKV128] = {DS4_TENSOR_F16, E, D},
        [T_CGATE128] = {DS4_TENSOR_F16, E, D},
        [T_CAPE128] = {DS4_TENSOR_F16, D, 128},
        [T_CNORM] = {DS4_TENSOR_F32, D, 0},
        [T_IKV] = {DS4_TENSOR_F16, E, 2 * ID},
        [T_IGATE] = {DS4_TENSOR_F16, E, 2 * ID},
        [T_IAPE] = {DS4_TENSOR_F16, 2 * ID, 4},
        [T_INORM] = {DS4_TENSOR_F32, ID, 0},
        [T_IPROJ] = {DS4_TENSOR_F16, E, IH},
        [T_IQB] = {DS4_TENSOR_F16, LQ, IH * ID},
    };
    ds4_tensor t[N_T];
    uint64_t off[N_T], total = 0;
    for (int i = 0; i < N_T; i++) {
        const uint64_t elems = spec[i].d0 * (spec[i].d1 ? spec[i].d1 : 1);
        const uint64_t bytes = spec[i].type == DS4_TENSOR_Q8_0 ? spec[i].d1 * (spec[i].d0 / 32) * 34
                             : elems * (spec[i].type == DS4_TENSOR_F16 ? 2 : 4);
        off[i] = total;
        total += (bytes + 63) / 64 * 64;
        memset(&t[i], 0, sizeof(t[i]));
        t[i].type = spec[i].type;
        t[i].ndim = spec[i].d1 ? 2 : 1;
        t[i].dim[0] = spec[i].d0;
        t[i].dim[1] = spec[i].d1;
        t[i].abs_offset = off[i];
        t[i].elements = elems;
        t[i].bytes = bytes;
    }
    quant_buffer wb = quant_alloc(total);
    uint8_t *w = wb.data;
    for (int i = 0; i < N_T; i++) {
        if (spec[i].type == DS4_TENSOR_F16) {
            uint16_t *p = (uint16_t *)(w + off[i]);
            const float scale = i == T_HCFN ? 0.01f : 0.05f;
            for (uint64_t k = 0; k < t[i].elements; k++) p[k] = f32_to_f16(quant_float() * scale);
        } else if (spec[i].type == DS4_TENSOR_Q8_0) {
            for (uint64_t b = 0; b < t[i].bytes / 34; b++) {
                uint8_t *blk = w + off[i] + b * 34;
                const uint16_t d = ffn_graph_half();
                memcpy(blk, &d, 2);
                for (int k = 0; k < 32; k++) blk[2 + k] = (uint8_t)quant_random();
            }
        } else {
            attn_test_fill((float *)(w + off[i]), t[i].elements, 1.0f);
        }
    }
    float *hcscale = (float *)(w + off[T_HCSCALE]);
    for (int i = 0; i < 3; i++) hcscale[i] = 0.5f + 0.25f * (float)i;
    ds4_model m = {.map = w, .size = total};

    const uint32_t ratios[3] = {0, 4, 128}, ils[3] = {0, 2, 3};
    ds4_layer_weights L[3];
    memset(L, 0, sizeof(L));
    for (int r = 0; r < 3; r++) {
        ds4_layer_weights *l = &L[r];
        l->hc_attn_fn = &t[T_HCFN]; l->hc_attn_scale = &t[T_HCSCALE]; l->hc_attn_base = &t[T_HCBASE];
        l->attn_norm = &t[T_NORM]; l->attn_sinks = &t[T_SINKS];
        l->attn_q_a = &t[T_QA]; l->attn_q_a_norm = &t[T_QANORM]; l->attn_q_b = &t[T_QB];
        l->attn_kv = &t[T_KV]; l->attn_kv_a_norm = &t[T_KVNORM];
        l->attn_output_a = &t[T_OA]; l->attn_output_b = &t[T_OB];
        if (ratios[r] == 4) {
            l->attn_compressor_kv = &t[T_CKV4]; l->attn_compressor_gate = &t[T_CGATE4];
            l->attn_compressor_ape = &t[T_CAPE4]; l->attn_compressor_norm = &t[T_CNORM];
            l->indexer_compressor_kv = &t[T_IKV]; l->indexer_compressor_gate = &t[T_IGATE];
            l->indexer_compressor_ape = &t[T_IAPE]; l->indexer_compressor_norm = &t[T_INORM];
            l->indexer_proj = &t[T_IPROJ]; l->indexer_attn_q_b = &t[T_IQB];
        } else if (ratios[r] == 128) {
            l->attn_compressor_kv = &t[T_CKV128]; l->attn_compressor_gate = &t[T_CGATE128];
            l->attn_compressor_ape = &t[T_CAPE128]; l->attn_compressor_norm = &t[T_CNORM];
        }
        CHECK(attn_graph_supported(l, ratios[r]));
        g_ds4_compress_ratios[ils[r]] = ratios[r];
    }

    const uint32_t cap_raw = 16, comp_cap = 64;
    attn_test_cache ca[3], cb[3];
    for (int r = 0; r < 3; r++) {
        attn_test_cache_alloc(&ca[r], ratios[r], cap_raw, comp_cap);
        attn_test_cache_alloc(&cb[r], ratios[r], cap_raw, comp_cap);
    }
    ds4_cpu_decode_scratch scratch;
    cpu_decode_scratch_init(&scratch, 256);
    float *res = xmalloc(hc_dim * sizeof(float)), *dirs = xmalloc(4 * E * sizeof(float));
    float *hc0 = xmalloc(hc_dim * sizeof(float)), *q0 = xmalloc(q_dim * sizeof(float));
    float *kv0 = xmalloc(D * sizeof(float)), *heads0 = xmalloc(q_dim * sizeof(float));
    float *out0 = xmalloc(E * sizeof(float));
    float *iq0 = xmalloc(IH * ID * sizeof(float)), *iqt0 = xmalloc(IH * ID * sizeof(float));
    float *isc0 = xmalloc(comp_cap * sizeof(float));
    attn_test_fill(dirs, 4 * E, 0.01f);

    /* top_k 0 keeps the indexer's 6; the attention scores every row while
     * the indexer selects when at most ATTN_SCORE_ALL_DROP_MAX rows drop
     * out, so the cases with more compressed rows have chunks of them with
     * no, some and all of their rows listed, and 6 of 61 waits for the
     * selection. */
    const struct { int r; uint32_t pos, n_raw, n_comp; bool steer; uint32_t top_k; } cases[] = {
        {0, 5, 5, 0, false, 0},       /* dense layer, window filling */
        {1, 131, 16, 20, false, 0},   /* ratio 4: full window, compression, 21 > top-k rows: indexer */
        {1, 130, 16, 20, false, 0},   /* ratio 4: the indexer selects without a compression step */
        {1, 130, 9, 3, false, 0},     /* ratio 4: no compression, every compressed row visible */
        {2, 255, 16, 1, false, 0},    /* ratio 128: compression */
        {1, 131, 16, 20, true, 0},    /* steering keeps the HC post serial */
        {1, 7, 8, 1, false, 0},       /* ratio 4: compression into the second row */
        {1, 251, 16, 60, false, 0},   /* ratio 4: 6 of 61 compressed rows */
        {1, 250, 16, 60, false, 57},  /* ratio 4: 57 of 60, whole chunks listed */
        {1, 251, 16, 60, false, 33},  /* ratio 4: 33 of 61 */
    };
    const size_t n_cases = sizeof(cases) / sizeof(*cases);

    ds4_numa_init();
    const bool numa = g_numa.on;
    const uint32_t threads[] = {1, 3, 16, 40};
    for (int mode = numa; mode >= 0; mode--) {
        g_numa.on = mode;
        for (size_t ti = 0; ti < sizeof(threads) / sizeof(*threads); ti++) {
            g_requested_threads = threads[ti];
            ds4_threads_init();
            for (size_t ci = 0; ci < n_cases; ci++) {
                const int r = cases[ci].r;
                const float *steer = cases[ci].steer ? dirs : NULL;
                g_ds4_shape.n_indexer_top_k = cases[ci].top_k ? cases[ci].top_k : 6;
                attn_test_cache_fill(&ca[r], cases[ci].n_raw, cases[ci].n_comp);
                attn_test_cache_copy(&cb[r], &ca[r]);
                attn_test_fill(res, hc_dim, ci & 1 ? 20.0f : 1.0f);

                layer_attention_plain_one(&m, &L[r], &ca[r].c, res, ils[r], cases[ci].pos,
                                          steer, 0.1f, &scratch, NULL);
                memcpy(hc0, scratch.after_attn_hc, hc_dim * sizeof(float));
                memcpy(q0, scratch.q, q_dim * sizeof(float));
                memcpy(kv0, scratch.kv, D * sizeof(float));
                memcpy(heads0, scratch.heads, q_dim * sizeof(float));
                memcpy(out0, scratch.attn_out, E * sizeof(float));
                /* When the indexer selects, its query (after the FP4 round
                 * trip), the transposed query and the scores must match too. */
                const uint32_t n_index = ca[r].c.n_index_comp;
                const bool indexed = ratios[r] == 4 && n_index > DS4_N_INDEXER_TOP_K;
                if (indexed) {
                    memcpy(iq0, scratch.index_q, IH * ID * sizeof(float));
                    memcpy(iqt0, scratch.index_qt, IH * ID * sizeof(float));
                    memcpy(isc0, scratch.index_scores, n_index * sizeof(float));
                    memset(scratch.index_q, 0xff, IH * ID * sizeof(float));
                    memset(scratch.index_qt, 0xff, IH * ID * sizeof(float));
                    memset(scratch.index_scores, 0xff, n_index * sizeof(float));
                }
                memset(scratch.after_attn_hc, 0xff, hc_dim * sizeof(float));
                memset(scratch.q, 0xff, q_dim * sizeof(float));
                memset(scratch.kv, 0xff, D * sizeof(float));
                memset(scratch.heads, 0xff, q_dim * sizeof(float));
                memset(scratch.attn_out, 0xff, E * sizeof(float));

                /* Every other case takes the profile marks (the decode
                 * profile path): the ready times must come in stage order. */
                double marks[5] = {0, 0, 0, 0, 0};
                layer_attention_graph(&m, &L[r], &cb[r].c, res, ils[r], cases[ci].pos,
                                      steer, 0.1f, &scratch, ci & 1 ? marks : NULL);
                if (ci & 1) {
                    CHECK(marks[0] > 0.0 && marks[0] <= marks[1] && marks[1] <= marks[2] &&
                          marks[2] <= marks[3] && marks[3] <= marks[4]);
                }
                attn_test_equal("attention graph q", q0, scratch.q, q_dim);
                attn_test_equal("attention graph kv", kv0, scratch.kv, D);
                attn_test_equal("attention graph heads", heads0, scratch.heads, q_dim);
                attn_test_equal("attention graph output", out0, scratch.attn_out, E);
                attn_test_equal("attention graph HC post", hc0, scratch.after_attn_hc, hc_dim);
                if (indexed) {
                    attn_test_equal("attention graph indexer query", iq0, scratch.index_q, IH * ID);
                    attn_test_equal("attention graph indexer query transpose", iqt0, scratch.index_qt, IH * ID);
                    attn_test_equal("attention graph indexer scores", isc0, scratch.index_scores, n_index);
                }
                attn_test_cache_equal(&ca[r], &cb[r]);
                uint64_t live = 0;
                for (uint64_t i = 0; i < E; i++) live += isfinite(out0[i]) && out0[i] != 0.0f;
                CHECK(live > E / 2);   /* the weights exercise real values */
            }
            ds4_threads_shutdown();
            printf("CPU staged decode attention: %u threads%s PASS\n", threads[ti], mode ? " (NUMA)" : "");
        }
    }
    g_numa.on = numa;
    g_requested_threads = 0;
    for (int r = 0; r < 3; r++) {
        g_ds4_compress_ratios[ils[r]] = 0;
        attn_test_cache_free(&ca[r]);
        attn_test_cache_free(&cb[r]);
    }
    free(isc0); free(iqt0); free(iq0);
    free(out0); free(heads0); free(kv0); free(q0); free(hc0); free(dirs); free(res);
    cpu_decode_scratch_free(&scratch);
    quant_free(wb);
    g_ds4_shape = saved;
}

/* Prefix prefill attention must match the per-token attention each token
 * would run against its raw window and the compressed rows pushed so far,
 * with and without an indexer mask, for a fresh prompt and for one that
 * continues a cache (pos0 > 0, the window starting in the cached rows).  The
 * model shape takes the sixteen-lane kernels; 3 heads of 40 dims take the
 * partial-group and tail paths. */
static void check_prefix_attention_shape(uint32_t n_heads, uint32_t n_dims) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_shape.n_head = n_heads;
    g_ds4_shape.n_head_dim = n_dims;
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
    float *kv = xmalloc((size_t)(raw_cap + n_tok) * head_dim * sizeof(kv[0]));
    float *comp = xmalloc((size_t)n_comp_rows * head_dim * sizeof(comp[0]));
    float *batch = xmalloc((size_t)n_tok * q_dim * sizeof(batch[0]));
    float *one = xmalloc(q_dim * sizeof(one[0]));
    uint32_t *comp_counts = xmalloc((size_t)n_tok * sizeof(comp_counts[0]));
    uint8_t *mask = xmalloc(n_tok);
    uint8_t *bits = xmalloc((size_t)n_tok * allowed_stride);
    bool *allowed = xmalloc(n_comp_rows * sizeof(allowed[0]));
    for (uint64_t i = 0; i < (uint64_t)n_tok * q_dim; i++) q[i] = quant_float();
    for (uint64_t i = 0; i < (uint64_t)(raw_cap + n_tok) * head_dim; i++) kv[i] = quant_float();
    for (uint64_t i = 0; i < (uint64_t)n_comp_rows * head_dim; i++) comp[i] = quant_float();
    for (uint32_t t = 0; t < n_tok; t++) {
        comp_counts[t] = (t + 1) / 3 < n_comp_rows ? (t + 1) / 3 : n_comp_rows;
        mask[t] = (uint8_t)(t % 3 == 1);
        for (uint64_t b = 0; b < allowed_stride; b++) bits[t * allowed_stride + b] = (uint8_t)quant_random();
    }

    /* raw_base cached rows precede the new ones: pos0 below, at and past
     * the window. */
    const uint32_t starts[4] = {0, 3, raw_cap - 1, 10};
    for (unsigned si = 0; si < 4; si++) {
        const uint32_t pos0 = starts[si];
        const uint32_t raw_base = pos0 < raw_cap ? pos0 : raw_cap;
        const float *raw = kv + (uint64_t)(raw_cap - raw_base) * head_dim;
        for (unsigned masked = 0; masked < 2; masked++) {
            layer_attention_prefix_batch(batch, &m, &layer, q, raw, comp, comp_counts,
                                         masked ? mask : NULL, masked ? bits : NULL,
                                         allowed_stride, n_tok, raw_cap, pos0, raw_base);
            for (uint32_t t = 0; t < n_tok; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t raw_count = pos + 1 < raw_cap ? pos + 1 : raw_cap;
                const bool use_mask = masked && mask[t];
                for (uint32_t c = 0; c < comp_counts[t]; c++)
                    allowed[c] = (bits[t * allowed_stride + (c >> 3)] >> (c & 7u)) & 1u;
                layer_attention_mixed_one(one, &m, &layer, q + t * q_dim,
                                          raw + (uint64_t)(raw_base + t + 1 - raw_count) * head_dim, raw_count,
                                          comp, comp_counts[t], use_mask ? allowed : NULL);
                for (uint64_t i = 0; i < q_dim; i++)
                    quant_bits_equal(masked ? "masked prefix attention" : "prefix attention",
                                     one[i], batch[t * q_dim + i], (unsigned)(t * q_dim + i));
            }
        }
    }

    /* Raw-only layers: no compressed rows, the per-token path is rows_one. */
    memset(comp_counts, 0, (size_t)n_tok * sizeof(comp_counts[0]));
    for (unsigned si = 0; si < 4; si++) {
        const uint32_t pos0 = starts[si];
        const uint32_t raw_base = pos0 < raw_cap ? pos0 : raw_cap;
        const float *raw = kv + (uint64_t)(raw_cap - raw_base) * head_dim;
        layer_attention_prefix_batch(batch, &m, &layer, q, raw, NULL, comp_counts, NULL, NULL, 0,
                                     n_tok, raw_cap, pos0, raw_base);
        for (uint32_t t = 0; t < n_tok; t++) {
            const uint32_t pos = pos0 + t;
            const uint32_t raw_count = pos + 1 < raw_cap ? pos + 1 : raw_cap;
            layer_attention_rows_one(one, &m, &layer, q + t * q_dim,
                                     raw + (uint64_t)(raw_base + t + 1 - raw_count) * head_dim, raw_count);
            for (uint64_t i = 0; i < q_dim; i++)
                quant_bits_equal("raw prefix attention", one[i], batch[t * q_dim + i], (unsigned)(t * q_dim + i));
        }
    }

    free(allowed); free(bits); free(mask); free(comp_counts);
    free(one); free(batch); free(comp); free(kv); free(q);
    quant_free(sb);
    g_ds4_shape = saved;
}

static void check_prefix_attention(void) {
    check_prefix_attention_shape(3, 40);
    check_prefix_attention_shape(64, 512);
}

/* Fill a Q8_K row either from random floats through the real quantizer or
 * with adversarial bytes (pattern < 6) and consistent block sums. */
static void fill_q8_k_row(block_q8_K *y, unsigned blocks, unsigned pattern) {
    if (pattern >= 6) {
        float *x = malloc(blocks * QK_K * sizeof(float));
        CHECK(x != NULL);
        for (unsigned j = 0; j < blocks * QK_K; j++) {
            x[j] = ((float)(quant_random() % 20001) - 10000.0f) * (pattern & 1 ? 1e-3f : 3.7f);
        }
        ds4_quantize_row_q8_K(x, y, (int64_t)(blocks * QK_K));
        free(x);
        return;
    }
    const float scales[] = {0.0f, -0.0f, 0x1p-20f, -0.00173f, 0.375f, -1.13f, 17.3f, -1024.0f};
    for (unsigned b = 0; b < blocks; b++) {
        y[b].d = scales[(pattern + b) % 8];
        for (unsigned j = 0; j < QK_K; j++) y[b].qs[j] = quant_byte(pattern, j + b);
        for (unsigned group = 0; group < 16; group++) {
            int sum = 0;
            for (unsigned j = 0; j < 16; j++) sum += y[b].qs[group * 16 + j];
            y[b].bsums[group] = (int16_t)sum;
        }
    }
}

/* IQ2_XXS rows: random, or a sweep that walks every grid index, every sign
 * pattern and every sub-block scale across the row. */
static void fill_iq2_row(block_iq2_xxs *x, unsigned blocks, bool sweep, unsigned *counter) {
    const uint16_t half[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x8400,
                            0x1801, 0x9a17, 0x3555, 0xba31, 0x7bff, 0xfbff};
    for (unsigned b = 0; b < blocks; b++) {
        x[b].d = half[(*counter + b) % 12];
        for (unsigned ib = 0; ib < 8; ib++) {
            uint32_t aux[2];
            if (sweep) {
                uint8_t grid[4];
                for (unsigned l = 0; l < 4; l++) grid[l] = (uint8_t)((*counter)++);
                memcpy(&aux[0], grid, 4);
                aux[1] = 0;
                for (unsigned l = 0; l < 4; l++) aux[1] |= (uint32_t)((*counter + 37u * l) & 127u) << (7 * l);
                aux[1] |= (uint32_t)(*counter & 15u) << 28;
            } else {
                aux[0] = quant_random();
                aux[1] = quant_random();
            }
            memcpy(x[b].qs + 4 * ib, aux, 8);
        }
    }
}

static void fill_q2_row(block_q2_K *x, unsigned blocks, unsigned *counter) {
    const uint16_t half[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x8400,
                            0x1801, 0x9a17, 0x3555, 0xba31, 0x7bff, 0xfbff};
    for (unsigned b = 0; b < blocks; b++) {
        x[b].d = half[(*counter + 3 * b) % 12];
        x[b].dmin = half[(*counter + 5 * b + 7) % 12];
        for (unsigned j = 0; j < sizeof(x[b].qs); j++) x[b].qs[j] = (uint8_t)quant_random();
        /* Every scale byte value appears over the sweep. */
        for (unsigned j = 0; j < 16; j++) x[b].scales[j] = (uint8_t)((*counter)++);
    }
}

/* The per-superblock x16 tile kernels, run over whole rows the way the
 * workers do, against the single-row reference dots, lane by lane, on dense
 * tiles (vector path) and sparse tiles (per-row path), including padding
 * lanes, odd block counts and adversarial activations. */
static void check_x16_kernels(void) {
#if DS4_MOE_X16
    const unsigned counts[] = {1, 2, 3, 5, 8, 16};
    const unsigned lanes[] = {16, 15, 7, DS4_MOE_X16_MIN_LANES, DS4_MOE_X16_MIN_LANES - 1, 1};
    unsigned counter = 0;
    for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        const unsigned nb = counts[c];
        quant_buffer ib0 = quant_alloc(nb * sizeof(block_iq2_xxs)), ib1 = quant_alloc(nb * sizeof(block_iq2_xxs));
        quant_buffer ib2 = quant_alloc(nb * sizeof(block_iq2_xxs)), ib3 = quant_alloc(nb * sizeof(block_iq2_xxs));
        quant_buffer qb0 = quant_alloc(nb * sizeof(block_q2_K)), qb1 = quant_alloc(nb * sizeof(block_q2_K));
        quant_buffer qb2 = quant_alloc(nb * sizeof(block_q2_K)), qb3 = quant_alloc(nb * sizeof(block_q2_K));
        quant_buffer yb = quant_alloc(DS4_MOE_TILE * nb * sizeof(block_q8_K));
        block_iq2_xxs *iq0 = (block_iq2_xxs *)ib0.data, *iq1 = (block_iq2_xxs *)ib1.data;
        block_iq2_xxs *iq2 = (block_iq2_xxs *)ib2.data, *iq3 = (block_iq2_xxs *)ib3.data;
        block_q2_K *q20 = (block_q2_K *)qb0.data, *q21 = (block_q2_K *)qb1.data;
        block_q2_K *q22 = (block_q2_K *)qb2.data, *q23 = (block_q2_K *)qb3.data;
        block_q8_K *acts = (block_q8_K *)yb.data;
        iq2xxs_q8 *acts_iq2 = malloc((size_t)DS4_MOE_TILE * nb * sizeof(*acts_iq2));
        CHECK(acts_iq2);
        ds4_moe_tile tile;
        tile.q = malloc((size_t)nb * DS4_MOE_TILE_Q_STRIDE);
        tile.d = malloc((size_t)nb * DS4_MOE_TILE * sizeof(float));
        tile.bsums = malloc((size_t)nb * DS4_MOE_TILE_BS_STRIDE * sizeof(int16_t));
        uint8_t *w4 = malloc((size_t)4 * nb * QK_K);
        int32_t *ls4 = malloc((size_t)4 * nb * (QK_K / 32) * sizeof(int32_t));
        float *ds4 = malloc((size_t)4 * nb * sizeof(float));
        float *dd4 = malloc((size_t)4 * nb * sizeof(float)), *dm4 = malloc((size_t)4 * nb * sizeof(float));
        CHECK(tile.q && tile.d && tile.bsums && w4 && ls4 && ds4 && dd4 && dm4);
        for (unsigned p = 0; p < 24; p++) {
            for (unsigned lane = 0; lane < DS4_MOE_TILE; lane++) {
                fill_q8_k_row(acts + lane * nb, nb, (p + lane) % 8);
            }
            iq2xxs_q8_prepare(acts_iq2, acts, (uint64_t)DS4_MOE_TILE * nb);
            fill_iq2_row(iq0, nb, p & 1, &counter);
            fill_iq2_row(iq1, nb, (p & 2) != 0, &counter);
            fill_iq2_row(iq2, nb, (p & 4) != 0, &counter);
            fill_iq2_row(iq3, nb, (p & 1) == 0, &counter);
            fill_q2_row(q20, nb, &counter);
            fill_q2_row(q21, nb, &counter);
            fill_q2_row(q22, nb, &counter);
            fill_q2_row(q23, nb, &counter);
            for (unsigned l = 0; l < sizeof(lanes) / sizeof(*lanes); l++) {
                tile.n = lanes[l];
                tile.expert = 0;
                for (unsigned lane = 0; lane < DS4_MOE_TILE; lane++) tile.pair[lane] = lane;
                if (tile.n < DS4_MOE_X16_MIN_LANES) continue;   /* the workers use the pair kernels */
                moe_tiles_load_ctx load = { .tiles = &tile, .rows = acts, .iq2_rows = acts_iq2, .nb = nb,
                                            .pairs_per_row = 1, .iq2_order = true };
                moe_tiles_load_worker(&load, 0, 1);

                /* Four IQ2_XXS rows decoded superblock-major (n_rows 4), the
                 * per-superblock four-row kernel over every superblock as the
                 * gate/up worker runs it, then the final 1/8: row by row
                 * against the reference dot. */
                {
                    const block_iq2_xxs *x4[4] = { iq0, iq1, iq2, iq3 };
                    float out4[4][DS4_MOE_TILE];
                    __m512 sumf[4];
                    for (unsigned r = 0; r < 4; r++) moe_iq2_xxs_decode_row_sb(x4[r], nb, 4, r, w4, ls4, ds4);
                    for (unsigned r = 0; r < 4; r++) sumf[r] = _mm512_setzero_ps();
                    for (unsigned i = 0; i < nb; i++) {
                        moe_iq2_xxs_x16_sb4(w4 + (size_t)i * 4 * QK_K, ls4 + (size_t)i * 4 * (QK_K / 32), ds4 + (size_t)i * 4,
                                            tile.q + (size_t)i * DS4_MOE_TILE_Q_STRIDE,
                                            tile.bsums + (size_t)i * DS4_MOE_TILE_BS_STRIDE,
                                            _mm512_loadu_ps(tile.d + (size_t)i * DS4_MOE_TILE), sumf);
                    }
                    for (unsigned r = 0; r < 4; r++) {
                        _mm512_storeu_ps(out4[r], _mm512_mul_ps(_mm512_set1_ps(0.125f), sumf[r]));
                        for (unsigned lane = 0; lane < DS4_MOE_TILE; lane++) {
                            float ref = 0.0f;
                            if (lane < tile.n) ds4_vec_dot_iq2_xxs_q8_K((int)(nb * QK_K), &ref, x4[r], acts_iq2 + lane * nb);
                            quant_equal("IQ2_XXS x16 superblock rows", ref, out4[r][lane], nb, (p * 100 + lane) * 4 + r);
                        }
                    }
                }
                /* The same for four Q2_K rows against the tile in natural order. */
                load.iq2_order = false;
                moe_tiles_load_worker(&load, 0, 1);
                {
                    const block_q2_K *x4[4] = { q20, q21, q22, q23 };
                    float out4[4][DS4_MOE_TILE];
                    __m512 sumf[4];
                    for (unsigned r = 0; r < 4; r++) moe_q2_k_decode_row_sb(x4[r], nb, 4, r, w4, ls4, dd4, dm4);
                    for (unsigned r = 0; r < 4; r++) sumf[r] = _mm512_setzero_ps();
                    for (unsigned i = 0; i < nb; i++) {
                        moe_q2_k_x16_sb4(w4 + (size_t)i * 4 * QK_K, ls4 + (size_t)i * 4 * (QK_K / 32),
                                         dd4 + (size_t)i * 4, dm4 + (size_t)i * 4,
                                         tile.q + (size_t)i * DS4_MOE_TILE_Q_STRIDE,
                                         tile.bsums + (size_t)i * DS4_MOE_TILE_BS_STRIDE,
                                         _mm512_loadu_ps(tile.d + (size_t)i * DS4_MOE_TILE), sumf);
                    }
                    for (unsigned r = 0; r < 4; r++) {
                        _mm512_storeu_ps(out4[r], sumf[r]);
                        for (unsigned lane = 0; lane < DS4_MOE_TILE; lane++) {
                            float ref = 0.0f;
                            if (lane < tile.n) ds4_vec_dot_q2_K_q8_K((int)(nb * QK_K), &ref, x4[r], acts + lane * nb);
                            quant_equal("Q2_K x16 superblock rows", ref, out4[r][lane], nb, (p * 100 + lane) * 4 + r);
                        }
                    }
                }
            }
        }
        free(dm4); free(dd4); free(ds4); free(ls4); free(w4);
        free(acts_iq2); free(tile.bsums); free(tile.d); free(tile.q);
        quant_free(yb); quant_free(qb3); quant_free(qb2); quant_free(qb1); quant_free(qb0);
        quant_free(ib3); quant_free(ib2); quant_free(ib1); quant_free(ib0);
    }
#endif
}

/* The grouped routed FFN against a per-token re-implementation of the decode
 * path (matvec_iq2_xxs_mid_worker + matvec_q2_k_accum_worker): gate/up dots,
 * clamp, SwiGLU, router weight, Q8_K requantization, down dots and the
 * slot-ordered accumulation, bitwise.  The expert tensors and activations end
 * at a guard page; the routing is skewed so tile sizes span one lane to full. */
static void check_grouped_ffn(uint64_t in_dim, uint64_t mid_dim, uint64_t out_dim) {
    const uint32_t saved_n_expert = g_ds4_shape.n_expert;
    const uint32_t saved_n_used = g_ds4_shape.n_expert_used;
    const uint32_t n_expert = 8, n_used = 3, n_tok = 61;
    const unsigned nb_in = (unsigned)(in_dim / QK_K), nb_mid = (unsigned)(mid_dim / QK_K);
    g_ds4_shape.n_expert = n_expert;
    g_ds4_shape.n_expert_used = n_used;

    const uint64_t gate_row_bytes = nb_in * sizeof(block_iq2_xxs);
    const uint64_t down_row_bytes = nb_mid * sizeof(block_q2_K);
    quant_buffer gb = quant_alloc((size_t)n_expert * mid_dim * gate_row_bytes);
    quant_buffer ub = quant_alloc((size_t)n_expert * mid_dim * gate_row_bytes);
    quant_buffer db = quant_alloc((size_t)n_expert * out_dim * down_row_bytes);
    quant_buffer xb = quant_alloc((size_t)n_tok * in_dim * sizeof(float));
    block_iq2_xxs *gate = (block_iq2_xxs *)gb.data, *up = (block_iq2_xxs *)ub.data;
    block_q2_K *down = (block_q2_K *)db.data;
    float *norm = (float *)xb.data;
    int *selected = malloc((size_t)n_tok * n_used * sizeof(int));
    float *pair_weight = malloc((size_t)n_tok * n_used * sizeof(float));
    float *moe = malloc((size_t)n_tok * out_dim * sizeof(float));
    float *mid = malloc(mid_dim * sizeof(float));
    float *acc = malloc(out_dim * sizeof(float));
    block_q8_K *xq = malloc(nb_in * sizeof(block_q8_K));
    iq2xxs_q8 *xq_iq2 = malloc(nb_in * sizeof(iq2xxs_q8));
    block_q8_K *midq = malloc(nb_mid * sizeof(block_q8_K));
    CHECK(selected && pair_weight && moe && mid && acc && xq && xq_iq2 && midq);

    unsigned counter = 0;
    for (uint64_t r = 0; r < (uint64_t)n_expert * mid_dim; r++) {
        fill_iq2_row(gate + r * nb_in, nb_in, false, &counter);
        fill_iq2_row(up + r * nb_in, nb_in, false, &counter);
    }
    for (uint64_t r = 0; r < (uint64_t)n_expert * out_dim; r++) fill_q2_row(down + r * nb_mid, nb_mid, &counter);
    for (uint64_t j = 0; j < (uint64_t)n_tok * in_dim; j++) {
        norm[j] = ((float)(quant_random() % 20001) - 10000.0f) * 1e-4f;
    }
    /* Distinct experts per token, expert e >= 1 drawn with weight e. */
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t slot = 0; slot < n_used; slot++) {
            int e;
            bool dup;
            do {
                uint32_t r = quant_random() % (n_expert * (n_expert - 1) / 2);
                for (e = 1; r >= (uint32_t)e; e++) r -= (uint32_t)e;
                dup = false;
                for (uint32_t s = 0; s < slot; s++) dup = dup || selected[t * n_used + s] == e;
            } while (dup);
            selected[t * n_used + slot] = e;
            pair_weight[t * n_used + slot] = (float)(quant_random() % 1000 + 1) * 1e-3f;
        }
    }
    selected[0] = 0; /* expert 0 gets a single-lane tile */

    for (unsigned pass = 0; pass < 2; pass++) {
        const float clamp = pass ? 7.0f : 0.0f;
        moe_grouped_ffn(moe, (const uint8_t *)gate, (const uint8_t *)up, (const uint8_t *)down,
                        gate_row_bytes, gate_row_bytes, down_row_bytes, in_dim, mid_dim, out_dim,
                        norm, selected, pair_weight, n_tok, clamp);

        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint64_t row = 0; row < out_dim; row++) acc[row] = 0.0f;
            ds4_quantize_row_q8_K(norm + t * in_dim, xq, (int64_t)in_dim);
            iq2xxs_q8_prepare(xq_iq2, xq, nb_in);
            for (uint32_t slot = 0; slot < n_used; slot++) {
                const uint32_t e = (uint32_t)selected[t * n_used + slot];
                const float weight = pair_weight[t * n_used + slot];
                for (uint64_t row = 0; row < mid_dim; row++) {
                    float g = 0.0f, u = 0.0f;
                    ds4_vec_dot_iq2_xxs_pair_q8_K((int)in_dim, &g, &u,
                                                  gate + (e * mid_dim + row) * nb_in,
                                                  up + (e * mid_dim + row) * nb_in, xq_iq2);
                    if (clamp > 1.0e-6f) {
                        if (g > clamp) g = clamp;
                        if (u > clamp) u = clamp;
                        if (u < -clamp) u = -clamp;
                    }
                    mid[row] = silu(g) * u * weight;
                }
                ds4_quantize_row_q8_K(mid, midq, (int64_t)mid_dim);
                for (uint64_t row = 0; row < out_dim; row++) {
                    float v = 0.0f;
                    ds4_vec_dot_q2_K_q8_K((int)mid_dim, &v, down + (e * out_dim + row) * nb_mid, midq);
                    acc[row] += v;
                }
            }
            for (uint64_t row = 0; row < out_dim; row++) {
                quant_equal("grouped routed FFN", acc[row], moe[t * out_dim + row], (unsigned)row, t * 10 + pass);
            }
        }
    }

    g_ds4_shape.n_expert = saved_n_expert;
    g_ds4_shape.n_expert_used = saved_n_used;
    free(midq); free(xq_iq2); free(xq); free(acc); free(mid); free(moe); free(pair_weight); free(selected);
    quant_free(xb); quant_free(db); quant_free(ub); quant_free(gb);
}

/* The grouped V4.1 Q8_0 dispatch against one reference matvec per group,
 * with special scales and activations: every row keeps its kernel and its
 * input slice, so even NaN payloads must match. */
static void check_ds41_q8_grouped(void) {
    static const uint16_t scales[] = {0x0000, 0x8000, 0x0001, 0x83ff, 0x7bff,
                                      0x7c00, 0xfc00, 0x7e00, 0x7c01, 0x3c00};
    static const uint32_t specials[] = {0x00000000, 0x80000000, 0x00000001,
        0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345, 0x7f7fffff};
    const uint64_t dims[] = {32, 100, 4096};
    const uint64_t group_rows[] = {16, 32, 1024};
    for (size_t di = 0; di < sizeof(dims) / sizeof(*dims); di++)
    for (size_t gi = 0; gi < sizeof(group_rows) / sizeof(*group_rows); gi++)
    for (uint64_t groups = 1; groups <= 8; groups *= 2) {
        const uint64_t in_dim = dims[di], rows = group_rows[gi] * groups;
        const uint64_t blocks = (in_dim + 31) / 32, bytes = rows * blocks * 34;
        uint8_t *w = malloc(bytes);
        float *x = malloc(groups * in_dim * sizeof(float));
        float *want = malloc(rows * sizeof(float)), *got = malloc(rows * sizeof(float));
        CHECK(w && x && want && got);
        for (uint64_t i = 0; i < bytes; i++) w[i] = (uint8_t)quant_random();
        for (uint64_t b = 0; b < rows * blocks; b++) {
            const uint32_t r = quant_random();
            const uint16_t d = r % 4 ? (uint16_t)(0x1800 + r % 0x1000) :
                scales[(r >> 8) % (sizeof(scales) / sizeof(*scales))];
            memcpy(w + b * 34, &d, sizeof(d));
            if (r % 7 == 0) w[b * 34 + 2 + (r >> 16) % 32] = 0x80;
        }
        for (uint64_t i = 0; i < groups * in_dim; i++) {
            const uint32_t r = quant_random();
            if (r % 16 == 0) memcpy(&x[i], &specials[(r >> 8) % 8], sizeof(float));
            else x[i] = (float)((int32_t)(r >> 8) % 20001 - 10000) * 1e-4f;
        }
        for (uint64_t g = 0; g < groups; g++) {
            matvec_q8_0_f32_ref_ctx ref = {
                .out = want + g * group_rows[gi], .data = w + g * group_rows[gi] * blocks * 34,
                .x = x + g * in_dim, .in_dim = in_dim, .blocks = blocks,
            };
            matvec_q8_0_f32_ref_worker(&ref, 0, group_rows[gi]);
        }
        const ds4_model m = {.map = w, .size = bytes};
        const ds4_tensor t = {.type = DS4_TENSOR_Q8_0, .ndim = 2,
                              .dim = {in_dim, rows}, .elements = in_dim * rows, .bytes = bytes};
        ds41_cpu_matvec_q8_grouped(got, &m, &t, x, group_rows[gi], in_dim);
        CHECK(memcmp(want, got, rows * sizeof(float)) == 0);
        if (groups == 1) {
            ds41_cpu_matvec_q8(got, &m, &t, x);
            CHECK(memcmp(want, got, rows * sizeof(float)) == 0);
        }
#if defined(__AVX512F__) && defined(__AVX512BW__)
        if (groups == 2) {
            /* The two groups as two matrices that share the first input. */
            ds4_tensor t0 = t, t1 = t;
            t0.dim[1] = t1.dim[1] = group_rows[gi];
            t0.elements = t1.elements = in_dim * group_rows[gi];
            t0.bytes = t1.bytes = bytes / 2;
            t1.abs_offset = bytes / 2;
            float *pair = malloc(rows * sizeof(float));
            CHECK(pair);
            for (uint64_t g = 0; g < 2; g++) {
                matvec_q8_0_f32_ref_ctx ref = {
                    .out = want + g * group_rows[gi], .data = w + g * group_rows[gi] * blocks * 34,
                    .x = x, .in_dim = in_dim, .blocks = blocks,
                };
                matvec_q8_0_f32_ref_worker(&ref, 0, group_rows[gi]);
            }
            CHECK(ds41_cpu_matvec_q8_pair(pair, pair + group_rows[gi], &m, &t0, &t1, x));
            CHECK(memcmp(want, pair, rows * sizeof(float)) == 0);
            free(pair);
        }
#endif
        free(w); free(x); free(want); free(got);
    }
}

int main(void) {
    check_ffn_graph();
    check_raw_window();
    puts("CPU raw KV window slides like the memmove it replaces PASS");
    check_attn_graph();
    check_thread_pool();
    check_ds41_q8_grouped();
    puts("CPU V4.1 grouped and paired Q8_0 projections bitwise against the reference PASS");
    check_f16_round_trip();
    check_compressor_project_batch();
    check_compressor_deferred();
    check_prefix_attention();
    puts("CPU prefill compressor batch and prefix attention bitwise oracles PASS");
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
#if defined(DS4_NO_GPU) && defined(__AVX512VNNI__)
    check_q8_0_scale_f32();
    const uint32_t threads[] = {1, 3, 16};
    for (size_t i = 0; i < sizeof(threads) / sizeof(*threads); i++) {
        g_requested_threads = threads[i];
        check_q8_0_batch();
        ds4_threads_shutdown();
    }
    g_requested_threads = 0;
    puts("CPU AVX-512 VNNI Q8_0 tile GEMM bitwise against dot_q8_0_row PASS");
#endif
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
    check_f16_dispatch();
    check_attn_input_projections();
    puts("CPU grouped F16 dispatch and fused decode projections bitwise PASS");
    check_x16_kernels();
#if DS4_MOE_X16
    puts("CPU AVX-512 VNNI x16 expert kernels bitwise PASS");
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
    /* nb_in > nb_mid with a partial last row block, then nb_in < nb_mid,
     * then a full down row block followed by a partial one. */
    check_grouped_ffn(2 * QK_K, QK_K, 2 * DS4_MOE_ROW_BLOCK + 6);
    check_grouped_ffn(QK_K, 2 * QK_K, 2 * DS4_MOE_ROW_BLOCK);
    check_grouped_ffn(QK_K, QK_K, DS4_MOE_DOWN_ROW_BLOCK + 2 * DS4_MOE_ROW_BLOCK + 6);
    puts("CPU grouped routed FFN bitwise with the per-token decode path PASS");
    return 0;
}
