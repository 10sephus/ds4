/* Compare the V4.1 row-SIMD projections against the existing CPU reference.
 * Build with the CPU core's ordinary flags (without fast-math). */
#include "../ds4.c"
#include "../ds4_deepseek41_cpu_x86.inc"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static uint32_t test_rng = 0x71329acdu;
static uint32_t test_random(void) {
    test_rng ^= test_rng << 13;
    test_rng ^= test_rng >> 17;
    test_rng ^= test_rng << 5;
    return test_rng;
}

static float test_float(void) {
    return (float)((int)(test_random() % 2001) - 1000) * 0.00117f;
}

static uint16_t test_half(void) {
    const uint16_t edges[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff,
                             0x0400, 0x8400, 0x1800, 0x9800, 0x3800, 0xb800};
    const uint32_t r = test_random();
    return (r & 3) ? (uint16_t)((r & 0x8000) | (0x1000 + r % 0x2800)) :
                    edges[(r >> 2) % (sizeof(edges) / sizeof(*edges))];
}

/* Guard pages catch gather overreads too: ASan cannot instrument an intrinsic
 * gather's individual lanes. Place the last stored block at the page edge. */
typedef struct { void *mapping; size_t mapped; uint8_t *data; } test_buffer;
static test_buffer test_alloc(size_t bytes) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const size_t rounded = (bytes + page - 1) / page * page;
    void *mapping = mmap(NULL, rounded + page, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(mapping != MAP_FAILED);
    CHECK(mprotect((uint8_t *)mapping + rounded, page, PROT_NONE) == 0);
    return (test_buffer){mapping, rounded + page, (uint8_t *)mapping + rounded - bytes};
}
static void test_free(test_buffer b) { CHECK(munmap(b.mapping, b.mapped) == 0); }

static void test_equal(const char *what, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t aa, bb;
        memcpy(&aa, a + i, sizeof(aa));
        memcpy(&bb, b + i, sizeof(bb));
        if (aa != bb) {
            fprintf(stderr, "%s[%zu]: %08x != %08x (%g != %g)\n",
                    what, i, aa, bb, a[i], b[i]);
            exit(1);
        }
    }
}

static void test_fill_quants(uint8_t *data, size_t bytes, uint32_t type) {
    for (size_t i = 0; i < bytes; i++) data[i] = (uint8_t)test_random();
    if (type == DS4_TENSOR_Q8_0) {
        for (size_t i = 0; i < bytes; i += 34) {
            const uint16_t h = test_half();
            memcpy(data + i, &h, sizeof(h));
        }
    } else if (type == DS4_TENSOR_Q4_K) {
        for (size_t i = 0; i < bytes; i += sizeof(block_q4_K)) {
            const uint16_t h[] = {test_half(), test_half()};
            memcpy(data + i, h, sizeof(h));
        }
    } else if (type == DS4_TENSOR_IQ2_XXS) {
        for (size_t i = 0; i < bytes; i += sizeof(block_iq2_xxs)) {
            const uint16_t h = test_half();
            memcpy(data + i, &h, sizeof(h));
        }
    } else {
        CHECK(type == DS4_TENSOR_Q2_K);
        for (size_t i = 0; i < bytes; i += sizeof(block_q2_K)) {
            const uint16_t h[] = {test_half(), test_half()};
            memcpy(data + i + offsetof(block_q2_K, d), h, sizeof(h));
        }
    }
}

static void check_q8(void) {
    const uint64_t dims[] = {1, 3, 4, 7, 28, 31, 32, 33, 63, 64, 65, 256, 1280, 5120};
    const uint64_t rows[] = {1, 15, 16, 17, 31, 32, 33, 511, 513};
    for (size_t d = 0; d < sizeof(dims) / sizeof(*dims); d++) {
        for (size_t r = 0; r < sizeof(rows) / sizeof(*rows); r++) {
            const uint64_t k = dims[d], n = rows[r];
            const size_t bytes = (size_t)n * ((k + 31) / 32) * 34;
            test_buffer w = test_alloc(bytes), xb = test_alloc(k * sizeof(float));
            test_buffer ab = test_alloc(n * sizeof(float)), bb = test_alloc(n * sizeof(float));
            float *x = (float *)xb.data, *a = (float *)ab.data, *b = (float *)bb.data;
            test_fill_quants(w.data, bytes, DS4_TENSOR_Q8_0);
            for (uint64_t i = 0; i < k; i++) x[i] = test_float();
            ds4_model m = {.map = w.data, .size = bytes};
            ds4_tensor t = {.type = DS4_TENSOR_Q8_0, .ndim = 2,
                           .dim = {k, n}, .elements = k * n, .bytes = bytes};
            matvec_q8_0_f32_ref(a, &m, &t, x);
            ds41_cpu_matvec_q8(b, &m, &t, x);
            test_equal("Q8", a, b, n);
            test_free(bb); test_free(ab); test_free(xb); test_free(w);
        }
    }
}

static size_t test_block_bytes(uint32_t type) {
    return type == DS4_TENSOR_Q4_K ? sizeof(block_q4_K) :
           type == DS4_TENSOR_IQ2_XXS ? sizeof(block_iq2_xxs) : sizeof(block_q2_K);
}

/* Gate and up are of type, down of down_type (the Q2 models: IQ2_XXS, Q2_K). */
static void check_moe(uint32_t embd, uint32_t hidden, uint32_t type, uint32_t down_type, float clamp) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    g_ds4_shape.n_embd = embd;
    g_ds4_shape.n_ff_exp = hidden;
    g_ds4_shape.n_expert = 7;
    g_ds4_shape.n_expert_used = 3;
    g_ds4_shape.swiglu_clamp_exp = clamp;
    const int selected[] = {6, 2, 0};
    const float weights[] = {0.2173f, -0.6129f, 1.397f};
    const size_t bytes = (size_t)embd * hidden * 7 / QK_K * test_block_bytes(type);
    const size_t down_bytes = (size_t)embd * hidden * 7 / QK_K * test_block_bytes(down_type);
    test_buffer w = test_alloc(2 * bytes + down_bytes), xb = test_alloc(embd * sizeof(float));
    test_buffer ma = test_alloc(3 * hidden * sizeof(float)), mb = test_alloc(3 * hidden * sizeof(float));
    test_buffer oa = test_alloc(embd * sizeof(float)), ob = test_alloc(embd * sizeof(float));
    test_fill_quants(w.data, 2 * bytes, type);
    test_fill_quants(w.data + 2 * bytes, down_bytes, down_type);
    float *x = (float *)xb.data;
    for (uint32_t i = 0; i < embd; i++) x[i] = test_float();
    ds4_model m = {.map = w.data, .size = 2 * bytes + down_bytes};
    ds4_tensor gate = {.type = type, .ndim = 3, .dim = {embd, hidden, 7},
                      .elements = (uint64_t)embd * hidden * 7, .bytes = bytes};
    ds4_tensor up = gate, down = gate;
    up.abs_offset = bytes;
    down.type = down_type;
    down.abs_offset = 2 * bytes;
    down.bytes = down_bytes;
    down.dim[0] = hidden; down.dim[1] = embd;
    ds4_layer_weights l = {.ffn_gate_exps = &gate, .ffn_up_exps = &up, .ffn_down_exps = &down};
    layer_glm_routed_moe_one_f32_ref((float *)oa.data, (float *)ma.data, &m, &l, x, selected, weights);
    ds41_cpu_routed_moe((float *)ob.data, (float *)mb.data, &m, &l, x, selected, weights);
    test_equal("MoE mid", (float *)ma.data, (float *)mb.data, 3 * hidden);
    test_equal("MoE out", (float *)oa.data, (float *)ob.data, embd);
    test_free(ob); test_free(oa); test_free(mb); test_free(ma); test_free(xb); test_free(w);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
static void check_half(void) {
    for (uint32_t start = 0; start < 65536; start += 16) {
        uint32_t h[16];
        float a[16], b[16];
        for (uint32_t i = 0; i < 16; i++) {
            h[i] = start + i;
            a[i] = f16_to_f32((uint16_t)h[i]);
        }
        _mm512_storeu_ps(b, ds41_cpu_half16(_mm512_loadu_si512(h)));
        test_equal("half", a, b, 16);
    }
}

/* The IQ2_XXS pair kernel against the scalar dot, row by row: every grid
 * index and every sign index in each of the four positions of a sub-block,
 * all scale nibbles, edge-case F16 scales and signed zero activations. */
static void check_iq2xxs_pair(void) {
    const uint64_t k = 5120, rows = 64, stride = k / QK_K * sizeof(block_iq2_xxs);
    test_buffer g = test_alloc(rows * stride), u = test_alloc(rows * stride);
    test_buffer xb = test_alloc(k * sizeof(float));
    float *x = (float *)xb.data;
    for (uint64_t i = 0; i < k; i++) {
        x[i] = test_float();
        if (test_random() % 61 == 0) x[i] = (test_random() & 1) ? -0.0f : 0.0f;
    }
    uint32_t n = 0;
    for (unsigned m = 0; m < 2; m++) {
        uint8_t *w = m ? u.data : g.data;
        test_fill_quants(w, rows * stride, DS4_TENSOR_IQ2_XXS);
        for (uint64_t r = 0; r < rows; r++) {
            for (uint64_t b = 0; b < k / QK_K; b++) {
                uint8_t *q = w + r * stride + b * sizeof(block_iq2_xxs) + offsetof(block_iq2_xxs, qs);
                for (unsigned ib = 0; ib < QK_K / 32; ib++, n++) {
                    uint32_t aux[2] = {0, (n % 16u) << 28};
                    for (unsigned l = 0; l < 4; l++) {
                        aux[0] |= ((n + 64u * l) & 255u) << (8 * l);
                        aux[1] |= ((n / 2u + 32u * l) & 127u) << (7 * l);
                    }
                    memcpy(q + ib * 8u, aux, sizeof(aux));
                }
            }
        }
    }
    for (uint64_t row = 0; row < rows; row += 16) {
        __m512 sums[2];
        float a[2][16], b[2][16];
        ds41_cpu_iq2xxs_pair16(sums, g.data + row * stride, stride,
                               u.data + row * stride, stride, x, k);
        _mm512_storeu_ps(b[0], sums[0]);
        _mm512_storeu_ps(b[1], sums[1]);
        for (unsigned lane = 0; lane < 16; lane++) {
            a[0][lane] = ds4_vec_dot_iq2_xxs_f32((int)k,
                (const block_iq2_xxs *)(g.data + (row + lane) * stride), x);
            a[1][lane] = ds4_vec_dot_iq2_xxs_f32((int)k,
                (const block_iq2_xxs *)(u.data + (row + lane) * stride), x);
        }
        test_equal("IQ2_XXS pair gate", a[0], b[0], 16);
        test_equal("IQ2_XXS pair up", a[1], b[1], 16);
    }
    test_free(xb); test_free(u); test_free(g);
}

/* The Q2_K pair kernel against the scalar dot, row by row, two matrices
 * with their own inputs: random quants and scale bytes, edge-case F16 d and
 * dmin, signed zero activations. */
static void check_q2k_pair(void) {
    const uint64_t k = 2304, rows = 48, stride = k / QK_K * sizeof(block_q2_K);
    test_buffer w0 = test_alloc(rows * stride), w1 = test_alloc(rows * stride);
    test_buffer xb = test_alloc(2 * k * sizeof(float));
    float *x = (float *)xb.data;
    for (uint64_t i = 0; i < 2 * k; i++) {
        x[i] = test_float();
        if (test_random() % 61 == 0) x[i] = (test_random() & 1) ? -0.0f : 0.0f;
    }
    test_fill_quants(w0.data, rows * stride, DS4_TENSOR_Q2_K);
    test_fill_quants(w1.data, rows * stride, DS4_TENSOR_Q2_K);
    for (uint64_t row = 0; row < rows; row += 16) {
        __m512 sums[2];
        float a[2][16], b[2][16];
        ds41_cpu_q2k_pair16(sums, w0.data + row * stride, stride, x,
                            w1.data + row * stride, stride, x + k, k);
        _mm512_storeu_ps(b[0], sums[0]);
        _mm512_storeu_ps(b[1], sums[1]);
        for (unsigned lane = 0; lane < 16; lane++) {
            a[0][lane] = ds4_vec_dot_q2_K_f32((int)k,
                (const block_q2_K *)(w0.data + (row + lane) * stride), x);
            a[1][lane] = ds4_vec_dot_q2_K_f32((int)k,
                (const block_q2_K *)(w1.data + (row + lane) * stride), x + k);
        }
        test_equal("Q2_K pair 0", a[0], b[0], 16);
        test_equal("Q2_K pair 1", a[1], b[1], 16);
    }
    test_free(xb); test_free(w1); test_free(w0);
}

/* Full MoE layouts require multiples of 256. Exercise worker row tails
 * directly as well, including a worker range split at an expert boundary. */
static void check_q4_tails(void) {
    const uint64_t rows[] = {1, 15, 16, 17, 31, 33};
    const uint64_t dims[] = {256, 512, 2304, 5120};
    for (size_t d = 0; d < sizeof(dims) / sizeof(*dims); d++) {
        const uint64_t k = dims[d];
        for (size_t r = 0; r < sizeof(rows) / sizeof(*rows); r++) {
            const uint64_t n = rows[r], stride = k / QK_K * sizeof(block_q4_K);
            const size_t bytes = n * stride;
            test_buffer w = test_alloc(bytes), xb = test_alloc(2 * k * sizeof(float));
            float *x = (float *)xb.data;
            test_fill_quants(w.data, bytes, DS4_TENSOR_Q4_K);
            for (uint64_t i = 0; i < 2 * k; i++) x[i] = test_float();
            float a[66], b[66];
            glm_routed_moe_f32_mid_ctx mc = {
                .mid = a, .x = x, .gate_base = {w.data, w.data}, .up_base = {w.data, w.data},
                .expert_weight = {.213f, -.917f}, .in_dim = k, .out_dim = n,
                .gate_row_bytes = {stride, stride}, .up_row_bytes = {stride, stride},
                .gate_type = DS4_TENSOR_Q4_K, .up_type = DS4_TENSOR_Q4_K, .n_expert = 2, .clamp = 10,
            };
            glm_routed_moe_f32_mid_worker(&mc, 0, 2 * n);
            mc.mid = b;
            const uint64_t tiles = 2 * ((n + 15) / 16);
            ds41_cpu_mid_worker(&mc, 0, 1);
            ds41_cpu_mid_worker(&mc, 1, tiles);
            test_equal("Q4 mid tails", a, b, 2 * n);
            glm_routed_moe_f32_down_ctx dc = {
                .out = a, .mid = x, .down_base = {w.data, w.data},
                .in_dim = k, .out_dim = n, .down_row_bytes = {stride, stride},
                .down_type = DS4_TENSOR_Q4_K, .n_expert = 2,
            };
            glm_routed_moe_f32_down_worker(&dc, 0, n);
            dc.out = b;
            ds41_cpu_down_worker(&dc, 0, (n + 15) / 16);
            test_equal("Q4 down tails", a, b, n);
            const uint64_t stride2 = k / QK_K * sizeof(block_q2_K);
            test_buffer w2 = test_alloc(n * stride2);
            test_fill_quants(w2.data, n * stride2, DS4_TENSOR_Q2_K);
            dc.out = a;
            dc.down_base[0] = dc.down_base[1] = w2.data;
            dc.down_row_bytes[0] = dc.down_row_bytes[1] = stride2;
            dc.down_type = DS4_TENSOR_Q2_K;
            glm_routed_moe_f32_down_worker(&dc, 0, n);
            dc.out = b;
            ds41_cpu_down_worker(&dc, 0, (n + 15) / 16);
            test_equal("Q2_K down tails", a, b, n);
            test_buffer w5 = test_alloc(n * stride2), m3 = test_alloc(3 * k * sizeof(float));
            test_fill_quants(w5.data, n * stride2, DS4_TENSOR_Q2_K);
            for (uint64_t i = 0; i < 3 * k; i++) ((float *)m3.data)[i] = test_float();
            dc.out = a;
            dc.mid = (const float *)m3.data;
            dc.down_base[2] = w5.data;
            dc.down_row_bytes[2] = stride2;
            dc.n_expert = 3;
            glm_routed_moe_f32_down_worker(&dc, 0, n);
            dc.out = b;
            ds41_cpu_down_worker(&dc, 0, (n + 15) / 16);
            test_equal("Q2_K down tails, 3 experts", a, b, n);
            test_free(m3); test_free(w5);
            const uint64_t stride3 = k / QK_K * sizeof(block_iq2_xxs);
            test_buffer w3 = test_alloc(n * stride3), w4 = test_alloc(n * stride3);
            test_fill_quants(w3.data, n * stride3, DS4_TENSOR_IQ2_XXS);
            test_fill_quants(w4.data, n * stride3, DS4_TENSOR_IQ2_XXS);
            mc.mid = a;
            mc.gate_base[0] = mc.up_base[1] = w3.data;
            mc.gate_base[1] = mc.up_base[0] = w4.data;
            mc.gate_row_bytes[0] = mc.gate_row_bytes[1] = stride3;
            mc.up_row_bytes[0] = mc.up_row_bytes[1] = stride3;
            mc.gate_type = mc.up_type = DS4_TENSOR_IQ2_XXS;
            glm_routed_moe_f32_mid_worker(&mc, 0, 2 * n);
            mc.mid = b;
            ds41_cpu_mid_worker(&mc, 0, 1);
            ds41_cpu_mid_worker(&mc, 1, tiles);
            test_equal("IQ2_XXS mid tails", a, b, 2 * n);
            test_free(w4); test_free(w3); test_free(w2); test_free(xb); test_free(w);
        }
    }
}
#endif

int main(void) {
    const ds4_shape saved = g_ds4_shape;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    CHECK(pthread_once(&ds41_iq2xxs_once, ds41_iq2xxs_tables_init) == 0);
    check_half();
    check_iq2xxs_pair();
    check_q2k_pair();
    check_q4_tails();
#endif
    /* Both pool layouts: NUMA halves per group (on a two-node host) and one cursor. */
    ds4_numa_init();
    const bool numa = g_numa.on;
    const uint32_t threads[] = {1, 3, 16};
    for (int mode = numa; mode >= 0; mode--) {
        g_numa.on = mode;
        for (size_t i = 0; i < sizeof(threads) / sizeof(*threads); i++) {
            g_requested_threads = threads[i];
            check_q8();
            check_moe(256, 512, DS4_TENSOR_Q4_K, DS4_TENSOR_Q4_K, 10.0f);
            check_moe(512, 256, DS4_TENSOR_Q4_K, DS4_TENSOR_Q4_K, 0.0f);
            check_moe(256, 256, DS4_TENSOR_Q2_K, DS4_TENSOR_Q2_K, 10.0f);
            check_moe(512, 768, DS4_TENSOR_IQ2_XXS, DS4_TENSOR_Q2_K, 10.0f);
            check_moe(5120, 2304, DS4_TENSOR_IQ2_XXS, DS4_TENSOR_Q2_K, 10.0f);
            ds4_threads_shutdown();
            printf("V4.1 CPU Q8/Q4/Q2 bitwise reference checks: %u threads%s PASS\n",
                   threads[i], mode ? ", NUMA halves" : "");
        }
    }
    g_numa.on = numa;
    g_requested_threads = 0;
    g_ds4_shape = saved;
#if !defined(__AVX512F__) || !defined(__AVX512BW__)
    puts("AVX512 unavailable at compile time: scalar fallback checked");
#endif
    return 0;
}
