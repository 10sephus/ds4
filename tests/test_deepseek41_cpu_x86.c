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

/* A random V4.1 Q2 layer with eight experts, its tensors packed in one
 * mapping: what the staged FFN and attention projections read. */
typedef struct {
    test_buffer map;
    ds4_model m;
    ds4_tensor t[32];
    uint32_t n;
    uint64_t used;
} test_layer;

static ds4_tensor *test_tensor(test_layer *tl, uint32_t type, uint64_t d0,
                               uint64_t d1, uint64_t d2) {
    ds4_tensor *t = &tl->t[tl->n++];
    uint64_t row_bytes;
    CHECK(tl->n <= 32 && tensor_nbytes(type, d0, &row_bytes));
    *t = (ds4_tensor){.type = type, .ndim = d2 ? 3 : d1 ? 2 : 1, .dim = {d0, d1 ? d1 : 1, d2 ? d2 : 1},
                      .elements = d0 * (d1 ? d1 : 1) * (d2 ? d2 : 1),
                      .bytes = row_bytes * (d1 ? d1 : 1) * (d2 ? d2 : 1), .abs_offset = tl->used};
    tl->used = (tl->used + t->bytes + 63) / 64 * 64;
    CHECK(tl->used <= tl->map.mapped - (size_t)sysconf(_SC_PAGESIZE));
    uint8_t *data = tl->map.data + t->abs_offset;
    if (type == DS4_TENSOR_F32) {
        for (uint64_t i = 0; i < t->elements; i++) ((float *)data)[i] = test_float();
    } else if (type == DS4_TENSOR_F16) {
        for (uint64_t i = 0; i < t->elements; i++) ((uint16_t *)data)[i] = test_half();
    } else {
        test_fill_quants(data, t->bytes, type);
    }
    return t;
}

static void test_layer_init(test_layer *tl, ds4_layer_weights *l) {
    memset(tl, 0, sizeof(*tl));
    memset(l, 0, sizeof(*l));
    tl->map = test_alloc((size_t)400 << 20);
    tl->m = (ds4_model){.map = tl->map.data, .size = (uint64_t)400 << 20};
    l->hc_attn_fn = test_tensor(tl, DS4_TENSOR_F16, 20480, 24, 0);
    l->hc_attn_scale = test_tensor(tl, DS4_TENSOR_F32, 3, 0, 0);
    l->hc_attn_base = test_tensor(tl, DS4_TENSOR_F32, 24, 0, 0);
    l->hc_ffn_fn = test_tensor(tl, DS4_TENSOR_F16, 20480, 24, 0);
    l->hc_ffn_scale = test_tensor(tl, DS4_TENSOR_F32, 3, 0, 0);
    l->hc_ffn_base = test_tensor(tl, DS4_TENSOR_F32, 24, 0, 0);
    l->ffn_gate_inp = test_tensor(tl, DS4_TENSOR_F32, 5120, 8, 0);
    l->ffn_exp_probs_b = test_tensor(tl, DS4_TENSOR_F32, 8, 0, 0);
    l->ffn_gate_shexp = test_tensor(tl, DS4_TENSOR_Q8_0, 5120, 2304, 0);
    l->ffn_up_shexp = test_tensor(tl, DS4_TENSOR_Q8_0, 5120, 2304, 0);
    l->ffn_down_shexp = test_tensor(tl, DS4_TENSOR_Q8_0, 2304, 5120, 0);
    l->ffn_gate_exps = test_tensor(tl, DS4_TENSOR_IQ2_XXS, 5120, 2304, 8);
    l->ffn_up_exps = test_tensor(tl, DS4_TENSOR_IQ2_XXS, 5120, 2304, 8);
    l->ffn_down_exps = test_tensor(tl, DS4_TENSOR_Q2_K, 2304, 5120, 8);
    l->attn_q_a = test_tensor(tl, DS4_TENSOR_Q8_0, 5120, 1280, 0);
    l->attn_q_a_norm = test_tensor(tl, DS4_TENSOR_F32, 1280, 0, 0);
    l->attn_q_b = test_tensor(tl, DS4_TENSOR_Q8_0, 1280, 32768, 0);
    l->attn_kv = test_tensor(tl, DS4_TENSOR_Q8_0, 5120, 512, 0);
    l->attn_kv_a_norm = test_tensor(tl, DS4_TENSOR_F32, 512, 0, 0);
    l->attn_compressor_kv = test_tensor(tl, DS4_TENSOR_F16, 5120, 512, 0);
    l->attn_compressor_gate = test_tensor(tl, DS4_TENSOR_F16, 5120, 512, 0);
    l->attn_compressor_norm = test_tensor(tl, DS4_TENSOR_F32, 512, 0, 0);
    l->indexer_attn_k = test_tensor(tl, DS4_TENSOR_F16, 512, 128, 0);
    l->indexer_k_norm = test_tensor(tl, DS4_TENSOR_F32, 128, 0, 0);
    l->indexer_attn_q_b = test_tensor(tl, DS4_TENSOR_F16, 1280, 4096, 0);
    l->indexer_proj = test_tensor(tl, DS4_TENSOR_F16, 5120, 32, 0);
    l->attn_sinks = test_tensor(tl, DS4_TENSOR_F32, 64, 0, 0);
    l->attn_output_a = test_tensor(tl, DS4_TENSOR_Q8_0, 4096, 8192, 0);
    l->attn_output_b = test_tensor(tl, DS4_TENSOR_Q8_0, 8192, 5120, 0);
    l->ffn_norm = test_tensor(tl, DS4_TENSOR_F32, 5120, 0, 0);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* The staged FFN against the plain path: same split, block and HC state. */
static void check_ffn_graph(test_layer *tl, const ds4_layer_weights *l) {
    ds41_cpu_state *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
    CHECK(a && b);
    a->model = &tl->m;
    for (uint32_t i = 0; i < 4 * 5120; i++) a->after_attn[i] = ds41_cpu_bf16(test_float());
    for (uint32_t i = 0; i < 5120; i++) a->norm[i] = ds41_cpu_bf16(test_float());
    memcpy(b, a, sizeof(*a));
    const ds41_cpu_hc_job ha = ds41_cpu_hc_mixer(a, l, true), hb = ds41_cpu_hc_mixer(b, l, true);
    CHECK(ds41_cpu_hc_run(&ha) && ds41_cpu_moe(a, l));
    ds41_cpu_hc_expand(a->residual, a->block, a->after_attn, a->ffn_split);
    CHECK(ds41_cpu_ffn_graph_supported(l, &hb));
    ds41_cpu_ffn_graph_run(b, l, &hb);
    test_equal("FFN split", a->ffn_split, b->ffn_split, 24);
    test_equal("FFN block", a->block, b->block, 5120);
    test_equal("FFN HC state", a->residual, b->residual, 4 * 5120);
    for (uint32_t i = 0; i < 4 * 5120; i++) CHECK(isfinite(b->residual[i]));
    free(b); free(a);
}

/* The staged attention projections against the plain path for a KV source
 * layer of each compression ratio and an index-only layer, over enough
 * positions to publish rows and select from them. */
static void check_proj_graph(test_layer *tl, const ds4_layer_weights *l) {
    static const uint32_t layers[] = {2, 20, 24};
    for (size_t li = 0; li < sizeof(layers) / sizeof(*layers); li++) {
        const uint32_t il = layers[li];
        ds41_cpu_attention_state *a = ds41_cpu_attention_create(64);
        ds41_cpu_attention_state *b = ds41_cpu_attention_create(64);
        float *norm = malloc(5120 * sizeof(float)), *residual = malloc(4 * 5120 * sizeof(float));
        float *flat[2], *mix[2], *split[2];
        CHECK(a && b && norm && residual);
        for (int k = 0; k < 2; k++) {
            flat[k] = malloc(4 * 5120 * sizeof(float));
            mix[k] = malloc(24 * sizeof(float));
            split[k] = malloc(24 * sizeof(float));
            CHECK(flat[k] && mix[k] && split[k]);
        }
        /* Layers above 20 read the index rows layer 20 published. */
        for (uint32_t r = 0; r < 64 * 128; r++)
            a->index_cache[3][r] = b->index_cache[3][r] = ds41_cpu_bf16(test_float());
        for (uint32_t pos = 0; pos < 12; pos++) {
            for (uint32_t i = 0; i < 5120; i++) norm[i] = ds41_cpu_bf16(test_float());
            for (uint32_t i = 0; i < 4 * 5120; i++) residual[i] = ds41_cpu_bf16(test_float());
            ds41_cpu_hc_job h[2];
            for (int k = 0; k < 2; k++)
                h[k] = (ds41_cpu_hc_job){.m = &tl->m, .fn = l->hc_attn_fn,
                    .scale = l->hc_attn_scale, .base = l->hc_attn_base, .state = residual,
                    .flat = flat[k], .mix = mix[k], .split = split[k]};
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0u;
            const bool index = n_comp && ds41_index_source(il);
            const uint32_t owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
            CHECK(ds41_cpu_attention_project(a, &tl->m, l, il, pos, norm, &h[0]));
            CHECK(ds41_cpu_proj_graph_supported(l, il, index, &h[1]));
            CHECK(ds41_cpu_proj_graph_run(b, &tl->m, l, il, pos, norm, index, &h[1]));
            ds41_cpu_rope(b->q, 512, 64, pos, ratio != 0, false);
            if (index) ds41_cpu_attention_rank(b, il, owner, n_comp);
            test_equal("proj split", split[0], split[1], 24);
            test_equal("proj qr", a->qr, b->qr, 1280);
            test_equal("proj q", a->q, b->q, 64 * 512);
            for (uint32_t i = 0; i < 64 * 512; i++) CHECK(isfinite(b->q[i]));
            for (uint32_t i = 0; i < 24; i++) CHECK(isfinite(split[1][i]));
            test_equal("proj window", a->window[il], b->window[il], 128 * 512);
            test_equal("proj compressed", a->compressed[owner], b->compressed[owner],
                       (size_t)a->comp_cap[owner] * 512);
            test_equal("proj index cache", a->index_cache[owner], b->index_cache[owner],
                       (size_t)a->comp_cap[owner] * 128);
            test_equal("proj previous", a->previous_kv[0], b->previous_kv[0], 3 * 512);
            if (index) {
                test_equal("proj index q", a->index_q, b->index_q, 32 * 128);
                test_equal("proj index weights", a->index_weights, b->index_weights, 32);
                CHECK(memcmp(a->selected_comp, b->selected_comp, sizeof(a->selected_comp)) == 0);
            }
        }
        for (int k = 0; k < 2; k++) { free(flat[k]); free(mix[k]); free(split[k]); }
        free(residual); free(norm);
        ds41_cpu_attention_free(b); ds41_cpu_attention_free(a);
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* The staged heads, output projections and HC post against the plain
 * sequence of ds41_cpu_attention_forward() and ds41_cpu_hc_post_run(). */
static void check_out_graph(test_layer *tl, const ds4_layer_weights *l) {
    static const uint32_t rows[] = {1, 17, 130, 640};
    ds41_cpu_attention_state *a = ds41_cpu_attention_create(64);
    ds41_cpu_attention_state *b = ds41_cpu_attention_create(64);
    ds41_cpu_state *sa = calloc(1, sizeof(*sa)), *sb = calloc(1, sizeof(*sb));
    CHECK(a && b && sa && sb);
    for (size_t ri = 0; ri < sizeof(rows) / sizeof(*rows); ri++) {
        const uint32_t n = rows[ri], pos = 700 + (uint32_t)ri;
        for (uint32_t i = 0; i < n * 512; i++) a->attended_kv[i] = test_float();
        for (uint32_t i = 0; i < 64 * 512; i++) a->q[i] = test_float();
        for (uint32_t i = 0; i < 4 * 5120; i++) sa->residual[i] = ds41_cpu_bf16(test_float());
        for (uint32_t i = 0; i < 24; i++) sa->attn_split[i] = test_float();
        memcpy(b->attended_kv, a->attended_kv, n * 512 * sizeof(float));
        memcpy(b->q, a->q, sizeof(a->q));
        memcpy(sb->residual, sa->residual, sizeof(sa->residual));
        memcpy(sb->attn_split, sa->attn_split, sizeof(sa->attn_split));
        ds41_cpu_hc_post_job post[2];
        ds41_cpu_state *st[2] = {sa, sb};
        for (int k = 0; k < 2; k++)
            post[k] = (ds41_cpu_hc_post_job){.m = &tl->m, .norm_weight = l->ffn_norm,
                .block = st[k]->block, .residual = st[k]->residual, .split = st[k]->attn_split,
                .after = st[k]->after_attn, .x = st[k]->x, .norm = st[k]->norm};
        ds41_cpu_attention_heads(a, tensor_data(&tl->m, l->attn_sinks), n);
        ds41_cpu_bf16_rows(a->heads, 64u * 512u);
        ds41_cpu_rope(a->heads, 512, 64, pos, true, true);
        ds41_cpu_matvec_q8_grouped(a->low, &tl->m, l->attn_output_a, a->heads, 1024u, 4096u);
        ds41_cpu_bf16_rows(a->low, 8192u);
        CHECK(ds41_cpu_matvec(sa->block, &tl->m, l->attn_output_b, a->low, false));
        ds41_cpu_bf16_rows(sa->block, 5120);
        CHECK(ds41_cpu_hc_post_run(&post[0]));
        CHECK(ds41_cpu_out_graph_supported(l, &post[1]));
        ds41_cpu_out_graph_run(b, &tl->m, l, pos, true, n, sb->block, &post[1]);
        test_equal("out heads", a->heads, b->heads, 64 * 512);
        test_equal("out low", a->low, b->low, 8192);
        test_equal("out block", sa->block, sb->block, 5120);
        test_equal("out HC state", sa->after_attn, sb->after_attn, 4 * 5120);
        test_equal("out FFN input", sa->x, sb->x, 5120);
        test_equal("out FFN norm", sa->norm, sb->norm, 5120);
        for (uint32_t i = 0; i < 5120; i++) CHECK(isfinite(sb->norm[i]));
    }
    free(sb); free(sa);
    ds41_cpu_attention_free(b); ds41_cpu_attention_free(a);
}
#endif
/* The batched prefill kernels against the one-token ones, token by token:
 * Q8_0 with BF16 rounding and row groups, then the routed experts grouped
 * by expert with the slots added in order. */
static void check_q8_batch(void) {
    const uint64_t dims[] = {32, 64, 256, 1280};
    const uint64_t rows[] = {16, 48, 512};
    const uint32_t tokens[] = {1, 2, 3, 5, 33, 70};
    for (size_t d = 0; d < sizeof(dims) / sizeof(*dims); d++) {
        for (size_t r = 0; r < sizeof(rows) / sizeof(*rows); r++) {
            for (size_t tk = 0; tk < sizeof(tokens) / sizeof(*tokens); tk++) {
                const uint64_t k = dims[d], n = rows[r];
                const uint32_t nt = tokens[tk];
                const uint64_t groups = n % 32 == 0 ? 2 : 1;
                const size_t bytes = (size_t)n * (k / 32) * 34;
                const uint64_t xs = k * groups + 3, os = n + 5;
                test_buffer w = test_alloc(bytes), xb = test_alloc(nt * xs * sizeof(float));
                test_buffer ab = test_alloc(nt * os * sizeof(float));
                test_buffer bb = test_alloc(nt * os * sizeof(float));
                float *x = (float *)xb.data, *a = (float *)ab.data, *b = (float *)bb.data;
                test_fill_quants(w.data, bytes, DS4_TENSOR_Q8_0);
                for (uint64_t i = 0; i < nt * xs; i++) x[i] = test_float();
                ds4_model m = {.map = w.data, .size = bytes};
                ds4_tensor t = {.type = DS4_TENSOR_Q8_0, .ndim = 2,
                               .dim = {k, n}, .elements = k * n, .bytes = bytes};
                for (int round = 0; round < 2; round++) {
                    memset(a, 0, nt * os * sizeof(float));
                    memset(b, 0, nt * os * sizeof(float));
                    for (uint32_t i = 0; i < nt; i++) {
                        ds41_cpu_matvec_q8_grouped(a + i * os, &m, &t, x + i * xs,
                                                   n / groups, k);
                        if (round) ds41_cpu_bf16_rows(a + i * os, n);
                    }
                    if (ds41_cpu_matmul_q8(b, os, &m, &t, x, xs, nt, n / groups, k, round))
                        test_equal("Q8 batch", a, b, nt * os);
                }
                test_free(bb); test_free(ab); test_free(xb); test_free(w);
            }
        }
    }
}

static void check_f32_batch(void) {
    const uint64_t dims[] = {1, 7, 256, 5120};
    const uint64_t rows[] = {1, 3, 4, 5, 24, 384};
    const uint32_t tokens[] = {1, 7, 8, 9, 33};
    for (size_t d = 0; d < sizeof(dims) / sizeof(*dims); d++) {
        for (size_t r = 0; r < sizeof(rows) / sizeof(*rows); r++) {
            for (size_t tk = 0; tk < sizeof(tokens) / sizeof(*tokens); tk++) {
                const uint64_t k = dims[d], n = rows[r];
                const uint32_t nt = tokens[tk];
                const uint64_t xs = k + 3, os = n + 2;
                float *w = malloc(k * n * sizeof(float)), *x = malloc(nt * xs * sizeof(float));
                float *a = calloc(nt * os, sizeof(float)), *b = calloc(nt * os, sizeof(float));
                double *xt = malloc((nt + 7) / 8 * 8 * k * sizeof(double));
                CHECK(w && x && a && b && xt);
                for (uint64_t i = 0; i < k * n; i++) w[i] = test_float() * 3.1f;
                for (uint64_t i = 0; i < nt * xs; i++) x[i] = test_float();
                ds4_model m = {.map = (uint8_t *)w, .size = k * n * sizeof(float)};
                ds4_tensor t = {.type = DS4_TENSOR_F32, .ndim = 2, .dim = {k, n},
                               .elements = k * n, .bytes = k * n * sizeof(float)};
                for (uint32_t i = 0; i < nt; i++) {
                    matvec_f32_ctx ctx = {.out = a + i * os, .data = w, .x = x + i * xs, .in_dim = k};
                    matvec_f32_worker(&ctx, 0, n);
                }
                if (ds41_cpu_matmul_f32(b, os, &m, &t, x, xs, nt, xt))
                    test_equal("F32 batch", a, b, nt * os);
                free(xt); free(b); free(a); free(x); free(w);
            }
        }
    }
}

static void check_routed_batch(uint32_t embd, uint32_t hidden, float clamp) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    g_ds4_shape.n_embd = embd;
    g_ds4_shape.n_ff_exp = hidden;
    g_ds4_shape.n_expert = 7;
    g_ds4_shape.n_expert_used = 3;
    g_ds4_shape.swiglu_clamp_exp = clamp;
    const size_t gbytes = (size_t)embd * hidden * 7 / QK_K * sizeof(block_iq2_xxs);
    const size_t dbytes = (size_t)embd * hidden * 7 / QK_K * sizeof(block_q2_K);
    test_buffer w = test_alloc(2 * gbytes + dbytes);
    test_fill_quants(w.data, gbytes, DS4_TENSOR_IQ2_XXS);
    test_fill_quants(w.data + gbytes, gbytes, DS4_TENSOR_IQ2_XXS);
    test_fill_quants(w.data + 2 * gbytes, dbytes, DS4_TENSOR_Q2_K);
    ds4_model m = {.map = w.data, .size = 2 * gbytes + dbytes};
    ds4_tensor gate = {.type = DS4_TENSOR_IQ2_XXS, .ndim = 3, .dim = {embd, hidden, 7},
                      .elements = (uint64_t)embd * hidden * 7, .bytes = gbytes};
    ds4_tensor up = gate, down = gate;
    up.abs_offset = gbytes;
    down.type = DS4_TENSOR_Q2_K;
    down.abs_offset = 2 * gbytes;
    down.bytes = dbytes;
    down.dim[0] = hidden; down.dim[1] = embd;
    ds4_layer_weights l = {.ffn_gate_exps = &gate, .ffn_up_exps = &up, .ffn_down_exps = &down};
    const uint32_t tokens[] = {1, 2, 5, 9, 40};
    for (size_t tk = 0; tk < sizeof(tokens) / sizeof(*tokens); tk++) {
        const uint32_t nt = tokens[tk];
        float *x = malloc((size_t)nt * embd * sizeof(float));
        float *ma = malloc((size_t)nt * 3 * hidden * sizeof(float));
        float *mb = malloc((size_t)nt * 3 * hidden * sizeof(float));
        float *oa = malloc((size_t)nt * embd * sizeof(float));
        float *ob = malloc((size_t)nt * embd * sizeof(float));
        float *partial = malloc((size_t)nt * 3 * embd * sizeof(float));
        float *weights = malloc((size_t)nt * 3 * sizeof(float));
        int *selected = malloc((size_t)nt * 3 * sizeof(int));
        uint32_t *entry = malloc((size_t)nt * 3 * sizeof(uint32_t));
        CHECK(x && ma && mb && oa && ob && partial && weights && selected && entry);
        for (uint32_t i = 0; i < nt * embd; i++) x[i] = test_float();
        for (uint32_t t = 0; t < nt; t++) {
            /* Three distinct experts, mostly 0..2 so that experts are shared. */
            const uint32_t base = test_random() % (t % 3 ? 5 : 2);
            for (uint32_t k = 0; k < 3; k++) {
                selected[t * 3 + k] = (int)((base + k * (1 + t % 2)) % 7);
                weights[t * 3 + k] = test_float();
            }
            if (selected[t * 3] == selected[t * 3 + 2]) selected[t * 3 + 2] = (selected[t * 3 + 2] + 1) % 7;
            if (selected[t * 3 + 1] == selected[t * 3 + 2]) selected[t * 3 + 2] = (selected[t * 3 + 2] + 1) % 7;
            if (selected[t * 3] == selected[t * 3 + 2]) selected[t * 3 + 2] = (selected[t * 3 + 2] + 1) % 7;
            ds41_cpu_routed_moe(oa + (size_t)t * embd, ma + (size_t)t * 3 * hidden, &m, &l,
                                x + (size_t)t * embd, selected + t * 3, weights + t * 3);
        }
        if (ds41_cpu_routed_moe_batch(partial, mb, entry, &m, &l, x, embd, nt, selected, weights)) {
            for (uint32_t t = 0; t < nt; t++)
                for (uint32_t i = 0; i < embd; i++) {
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < 3; k++)
                        sum = sum + partial[((size_t)t * 3 + k) * embd + i];
                    ob[(size_t)t * embd + i] = sum;
                }
            test_equal("routed batch mid", ma, mb, (size_t)nt * 3 * hidden);
            test_equal("routed batch out", oa, ob, (size_t)nt * embd);
        }
        free(entry); free(selected); free(weights); free(partial);
        free(ob); free(oa); free(mb); free(ma); free(x);
    }
    test_free(w);
}

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
            check_q8_batch();
            check_f32_batch();
            check_routed_batch(256, 512, 10.0f);
            check_routed_batch(512, 256, 0.0f);
            ds4_threads_shutdown();
            printf("V4.1 CPU Q8/Q4/Q2 bitwise reference checks: %u threads%s PASS\n",
                   threads[i], mode ? ", NUMA halves" : "");
        }
    }
    g_numa.on = numa;
    g_requested_threads = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    {
        g_ds4_shape = DS4_SHAPE_FLASH41;
        g_ds4_shape.n_expert = 8;
        const uint32_t ratios[40] = {[2] = 2, [20] = 1, [24] = 1};
        memcpy(g_ds4_compress_ratios, ratios, sizeof(ratios));
        test_layer tl;
        ds4_layer_weights l;
        test_layer_init(&tl, &l);
        for (size_t i = 0; i < sizeof(threads) / sizeof(*threads); i++) {
            g_requested_threads = threads[i];
            check_ffn_graph(&tl, &l);
            check_proj_graph(&tl, &l);
            check_out_graph(&tl, &l);
            ds4_threads_shutdown();
            printf("V4.1 CPU staged FFN, projections and output bitwise checks: %u threads PASS\n",
                   threads[i]);
        }
        g_requested_threads = 0;
        test_free(tl.map);
    }
#endif
    g_ds4_shape = saved;
#if !defined(__AVX512F__) || !defined(__AVX512BW__)
    puts("AVX512 unavailable at compile time: scalar fallback checked");
#endif
    return 0;
}
