/* Standalone numerical oracles: cc -O2 -std=c99 -Wall -Wextra -I.
 * tests/test_deepseek41_cpu_math.c -lm -o /tmp/test_deepseek41_cpu_math */
#include "../ds4_deepseek41_cpu_math.inc"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t bits_of(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static float from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

/* An arithmetic oracle, independent of the production integer rounding trick. */
static float reference_bf16(float x) {
    if (!isfinite(x)) return from_bits(bits_of(x) & 0xffff0000u);
    if (x == 0.0f) return x;
    int exponent;
    const double magnitude = fabs((double)x);
    frexp(magnitude, &exponent);
    const double step = ldexp(1.0, exponent - 8 > -133 ? exponent - 8 : -133);
    const double scaled = magnitude / step;
    double integer = floor(scaled);
    const double fraction = scaled - integer;
    if (fraction > 0.5 || (fraction == 0.5 && fmod(integer, 2.0) != 0.0)) integer++;
    return copysignf((float)(integer * step), x);
}

static int check_bf16(void) {
    const uint32_t low[] = {0, 0x7fffu, 0x8000u, 0x8001u, 0xffffu};
    for (uint32_t upper = 0; upper < 65536u; upper++) {
        for (size_t i = 0; i < sizeof(low) / sizeof(*low); i++) {
            const float x = from_bits((upper << 16u) | low[i]);
            CHECK(bits_of(ds41_cpu_bf16(x)) == bits_of(reference_bf16(x)));
        }
    }
    float vector[] = {1.00390625f, 1.01171875f, -0.0f, 0x1p-133f, -0x1p-134f};
    float expected[sizeof(vector) / sizeof(*vector)];
    for (size_t i = 0; i < sizeof(vector) / sizeof(*vector); i++) expected[i] = reference_bf16(vector[i]);
    ds41_cpu_bf16_rows(vector, sizeof(vector) / sizeof(*vector));
    CHECK(memcmp(vector, expected, sizeof(vector)) == 0);
    puts("BF16 rounding: exhaustive exponent/mantissa boundaries, ties and signed zero: PASS");
    return 1;
}

/* Brute-force search of the format's representable values. */
static double reference_value(unsigned code, int fp4) {
    const double fp4_values[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    if (fp4) return fp4_values[code];
    return code < 8u ? (double)code / 512.0 :
        ldexp(8.0 + (code % 8u), (int)(code / 8u) - 10);
}

static float reference_nearest(float x, int fp4) {
    unsigned best = 0;
    double error = INFINITY;
    for (unsigned i = 0; i < (fp4 ? 8u : 127u); i++) {
        const double d = fabs(fabs((double)x) - reference_value(i, fp4));
        if (d < error || (d == error && !(i & 1u) && (best & 1u))) {
            best = i;
            error = d;
        }
    }
    return copysignf((float)reference_value(best, fp4), x);
}

static float reference_pow2(float x) {
    int e;
    const double mantissa = frexp((double)x, &e);
    return (float)ldexp(1.0, mantissa == 0.5 ? e - 1 : e);
}

static void reference_quantize(float *x, size_t count, int format) {
    const size_t block = format == DS41_CPU_FP4_E4M3 ? 16u : 32u;
    for (size_t i = 0; i < count; i++) x[i] = reference_bf16(x[i]);
    if (format == DS41_CPU_BF16) return;
    for (size_t first = 0; first < count; first += block) {
        float maximum = 0.0f;
        for (size_t i = 0; i < block; i++) maximum = fmaxf(maximum, fabsf(x[first + i]));
        const float scale = format == DS41_CPU_FP8_E8M0 ?
            reference_pow2(fmaxf(maximum, 1e-4f) * (1.0f / 448.0f)) :
            format == DS41_CPU_FP4_E8M0 ?
            reference_pow2(fmaxf(maximum, 0x1.8p-124f) * (1.0f / 6.0f)) :
            reference_nearest(fmaxf(maximum, 6.0f / 512.0f) / 6.0f, 0);
        for (size_t i = 0; i < block; i++)
            x[first + i] = reference_bf16(reference_nearest(x[first + i] / scale,
                format != DS41_CPU_FP8_E8M0) * scale);
    }
}

static uint32_t random_state = 4141;
static float random_value(void) {
    random_state ^= random_state << 13u;
    random_state ^= random_state >> 17u;
    random_state ^= random_state << 5u;
    return ((int)(random_state % 65537u) - 32768) / 8192.0f;
}

static int check_quantize(void) {
    enum { WIDTH = 512, ROWS = 5, N = WIDTH * ROWS };
    float actual[N], expected[N];
    const float scales[] = {0.0f, 0x1p-140f, 0x1p-132f, 0x1p-123f, 1e-4f, .01f, 1, 16, 1024};
    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS41_CPU_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) actual[i] = random_value() * scales[(i / block) % 9];
        actual[0] = -0.0f;
        memcpy(expected, actual, sizeof(actual));
        reference_quantize(expected, N, mode);
        ds41_cpu_quantize(actual, WIDTH, ROWS, mode);
        for (int i = 0; i < N; i++) {
            if (bits_of(actual[i]) != bits_of(expected[i])) {
                fprintf(stderr, "quant mode %d index %d: %a != %a\n", mode, i, actual[i], expected[i]);
                return 0;
            }
        }
    }
    /* Anchor each block at the largest code, forcing scale=1. This exercises
     * every half-way tie (both signs) through the actual block quantizer. */
    for (int mode = 1; mode < 4; mode++) {
        const int fp4 = mode != DS41_CPU_FP8_E8M0;
        const unsigned block = mode == DS41_CPU_FP4_E4M3 ? 16u : 32u;
        for (unsigned code = 0; code < (fp4 ? 7u : 126u); code++) {
            float x[32] = {0}, y[32];
            x[0] = fp4 ? 6.0f : 448.0f;
            const float midpoint = (float)((reference_value(code, fp4) + reference_value(code + 1, fp4)) / 2);
            x[1] = midpoint; x[2] = -midpoint;
            x[3] = from_bits(bits_of(midpoint) - 0x10000u);
            x[4] = from_bits(bits_of(midpoint) + 0x10000u);
            x[5] = -x[3]; x[6] = -x[4]; x[7] = -0.0f;
            memcpy(y, x, sizeof(x));
            reference_quantize(y, block, mode);
            ds41_cpu_quantize(x, block, 1, mode);
            CHECK(memcmp(x, y, block * sizeof(float)) == 0);
        }
    }
    puts("FP8/FP4 activation blocks: independent oracle, all ties, subnormals and signed zero: PASS");
    return 1;
}

/* Fixed float32 frequencies from the released pow/reciprocal/YaRN definition.
 * Freezing these catches phase drift caused by compiler reassociation. */
static const float reference_frequencies[2][32] = {
    {
        0x1p+0f, 0x1.7ff222p-1f, 0x1.1feb34p-1f, 0x1.afd136p-2f,
        0x1.43d136p-2f, 0x1.e5a846p-3f, 0x1.6c310ep-3f, 0x1.111aeep-3f,
        0x1.99999ap-4f, 0x1.33281ap-4f, 0x1.ccab84p-5f, 0x1.59742ap-5f,
        0x1.030dc6p-5f, 0x1.84869ep-6f, 0x1.235a72p-6f, 0x1.b4f7e4p-7f,
        0x1.47ae14p-7f, 0x1.eb736p-8f, 0x1.708938p-8f, 0x1.145ceep-8f,
        0x1.9e7c7p-9f, 0x1.36d21ap-9f, 0x1.d22a5p-10f, 0x1.5d931cp-10f,
        0x1.0624dep-10f, 0x1.892918p-11f, 0x1.26d42cp-11f, 0x1.ba2e4ap-12f,
        0x1.4b96cp-12f, 0x1.f1502ap-13f, 0x1.74eea6p-13f, 0x1.17a8e4p-13f
    }, {
        0x1p+0f, 0x1.601474p-1f, 0x1.e4383ep-2f, 0x1.4cfa02p-2f,
        0x1.c9f25cp-3f, 0x1.3ae8eap-3f, 0x1.b11968p-4f, 0x1.29d2c6p-4f,
        0x1.99999ap-5f, 0x1.19a9f6p-5f, 0x1.836032p-6f, 0x1.0a619cp-6f,
        0x1.6e5b7ep-7f, 0x1.f7db1p-8f, 0x1.5a7abcp-8f, 0x1.dc846ep-9f,
        0x1.28f5c2p-9f, 0x1.6e29cp-10f, 0x1.bd7b6cp-11f, 0x1.0a619cp-11f,
        0x1.37675ep-12f, 0x1.60b2fp-13f, 0x1.7d209ap-14f, 0x1.7d36bep-15f,
        0x1.47ae18p-16f, 0x1.6887ap-18f, 0x1.efd74cp-19f, 0x1.54f7dp-19f,
        0x1.d4effep-20f, 0x1.4277b8p-20f, 0x1.bb7e6p-21f, 0x1.30f89ap-21f
    }
};

static int check_rope(void) {
    enum { WIDTH = 96, HEADS = 3, N = WIDTH * HEADS };
    const uint32_t positions[] = {0, 1, 127, 128, 511, 4095, 65535, 100000, 262143, 1048575};
    float x[N], original[N];
    for (int kind = 0; kind < 2; kind++) for (int inverse = 0; inverse < 2; inverse++) {
        for (size_t p = 0; p < sizeof(positions) / sizeof(*positions); p++) {
            for (int i = 0; i < N; i++) original[i] = reference_bf16(random_value());
            memcpy(x, original, sizeof(x));
            ds41_cpu_rope(x, WIDTH, HEADS, positions[p], kind != 0, inverse != 0);
            double error2 = 0, norm2 = 0;
            for (unsigned h = 0; h < HEADS; h++) {
                CHECK(memcmp(x + h * WIDTH, original + h * WIDTH, (WIDTH - 64) * sizeof(float)) == 0);
                for (unsigned pair = 0; pair < 32; pair++) {
                    const float theta = (float)positions[p] * reference_frequencies[kind][pair];
                    const double c = cos(theta), s = inverse ? -sin(theta) : sin(theta);
                    const size_t i = h * WIDTH + WIDTH - 64 + pair * 2;
                    const float expected[] = {
                        reference_bf16((float)(original[i] * c - original[i + 1] * s)),
                        reference_bf16((float)(original[i] * s + original[i + 1] * c))
                    };
                    for (unsigned j = 0; j < 2; j++) {
                        const double error = x[i + j] - expected[j];
                        error2 += error * error;
                        norm2 += (double)expected[j] * expected[j];
                    }
                }
            }
            if (sqrt(error2 / norm2) > 1e-4) {
                fprintf(stderr, "RoPE kind %d inverse %d position %u relative RMS %.9g\n",
                    kind, inverse, positions[p], sqrt(error2 / norm2));
                return 0;
            }
            ds41_cpu_rope(x, WIDTH, HEADS, positions[p], kind != 0, inverse == 0);
            for (int i = 0; i < N; i++) CHECK(fabsf(x[i] - original[i]) <= 0.03125f);
        }
    }
    puts("V4.1 RoPE: fixed frequencies, compressed/uncompressed, inverse, positions through 1M: PASS");
    return 1;
}

static int check_engram(void) {
    enum { WIDTH = 5120, N = WIDTH * 4 };
    float *x = malloc(N * sizeof(float)), *original = malloc(N * sizeof(float));
    float *kv = malloc(WIDTH * 5 * sizeof(float));
    float *qw = malloc(N * sizeof(float)), *kw = malloc(N * sizeof(float));
    CHECK(x && original && kv && qw && kw);
    for (int i = 0; i < N; i++) {
        x[i] = original[i] = reference_bf16(random_value());
        qw[i] = random_value(); kw[i] = random_value();
    }
    for (int i = 0; i < WIDTH * 5; i++) kv[i] = random_value();
    /* Exercise the epsilon path independently for zero residual and zero key. */
    memset(x, 0, WIDTH * sizeof(float));
    memset(original, 0, WIDTH * sizeof(float));
    memset(kv + WIDTH, 0, WIDTH * sizeof(float));
    ds41_cpu_engram_add(x, kv, qw, kw, WIDTH, 1e-20f);
    double error2 = 0, norm2 = 0;
    size_t different = 0;
    for (unsigned h = 0; h < 4; h++) {
        double dot = 0, h2 = 0, k2 = 0;
        for (unsigned i = 0; i < WIDTH; i++) {
            const unsigned j = h * WIDTH + i;
            const double a = original[j], b = reference_bf16(kv[j]);
            h2 += a * a; k2 += b * b;
            dot += a * (float)(qw[j] * kw[j]) * b;
        }
        dot /= sqrt(h2 / WIDTH + 1e-20) * sqrt(k2 / WIDTH + 1e-20) * sqrt(WIDTH);
        const double gate = 1.0 / (1.0 + exp(-copysign(sqrt(fmax(fabs(dot), 1e-6)), dot)));
        for (unsigned i = 0; i < WIDTH; i++) {
            const unsigned j = h * WIDTH + i;
            const float expected = reference_bf16(original[j] + (float)gate * reference_bf16(kv[4 * WIDTH + i]));
            const double d = x[j] - expected;
            CHECK(isfinite(x[j]));
            CHECK(fabs(d) <= fmax(1e-6, fabs(expected) / 128));
            error2 += d * d; norm2 += (double)expected * expected;
            different += x[j] != expected;
        }
    }
    CHECK(different < N / 1000);
    CHECK(sqrt(error2 / norm2) < 1e-4);
    free(x); free(original); free(kv); free(qw); free(kw);
    printf("Engram gate: independent double oracle, %zu BF16 boundary differences: PASS\n", different);
    return 1;
}

int main(void) {
    return !(check_bf16() && check_quantize() && check_rope() && check_engram());
}
