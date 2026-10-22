/* V4.1 scalar activation semantics. Kept independent of model state so the
 * CPU reference and small numerical tests exercise the same implementation. */
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* In particular, reassociating the YaRN interpolation changes long-context
 * phases. Preserve BF16 boundaries even when the surrounding engine is built
 * with -ffast-math. */
#if defined(__clang__)
#pragma float_control(precise, on, push)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#endif

enum {
    DS41_CPU_BF16 = 0,
    DS41_CPU_FP8_E8M0 = 1,
    DS41_CPU_FP4_E8M0 = 2,
    DS41_CPU_FP4_E4M3 = 3
};

static float ds41_cpu_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    bits &= 0xffff0000u;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static void ds41_cpu_bf16_rows(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = ds41_cpu_bf16(x[i]);
}

/* Nearest E4M3FN value, ties to an even code, saturating at 448 (fminf also
 * maps NaN to 448).  Below 2^-6 the grid is the subnormal step 2^-9; above it
 * a float keeps three mantissa bits, rounded to nearest even in the bits.
 * Equal to a search over the 127 positive codes for every float. */
static float ds41_cpu_e4m3(float x) {
    const float a = fminf(fabsf(x), 448.0f);
    float q;
    if (a < 0.015625f) {
        q = rintf(a * 512.0f) * 0.001953125f;
    } else {
        uint32_t bits;
        memcpy(&bits, &a, sizeof(bits));
        bits = (bits + 0x7ffffu + ((bits >> 20) & 1u)) & ~0xfffffu;
        memcpy(&q, &bits, sizeof(q));
    }
    return copysignf(q, x);
}

static float ds41_cpu_e2m1(float x) {
    static const float values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = fminf(fabsf(x), 6.0f);
    unsigned best = 0;
    float error = a;
    for (unsigned i = 1; i < 8u; i++) {
        const float next = fabsf(a - values[i]);
        if (next < error || (next == error && !(i & 1u) && (best & 1u))) {
            best = i;
            error = next;
        }
    }
    return copysignf(values[best], x);
}

/* The format-specific floors below ensure x is at least FLT_MIN. */
static float ds41_cpu_pow2_ceil(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits = (bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u);
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static void ds41_cpu_quantize(float *x, uint32_t width, uint32_t rows, int format) {
    const uint32_t block = format == DS41_CPU_FP4_E4M3 ? 16u : 32u;
    assert(x && width && rows && (size_t)rows <= SIZE_MAX / width);
    assert(format >= DS41_CPU_BF16 && format <= DS41_CPU_FP4_E4M3);
    if (format == DS41_CPU_BF16) {
        ds41_cpu_bf16_rows(x, (size_t)width * rows);
        return;
    }
    assert(width % block == 0);
    for (size_t first = 0; first < (size_t)width * rows; first += block) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < block; i++) {
            x[first + i] = ds41_cpu_bf16(x[first + i]);
            amax = fmaxf(amax, fabsf(x[first + i]));
        }
        float scale;
        if (format == DS41_CPU_FP8_E8M0)
            scale = ds41_cpu_pow2_ceil(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f));
        else if (format == DS41_CPU_FP4_E8M0)
            scale = ds41_cpu_pow2_ceil(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f));
        else
            scale = ds41_cpu_e4m3(fmaxf(amax, 0.01171875f) / 6.0f);
        for (uint32_t i = 0; i < block; i++) {
            const float value = x[first + i];
            const float a = fabsf(value) / scale;
            const float q = format == DS41_CPU_FP8_E8M0 ? ds41_cpu_e4m3(a) : ds41_cpu_e2m1(a);
            x[first + i] = ds41_cpu_bf16(copysignf(q, value) * scale);
        }
    }
}

/* One token, heads contiguous; only the final 64 dimensions are rotated.
 * V4.1 uses adjacent complex pairs and unit magnitude, including compression.
 * Pairs [pair0, pair1) only: a pair's values depend on nothing else. */
static void ds41_cpu_rope_pairs(float *x, uint32_t width, uint32_t heads,
                                uint32_t pos, bool compressed, bool inverse,
                                uint32_t pair0, uint32_t pair1) {
    assert(x && width >= 64u && heads && pos < 1048576u && pair1 <= 32u);
    const float base = compressed ? 160000.0f : 10000.0f;
    const double pi = 3.14159265358979323846;
    const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * pi)) / (2.0 * log(base)));
    const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * pi)) / (2.0 * log(base)));
    for (uint32_t pair = pair0; pair < pair1; pair++) {
        const float denominator = powf(base, (float)pair / 32.0f);
        float frequency = 1.0f / denominator;
        if (compressed) {
            const float ramp = fminf(1.0f, fmaxf(0.0f, ((float)pair - low) / (high - low)));
            const float smooth = 1.0f - ramp;
            const float interpolate = (frequency / 16.0f) * (1.0f - smooth);
            const float extrapolate = frequency * smooth;
            frequency = interpolate + extrapolate;
        }
        const float theta = (float)pos * frequency;
        const float c = cosf(theta), s = inverse ? -sinf(theta) : sinf(theta);
        for (uint32_t head = 0; head < heads; head++) {
            const size_t i = (size_t)head * width + width - 64u + 2u * pair;
            const float re = x[i], im = x[i + 1u];
            x[i] = ds41_cpu_bf16(re * c - im * s);
            x[i + 1u] = ds41_cpu_bf16(re * s + im * c);
        }
    }
}

static void ds41_cpu_rope(float *x, uint32_t width, uint32_t heads,
                         uint32_t pos, bool compressed, bool inverse) {
    ds41_cpu_rope_pairs(x, width, heads, pos, compressed, inverse, 0, 32);
}

/* One token: four residual/key streams share the fifth projected KV vector.
 * Key/value projections cross a BF16 boundary before computing the gate. */
static void ds41_cpu_engram_add(float *residual, const float *kv,
                               const float *qweight, const float *kweight,
                               uint32_t width, float eps) {
    assert(residual && kv && qweight && kweight && width && eps > 0.0f);
    for (uint32_t stream = 0; stream < 4u; stream++) {
        const size_t offset = (size_t)stream * width;
        float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
        for (uint32_t i = 0; i < width; i++) {
            const float h = residual[offset + i];
            const float k = ds41_cpu_bf16(kv[offset + i]);
            h2 += h * h;
            k2 += k * k;
            dot += h * (qweight[offset + i] * kweight[offset + i]) * k;
        }
        dot = dot * (1.0f / sqrtf(h2 / width + eps)) *
              (1.0f / sqrtf(k2 / width + eps)) * (1.0f / sqrtf((float)width));
        const float logit = copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot);
        const float gate = 1.0f / (1.0f + expf(-logit));
        for (uint32_t i = 0; i < width; i++)
            residual[offset + i] = ds41_cpu_bf16(residual[offset + i] +
                gate * ds41_cpu_bf16(kv[(size_t)4u * width + i]));
    }
}

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
