/* Public-API state consistency check against real V4.1 weights.
 * Five token evaluations cover an odd compressor frontier and its continuation.
 * This compares one backend with itself; it is not an independent accuracy test. */
#include "ds4.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void note_prefill(void *ud, const char *event, int current, int total) {
    (void)current; (void)total;
    if (!strcmp(event, "prefill_chunk")) (*(unsigned *)ud)++;
}

static int copy_finite_logits(ds4_session *s, int pos, float *out, int n,
                              const char *label) {
    if (!ds4_session_checkpoint_valid(s) || ds4_session_pos(s) != pos ||
        ds4_session_copy_logits(s, out, n) != n) {
        fprintf(stderr, "%s: invalid checkpoint, position or logits length\n", label);
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (!isfinite(out[i])) {
            fprintf(stderr, "%s: nonfinite logit at token %d\n", label, i);
            return 0;
        }
    }
    return 1;
}

static int compare_logits(const float *expected, const float *actual, int n,
                          const char *label) {
    int differences = 0, first = -1;
    double max_abs = 0.0, max_relative = 0.0;
    for (int i = 0; i < n; i++) {
        if (memcmp(expected + i, actual + i, sizeof(float)) == 0) continue;
        if (first < 0) first = i;
        differences++;
        const double delta = fabs((double)expected[i] - actual[i]);
        const double scale = fmax(fabs((double)expected[i]), fabs((double)actual[i]));
        if (delta > max_abs) max_abs = delta;
        if (scale > 0 && delta / scale > max_relative) max_relative = delta / scale;
    }
    if (differences) {
        fprintf(stderr,
                "%s: %d/%d logits differ; first token %d: %.9g vs %.9g; "
                "max_abs=%.9g max_relative=%.9g\n",
                label, differences, n, first, expected[first], actual[first],
                max_abs, max_relative);
        return 0;
    }
    printf("%s: bitwise exact (%d finite logits)\n", label, n);
    fflush(stdout);
    return 1;
}

static int write_logits(const char *path, const float *odd, const float *continued, int n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        return 0;
    }
    /* No header: two consecutive native float32 vocabulary rows, at positions
     * one and two, so scalar/SIMD builds can be compared directly with cmp. */
    int ok = fwrite(odd, sizeof(*odd), (size_t)n, fp) == (size_t)n &&
             fwrite(continued, sizeof(*continued), (size_t)n, fp) == (size_t)n;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "failed to write logits to %s\n", path);
        (void)remove(path);
    }
    return ok;
}

static void usage(const char *program) {
    fprintf(stderr, "Usage: %s MODEL.gguf [THREADS [OUTPUT_LOGITS.bin]]\n", program);
    fputs("Real V4.1 CPU state consistency: five token evaluations, two sessions, "
          "16-token context. Default: 16 threads.\n", stderr);
    fputs("Optional output contains odd-frontier then continuation logits as raw "
          "float32 rows, written only after all checks pass.\n", stderr);
}

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "line %d: %s failed%s%s\n", __LINE__, #condition, \
                err[0] ? ": " : "", err); \
        goto done; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        usage(argv[0]);
        return 0;
    }
    if (argc < 2 || argc > 4) {
        usage(argv[0]);
        return 2;
    }
    int threads = 16;
    if (argc >= 3) {
        char *end = NULL;
        errno = 0;
        const long value = strtol(argv[2], &end, 10);
        if (errno || !argv[2][0] || *end || value < 1 || value > 32) {
            fputs("THREADS must be an integer from 1 through 32\n", stderr);
            return 2;
        }
        threads = (int)value;
    }

    int rc = 1;
    char err[256] = {0};
    ds4_engine *engine = NULL;
    ds4_session *reference = NULL, *candidate = NULL;
    ds4_session_snapshot snapshot = {0};
    ds4_tokens tokens = {0};
    unsigned prefill_events = 0;
    float *odd_logits = NULL, *expected = NULL, *actual = NULL;
    const ds4_engine_options opt = {
        .model_path = argv[1],
        .backend = DS4_BACKEND_CPU,
        .n_threads = threads,
        .context_size = 16,
        .power_percent = 100,
    };

    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    REQUIRE(ds4_engine_is_deepseek41(engine));
    const int n_vocab = ds4_engine_vocab_size(engine);
    REQUIRE(n_vocab > 0);
    ds4_tokenize_text(engine, "Hello world.", &tokens);
    REQUIRE(tokens.len >= 2);
    const ds4_tokens odd = {.v = tokens.v, .len = 1, .cap = 1};
    const ds4_tokens full = {.v = tokens.v, .len = 2, .cap = 2};
    printf("Raw token IDs: %d %d; threads=%d; exactly five forward-token evaluations\n",
           tokens.v[0], tokens.v[1], threads);
    fflush(stdout);
    odd_logits = malloc((size_t)n_vocab * sizeof(*odd_logits));
    expected = malloc((size_t)n_vocab * sizeof(*expected));
    actual = malloc((size_t)n_vocab * sizeof(*actual));
    REQUIRE(odd_logits && expected && actual);
    REQUIRE(ds4_session_create(&reference, engine, 16) == 0);
    REQUIRE(ds4_session_create(&candidate, engine, 16) == 0);
    ds4_session_set_progress(candidate, note_prefill, &prefill_events);

    fputs("Evaluating odd frontier and uninterrupted continuation...\n", stderr);
    REQUIRE(ds4_session_sync(reference, &odd, err, sizeof(err)) == 0);
    REQUIRE(copy_finite_logits(reference, 1, odd_logits, n_vocab, "odd frontier"));
    REQUIRE(ds4_session_save_snapshot(reference, &snapshot, err, sizeof(err)) == 0);
    REQUIRE(snapshot.ptr && snapshot.len > 0);
    REQUIRE(ds4_session_eval(reference, tokens.v[1], err, sizeof(err)) == 0);
    REQUIRE(copy_finite_logits(reference, 2, expected, n_vocab, "uninterrupted continuation"));

    fputs("Evaluating the same two-token prefix in a fresh session...\n", stderr);
    REQUIRE(ds4_session_sync(candidate, &full, err, sizeof(err)) == 0);
    REQUIRE(copy_finite_logits(candidate, 2, actual, n_vocab, "fresh prefix"));
    REQUIRE(compare_logits(expected, actual, n_vocab, "prefix sync versus incremental eval"));
    const unsigned previous_events = prefill_events;
    REQUIRE(ds4_session_sync(candidate, &full, err, sizeof(err)) == 0);
    REQUIRE(prefill_events == previous_events);
    REQUIRE(copy_finite_logits(candidate, 2, actual, n_vocab, "repeated prefix"));
    REQUIRE(compare_logits(expected, actual, n_vocab, "repeated prefix reuse"));

    fputs("Restoring the odd frontier and replaying its continuation...\n", stderr);
    REQUIRE(ds4_session_load_snapshot(candidate, &snapshot, err, sizeof(err)) == 0);
    REQUIRE(copy_finite_logits(candidate, 1, actual, n_vocab, "restored frontier"));
    REQUIRE(compare_logits(odd_logits, actual, n_vocab, "snapshot restored logits"));
    const ds4_tokens *restored = ds4_session_tokens(candidate);
    REQUIRE(restored && restored->len == 1 && restored->v[0] == tokens.v[0]);
    REQUIRE(ds4_session_eval(candidate, tokens.v[1], err, sizeof(err)) == 0);
    REQUIRE(copy_finite_logits(candidate, 2, actual, n_vocab, "restored continuation"));
    REQUIRE(compare_logits(expected, actual, n_vocab, "snapshot continuation versus uninterrupted reference"));
    if (argc == 4) REQUIRE(write_logits(argv[3], odd_logits, expected, n_vocab));
    printf("V4.1 CPU live state consistency: PASS (snapshot %llu bytes). "
           "Independent model accuracy remains untested.\n",
           (unsigned long long)snapshot.len);
    rc = 0;

done:
    free(actual);
    free(expected);
    free(odd_logits);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(candidate);
    ds4_session_free(reference);
    ds4_tokens_free(&tokens);
    ds4_engine_close(engine);
    return rc;
}
