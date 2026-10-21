/* No model required. Compare CPU snapshots with an independently written
 * V4.1 GPU-format payload, then exercise restore boundaries and failed I/O. */
#include "../ds4.c"
#include <assert.h>

static const float untouched = -12345.0f;
static ds4_engine test_engine = {.backend = DS4_BACKEND_CPU};

static void fill_values(float *x, size_t n, unsigned tag) {
    for (size_t i = 0; i < n; i++)
        x[i] = tag ? (float)(tag * 1000u) + (float)(i % 257u) * 0.25f : untouched;
}

static ds4_session *fixture(uint32_t ctx, uint32_t pos, bool populated) {
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &test_engine;
    s->ctx_size = (int)ctx;
    s->ds41_cpu = calloc(1, sizeof(*s->ds41_cpu));
    assert(s->ds41_cpu);
    ds41_cpu_state *g = s->ds41_cpu;
    g->attention = calloc(1, sizeof(*g->attention));
    g->token_map = malloc((size_t)DS4_N_VOCAB * sizeof(*g->token_map));
    s->logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    assert(g->attention && g->token_map && s->logits);
    g->ctx = g->attention->ctx = ctx;
    g->pos = pos;
    g->valid = s->checkpoint_valid = true;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++)
        g->token_map[i] = (i * 7u + 29u) % 99092u;
    for (uint32_t i = 0; i < pos; i++)
        ds4_tokens_push(&s->checkpoint, (int)((i * 17u + 11u) % DS4_N_VOCAB));
    fill_values(s->logits, DS4_N_VOCAB, populated ? 1 : 0);
    for (uint32_t il = 0; il < 40; il++) {
        g->attention->window[il] = malloc(128u * 512u * sizeof(float));
        assert(g->attention->window[il]);
        fill_values(g->attention->window[il], 128u * 512u, populated ? il + 2u : 0);
    }
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t cap = ctx / (i < 3 ? 2u : 1u) + 1u;
        g->attention->comp_cap[i] = cap;
        g->attention->compressed[i] = malloc((size_t)cap * 512u * sizeof(float));
        g->attention->index_cache[i] = malloc((size_t)cap * 128u * sizeof(float));
        assert(g->attention->compressed[i] && g->attention->index_cache[i]);
        fill_values(g->attention->compressed[i], (size_t)cap * 512u, populated ? 50u + i : 0);
        fill_values(g->attention->index_cache[i], (size_t)cap * 128u, populated ? 60u + i : 0);
        if (i < 3) {
            fill_values(g->attention->previous_kv[i], 512u, populated ? 70u + i : 0);
            fill_values(g->attention->previous_score[i], 512u, populated ? 80u + i : 0);
        }
    }
    memset(&g->history, 0x5a, sizeof(g->history));
    return s;
}

static void fixture_free(ds4_session *s) {
    ds41_cpu_state *g = s->ds41_cpu;
    for (uint32_t i = 0; i < 40; i++) free(g->attention->window[i]);
    for (uint32_t i = 0; i < 4; i++) {
        free(g->attention->compressed[i]);
        free(g->attention->index_cache[i]);
    }
    free(g->attention);
    free(g->token_map);
    free(g);
    free(s->logits);
    ds4_tokens_free(&s->checkpoint);
    free(s);
}

static void golden_u32(FILE *fp, uint32_t v) {
    const unsigned char b[] = {(unsigned char)v, (unsigned char)(v >> 8),
                               (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    assert(fwrite(b, 1, sizeof(b), fp) == sizeof(b));
}

static void golden_floats(FILE *fp, const float *p, size_t n) {
    assert(fwrite(p, sizeof(float), n, fp) == n);
}

/* Keep this traversal independent of ds41_cpu_payload_spans: in particular,
 * an odd frontier stores its pending pair immediately after each owner cache. */
static FILE *golden_payload(ds4_session *s) {
    FILE *fp = tmpfile();
    assert(fp);
    ds41_cpu_state *g = s->ds41_cpu;
    const uint32_t pos = g->pos;
    const uint32_t header[] = {DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION,
        g->ctx, 1, 128, 128, g->ctx + 1, pos, 40, 512, 128, DS4_N_VOCAB, 0x413431u};
    assert(sizeof(header) / sizeof(header[0]) == DS4_SESSION_PAYLOAD_U32_FIELDS);
    for (size_t i = 0; i < sizeof(header) / sizeof(header[0]); i++) golden_u32(fp, header[i]);
    for (uint32_t i = 0; i < pos; i++) golden_u32(fp, (uint32_t)s->checkpoint.v[i]);
    golden_floats(fp, s->logits, DS4_N_VOCAB);
    for (unsigned il = 0; il < 40; il++)
        golden_floats(fp, g->attention->window[il], (size_t)(pos < 128 ? pos : 128) * 512);
    for (unsigned i = 0; i < 4; i++) {
        const size_t rows = i < 3 ? pos / 2 : pos;
        golden_floats(fp, g->attention->compressed[i], rows * 512);
        golden_floats(fp, g->attention->index_cache[i], rows * 128);
        if (i < 3 && pos % 2) {
            golden_floats(fp, g->attention->previous_kv[i], 512);
            golden_floats(fp, g->attention->previous_score[i], 512);
        }
    }
    assert(fflush(fp) == 0);
    rewind(fp);
    return fp;
}

static uint64_t expected_size(uint32_t pos) {
    const uint64_t raw = pos < 128u ? pos : 128u;
    const uint64_t values = DS4_N_VOCAB + 40u * raw * 512u +
        (3u * (uint64_t)(pos / 2u) + pos) * 640u + (pos & 1u) * 3u * 1024u;
    return (DS4_SESSION_PAYLOAD_U32_FIELDS + (uint64_t)pos) * 4u + values * sizeof(float);
}

static void assert_files_equal(FILE *a, FILE *b, uint64_t bytes) {
    rewind(a);
    rewind(b);
    unsigned char ab[8192], bb[8192];
    while (bytes) {
        const size_t n = bytes < sizeof(ab) ? (size_t)bytes : sizeof(ab);
        assert(fread(ab, 1, n, a) == n);
        assert(fread(bb, 1, n, b) == n);
        assert(memcmp(ab, bb, n) == 0);
        bytes -= n;
    }
    assert(fgetc(a) == EOF && fgetc(b) == EOF);
}

static void assert_region(const float *actual, const float *source, size_t live, size_t cap) {
    assert(memcmp(actual, source, live * sizeof(float)) == 0);
    for (size_t i = live; i < cap; i++) assert(actual[i] == untouched);
}

static void assert_restored(ds4_session *dst, const ds4_session *src) {
    ds41_cpu_state *g = dst->ds41_cpu, *original = src->ds41_cpu;
    const uint32_t pos = original->pos;
    assert(g->valid && dst->checkpoint_valid && !dst->mtp_draft_valid);
    assert(g->pos == pos && dst->checkpoint.len == (int)pos);
    assert(memcmp(dst->checkpoint.v, src->checkpoint.v, (size_t)pos * sizeof(int)) == 0);
    assert(memcmp(dst->logits, src->logits, (size_t)DS4_N_VOCAB * sizeof(float)) == 0);
    for (unsigned il = 0; il < 40; il++)
        assert_region(g->attention->window[il], original->attention->window[il],
                       (size_t)(pos < 128 ? pos : 128) * 512, 128u * 512u);
    for (unsigned i = 0; i < 4; i++) {
        const size_t rows = i < 3 ? pos / 2 : pos;
        assert_region(g->attention->compressed[i], original->attention->compressed[i],
                       rows * 512, (size_t)g->attention->comp_cap[i] * 512);
        assert_region(g->attention->index_cache[i], original->attention->index_cache[i],
                       rows * 128, (size_t)g->attention->comp_cap[i] * 128);
        if (i < 3) {
            const size_t live = pos % 2 ? 512 : 0;
            assert_region(g->attention->previous_kv[i], original->attention->previous_kv[i], live, 512);
            assert_region(g->attention->previous_score[i], original->attention->previous_score[i], live, 512);
        }
    }
    for (uint32_t i = 0; i < DS4_ENGRAM_NGRAM - 1u; i++) {
        const int32_t expected = i < pos ?
            (int32_t)g->token_map[src->checkpoint.v[pos - 1u - i]] : DS4_ENGRAM_DEAD;
        assert(g->history.tail[i] == expected);
    }
}

static void round_trip(uint32_t pos, uint32_t destination_ctx) {
    ds4_session *src = fixture(257, pos, true);
    ds4_session *dst = fixture(destination_ctx, 0, false);
    FILE *expected = golden_payload(src), *actual = tmpfile();
    assert(actual);
    char err[256] = {0};
    const uint64_t bytes = expected_size(pos);
    assert(ds4_session_payload_bytes(src) == bytes);
    assert(ds41_cpu_payload_body_bytes(src->ds41_cpu, pos) +
            DS4_SESSION_PAYLOAD_U32_FIELDS * 4u == bytes);
    assert(ds4_session_save_payload(src, actual, err, sizeof(err)) == 0);
    assert_files_equal(expected, actual, bytes);
    rewind(expected);
    dst->mtp_draft_valid = true;
    assert(ds4_session_load_payload(dst, expected, bytes, err, sizeof(err)) == 0);
    assert_restored(dst, src);
    /* Exercise the public in-memory snapshot wrapper's exact-size contract. */
    ds4_session_snapshot snapshot = {0};
    assert(ds4_session_save_snapshot(src, &snapshot, err, sizeof(err)) == 0);
    assert(snapshot.len == bytes);
    assert(ds4_session_load_snapshot(dst, &snapshot, err, sizeof(err)) == 0);
    assert_restored(dst, src);
    ds4_session_snapshot_free(&snapshot);
    fclose(expected);
    fclose(actual);
    fixture_free(dst);
    fixture_free(src);
}

static void rejected_loads(void) {
    ds4_session *src = fixture(257, 129, true);
    ds4_session *dst = fixture(129, 1, false);
    FILE *fp = golden_payload(src);
    char err[256];
    assert(ds4_session_load_payload(dst, fp, expected_size(129), err, sizeof(err)) != 0);
    assert(dst->checkpoint_valid && dst->ds41_cpu->valid && dst->ds41_cpu->pos == 1);
    fclose(fp);
    fixture_free(dst);
    dst = fixture(257, 1, false);

    /* Rejected header/body lengths and token IDs do not touch live state. */
    const unsigned fields[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
        fp = golden_payload(src);
        assert(fseek(fp, (long)fields[i] * 4, SEEK_SET) == 0);
        golden_u32(fp, 0xffffffffu);
        rewind(fp);
        assert(ds4_session_load_payload(dst, fp, expected_size(129), err, sizeof(err)) != 0);
        assert(dst->checkpoint_valid && dst->ds41_cpu->valid && dst->ds41_cpu->pos == 1);
        fclose(fp);
    }
    fp = golden_payload(src);
    assert(ds4_session_load_payload(dst, fp, expected_size(129) - 4, err, sizeof(err)) != 0);
    assert(dst->checkpoint_valid && dst->ds41_cpu->valid);
    rewind(fp);
    assert(fseek(fp, DS4_SESSION_PAYLOAD_U32_FIELDS * 4, SEEK_SET) == 0);
    golden_u32(fp, DS4_N_VOCAB);
    rewind(fp);
    assert(ds4_session_load_payload(dst, fp, expected_size(129), err, sizeof(err)) != 0);
    assert(dst->checkpoint_valid && dst->ds41_cpu->valid);
    fclose(fp);

    /* A complete declared length with truncated storage fails after writes
     * begin. The old frontier must no longer be usable. */
    fp = golden_payload(src);
    assert(ftruncate(fileno(fp), (off_t)(expected_size(129) - 4)) == 0);
    rewind(fp);
    assert(ds4_session_load_payload(dst, fp, expected_size(129), err, sizeof(err)) != 0);
    assert(!dst->checkpoint_valid && !dst->ds41_cpu->valid);
    assert(ds4_session_payload_bytes(dst) == 0);
    fclose(fp);
    fp = golden_payload(src);
    assert(ds4_session_load_payload(dst, fp, expected_size(129), err, sizeof(err)) == 0);
    assert_restored(dst, src);
    fclose(fp);
    fixture_free(dst);
    fixture_free(src);
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    const uint32_t positions[] = {1, 2, 3, 127, 128, 129};
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); i++)
        round_trip(positions[i], 257);
    round_trip(128, 129); /* A smaller context can load any frontier that fits. */
    round_trip(129, 513);
    rejected_loads();
    puts("V4.1 CPU payload compatibility and failure recovery: PASS");
    return 0;
}
